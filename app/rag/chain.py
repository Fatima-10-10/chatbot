"""
RAG chain with manual memory management.

RunnableWithMessageHistory doesn't work with PydanticOutputParser because
it can't save custom objects to history. Fix: save the raw model response
to memory manually BEFORE parsing, then parse separately.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.llm import get_llm
from app.core.memory import get_session_history, get_history_messages
from app.core.parsers import chat_answer_parser
from app.core.prompts import get_rag_prompt, get_rephrase_prompt
from app.rag.retriever import get_retriever


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chatbot(source_file: str | None = None):
    retriever = get_retriever(k=10, source_file=source_file)
    model = get_llm()
    rephrase_prompt = get_rephrase_prompt()
    rag_prompt = get_rag_prompt()
    str_parser = StrOutputParser()

    def rephrase_question(x: dict) -> str:
        session_id = x.get("session_id", "")
        history = get_history_messages(session_id) if session_id else []
        if not history:
            return x["input"]
        messages = rephrase_prompt.format_messages(
            history=history,
            input=x["input"],
        )
        return str_parser.invoke(model.invoke(messages))

    def is_document_question(x: dict) -> bool:
        session_id = x.get("session_id", "")
        history = get_history_messages(session_id) if session_id else []
        if not history:
            return True
        router_prompt = (
            "Classify this message as either 'document' or 'conversational'.\n"
            "'conversational' means: greetings, small talk, or questions about "
            "what was said earlier in THIS conversation.\n"
            "'document' means: any question needing information from a document.\n"
            "Reply with ONLY one word: 'document' or 'conversational'.\n\n"
            f"Message: {x['input']}"
        )
        result = str_parser.invoke(model.invoke(router_prompt)).strip().lower()
        return result != "conversational"

    def get_context(x: dict) -> str:
        if x.get("needs_retrieval", True):
            return _format_docs(retriever.invoke(x["rephrased"]))
        return ""

    def invoke(user_input: str, session_id: str) -> dict:
        """
        Manually manages the full RAG + memory flow:
        1. Load history
        2. Rephrase + route
        3. Retrieve context if needed
        4. Build prompt + call model
        5. Save turn to memory
        6. Parse and return
        """
        # Step 1: load history
        history = get_history_messages(session_id)

        # Step 2: classify and rephrase
        x = {"input": user_input, "session_id": session_id}
        needs_retrieval = is_document_question(x)
        rephrased = rephrase_question(x)

        # Step 3: retrieve context
        context = _format_docs(retriever.invoke(rephrased)) if needs_retrieval else ""

        # Step 4: build prompt and call model
        messages = rag_prompt.format_messages(
            context=context,
            history=history,
            input=user_input,
        )
        raw_response = model.invoke(messages)

        # Step 5: save to memory BEFORE parsing
        session = get_session_history(session_id)
        session.add_message(HumanMessage(content=user_input))
        session.add_message(AIMessage(content=raw_response.content))

        # Step 6: parse and return
        try:
            parsed = chat_answer_parser.parse(raw_response.content)
            return {"answer": parsed.answer, "confidence": parsed.confidence}
        except Exception:
            return {"answer": raw_response.content, "confidence": "low"}

    return invoke