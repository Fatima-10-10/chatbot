"""
RAG chain: ties together Retriever + Prompt + Model + Memory + Output Parser.
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.llm import get_llm
from app.core.memory import get_session_history
from app.core.parsers import chat_answer_parser
from app.core.prompts import get_rag_prompt
from app.rag.retriever import get_retriever


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chatbot(source_file: str | None = None) -> RunnableWithMessageHistory:
    """
    Returns a RAG chatbot. If source_file is provided, answers are grounded
    only in that document's chunks instead of the entire index.
    """
    retriever = get_retriever(k=3, source_file=source_file)
    model = get_llm()
    prompt = get_rag_prompt()

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | model
        | chat_answer_parser
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )