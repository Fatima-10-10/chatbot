"""
RAG chain with:
- Manual memory management (fixes RunnableWithMessageHistory + PydanticOutputParser incompatibility)
- Question routing (document vs conversational)
- Query rephrasing (standalone questions from follow-ups)
- Query expansion (multiple phrasings for better retrieval)
- Result deduplication (merge results from multiple queries)
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm
from app.core.memory import get_session_history, get_history_messages
from app.core.parsers import chat_answer_parser
from app.core.prompts import get_rag_prompt, get_rephrase_prompt, get_query_expansion_prompt
from app.rag.retriever import get_retriever
from app.rag.reranker import rerank

# Load once at module level so the embedding model isn't reloaded on every request
_retriever_cache: dict = {}

def _get_cached_retriever(k: int, source_file: str | None):
    key = (k, source_file)
    if key not in _retriever_cache:
        _retriever_cache[key] = get_retriever(k=k, source_file=source_file)
    return _retriever_cache[key]

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks based on content."""
    seen = set()
    unique = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


def get_rag_chatbot(source_file: str | None = None):
    retriever = _get_cached_retriever(k=10, source_file=source_file)
    model = get_llm()
    str_parser = StrOutputParser()
    rephrase_prompt = get_rephrase_prompt()
    rag_prompt = get_rag_prompt()
    expansion_prompt = get_query_expansion_prompt()

    def rephrase_question(user_input: str, session_id: str) -> str:
        """Rewrites follow-up questions into standalone questions."""
        history = get_history_messages(session_id)
        if not history:
            return user_input
        messages = rephrase_prompt.format_messages(
            history=history,
            input=user_input,
        )
        return str_parser.invoke(model.invoke(messages))

    def expand_query(question: str) -> list[str]:
        """
        Generates 3 alternative phrasings of the question.
        Query expansion: searching with multiple phrasings catches more relevant chunks.
        Falls back to original question if expansion fails.
        """
        try:
            messages = expansion_prompt.format_messages(question=question)
            raw = str_parser.invoke(model.invoke(messages))
            # Strip markdown code fences if present
            raw = raw.strip().strip("```json").strip("```").strip()
            variants = json.loads(raw)
            if isinstance(variants, list):
                return [question] + variants[:3]
        except Exception:
            pass
        return [question]

    def is_document_question(user_input: str, session_id: str) -> bool:
        """Classifies whether the question needs document retrieval or can be answered from history."""
        history = get_history_messages(session_id)
        if not history:
            return True
        router_prompt = (
            "Classify this message as either 'document' or 'conversational'.\n"
            "'conversational': greetings, small talk, or questions about what was said earlier.\n"
            "'document': any question needing information from a document.\n"
            "Reply with ONLY one word: 'document' or 'conversational'.\n\n"
            f"Message: {user_input}"
        )
        result = str_parser.invoke(model.invoke(router_prompt)).strip().lower()
        return result != "conversational"

    def retrieve_with_expansion(question: str) -> list[Document]:
        """
        Retrieves documents using multiple query phrasings and deduplicates results.
        This is the query expansion step -- more phrasings = better coverage.
        """
        variants = expand_query(question)
        all_docs = []
        for variant in variants:
            docs = retriever.invoke(variant)
            all_docs.extend(docs)
        return _deduplicate(all_docs)

    def invoke(user_input: str, session_id: str) -> dict:
        """
        Full RAG pipeline:
        1. Load history
        2. Classify question (document vs conversational)
        3. Rephrase into standalone question
        4. Expand query into multiple phrasings
        5. Retrieve + deduplicate chunks
        6. Build prompt + call model
        7. Save to memory
        8. Parse and return
        """
        history = get_history_messages(session_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            routing_future = executor.submit(is_document_question, user_input, session_id)
            rephrase_future = executor.submit(rephrase_question, user_input, session_id)
            needs_retrieval = routing_future.result()
            rephrased = rephrase_future.result()


        if needs_retrieval:
                # Step 1: retrieve top 10 candidates from Pinecone
                docs = retriever.invoke(rephrased)
                # Step 2: deduplicate
                docs = _deduplicate(docs)
                # Step 3: rerank — cross-encoder scores each chunk against the query
                # and returns top 5 in true relevance order
                docs = rerank(rephrased, docs, top_n=5)
                context = _format_docs(docs)
        else:
            context = ""
    

        needs_retrieval = is_document_question(user_input, session_id)
        rephrased = rephrase_question(user_input, session_id)

        if needs_retrieval:
            docs = retrieve_with_expansion(rephrased)
            context = _format_docs(docs)
        else:
            context = ""

        messages = rag_prompt.format_messages(
            context=context,
            history=history,
            input=user_input,
        )
        raw_response = model.invoke(messages)

        # Save to memory BEFORE parsing
        session = get_session_history(session_id)
        session.add_message(HumanMessage(content=user_input))
        session.add_message(AIMessage(content=raw_response.content))

        try:
            parsed = chat_answer_parser.parse(raw_response.content)
            return {"answer": parsed.answer, "confidence": parsed.confidence}
        except Exception:
            return {"answer": raw_response.content, "confidence": "low"}

    return invoke