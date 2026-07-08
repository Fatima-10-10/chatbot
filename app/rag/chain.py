"""
RAG chain with:
- Manual memory management (fixes RunnableWithMessageHistory + PydanticOutputParser incompatibility)
- Question routing (document vs conversational)
- Query rephrasing (standalone questions from follow-ups)
- Query expansion (multiple phrasings for better retrieval)
- Reranking (cross-encoder reordering of retrieved chunks)
- Document-aware retrieval (per-document retrieval in all-documents mode)
- Streaming support
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

_retriever_cache: dict = {}


def _get_cached_retriever(k: int, source_file: str | None):
    key = (k, source_file)
    if key not in _retriever_cache:
        _retriever_cache[key] = get_retriever(k=k, source_file=source_file)
    return _retriever_cache[key]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _deduplicate(docs: list[Document]) -> list[Document]:
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
        history = get_history_messages(session_id)
        if not history:
            return user_input
        messages = rephrase_prompt.format_messages(
            history=history,
            input=user_input,
        )
        return str_parser.invoke(model.invoke(messages))

    def expand_query(question: str) -> list[str]:
        try:
            messages = expansion_prompt.format_messages(question=question)
            raw = str_parser.invoke(model.invoke(messages))
            raw = raw.strip().strip("```json").strip("```").strip()
            variants = json.loads(raw)
            if isinstance(variants, list):
                return [question] + variants[:2]
        except Exception:
            pass
        return [question]

    def is_document_question(user_input: str, session_id: str) -> bool:
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
        variants = expand_query(question)
        all_docs = []
        with ThreadPoolExecutor(max_workers=len(variants)) as executor:
            futures = {executor.submit(retriever.invoke, v): v for v in variants}
            for future in as_completed(futures):
                try:
                    all_docs.extend(future.result())
                except Exception:
                    pass
        return _deduplicate(all_docs)

    def _build_context(needs_retrieval: bool, rephrased: str, sf: str | None) -> str:
        if not needs_retrieval:
            return ""

        if sf:
            # Single document mode: retrieve + rerank from that document only
            docs = retriever.invoke(rephrased)
            if len(rephrased.split()) <= 5:
                extra_docs = retrieve_with_expansion(rephrased)
                docs = docs + extra_docs
            docs = _deduplicate(docs)
            docs = rerank(rephrased, docs, top_n=8)
            return _format_docs(docs)
        else:
            # All documents mode: retrieve from each document separately
            # so no single document dominates the results
            from app.ingestion.vectorstore import list_documents, get_vectorstore
            all_doc_names = list_documents()

            if not all_doc_names:
                return ""

            all_docs = []
            vs = get_vectorstore()

            for doc_name in all_doc_names:
                doc_results = vs.similarity_search(
                    rephrased,
                    k=3,
                    filter={"source_file": {"$eq": doc_name}}
                )
                if doc_results:
                    reranked = rerank(rephrased, doc_results, top_n=2)
                    all_docs.extend(reranked)

            return _format_docs(all_docs)

    def _prepare(user_input: str, session_id: str, sf: str | None = None):
        """Shared preparation logic for both invoke and stream."""
        history = get_history_messages(session_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            routing_future = executor.submit(is_document_question, user_input, session_id)
            rephrase_future = executor.submit(rephrase_question, user_input, session_id)
            needs_retrieval = routing_future.result()
            rephrased = rephrase_future.result()

        context = _build_context(needs_retrieval, rephrased, sf)
        messages = rag_prompt.format_messages(
            context=context,
            history=history,
            input=user_input,
        )
        return history, messages

    def invoke(user_input: str, session_id: str) -> dict:
        # Pass source_file (captured from outer scope) to _prepare
        history, messages = _prepare(user_input, session_id, source_file)
        raw_response = model.invoke(messages)

        session = get_session_history(session_id)
        session.add_message(HumanMessage(content=user_input))
        session.add_message(AIMessage(content=raw_response.content))

        try:
            parsed = chat_answer_parser.parse(raw_response.content)
            return {"answer": parsed.answer, "confidence": parsed.confidence}
        except Exception:
            return {"answer": raw_response.content, "confidence": "low"}

    def stream(user_input: str, session_id: str):
        # Pass source_file (captured from outer scope) to _prepare
        history, messages = _prepare(user_input, session_id, source_file)

        full_response = ""
        for chunk in model.stream(messages):
            token = chunk.content
            full_response += token
            yield token

        session = get_session_history(session_id)
        session.add_message(HumanMessage(content=user_input))
        session.add_message(AIMessage(content=full_response))

    return invoke, stream