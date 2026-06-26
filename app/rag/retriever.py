"""
Retriever setup.

Supports optional filtering by source_file metadata so you can chat
with a specific document instead of searching across all of them.
"""

from langchain_core.retrievers import BaseRetriever
from app.ingestion.vectorstore import get_vectorstore


def get_retriever(k: int = 3, source_file: str | None = None) -> BaseRetriever:
    """
    Returns a retriever that fetches top-k relevant chunks.
    If source_file is provided, only searches chunks from that document.
    """
    vectorstore = get_vectorstore()

    search_kwargs: dict = {"k": k}
    if source_file:
        search_kwargs["filter"] = {"source_file": {"$eq": source_file}}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)