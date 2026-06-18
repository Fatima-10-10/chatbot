"""
Retriever setup.

A retriever is the standard interface for "find relevant documents for this
query," regardless of what's underneath -- here it's Pinecone, but the rest of
the app never needs to know that. This is what lets you swap vector stores
later without touching any code outside this file.
"""

from langchain_core.retrievers import BaseRetriever

from app.ingestion.vectorstore import get_vectorstore


def get_retriever(k: int = 3) -> BaseRetriever:
    """Returns a retriever that fetches the top-k most relevant chunks for a query."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})