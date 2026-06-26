"""
Vector store setup (Pinecone).

create_index_if_missing(): idempotent -- safe to call every time the app starts,
since it only actually creates anything the first time.

upsert_documents(): embeds and pushes document chunks into the index. This is
what app/api/routes_ingest.py will eventually call when you hit POST /ingest.
"""

import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.embeddings import get_embeddings


def _get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def create_index_if_missing() -> None:
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name in existing:
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
        time.sleep(1)


def get_vectorstore() -> PineconeVectorStore:
    """Connects to the (already-created) index, ready for search or upsert."""
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )


def upsert_documents(chunks: list[Document], filename: str) -> int:
    """Embeds and pushes chunks into the index, tagged with the source filename."""
    create_index_if_missing()
    vectorstore = get_vectorstore()

    # Tag each chunk with the filename so we can filter by document later
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    vectorstore.add_documents(chunks)
    return len(chunks)