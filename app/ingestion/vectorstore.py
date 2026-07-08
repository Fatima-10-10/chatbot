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

def delete_document(filename: str) -> None:
    """
    Deletes all chunks tagged with source_file=filename from Pinecone.
    Uses Pinecone's metadata filtering to find and delete matching vectors.
    """
    pc = _get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    # Pinecone doesn't support delete-by-metadata directly in serverless,
    # so we fetch matching IDs first then delete by ID
    results = index.query(
        vector=[0.0] * settings.embedding_dimension,
        filter={"source_file": {"$eq": filename}},
        top_k=10000,
        include_metadata=False,
    )

    ids = [match["id"] for match in results["matches"]]
    if ids:
        index.delete(ids=ids)


def list_documents() -> list[str]:
    """Returns list of unique source_file values in the index."""
    pc = _get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    seen_docs = set()

    # Use multiple different queries to sample different parts of the index
    # A zero vector is biased toward one cluster; random vectors cover more ground
    import random
    random.seed(42)
    
    for _ in range(5):
        # Random vector to sample different parts of the index
        random_vector = [random.uniform(-1, 1) for _ in range(settings.embedding_dimension)]
        results = index.query(
            vector=random_vector,
            top_k=100,
            include_metadata=True,
        )
        for match in results["matches"]:
            source = match.get("metadata", {}).get("source_file")
            if source:
                seen_docs.add(source)

    return list(seen_docs)