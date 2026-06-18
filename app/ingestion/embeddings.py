"""
Embedding model setup.

HuggingFaceEmbeddings runs locally (downloads the model once, then works
offline/free) and produces vectors of a fixed size -- settings.embedding_dimension
must match whatever this model actually outputs, since Pinecone needs to know
the vector size up front when creating an index.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)