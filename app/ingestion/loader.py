"""
Document loading and splitting.

Loaders read raw files into LangChain's Document objects (text + metadata).
Splitters break long documents into smaller chunks -- models and vector DBs
work better on small, focused chunks than one giant blob of text.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def load_and_split(file_path: str) -> list[Document]:
    """Loads a text file and splits it into overlapping chunks ready for embedding."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)