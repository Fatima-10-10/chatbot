"""
Document loading and splitting.

Uses semantic chunking instead of fixed character splitting:
- RecursiveCharacterTextSplitter respects natural boundaries (paragraphs, sentences)
- For structured documents, we first split by section markers then by size
- This keeps related information together instead of cutting mid-section

Supports .txt, .pdf, .docx, and .csv files.
"""

import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv"}

# Semantic separators in priority order:
# Try to split on double newlines (sections) first,
# then single newlines (paragraphs),
# then sentences, then words, then characters as last resort.
# This means chunks break at natural boundaries whenever possible.
SEMANTIC_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]


def load_and_split(file_path: str) -> list[Document]:
    """
    Loads a file and splits using semantic boundaries.
    RecursiveCharacterTextSplitter tries each separator in order,
    only falling back to smaller units when the chunk would exceed chunk_size.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEMANTIC_SEPARATORS,
        # Keep separator at end of chunk so context isn't lost
        keep_separator=True,
    )

    chunks = splitter.split_documents(documents)

    # Filter out chunks that are too small to be useful (e.g. just a heading)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    return chunks