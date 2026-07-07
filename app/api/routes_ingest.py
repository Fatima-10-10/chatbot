"""
Ingest endpoint: POST /ingest
Accepts real file uploads (multipart/form-data) from the browser.
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.api.schemas import IngestResponse
from app.ingestion.loader import load_and_split, SUPPORTED_EXTENSIONS
from app.ingestion.vectorstore import upsert_documents

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.delete("/ingest/{filename}", response_model=IngestResponse)
def delete_document(filename: str) -> IngestResponse:
    """
    Deletes all chunks belonging to a specific document from Pinecone.
    Uses the source_file metadata tag we added during ingestion.
    """
    try:
        from app.ingestion.vectorstore import delete_document as delete_from_store
        count = delete_from_store(filename)
        return IngestResponse(chunks_upserted=0, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_split(save_path)
        count = upsert_documents(chunks, filename=file.filename)  # pass filename
    finally:
        os.remove(save_path)

    return IngestResponse(chunks_upserted=count, filename=file.filename)