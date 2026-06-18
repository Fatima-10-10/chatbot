"""
Ingest endpoint: POST /ingest
"""

from fastapi import APIRouter

from app.api.schemas import IngestRequest, IngestResponse
from app.ingestion.loader import load_and_split
from app.ingestion.vectorstore import upsert_documents

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    chunks = load_and_split(request.file_path)
    count = upsert_documents(chunks)
    return IngestResponse(chunks_upserted=count)