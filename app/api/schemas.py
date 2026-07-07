from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_filter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: str = "high"


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    chunks_upserted: int
    filename: str


class DeleteResponse(BaseModel):
    filename: str
    message: str