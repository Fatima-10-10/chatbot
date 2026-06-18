"""
Request/response schemas for the API.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    chunks_upserted: int