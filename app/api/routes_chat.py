"""
Chat endpoint: POST /chat
"""

from fastapi import APIRouter
from app.api.schemas import ChatRequest, ChatResponse
from app.rag.chain import get_rag_chatbot

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    chatbot = get_rag_chatbot(source_file=request.document_filter)
    result = chatbot(
        user_input=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(answer=result["answer"], confidence=result["confidence"])