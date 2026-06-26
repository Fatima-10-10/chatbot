"""
Chat endpoint: POST /chat
"""

from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.rag.chain import get_rag_chatbot

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Build chatbot with optional document filter per request
    chatbot = get_rag_chatbot(source_file=request.document_filter)

    result = chatbot.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": request.session_id}},
    )
    return ChatResponse(answer=result.answer, confidence=result.confidence)