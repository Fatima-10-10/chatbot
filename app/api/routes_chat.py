"""
Chat endpoints:
- POST /chat         — standard request/response
- POST /chat/stream  — streaming response (tokens arrive one by one)
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.exceptions import OutputParserException
import json

from app.api.schemas import ChatRequest, ChatResponse
from app.rag.chain import get_rag_chatbot

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        invoke, _ = get_rag_chatbot(source_file=request.document_filter)
        result = invoke(
            user_input=request.message,
            session_id=request.session_id,
        )
        return ChatResponse(answer=result["answer"], confidence=result["confidence"])
    except OutputParserException:
        return ChatResponse(
            answer="I had trouble formatting my response. Please try again.",
            confidence="low"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streams the response token by token using Server-Sent Events (SSE).
    The frontend reads these tokens and appends them to the chat bubble in real time.
    """
    _, stream = get_rag_chatbot(source_file=request.document_filter)

    def generate():
        for token in stream(
            user_input=request.message,
            session_id=request.session_id,
        ):
            # SSE format: "data: <token>\n\n"
            yield f"data: {json.dumps(token)}\n\n"
        # Signal stream is done
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )