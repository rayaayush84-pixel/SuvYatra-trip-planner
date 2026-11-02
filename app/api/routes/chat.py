import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag
from app.rag.fallback import TEMP_SERVICE_MESSAGE
from app.rag.rag import RAGService
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, rag: RAGService = Depends(get_rag)) -> ChatResponse:
    # The service implements the full fallback hierarchy and only returns
    # user-safe answers. This catch-all is belt-and-braces: even an
    # unexpected bug becomes a friendly message, never a technical error.
    try:
        answer, sources = await rag.chat(body.question, body.history, body.trip_context)
    except Exception:
        logger.exception("Chat request failed unexpectedly")
        answer, sources = TEMP_SERVICE_MESSAGE, []
    return ChatResponse(answer=answer, sources=sources)


@router.post("/stream")
async def chat_stream(body: ChatRequest, rag: RAGService = Depends(get_rag)):
    """Server-Sent Events: {"token": "..."} events, then {"done": true, "sources": [...]}."""

    async def events():
        try:
            async for token, sources in rag.stream(body.question, body.history, body.trip_context):
                if sources is None:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    payload = {"done": True, "sources": [s.model_dump() for s in sources]}
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            logger.exception("Streaming chat failed unexpectedly")
            yield f"data: {json.dumps({'token': TEMP_SERVICE_MESSAGE})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
