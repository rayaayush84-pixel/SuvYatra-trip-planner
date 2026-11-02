from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag
from app.rag.rag import RAGService

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
def ready(rag: RAGService = Depends(get_rag)) -> dict:
    """Readiness probe: the knowledge base has data."""
    count = rag.count()
    if count == 0:
        raise HTTPException(503, "Travel data is still loading")
    return {"status": "ready", "chunks": count}
