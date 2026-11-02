import logging

import requests
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag, require_admin
from app.rag.rag import RAGService
from app.schemas.ingest import IngestResponse, IngestTextRequest, IngestUrlRequest

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_admin)]
)


@router.post("/text", response_model=IngestResponse)
def ingest_text(body: IngestTextRequest, rag: RAGService = Depends(get_rag)):
    added = rag.add_text(body.text, body.source)
    return IngestResponse(chunks_added=added, total_chunks=rag.count())


@router.post("/url", response_model=IngestResponse)
def ingest_url(body: IngestUrlRequest, rag: RAGService = Depends(get_rag)):
    try:
        added = rag.add_url(str(body.url))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except requests.RequestException as exc:
        raise HTTPException(502, f"Could not fetch URL: {exc}")
    return IngestResponse(chunks_added=added, total_chunks=rag.count())


@router.post("/reload", response_model=IngestResponse)
def ingest_data_folder(rag: RAGService = Depends(get_rag)):
    """Re-scan the ./data folder and add any new content."""
    added = rag.ingest_directory()
    return IngestResponse(chunks_added=added, total_chunks=rag.count())
