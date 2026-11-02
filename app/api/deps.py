import secrets

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import Settings, get_settings
from app.rag.rag import RAGService


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


def require_admin(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Protect admin endpoints with an X-API-Key header."""
    if settings.admin_api_key is None:
        if settings.environment == "production":
            raise HTTPException(503, "ADMIN_API_KEY is not configured on the server")
        return  # open in development for convenience
    if not secrets.compare_digest(x_api_key or "", settings.admin_api_key):
        raise HTTPException(401, "Invalid or missing X-API-Key")
