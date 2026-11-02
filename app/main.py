import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.rag import RAGService

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag = RAGService(settings)
    app.state.rag = rag

    if settings.auto_ingest and rag.is_empty():
        try:
            added = await run_in_threadpool(rag.ingest_directory)
            logger.info("Startup ingest finished: %d chunks", added)
        except Exception:
            logger.exception("Startup ingest failed (server will still start)")

    logger.info("Ready with %d chunks", rag.count())
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG chatbot for Nepal travel: FastAPI + LangChain + Gemini + ChromaDB",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:8])
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.0f ms) [%s]",
            request.method, request.url.path, response.status_code, ms, request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        # User-safe detail: the frontend never surfaces raw errors, but this
        # text must still read cleanly if it ever reaches a screen.
        return JSONResponse(
            {"detail": "The travel assistant had trouble with that request. Please try again in a moment."},
            status_code=500,
        )

    app.include_router(health.router)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
