"""Core RAG service: ChromaDB retrieval + Gemini generation, built with LangChain."""
import hashlib
import asyncio
import logging
import re
from collections.abc import AsyncIterator
from urllib.parse import urljoin

import requests
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.rag.fallback import (
    TEMP_SERVICE_MESSAGE,
    build_data_answer,
    general_answer,
)
from app.rag.guard import (
    format_trip_context,
    is_travel_related,
    off_topic_response,
)
from app.rag.loaders import fetch_url_text, read_text_files
from app.rag.prompts import RAG_PROMPT
from app.rag.understand import (
    describe_intent_flags,
    detect_intents,
    expand_query,
    find_budget_npr,
    find_places,
    history_places,
)
from app.schemas.chat import Message, Source

logger = logging.getLogger(__name__)


class KnowledgeBaseEmpty(Exception):
    """Raised when a question is asked before any data was ingested."""


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embed_model,
            google_api_key=settings.google_api_key,
        )
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_dir,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.top_k}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        llm_kwargs = {}
        if settings.thinking_budget >= 0:
            llm_kwargs["thinking_budget"] = settings.thinking_budget

        llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            temperature=0,
            max_output_tokens=settings.max_output_tokens,
            max_retries=2,
            google_api_key=settings.google_api_key,
            **llm_kwargs,
        )
        self.chain = RAG_PROMPT | llm | StrOutputParser()

    # ------------------------------------------------------------------ data
    def count(self) -> int:
        return self.vectorstore._collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0

    def add_text(self, text: str, source: str) -> int:
        """Chunk + embed + store text. Chunks already in the DB are skipped."""
        docs = self.splitter.create_documents([text], metadatas=[{"source": source}])
        unique = {}
        for doc in docs:
            key = hashlib.sha256(f"{source}:{doc.page_content}".encode()).hexdigest()[:32]
            unique[key] = doc
        if not unique:
            return 0

        existing = set(self.vectorstore.get(ids=list(unique))["ids"])
        new = {k: d for k, d in unique.items() if k not in existing}
        if not new:
            return 0

        self.vectorstore.add_documents(list(new.values()), ids=list(new.keys()))
        logger.info("Added %d chunks from %s", len(new), source)
        return len(new)

    def ingest_directory(self) -> int:
        added = 0
        for name, text in read_text_files(self.settings.data_dir):
            added += self.add_text(text, name)
        return added

    def add_url(self, url: str) -> int:
        return self.add_text(fetch_url_text(url), url)

    # ------------------------------------------------------------------ chat
    async def _prepare(
        self, question: str, history: list[Message], trip_context=None
    ) -> tuple[dict, list[Source], dict[str, list[dict]], list[str]]:
        """Gather retrieval inputs. NEVER raises: every source is optional.

        Returns (prompt inputs, sources, live records by route, RAG snippets).
        An empty knowledge base or a downed embedding/API service only yields
        notes — the caller decides between Gemini, data fallback, or general
        guidance. Nothing technical is ever included for the user.
        """
        docs: list = []
        notes: list[str] = []
        try:
            if self.is_empty():
                notes.append("No retrieved travel-guide knowledge is available right now.")
            else:
                expanded = expand_query(
                    question, [m for m in history], trip_context
                )
                docs = await self.retriever.ainvoke(expanded)
        except Exception:
            logger.exception("Knowledge-base retrieval failed; continuing without it")
            notes.append("Travel-guide lookup is temporarily unavailable.")
            docs = []

        try:
            live_context, live_sources, live_records = await self._get_live_context(
                question, history, trip_context
            )
        except Exception:
            logger.exception("Live lookup wrapper failed; continuing without it")
            live_context, live_sources, live_records = (
                "Live travel records are temporarily unavailable.",
                [],
                {},
            )

        trip_text = format_trip_context(trip_context) or "(none)"
        context_parts = [d.page_content for d in docs]
        context_parts.extend(notes)
        inputs = {
            "context": "\n\n---\n\n".join(context_parts) or "(none)",
            "live_context": live_context,
            "history": "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
            or "(none)",
            "trip_context": trip_text,
            "question": question,
        }
        sources = [
            Source(
                source=str(d.metadata.get("source", "unknown")),
                snippet=d.page_content[:200],
            )
            for d in docs
        ]
        snippets = [d.page_content for d in docs]
        return inputs, sources + live_sources, live_records, snippets

    def _primary_destination(
        self, question: str, history: list[Message] | None, trip_context=None
    ) -> str | None:
        places = find_places(question)
        if places:
            return places[0]
        for place in history_places(history):
            return place
        return None

    async def _get_live_context(
        self, question: str, history: list[Message] | None = None, trip_context=None
    ) -> tuple[str, list[Source], dict[str, list[dict]]]:
        """Query SuvYatra live collections by detected intent + entities.

        Never raises: per-route failures are logged and reported as plain
        "could not be verified" notes.
        """
        expanded = expand_query(question, history, trip_context)
        intents = detect_intents(expanded)
        places = find_places(expanded)
        if not intents and places:
            intents = ["destination"]
        if not intents and (history or trip_context):
            # Contextual follow-up ("Any cheap hotels?", "How far is it?"):
            # fall back to destinations around the recent place.
            intents = ["destination"]

        routes: list[str] = []
        for intent in intents:
            for route in {
                "hotel": ("hotels",),
                "restaurant": ("restaurants",),
                "transport": ("transports",),
                "flight": ("flights",),
                "destination": ("destinations",),
                "activity": ("places",),
                "service": ("places",),
            }.get(intent, ()):
                if route not in routes:
                    routes.append(route)

        if not routes:
            return "No live database lookup was needed for this question.", [], {}
        if not self.settings.suvyatra_api_url:
            return (
                "Live inventory lookup is not configured. Do not claim current availability, prices, or schedules.",
                [],
                {},
            )

        search_term = ""
        match = re.search(
            r"\b(?:in|near|around|from|to)\s+([a-z][a-z '\-]{1,45}?)(?=\s+(?:and|for|with)\b|[?.!,;:]|$)",
            expanded,
            re.IGNORECASE,
        )
        if match:
            search_term = match.group(1).strip()
        if not search_term and places:
            search_term = places[0]
        flags = describe_intent_flags(expanded)
        budget_cap = find_budget_npr(expanded)

        lowered = expanded.lower()
        places_module = ""
        if "activity" in intents or "activity" in lowered:
            places_module = "activity"
        elif "hospital" in lowered or "medical" in lowered:
            places_module = "hospital"
        elif "atm" in lowered:
            places_module = "atm"
        elif "petrol" in lowered or "fuel" in lowered:
            places_module = "petrol"

        base_url = self.settings.suvyatra_api_url.rstrip("/") + "/"
        contexts: list[str] = []
        sources: list[Source] = []
        records: dict[str, list[dict]] = {}

        for route in routes:
            params: dict[str, object] = {}
            if search_term:
                params["search"] = search_term
            if route == "hotels" and budget_cap:
                params["maxPrice"] = budget_cap
            if route == "transports" and flags["cheap"]:
                params["sort"] = "cheapest"
            if route == "destinations" and flags["hidden"]:
                params["featured"] = "hidden"
            if route == "places" and places_module:
                params["module"] = places_module
            url = urljoin(base_url, route)
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    url,
                    params=params or None,
                    timeout=3,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError(f"Expected a list from the {route} endpoint")
            except (requests.RequestException, ValueError) as error:
                logger.warning("Live %s lookup failed: %s", route, error)
                contexts.append(
                    f"{route.title()} live records could not be verified right now."
                )
                continue

            safe_records = []
            for record in payload[:8]:
                if not isinstance(record, dict):
                    continue
                allowed_fields = (
                    "name",
                    "airline",
                    "flightNumber",
                    "from",
                    "to",
                    "departureTime",
                    "arrivalTime",
                    "date",
                    "duration",
                    "location",
                    "city",
                    "description",
                    "available",
                    "pricePerNight",
                    "priceRange",
                    "rating",
                    "type",
                    "route",
                    "price",
                    "province",
                    "highlights",
                )
                safe_record = {
                    field: record[field]
                    for field in allowed_fields
                    if field in record and record[field] not in (None, "")
                }
                if safe_record:
                    safe_records.append(safe_record)

            if safe_records:
                records[route] = safe_records
                contexts.append(f"{route.title()} records: {safe_records}")
                sources.append(
                    Source(source=f"SuvYatra live database: {route}", snippet=str(safe_records)[:200])
                )
            else:
                contexts.append(f"No current {route} records were returned.")

        return "\n".join(contexts), sources, records

    async def chat(
        self, question: str, history: list[Message] | None = None, trip_context=None
    ) -> tuple[str, list[Source]]:
        history = history or []
        # LEVEL 4 — domain check BEFORE retrieval and BEFORE any Gemini call:
        # clearly off-topic questions never reach the model.
        if not is_travel_related(question, history, trip_context):
            logger.info("Off-topic question refused without LLM call")
            return off_topic_response(question), []
        try:
            inputs, sources, live_records, snippets = await self._prepare(
                question, history, trip_context
            )
            intents = detect_intents(expand_query(question, history, trip_context))
            destination = self._primary_destination(question, history, trip_context)
            try:
                # LEVEL 1 — Gemini + retrieved data.
                answer = await self.chain.ainvoke(inputs)
                return answer, sources
            except Exception:
                logger.exception("Gemini generation failed; trying data fallback")
                # LEVEL 2 — Gemini down but data exists: answer from data.
                data_answer = build_data_answer(live_records, snippets, destination)
                if data_answer:
                    return data_answer, sources
                # LEVEL 3 — travel question, no usable data: general guidance.
                return general_answer(question, intents), sources
        except Exception:
            # LEVEL 5 — genuine system failure: log internally, friendly reply.
            logger.exception("Chat pipeline failed; returning friendly fallback")
            return TEMP_SERVICE_MESSAGE, []

    async def stream(
        self, question: str, history: list[Message] | None = None, trip_context=None
    ) -> AsyncIterator[tuple[str, list[Source] | None]]:
        """Yield (token, None) for each piece, then ("", sources) at the end."""
        history = history or []
        if not is_travel_related(question, history, trip_context):
            logger.info("Off-topic question refused without LLM call")
            refusal = off_topic_response(question)
            yield refusal, None
            yield "", []
            return
        try:
            inputs, sources, live_records, snippets = await self._prepare(
                question, history, trip_context
            )
            intents = detect_intents(expand_query(question, history, trip_context))
            destination = self._primary_destination(question, history, trip_context)
            sent_any = False
            try:
                async for token in self.chain.astream(inputs):
                    sent_any = True
                    yield token, None
            except Exception:
                logger.exception("Gemini streaming failed")
                if not sent_any:
                    fallback = build_data_answer(live_records, snippets, destination)
                    if fallback is None:
                        fallback = general_answer(question, intents)
                    yield fallback, None
                else:
                    yield (
                        "\n\n(Note: the live connection was interrupted — "
                        "please ask again if anything looks incomplete.)",
                        None,
                    )
            yield "", sources
        except Exception:
            logger.exception("Streaming pipeline failed; returning friendly fallback")
            yield TEMP_SERVICE_MESSAGE, None
            yield "", []
