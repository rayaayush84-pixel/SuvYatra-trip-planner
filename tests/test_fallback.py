"""Fallback hierarchy + retrieval understanding + endpoint sanitization.

Covers spec TESTs 1-12 behavior at unit level:
- travel questions get useful answers (L1/L2/L3),
- off-topic gets the refusal,
- dead Gemini + data -> structured data answer (L2),
- dead Gemini + no data -> general guidance (L3),
- total backend explosion -> friendly message, still HTTP 200 (L5),
- no technical string ("AI service", "Gemini", status codes, tracebacks)
  ever appears in a user-facing answer.
"""

import pytest
from types import SimpleNamespace

from app.rag.fallback import (
    TEMP_SERVICE_MESSAGE,
    build_data_answer,
    general_answer,
)
from app.rag.guard import OFF_TOPIC_RESPONSES
from app.rag.rag import RAGService
from app.rag.understand import (
    detect_intents,
    expand_query,
    find_budget_npr,
    find_places,
)
from app.schemas.chat import Message, Source

import re as _re

TECHNICAL_MARKERS = [
    "AI service",
    "Gemini",
    "Traceback",
    "Internal server",
    "Something went wrong",
    "Knowledge base",
    "Chroma",
    "TypeError",
    "ValueError",
]
# Bare HTTP codes only (not digits inside formatted prices like "2,500").
_TECHNICAL_CODE_RES = [_re.compile(r"(?<![\d,])500\b"), _re.compile(r"(?<![\d,])503\b")]


def assert_user_safe(answer: str):
    for marker in TECHNICAL_MARKERS:
        assert marker not in answer, f"technical leak {marker!r} in: {answer[:200]}"
    for pattern in _TECHNICAL_CODE_RES:
        assert not pattern.search(answer), f"technical code leak in: {answer[:200]}"


POKHARA_HISTORY = [
    Message(role="user", content="Tell me about Pokhara."),
    Message(role="assistant", content="Pokhara has a lake and caves."),
]

HOTEL_RECORDS = {
    "hotels": [
        {"name": "Lake View Hotel", "location": "Lakeside, Pokhara", "pricePerNight": 2500, "currency": "NPR", "rating": 4.5, "type": "Hotel"},
        {"name": "Budget Stay Inn", "location": "Pokhara", "pricePerNight": 1200, "currency": "NPR", "rating": 4.0},
    ]
}


def make_service(monkeypatch, *, chain=None, docs=None, live=None, empty=True):
    """Bare RAGService with stubbed retrieval/generation (no network)."""
    service = RAGService.__new__(RAGService)
    service.settings = SimpleNamespace(suvyatra_api_url=None)

    class FakeRetriever:
        async def ainvoke(self, query):
            if isinstance(docs, Exception):
                raise docs
            return docs or []

    service.retriever = FakeRetriever()
    monkeypatch.setattr(service, "is_empty", lambda: empty)
    if live is not None:
        async def fake_live(question, history=None, trip_context=None):
            if isinstance(live, Exception):
                raise live
            return live

        monkeypatch.setattr(service, "_get_live_context", fake_live)
    if chain is not None:
        service.chain = chain
    return service


class DeadChain:
    async def ainvoke(self, inputs):
        raise RuntimeError("Gemini API error 503 overloaded")

    async def astream(self, inputs):
        raise RuntimeError("Gemini API error 503 overloaded")
        yield  # pragma: no cover


# ---------------------------------------------------------------- L2/L3 units

def test_data_answer_lists_real_records():
    answer = build_data_answer(HOTEL_RECORDS, [], "Pokhara")
    assert answer is not None
    assert "Lake View Hotel" in answer
    assert "Budget Stay Inn" in answer
    assert "2,500" in answer
    assert "Pokhara" in answer
    assert_user_safe(answer)


def test_data_answer_none_without_data():
    assert build_data_answer({}, [], None) is None
    assert build_data_answer({"hotels": []}, [], None) is None


def test_data_answer_uses_rag_snippets_when_no_live_records():
    answer = build_data_answer({}, ["Phewa Lake is the heart of Pokhara tourism."], "Pokhara")
    assert answer is not None
    assert "Phewa Lake" in answer
    assert_user_safe(answer)


@pytest.mark.parametrize("question,intent_word", [
    ("What hotels are available in Pokhara?", "stay"),
    ("Any cheap restaurants?", "Thamel"),
    ("How can I travel from Kathmandu to Pokhara?", "bus"),
    ("What is the weather like for my trip?", "monsoon"),
])
def test_general_answers_useful_and_safe(question, intent_word):
    answer = general_answer(question, ["hotel"])
    assert len(answer) > 50
    assert_user_safe(answer)


# ------------------------------------------------- understanding / retrieval

def test_expand_query_adds_history_place():
    expanded = expand_query("Any cheap hotels?", POKHARA_HISTORY)
    assert "pokhara" in expanded.lower()
    assert "cheap hotels" in expanded.lower()


def test_detect_intents_and_budget():
    assert "hotel" in detect_intents("Find me cheap hotels in Kathmandu")
    assert "restaurant" in detect_intents("Which restaurants are near Thamel?")
    assert "transport" in detect_intents("How can I travel from Kathmandu to Pokhara?")
    assert find_budget_npr("Plan a 5 day Nepal trip under NPR 30,000") == 30000
    assert find_places("hidden gems around Kathmandu") == ["kathmandu"]


# ------------------------------------------------- service fallback levels

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gemini_down_with_data_returns_l2_answer(anyio_backend, monkeypatch):
    """TEST 10: Gemini unavailable, database has hotels -> useful answer."""
    service = make_service(
        monkeypatch,
        chain=DeadChain(),
        docs=[],
        live=("Hotels records: [{'name': 'Lake View'}]", [Source(source="live", snippet="x")], HOTEL_RECORDS),
        empty=True,
    )
    answer, sources = await service.chat("What hotels are available in Pokhara?")
    assert "Lake View Hotel" in answer
    assert "2,500" in answer
    assert len(sources) >= 1
    assert_user_safe(answer)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gemini_down_without_data_returns_l3_answer(anyio_backend, monkeypatch):
    """TEST 11: nothing retrieved -> clear guidance, never 'AI service failed'."""
    service = make_service(
        monkeypatch, chain=DeadChain(), docs=[], live=("none", [], {}), empty=True
    )
    answer, sources = await service.chat("What hotels are available in Pokhara?")
    assert len(answer) > 50
    assert_user_safe(answer)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_total_backend_failure_returns_friendly_message(anyio_backend, monkeypatch):
    """TEST 12: every subsystem throws -> logged internally, friendly reply."""
    service = make_service(monkeypatch, chain=DeadChain(), docs=[], live=None, empty=True)
    monkeypatch.setattr(service, "_prepare", _exploding_prepare(service))
    answer, sources = await service.chat("Plan a 3 day trip to Pokhara.")
    assert answer == TEMP_SERVICE_MESSAGE
    assert sources == []
    assert_user_safe(answer)


def _exploding_prepare(service):
    async def _prepare(question, history, trip_context=None):
        raise ConnectionError("Chroma connection refused")

    return _prepare


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_off_topic_never_reaches_dead_gemini(anyio_backend, monkeypatch):
    """TEST 7/8/9: off-topic -> refusal even when Gemini is broken."""
    service = make_service(monkeypatch, chain=DeadChain(), docs=[], live=None, empty=True)
    for question in ("What is Python?", "Who is Cristiano Ronaldo?", "Write my Java assignment."):
        answer, sources = await service.chat(question)
        assert answer in OFF_TOPIC_RESPONSES
        assert answer.strip().endswith("?")
        assert sources == []


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_stream_falls_back_without_llm(anyio_backend, monkeypatch):
    service = make_service(
        monkeypatch,
        chain=DeadChain(),
        docs=[],
        live=("Hotels records: [{'name': 'Lake View'}]", [], HOTEL_RECORDS),
        empty=True,
    )
    events = [e async for e in service.stream("What hotels are available in Pokhara?")]
    text = "".join(token for token, _ in events)
    assert "Lake View Hotel" in text
    assert_user_safe(text)
    assert events[-1][1] is not None


# ------------------------------------------------- endpoint sanitization

def test_endpoint_returns_200_friendly_when_service_explodes(client, monkeypatch):
    async def broken(question, history=None, trip_context=None):
        raise RuntimeError("unexpected Gemini outage")

    monkeypatch.setattr(client.app.state.rag, "chat", broken)
    res = client.post("/api/v1/chat", json={"question": "Plan a 3 day trip to Pokhara."})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == TEMP_SERVICE_MESSAGE
    assert body["sources"] == []
    assert_user_safe(body["answer"])


def test_stream_endpoint_never_emits_technical_errors(client, monkeypatch):
    async def broken(question, history=None, trip_context=None):
        raise RuntimeError("unexpected Gemini outage")
        yield  # pragma: no cover

    monkeypatch.setattr(client.app.state.rag, "stream", broken)
    res = client.post("/api/v1/chat/stream", json={"question": "Hi"})
    assert res.status_code == 200
    text = res.text
    assert "AI service" not in text
    assert "Gemini" not in text
    assert "Traceback" not in text
