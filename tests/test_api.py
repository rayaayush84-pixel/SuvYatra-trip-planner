from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.config import Settings
from app.rag.rag import RAGService
from app.schemas.chat import Source


def test_settings_resolve_project_paths_from_any_cwd(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    settings = Settings()
    assert settings.google_api_key
    assert Path(settings.chroma_dir).is_absolute()
    assert Path(settings.data_dir).is_absolute()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_fails_when_empty(client):
    assert client.get("/ready").status_code == 503


def test_chat_validation(client):
    assert client.post("/api/v1/chat", json={"question": ""}).status_code == 422


def test_chat_ok(client, monkeypatch):
    async def fake_chat(question, history=None, trip_context=None):
        return "Visit Pokhara!", [Source(source="test", snippet="...")]

    monkeypatch.setattr(client.app.state.rag, "chat", fake_chat)
    res = client.post("/api/v1/chat", json={"question": "Where to go?"})
    assert res.status_code == 200
    assert res.json()["answer"] == "Visit Pokhara!"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_live_context_is_explicitly_unavailable_without_api_url(anyio_backend):
    service = RAGService.__new__(RAGService)
    service.settings = SimpleNamespace(suvyatra_api_url=None)

    context, sources, records = await service._get_live_context("Hotels near Pokhara")

    assert "not configured" in context
    assert sources == []
    assert records == {}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_live_context_uses_configured_hotel_and_flight_endpoints(
    anyio_backend, monkeypatch
):
    service = RAGService.__new__(RAGService)
    service.settings = SimpleNamespace(suvyatra_api_url="http://suvyatra.test/api")
    requested_urls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"name": "Verified record", "price": 42, "password": "must not be returned"}]

    def fake_get(url, **kwargs):
        requested_urls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("app.rag.rag.requests.get", fake_get)

    context, sources, records = await service._get_live_context("Hotels and flights near Pokhara")

    assert [url for url, _ in requested_urls] == [
        "http://suvyatra.test/api/hotels",
        "http://suvyatra.test/api/flights",
    ]
    assert all(kwargs["params"] == {"search": "Pokhara"} for _, kwargs in requested_urls)
    assert "Verified record" in context
    assert "must not be returned" not in context
    assert len(sources) == 2
    assert set(records) == {"hotels", "flights"}
