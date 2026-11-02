"""Understand a travel question: extract location entities, price hints, and
service intent so retrieval can search the right fields instead of relying
on a single raw keyword.

Used by live-database lookup, RAG query expansion, and the data-driven
fallback answer builder. Pure local code — never calls the network.
"""

from __future__ import annotations

import re
from typing import Any

# Reuse the curated Nepal gazetteer from the domain guard so entity
# detection and relevance stay consistent.
from app.rag.guard import _NEPAL_PLACES

# Multi-word places first so "poon hill" wins over shorter overlaps.
_SORTED_PLACES = sorted(_NEPAL_PLACES, key=len, reverse=True)

# Route intent -> SuvYatra API routes to query.
INTENT_ROUTES: dict[str, tuple[str, ...]] = {
    "hotel": ("hotels",),
    "restaurant": ("restaurants",),
    "transport": ("transports",),
    "flight": ("flights",),
    "destination": ("destinations",),
    "activity": ("places",),
    "service": ("places",),
}

_INTENT_PATTERNS: dict[str, list[str]] = {
    "hotel": [
        r"\bhotels?\b", r"\bmotels?\b", r"\bhostels?\b", r"\bresorts?\b",
        r"\blodges?\b", r"\bhome\s*stays?\b", r"\btea\s*houses?\b",
        r"\bstay\b", r"\bstays\b", r"\baccommodation",
        r"\brooms?\b", r"where\s+(should|can|do)\s+i\s+stay\b",
        r"where\s+to\s+stay\b", r"place(s)?\s+to\s+stay\b",
    ],
    "restaurant": [
        r"\brestaurants?\b", r"\bfoods?\b", r"\bmeals?\b", r"\beat\b",
        r"\bdining\b", r"\bcuisines?\b", r"\bmomo\b", r"\bdal\s*bhat\b",
    ],
    "transport": [
        r"\btransports?\b", r"\btransportation\b", r"\btravell?(ing|er|ers)?\b",
        r"\bbus(es)?\b",
        r"\bjeeps?\b", r"\bcars?\b", r"\bbikes?\b", r"\bmotorbikes?\b",
        r"\btaxi(s)?\b", r"\bcable\s*cars?\b", r"\bropeways?\b",
        r"\bdrive\b", r"\bride\b", r"\broutes?\b", r"\bdistances?\b",
        r"\bhow\s+far\b", r"\bflights?\b.*\b(from|to)\b",
    ],
    "flight": [
        r"\bflights?\b", r"\bairports?\b", r"\bairlines?\b",
    ],
    "destination": [
        r"\bdestinations?\b", r"\bplaces?\s+to\s+visit\b",
        r"\bhidden\s+gems?\b", r"\battractions?\b", r"\bsightseeing\b",
        r"\bwhere\s+(should|can)\s+(i|we)\s+go\b", r"\bwhat\s+to\s+(do|see|visit)\b",
        r"\bthings\s+to\s+do\b",
    ],
    "activity": [
        r"\bactivit(y|ies)\b", r"\btreks?\b", r"\btrekking\b",
        r"\bhik(e|es|ing)\b", r"\bparagliding\b", r"\brafting\b",
        r"\bboating\b", r"\bsafari\b", r"\bbungee\b", r"\bzipline\b",
        r"\badventure\b",
    ],
    "service": [
        r"\bhospitals?\b", r"\bmedical\b", r"\bpharmac(y|ies)\b",
        r"\batms?\b", r"\bpetrol\b", r"\bfuel\b", r"\bmoney\b",
        r"\bexchange\b", r"\bsafety\b", r"\bsafe\b",
    ],
}
_INTENT_RES: dict[str, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in _INTENT_PATTERNS.items()
}

_BUDGET_PATTERNS = [
    re.compile(r"under\s+npr\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"npr\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"rs\.?\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"budget\s+(?:of\s+)?(?:npr\s*)?([\d,]+)", re.IGNORECASE),
]

_CHEAP_HINT = re.compile(r"\b(cheap|cheapest|cheaper|budget|affordable|low\s*cost)\b", re.IGNORECASE)
_HIDDEN_HINT = re.compile(r"\bhidden\b", re.IGNORECASE)
_FAMILY_HINT = re.compile(r"\b(family|families|kids|children|couple|honeymoon|group)\b", re.IGNORECASE)


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def find_places(text: str) -> list[str]:
    """Nepal place entities mentioned in text (longest match wins)."""
    lowered = f" {(text or '').lower()} "
    found: list[str] = []
    for place in _SORTED_PLACES:
        if place in lowered:
            found.append(place)
            lowered = lowered.replace(place, " ")
    return found


def find_budget_npr(text: str) -> int | None:
    """Explicit NPR budget cap mentioned in the question, if any."""
    for pattern in _BUDGET_PATTERNS:
        match = pattern.search(text or "")
        if match:
            try:
                value = int(match.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
            if 0 < value <= 10_000_000:
                return value
    return None


def detect_intents(text: str) -> list[str]:
    """Service intents in priority order (hotel before generic, etc.)."""
    found = [intent for intent, res in _INTENT_RES.items() if any(r.search(text or "") for r in res)]
    priority = ["hotel", "restaurant", "flight", "transport", "activity", "destination", "service"]
    return sorted(found, key=priority.index)


def history_places(history: list[Any] | None, limit: int = 3) -> list[str]:
    """Most recent place entities from the conversation (for follow-ups)."""
    if not history:
        return []
    seen: list[str] = []
    for message in reversed(history[-6:]):
        for place in find_places(_message_text(message)):
            if place not in seen:
                seen.append(place)
            if len(seen) >= limit:
                return seen
    return seen


def expand_query(question: str, history: list[Any] | None = None, trip_context: Any = None) -> str:
    """Build a retrieval query: the question plus unresolved context.

    Short follow-ups ("Any cheap hotels?") gain the recent destination
    ("... in Pokhara") so both vector search and live lookup stay on topic.
    """
    parts = [(question or "").strip()]
    places_in_question = set(find_places(question))
    for place in history_places(history):
        if place not in places_in_question:
            parts.append(place)
            break
    try:
        from app.rag.guard import format_trip_context

        trip_text = format_trip_context(trip_context)
        for place in find_places(trip_text):
            if place not in places_in_question and len(parts) < 3:
                parts.append(place)
    except Exception:
        pass
    return " ".join(p for p in parts if p)


def describe_intent_flags(question: str) -> dict[str, bool]:
    lowered = (question or "").lower()
    return {
        "cheap": bool(_CHEAP_HINT.search(lowered)),
        "hidden": bool(_HIDDEN_HINT.search(lowered)),
        "family": bool(_FAMILY_HINT.search(lowered)),
    }
