"""Fallback answers for the SuvYatra chatbot.

Fallback hierarchy (no technical detail ever leaves the backend):
- LEVEL 2: Gemini failed but retrieved data exists -> build a structured
  answer directly from that data (names, prices, locations as returned).
- LEVEL 3: travel question, no usable data -> safe general Nepal travel
  guidance with NO invented listings, prices, availability, schedules,
  ratings, addresses, weather, or phone numbers.
- LEVEL 5: genuine system failure -> one friendly temporary-service message.

LEVEL 1 (Gemini + data) and LEVEL 4 (out-of-scope refusal) live in
rag.py / guard.py respectively.
"""

from __future__ import annotations

TEMP_SERVICE_MESSAGE = (
    "I'm having trouble accessing some travel information right now. "
    "Please try your question again in a moment."
)

NO_VERIFIED_INFO_MESSAGE = (
    "I don't currently have verified availability for that specific request, "
    "but I can still help you plan. Tell me your destination and dates and "
    "I'll suggest suitable options to check."
)

_MAX_ITEMS_PER_CATEGORY = 3


def _fmt_amount(value: object, currency: object = "NPR") -> str | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number == float("inf") or number == float("-inf"):
        return None
    text = f"{number:,.0f}" if float(number).is_integer() else f"{number:,.2f}"
    return f"{currency} {text}" if currency else text


def _record_lines(record: dict, kind: str) -> list[str]:
    """Render one retrieved record using ONLY fields the API returned."""
    lines: list[str] = []
    name = record.get("name") or record.get("title")
    if name:
        lines.append(f"   * Name: {name}")
    location = record.get("location") or record.get("city") or record.get("province")
    if location:
        lines.append(f"   * Location: {location}")
    if kind == "hotels":
        price = _fmt_amount(record.get("pricePerNight"), record.get("currency", "NPR"))
        if price:
            lines.append(f"   * Price: {price} per night")
        elif record.get("priceRange"):
            lines.append(f"   * Price: {record['priceRange']}")
        if record.get("rating") is not None:
            lines.append(f"   * Rating: {record['rating']}")
        if record.get("type"):
            lines.append(f"   * Type: {record['type']}")
    elif kind == "restaurants":
        if record.get("priceRange"):
            lines.append(f"   * Price: {record['priceRange']}")
        if record.get("type"):
            lines.append(f"   * Cuisine: {record['type']}")
        if record.get("rating") is not None:
            lines.append(f"   * Rating: {record['rating']}")
    elif kind == "transports":
        price = _fmt_amount(record.get("price"), record.get("currency", "NPR"))
        if price:
            lines.append(f"   * Price: {price}")
        route = record.get("route")
        if not route and (record.get("from") or record.get("to")):
            route = f"{record.get('from', '')} → {record.get('to', '')}".strip(" →")
        if route:
            lines.append(f"   * Route: {route}")
        if record.get("type"):
            lines.append(f"   * Type: {record['type']}")
        if record.get("duration"):
            lines.append(f"   * Duration: {record['duration']}")
    elif kind == "flights":
        price = _fmt_amount(record.get("price"), record.get("currency", "NPR"))
        if price:
            lines.append(f"   * Price: {price}")
        if record.get("airline"):
            lines.append(f"   * Airline: {record['airline']}")
        if record.get("flightNumber"):
            lines.append(f"   * Flight: {record['flightNumber']}")
        if record.get("from") or record.get("to"):
            lines.append(f"   * Route: {record.get('from', '')} → {record.get('to', '')}".strip(" →"))
        if record.get("departureTime") or record.get("arrivalTime"):
            lines.append(
                f"   * Time: {record.get('departureTime', '?')} – {record.get('arrivalTime', '?')}"
            )
    elif kind == "destinations":
        if record.get("description") and isinstance(record["description"], str):
            lines.append(f"   * About: {record['description'][:220]}")
        if record.get("highlights"):
            highlights = record["highlights"]
            if isinstance(highlights, list):
                lines.append(f"   * Highlights: {', '.join(str(h) for h in highlights[:5])}")
            elif highlights:
                lines.append(f"   * Highlights: {highlights}")
    elif kind == "places":
        price = _fmt_amount(record.get("price"), record.get("currency", "NPR"))
        if price:
            lines.append(f"   * Price: {price}")
        if record.get("type"):
            lines.append(f"   * Type: {record['type']}")
        if record.get("description") and isinstance(record["description"], str):
            lines.append(f"   * About: {record['description'][:220]}")
    if record.get("available") is False:
        lines.append("   * Note: currently listed as unavailable")
    return lines


_CATEGORY_TITLES = {
    "hotels": "places to stay",
    "restaurants": "places to eat",
    "transports": "transport options",
    "flights": "flights",
    "destinations": "destinations",
    "places": "activities and places",
}


def build_data_answer(
    live_records: dict[str, list[dict]],
    rag_snippets: list[str] | None = None,
    destination: str | None = None,
) -> str | None:
    """LEVEL 2: structured answer built ONLY from retrieved data.

    Returns None when there is nothing usable to present.
    """
    sections: list[str] = []
    where = f" in {destination.title()}" if destination else ""
    index = 0
    for kind in ("hotels", "restaurants", "transports", "flights", "destinations", "places"):
        records = [r for r in (live_records.get(kind) or []) if isinstance(r, dict)]
        if not records:
            continue
        sections.append(f"Here are some {_CATEGORY_TITLES[kind]}{where}:")
        sections.append("")
        for record in records[:_MAX_ITEMS_PER_CATEGORY]:
            index += 1
            name = record.get("name") or record.get("title") or f"Option {index}"
            sections.append(f"{index}. {name}")
            sections.extend(_record_lines(record, kind))
            sections.append("")
    if sections:
        sections.append("Would you like me to help you choose one?")
        return "\n".join(sections).strip()

    snippets = [s.strip() for s in (rag_snippets or []) if s and s.strip()]
    if snippets:
        lines = [f"Here's what I found in our Nepal travel guide{where}:"]
        lines.append("")
        for snippet in snippets[:3]:
            lines.append(f"- {snippet[:320]}")
            lines.append("")
        lines.append("Want details on any of these?")
        return "\n".join(lines).strip()
    return None


_GENERAL_ANSWERS: dict[str, str] = {
    "hotel": (
        "I couldn't pull a live hotel list just now, but I can still help you plan your stay. "
        "Popular areas are Thamel in Kathmandu and Lakeside in Pokhara, with guesthouses for shorter "
        "stays and mid-range hotels if you want more comfort. Tell me your destination and dates and "
        "I'll fetch current options for you."
    ),
    "restaurant": (
        "I couldn't pull live restaurant listings just now. While travelling in Nepal, try dal bhat, "
        "momos, and Newari or Thakali dishes — tourist hubs like Thamel (Kathmandu) and Lakeside "
        "(Pokhara) have the widest choice. Tell me where you'll be and I'll look up options."
    ),
    "transport": (
        "I couldn't check live transport just now. Between major cities you can usually choose "
        "tourist buses, local buses, jeeps, or domestic flights; for Everest-region trips, flights "
        "to Lukla are the common link. Tell me your route and date and I'll look up what's running."
    ),
    "flight": (
        "I couldn't check live flights just now. Domestic airlines connect Kathmandu with Pokhara, "
        "Bharatpur (Chitwan), Bhairahawa (Lumbini), and Lukla (Everest). Tell me your route and date "
        "and I'll look up current options."
    ),
    "destination": (
        "I couldn't pull destination details just now, but Nepal's most-visited spots are Kathmandu "
        "Valley, Pokhara, Chitwan, Lumbini, and the Everest and Annapurna regions. Tell me how many "
        "days you have and what you enjoy, and I'll suggest an itinerary."
    ),
    "activity": (
        "I couldn't pull live activities just now. Popular options include boating on Phewa Lake, "
        "paragliding in Pokhara, jungle safaris in Chitwan, and day hikes around Kathmandu Valley. "
        "Tell me your destination and I'll look up what's available."
    ),
    "service": (
        "I couldn't verify that facility just now. In tourist areas you'll generally find ATMs, "
        "pharmacies, and clinics; for anything urgent, contact local emergency services or your "
        "hotel front desk. Tell me where you are and what you need, and I'll help locate it."
    ),
    "weather": (
        "I can't verify the current weather right now. In general, October–November and March–April "
        "bring the clearest skies for trekking and sightseeing, while June–September is monsoon "
        "season. Tell me your destination and dates and I'll check the outlook."
    ),
    "budget": (
        "I couldn't work out exact figures just now. A practical approach: set a daily amount covering "
        "stay, food, local transport, and one paid activity, then keep a buffer for permits or "
        "flights. Tell me your destination, days, and group size and I'll help you plan it."
    ),
    "itinerary": (
        "I couldn't build the full itinerary just now. A good starting shape is: arrival + old-city "
        "sightseeing, one nature/adventure day, one flexible day, then departure. Tell me your "
        "destination and number of days and I'll draft the day-by-day plan."
    ),
}


def _general_intent(question: str, intents: list[str] | None) -> str:
    lowered = (question or "").lower()
    if any(w in lowered for w in ("weather", "rain", "temperature", "monsoon", "climate")):
        return "weather"
    if any(w in lowered for w in ("budget", "cost", "expensive", "cheap", "price", "npr")):
        return "budget"
    if any(w in lowered for w in ("itinerary", "plan", "days", "day trip")):
        return "itinerary"
    for intent in intents or []:
        if intent in _GENERAL_ANSWERS:
            return intent
    return "destination"


def general_answer(question: str, intents: list[str] | None = None) -> str:
    """LEVEL 3: safe general guidance with zero invented specifics."""
    return _GENERAL_ANSWERS[_general_intent(question, intents)]
