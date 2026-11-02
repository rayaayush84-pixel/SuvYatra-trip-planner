"""Travel-domain guard for the SuvYatra chatbot.

The guard runs BEFORE retrieval and BEFORE any Gemini call. Clearly off-topic
questions are refused locally with a short redirect so they never reach the
LLM. Valid travel questions (including ambiguous follow-ups resolved through
conversation / trip context) pass through to RAG + Gemini.

How it works (meaning/context based, not single-keyword matching):
- STRONG signals (Nepal places, trip/trek/visa words, travel-intent phrases
  and question shapes) prove travel intent on their own.
- WEAK signals are generic English words (food, price, safe, far, book, map
  ...) that only count when reinforced: by a second weak signal, or by an
  ongoing travel conversation / active trip. Alone they are refused, so
  "What is the price of Bitcoin?" or "Is it safe to invest in crypto?" never
  reach the model.
- Hard off-topic intents (code, maths, jokes, assignments, quantum, ...) are
  refused even if a travel word appears next to them.
- User messages are untrusted: prompt-injection clauses ("ignore your
  previous instructions ...") are stripped before classification, and the
  underlying request is classified on its own merits.
"""

from __future__ import annotations

import re
from typing import Any

OFF_TOPIC_REFUSAL = (
    "I'm SuvYatra's Nepal travel assistant. I can help with Nepal destinations, "
    "itineraries, hotels, transportation, activities, restaurants, budgets, "
    "weather, and travel planning."
)

# Varied polite redirects for off-topic questions. Every entry follows the
# same pattern: (1) briefly acknowledge, (2) explain SuvYatra focuses on
# Nepal travel, (3) ask one relevant travel question. The services pick one
# deterministically per question (see off_topic_response) so the bot does
# not repeat the exact same reply every time, while staying deterministic
# for tests. Keep every entry short, friendly, free of technical detail,
# and ending with a travel question — never answer the off-topic request.
OFF_TOPIC_RESPONSES = [
    "😊 That's outside my area of expertise! I'm here to help you explore Nepal "
    "and plan your travels. Where would you like to go on your next trip?",
    "That's a little outside my travel-planning role! 😊 But I'd love to help "
    "you plan an adventure in Nepal. Are you interested in exploring mountains, "
    "lakes, cultural sites, or hidden gems?",
    "I'm your Nepal travel assistant, so I focus on helping you discover amazing "
    "places! 😊 Are you looking for popular destinations or less-crowded places "
    "to explore in Nepal?",
    "I specialize in Nepal travel rather than that topic. 😊 Would you like help "
    "planning a trip, finding transportation, or discovering activities in Nepal?",
    "Good question, but I stick to Nepal travel and trip planning! 😊 Which "
    "destination, activity, or transport option can I help you with?",
]


def off_topic_response(question: str) -> str:
    """Return a varied polite redirect for an off-topic question.

    The choice is a stable function of the question text (no randomness),
    so repeated asks get a consistent reply while different questions vary
    instead of repeating one canned line. Falls back to OFF_TOPIC_REFUSAL
    for empty input.
    """
    import hashlib

    if not (question or "").strip():
        return OFF_TOPIC_REFUSAL
    digest = hashlib.md5((question or "").strip().lower().encode()).hexdigest()
    return OFF_TOPIC_RESPONSES[int(digest, 16) % len(OFF_TOPIC_RESPONSES)]

# ---------------------------------------------------------------------------
# Prompt-injection handling: these clauses are stripped before classification.
# The remaining text (the actual request) is what gets classified.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+)*instructions?",
    r"disregard\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+)*(rules|instructions?|prompts?|orders)",
    r"forget\s+(everything|all|your|previous|prior).{0,30}(instructions?|rules|prompts?|training)",
    r"override\s+(your\s+|previous\s+|prior\s+)*(instructions?|rules|restrictions)",
    r"you\s+are\s+now\b.{0,60}",
    r"\bact\s+as\b.{0,60}",
    r"\bpretend\s+(you\s+are|to\s+be)\b.{0,60}",
    r"\broleplay\b.{0,60}",
    r"\bjailbreak\b.{0,40}",
    r"\bdan\b.{0,10}mode",
    r"do\s+anything\s+now.{0,40}",
    r"\bbypass\b.{0,30}(rules|restrictions?|safety|filters?)",
    r"\bcircumvent\b.{0,30}(rules|restrictions?|safety|filters?)",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)",
]
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Explicit "not about travel" disclaimers -> always off-topic.
_NON_TRAVEL_DISCLAIMERS = [
    r"unrelat\w*\s+to\s+travell?ing",
    r"not\s+(about|related\s+to|for)\s+travell?",
    r"non[\s-]?travel",
    r"outside\s+(of\s+)?travell?",
    r"nothing\s+to\s+do\s+with\s+travell?",
]
_NON_TRAVEL_RES = [re.compile(p, re.IGNORECASE) for p in _NON_TRAVEL_DISCLAIMERS]

# ---------------------------------------------------------------------------
# Hard off-topic intents: never travel-related, refused even if a travel word
# happens to appear next to them.
# ---------------------------------------------------------------------------
_HARD_OFF_TOPIC = [
    # coding / programming
    r"\bpython\b.{0,25}(program|code|script|function|tutorial|course|develop)",
    r"(program|code|script|tutorial|course)\s+in\s+python\b",
    r"\bjava\b.{0,15}\bprogram\b",
    r"write\s+.*\b(java|javascript|python|c\+\+|c#|php|ruby|go|rust|swift|kotlin|typescript)\b",
    r"write\s+(me\s+|us\s+|a\s+|an\s+|the\s+)*(code|program|script|software|website|web\s*site|web\s*app|app)\b",
    r"how\s+to\s+code\b",
    r"\bcod(e|ing)\s+(a|an|the|my|this)\s+(website|program|app|script|software)\b",
    r"\bprogramm(ing|er)\b.{0,25}(language|tutorial|course|assignment|code|job|interview)",
    r"\bdebug\b.{0,20}(code|program|script|error)",
    r"\bbus\s+error\b",
    r"\bsafari\b.{0,20}(browser|cache|history|tabs?)",
    r"machine\s+learning",
    r"neural\s+network",
    # maths / science homework style
    r"\bmathematics?\b",
    r"\bcalculus\b",
    r"\balgebra\b",
    r"solve\s+(this|the|that|my).{0,20}(math|equation|problem|sum)",
    r"\bcalculate\b.{0,30}\d+",
    r"\bsolve\b.{0,30}\d+\s*[×x\*\+\-\/]\s*\d+",
    r"\b\d+\s*[×÷]\s*\d+",
    r"\b\d+\s+(times|multiplied\s+by|divided\s+by)\s+\d+",
    r"^what\s+is\s+\d+\s*[x×\*\+\-\/]\s*\d+",
    r"quantum\s+(mechanics|physics|computing)",
    r"theory\s+of\s+relativity",
    r"\btour\s+de\s+france\b",
    # jokes / entertainment
    r"\bjoke(s)?\b",
    r"make\s+me\s+laugh",
    r"funny\s+(story|stories)",
    # schoolwork
    r"\bassignment(s)?\b",
    r"\bhomework\b",
    r"\bbook\s+report\b",
    r"write\s+my\s+(college|school|university|class).{0,20}(essay|thesis|paper|work)",
    r"do\s+my\s+(homework|assignment)",
    # celebrity / non-travel trivia
    r"\b(cristiano\s+ronaldo|elon\s+musk|lionel\s+messi)\b",
]
_HARD_RES = [re.compile(p, re.IGNORECASE) for p in _HARD_OFF_TOPIC]

# ---------------------------------------------------------------------------
# Soft off-topic signals: refused when there is no travel evidence.
# ---------------------------------------------------------------------------
_SOFT_OFF_TOPIC = [
    r"\bpython\b",
    r"\bjava\b",
    r"\brecipe(s)?\b",
    r"\bwho\s+(is|was|are|were)\b",
    r"\bcapital\s+of\b",
    r"\bphysics\b",
    r"\bchemistry\b",
    r"\bbiology\b",
    r"\bhistory\s+of\s+(france|germany|usa|america|england|china|japan|russia)\b",
]
_SOFT_RES = [re.compile(p, re.IGNORECASE) for p in _SOFT_OFF_TOPIC]

# ---------------------------------------------------------------------------
# Travel signals
# ---------------------------------------------------------------------------
_NEPAL_PLACES = [
    "nepal", "nepali", "nepalese", "himalaya", "himalayas", "terai",
    "kathmandu", "bhaktapur", "bhadgaon", "patan", "lalitpur", "kirtipur",
    "thamel", "lazimpat", "boudha", "boudhanath", "swayambhu", "swayambhunath",
    "pashupati", "pashupatinath", "durbar square", "narayanhiti", "kopan",
    "pharping", "dakshinkali", "godavari", "nagarjun", "shivapuri", "kakani",
    "nagarkot", "dhulikhel", "chandragiri", "champadevi",
    "pokhara", "lakeside", "phewa", "fewa", "begnas", "rupa lake",
    "sarangkot", "bat cave", "mahendra cave", "davis falls", "gupteshwor",
    "chitwan", "chitawan", "sauraha", "meghauli", "narayani", "rapti",
    "lumbini", "kapilvastu", "tulsipur",
    "everest", "sagarmatha", "lukla", "namche", "tengboche", "dingboche",
    "gorak shep", "kala patthar", "gokyo", "renjo",
    "annapurna", "machhapuchhre", "fishtail", "mardi", "ghandruk",
    "chhomrong", "ghorepani", "poon hill", "poonhill", "ulleri", "tatopani",
    "jomsom", "kagbeni", "marpha", "muktinath", "mustang", "manang",
    "tilicho", "manaslu", "langtang", "helambu", "gosaikunda",
    "dolpo", "shey", "phoksundo", "rara", "khaptad",
    "bandipur", "gorkha", "tansen", "palpa", "gulmi", "syangja",
    "baglung", "beni", "daman", "chitlang", "markhu", "kulekhani",
    "charikot", "dolakha", "kalinchowk", "kuri village",
    "pathibhara", "taplejung", "ilam", "dharan", "dhankuta",
    "butwal", "bhairahawa", "siddharthanagar", "nepalgunj", "nepalganj",
    "dhangadhi", "janakpur", "janakpurdham", "biratnagar", "birgunj",
    "hetauda", "damauli", "tanahun",
    "bardia", "bardiya", "shuklaphanta", "koshi tappu", "koshi",
    "trishuli", "bhote koshi", "sunkoshi", "karnali", "kaligandaki",
    "marsyangdi", "seti river", "budhi gandaki",
    "tims", "acap", "unesco",
    "buddha", "buddhist", "thakali", "newari", "tharu",
    "dal bhat", "dhalbhat", "momo",
]

# Distinctive multi-word travel phrases matched by substring (strong).
_TRAVEL_PHRASES = [
    "hidden gem", "hidden gems", "off the beaten", "offbeat",
    "best time", "google maps", "tea house", "teahouse",
    "guest house", "guesthouse", "home stay", "homestay",
    "jungle safari", "elephant safari", "cable car", "cablecar",
    "travel agency", "travel agent", "tour package", "travel guide",
    "tourist attraction", "tourist attractions",
    "things to do", "places to visit", "place to visit",
    "check in", "check-in", "checkin", "packing list",
    "money exchange", "fuel station", "petrol pump",
    "health post", "travel insurance", "travel date", "travel plan",
    "trip plan", "my trip", "our trip", "this trip",
    "national park", "wildlife reserve", "conservation area",
    "world heritage", "durbar square",
    "bus park", "bus station", "tourist bus", "local bus",
    "domestic flight", "international flight", "yet airlines",
    "buddha air", "shree airlines", "nepal airlines",
    "trekking permit", "trekking route",
]

# STRONG single words: unambiguous travel-domain or Nepal-specific terms.
# One of these is enough to treat the question as travel-related.
_STRONG_WORDS = [
    "travel", "travelling", "traveling", "traveller", "travellers",
    "traveler", "travelers",
    "trip", "trips", "itinerary", "itineraries",
    "trek", "treks", "trekking",
    "tour", "tours", "tourism", "tourist", "tourists",
    "vacation", "vacations", "holiday", "holidays", "getaway",
    "expedition", "expeditions", "pilgrimage",
    "safari", "safaris",
    "hostel", "hostels",
    "accommodation", "accommodations",
    "transport", "transportation", "jeep", "jeeps", "ropeway",
    "airport", "airports", "airline", "airlines",
    "visa", "visas", "passport", "passports", "permit", "permits",
    "weather",
    "monsoon",
    "destination", "destinations", "sightseeing",
    "stupa", "stupas", "monastery", "monasteries",
    "paragliding", "rafting", "kayaking", "bungee", "zipline", "ultralight",
    "wildlife",
    "booking", "bookings", "reservation", "reservations",
    "npr", "rupee", "rupees", "airfare",
    "dashain", "tihar", "holi", "teej",
]

# WEAK single words: generic English with non-travel meanings. Each counts
# for less, so one alone (e.g. "price" in "price of Bitcoin") is refused;
# two together, or one inside a travel conversation, passes.
_WEAK_WORDS = [
    "hotel", "hotels", "motel", "motels", "resort", "resorts",
    "lodge", "lodges",
    "restaurant", "restaurants",
    "food", "foods", "meal", "meals", "eat", "eating", "dining", "cuisine",
    "flight", "flights",
    "bus", "buses", "car", "cars", "bike", "bikes",
    "taxi", "taxis", "cab", "cabs", "motorbike", "motorcycle", "scooter",
    "train", "trains",
    "stay", "stays", "staying",
    "book", "guide", "guides",
    "map", "maps", "direction", "directions", "navigate",
    "route", "routes", "distance", "distances", "far",
    "season", "seasons", "climate", "temperature", "rainfall", "snowfall",
    "pack", "packing", "luggage", "baggage", "backpack",
    "insurance",
    "safety", "safe",
    "hospital", "hospitals", "medical", "clinic", "clinics",
    "doctor", "doctors", "pharmacy", "emergency", "altitude",
    "atm", "atms", "exchange", "currency", "cash",
    "petrol", "fuel",
    "budget", "budgets", "price", "prices", "pricing",
    "cost", "costs", "fare", "fares", "fee", "fees",
    "expensive", "cheap", "cheapest", "cheaper", "nearest", "closest",
    "hike", "hikes", "hiking",
    "activity", "activities", "adventure", "adventures",
    "attraction", "attractions",
    "boating", "cycling",
    "rhino", "rhinos",
    "viewpoint", "viewpoints", "sunrise", "sunset",
    "culture", "cultural", "heritage",
    "temple", "temples", "museum", "museums",
    "festival", "festivals",
    "bazaar", "souvenir", "souvenirs",
]

# Question shapes that are travel-related even with few keywords.
# Cost/distance/safety adjectives are deliberately NOT standalone patterns:
# they must co-occur with a travel anchor (place, trip word, measure), so
# "Is Pokhara expensive?" passes while "Are diamonds expensive?" is refused.
_TRAVEL_QUESTION_PATTERNS = [
    r"\bplan\b.{0,50}\btrip\b",
    r"\bplan\s+(me|my|a|an|our|the)\b",
    r"where\s+(should|can|do)\s+i\s+stay\b",
    r"where\s+to\s+stay\b",
    r"place(s)?\s+to\s+stay\b",
    r"how\s+(do|can|should)\s+i\s+(get|travel|go|reach)\b",
    r"how\s+to\s+(get|travel|go|reach)\b",
    # distance/cost only with a travel anchor (place, trip word, measure)
    r"how\s+far\b.{0,60}(km|kilomete?r?s?|miles?|hours?|minutes?|mins?|days?|nepal|trek|travel|airport|station|hotel|lake|temple|park|city|town|village|pokhara|kathmandu|chitwan|everest|annapurna|lumbini|mustang)",
    r"distance\s+(between|from)\b.{0,60}(nepal|kathmandu|pokhara|chitwan|everest|annapurna|lumbini|km|kilomete?rs?|miles?|city|town|airport|hotel|trek)",
    ".{0,0}",  # placeholder replaced below
    r"how\s+much\b.{0,50}(trip|trek|tour|travel|flight|hotel|visa|permit|guide|porter|nepal|package)",
    r"cost\s+of\s+(the|a|my|this)\s+(trip|trek|tour|travel)\b",
    r"best\s+time\s+(to|for)\b",
    r"when\s+(to|should\s+i)\s+(visit|go|trek|travel)\b",
    r"when\s+is\s+the\s+best\b",
    r"what\s+to\s+(pack|bring|carry)\b.{0,60}(trek|nepal|trip|travel|tour|hik(e|ing)|everest|annapurna|monsoon|winter|summer|spring|flight|luggage|backpack|pokhara|kathmandu|chitwan|ebc|abc|packing\s+list)",
    r"what\s+should\s+i\s+(pack|bring|carry|wear)\b.{0,60}(trek|nepal|trip|travel|tour|hik(e|ing)|everest|annapurna|monsoon|winter|summer|spring|flight|luggage|backpack|pokhara|kathmandu|chitwan|ebc|abc|for\s+my|packing\s+list)",
    r"packing\s+list\b",
    r"which\s+hotel\b",
    r"suggest\s+(a|me|some)\s+(hotel|itinerary|trip|place)",
    r"recommend\s+.*(hotel|trip|itinerary|place|restaurant|trek)",
    r"hotel.*\bbetter\b",
    r"what\s+about\s+the\s+weather\b",
    r"how('s|\u2019s|\s+is)\s+the\s+weather\b",
    r"weather\s+like\b",
    r"weather\s+in\b",
    r"things\s+to\s+do\b",
    r"good\s+for\s+trekking\b",
    r"do\s+i\s+need\s+(a\s+)?(visa|permit|insurance|guide)\b",
    r"my\s+\w*\s*trip\b",
    r"for\s+my\s+trip\b",
    r"\bfor\s+(a\s+)?(family|kids|children|couple|honeymoon|group)\s+(in|trip|visit|holiday)",
    r"\bjeep\s+or\s+bus\b",
    r"\bbus\s+or\s+jeep\b",
]
# Drop the placeholder entry used to keep diffs readable.
_TRAVEL_QUESTION_PATTERNS = [p for p in _TRAVEL_QUESTION_PATTERNS if p != ".{0,0}"]
_TRAVEL_QUESTION_RES = [re.compile(p, re.IGNORECASE) for p in _TRAVEL_QUESTION_PATTERNS]


def _word_res(words: list[str]) -> list[re.Pattern]:
    return [re.compile(r"\b" + re.escape(w) + r"s?\b", re.IGNORECASE) for w in words]


_STRONG_WORD_RES = _word_res(_STRONG_WORDS)
_WEAK_WORD_RES = _word_res(_WEAK_WORDS)

# Follow-up shapes: travel-related ONLY inside a travel conversation
# ("What can I do there?", "Anything cheaper?"). Checked after the soft
# off-topic list, so "What about Python?" style questions still refuse.
_FOLLOWUP_PATTERNS = [
    r"what can i do (there|here|around|nearby)",
    r"how far is it\b",
    r"is it (expensive|cheap|cheaper|safe|good|worth(\s+it)?|far|open|nearby|close)",
    r"anything (cheaper|else|nearby|near|good)",
    r"what about ",
    r"how about ",
    r"^any\s+\w+\s*\??$",
    r"how (do|can) (i|we) get there\b",
    r"\band\s+(the|for|in|near|around)\b",
]
_FOLLOWUP_RES = [re.compile(p, re.IGNORECASE) for p in _FOLLOWUP_PATTERNS]

_GREETING_PATTERNS = [
    # Plain greetings, optionally addressed to SuvYatra ("Hi SuvYatra!").
    # Anchored to the full message so "Hi, teach me Python" still refuses
    # via the hard off-topic check that runs before this.
    r"^(hi|hey|hello|namaste|namaskar)(\s+suvyatra)?[\s!.?,]*$",
    r"^good\s+(morning|afternoon|evening)(\s+suvyatra)?[\s!.?,]*$",
    r"^how\s+are\s+you(\s+doing)?(\s+suvyatra)?[\s?.!]*$",
    r"^(thanks?(\s+you)?|thank\s+you|bye|bye-bye|see\s+you)(\s+suvyatra)?[\s!.?,]*$",
]
_GREETING_RES = [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS]


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _history_text(history: list[Any] | None) -> str:
    if not history:
        return ""
    return "\n".join(_message_text(m) for m in history)


def format_trip_context(trip_context: Any) -> str:
    """Render an optional trip-context object into prompt/guard text."""
    if not trip_context:
        return ""
    if isinstance(trip_context, str):
        return trip_context.strip()
    if isinstance(trip_context, dict):
        items = trip_context
    else:
        try:
            items = trip_context.model_dump(exclude_none=True)
        except AttributeError:
            try:
                items = dict(vars(trip_context))
            except TypeError:
                return str(trip_context).strip()
    parts = []
    labels = {
        "destination": "Destination",
        "start_date": "Start date",
        "end_date": "End date",
        "dates": "Dates",
        "travellers": "Travellers",
        "travelers": "Travellers",
        "budget": "Budget",
        "preferences": "Preferences",
        "hotels": "Selected hotels",
        "selected_hotels": "Selected hotels",
        "activities": "Selected activities",
        "selected_activities": "Selected activities",
        "transportation": "Transportation",
        "transport": "Transportation",
        "itinerary": "Itinerary",
    }
    for key, value in items.items():
        if value in (None, "", [], {}):
            continue
        label = labels.get(key, str(key).replace("_", " ").title())
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{label}: {value}")
    return "; ".join(parts)


def _strip_injection(text: str) -> str:
    cleaned = text
    for pattern in _INJECTION_RES:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_strong_signal(text: str) -> bool:
    lowered = text.lower()
    for place in _NEPAL_PLACES:
        if place in lowered:
            return True
    for phrase in _TRAVEL_PHRASES:
        if phrase in lowered:
            return True
    for word_re in _STRONG_WORD_RES:
        if word_re.search(text):
            return True
    for question_re in _TRAVEL_QUESTION_RES:
        if question_re.search(text):
            return True
    return False


def _count_weak_signals(text: str) -> int:
    return sum(1 for word_re in _WEAK_WORD_RES if word_re.search(text))


def _has_soft_off_topic(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SOFT_RES)


def classify(
    question: str,
    history: list[Any] | None = None,
    trip_context: Any = None,
) -> str:
    """Return "travel" or "off_topic" for a user question.

    Scoring (meaning/context based, not single-keyword matching):
    - one STRONG signal (Nepal place, trip/trek/visa word, travel phrase or
      anchored travel question shape) -> travel;
    - two or more WEAK generic words (food + price, bus + fare, ...) -> travel;
    - one WEAK word is only travel inside an ongoing travel conversation or
      with an active trip ("Suggest a hotel." after Pokhara talk);
    - bare follow-up shapes ("What can I do there?", "Anything cheaper?")
      are travel only with travel context behind them;
    - a lone generic word with no travel evidence ("price of Bitcoin") is
      refused and never reaches the model.
    """
    text = (question or "").strip()
    if not text:
        return "off_topic"
    core = _strip_injection(text)

    for pattern in _NON_TRAVEL_RES:
        if pattern.search(core):
            return "off_topic"

    for pattern in _HARD_RES:
        if pattern.search(core):
            return "off_topic"

    if _has_strong_signal(core):
        return "travel"

    weak_count = _count_weak_signals(core)
    if weak_count >= 2:
        return "travel"

    if _has_soft_off_topic(core):
        return "off_topic"

    context = "\n".join(
        part for part in (_history_text(history), format_trip_context(trip_context)) if part
    )
    if context and _has_strong_signal(context):
        if any(pattern.search(core) for pattern in _FOLLOWUP_RES):
            # Bare follow-up inside a travel conversation -> travel.
            return "travel"
        if weak_count >= 1:
            # Ambiguous follow-up inside a travel conversation -> travel.
            return "travel"

    for pattern in _GREETING_RES:
        if pattern.match(core):
            return "travel"

    # No travel evidence anywhere -> off-topic. The hardened system
    # prompt remains as a second layer of defence for anything borderline
    # that reaches the model.
    return "off_topic"


def is_travel_related(
    question: str,
    history: list[Any] | None = None,
    trip_context: Any = None,
) -> bool:
    return classify(question, history, trip_context) == "travel"
