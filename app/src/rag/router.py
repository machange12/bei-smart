"""Language detection and intent classification for AgriPulse RAG.

Both are rule-based first. langdetect is only a fallback for language, and
the LLM is only a fallback for intent, because short mixed English/Swahili/
Sheng questions are exactly the input both of those general-purpose tools
are weakest on.
"""

from __future__ import annotations

import re

SWAHILI_MARKERS = [
    "bei", "mahindi", "maharagwe", "ngapi", "wapi", "soko", "mwezi",
    "gharama", "jumla", "reja",
]

_WORD_RE = re.compile(r"[a-zA-Z']+")


def detect_language(text: str) -> str:
    """Return 'en' or 'sw'. Swahili markers are checked before langdetect,
    which is unreliable on short mixed-language queries."""
    words = set(_WORD_RE.findall(text.lower()))
    if any(marker in words or marker in text.lower() for marker in SWAHILI_MARKERS):
        return "sw"

    try:
        from langdetect import detect, LangDetectException

        try:
            code = detect(text)
        except LangDetectException:
            return "en"
        return "sw" if code == "sw" else "en"
    except ImportError:
        return "en"


_INTENT_KEYWORDS = {
    "explain": [
        "why", "how are these", "how are forecasts", "how does the",
        "explain", "kwa nini", "maana ya", "sababu", "how come",
    ],
    "seasonality": [
        "best time to sell", "when should i sell", "when to sell",
        "best month to sell", "when is the best time", "when do prices peak",
        "wakati mzuri wa kuuza", "lini niuze", "muda mzuri wa kuuza",
    ],
    "history": [
        "was the price", "has the price been", "has been", "used to cost",
        "used to be", "historically", "in the past", "past price",
        "history of", "last year", "last month", "ilikuwa", "zamani",
        "hapo awali", "mwaka jana", "mwezi jana", "historia ya",
    ],
    "compare": [
        "compare", "cheapest", "cheaper", "cheap", "which market", "best price",
        "lowest price", "vs", "versus", "nafuu", "gani ina bei", "gani ni bora",
        "wapi bei", "kati ya",
        "most expensive", "highest price", "priciest", "dearest", "ghali",
        "bei ya juu",
    ],
    "alert": [
        "spike", "anomaly", "anomalies", "alert", "unusual", "abnormal",
        "volatility", "volatile", "kawaida", "ghafla", "imepanda", "imeshuka",
        "mabadiliko",
    ],
    "forecast": [
        "forecast", "predict", "price of", "price for", "cost", "will be",
        "itakuwa", "utauzwa", "bei ya", "gharama ya", "utabiri",
    ],
}

# seasonality and history are checked ahead of forecast/compare/alert since
# a question like "when is maize cheapest to sell" or "what was the price"
# would otherwise fall through to the forecast/compare default.
_INTENT_ORDER = ["explain", "seasonality", "history", "compare", "alert", "forecast"]


def classify_intent(text: str, llm_fallback=None) -> str:
    """Rule-based keyword matching first, checked in a fixed priority order
    so a question that mentions both "why" and "price" is treated as an
    explanation request. Falls back to an LLM only when no keyword hits."""
    lower = text.lower()

    for intent in _INTENT_ORDER:
        if any(keyword in lower for keyword in _INTENT_KEYWORDS[intent]):
            return intent

    # A bare past-year mention ("maize price in 2023", or even an
    # out-of-range one like 1990) carries the same intent as an explicit
    # "historically"/"last year" phrase but won't match any fixed keyword
    # above. lookup.get_historical_price reports "no data for that period"
    # gracefully for years outside the actual data span.
    year_match = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", lower)
    if year_match and int(year_match.group(1)) < 2026:
        return "history"

    if llm_fallback is not None:
        try:
            result = llm_fallback(text)
            if result in _INTENT_ORDER:
                return result
        except Exception:
            pass

    # No keyword matched and no usable LLM fallback: a bare price question
    # ("maize Nairobi?") is the most common shape, so default to forecast.
    return "forecast"


_EXPENSIVE_KEYWORDS = [
    "most expensive", "highest price", "priciest", "dearest", "expensive",
    "highest", "ghali", "bei ya juu",
]


def detect_price_direction(text: str) -> str:
    """For a compare-intent question, is this asking for the cheapest
    market or the most expensive one? Defaults to cheapest, since that is
    by far the more common phrasing ("cheapest", "best price", "nafuu")."""
    lower = text.lower()
    if any(keyword in lower for keyword in _EXPENSIVE_KEYWORDS):
        return "expensive"
    return "cheapest"
