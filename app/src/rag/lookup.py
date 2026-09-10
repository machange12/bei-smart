"""Deterministic dataframe lookups for AgriPulse RAG.

No LLM calls and no embeddings live in this module. Every number returned
here comes straight from a CSV/JSON filter. Forecast rows for different
markets are near-identical in text ("Maize, Nairobi, Wholesale, 24.62 KES/kg"
vs "Maize, Nakuru, Wholesale, 22.10 KES/kg") and embed too similarly for
vector search to reliably pick the right one — so entity extraction plus
exact dataframe filtering is the only path to a numeric answer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

FUZZY_THRESHOLD = 85

# ---------------------------------------------------------------------------
# Static vocabularies
# ---------------------------------------------------------------------------

SWAHILI_COMMODITY_MAP = {
    "mahindi": "Maize",
    # generic Swahili "beans" -- routes to the larger of the two covered
    # bean series since plain "Beans" no longer exists post-rebuild.
    "maharagwe": "Beans (mixed)",
    "mtama": "Sorghum",
    "unga": "Maize meal",
    "viazi": "Potatoes",
}

# Major Kenyan counties/towns that this system does NOT cover. Listing them
# explicitly means a real, well-known place (Kiambu) is reported as "not
# covered" rather than silently fuzzy-matching to some unrelated market that
# happens to share a few letters.
#
# Updated after the FEWS-primary rebuild: Baringo, Bomet, Busia, Garissa,
# Kajiado, Kiambu, Kilifi, Laikipia, Lamu, Migori, Narok, Nyeri, Samburu,
# Siaya, Taita Taveta, Tharaka Nithi and Turkana moved to COVERED_MARKETS
# and were removed from this list. The remainder are still genuinely
# uncovered by exact market name, even where a same-county town (e.g.
# Butere/Mumias in Kakamega County) is now covered.
KNOWN_UNCOVERED_MARKETS = [
    "Kakamega", "Bungoma", "Machakos", "Kericho", "Kisii", "Murang'a",
    "Embu", "Homa Bay", "Kwale", "Nyandarua", "Kirinyaga",
    "West Pokot", "Trans Nzoia", "Nandi", "Vihiga",
]

# Colloquial short forms that fuzzy matching alone can't bridge -- e.g.
# "nakuru" vs "Nakuru Town" scores 71 (below the 85 threshold), so a plain
# fuzzy match would incorrectly report Nakuru as uncovered even though
# it is. Checked as a direct word match before fuzzy matching runs.
MARKET_ALIASES = {
    "nakuru": "Nakuru Town",
    "machakos": "Machakos Town",
    "embu": "Embu-Mbeere",
}

# A handful of well-known commodities this system does not model. Not
# exhaustive -- just enough that "rice" resolves to a name we can report as
# uncovered, rather than falling through as an unrecognised token.
KNOWN_UNCOVERED_COMMODITIES = ["Rice", "Potatoes", "Wheat flour", "Sugar", "Milk"]

_PRICETYPE_KEYWORDS = {
    "wholesale": "Wholesale",
    "jumla": "Wholesale",
    "retail": "Retail",
    "reja": "Retail",  # matches "reja" and "reja-reja"
}

_WORD_RE = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------
# Data loading (once, at import time)
# ---------------------------------------------------------------------------


def _load():
    forecasts = pd.read_csv(
        DATA_DIR / "production_forecasts.csv", parse_dates=["forecast_date"]
    )
    anomalies = pd.read_csv(DATA_DIR / "price_anomalies.csv", parse_dates=["latest_date"])
    volatility = pd.read_csv(DATA_DIR / "market_volatility.csv")
    model_metrics = pd.read_csv(DATA_DIR / "model_metrics.csv")
    historical = pd.read_csv(
        DATA_DIR / "bei_smart_forecasting_final.csv",
        usecols=["market", "admin1"],
    )
    market_region_map = (
        historical.dropna(subset=["admin1"])
        .drop_duplicates(subset=["market"])
        .set_index("market")["admin1"]
        .to_dict()
    )
    return forecasts, anomalies, volatility, model_metrics, market_region_map


FORECASTS, ANOMALIES, VOLATILITY, MODEL_METRICS, MARKET_REGION_MAP = _load()

COVERED_MARKETS = sorted(FORECASTS["market"].unique().tolist())
COVERED_COMMODITIES = sorted(FORECASTS["commodity"].unique().tolist())

_ALL_MARKET_CHOICES = COVERED_MARKETS + KNOWN_UNCOVERED_MARKETS
_ALL_COMMODITY_CHOICES = COVERED_COMMODITIES + KNOWN_UNCOVERED_COMMODITIES

# Unreliable series: present in model_metrics.csv but failed the 25% MAPE
# reliability threshold, so they are not served as live forecasts.
_UNRELIABLE = MODEL_METRICS[MODEL_METRICS["reliable"] == False]  # noqa: E712


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------


def _windows(text: str, max_len: int = 3) -> list[str]:
    """3-, 2- and 1-word sliding windows over the words in `text`, longest
    first. Longest-first ordering matters: "maize meal" and "Maize" can
    both score a tied 100 against their respective choices, and checking
    the more specific multi-word phrase first means the tie-break below
    (strict `>`) keeps the more specific match instead of the shorter one."""
    words = _WORD_RE.findall(text)
    out = []
    for size in range(max_len, 0, -1):
        for i in range(len(words) - size + 1):
            out.append(" ".join(words[i : i + size]))
    return out


def _best_fuzzy_match(text: str, choices: list[str], threshold: int = FUZZY_THRESHOLD):
    """Return (matched_choice, score) for the best window-vs-choice match, or (None, 0)."""
    best_choice, best_score = None, 0
    for window in _windows(text):
        for choice in choices:
            score = fuzz.ratio(window.lower(), choice.lower())
            if score > best_score:
                best_score, best_choice = score, choice
    if best_score >= threshold:
        return best_choice, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _extract_commodity(question: str) -> str | None:
    lower = question.lower()
    for word in _WORD_RE.findall(lower):
        if word in SWAHILI_COMMODITY_MAP:
            return SWAHILI_COMMODITY_MAP[word]
    match, _ = _best_fuzzy_match(question, _ALL_COMMODITY_CHOICES)
    return match


def _extract_market(question: str) -> str | None:
    # Aliases are added as extra fuzzy-matchable choices (not just an exact
    # word check) so a typo on the colloquial form -- "Nakru" for Nakuru --
    # still resolves, the same way typos on official names do.
    match, _ = _best_fuzzy_match(question, _ALL_MARKET_CHOICES + list(MARKET_ALIASES.keys()))
    if match is not None:
        return MARKET_ALIASES.get(match.lower(), match)
    return None


def _extract_pricetype(question: str) -> str | None:
    lower = question.lower()
    for keyword, canonical in _PRICETYPE_KEYWORDS.items():
        if keyword in lower:
            return canonical
    return None


def _extract_horizon(question: str) -> int:
    lower = question.lower()
    if "miezi sita" in lower or "six month" in lower or "6 month" in lower:
        return 6
    if "mwaka ujao" in lower or "next year" in lower or re.search(r"\b12\s*month", lower):
        return 12
    if "mwezi ujao" in lower or "next month" in lower or re.search(r"\b3\s*month", lower):
        return 3
    if re.search(r"\byear\b|\bmwaka\b", lower):
        return 12
    return 3


def extract_entities(question: str) -> dict:
    """Pull commodity, market, pricetype and horizon out of free text.

    Values are the best fuzzy-matched canonical names (which may or may not
    be *covered* -- that classification happens in get_forecast) or None
    when nothing clears the match threshold.
    """
    return {
        "commodity": _extract_commodity(question),
        "market": _extract_market(question),
        "pricetype": _extract_pricetype(question),
        "horizon_months": _extract_horizon(question),
    }


# ---------------------------------------------------------------------------
# Forecast lookup
# ---------------------------------------------------------------------------


def _available() -> dict:
    return {"commodities": COVERED_COMMODITIES, "markets": COVERED_MARKETS}


def _combos_for_market(market: str) -> list[tuple[str, str]]:
    sub = FORECASTS[FORECASTS["market"] == market][["commodity", "pricetype"]]
    return sorted(set(map(tuple, sub.values.tolist())))


def _combos_for_commodity(commodity: str) -> list[tuple[str, str]]:
    sub = FORECASTS[FORECASTS["commodity"] == commodity][["market", "pricetype"]]
    return sorted(set(map(tuple, sub.values.tolist())))


def _pricetypes_for(commodity: str, market: str) -> list[str]:
    sub = FORECASTS[(FORECASTS["commodity"] == commodity) & (FORECASTS["market"] == market)]
    return sorted(sub["pricetype"].unique().tolist())


def _unreliable_row(commodity: str, market: str, pricetype: str | None):
    sub = _UNRELIABLE[(_UNRELIABLE["commodity"] == commodity) & (_UNRELIABLE["market"] == market)]
    if pricetype:
        exact = sub[sub["pricetype"] == pricetype]
        if not exact.empty:
            return exact.iloc[0]
    if not sub.empty:
        return sub.iloc[0]
    return None


def get_forecast(entities: dict) -> dict:
    """Resolve a forecast request to exactly one of five statuses.

    Statuses: found, no_market, no_commodity, no_combination, unreliable.
    Every branch includes `available` (covered commodities/markets) and an
    `alternatives` list so a caller never has to guess what else to try.
    """
    commodity = entities.get("commodity")
    market = entities.get("market")
    pricetype = entities.get("pricetype")
    horizon_months = entities.get("horizon_months") or 3

    available = _available()

    if not market or market not in COVERED_MARKETS:
        known_place = bool(market) and market in KNOWN_UNCOVERED_MARKETS
        return {
            "status": "no_market",
            "message": (
                f"{market} is a real place but is not one of the markets this system covers."
                if known_place
                else f"'{market or 'that market'}' was not recognised as a covered market."
            ),
            "requested": {"commodity": commodity, "market": market, "pricetype": pricetype},
            "alternatives": COVERED_MARKETS,
            "available": available,
        }

    if not commodity or commodity not in COVERED_COMMODITIES:
        return {
            "status": "no_commodity",
            "message": f"'{commodity or 'that commodity'}' is not one of the commodities modelled here.",
            "requested": {"commodity": commodity, "market": market, "pricetype": pricetype},
            "alternatives": COVERED_COMMODITIES,
            "available": available,
        }

    combos = _pricetypes_for(commodity, market)
    if pricetype is None:
        # Not specified by the question: default to whichever pricetype
        # this series is actually tracked as.
        resolved_pricetype = combos[0] if combos else None
    elif pricetype in combos:
        resolved_pricetype = pricetype
    else:
        # Explicitly asked for a pricetype this series does not have --
        # this must fall through to no_combination, never silently swap in
        # the pricetype that *is* available.
        resolved_pricetype = None

    if resolved_pricetype and resolved_pricetype in combos:
        rows = FORECASTS[
            (FORECASTS["commodity"] == commodity)
            & (FORECASTS["market"] == market)
            & (FORECASTS["pricetype"] == resolved_pricetype)
            & (FORECASTS["horizon_months"] == horizon_months)
        ]
        if not rows.empty:
            row = rows.iloc[0]
            return {
                "status": "found",
                "commodity": commodity,
                "market": market,
                "region": row["region"],
                "pricetype": resolved_pricetype,
                "pricetype_assumed": pricetype is None,
                "horizon_months": int(row["horizon_months"]),
                "forecast_price_kes": float(row["forecast_price_kes"]),
                "forecast_lower": float(row["forecast_lower"]),
                "forecast_upper": float(row["forecast_upper"]),
                "test_mape": float(row["test_mape"]),
                "confidence": row["confidence"],
                "strategy": row["strategy"],
                "available": available,
            }

    unreliable = _unreliable_row(commodity, market, pricetype)
    if unreliable is not None:
        return {
            "status": "unreliable",
            "message": (
                f"{commodity} in {market} ({unreliable['pricetype']}) was tested but its "
                f"error rate of {unreliable['mape']:.1f}% exceeds the 25% reliability "
                "threshold, so it is not served as a live forecast."
            ),
            "commodity": commodity,
            "market": market,
            "pricetype": unreliable["pricetype"],
            "test_mape": float(unreliable["mape"]),
            "requested": {"commodity": commodity, "market": market, "pricetype": pricetype},
            "alternatives": combos,
            "available": available,
        }

    if combos:
        return {
            "status": "no_combination",
            "message": (
                f"{commodity} in {market} is tracked as {', '.join(combos)}, not "
                f"{pricetype or 'that price type'}."
            ),
            "requested": {"commodity": commodity, "market": market, "pricetype": pricetype},
            "alternatives": combos,
            "available": available,
        }

    return {
        "status": "no_combination",
        "message": f"{commodity} is not tracked in {market} at all.",
        "requested": {"commodity": commodity, "market": market, "pricetype": pricetype},
        "alternatives": {
            "markets_for_commodity": [m for m, _ in _combos_for_commodity(commodity)],
            "commodities_for_market": [c for c, _ in _combos_for_market(market)],
        },
        "available": available,
    }


# ---------------------------------------------------------------------------
# Anomaly / volatility lookups
# ---------------------------------------------------------------------------


def get_anomaly(entities: dict) -> dict:
    commodity = entities.get("commodity")
    market = entities.get("market")
    pricetype = entities.get("pricetype")

    if not market or market not in COVERED_MARKETS:
        return {"status": "no_market", "available": _available()}
    if not commodity or commodity not in COVERED_COMMODITIES:
        return {"status": "no_commodity", "available": _available()}

    sub = ANOMALIES[(ANOMALIES["commodity"] == commodity) & (ANOMALIES["market"] == market)]
    if pricetype:
        narrowed = sub[sub["pricetype"] == pricetype]
        if not narrowed.empty:
            sub = narrowed

    if sub.empty:
        return {
            "status": "no_combination",
            "message": f"{commodity} in {market} is not in the anomaly-monitored set.",
            "available": _available(),
        }

    row = sub.iloc[0]
    return {
        "status": "found",
        "commodity": commodity,
        "market": market,
        "pricetype": row["pricetype"],
        "latest_price": float(row["latest_price"]),
        "normal_low": float(row["normal_low"]),
        "normal_high": float(row["normal_high"]),
        "anomaly_status": row["status"],
        "deviation_pct": float(row["deviation_pct"]),
    }


def get_volatility(entities: dict) -> dict:
    commodity = entities.get("commodity")
    market = entities.get("market")
    pricetype = entities.get("pricetype")

    if not market or market not in COVERED_MARKETS:
        return {"status": "no_market", "available": _available()}
    if not commodity or commodity not in COVERED_COMMODITIES:
        return {"status": "no_commodity", "available": _available()}

    sub = VOLATILITY[(VOLATILITY["commodity"] == commodity) & (VOLATILITY["market"] == market)]
    if pricetype:
        narrowed = sub[sub["pricetype"] == pricetype]
        if not narrowed.empty:
            sub = narrowed

    if sub.empty:
        return {
            "status": "no_combination",
            "message": f"{commodity} in {market} has no volatility data.",
            "available": _available(),
        }

    row = sub.iloc[0]
    return {
        "status": "found",
        "commodity": commodity,
        "market": market,
        "pricetype": row["pricetype"],
        "current_volatility_pct": float(row["current_volatility_pct"]),
        "median_volatility_pct": float(row["median_volatility_pct"]),
        "trend": row["trend"],
    }


# ---------------------------------------------------------------------------
# Comparison lookup
# ---------------------------------------------------------------------------


def get_cheapest_market(commodity: str | None, region: str | None = None, horizon_months: int = 3) -> dict:
    if not commodity or commodity not in COVERED_COMMODITIES:
        return {"status": "no_commodity", "available": _available()}

    sub = FORECASTS[
        (FORECASTS["commodity"] == commodity) & (FORECASTS["horizon_months"] == horizon_months)
    ]
    if region:
        sub = sub[sub["region"] == region]

    if sub.empty:
        return {
            "status": "no_combination",
            "message": f"No markets found for {commodity}" + (f" in {region}" if region else ""),
            "available": _available(),
        }

    ranked = sub.sort_values("forecast_price_kes")[
        ["market", "region", "pricetype", "forecast_price_kes", "confidence", "test_mape"]
    ].to_dict("records")

    cheapest = ranked[0]
    return {
        "status": "found",
        "commodity": commodity,
        "cheapest_market": cheapest["market"],
        "cheapest_price_kes": float(cheapest["forecast_price_kes"]),
        "cheapest_pricetype": cheapest["pricetype"],
        "confidence": cheapest["confidence"],
        "test_mape": float(cheapest["test_mape"]),
        "ranked_markets": ranked,
    }
