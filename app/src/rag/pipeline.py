"""AgriPulse RAG pipeline: routed hybrid, not naive RAG.

Numeric intents (forecast, compare, alert) are answered entirely from
lookup.py's deterministic dataframe filters. Only the `explain` intent
touches the vector store. The LLM, when available, is used purely to
phrase the deterministic result naturally in the detected language --
it never supplies a number of its own.
"""

from __future__ import annotations

from . import generate, lookup, router, vectorstore


def _confidence_label(status: str, test_mape: float | None) -> str | None:
    if status != "found" or test_mape is None:
        return None
    if test_mape < 10:
        return "high"
    if test_mape <= 20:
        return "medium"
    return "low"


def _missing_followup(
    entities: dict, language: str, need_commodity: bool = True, need_market: bool = True
) -> dict | None:
    """A question with no market/commodity named at all gets a direct
    follow-up question instead of a dump of every covered option -- that
    dump is still the right response once a *specific but uncovered* value
    has been named (handled separately by no_market/no_commodity)."""
    missing_commodity = need_commodity and not entities.get("commodity")
    missing_market = need_market and not entities.get("market")
    if not missing_commodity and not missing_market:
        return None
    if missing_commodity and missing_market:
        return {"answer": generate.ask_followup("both", language), "status": "missing_both"}
    if missing_commodity:
        return {"answer": generate.ask_followup("commodity", language), "status": "missing_commodity"}
    return {"answer": generate.ask_followup("market", language), "status": "missing_market"}


def _answer_forecast(question: str, entities: dict, language: str) -> dict:
    missing = _missing_followup(entities, language)
    if missing:
        return {**missing, "sources": [], "confidence": None}

    result = lookup.get_forecast(entities)
    status = result["status"]

    if status != "found":
        text = generate.render_status_message(result, language)
        return {
            "answer": text,
            "status": status,
            "sources": ["production_forecasts.csv", "model_metrics.csv"],
            "confidence": None,
        }

    anomaly = lookup.get_anomaly(entities)
    deterministic = generate.build_forecast_answer(result, language, anomaly=anomaly)
    context = generate.forecast_context(result, language)
    note = generate.anomaly_note(anomaly, language)
    if note:
        context = f"{context}\nAnomaly note: {note}"

    text = generate.generate_from_context(context, question, language, deterministic)
    return {
        "answer": text,
        "status": status,
        "sources": ["production_forecasts.csv", "model_routing.json"],
        "confidence": result["confidence"],
    }


def _answer_compare(question: str, entities: dict, language: str) -> dict:
    missing = _missing_followup(entities, language, need_market=False)
    if missing:
        return {**missing, "sources": [], "confidence": None}

    direction = router.detect_price_direction(question)
    result = lookup.get_price_extreme(entities.get("commodity"), direction=direction)
    status = result["status"]

    if status != "found":
        text = generate.render_status_message(result, language)
        return {
            "answer": text,
            "status": status,
            "sources": ["production_forecasts.csv"],
            "confidence": None,
        }

    deterministic = generate.build_extreme_answer(result, language)
    label = "Cheapest" if direction != "expensive" else "Most expensive"
    context = (
        f"Commodity: {result['commodity']}\n"
        f"{label} market: {result['extreme_market']} at {result['extreme_price_kes']:.2f} KES/kg\n"
        f"Confidence statement to use verbatim: "
        f"{generate.confidence_words(result['test_mape'], language)}\n"
        f"All markets ranked by price ({direction} first): "
        + ", ".join(
            f"{row['market']} {row['forecast_price_kes']:.2f} KES/kg"
            for row in result["ranked_markets"]
        )
    )
    named_market = entities.get("market")
    if named_market and named_market in lookup.COVERED_MARKETS:
        for row in result["ranked_markets"]:
            if row["market"] == named_market:
                context += (
                    f"\nSpecifically named market: {named_market} at "
                    f"{row['forecast_price_kes']:.2f} KES/kg"
                )
                break
    text = generate.generate_from_context(context, question, language, deterministic)
    return {
        "answer": text,
        "status": status,
        "sources": ["production_forecasts.csv"],
        "confidence": _confidence_label("found", result["test_mape"]),
    }


def _answer_alert(question: str, entities: dict, language: str) -> dict:
    missing = _missing_followup(entities, language)
    if missing:
        return {**missing, "sources": [], "confidence": None}

    result = lookup.get_anomaly(entities)
    status = result["status"]

    if status != "found":
        if status in ("no_market", "no_commodity"):
            text = generate.render_status_message(
                {"status": status, "requested": entities, "available": result["available"]},
                language,
            )
            return {
                "answer": text,
                "status": status,
                "sources": ["price_anomalies.csv"],
                "confidence": None,
            }

        # Commodity + market are individually valid but this exact
        # combination is absent from the anomaly table -- before giving up,
        # check whether volatility data (a separate table) has it instead.
        # This is what a "how volatile is X" question actually needs.
        volatility_only = lookup.get_volatility(entities)
        if volatility_only.get("status") == "found":
            deterministic = generate.build_volatility_answer(volatility_only, language)
            context = (
                f"Commodity: {volatility_only['commodity']}\nMarket: {volatility_only['market']}\n"
                f"Pricetype: {volatility_only['pricetype']}\n"
                f"Current volatility: {volatility_only['current_volatility_pct']:.1f}%\n"
                f"Trend: {volatility_only['trend']}\n"
            )
            text = generate.generate_from_context(context, question, language, deterministic)
            return {
                "answer": text,
                "status": "found",
                "sources": ["market_volatility.csv"],
                "confidence": None,
            }

        text = (
            "Hakuna taarifa za ufuatiliaji wa bei kwa mchanganyiko huu."
            if language == "sw"
            else result.get("message", "No anomaly data is monitored for this combination.")
        )
        return {
            "answer": text,
            "status": status,
            "sources": ["price_anomalies.csv", "market_volatility.csv"],
            "confidence": None,
        }

    volatility = lookup.get_volatility(entities)
    deterministic = generate.build_anomaly_answer(result, language)
    context = (
        f"Commodity: {result['commodity']}\nMarket: {result['market']}\n"
        f"Pricetype: {result['pricetype']}\nLatest price: {result['latest_price']:.2f} KES/kg\n"
        f"Anomaly status: {result['anomaly_status']}\n"
    )
    if volatility.get("status") == "found":
        context += f"Volatility trend: {volatility['trend']}\n"

    text = generate.generate_from_context(context, question, language, deterministic)
    return {
        "answer": text,
        "status": status,
        "sources": ["price_anomalies.csv", "market_volatility.csv"],
        "confidence": None,
    }


def _answer_history(question: str, entities: dict, language: str) -> dict:
    missing = _missing_followup(entities, language)
    if missing:
        return {**missing, "sources": [], "confidence": None}

    result = lookup.get_historical_price(entities)
    status = result["status"]

    if status != "found":
        text = generate.render_status_message(result, language)
        return {
            "answer": text,
            "status": status,
            "sources": ["bei_smart_forecasting_final.csv"],
            "confidence": None,
        }

    deterministic = generate.build_history_answer(result, language)
    context = generate.historical_context(result, language)
    text = generate.generate_from_context(context, question, language, deterministic)
    return {
        "answer": text,
        "status": status,
        "sources": ["bei_smart_forecasting_final.csv"],
        "confidence": None,
    }


def _answer_seasonality(question: str, entities: dict, language: str) -> dict:
    missing = _missing_followup(entities, language)
    if missing:
        return {**missing, "sources": [], "confidence": None}

    result = lookup.get_seasonality(entities)
    status = result["status"]

    if status != "found":
        text = generate.render_status_message(result, language)
        return {
            "answer": text,
            "status": status,
            "sources": ["bei_smart_forecasting_final.csv"],
            "confidence": None,
        }

    deterministic = generate.build_seasonality_answer(result, language)
    context = (
        f"Commodity: {result['commodity']}\nMarket: {result['market']}\n"
        f"Pricetype: {result['pricetype']}\n"
        f"Best month to sell (highest average price): month {result['best_month']}, "
        f"avg {result['best_month_avg_price']:.2f} KES/kg\n"
        f"Worst month to sell (lowest average price): month {result['worst_month']}, "
        f"avg {result['worst_month_avg_price']:.2f} KES/kg\n"
        f"Based on {result['n_years']} years of history. This is a seasonal pattern from "
        f"actual past prices, not a forecast -- do not attach a confidence statement."
    )
    text = generate.generate_from_context(context, question, language, deterministic)
    return {
        "answer": text,
        "status": status,
        "sources": ["bei_smart_forecasting_final.csv"],
        "confidence": None,
    }


def _answer_explain(question: str, language: str) -> dict:
    chunks = vectorstore.query(question, n_results=3)
    text = generate.generate_explain_answer(question, chunks, language)
    status = "found" if chunks else "no_context"
    sources = [c["source"] for c in chunks]
    return {"answer": text, "status": status, "sources": sources, "confidence": None}


_FOLLOWUP_STATUSES = {"missing_commodity", "missing_market", "missing_both"}


def _dispatch(intent: str, question: str, entities: dict, language: str) -> dict:
    if intent == "explain":
        return _answer_explain(question, language)
    if intent == "seasonality":
        return _answer_seasonality(question, entities, language)
    if intent == "history":
        return _answer_history(question, entities, language)
    if intent == "compare":
        return _answer_compare(question, entities, language)
    if intent == "alert":
        return _answer_alert(question, entities, language)
    return _answer_forecast(question, entities, language)


def _finalize(intent: str, language: str, entities: dict, result: dict) -> dict:
    response = {
        "answer": result["answer"],
        "intent": intent,
        "language": language,
        "status": result["status"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "entities": entities,
    }
    if result["status"] in _FOLLOWUP_STATUSES:
        # Carried by the caller (e.g. the Chat page) so the *next* message
        # can be treated as answering this question rather than a fresh one.
        # Language is carried forward too: a follow-up reply is often just
        # one or two words ("Nairobi", "maize"), exactly the short text
        # langdetect is least reliable on, so re-detecting from it alone
        # risks flipping languages mid-conversation for no real reason.
        response["pending"] = {"intent": intent, "entities": entities, "language": language}
    return response


def answer(question: str) -> dict:
    """Answer a single question end to end.

    Returns a dict with keys: answer, intent, language, status, sources,
    confidence, entities, and -- only when the response is a follow-up
    question -- pending, which a caller should hand back to continue_answer
    along with the user's next message.
    """
    language = router.detect_language(question)
    intent = router.classify_intent(question)
    entities = lookup.extract_entities(question)
    result = _dispatch(intent, question, entities, language)
    return _finalize(intent, language, entities, result)


def continue_answer(pending: dict, follow_up_text: str) -> dict:
    """Resume a pending intent after a follow-up question, merging whatever
    the follow-up message names into the entities that were already known.
    Only fills gaps -- an entity already resolved from the original question
    is never overwritten by the follow-up."""
    intent = pending["intent"]
    entities = dict(pending["entities"])
    # Inherit the original question's language rather than re-detecting from
    # the follow-up alone -- see the comment on `pending` in _finalize.
    language = pending.get("language") or router.detect_language(follow_up_text)

    new_entities = lookup.extract_entities(follow_up_text)
    for key in ("commodity", "market", "pricetype"):
        if not entities.get(key) and new_entities.get(key):
            entities[key] = new_entities[key]

    result = _dispatch(intent, follow_up_text, entities, language)
    return _finalize(intent, language, entities, result)
