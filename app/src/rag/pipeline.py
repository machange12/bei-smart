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


def _answer_forecast(question: str, entities: dict, language: str) -> dict:
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
    result = lookup.get_cheapest_market(entities.get("commodity"))
    status = result["status"]

    if status != "found":
        text = generate.render_status_message(result, language)
        return {
            "answer": text,
            "status": status,
            "sources": ["production_forecasts.csv"],
            "confidence": None,
        }

    deterministic = generate.build_cheapest_answer(result, language)
    context = (
        f"Commodity: {result['commodity']}\n"
        f"Cheapest market: {result['cheapest_market']} at {result['cheapest_price_kes']:.2f} KES/kg\n"
        f"Confidence statement to use verbatim: "
        f"{generate.confidence_words(result['test_mape'], language)}\n"
        f"All markets ranked by price: "
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


def answer(question: str) -> dict:
    """Answer a single question end to end.

    Returns a dict with keys: answer, intent, language, status, sources,
    confidence.
    """
    language = router.detect_language(question)
    intent = router.classify_intent(question)
    entities = lookup.extract_entities(question)

    if intent == "explain":
        result = _answer_explain(question, language)
    elif intent == "seasonality":
        result = _answer_seasonality(question, entities, language)
    elif intent == "history":
        result = _answer_history(question, entities, language)
    elif intent == "compare":
        result = _answer_compare(question, entities, language)
    elif intent == "alert":
        result = _answer_alert(question, entities, language)
    else:
        result = _answer_forecast(question, entities, language)

    return {
        "answer": result["answer"],
        "intent": intent,
        "language": language,
        "status": result["status"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "entities": entities,
    }
