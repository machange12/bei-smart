"""Prompt assembly and response generation for AgriPulse RAG.

Two response paths:
  1. Deterministic templates for every "nothing to forecast" status
     (no_market, no_commodity, no_combination, unreliable) and as an
     offline fallback when the LLM is unavailable. These never guess.
  2. LLM phrasing for a `found` forecast/comparison/alert or an `explain`
     answer, constrained by a system prompt to the supplied context only.

Either way, no number is ever surfaced that did not come from lookup.py or
the knowledge base.
"""

from __future__ import annotations

from . import llm
from .lookup import MONTH_NAMES_EN, MONTH_NAMES_SW

SYSTEM_PROMPT = """You are the AgriPulse assistant, answering questions about Kenyan food prices.

Rules:
- Answer only from the context provided below. Never state a number that is not present in that context.
- Reply entirely in the requested language. If the language is Swahili, answer entirely in Swahili with no English words mixed in.
- The response must contain the fact itself and -- only if the context supplies one -- the confidence statement (repeat it verbatim, do not invent your own wording) or an anomaly note about unusual current prices. If the context is an actual historical observation rather than a forecast, do not add a confidence statement at all -- observed prices are facts, not predictions. Add nothing else unless the question explicitly asks for more.
- Do not name the underlying model (Prophet, SARIMA, Chronos, ensemble) or quote volatility percentages unless the question specifically asks about the model or volatility.
- Keep the answer short: two to three sentences.
"""


def confidence_words(test_mape: float, language: str) -> str:
    """Confidence phrasing driven by measured test MAPE, not a guess."""
    if test_mape < 10:
        return "Utabiri huu ni wa kuaminika" if language == "sw" else "This forecast is reliable"
    if test_mape <= 20:
        return (
            "Utabiri huu una uhakika wa wastani"
            if language == "sw"
            else "This forecast is moderately confident"
        )
    return (
        "Bei katika soko hili ni vigumu kutabiri, hii inaweza kukosea kwa robo"
        if language == "sw"
        else "Prices here are difficult to predict, this could be off by a quarter"
    )


# ---------------------------------------------------------------------------
# Deterministic templates for the non-"found" lookup statuses
# ---------------------------------------------------------------------------


def _join(items) -> str:
    return ", ".join(items)


def _template_no_market(result: dict, language: str) -> str:
    market = (result.get("requested") or {}).get("market")
    covered = _join(result["available"]["markets"])
    if language == "sw":
        return (
            f"Soko '{market or 'hilo'}' halijaandikishwa katika mfumo huu. "
            f"Masoko yanayofuatiliwa ni: {covered}."
        )
    return (
        f"'{market or 'That market'}' is not covered by this system. "
        f"Markets currently tracked are: {covered}."
    )


def _template_no_commodity(result: dict, language: str) -> str:
    commodity = (result.get("requested") or {}).get("commodity")
    covered = _join(result["available"]["commodities"])
    if language == "sw":
        return (
            f"Bidhaa '{commodity or 'hiyo'}' haifuatiliwi na mfumo huu. "
            f"Bidhaa zinazofuatiliwa ni: {covered}."
        )
    return (
        f"'{commodity or 'That commodity'}' is not modelled by this system. "
        f"Commodities currently tracked are: {covered}."
    )


def _template_no_combination(result: dict, language: str) -> str:
    req = result.get("requested") or {}
    commodity, market, pricetype = req.get("commodity"), req.get("market"), req.get("pricetype")
    alternatives = result.get("alternatives")

    if isinstance(alternatives, list) and alternatives:
        alt_str = _join(alternatives)
        if language == "sw":
            return (
                f"{commodity} huko {market} inafuatiliwa kama {alt_str}, si "
                f"{pricetype or 'aina hiyo ya bei'}. Jaribu kutumia {alt_str}."
            )
        return (
            f"{commodity} in {market} is tracked as {alt_str}, not "
            f"{pricetype or 'that price type'}. Try asking for {alt_str} instead."
        )

    if language == "sw":
        return f"{commodity} haifuatiliwi katika soko la {market}."
    return f"{commodity} is not tracked in {market} at all."


def _template_unreliable(result: dict, language: str) -> str:
    commodity, market = result.get("commodity"), result.get("market")
    mape = result.get("test_mape")
    if language == "sw":
        return (
            f"{commodity} huko {market} ilijaribiwa lakini kosa lake la {mape:.1f}% "
            "linazidi kiwango cha kukubalika, hivyo haitolewi kama utabiri wa moja kwa moja."
        )
    return (
        f"{commodity} in {market} was tested but its error rate of {mape:.1f}% exceeds "
        "the reliability threshold, so it is not served as a live forecast."
    )


def _template_no_data_for_period(result: dict, language: str) -> str:
    req = result.get("requested") or {}
    commodity, market = req.get("commodity"), req.get("market")
    period = result.get("period") or {}
    year, month = period.get("year"), period.get("month")
    when = f"{MONTH_NAMES_EN.get(month, '').title()} {year}".strip() if month else str(year)
    if language == "sw":
        when_sw = f"{MONTH_NAMES_SW.get(month, '').title()} {year}".strip() if month else str(year)
        return f"Hakuna rekodi za {commodity} huko {market} kwa {when_sw}."
    return f"No records of {commodity} in {market} for {when}."


def _template_insufficient_history(result: dict, language: str) -> str:
    req = result.get("requested") or {}
    commodity, market = req.get("commodity"), req.get("market")
    if language == "sw":
        return (
            f"{commodity} huko {market} haina historia ya kutosha kubaini "
            "msimu wa bei kwa uhakika."
        )
    return result.get(
        "message", f"{commodity} in {market} does not have enough history for a seasonal pattern."
    )


_TEMPLATES = {
    "no_market": _template_no_market,
    "no_commodity": _template_no_commodity,
    "no_combination": _template_no_combination,
    "unreliable": _template_unreliable,
    "no_data_for_period": _template_no_data_for_period,
    "insufficient_history": _template_insufficient_history,
}


def render_status_message(result: dict, language: str) -> str:
    status = result["status"]
    template = _TEMPLATES.get(status)
    if template is None:
        return result.get("message", "")
    return template(result, language)


# ---------------------------------------------------------------------------
# Context builders for the "found" case
# ---------------------------------------------------------------------------


def forecast_context(result: dict, language: str) -> str:
    conf_words = confidence_words(result["test_mape"], language)
    lines = [
        f"Commodity: {result['commodity']}",
        f"Market: {result['market']}",
        f"Pricetype: {result['pricetype']}",
        f"Horizon: {result['horizon_months']} months",
        f"Forecast price: {result['forecast_price_kes']:.2f} KES/kg "
        f"(range {result['forecast_lower']:.2f}-{result['forecast_upper']:.2f})",
        f"Confidence statement to use verbatim: {conf_words}",
    ]
    return "\n".join(lines)


def anomaly_note(anomaly: dict | None, language: str) -> str | None:
    if not anomaly or anomaly.get("status") != "found":
        return None
    if anomaly.get("anomaly_status") == "NORMAL":
        return None
    if language == "sw":
        return "Bei za sasa hazina mwelekeo wa kawaida (zimeonekana kuwa juu au chini ya kawaida)."
    return "Current prices here are unusual compared with this market's normal range."


def build_forecast_answer(result: dict, language: str, anomaly: dict | None = None) -> str:
    """Deterministic fallback for a `found` forecast -- used when the LLM
    is unavailable, and as the ground truth the LLM's phrasing is checked
    against conceptually."""
    conf = confidence_words(result["test_mape"], language)
    price = result["forecast_price_kes"]
    unit = "KES/kg"

    if language == "sw":
        base = (
            f"Bei ya {result['commodity']} {result['market']} ({result['pricetype']}) "
            f"kwa miezi {result['horizon_months']} inakadiriwa kuwa {price:.2f} {unit}. {conf}."
        )
    else:
        base = (
            f"The {result['horizon_months']}-month forecast for {result['commodity']} in "
            f"{result['market']} ({result['pricetype']}) is {price:.2f} {unit}. {conf}."
        )

    note = anomaly_note(anomaly, language)
    if note:
        base = f"{base} {note}"
    return base


def build_cheapest_answer(result: dict, language: str) -> str:
    market, price = result["cheapest_market"], result["cheapest_price_kes"]
    conf = confidence_words(result["test_mape"], language)
    if language == "sw":
        return (
            f"Bei nafuu zaidi ya {result['commodity']} ni {market} kwa {price:.2f} KES/kg. {conf}."
        )
    return f"The cheapest market for {result['commodity']} is {market} at {price:.2f} KES/kg. {conf}."


def build_volatility_answer(result: dict, language: str) -> str:
    trend = result["trend"]
    pct = result["current_volatility_pct"]
    if language == "sw":
        return (
            f"Mabadiliko ya bei ya {result['commodity']} {result['market']} "
            f"({result['pricetype']}) kwa sasa ni {pct:.1f}% na mwelekeo wake ni {trend}."
        )
    return (
        f"Price volatility for {result['commodity']} in {result['market']} "
        f"({result['pricetype']}) is currently {pct:.1f}%, trending {trend}."
    )


def build_anomaly_answer(result: dict, language: str) -> str:
    status = result["anomaly_status"]
    price = result["latest_price"]
    if language == "sw":
        if status == "NORMAL":
            return (
                f"Bei ya {result['commodity']} {result['market']} ({result['pricetype']}) ni "
                f"{price:.2f} KES/kg, ambayo ni ya kawaida."
            )
        return (
            f"Bei ya {result['commodity']} {result['market']} ({result['pricetype']}) ni "
            f"{price:.2f} KES/kg. Bei hii si ya kawaida (SPIKE)."
        )
    if status == "NORMAL":
        return (
            f"The price of {result['commodity']} in {result['market']} ({result['pricetype']}) "
            f"is {price:.2f} KES/kg, which is within its normal range."
        )
    return (
        f"The price of {result['commodity']} in {result['market']} ({result['pricetype']}) is "
        f"{price:.2f} KES/kg. This is currently unusual (flagged as a SPIKE)."
    )


# ---------------------------------------------------------------------------
# LLM-backed generation with deterministic fallback
# ---------------------------------------------------------------------------


def generate_from_context(context: str, question: str, language: str, deterministic_fallback: str) -> str:
    lang_instruction = (
        "Answer entirely in Swahili." if language == "sw" else "Answer entirely in English."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n{lang_instruction}"

    try:
        return llm.generate(prompt, SYSTEM_PROMPT)
    except Exception:
        return deterministic_fallback


# ---------------------------------------------------------------------------
# Historical price + seasonality answers
# ---------------------------------------------------------------------------


def _month_name(month: int, language: str) -> str:
    names = MONTH_NAMES_SW if language == "sw" else MONTH_NAMES_EN
    return names.get(month, "").title()


def historical_context(result: dict, language: str) -> str:
    lines = [
        f"Commodity: {result['commodity']}",
        f"Market: {result['market']}",
        f"Pricetype: {result['pricetype']}",
        f"Date of this actual observation: {result['date']}",
        f"Observed price: {result['price_per_kg']:.2f} KES/kg",
        "This is an actual recorded price, not a forecast -- do not attach a confidence statement to it.",
    ]
    return "\n".join(lines)


def build_history_answer(result: dict, language: str) -> str:
    price = result["price_per_kg"]
    date_str = result["date"].strftime("%B %Y")
    if language == "sw":
        if result["is_exact_period"]:
            return (
                f"Bei ya {result['commodity']} {result['market']} ({result['pricetype']}) "
                f"mnamo {date_str} ilikuwa KES {price:.2f}/kg."
            )
        return (
            f"Bei ya mwisho iliyorekodiwa ya {result['commodity']} {result['market']} "
            f"({result['pricetype']}) ilikuwa KES {price:.2f}/kg, mnamo {date_str}."
        )
    if result["is_exact_period"]:
        return (
            f"The price of {result['commodity']} in {result['market']} ({result['pricetype']}) "
            f"in {date_str} was KES {price:.2f}/kg."
        )
    return (
        f"The most recently recorded price of {result['commodity']} in {result['market']} "
        f"({result['pricetype']}) was KES {price:.2f}/kg, as of {date_str}."
    )


def build_seasonality_answer(result: dict, language: str) -> str:
    best_name = _month_name(result["best_month"], language)
    worst_name = _month_name(result["worst_month"], language)
    if language == "sw":
        return (
            f"Kwa {result['commodity']} {result['market']} ({result['pricetype']}), bei huwa "
            f"juu zaidi mnamo {best_name} (wastani KES {result['best_month_avg_price']:.2f}/kg) "
            f"na chini zaidi mnamo {worst_name} (wastani KES {result['worst_month_avg_price']:.2f}/kg), "
            f"kulingana na miaka {result['n_years']} ya data. {best_name} ndio wakati mzuri zaidi wa kuuza."
        )
    return (
        f"For {result['commodity']} in {result['market']} ({result['pricetype']}), prices are "
        f"typically highest in {best_name} (averaging KES {result['best_month_avg_price']:.2f}/kg) "
        f"and lowest in {worst_name} (averaging KES {result['worst_month_avg_price']:.2f}/kg), "
        f"based on {result['n_years']} years of data. {best_name} is the best time to sell."
    )


def generate_explain_answer(question: str, chunks: list[dict], language: str) -> str:
    if not chunks:
        if language == "sw":
            return "Sina taarifa za kutosha kujibu swali hilo kutoka kwenye hifadhi ya maarifa."
        return "I don't have enough information in the knowledge base to answer that."

    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    deterministic_fallback = chunks[0]["text"]

    lang_instruction = (
        "Answer entirely in Swahili." if language == "sw" else "Answer entirely in English."
    )
    prompt = (
        f"Context from the AgriPulse knowledge base:\n{context}\n\n"
        f"Question: {question}\n\n{lang_instruction} "
        "Answer only using the context above."
    )
    try:
        return llm.generate(prompt, SYSTEM_PROMPT)
    except Exception:
        return deterministic_fallback
