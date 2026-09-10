"""Forecast page: historical prices + forecast with confidence band."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import confidence_badge, model_provenance_caption, render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Forecast | AgriPulse", page_icon="📈", layout="wide")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "production_forecasts.csv", parse_dates=["forecast_date"])


@st.cache_data
def load_historical() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "bei_smart_forecasting_final.csv", parse_dates=["date"])


@st.cache_data
def load_routing() -> dict:
    return json.loads((DATA_DIR / "model_routing.json").read_text())


forecasts_all = load_forecasts()
historical = load_historical()
routing = load_routing()

render_header("Price Forecast", "📈")

# The data layer can carry series up to 36 months stale (with confidence
# capped accordingly -- see model_routing.json), but this page only offers
# genuinely current series so a user never has to second-guess freshness.
if "months_stale" in forecasts_all.columns:
    forecasts = forecasts_all[forecasts_all["months_stale"] <= 6]
else:
    forecasts = forecasts_all

commodities = sorted(forecasts["commodity"].unique())
commodity = st.selectbox("Commodity", commodities)

markets = sorted(forecasts.loc[forecasts["commodity"] == commodity, "market"].unique())
market = st.selectbox("Market", markets)

pricetypes = sorted(
    forecasts.loc[
        (forecasts["commodity"] == commodity) & (forecasts["market"] == market), "pricetype"
    ].unique()
)
pricetype = st.selectbox("Price type", pricetypes)

horizon = st.radio("Horizon (months)", [3, 6, 12], horizontal=True)

subset = forecasts[
    (forecasts["commodity"] == commodity)
    & (forecasts["market"] == market)
    & (forecasts["pricetype"] == pricetype)
    & (forecasts["horizon_months"] == horizon)
].sort_values("forecast_date")

if subset.empty:
    st.warning("No forecast available for this combination.")
    st.stop()

# Confidence: prefer routing.json, fall back to the CSV's own columns —
# mirrors the API's /forecast reconciliation logic exactly.
key = f"{commodity}|{market}|{pricetype}"
route = routing.get(key)
if route is not None:
    confidence = route["confidence"]
    test_mape = route["test_mape"]
    confidence_source = "routing"
else:
    confidence = subset["confidence"].iloc[0]
    test_mape = subset["test_mape"].iloc[0]
    confidence_source = "csv_fallback"

strategy = subset["strategy"].iloc[0]

confidence_badge(confidence)
model_provenance_caption(strategy, test_mape)
st.caption(f"Confidence source: `{confidence_source}`")

hist_subset = historical[
    (historical["commodity"] == commodity)
    & (historical["market"] == market)
    & (historical["pricetype"] == pricetype)
].sort_values("date")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=hist_subset["date"],
        y=hist_subset["price_per_kg"],
        name="Historical price",
        mode="lines",
        line=dict(color="#1f77b4"),
    )
)

fig.add_trace(
    go.Scatter(
        x=subset["forecast_date"],
        y=subset["forecast_upper"],
        name="Upper bound",
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=subset["forecast_date"],
        y=subset["forecast_lower"],
        name="Confidence band",
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255,127,14,0.2)",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=subset["forecast_date"],
        y=subset["forecast_price_kes"],
        name="Forecast",
        mode="lines+markers",
        line=dict(color="#ff7f0e"),
    )
)

fig.update_layout(
    title=f"{commodity} — {market} ({pricetype})",
    xaxis_title="Date",
    yaxis_title="Price (KES)",
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

render_footer()
