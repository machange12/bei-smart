"""Forecast page: historical prices + forecast with confidence band.

Selection cascades Region -> Market -> Commodity -> Price type -> Horizon,
each filtered to what the previous choice actually has, so no combination
in the sidebar can ever resolve to an empty result.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import (  # noqa: E402
    confidence_badge,
    forecast_chart,
    inject_css,
    model_provenance_caption,
    sidebar_brand,
    sidebar_footer,
)

st.set_page_config(page_title="Forecast | AgriPulse", page_icon="📈", layout="wide")
inject_css()
sidebar_brand()


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

# The data layer can carry series up to 36 months stale (with confidence
# capped accordingly -- see model_routing.json), but this page only offers
# genuinely current series so a user never has to second-guess freshness.
if "months_stale" in forecasts_all.columns:
    forecasts = forecasts_all[forecasts_all["months_stale"] <= 6]
else:
    forecasts = forecasts_all

st.subheader("Price Forecast")

with st.sidebar:
    st.markdown("#### Filters")

    regions = sorted(forecasts["region"].dropna().unique()) if "region" in forecasts.columns else []
    region = st.selectbox("Region", regions) if regions else None

    scoped = forecasts[forecasts["region"] == region] if region else forecasts

    markets = sorted(scoped["market"].unique())
    market = st.selectbox("Market", markets)
    scoped = scoped[scoped["market"] == market]

    commodities = sorted(scoped["commodity"].unique())
    commodity = st.selectbox("Commodity", commodities)
    scoped = scoped[scoped["commodity"] == commodity]

    pricetypes = sorted(scoped["pricetype"].unique())
    pricetype = st.selectbox("Price type", pricetypes)
    scoped = scoped[scoped["pricetype"] == pricetype]

    horizons = sorted(scoped["horizon_months"].unique())
    horizon = st.radio("Horizon (months)", horizons, horizontal=True)

subset = scoped[scoped["horizon_months"] == horizon].sort_values("forecast_date")

if subset.empty:
    st.warning("No forecast available for this combination.")
    st.stop()

# Confidence: prefer routing.json, fall back to the CSV's own columns --
# mirrors the API's /forecast reconciliation logic exactly.
key = f"{commodity}|{market}|{pricetype}"
route = routing.get(key)
if route is not None:
    confidence = route["confidence"]
    test_mape = route["test_mape"]
else:
    confidence = subset["confidence"].iloc[0]
    test_mape = subset["test_mape"].iloc[0]

strategy = subset["strategy"].iloc[0]

hist_subset = historical[
    (historical["commodity"] == commodity)
    & (historical["market"] == market)
    & (historical["pricetype"] == pricetype)
].sort_values("date")

last_observed = hist_subset["date"].max()
last_observed_str = last_observed.strftime("%B %Y") if pd.notna(last_observed) else None

st.plotly_chart(forecast_chart(hist_subset, subset), width="stretch")

metric_cols = st.columns(3)
forecast_at_horizon = subset.iloc[-1]
current_price = hist_subset["price_per_kg"].iloc[-1] if not hist_subset.empty else None
delta = None
if current_price is not None:
    delta = f"{forecast_at_horizon['forecast_price_kes'] - current_price:+.2f} KES"

metric_cols[0].metric(
    f"Forecast at {horizon} months",
    f"KES {forecast_at_horizon['forecast_price_kes']:.2f}",
    delta=delta,
)
with metric_cols[1]:
    st.markdown("**Confidence**")
    confidence_badge(confidence)
metric_cols[2].metric("Test MAPE", f"{test_mape:.2f}%")

model_provenance_caption(strategy, test_mape, last_observed_str)

sidebar_footer()
