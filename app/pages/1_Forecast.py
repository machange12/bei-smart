"""Forecast page: historical prices + forecast with confidence band.

Selection cascades Region -> Market -> Commodity -> Price type -> Horizon,
each filtered to what the previous choice actually has, so no combination
in the sidebar can ever resolve to an empty result.
"""

import json
import re
import sys
from urllib.parse import urlencode
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.units import unit_label  # noqa: E402
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
    return pd.read_csv(DATA_DIR / "bei_smart_forecasting_chained.csv", parse_dates=["date"])


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


def seed_from_url(key: str, options: list, param: str) -> None:
    """Keep the widget's current value if still valid; otherwise take the URL
    value when it's a valid option at this cascade level, else the first option.
    A bad or stale shared link therefore degrades to defaults, never crashes."""
    if st.session_state.get(key) not in options:
        wanted = st.query_params.get(param)
        st.session_state[key] = next((o for o in options if str(o) == wanted), options[0])


with st.sidebar:
    st.markdown("#### Filters")

    regions = sorted(forecasts["region"].dropna().unique()) if "region" in forecasts.columns else []
    region = None
    if regions:
        seed_from_url("fc_region", regions, "region")
        region = st.selectbox("Region", regions, key="fc_region")

    scoped = forecasts[forecasts["region"] == region] if region else forecasts

    markets = sorted(scoped["market"].unique())
    seed_from_url("fc_market", markets, "market")
    market = st.selectbox("Market", markets, key="fc_market")
    scoped = scoped[scoped["market"] == market]

    commodities = sorted(scoped["commodity"].unique())
    seed_from_url("fc_commodity", commodities, "commodity")
    commodity = st.selectbox("Commodity", commodities, key="fc_commodity")
    scoped = scoped[scoped["commodity"] == commodity]

    pricetypes = sorted(scoped["pricetype"].unique())
    seed_from_url("fc_pricetype", pricetypes, "pricetype")
    pricetype = st.selectbox(
        "Price type", pricetypes, key="fc_pricetype",
        help="Wholesale = bulk price traders pay; Retail = what shoppers pay.",
    )
    scoped = scoped[scoped["pricetype"] == pricetype]

    horizons = [int(h) for h in sorted(scoped["horizon_months"].unique())]
    seed_from_url("fc_horizon", horizons, "horizon")
    horizon = st.radio("Horizon (months)", horizons, horizontal=True, key="fc_horizon")

# Mirror the selection into the URL so the address bar is a shareable link.
link_params = {"market": market, "commodity": commodity, "pricetype": pricetype, "horizon": horizon}
if region:
    link_params = {"region": region, **link_params}
st.query_params.from_dict({k: str(v) for k, v in link_params.items()})

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
    f"KES {forecast_at_horizon['forecast_price_kes']:.2f}/{unit_label(commodity)}",
    delta=delta,
)
with metric_cols[1]:
    st.markdown("**Confidence**")
    confidence_badge(confidence)
metric_cols[2].metric(
    "Typical error",
    f"±{test_mape:.1f}%",
    help="Test MAPE: the average gap between forecast and actual price on months the model never saw.",
)

model_provenance_caption(strategy, test_mape, last_observed_str)

export = pd.concat(
    [
        hist_subset[["date", "price_per_kg"]]
        .rename(columns={"price_per_kg": "actual_price_kes"})
        .assign(row_type="history"),
        subset[["forecast_date", "forecast_price_kes", "forecast_lower", "forecast_upper"]]
        .rename(columns={"forecast_date": "date"})
        .assign(row_type="forecast"),
    ]
).assign(commodity=commodity, market=market, pricetype=pricetype, horizon_months=horizon)
slug = re.sub(r"[^a-z0-9]+", "_", f"{commodity} {market} {pricetype} {horizon}m".lower()).strip("_")
dl_col, link_col = st.columns(2)
dl_col.download_button(
    "⬇️ Download this forecast (CSV)",
    export.to_csv(index=False),
    file_name=f"agripulse_forecast_{slug}.csv",
    mime="text/csv",
)
with link_col:
    query = "?" + urlencode(st.query_params.to_dict())
    base = (st.context.url or "").split("?")[0]
    st.caption("🔗 Copy link to this forecast")
    st.code(base + query, language=None, wrap_lines=True)

sidebar_footer()
