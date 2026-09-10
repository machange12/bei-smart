"""AgriPulse landing page: headline metrics."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import render_footer, render_header  # noqa: E402

st.set_page_config(page_title="AgriPulse", page_icon="🌾", layout="wide")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "production_forecasts.csv", parse_dates=["forecast_date"])


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "price_anomalies.csv", parse_dates=["latest_date"])


@st.cache_data
def load_validation_archive() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation_archive.csv", parse_dates=["date"])


forecasts = load_forecasts()
anomalies = load_anomalies()
validation_archive = load_validation_archive()

render_header()

st.markdown(
    "Use the sidebar to explore forecasts, anomaly alerts, market locations, "
    "the price-movement classifier, and the validation archive."
)

col1, col2, col3, col4 = st.columns(4)

live_series = forecasts
if "months_stale" in forecasts.columns:
    live_series = forecasts[forecasts["months_stale"] <= 6]
live_series_count = len(live_series[["commodity", "market", "pricetype"]].drop_duplicates())

# Median error: prefer the discontinued-series validation archive when it has
# data (accuracy checked against real outcomes); otherwise fall back to the
# deployed series' own measured test MAPE.
if not validation_archive.empty:
    median_forecast_error = validation_archive["error_pct"].median()
    error_source = "measured on discontinued series retained for method validation"
else:
    median_forecast_error = forecasts.drop_duplicates(["commodity", "market", "pricetype"])["test_mape"].median()
    error_source = "measured test error across all currently deployed series"

markets_covered = live_series["market"].nunique()
active_alerts = int((anomalies["status"] != "NORMAL").sum())

col1.metric("Live series", f"{live_series_count:,}")
col2.metric("Median forecast error", f"{median_forecast_error:.1f}%")
col3.metric("Markets covered", f"{markets_covered:,}")
col4.metric("Active price alerts", f"{active_alerts:,}")

st.caption(f"Median forecast error is {error_source} — see the Validation page for details.")

st.divider()
st.markdown(
    """
    ### Pages
    - **Forecast** — historical prices plus forward-looking forecasts with confidence bands
    - **Alerts** — price anomaly table and market volatility
    - **Map** — market locations colored by anomaly status
    - **Classify** — predict whether a price point is cheap, average, or expensive
    - **Validation** — forecast accuracy evidence from discontinued series
    """
)

render_footer()
