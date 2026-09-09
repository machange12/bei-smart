"""Alerts page: anomaly table and market volatility."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Alerts | AgriPulse", page_icon="🚨", layout="wide")


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "price_anomalies.csv", parse_dates=["latest_date"])


@st.cache_data
def load_volatility() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "market_volatility.csv")


@st.cache_data
def load_market_region_map() -> dict:
    historical = pd.read_csv(
        DATA_DIR / "bei_smart_forecasting_final.csv", usecols=["market", "admin1"]
    )
    return (
        historical.dropna(subset=["admin1"])
        .drop_duplicates(subset=["market"])
        .set_index("market")["admin1"]
        .to_dict()
    )


anomalies = load_anomalies()
volatility = load_volatility()
market_region_map = load_market_region_map()

anomalies = anomalies.copy()
anomalies["region"] = anomalies["market"].map(market_region_map)

volatility = volatility.copy()
volatility["region"] = volatility["region"].replace("—", "Unknown").fillna("Unknown")

render_header("Price Alerts & Volatility", "🚨")

regions = sorted(anomalies["region"].dropna().unique())
commodities = sorted(anomalies["commodity"].unique())

col1, col2 = st.columns(2)
selected_regions = col1.multiselect("Region", regions)
selected_commodities = col2.multiselect("Commodity", commodities)

filtered = anomalies
if selected_regions:
    filtered = filtered[filtered["region"].isin(selected_regions)]
if selected_commodities:
    filtered = filtered[filtered["commodity"].isin(selected_commodities)]

st.subheader("Active alerts")
active = filtered[filtered["status"] != "NORMAL"]
if active.empty:
    st.info("No active anomalies for the selected filters.")
else:
    st.dataframe(active, width="stretch", hide_index=True)

with st.expander(f"All monitored markets ({len(filtered)})"):
    st.dataframe(filtered, width="stretch", hide_index=True)

st.divider()
st.subheader("Market volatility")

vol_filtered = volatility
if selected_regions:
    vol_filtered = vol_filtered[vol_filtered["region"].isin(selected_regions)]
if selected_commodities:
    vol_filtered = vol_filtered[vol_filtered["commodity"].isin(selected_commodities)]

vol_filtered = vol_filtered.sort_values("current_volatility_pct", ascending=False)
st.dataframe(vol_filtered, width="stretch", hide_index=True)

render_footer()
