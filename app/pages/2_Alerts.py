"""Alerts page: active price anomaly cards, full monitoring table, volatility."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import inject_css, sidebar_brand, sidebar_footer, status_badge  # noqa: E402

st.set_page_config(page_title="Alerts | AgriPulse", page_icon="🚨", layout="wide")
inject_css()
sidebar_brand()


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

st.subheader("Price Alerts & Volatility")

with st.sidebar:
    st.markdown("#### Filters")
    regions = sorted(anomalies["region"].dropna().unique())
    commodities = sorted(anomalies["commodity"].unique())
    selected_regions = st.multiselect("Region", regions)
    selected_commodities = st.multiselect("Commodity", commodities)

filtered = anomalies
if selected_regions:
    filtered = filtered[filtered["region"].isin(selected_regions)]
if selected_commodities:
    filtered = filtered[filtered["commodity"].isin(selected_commodities)]

active = filtered[filtered["status"] != "NORMAL"]
st.caption(f"{len(filtered):,} monitored · {len(active):,} flagged")

st.markdown("#### Active alerts")
if active.empty:
    st.info("No active anomalies for the selected filters.")
else:
    for _, row in active.iterrows():
        with st.container(border=True):
            head_col, badge_col = st.columns([4, 1])
            head_col.markdown(f"**{row['commodity']}** · {row['market']} ({row['pricetype']})")
            with badge_col:
                status_badge(row["status"])
            st.caption(
                f"Latest: KES {row['latest_price']:.2f} · Normal range: "
                f"KES {row['normal_low']:.2f}–{row['normal_high']:.2f} · "
                f"Deviation: {row['deviation_pct']:+.1f}%"
            )

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

sidebar_footer()
