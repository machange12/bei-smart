"""Map page: markets colored by anomaly status."""

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
KENYA_CENTER = [0.0236, 37.9062]

STATUS_COLORS = {"NORMAL": "green", "SPIKE": "red"}

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

st.set_page_config(page_title="Map | AgriPulse", page_icon="🗺️", layout="wide")
inject_css()
sidebar_brand()


@st.cache_data
def load_markets() -> pd.DataFrame:
    historical = pd.read_csv(
        DATA_DIR / "bei_smart_forecasting_final.csv",
        usecols=["market", "latitude", "longitude", "admin1"],
    )
    return (
        historical.dropna(subset=["latitude", "longitude"])
        .groupby("market")
        .first()
        .reset_index()
    )


@st.cache_data
def load_market_status() -> pd.DataFrame:
    anomalies = pd.read_csv(DATA_DIR / "price_anomalies.csv")

    def worst_status(statuses: pd.Series) -> str:
        return "SPIKE" if (statuses == "SPIKE").any() else "NORMAL"

    grouped = anomalies.groupby("market").agg(
        status=("status", worst_status),
        commodities=("commodity", lambda s: ", ".join(sorted(s.unique()))),
    )
    return grouped.reset_index()


markets = load_markets()
market_status = load_market_status()

markets = markets.merge(market_status, on="market", how="left")

st.subheader("Market Map")
st.markdown("🟢 Normal · 🔴 Spike · ⚪ No anomaly data")

m = folium.Map(location=KENYA_CENTER, zoom_start=6)

for row in markets.to_dict("records"):
    status = row["status"] if pd.notna(row["status"]) else None
    color = STATUS_COLORS.get(status, "gray")
    commodities = row["commodities"] if pd.notna(row.get("commodities")) else "No anomaly data"
    popup = folium.Popup(
        f"<b>{row['market']}</b><br>Region: {row['admin1']}<br>"
        f"Status: {status or 'Unknown'}<br>Commodities: {commodities}",
        max_width=300,
    )
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=popup,
    ).add_to(m)

st_folium(m, use_container_width=True, height=600)

sidebar_footer()
