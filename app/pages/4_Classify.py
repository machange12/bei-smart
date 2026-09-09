"""Classify page: form that calls the /classify API endpoint."""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
API_BASE_URL = os.environ.get("AGRIPULSE_API_URL", "http://localhost:8000")

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Classify | AgriPulse", page_icon="🔮", layout="wide")


@st.cache_data(ttl=300)
def fetch_metadata() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/metadata", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


render_header("Price Classification", "🔮")
st.markdown("Predict whether a price point is cheap, average, or expensive relative to norms.")

metadata = fetch_metadata()

if metadata is None:
    st.error(
        f"Cannot reach AgriPulse API at `{API_BASE_URL}`. "
        "Is it running? (`uvicorn api.main:app --reload`)"
    )
    render_footer()
    st.stop()

valid_values = metadata["valid_values"]

with st.form("classify_form"):
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", valid_values["region"])
        county = st.selectbox("County", valid_values["county"])
        market = st.selectbox("Market", valid_values["market"])
        category = st.selectbox("Category", valid_values["category"])
        commodity = st.selectbox("Commodity", valid_values["commodity"])
        season = st.selectbox("Season", valid_values["season"])

    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        year = st.number_input("Year", value=2025, step=1)
        rainfall_mm = st.number_input("Rainfall (mm)", value=80.0)
        diesel_price_kes = st.number_input("Diesel price (KES)", value=150.0)
        rolling_12m_avg = st.number_input("Rolling 12-month average price", value=60.0)
        latitude = st.number_input("Latitude", value=-1.28, format="%.4f")
        longitude = st.number_input("Longitude", value=36.82, format="%.4f")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "region": region,
        "county": county,
        "market": market,
        "category": category,
        "commodity": commodity,
        "season": season,
        "month": int(month),
        "year": int(year),
        "rainfall_mm": rainfall_mm,
        "diesel_price_kes": diesel_price_kes,
        "rolling_12m_avg": rolling_12m_avg,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/classify", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach AgriPulse API at `{API_BASE_URL}`. "
            "Is it running? (`uvicorn api.main:app --reload`)"
        )
        render_footer()
        st.stop()

    if resp.status_code == 200:
        result = resp.json()
        st.success(f"Predicted class: **{result['predicted_class']}**")
        proba_df = pd.DataFrame(
            {
                "class": list(result["probabilities"].keys()),
                "probability": list(result["probabilities"].values()),
            }
        ).set_index("class")
        st.bar_chart(proba_df)
    elif resp.status_code == 422:
        body = resp.json()
        errors = body.get("detail", {}).get("errors", [])
        if errors:
            for err in errors:
                st.error(
                    f"`{err['field']}` = {err['invalid_value']!r} is not valid. "
                    f"Allowed values: {', '.join(err['allowed_values'][:10])}"
                    + ("..." if len(err["allowed_values"]) > 10 else "")
                )
        else:
            st.error(f"Validation error: {body}")
    else:
        st.error(f"Unexpected error ({resp.status_code}): {resp.text}")

render_footer()
