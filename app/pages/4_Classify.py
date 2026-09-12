"""Classify page: simple commodity + region form, predicting in-process.

The underlying model needs 13 features (region, county, market, category,
commodity, season, month, year, rainfall_mm, diesel_price_kes,
rolling_12m_avg, latitude, longitude) -- that's model plumbing, not
something a user should have to think about. This page asks for only the
two things that actually vary by question (commodity, region) and fills
the rest from a representative market per region, the current date, and
recent historical averages. The "Show what's being sent" expander keeps
this transparent rather than hiding it outright.

Prediction runs directly in this process (loading the ~13MB classifier
artifacts once via st.cache_resource) rather than calling out to the
FastAPI service -- Streamlit Cloud has no way to reach a locally-run API,
and shipping one extra hosted service just for this one endpoint isn't
worth it when the artifacts are small enough to ship with the app. The
FastAPI backend still exposes the same /classify endpoint for local dev
and any other client that wants it; this page just no longer depends on it
being up.
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

st.set_page_config(page_title="Classify | AgriPulse", page_icon="🔮", layout="wide")
inject_css()
sidebar_brand()

# One representative (county, market, lat, long, rainfall) per classifier
# region, verified against the classifier's own valid_values so the model
# never sees a county/market it wasn't trained on. Rainfall is a recent
# regional average from the training data.
REGION_DEFAULTS = {
    "Coast": {"county": "Kilifi", "market": "Kilifi", "latitude": -3.63, "longitude": 39.85, "rainfall_mm": 47.5},
    "Eastern": {"county": "Kitui", "market": "Kitui", "latitude": -1.37, "longitude": 38.02, "rainfall_mm": 33.1},
    "Nairobi": {"county": "Nairobi", "market": "Nairobi", "latitude": -1.28, "longitude": 36.82, "rainfall_mm": 50.5},
    "North Eastern": {"county": "Mandera", "market": "Mandera", "latitude": 3.94, "longitude": 41.86, "rainfall_mm": 17.9},
    "Rift Valley": {"county": "Turkana", "market": "Lodwar (Turkana)", "latitude": 3.12, "longitude": 35.60, "rainfall_mm": 107.1},
}

COMMODITY_CATEGORY = {
    "Beans": "pulses and nuts", "Beans (dry)": "pulses and nuts", "Bread": "cereals and tubers",
    "Cooking fat": "oil and fats", "Cowpeas (dry)": "pulses and nuts", "Kale": "vegetables and fruits",
    "Maize": "cereals and tubers", "Maize (white)": "cereals and tubers",
    "Maize (white, dry)": "cereals and tubers", "Maize flour": "cereals and tubers",
    "Maize flour (white)": "cereals and tubers", "Meat (beef)": "meat, fish and eggs",
    "Meat (camel)": "meat, fish and eggs", "Meat (goat)": "meat, fish and eggs",
    "Milk (UHT)": "milk and dairy", "Milk (camel, fresh)": "milk and dairy",
    "Milk (cow, fresh)": "milk and dairy", "Milk (cow, pasteurized)": "milk and dairy",
    "Oil (vegetable)": "oil and fats", "Oil (vegetable, fortified)": "oil and fats",
    "Onions (red)": "vegetables and fruits", "Pigeon peas (dry)": "pulses and nuts",
    "Potatoes (Irish)": "cereals and tubers", "Rice": "cereals and tubers",
    "Rice (aromatic)": "cereals and tubers", "Salt": "miscellaneous food",
    "Sorghum": "cereals and tubers", "Spinach": "vegetables and fruits",
    "Sugar": "miscellaneous food", "Tomatoes": "vegetables and fruits", "Wheat flour": "cereals and tubers",
}

# Recent 12-month average price per commodity (falls back to the full-history
# average for the handful of commodities with too few recent records) --
# used as the rolling_12m_avg feature so the user never has to supply it.
COMMODITY_ROLLING_AVG = {
    "Beans": 138.33, "Beans (dry)": 121.74, "Bread": 113.06, "Cooking fat": 188.72,
    "Cowpeas (dry)": 146.64, "Kale": 82.74, "Maize": 60.55, "Maize (white)": 53.33,
    "Maize (white, dry)": 78.68, "Maize flour": 87.42, "Maize flour (white)": 90.29,
    "Meat (beef)": 642.46, "Meat (camel)": 635.17, "Meat (goat)": 732.27, "Milk (UHT)": 165.48,
    "Milk (camel, fresh)": 184.26, "Milk (cow, fresh)": 166.68, "Milk (cow, pasteurized)": 146.00,
    "Oil (vegetable)": 279.89, "Oil (vegetable, fortified)": 282.23, "Onions (red)": 57.00,
    "Pigeon peas (dry)": 161.00, "Potatoes (Irish)": 76.21, "Rice": 120.68,
    "Rice (aromatic)": 128.22, "Salt": 75.42, "Sorghum": 71.75, "Spinach": 79.90,
    "Sugar": 151.46, "Tomatoes": 58.00, "Wheat flour": 96.94,
}

# Kenya's four-season split by calendar month, matching the classifier's
# own training data exactly (verified against bei_smart_classification.csv).
MONTH_TO_SEASON = {
    1: "Cool Dry", 2: "Cool Dry",
    3: "Long Rains", 4: "Long Rains", 5: "Long Rains",
    6: "Dry Season", 7: "Dry Season", 8: "Dry Season", 9: "Dry Season",
    10: "Short Rains", 11: "Short Rains", 12: "Short Rains",
}

DIESEL_PRICE_DEFAULT_KES = 178.26  # recent national average


@st.cache_resource
def load_classifier() -> dict:
    import json

    metadata = json.loads((MODELS_DIR / "classifier_metadata.json").read_text())
    return {
        "metadata": metadata,
        "encoders": joblib.load(MODELS_DIR / "classifier_encoders.joblib"),
        "label_encoder": joblib.load(MODELS_DIR / "classifier_label_encoder.joblib"),
        "xgb_model": joblib.load(MODELS_DIR / "classifier_xgb.joblib"),
    }


def predict(payload: dict, artifacts: dict) -> tuple[str, dict[str, float]]:
    """Mirrors api/main.py's /classify handler exactly: validate categoricals
    against valid_values, encode, predict, map probabilities to class names."""
    metadata = artifacts["metadata"]
    categorical_features = metadata["categorical_features"]

    X = pd.DataFrame([payload])[metadata["feature_order"]]
    for col in categorical_features:
        X[col] = artifacts["encoders"][col].transform(X[col].astype(str))

    xgb_model = artifacts["xgb_model"]
    label_encoder = artifacts["label_encoder"]
    pred_idx = xgb_model.predict(X)
    predicted_class = label_encoder.inverse_transform(pred_idx)[0]
    proba = xgb_model.predict_proba(X)[0]
    probabilities = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}
    return predicted_class, probabilities


st.subheader("Price Classification")
st.markdown("Predict whether a price point is cheap, average, or expensive relative to norms.")

try:
    artifacts = load_classifier()
except FileNotFoundError as exc:
    st.error(f"Classifier artifacts missing: {exc}")
    sidebar_footer()
    st.stop()

valid_values = artifacts["metadata"]["valid_values"]
regions = [r for r in valid_values["region"] if r in REGION_DEFAULTS]

with st.form("classify_form"):
    col1, col2 = st.columns(2)
    with col1:
        commodity = st.selectbox("Commodity", valid_values["commodity"])
    with col2:
        region = st.selectbox("Region", regions)
    submitted = st.form_submit_button("Predict")

if submitted:
    defaults = REGION_DEFAULTS[region]
    today = date.today()
    season = MONTH_TO_SEASON[today.month]

    payload = {
        "region": region,
        "county": defaults["county"],
        "market": defaults["market"],
        "category": COMMODITY_CATEGORY.get(commodity, "miscellaneous food"),
        "commodity": commodity,
        "season": season,
        "month": today.month,
        "year": today.year,
        "rainfall_mm": defaults["rainfall_mm"],
        "diesel_price_kes": DIESEL_PRICE_DEFAULT_KES,
        "rolling_12m_avg": COMMODITY_ROLLING_AVG.get(commodity, 100.0),
        "latitude": defaults["latitude"],
        "longitude": defaults["longitude"],
    }

    with st.expander("What was sent to the model"):
        st.json(payload)

    try:
        predicted_class, probabilities = predict(payload, artifacts)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        sidebar_footer()
        st.stop()

    st.success(f"Predicted class: **{predicted_class}**")
    proba_df = pd.DataFrame(
        {"class": list(probabilities.keys()), "probability": list(probabilities.values())}
    ).set_index("class")
    st.bar_chart(proba_df)

sidebar_footer()
