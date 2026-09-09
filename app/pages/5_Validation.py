"""Validation page: forecast-vs-actual evidence from discontinued series."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Validation | AgriPulse", page_icon="✅", layout="wide")


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "validation_summary.csv",
        parse_dates=["period_start", "period_end", "series_ended"],
    )


@st.cache_data
def load_archive() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation_archive.csv", parse_dates=["date", "series_ended"])


summary = load_summary()
archive = load_archive()

render_header("Validation Archive", "✅")

st.info(
    "These 17 series were discontinued by the data provider between 2019 and 2022 "
    "(Nairobi, Nakuru, Mombasa, Kisumu, and Eldoret wholesale markets). They are **not** "
    "current prices — the forecast period for each has already occurred, so the error "
    "shown here is verified against real outcomes rather than claimed in advance. They "
    "are retained solely as evidence of forecasting method accuracy."
)

overall_median_error = archive["error_pct"].median()
st.metric(
    "Overall median error (all forecast-actual pairs)",
    f"{overall_median_error:.1f}%",
    help=f"Computed across {len(archive)} monthly forecast-actual pairs from all 17 archived series.",
)

st.divider()

commodities = sorted(summary["commodity"].unique())
commodity = st.selectbox("Commodity", commodities)

markets = sorted(summary.loc[summary["commodity"] == commodity, "market"].unique())
market = st.selectbox("Market", markets)

series_summary = summary[(summary["commodity"] == commodity) & (summary["market"] == market)]

if series_summary.empty:
    st.warning("No archived series for this combination.")
else:
    row = series_summary.iloc[0]
    st.caption(
        f"Backtest · {row['pricetype']} · {row['region']} · {int(row['n_months'])} months "
        f"({row['period_start'].date()} to {row['period_end'].date()}) · "
        f"Median error: {row['median_error']:.1f}% · Mean error: {row['mean_error']:.1f}% · "
        f"Series ended: {row['series_ended'].date()}"
    )

    series_pairs = archive[
        (archive["commodity"] == commodity) & (archive["market"] == market)
    ].sort_values("date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series_pairs["date"],
            y=series_pairs["actual_kes"],
            name="Actual",
            mode="lines+markers",
            line=dict(color="#1a7f37"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series_pairs["date"],
            y=series_pairs["forecast_kes"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#ff7f0e", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.15)",
        )
    )

    fig.update_layout(
        title=f"{commodity} — {market} ({row['pricetype']}): forecast vs. actual",
        xaxis_title="Date",
        yaxis_title="Price (KES)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("All archived series, sorted by median error")

table = summary[
    ["commodity", "market", "pricetype", "region", "n_months", "median_error", "mean_error", "series_ended"]
].sort_values("median_error")
st.dataframe(table, width="stretch", hide_index=True)

render_footer()
