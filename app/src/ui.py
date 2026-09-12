"""Shared visual identity and UI components for AgriPulse pages.

Every page imports colours and components from here rather than hardcoding
styling -- this is the single place the palette, chart styling and shared
widgets (badges, header/footer, the standard forecast chart) are defined.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_NAME = "AgriPulse"
APP_TAGLINE = "Kenya food price forecasting and market intelligence."

# Palette -- referenced by name everywhere else, never as a literal hex.
GREEN_DARK = "#1B5E20"    # primary -- headings, historical price line
GREEN_MID = "#388E3C"     # secondary
GREEN_LIGHT = "#A5D6A7"   # fills, hover
YELLOW = "#F9A825"        # accent -- forecast line
YELLOW_LIGHT = "#FFF3CD"  # confidence band
EARTH = "#795548"         # tertiary, sparing
CREAM = "#FAF9F5"         # page background
GREY = "#E0E0E0"          # gridlines, dividers

CONFIDENCE_COLORS = {"high": GREEN_DARK, "medium": "#B8860B", "low": "#C0392B"}
STATUS_COLORS = {"SPIKE": "#C0392B", "CRASH": "#1565C0", "NORMAL": "#757575"}

DATA_SOURCES = (
    "FEWS NET",
    "WFP (via HDX)",
    "KAMIS (Kenya Ministry of Agriculture)",
    "KNBS",
    "ERA5 (via the World Bank Climate Change Knowledge Portal)",
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def inject_css() -> None:
    """Apply the shared look: tighter vertical rhythm, a capped content
    width so text lines don't stretch edge-to-edge on wide monitors, and a
    light card treatment for st.metric so headline numbers read as a unit."""
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: 1150px;
            padding-top: 2rem;
        }}
        div[data-testid="stVerticalBlock"] {{
            gap: 0.5rem;
        }}
        div[data-testid="stMetric"] {{
            background-color: #F1F5EE;
            border: 1px solid {GREY};
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {EARTH};
        }}
        h1, h2, h3 {{
            color: {GREEN_DARK};
        }}
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {GREY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """AgriPulse name + tagline at the top of the sidebar, replacing the
    header that used to be repeated at the top of every page's main panel."""
    with st.sidebar:
        st.markdown(f"## 🌾 {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()


def sidebar_footer() -> None:
    """Data source attribution as a caption at the bottom of the sidebar."""
    with st.sidebar:
        st.divider()
        st.caption("Data sources: " + " · ".join(DATA_SOURCES))


def render_hero() -> None:
    """A short hero band for the landing page: a Kenyan agricultural photo
    with a dark green overlay so white title text stays legible. Falls back
    to a plain gradient if the asset is ever missing, so a bad deploy never
    crashes the page over a missing image file."""
    img_path = ASSETS_DIR / "hero.jpg"
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode()
        layers = (
            f"linear-gradient(rgba(27,94,32,0.55), rgba(27,94,32,0.55)), "
            f"url('data:image/jpeg;base64,{encoded}')"
        )
    else:
        layers = f"linear-gradient(135deg, {GREEN_DARK}, {GREEN_MID})"

    st.markdown(
        f"""
        <div style="
            height:200px;
            background-image:{layers};
            background-size:cover;
            background-position:center top;
            border-radius:12px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;
            margin-bottom:0.5rem;
        ">
            <div style="color:white;font-size:2.1rem;font-weight:700;">🌾 {APP_NAME}</div>
            <div style="color:white;font-size:1.05rem;opacity:0.95;">{APP_TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if img_path.exists():
        st.caption(
            "Photo: [\"Maize farming in Kenya\"](https://commons.wikimedia.org/wiki/File:Maize_farming_in_Kenya.jpg) "
            "by Kuza Kilimo, Wiki Loves Africa 2017 · CC BY-SA 4.0"
        )


def render_footer() -> None:
    """Kept for pages that want an end-of-page attribution line in addition
    to (or instead of) the sidebar footer."""
    st.divider()
    st.caption("Data sources: " + " · ".join(DATA_SOURCES))


def render_header(page_title: str | None = None) -> None:
    """Page subtitle only -- the AgriPulse name/tagline now live once in the
    sidebar via sidebar_brand(), not repeated on every page."""
    if page_title:
        st.subheader(page_title)


def confidence_badge(confidence: str) -> None:
    """Coloured pill: green for high confidence, amber for medium, red for low."""
    color = CONFIDENCE_COLORS.get(confidence, "#666666")
    st.markdown(
        f"""
        <span style="
            background-color:{color};
            color:white;
            padding:4px 14px;
            border-radius:999px;
            font-weight:600;
            font-size:0.9rem;
            display:inline-block;
        ">Confidence: {confidence.upper()}</span>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> None:
    """Coloured pill for anomaly status: red SPIKE, blue CRASH, grey NORMAL."""
    color = STATUS_COLORS.get(status, "#666666")
    st.markdown(
        f"""
        <span style="
            background-color:{color};
            color:white;
            padding:3px 12px;
            border-radius:999px;
            font-weight:600;
            font-size:0.8rem;
            display:inline-block;
        ">{status}</span>
        """,
        unsafe_allow_html=True,
    )


def model_provenance_caption(strategy: str, test_mape: float, last_observed: str | None = None) -> None:
    """Caption naming the model serving a forecast, its test MAPE, and --
    when supplied -- the last date it actually observed real data."""
    text = f"Served by `{strategy}` · Test MAPE: {test_mape:.2f}%"
    if last_observed:
        text += f" · Last observed {last_observed}"
    st.caption(text)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """The standard forecast figure: solid green historical line, dashed
    yellow forecast line, shaded confidence band, horizontal gridlines only.
    No title inside the figure -- callers use st.subheader above it instead
    so typography stays consistent with the rest of the page."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["price_per_kg"],
            name="Historical price",
            mode="lines",
            line=dict(color=GREEN_DARK, width=2),
            hovertemplate="%{x|%b %Y}<br>KES %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["forecast_upper"],
            name="Upper bound",
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["forecast_lower"],
            name="Confidence band",
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(249, 168, 37, 0.25)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["forecast_price_kes"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color=YELLOW, width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>KES %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        hovermode="x unified",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor=GREY, title="Price (KES)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def validation_chart(actual: pd.DataFrame, actual_col: str, forecast_col: str, date_col: str) -> go.Figure:
    """Actual-vs-forecast figure for the Validation page: solid green
    actual, dashed yellow forecast, the gap between them shaded."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual[date_col],
            y=actual[actual_col],
            name="Actual",
            mode="lines+markers",
            line=dict(color=GREEN_DARK, width=2),
            hovertemplate="%{x|%b %Y}<br>KES %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual[date_col],
            y=actual[forecast_col],
            name="Forecast",
            mode="lines+markers",
            line=dict(color=YELLOW, width=2, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(249, 168, 37, 0.25)",
            hovertemplate="%{x|%b %Y}<br>KES %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        hovermode="x unified",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor=GREY, title="Price (KES)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
