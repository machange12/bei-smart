"""Shared Streamlit UI components for AgriPulse pages."""

import streamlit as st

APP_NAME = "AgriPulse"
APP_TAGLINE = "Kenya food price forecasting and market intelligence."

DATA_SOURCES = (
    "WFP (via HDX)",
    "KAMIS (Kenya Ministry of Agriculture)",
    "FAO GIEWS",
    "KNBS",
    "ERA5 (via the World Bank Climate Change Knowledge Portal)",
)

CONFIDENCE_COLORS = {
    "high": "#1a7f37",
    "medium": "#b8860b",
    "low": "#c0392b",
}


def render_header(page_title: str | None = None, page_icon: str = "") -> None:
    """Render the shared AgriPulse name + tagline, optionally with a page subtitle."""
    st.title(f"🌾 {APP_NAME}")
    st.caption(APP_TAGLINE)
    if page_title:
        st.subheader(f"{page_icon} {page_title}".strip())


def render_footer() -> None:
    """Render the shared data-source attribution footer."""
    st.divider()
    st.caption("Data sources: " + " · ".join(DATA_SOURCES))


def confidence_badge(confidence: str) -> None:
    """Render a coloured pill badge for a forecast confidence level."""
    color = CONFIDENCE_COLORS.get(confidence, "#666666")
    st.markdown(
        f"""
        <span style="
            background-color:{color};
            color:white;
            padding:4px 14px;
            border-radius:999px;
            font-weight:600;
            font-size:0.95rem;
            display:inline-block;
        ">Confidence: {confidence.upper()}</span>
        """,
        unsafe_allow_html=True,
    )


def model_provenance_caption(strategy: str, test_mape: float) -> None:
    """Render a caption naming the model serving a forecast and its test MAPE."""
    st.caption(f"Served by model: `{strategy}` · Test MAPE: {test_mape:.2f}%")
