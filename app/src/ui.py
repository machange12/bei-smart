"""Shared visual identity and UI components for AgriPulse pages.

Every page imports colours and components from here rather than hardcoding
styling -- this is the single place the palette, chart styling and shared
widgets (badges, header/footer, the standard forecast chart) are defined.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.units import unit_label

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
# Routing strategy codes -> what a non-technical reader sees.
STRATEGY_NAMES = {
    "ensemble": "blend of the 2 best models",
    "chronos": "Chronos (AI foundation model)",
    "prophet": "Prophet",
    "sarima": "SARIMA (statistical)",
    "lgbm": "LightGBM (learns across all markets)",
}

DATA_SOURCES = (
    "FEWS NET",
    "WFP (via HDX)",
    "KAMIS (Kenya Ministry of Agriculture)",
    "KNBS",
    "FAO GIEWS FPMA",
    "ERA5 (via the World Bank Climate Change Knowledge Portal)",
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SERIES_KEY = ["commodity", "market", "pricetype"]
# Matches the GitHub Actions cron (5th of each month, 03:00 UTC).
REFRESH_NOTE = "refreshes automatically on the 5th of each month"


@st.cache_data(ttl="6h")
def latest_prices() -> pd.DataFrame:
    """Last observed price per series. Shared by the freshness banner and
    the landing cards so the big history CSV is parsed once per process.
    The ttl lets a monthly data refresh show up without a restart."""
    try:
        history = pd.read_csv(
            DATA_DIR / "bei_smart_forecasting_chained.csv",
            usecols=SERIES_KEY + ["date", "price_per_kg"],
            parse_dates=["date"],
        )
    except (OSError, ValueError):  # missing file or columns -- banner just hides
        return pd.DataFrame(columns=SERIES_KEY + ["date", "price_per_kg"])
    return history.dropna(subset=["date"]).sort_values("date").groupby(SERIES_KEY).tail(1)


def _freshness_banner() -> None:
    """One slim line so nobody has to wonder how current the numbers are."""
    as_of = latest_prices()["date"].max()
    if pd.notna(as_of):
        st.caption(f"📅 Data as of {as_of:%B %Y} · {REFRESH_NOTE}")


@st.cache_resource
def _warm_chat() -> None:
    """Load the RAG stack in a background thread, once per process. The first
    chat question otherwise pays ~60s for chromadb + the embedding model; the
    modules keep their loaded client/collection in globals, so warming once
    is enough. Failures are swallowed: chat just falls back to loading lazily."""
    def warm() -> None:
        try:
            from src.rag import pipeline
            pipeline.vectorstore.query("warm up", n_results=1)
        except Exception:
            pass

    # ponytail: a question asked mid-warm-up may load the model a second time; add a lock in vectorstore if memory bites.
    threading.Thread(target=warm, daemon=True).start()


def inject_css() -> None:
    """Apply the shared look: tighter vertical rhythm, a capped content
    width so text lines don't stretch edge-to-edge on wide monitors, and a
    light card treatment for st.metric so headline numbers read as a unit."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .block-container {{
            max-width: 1150px;
            padding-top: 4rem;  /* clears the 60px top bar so the freshness line isn't hidden */
        }}
        div[data-testid="stVerticalBlock"] {{
            gap: 0.5rem;
        }}
        div[data-testid="stMetric"] {{
            background-color: #FFFFFF;
            border: 1px solid {GREY};
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
        .st-key-chat_fab {{
            position: fixed;
            /* Streamlit Cloud pins its own "Manage app" button and hosting badge to the
               bottom-right corner; sit above them instead of underneath. */
            bottom: 5rem;
            right: 1.5rem;
            z-index: 1000;
            width: auto;
        }}
        .st-key-chat_fab button {{
            border-radius: 999px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }}
        /* Chat dialog as a floating window: no dimmed backdrop, the page stays
           clickable behind it, and the title bar is the drag handle (CHAT_DRAG_JS). */
        div[data-testid="stDialog"] {{
            background: transparent !important;
            pointer-events: none;
            overflow: hidden;
        }}
        div[data-testid="stDialog"] > div {{
            pointer-events: auto;
            width: min(640px, 94vw) !important;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.28);
            border: 1px solid {GREY};
        }}
        div[data-testid="stDialog"] [role="dialog"] h2 {{
            cursor: move;
            user-select: none;
        }}
        /* Landing "This month" cards: the bordered container is the card,
           so drop the metric's own card styling inside it. */
        .st-key-this_month div[data-testid="stMetric"] {{
            background: none;
            border: none;
            padding: 0;
        }}
        /* ---------- polish: type, motion, buttons, cards, tabs, links ---------- */
        html, body, .stApp, button, input, textarea, [data-baseweb] {{
            font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
        }}
        :root {{
            --ap-ease: cubic-bezier(.2, .7, .2, 1);
            --ap-shadow: 0 1px 2px rgba(27, 94, 32, .06), 0 4px 14px rgba(27, 94, 32, .07);
            --ap-shadow-lift: 0 2px 4px rgba(27, 94, 32, .08), 0 12px 28px rgba(27, 94, 32, .14);
        }}
        h1, h2, h3 {{ letter-spacing: -0.015em; font-weight: 700; }}

        /* each page eases in instead of snapping */
        @keyframes ap-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
        .block-container {{ animation: ap-rise .35s var(--ap-ease) both; }}

        /* buttons: one shape, clear hover / press / keyboard focus */
        div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {{
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
            transition: transform .15s var(--ap-ease), box-shadow .15s var(--ap-ease),
                        background-color .15s, border-color .15s, color .15s;
        }}
        div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {{
            transform: translateY(-1px);
            box-shadow: var(--ap-shadow-lift);
            border-color: {GREEN_MID};
        }}
        div[data-testid="stButton"] button:active, div[data-testid="stDownloadButton"] button:active,
        div[data-testid="stFormSubmitButton"] button:active {{
            transform: translateY(0) scale(.98);
            box-shadow: none;
        }}
        button:focus-visible, a:focus-visible, [role="tab"]:focus-visible {{
            outline: 3px solid {GREEN_LIGHT};
            outline-offset: 2px;
        }}
        button[kind="primary"], button[kind="primaryFormSubmit"] {{
            background: linear-gradient(180deg, {GREEN_MID}, {GREEN_DARK});
            border: 1px solid {GREEN_DARK};
        }}
        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{ filter: brightness(1.08); }}

        /* cards: metrics, Home cards, alert cards, the sign-up form */
        div[data-testid="stMetric"], [class*="st-key-card_"], [class*="st-key-alert_"], div[data-testid="stForm"] {{
            border-radius: 14px !important;
            box-shadow: var(--ap-shadow);
            transition: transform .2s var(--ap-ease), box-shadow .2s var(--ap-ease), border-color .2s;
        }}
        [class*="st-key-card_"], [class*="st-key-alert_"], div[data-testid="stForm"] {{ background: #FFFFFF; }}
        div[data-testid="stMetric"]:hover, [class*="st-key-card_"]:hover, [class*="st-key-alert_"]:hover {{
            transform: translateY(-2px);
            box-shadow: var(--ap-shadow-lift);
            border-color: {GREEN_LIGHT} !important;
        }}
        div[data-testid="stMetricValue"] {{ font-weight: 700; letter-spacing: -0.02em; }}
        .st-key-this_month div[data-testid="stMetric"], .st-key-this_month div[data-testid="stMetric"]:hover {{
            box-shadow: none;
            transform: none;
        }}

        /* links that go somewhere: the arrow nudges on hover */
        a[data-testid="stPageLink-NavLink"] {{
            border-radius: 8px;
            padding: 0.25rem 0.5rem;
            margin-left: -0.5rem;
            transition: background-color .15s, padding-left .2s var(--ap-ease);
        }}
        a[data-testid="stPageLink-NavLink"] p {{ color: {GREEN_DARK}; font-weight: 600; }}
        a[data-testid="stPageLink-NavLink"]:hover {{ background: #F1F5EE; padding-left: 0.8rem; }}

        /* sidebar navigation */
        a[data-testid="stSidebarNavLink"] {{
            border-radius: 10px;
            transition: background-color .15s, transform .15s var(--ap-ease);
        }}
        a[data-testid="stSidebarNavLink"]:hover {{ transform: translateX(3px); }}
        a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: #E3EEDD; }}
        a[data-testid="stSidebarNavLink"][aria-current="page"] span {{ font-weight: 600; color: {GREEN_DARK}; }}

        /* tabs */
        div[data-testid="stTabs"] [role="tab"] {{
            padding: 0.6rem 1.1rem;
            font-weight: 600;
            border-radius: 10px 10px 0 0;
            transition: background-color .15s, color .15s;
        }}
        div[data-testid="stTabs"] [role="tab"]:hover {{ background: #F1F5EE; }}
        div[data-testid="stTabs"] [role="tabpanel"] {{ animation: ap-rise .3s var(--ap-ease) both; }}

        /* inputs, expanders, tables, charts */
        div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] {{
            border-radius: 10px !important;
            transition: border-color .15s, box-shadow .15s;
        }}
        div[data-baseweb="select"] > div:hover {{ border-color: {GREEN_MID} !important; }}
        div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within {{
            box-shadow: 0 0 0 3px {GREEN_LIGHT}66;
        }}
        details[data-testid="stExpander"], div[data-testid="stExpander"] details {{
            border-radius: 12px;
            overflow: hidden;
            transition: box-shadow .2s var(--ap-ease);
        }}
        div[data-testid="stExpander"] summary {{ font-weight: 600; transition: background-color .15s; }}
        div[data-testid="stExpander"] summary:hover {{ background: #F1F5EE; color: {GREEN_DARK}; }}
        div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"], div[data-testid="stCustomComponentV1"],
        iframe[title="streamlit_folium.st_folium"] {{
            border-radius: 12px;
            overflow: hidden;
        }}
        div[data-testid="stPlotlyChart"] {{ box-shadow: var(--ap-shadow); background: #FFFFFF; padding: 0.5rem; }}
        div[data-testid="stAlert"] {{ border-radius: 12px; animation: ap-rise .25s var(--ap-ease) both; }}

        /* hero */
        .ap-hero {{
            height: 220px;
            background-size: cover;
            background-position: center 30%;
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            margin-bottom: 0.5rem;
            box-shadow: var(--ap-shadow-lift);
            overflow: hidden;
        }}
        .ap-hero .ap-title {{
            color: #fff; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
            text-shadow: 0 2px 12px rgba(0, 0, 0, .35);
            animation: ap-rise .6s var(--ap-ease) both;
        }}
        .ap-hero .ap-tag {{
            color: #fff; font-size: 1.1rem; opacity: .95; margin-top: .25rem;
            text-shadow: 0 1px 8px rgba(0, 0, 0, .35);
            animation: ap-rise .6s .12s var(--ap-ease) both;
        }}

        /* chat button */
        .st-key-chat_fab button {{ transition: transform .2s var(--ap-ease), box-shadow .2s var(--ap-ease); }}
        .st-key-chat_fab button:hover {{ transform: translateY(-2px) scale(1.04); box-shadow: 0 8px 22px rgba(0, 0, 0, .3); }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ animation: none !important; transition: none !important; }}
        }}
        @media (max-width: 640px) {{
            .ap-hero {{ height: 170px; }}
            .ap-hero .ap-title {{ font-size: 1.8rem; }}
            .block-container {{
                padding: 4rem 1rem 8rem;  /* bottom room so the chat button never hides the last row */
            }}
            .st-key-chat_fab {{
                bottom: 4.5rem;
                right: 1rem;
            }}
            .st-key-chat_fab button strong {{
                display: none;  /* icon-only on phones */
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Every page calls inject_css() first, so the chat launcher and the
    # freshness line ride along everywhere -- even on pages that st.stop() early.
    _freshness_banner()
    _warm_chat()
    with st.container(key="chat_fab"):
        st.html(CHAT_DRAG_JS, unsafe_allow_javascript=True)
        # Bold label so the mobile CSS can hide the words and keep the icon.
        if st.button("💬 **Ask AgriPulse**", type="primary"):
            _chat_dialog()


# Drag the chat window by its title. Installed once per browser tab; a new dialog
# (each open, each page) is picked up by the observer and placed where it was left.
CHAT_DRAG_JS = """
<script>
(() => {
  if (window.__agriDrag) return;
  window.__agriDrag = true;
  const pos = {x: 0, y: 0};
  const place = (card) => { card.style.transform = `translate(${pos.x}px, ${pos.y}px)`; };
  const wire = (card) => {
    if (card.dataset.drag) return;
    card.dataset.drag = "1";
    place(card);
    const handle = card.querySelector('[role="dialog"] h2');
    if (!handle) return;
    handle.addEventListener("pointerdown", (e) => {
      const start = {x: e.clientX - pos.x, y: e.clientY - pos.y};
      const r = card.getBoundingClientRect();
      const base = {left: r.left - pos.x, top: r.top - pos.y};  // untranslated position
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => {
        // fully on screen sideways; the title bar always stays grabbable vertically
        pos.x = Math.min(Math.max(ev.clientX - start.x, -base.left), innerWidth - r.width - base.left);
        pos.y = Math.min(Math.max(ev.clientY - start.y, -base.top), innerHeight - 60 - base.top);
        place(card);
      };
      const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  };
  new MutationObserver(() => {
    document.querySelectorAll('div[data-testid="stDialog"] > div').forEach(wire);
  }).observe(document.body, {childList: true, subtree: true});
})();
</script>
"""

CHAT_EXAMPLES = [
    "What is the maize price forecast for Nairobi wholesale next month?",
    "Bei ya mahindi itakuwa ngapi Nairobi jumla mwezi ujao?",
    "Why do prices vary so much by region?",
]


def _chat_meta(meta: dict) -> None:
    """Confidence pill + source caption under an assistant reply."""
    if meta.get("confidence"):
        color = CONFIDENCE_COLORS.get(meta["confidence"], "#666666")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:2px 10px;"
            f"border-radius:999px;font-size:0.8rem;'>Confidence: "
            f"{meta['confidence'].upper()}</span>",
            unsafe_allow_html=True,
        )
    if meta.get("sources"):
        st.caption(f"Source: {', '.join(meta['sources'])} · status: {meta['status']}")


@st.dialog("Ask AgriPulse", width="medium")
def _chat_dialog() -> None:
    """RAG chat in a modal. Dialogs rerun as fragments, so chatting doesn't
    close it; history lives in session_state and follows the user across pages."""
    from src.rag import pipeline  # lazy: chroma + embeddings only load once someone chats

    st.caption("Ask about forecasts, price alerts, or which market is cheapest -- "
               "in English or Kiswahili.")
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_followup", None)

    example_clicked = None
    for col, example in zip(st.columns(len(CHAT_EXAMPLES)), CHAT_EXAMPLES):
        if col.button(example, width="stretch"):
            example_clicked = example

    log = st.container(height=420)
    with log:
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant":
                    _chat_meta(turn.get("meta") or {})

    user_input = st.chat_input("Ask a question...") or example_clicked
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with log:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Looking this up..."):
                if st.session_state.pending_followup is not None:
                    # Answering last turn's follow-up (e.g. "which market?"), not a fresh question.
                    result = pipeline.continue_answer(st.session_state.pending_followup, user_input)
                else:
                    result = pipeline.answer(user_input)
            st.session_state.pending_followup = result.get("pending")
            meta = {k: result.get(k) for k in ("confidence", "sources", "status")}
            st.markdown(result["answer"])
            _chat_meta(meta)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["answer"], "meta": meta}
    )


def t(en: str, sw: str) -> str:
    """The English or Kiswahili wording, per the sidebar language switch."""
    return sw if st.session_state.get("lang") == "Kiswahili" else en


def sidebar_brand() -> None:
    """AgriPulse name + tagline at the top of the sidebar, replacing the
    header that used to be repeated at the top of every page's main panel."""
    with st.sidebar:
        st.markdown(f"## 🌾 {APP_NAME}")
        # Widget state is dropped on page change unless re-assigned, so keep the choice sticky.
        if "lang" in st.session_state:
            st.session_state["lang"] = st.session_state["lang"]
        st.radio("Language / Lugha", ["English", "Kiswahili"], key="lang", horizontal=True,
                 label_visibility="collapsed")
        st.caption(t(APP_TAGLINE, "Utabiri wa bei za chakula na taarifa za masoko Kenya."))
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
            f"linear-gradient(160deg, rgba(27,94,32,0.78), rgba(27,94,32,0.35)), "
            f"url('data:image/jpeg;base64,{encoded}')"
        )
    else:
        layers = f"linear-gradient(135deg, {GREEN_DARK}, {GREEN_MID})"

    st.markdown(
        f"""
        <div class="ap-hero" style="background-image:{layers};">
            <div class="ap-title">🌾 {APP_NAME}</div>
            <div class="ap-tag">{t(APP_TAGLINE, "Utabiri wa bei za chakula na taarifa za masoko Kenya.")}</div>
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
    """Plain-English caption naming the model serving a forecast, its typical
    error (test MAPE, same term as the Forecast page), and -- when supplied --
    the last date it actually observed real data."""
    model = STRATEGY_NAMES.get(strategy, strategy)
    text = f"Model: {model} · typical error ±{test_mape:.1f}%"
    if last_observed:
        text += f" · last price observed {last_observed}"
    st.caption(text)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """The standard forecast figure: solid green historical line, dashed
    yellow forecast line, shaded confidence band, horizontal gridlines only.
    No title inside the figure -- callers use st.subheader above it instead
    so typography stays consistent with the rest of the page."""
    unit = unit_label(forecast["commodity"].iloc[0]) if "commodity" in forecast and len(forecast) else "kg"
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["price_per_kg"],
            name=t("Past prices", "Bei zilizopita"),
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
            name=t("80% range", "Wigo wa 80%"),
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
            name=t("Forecast", "Utabiri"),
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
        yaxis=dict(showgrid=True, gridcolor=GREY, title=t(f"Price (KES/{unit})", f"Bei (KES/{unit})")),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
