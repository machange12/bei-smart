"""Chat page: conversational interface backed by the routed hybrid RAG pipeline."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.rag import pipeline  # noqa: E402
from src.ui import render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Chat | AgriPulse", page_icon="💬", layout="wide")

CONFIDENCE_COLORS = {"high": "#1a7f37", "medium": "#b8860b", "low": "#c0392b"}

render_header("Ask AgriPulse", "💬")
st.caption(
    "Ask about forecasts, price alerts, or which market is cheapest -- in English or Kiswahili."
)

EXAMPLE_QUESTIONS = [
    "What is the maize price forecast for Nairobi wholesale next month?",
    "Bei ya mahindi itakuwa ngapi Nairobi jumla mwezi ujao?",
    "Why do prices vary so much by region?",
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

example_cols = st.columns(len(EXAMPLE_QUESTIONS))
example_clicked = None
for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
    if col.button(example, width="stretch"):
        example_clicked = example

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("meta"):
            meta = turn["meta"]
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

user_input = st.chat_input("Ask a question...") or example_clicked

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking this up..."):
            result = pipeline.answer(user_input)
        st.markdown(result["answer"])
        if result.get("confidence"):
            color = CONFIDENCE_COLORS.get(result["confidence"], "#666666")
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 10px;"
                f"border-radius:999px;font-size:0.8rem;'>Confidence: "
                f"{result['confidence'].upper()}</span>",
                unsafe_allow_html=True,
            )
        if result.get("sources"):
            st.caption(f"Source: {', '.join(result['sources'])} · status: {result['status']}")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "confidence": result.get("confidence"),
                "sources": result.get("sources"),
                "status": result.get("status"),
            },
        }
    )

render_footer()
