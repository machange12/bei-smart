"""Coming soon: series still reporting that will clear the data gates within a year."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.ui import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

st.set_page_config(page_title="Coming soon | AgriPulse", page_icon="⏳", layout="wide")
inject_css()
sidebar_brand()


@st.cache_data
def load_watchlist() -> pd.DataFrame:
    path = DATA_DIR / "watchlist.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def served_markets() -> set:
    m = pd.read_csv(DATA_DIR / "model_metrics.csv")
    return set(m.loc[m["reliable"], "market"])


wl = load_watchlist()

st.subheader("Coming soon")
st.info(
    "Series we can't forecast yet, but will soon. We only forecast a series with at least "
    "**36 months** of prices and **fewer than 25% of months missing**. These are still "
    "reporting every month and will pass both rules within a year. When a series passes, "
    "the monthly refresh models it on its own, and it goes live if its error on 12 held-out "
    "months is 25% or less."
)

if wl.empty:
    st.success("Nothing is waiting: every series that's still reporting has already passed the data rules.")
    sidebar_footer()
    st.stop()

new_markets = sorted(set(wl["market"]) - served_markets())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Series waiting", len(wl))
c2.metric("Ready within 3 months", int((wl["months_to_go"] <= 3).sum()))
c3.metric("Markets", wl["market"].nunique())
c4.metric("New markets", len(new_markets), help="Markets with no forecast today: " + ", ".join(new_markets))

with st.sidebar:
    st.markdown("#### Filters")
    regions = st.multiselect("Region", sorted(wl["region"].unique()))
    commodities = st.multiselect("Commodity", sorted(wl["commodity"].unique()))
    only_new = st.toggle("Only new markets")

view = wl
if regions:
    view = view[view["region"].isin(regions)]
if commodities:
    view = view[view["commodity"].isin(commodities)]
if only_new:
    view = view[view["market"].isin(new_markets)]

st.dataframe(
    view[["commodity", "market", "pricetype", "region", "n_records", "months_to_go",
          "eligible_from", "last_obs", "source"]],
    hide_index=True,
    width="stretch",
    column_config={
        "commodity": "Commodity",
        "market": "Market",
        "pricetype": "Price type",
        "region": "Region",
        "n_records": st.column_config.NumberColumn("Months of data"),
        "months_to_go": st.column_config.ProgressColumn(
            "Months to go", min_value=0, max_value=12, format="%d"),
        "eligible_from": "Eligible from",
        "last_obs": "Last price",
        "source": "Source",
    },
)
st.caption("Months to go assumes the series keeps reporting every month. A missed month pushes it back.")

sidebar_footer()
