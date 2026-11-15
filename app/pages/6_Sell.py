"""Where to sell: wholesale prices in other markets, net of an adjustable transport cost."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.sell import DEFAULT_RATE, live_wholesale, market_coords, options  # noqa: E402
from src.ui import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

st.set_page_config(page_title="Where to sell | AgriPulse", page_icon="🚚", layout="wide")
inject_css()
sidebar_brand()


@st.cache_data(ttl="6h")
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(
        DATA_DIR / "bei_smart_forecasting_chained.csv", parse_dates=["date"],
        usecols=["commodity", "market", "pricetype", "date", "price_per_kg", "admin1", "latitude", "longitude"],
    )
    coords = market_coords(history)
    prices = live_wholesale(history)
    return prices[prices["market"].isin(coords["market"])], coords


prices, coords = load()
st.subheader("Where to sell")

counts = prices.groupby("commodity")["market"].nunique()
commodities = sorted(counts[counts >= 2].index)
if not commodities:
    st.info("No commodity has current wholesale prices in two or more markets right now.")
    st.stop()

with st.sidebar:
    st.markdown("#### Your produce")
    commodity = st.selectbox("Commodity", commodities,
                             index=commodities.index("Maize") if "Maize" in commodities else 0)
    here = prices[prices["commodity"] == commodity]
    origin = st.selectbox("Your nearest market", sorted(here["market"]))
    kg = st.number_input("Load (kg)", min_value=1, value=900, step=90, help="A 90 kg bag x 10 = 900 kg")
    rate = st.slider(
        "Transport cost (KES per kg per 100 km)", 0.5, 5.0, DEFAULT_RATE, 0.1,
        help="An estimate, not a quote. Around 1.5 is typical for a lorry on tarmac; "
             "use a higher figure for small loads or rough roads.",
    )

ranked = options(here, coords, origin, rate)
local = ranked[ranked["market"] == origin].iloc[0]
best = ranked.iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric(f"{origin} price", f"KES {local['price_per_kg']:.0f}/kg", help=f"Wholesale, {local['date']:%B %Y}")
if best["market"] == origin:
    col2.metric("Best option", "Sell locally")
    col3.metric("Extra from hauling", "KES 0")
    st.info(f"No other market pays enough for {commodity.lower()} to cover transport from {origin} at this rate.")
else:
    col2.metric("Best market", best["market"], f"{best['km']:.0f} km away", delta_color="off")
    col3.metric(f"Extra on {kg:,} kg", f"KES {best['net_gain'] * kg:,.0f}",
                f"{best['net_gain']:+.1f} KES/kg after transport")

table = ranked.assign(
    market=ranked["market"] + ranked["approx"].map({True: " ~", False: ""}),
    date=ranked["date"].dt.strftime("%b %Y"),
    total=ranked["net_gain"] * kg,
)[["market", "price_per_kg", "date", "km", "transport", "net_gain", "total"]]
st.dataframe(
    table, width="stretch", hide_index=True,
    column_config={
        "market": "Market",
        "price_per_kg": st.column_config.NumberColumn("Wholesale price (KES/kg)", format="%.0f"),
        "date": "Price as of",
        "km": st.column_config.NumberColumn("Distance (km)", format="%.0f"),
        "transport": st.column_config.NumberColumn("Transport (KES/kg)", format="%.1f"),
        "net_gain": st.column_config.NumberColumn("Gain vs selling locally (KES/kg)", format="%+.1f"),
        "total": st.column_config.NumberColumn(f"On {kg:,} kg (KES)", format="%+,.0f"),
    },
)
st.caption(
    "Prices are the latest monthly wholesale observations, not offers: confirm with a buyer before "
    "you load. Distance is the straight line x 1.3. Markets marked ~ are placed at the centre of "
    "their county, so their distances can be off by tens of km. "
    "Transport is your estimate from the slider; it excludes cess, loading and losses on the road."
)

sidebar_footer()
