"""AgriPulse landing page: hero band, "this month" cards, headline metrics, regional coverage."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.units import unit_label  # noqa: E402
from src.ui import SERIES_KEY, inject_css, latest_prices, render_hero, sidebar_brand, sidebar_footer  # noqa: E402

st.set_page_config(page_title="AgriPulse", page_icon="🌾", layout="wide")
inject_css()
sidebar_brand()


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "production_forecasts.csv", parse_dates=["forecast_date"])


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "price_anomalies.csv", parse_dates=["latest_date"])


@st.cache_data
def load_validation_archive() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation_archive.csv", parse_dates=["date"])


forecasts = load_forecasts()
anomalies = load_anomalies()
validation_archive = load_validation_archive()

render_hero()

# --- This month: three plain-language answers before any dashboard numbers ---
def trusted_current() -> pd.DataFrame:
    """3-month, high/medium-confidence forecasts for fully current series
    (months_stale == 0), so "this month" really means now."""
    current = forecasts
    if "months_stale" in forecasts.columns:
        current = forecasts[forecasts["months_stale"] == 0]
    current = current[current["horizon_months"] == 3]
    return current[current["confidence"].isin(["high", "medium"])]


def series_name(row) -> str:
    return f"{row['commodity']} in {row['market']} {row['pricetype'].lower()}"


def sell_card():
    """Maize/beans series with the biggest expected rise vs. its last real price."""
    trusted = trusted_current()
    staples = trusted[trusted["commodity"].str.contains("maize|beans", case=False)]
    ends = staples.sort_values("forecast_date").groupby(SERIES_KEY).tail(1).merge(latest_prices(), on=SERIES_KEY)
    ends["rise"] = (ends["forecast_price_kes"] / ends["price_per_kg"] - 1) * 100
    best = ends.sort_values("rise").iloc[-1]
    if best["rise"] <= 0:
        return None
    return ("📈 Best time & place to sell", f"+{best['rise']:.0f}%",
            f"{series_name(best)} expected to rise from KES {best['price_per_kg']:.0f} to "
            f"{best['forecast_price_kes']:.0f}/{unit_label(best['commodity'])} by {best['forecast_date']:%b %Y} · {best['confidence']} confidence")


def alerts_card():
    """SPIKE count plus the two biggest recent ones (placeholder product_* codes skipped)."""
    spikes = anomalies[anomalies["status"] == "SPIKE"]
    if spikes.empty:
        return None
    recent = spikes[(spikes["latest_date"] >= spikes["latest_date"].max() - pd.DateOffset(months=3))
                    & ~spikes["commodity"].str.startswith("product_")]
    top = recent.nlargest(2, "deviation_pct")
    detail = "; ".join(f"{series_name(r)} +{r['deviation_pct']:.0f}% above normal" for _, r in top.iterrows())
    return ("🚨 Price alerts", f"{len(spikes)} spikes", f"Biggest: {detail}" if detail else "See the Alerts page")


def buy_card():
    """Cheapest next-month retail maize forecast anywhere in the country."""
    trusted = trusted_current()
    maize = trusted[trusted["commodity"].isin(["Maize", "Maize (white)"]) & (trusted["pricetype"] == "Retail")]
    nxt = maize[maize["forecast_date"] == maize["forecast_date"].min()]
    row = nxt.sort_values("forecast_price_kes").iloc[0]
    where = f"{row['market']} ({row['region']})" if "region" in row and pd.notna(row["region"]) else row["market"]
    return ("🛒 Cheapest maize to buy", f"KES {row['forecast_price_kes']:.0f}/{unit_label(row['commodity'])}",
            f"{where} retail, forecast for {row['forecast_date']:%b %Y}")


cards = []
for build in (sell_card, alerts_card, buy_card):
    try:
        cards.append(build())
    except Exception:  # empty CSV / missing column mid-refresh: hide the card, never the page
        pass
cards = [c for c in cards if c]
if cards:
    st.subheader("This month")
    with st.container(key="this_month"):
        for col, (label, value, note) in zip(st.columns(len(cards)), cards):
            with col.container(border=True):
                st.metric(label, value)
                st.caption(note)

live_series = forecasts
if "months_stale" in forecasts.columns:
    live_series = forecasts[forecasts["months_stale"] <= 6]
live_series_count = len(live_series[["commodity", "market", "pricetype"]].drop_duplicates())

# Median error: prefer the discontinued-series validation archive when it has
# data (accuracy checked against real outcomes); otherwise fall back to the
# deployed series' own measured test MAPE.
if not validation_archive.empty:
    median_forecast_error = validation_archive["error_pct"].median()
    error_source = "measured on discontinued series retained for method validation"
else:
    median_forecast_error = forecasts.drop_duplicates(["commodity", "market", "pricetype"])["test_mape"].median()
    error_source = "measured test error across all currently deployed series"

markets_covered = live_series["market"].nunique()
active_alerts = int((anomalies["status"] != "NORMAL").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Live series", f"{live_series_count:,}")
col2.metric("Median forecast error", f"{median_forecast_error:.1f}%")
col3.metric("Markets covered", f"{markets_covered:,}")
col4.metric("Active price alerts", f"{active_alerts:,}")

st.caption(f"Median forecast error is {error_source} — see the Validation page for details.")

st.divider()

st.markdown(
    """
    AgriPulse forecasts retail and wholesale food prices across Kenya's markets, flags
    unusual price spikes and crashes as they happen, and classifies whether a given price
    point is cheap, average, or expensive relative to local norms. It draws on FEWS NET,
    WFP, KAMIS, KNBS and climate data to serve traders, buyers, policy analysts and
    anyone who needs to know where prices are heading before they get there.
    """
)

st.divider()
st.subheader("Regional coverage")

region_col = "region" if "region" in live_series.columns else None
if region_col:
    coverage = (
        live_series.groupby(region_col)
        .agg(
            markets=("market", "nunique"),
            series=("commodity", "count"),
            median_error=("test_mape", "median"),
        )
        .reset_index()
        .rename(columns={region_col: "Region", "markets": "Markets", "series": "Series", "median_error": "Median error (%)"})
        .sort_values("Series", ascending=False)
    )
    coverage["Median error (%)"] = coverage["Median error (%)"].round(1)
    st.dataframe(coverage, width="stretch", hide_index=True)
else:
    st.caption("Regional breakdown unavailable -- no region column in the forecast export.")

sidebar_footer()
