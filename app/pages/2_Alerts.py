"""Alerts page: active price alerts (list + map), SMS sign-up, full table, volatility."""

import re
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
KENYA_CENTER = [0.0236, 37.9062]
MAP_COLORS = {"NORMAL": "green", "SPIKE": "red", "CRASH": "blue"}
TOP_N = 10

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src import subscribe  # noqa: E402
from src.sell import market_coords  # noqa: E402
from src.ui import SERIES_KEY, inject_css, sidebar_brand, sidebar_footer, status_badge, t  # noqa: E402
from src.units import unit_label  # noqa: E402

st.set_page_config(page_title="Alerts | AgriPulse", page_icon="🚨", layout="wide")
inject_css()
sidebar_brand()


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "price_anomalies.csv", parse_dates=["latest_date"])


@st.cache_data
def load_volatility() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "market_volatility.csv")


@st.cache_data
def load_markets() -> tuple[dict, pd.DataFrame]:
    """market -> county, and market coordinates (surveyed, geocoded or county centre)."""
    historical = pd.read_csv(DATA_DIR / "bei_smart_forecasting_chained.csv",
                             usecols=["market", "admin1", "latitude", "longitude"])
    regions = (historical.dropna(subset=["admin1"]).drop_duplicates(subset=["market"])
               .set_index("market")["admin1"].to_dict())
    return regions, market_coords(historical)


@st.cache_data
def served_series() -> set:
    """Series the Forecast page offers, so an alert only links where a forecast exists."""
    fc = pd.read_csv(DATA_DIR / "production_forecasts.csv", usecols=SERIES_KEY + ["months_stale"])
    return set(map(tuple, fc[fc["months_stale"] <= 6][SERIES_KEY].drop_duplicates().to_numpy()))


anomalies = load_anomalies().copy()
volatility = load_volatility().copy()
market_region_map, coords = load_markets()
anomalies["region"] = anomalies["market"].map(market_region_map)
volatility["region"] = volatility["region"].replace("—", "Unknown").fillna("Unknown")

st.subheader(t("Price Alerts", "Tahadhari za Bei"))

with st.sidebar:
    st.markdown(t("#### Filters", "#### Chuja"))
    selected_regions = st.multiselect(t("County", "Kaunti"), sorted(anomalies["region"].dropna().unique()))
    selected_commodities = st.multiselect(t("Commodity", "Bidhaa"), sorted(anomalies["commodity"].unique()))

filtered = anomalies
if selected_regions:
    filtered = filtered[filtered["region"].isin(selected_regions)]
if selected_commodities:
    filtered = filtered[filtered["commodity"].isin(selected_commodities)]
active = filtered[filtered["status"] != "NORMAL"].sort_values("deviation_pct", ascending=False)

st.caption(t(f"{len(filtered):,} prices monitored · {len(active):,} unusual right now",
             f"Bei {len(filtered):,} zinafuatiliwa · {len(active):,} si za kawaida sasa"))


def alert_card(row) -> None:
    with st.container(border=True, key=f"alert_{row.name}"):
        head_col, badge_col = st.columns([4, 1])
        head_col.markdown(f"**{row['commodity']}** · {row['market']} ({row['pricetype']})")
        with badge_col:
            status_badge(row["status"])
        unit = unit_label(row["commodity"])
        where = t("above", "juu ya") if row["status"] == "SPIKE" else t("below", "chini ya")
        st.caption(
            f"KES {row['latest_price']:.0f}/{unit} · "
            + t(f"normal range KES {row['normal_low']:.0f}–{row['normal_high']:.0f} · "
                f"{row['deviation_pct']:.0f}% {where} it",
                f"kawaida KES {row['normal_low']:.0f}–{row['normal_high']:.0f} · "
                f"{row['deviation_pct']:.0f}% {where} kawaida")
            + f" · {row['latest_date']:%b %Y}"
        )
        if (row["commodity"], row["market"], row["pricetype"]) in served_series():
            st.page_link("pages/1_Forecast.py", label=t("See the forecast →", "Angalia utabiri →"),
                         query_params={"commodity": row["commodity"], "market": row["market"],
                                       "pricetype": row["pricetype"]})


list_tab, map_tab = st.tabs([t("List", "Orodha"), t("Map", "Ramani")])

with list_tab:
    if active.empty:
        st.info(t("No unusual prices for the selected filters.", "Hakuna bei zisizo za kawaida kwa chaguo hili."))
    for _, row in active.head(TOP_N).iterrows():
        alert_card(row)
    if len(active) > TOP_N:
        with st.expander(t(f"Show {len(active) - TOP_N} more", f"Onyesha {len(active) - TOP_N} zaidi")):
            for _, row in active.iloc[TOP_N:].iterrows():
                alert_card(row)

with map_tab:
    st.markdown(t("🔴 Spike (unusually high) · 🔵 Crash (unusually low) · 🟢 Normal",
                  "🔴 Bei juu isivyo kawaida · 🔵 Bei chini isivyo kawaida · 🟢 Kawaida"))

    def worst(statuses: pd.Series) -> str:
        return next((s for s in ("SPIKE", "CRASH") if (statuses == s).any()), "NORMAL")

    by_market = filtered.groupby("market").agg(
        status=("status", worst),
        flagged=("commodity", lambda s: ", ".join(sorted(set(s[filtered.loc[s.index, "status"] != "NORMAL"])))),
        n=("commodity", "size"),
    ).reset_index().merge(coords, on="market")
    fmap = folium.Map(location=KENYA_CENTER, zoom_start=6)
    # Normal markets first so spikes and crashes are drawn on top.
    for row in by_market.sort_values("status", key=lambda s: s != "NORMAL").to_dict("records"):
        color = MAP_COLORS[row["status"]]
        detail = row["flagged"] or t(f"{row['n']} prices, all normal", f"bei {row['n']}, zote kawaida")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=8 if row["status"] != "NORMAL" else 5,
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(f"<b>{row['market']}</b><br>{row['status']}<br>{detail}", max_width=300),
        ).add_to(fmap)
    st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
    st.caption(t(f"{len(by_market)} of {filtered['market'].nunique()} monitored markets are on the map; "
                 "a few are placed at the centre of their county.",
                 f"Masoko {len(by_market)} kati ya {filtered['market'].nunique()} yako kwenye ramani; "
                 "machache yamewekwa katikati ya kaunti."))

# --- SMS sign-up (hidden until the subscriber store is configured) ---
if subscribe.configured():
    st.divider()
    st.subheader(t("Get alerts by SMS", "Pata tahadhari kwa SMS"))
    st.caption(t("One text a month, only when a price you follow is unusual. Free. Unsubscribe here any time.",
                 "Ujumbe mmoja kwa mwezi, pale tu bei unayofuatilia si ya kawaida. Bure. Unaweza kujiondoa hapa wakati wowote."))
    with st.form("sms_signup", border=True):
        phone = st.text_input(t("Mobile number", "Namba ya simu"), placeholder="0712 345 678")
        c1, c2 = st.columns(2)
        want_commodities = c1.multiselect(t("Commodities (empty = all)", "Bidhaa (wazi = zote)"),
                                          sorted(anomalies["commodity"].unique()))
        want_markets = c2.multiselect(t("Markets (empty = all)", "Masoko (wazi = yote)"),
                                      sorted(anomalies["market"].unique()))
        sms_lang = st.radio(t("Message language", "Lugha ya ujumbe"), ["English", "Kiswahili"], horizontal=True)
        consent = st.checkbox(t("I agree to receive price alerts by SMS on this number.",
                                "Nakubali kupokea tahadhari za bei kwa SMS kwenye namba hii."))
        b1, b2 = st.columns(2)
        join = b1.form_submit_button(t("Subscribe", "Jiunge"), type="primary")
        leave = b2.form_submit_button(t("Unsubscribe this number", "Ondoa namba hii"))
    if join or leave:
        number = subscribe.normalise_phone(phone)
        if number is None:
            st.error(t("Enter a Kenyan mobile number, for example 0712 345 678.",
                       "Weka namba ya simu ya Kenya, kwa mfano 0712 345 678."))
        elif join and not consent:
            st.error(t("Tick the box to agree before subscribing.", "Weka alama kwenye kisanduku kukubali kwanza."))
        else:
            ok = (subscribe.subscribe(number, want_commodities, want_markets, "sw" if sms_lang == "Kiswahili" else "en")
                  if join else subscribe.unsubscribe(number))
            if not ok:
                st.error(t("That did not go through. Please try again later.", "Haikufanikiwa. Tafadhali jaribu tena baadaye."))
            elif join:
                st.success(t("Subscribed. Your first text comes with the next monthly update.",
                             "Umejiunga. Ujumbe wa kwanza utakuja na taarifa ya mwezi ujao."))
            else:
                st.success(t("If that number was subscribed, it has been removed.",
                             "Kama namba hiyo ilikuwa imejiunga, imeondolewa."))

st.divider()
LABELS = {
    "commodity": t("Commodity", "Bidhaa"), "market": t("Market", "Soko"), "pricetype": t("Price type", "Aina ya bei"),
    "region": t("County", "Kaunti"), "latest_date": t("Price as of", "Bei ya tarehe"),
    "latest_price": t("Latest price (KES)", "Bei ya sasa (KES)"), "normal_low": t("Normal low", "Kawaida chini"),
    "normal_high": t("Normal high", "Kawaida juu"), "status": t("Status", "Hali"),
    "deviation_pct": t("% outside normal", "% nje ya kawaida"),
    "current_volatility_pct": t("Volatility now (%)", "Mabadiliko sasa (%)"),
    "median_volatility_pct": t("Usual volatility (%)", "Mabadiliko ya kawaida (%)"),
    "trend": t("Trend", "Mwelekeo"),
}
with st.expander(t(f"All monitored prices ({len(filtered)})", f"Bei zote zinazofuatiliwa ({len(filtered)})")):
    st.dataframe(filtered, width="stretch", hide_index=True, column_config=LABELS)
    scope = "_".join(selected_regions + selected_commodities) or "all"
    st.download_button(
        t("⬇️ Download this table (CSV)", "⬇️ Pakua jedwali (CSV)"),
        filtered.to_csv(index=False),
        file_name=f"agripulse_alerts_{re.sub(r'[^a-z0-9]+', '_', scope.lower()).strip('_')[:80]}.csv",
        mime="text/csv",
    )

with st.expander(t("Market volatility", "Mabadiliko ya bei sokoni")):
    vol_filtered = volatility
    if selected_regions:
        vol_filtered = vol_filtered[vol_filtered["region"].isin(selected_regions)]
    if selected_commodities:
        vol_filtered = vol_filtered[vol_filtered["commodity"].isin(selected_commodities)]
    st.dataframe(vol_filtered.sort_values("current_volatility_pct", ascending=False),
                 width="stretch", hide_index=True, column_config=LABELS)

sidebar_footer()
