"""Where to sell: latest wholesale price in every market, net of a transport estimate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.units import PER_HEAD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROAD_FACTOR = 1.3      # road distance is roughly 1.3x the straight line
DEFAULT_RATE = 1.5     # KES per kg per 100 km; an assumption the user tunes on the page
FRESH_MONTHS = 2       # a price older than this is not "where to sell now"


def road_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a)) * ROAD_FACTOR


def market_coords(history: pd.DataFrame) -> pd.DataFrame:
    """market -> lat, lon. Surveyed coordinates where the source has them, else the
    centre of the market's county.

    ponytail: county centres put two markets in one county 0 km apart and can be
    50+ km off in big counties; geocode the market towns if this page gets used.
    """
    centres = json.loads((DATA_DIR / "county_centroids.json").read_text())
    m = history.groupby("market").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                      admin1=("admin1", "first")).reset_index()
    key = m["admin1"].astype(str).str.lower().str.replace("-", " ").str.replace("'", "").str.strip()
    m["approx"] = m["lat"].isna()
    m["lat"] = m["lat"].fillna(key.map(lambda k: centres.get(k, [None, None])[0]))
    m["lon"] = m["lon"].fillna(key.map(lambda k: centres.get(k, [None, None])[1]))
    return m.dropna(subset=["lat", "lon"])[["market", "lat", "lon", "approx"]]


def live_wholesale(history: pd.DataFrame) -> pd.DataFrame:
    """Latest wholesale KES/kg price per (commodity, market), recent ones only."""
    w = history[(history["pricetype"] == "Wholesale") & ~history["commodity"].isin(PER_HEAD)]
    w = w.dropna(subset=["price_per_kg"]).sort_values("date").groupby(["commodity", "market"]).tail(1)
    return w[w["date"] >= history["date"].max() - pd.DateOffset(months=FRESH_MONTHS)]


def options(prices: pd.DataFrame, coords: pd.DataFrame, origin: str, rate: float) -> pd.DataFrame:
    """One commodity's markets ranked by what is left per kg after hauling from `origin`.

    net_gain = price there - price at origin - transport; the origin row is 0 by definition.
    """
    p = prices.merge(coords, on="market")
    o = p[p["market"] == origin].iloc[0]
    p["km"] = [road_km(o["lat"], o["lon"], la, lo) for la, lo in zip(p["lat"], p["lon"])]
    p["transport"] = p["km"] / 100 * rate
    p["net_gain"] = p["price_per_kg"] - o["price_per_kg"] - p["transport"]
    return p.sort_values("net_gain", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    assert 520 < road_km(-1.286, 36.817, -4.043, 39.668) < 600  # Nairobi-Mombasa, ~485 km by road
    prices = pd.DataFrame({"market": ["A", "B", "C"], "price_per_kg": [40.0, 50.0, 41.0]})
    coords = pd.DataFrame({"market": ["A", "B", "C"], "lat": [0.0, 0.0, 0.0], "lon": [36.0, 37.0, 39.0],
                           "approx": False})
    out = options(prices, coords, "A", rate=2.0)
    assert out["market"].tolist() == ["B", "A", "C"], out          # B pays, C does not cover the haul
    assert abs(out["net_gain"][0] - (10 - road_km(0, 36, 0, 37) / 100 * 2)) < 1e-9 and out["net_gain"][1] == 0
    print("sell ok")
