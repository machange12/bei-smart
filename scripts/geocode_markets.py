"""Coordinates for active markets the price sources give no location for (mostly KAMIS).

Looks each market up on OpenStreetMap (Nominatim, 1 request/second) as
"<market>, <county>, Kenya" and keeps the hit only if it lands within MAX_KM of
the county's centre; otherwise the market is placed at the county centre.
Already-resolved markets are kept, so re-runs only look up new ones.

    python scripts/geocode_markets.py
Writes app/data/market_coords.json: {market: [lat, lon, "geocoded" | "county"]}.
Hand-edit that file to correct a market; this script will not overwrite it.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAIN = ROOT / "app" / "data" / "bei_smart_forecasting_chained.csv"
CENTRES = ROOT / "app" / "data" / "county_centroids.json"
OUT = ROOT / "app" / "data" / "market_coords.json"
MAX_KM = 120          # a hit further than this from the county centre is the wrong place
ACTIVE_MONTHS = 12


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def county_key(s) -> str:
    return str(s).lower().replace("-", " ").replace("'", "").strip()


def place_name(market: str) -> str:
    """'Kitale Municipality' -> 'Kitale', 'Kiminini - Kitale' -> 'Kiminini'."""
    name = re.split(r"\s+-\s+|\(", market)[0]
    return re.sub(r"\b(municipality|market|town|open air)\b", "", name, flags=re.I).strip()


def lookup(market: str, county: str) -> tuple[float, float] | None:
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": f"{place_name(market)}, {county}, Kenya", "format": "json",
                             "limit": 1, "countrycodes": "ke"},
                     headers={"User-Agent": "market-price-dashboard-geocoder/1.0"}, timeout=30)
    r.raise_for_status()
    hits = r.json()
    return (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None


def main() -> None:
    h = pd.read_csv(CHAIN, usecols=["market", "admin1", "latitude", "date"], parse_dates=["date"])
    m = h.groupby("market").agg(lat=("latitude", "first"), last=("date", "max"),
                                county=("admin1", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""))
    need = m[m["lat"].isna() & (m["last"] >= h["date"].max() - pd.DateOffset(months=ACTIVE_MONTHS))]
    centres = json.loads(CENTRES.read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for market, row in need.iterrows():
        centre = centres.get(county_key(row["county"]))
        if market in out or centre is None:
            continue
        try:
            hit = lookup(market, row["county"])
        except requests.RequestException as e:
            print(f"[retry later] {market}: {type(e).__name__}")
            continue
        time.sleep(1.1)  # Nominatim usage policy
        if hit and km(*hit, *centre) <= MAX_KM:
            out[market] = [round(hit[0], 4), round(hit[1], 4), "geocoded"]
        else:
            out[market] = [centre[0], centre[1], "county"]
        print(f"{market:<28} {row['county']:<16} {out[market][2]:<9}"
              f"{'' if not hit else f'{km(*hit, *centre):5.0f} km from county centre'}")
    OUT.write_text(json.dumps(out, indent=0, sort_keys=True))
    kinds = pd.Series([v[2] for v in out.values()]).value_counts().to_dict()
    print(f"\n{len(out)} markets -> {OUT.relative_to(ROOT)}: {kinds}")


if __name__ == "__main__":
    assert place_name("Kitale Municipality") == "Kitale" and place_name("Kiminini - Kitale") == "Kiminini"
    assert 430 < km(-1.286, 36.817, -4.043, 39.668) < 450  # Nairobi-Mombasa straight line
    main()
