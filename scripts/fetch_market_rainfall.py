"""
Pull real market-level monthly rainfall from NASA POWER's public API, using each
market's actual latitude/longitude instead of the admin1-region average currently
baked into bei_smart_forecasting_v3.csv's rainfall_mm column.

NASA POWER's monthly endpoint needs no API key and returns PRECTOTCORR in mm/day
(monthly average daily rate) -- this script converts that to total monthly mm by
multiplying by the number of days in each month, matching the convention already
used elsewhere in this project.

Output: data/raw/rainfall_by_market_nasapower.csv
Columns: market, admin1, latitude, longitude, date, rainfall_mm_market
"""
import time
import calendar
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data" / "cleaned" / "bei_smart_forecasting_v3.csv"
OUT_DIR = REPO_ROOT / "data" / "raw"
OUT_FILE = OUT_DIR / "rainfall_by_market_nasapower.csv"
CACHE_DIR = OUT_DIR / "_rainfall_cache"  # one JSON per market, so reruns skip completed ones

START_YEAR = 2006
END_YEAR = 2025  # NASA POWER's data currently ends 2025-12-31
API_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_market(lat: float, lon: float) -> dict:
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["properties"]["parameter"]["PRECTOTCORR"]


def to_monthly_rows(market: str, admin1: str, lat: float, lon: float, monthly: dict) -> list[dict]:
    rows = []
    for key, mm_per_day in monthly.items():
        if key.endswith("13"):  # annual average, not a real month
            continue
        year, month = int(key[:4]), int(key[4:6])
        if mm_per_day < 0:  # NASA POWER uses -999 for missing
            continue
        n_days = calendar.monthrange(year, month)[1]
        rows.append({
            "market": market,
            "admin1": admin1,
            "latitude": lat,
            "longitude": lon,
            "date": pd.Timestamp(year=year, month=month, day=1),
            "rainfall_mm_market": round(mm_per_day * n_days, 2),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE)
    markets = (df[["market", "admin1", "latitude", "longitude"]]
               .drop_duplicates(subset=["market"])
               .dropna(subset=["latitude", "longitude"])
               .reset_index(drop=True))
    print(f"{len(markets)} markets with coordinates to fetch")

    all_rows = []
    for i, row in markets.iterrows():
        cache_file = CACHE_DIR / f"{row['market'].replace('/', '_')}.json"
        if cache_file.exists():
            monthly = pd.read_json(cache_file, typ="series").to_dict()
        else:
            try:
                monthly = fetch_market(row["latitude"], row["longitude"])
            except Exception as e:
                print(f"  [{i+1}/{len(markets)}] {row['market']}: FAILED ({e})")
                continue
            pd.Series(monthly).to_json(cache_file)
            time.sleep(0.3)  # be polite to the API

        all_rows.extend(to_monthly_rows(row["market"], row["admin1"],
                                         row["latitude"], row["longitude"], monthly))
        if (i + 1) % 20 == 0 or i == len(markets) - 1:
            print(f"  [{i+1}/{len(markets)}] {row['market']} done")

    result = pd.DataFrame(all_rows).sort_values(["market", "date"])
    result.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(result):,} rows across {result['market'].nunique()} markets to {OUT_FILE}")


if __name__ == "__main__":
    main()
