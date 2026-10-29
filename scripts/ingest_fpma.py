"""Pull Kenya domestic price series from FAO GIEWS FPMA (open data, CC-BY-4.0).

The FPMA Tool front-end (fpma.fao.org/giews/fpmat4) reads an unauthenticated
Django-REST JSON API; this script calls it directly:

    GET {API}FpmaSerie/?iso3_country_code=KEN&limit=500         -> series metadata
    GET {API}FpmaSeriePrice/?uuid__in=<u1,u2,...>&periodicity=monthly -> datapoints

Unit conversion: every series carries `conversion_factor` = multiplier from its
reporting unit to the API's normalised unit (Kg -> 1, tonne -> 0.001,
500 gms -> 2, 2 Kg -> 0.5). price_per_kg = price_value * conversion_factor.
Series whose factor is null (per-head goats, Liter oil, 500 ml milk) are not
mass-denominated and are dropped. Prices are nominal local currency (KES).

Output (long): data/raw/new_sources/fpma_kenya_long.csv
    source, market, admin_1, commodity, pricetype, unit, currency, date, price
Raw JSON is cached in data/raw/new_sources/_fpma_raw/.

Run:
    python scripts/ingest_fpma.py          # fetch (or reuse cache) + write CSV
    python scripts/ingest_fpma.py --refresh  # drop the cache first (monthly CI run)
    python scripts/ingest_fpma.py --demo   # offline self-check, no network/write
"""
from __future__ import annotations

import argparse
import shutil
import json
import pathlib
import re
import sys
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_chained_all import _canonical, _norm_market  # noqa: E402

API = "https://fpma.fao.org/giews/v4/global/price_module/api/v1/"
RAW = ROOT / "data" / "raw" / "new_sources" / "_fpma_raw"
OUT = ROOT / "data" / "raw" / "new_sources" / "fpma_kenya_long.csv"
COLS = ["source", "market", "admin_1", "commodity", "pricetype", "unit", "currency", "date", "price"]
HDRS = {"User-Agent": "agripulse-research/0.1"}


def _get(url: str, cache: pathlib.Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    r = requests.get(url, headers=HDRS, timeout=120)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    time.sleep(1.0)
    return r.json()


def fetch() -> tuple[list[dict], list[dict]]:
    RAW.mkdir(parents=True, exist_ok=True)
    series = _get(f"{API}FpmaSerie/?iso3_country_code=KEN&limit=500", RAW / "series_KEN.json")["results"]
    prices = []
    uuids = [s["uuid"] for s in series]
    for i in range(0, len(uuids), 10):
        chunk = uuids[i:i + 10]
        url = f"{API}FpmaSeriePrice/?uuid__in={','.join(chunk)}&periodicity=monthly"
        prices += _get(url, RAW / f"prices_KEN_{i:03d}.json")["results"]
    return series, prices


def _admin1(s: str) -> str:
    # "Rift Valley province (1)" -> "Rift Valley"; county names pass through.
    s = re.sub(r"\s*\([^)]*\)", "", s or "")
    return re.sub(r"\s+province$", "", s, flags=re.I).strip() or None


def to_long(series: list[dict], prices: list[dict]) -> pd.DataFrame:
    meta = {s["uuid"]: s for s in series}
    rows = []
    for p in prices:
        s = meta[p["uuid"]]
        com = _canonical(s["commodity_name"])
        if com is None:
            continue
        for d in p["datapoints"]:
            f = d.get("conversion_factor") or s.get("conversion_factor")
            if not f or d.get("price_value") is None:
                continue  # not mass-denominated (per head / litre / ml)
            rows.append({
                "source": "FAO_FPMA",
                "market": _norm_market(s["market_name"]),
                "admin_1": _admin1(s.get("admin_unit")),
                "commodity": com,
                "pricetype": s["price_type"].title(),
                "unit": "KG",
                "currency": s["currency"],
                "date": pd.Timestamp(d["date"]).to_period("M").to_timestamp(),
                "price": float(d["price_value"]) * float(f),
            })
    df = pd.DataFrame(rows, columns=COLS)
    assert (df["currency"] == "KES").all(), df["currency"].unique()
    return (df.groupby(COLS[:-1], as_index=False, dropna=False)["price"].mean()
              .sort_values(["commodity", "market", "pricetype", "date"]))[COLS]


def demo() -> None:
    series = [
        {"uuid": "a", "market_name": "Eldoret", "admin_unit": "Rift Valley province (1)",
         "commodity_name": "Maize", "price_type": "WHOLESALE", "currency": "KES", "conversion_factor": 0.001},
        {"uuid": "b", "market_name": "Wajir", "admin_unit": "Wajir",
         "commodity_name": "Goats (local quality)", "price_type": "RETAIL", "currency": "KES", "conversion_factor": None},
        {"uuid": "c", "market_name": "Nairobi", "admin_unit": "",
         "commodity_name": "Bread", "price_type": "RETAIL", "currency": "KES", "conversion_factor": 2.0},
    ]
    prices = [
        {"uuid": "a", "datapoints": [{"price_value": 53881.0, "conversion_factor": 0.001, "date": "2022-11-01"}]},
        {"uuid": "b", "datapoints": [{"price_value": 6000.0, "conversion_factor": None, "date": "2022-11-01"}]},
        {"uuid": "c", "datapoints": [{"price_value": 60.0, "conversion_factor": 2.0, "date": "2022-11-15"}]},
    ]
    df = to_long(series, prices)
    assert list(df.columns) == COLS
    assert len(df) == 2, df  # goats (per head) dropped
    e = df[df.market == "Eldoret"].iloc[0]
    assert abs(e.price - 53.881) < 1e-9 and e.admin_1 == "Rift Valley" and e.pricetype == "Wholesale"
    b = df[df.commodity == "Bread"].iloc[0]
    assert b.price == 120.0 and b.date == pd.Timestamp("2022-11-01") and pd.isna(b.admin_1)
    print("[demo] OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore cached JSON and refetch")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if args.refresh:
        shutil.rmtree(RAW, ignore_errors=True)
    df = to_long(*fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} rows, {df.market.nunique()} markets, "
          f"{df.groupby(['commodity', 'market', 'pricetype']).ngroups} series")


if __name__ == "__main__":
    main()
