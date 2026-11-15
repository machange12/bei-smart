"""Weekly KAMIS prices for the served series, from the scraper's daily cache.

The chain is monthly, so between refreshes the app shows nothing newer than last
month. KAMIS reports daily; this keeps the last WEEKS weeks (median per week) for
every series the app forecasts, so the Forecast page can show "latest week".

Run after scrape_kamis.py (kamis_local_push.ps1 does):
    python scripts/build_weekly.py
Writes app/data/kamis_weekly.csv: commodity, market, pricetype, week (Monday), price, days.
"""
from __future__ import annotations

import re

import pandas as pd

from build_chained_all import _canonical, _norm_market
from scrape_kamis import CACHE, PRODUCTS, ROOT, _parse_price

OUT = ROOT / "app" / "data" / "kamis_weekly.csv"
SERVED = ROOT / "app" / "data" / "production_forecasts.csv"
KEY = ["commodity", "market", "pricetype"]
WEEKS = 8


def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily rows (Date, Market, Wholesale, Retail, commodity) -> weekly medians, KES/kg only."""
    daily = daily.assign(Date=pd.to_datetime(daily["Date"], errors="coerce")).dropna(subset=["Date", "Market"])
    parts = []
    for pt in ("Wholesale", "Retail"):
        parsed = daily[pt].map(_parse_price)
        ok = parsed.map(lambda t: t[0] is not None and (t[1] or "").lower() == "kg")  # not "Bag(90 Kg)"
        d = daily[ok].assign(price=parsed[ok].map(lambda t: t[0]), pricetype=pt)
        parts.append(d)
    d = pd.concat(parts)
    d["week"] = d["Date"].dt.to_period("W-SUN").dt.start_time
    d["market"] = d["Market"].map(_norm_market)
    return (d.groupby(KEY + ["week"])["price"].agg(price="median", days="size")
            .reset_index().round({"price": 2}))


def main() -> None:
    months = sorted({re.search(r"_(\d{4}-\d{2})\.xlsx$", p.name).group(1)
                     for p in CACHE.glob("prod*.xlsx") if p.stat().st_size > 0})[-(WEEKS // 4 + 2):]
    frames = []
    for p in CACHE.glob("prod*.xlsx"):
        pid = int(re.match(r"prod(\d+)_", p.name).group(1))
        if p.stat().st_size == 0 or pid not in PRODUCTS or not any(m in p.name for m in months):
            continue
        frames.append(pd.read_excel(p, usecols=["Date", "Market", "Wholesale", "Retail"])
                      .assign(commodity=_canonical(PRODUCTS[pid])))
    if not frames:
        print("[weekly] no recent cache files; left the old file in place")
        return
    w = weekly(pd.concat(frames).dropna(subset=["commodity"]))
    w = w.merge(pd.read_csv(SERVED, usecols=KEY).drop_duplicates(), on=KEY)  # served series only
    w = w[w["week"] > w["week"].max() - pd.Timedelta(weeks=WEEKS)].sort_values(KEY + ["week"])
    w.to_csv(OUT, index=False)
    print(f"[weekly] {len(w):,} rows, {w.groupby(KEY).ngroups} series, "
          f"weeks {w['week'].min():%Y-%m-%d} to {w['week'].max():%Y-%m-%d} -> {OUT.relative_to(ROOT)}")


def demo() -> None:
    d = pd.DataFrame({
        "Date": ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-12"],  # Mon, Tue, Wed, next Mon
        "Market": ["Kitale"] * 4, "commodity": ["Maize"] * 4,
        "Wholesale": ["40.00/Kg", "44.00/Kg", "3,600.00/Bag(90 Kg)", "50.00/Kg"],  # bag row dropped
        "Retail": ["-"] * 4,
    })
    w = weekly(d)
    assert w["price"].tolist() == [42.0, 50.0] and w["days"].tolist() == [2, 1], w
    assert w["week"].dt.dayofweek.eq(0).all() and set(w["pricetype"]) == {"Wholesale"}
    print("demo ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
