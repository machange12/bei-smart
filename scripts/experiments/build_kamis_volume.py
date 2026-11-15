"""Monthly KAMIS supply volume per (commodity, market), from the scraper's xlsx cache.

KAMIS reports a daily "Supply Volume" next to each price; the price scraper drops
it. Output (mean daily volume per month, same commodity/market names as the chain):
    data/raw/new_sources/kamis_volume_monthly.csv
"""
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from build_chained_all import _canonical, _norm_market  # noqa: E402
from scrape_kamis import CACHE, PRODUCTS  # noqa: E402

OUT = CACHE.parent / "new_sources" / "kamis_volume_monthly.csv"


def one(path: pathlib.Path) -> pd.DataFrame | None:
    if path.stat().st_size == 0:
        return None
    pid = int(re.match(r"prod(\d+)_", path.name).group(1))
    d = pd.read_excel(path, usecols=["Market", "Supply Volume", "Date"])
    d = d[d["Supply Volume"] > 0]
    if d.empty or pid not in PRODUCTS:
        return None
    d["date"] = pd.to_datetime(d["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    d["commodity"] = _canonical(PRODUCTS[pid])
    d["market"] = d["Market"].map(_norm_market)
    return d.dropna(subset=["date", "commodity"])[["commodity", "market", "date", "Supply Volume"]]


if __name__ == "__main__":
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, sorted(CACHE.glob("prod*.xlsx"))) if p is not None]
    v = (pd.concat(parts).groupby(["commodity", "market", "date"])["Supply Volume"]
         .agg(volume="mean", days="size").reset_index())
    v.to_csv(OUT, index=False)
    print(f"{len(v):,} rows, {v.groupby(['commodity', 'market']).ngroups} series, "
          f"{v.date.min():%Y-%m} to {v.date.max():%Y-%m} -> {OUT}")
