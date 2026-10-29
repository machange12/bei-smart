"""Download World Bank Real-Time Food Prices (RTFP) for Kenya and write it long.

Source: HDX dataset `kenya-real-time-prices` (CC BY 4.0, no login), resolved via
the CKAN API so a changed resource id doesn't break the CI fetch. Same data as
World Bank Microdata Library catalog 6167 (KEN_2021_RTFP_MKT).

RTFP ships one wide row per (market, month). For each commodity `x`:
    x           observed survey price (NaN when not surveyed)
    o_/h_/l_/c_x  ML-modelled open/high/low/close for EVERY month
We emit `price` = observed `x` where present, else `c_x`, and
`estimated` = True where `x` is NaN. Score accuracy on estimated == False only.

Output: data/raw/new_sources/rtfp_kenya_long.csv
    source, market, admin_1, commodity, pricetype, unit, currency, date, price, estimated

Run:
    python scripts/experiments/ingest_rtfp.py                # download + write
    python scripts/experiments/ingest_rtfp.py --no-download  # reuse cached raw CSV
    python scripts/experiments/ingest_rtfp.py --demo         # tiny self-check, no network/write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_chained_all import _canonical, _norm_market  # noqa: E402

RAW = ROOT / "data" / "raw" / "new_sources" / "_rtfp_raw" / "real-time-food-prices-for-kenya.csv"
OUT = ROOT / "data" / "raw" / "new_sources" / "rtfp_kenya_long.csv"
CKAN = "https://data.humdata.org/api/3/action/package_show?id=kenya-real-time-prices"
COLS = ["source", "market", "admin_1", "commodity", "pricetype", "unit",
        "currency", "date", "price", "estimated"]

# RTFP ticker -> (raw label fed to _canonical, pricetype, unit). Labels and units
# from the study metadata `ticker_info` ("Beans (dry)" KG, "Maize (white)" Kg,
# "Livestock (Goat, s local quality)" 1 unit). Goats are per head, NOT per kg:
# kept with unit "Head" so callers can filter; don't mix with KES/kg series.
# pricetype: RTFP doesn't say; observed beans/maize match our chain's WFP/FEWS
# RETAIL series exactly (median ratio 1.000), so "Retail" for all.
TICKERS = {
    "beans": ("Beans (dry)", "Retail", "KG"),
    "maize_fao": ("Maize (white)", "Retail", "KG"),
    "livestock_goat_s_fao": ("Goats (Local Quality)", "Retail", "Head"),
}


# _norm_market drops parentheticals, which would merge these two distinct
# markets into "Wakulima". Fold each onto its city, as KAMIS's
# "Nairobi Wakulima"/"Nakuru Wakulima" already are.
_MARKET_FIX = {"Wakulima (Nairobi)": "Nairobi", "Wakulima (Nakuru)": "Nakuru"}


def download(dest: pathlib.Path = RAW) -> pathlib.Path:
    pkg = requests.get(CKAN, timeout=60).json()["result"]
    url = next(r["url"] for r in pkg["resources"] if "food-prices" in r["url"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide[wide["mkt_name"] != "Market Average"]  # national pseudo-market
    parts = []
    for tick, (label, ptype, unit) in TICKERS.items():
        if "c_" + tick not in wide:
            continue
        obs = pd.to_numeric(wide[tick], errors="coerce") if tick in wide else pd.Series(float("nan"), wide.index)
        est = pd.to_numeric(wide["c_" + tick], errors="coerce")
        parts.append(pd.DataFrame({
            "source": "WB_RTFP",
            "market": wide["mkt_name"].replace(_MARKET_FIX).map(_norm_market),
            "admin_1": wide["adm1_name"],
            "commodity": _canonical(label),
            "pricetype": ptype,
            "unit": unit,
            "currency": wide["currency"],
            "date": pd.to_datetime(wide["DATES"]).dt.to_period("M").dt.to_timestamp(),
            "price": obs.fillna(est),
            "estimated": obs.isna(),
        }))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["price"])
    # Several raw names fold onto one (e.g. "Garissa" + "Garissa town (Garissa)",
    # Kalobeyei villages 1-3). Per month: mean of observed if any, else mean of
    # estimates.
    key = ["market", "commodity", "pricetype", "date"]
    out = out.sort_values("estimated")
    first = out.groupby(key, as_index=False).first()          # estimated=False wins
    price = out[out["estimated"] == out.groupby(key)["estimated"].transform("min")]
    first["price"] = first[key].merge(price.groupby(key, as_index=False)["price"].mean(), on=key)["price"]
    return first.sort_values(["commodity", "market", "date"])[COLS].reset_index(drop=True)


def demo() -> None:
    wide = pd.DataFrame({
        "mkt_name": ["Adele Center", "Adele Center", "Market Average",
                     "Garissa", "Garissa town (Garissa)", "Kitui", "Kitui town (Kitui)"],
        "adm1_name": ["Coast", "Coast", "Market Average",
                      "North Eastern", "North Eastern", "Eastern", "Eastern"],
        "currency": "KES", "DATES": ["2020-01-01", "2020-02-15"] + ["2020-01-01"] * 5,
        "beans": [100.0, None, 90.0, 50.0, 60.0, 70.0, None],
        "c_beans": [101.0, 105.0, 91.0, 51.0, 61.0, 71.0, 99.0],
        "maize_fao": [None] * 7, "c_maize_fao": [40.0, 41.0, 39.0, 1.0, 3.0, 5.0, 5.0],
    })
    out = to_long(wide)
    assert list(out.columns) == COLS
    assert set(out["market"]) == {"Adele", "Garissa", "Kitui"}, out["market"].unique()
    g = out[out["market"] != "Adele"].set_index(["market", "commodity"])
    assert g.loc[("Garissa", "Beans (dry)"), "price"] == 55.0        # mean of two observed
    assert g.loc[("Kitui", "Beans (dry)"), "price"] == 70.0          # observed beats estimate
    assert not g.loc[("Kitui", "Beans (dry)"), "estimated"]
    assert g.loc[("Garissa", "Maize"), "price"] == 2.0 and g.loc[("Garissa", "Maize"), "estimated"]
    b = out[(out["commodity"] == "Beans (dry)") & (out["market"] == "Adele")].set_index("date")
    assert b.loc["2020-01-01", "price"] == 100.0 and not b.loc["2020-01-01", "estimated"]
    assert b.loc["2020-02-01", "price"] == 105.0 and b.loc["2020-02-01", "estimated"]
    assert set(out.loc[out["commodity"] == "Maize", "estimated"]) == {True}
    print("demo ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    raw = RAW if a.no_download and RAW.exists() else download()
    out = to_long(pd.read_csv(raw, low_memory=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out):,} rows, {out['market'].nunique()} markets, "
          f"{out['date'].min():%Y-%m}..{out['date'].max():%Y-%m}, "
          f"estimated share {out['estimated'].mean():.1%}")


if __name__ == "__main__":
    main()
