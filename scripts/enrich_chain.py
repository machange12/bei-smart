"""Append external climate/commodity/FX features + lags to the chained series.

Reads:  data/cleaned/bei_smart_forecasting_chained_all.csv (the widened chain)
        data/external/combined_external.csv               (oni + corn + wheat + fx)
        data/external/epra_fuel.csv                        (optional)

Writes: overwrites the chain in place with new columns appended.

Lags:
  oni_anom          -> 1, 2, 3 + 3-month rolling mean
  corn_usd_bushel   -> 1, 2
  wheat_usd_bushel  -> 1, 2
  kes_per_usd       -> 1
  petrol / diesel   -> 1, 2   (only if epra_fuel.csv is present)

Existing columns are untouched.
"""
from __future__ import annotations
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAIN = ROOT / "data" / "cleaned" / "bei_smart_forecasting_chained_all.csv"
EXTERNAL = ROOT / "data" / "external" / "combined_external.csv"
EPRA = ROOT / "data" / "external" / "epra_fuel.csv"

LAGS = {
    "oni_anom": [1, 2, 3],
    "corn_usd_bushel": [1, 2],
    "wheat_usd_bushel": [1, 2],
    "kes_per_usd": [1],
}
EPRA_LAGS = {"petrol_nairobi_ksh_l": [1, 2], "diesel_nairobi_ksh_l": [1, 2]}


def _build_features(ext: pd.DataFrame) -> pd.DataFrame:
    """External monthly frame -> monthly frame with lag + rolling cols added."""
    ext = ext.sort_values("date").copy()
    # Fill gaps in the monthly external series before computing lags so a
    # missing month doesn't shift a "1-month lag" into a 2-month gap.
    for col in ext.columns:
        if col == "date":
            continue
        ext[col] = ext[col].ffill().bfill()

    for col, lags in LAGS.items():
        if col not in ext.columns:
            continue
        for L in lags:
            ext[f"{col}_lag{L}"] = ext[col].shift(L)

    if "oni_anom" in ext.columns:
        ext["oni_anom_roll3"] = ext["oni_anom"].rolling(3, min_periods=1).mean()
    return ext


def main() -> None:
    df = pd.read_csv(CHAIN, parse_dates=["date"])
    before_rows, before_cols = df.shape
    ext = pd.read_csv(EXTERNAL, parse_dates=["date"])
    # combined_external.csv ships with dup March rows most years -- keep first.
    ext = ext.drop_duplicates(subset="date", keep="first")

    if EPRA.exists():
        epra = pd.read_csv(EPRA, parse_dates=["date"])
        ext = ext.merge(epra, on="date", how="outer").sort_values("date")
        LAGS.update({k: v for k, v in EPRA_LAGS.items() if k in epra.columns})
        print(f"[epra] merged {len(epra)} rows, cols {list(epra.columns)}")
    else:
        print("[epra] not present -- skipping fuel lags")

    ext = _build_features(ext)
    new_cols = [c for c in ext.columns if c != "date"]

    # Drop any pre-existing versions of the new cols so we don't duplicate.
    drop = [c for c in new_cols if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        print(f"[replace] dropping stale {len(drop)} cols: {drop[:6]}{'...' if len(drop)>6 else ''}")

    merged = df.merge(ext, on="date", how="left")
    # Backfill any pre-2015 rows (chain starts 2006, external starts 2015).
    for c in new_cols:
        merged[c] = merged[c].bfill()

    print(f"[shape] rows {before_rows:,} -> {len(merged):,}  cols {before_cols} -> {merged.shape[1]}")
    print(f"[range] {merged.date.min().date()} .. {merged.date.max().date()}")
    print("[null %] per new column:")
    for c in new_cols:
        pct = merged[c].isna().mean() * 100
        print(f"  {c:<28} {pct:5.2f}%")

    merged.to_csv(CHAIN, index=False)
    print(f"[write] {CHAIN}")


if __name__ == "__main__":
    main()
