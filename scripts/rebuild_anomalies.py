"""Rebuild price_anomalies.csv from the widened chained dataset.

Ports the IQR-band rule from notebooks/modeling/03_forecasting.ipynb so
the anomaly file re-runs whenever the chain is rebuilt (the old file lagged
the widening, leaving 211/297 map markers grey).

Rule per (commodity, market, pricetype):
  - need n >= 24 monthly obs and staleness <= 12 months
  - build IQR band from the 36 months before the latest (at least +/-10% of median): lower = max(Q1 - 1.5*IQR, 0), upper = Q3 + 1.5*IQR
  - skip flat / zero-variance series
  - status: SPIKE if latest > upper, CRASH if < lower, else NORMAL

Column contract matches app/pages/3_Map.py's expectations exactly.
"""
from __future__ import annotations
import pathlib
import shutil
import pandas as pd

from rebuild_forecasts import SETTLEMENT_RE

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "cleaned" / "bei_smart_forecasting_chained_all.csv"
EXPORTS = ROOT / "data" / "exports"
APP_DATA = ROOT / "app" / "data"

# Band window: over 10-20 years of trending prices the IQR spans 0-235 KES and
# flags nothing; 3 years tracks the current price level.
BAND_MONTHS = 36
MIN_BAND = 0.10


def build(df: pd.DataFrame) -> pd.DataFrame:
    # Settlement prices follow aid distribution, so an IQR band flags noise.
    df = df[~df["market"].str.contains(SETTLEMENT_RE)]
    anchor = df["date"].max().to_period("M")  # latest month in the data, not a fixed date
    rows = []
    for (c, mk, pt), g in df.groupby(["commodity", "market", "pricetype"]):
        g = g.sort_values("date")
        if len(g) < 24:
            continue
        stale = (anchor - g["date"].max().to_period("M")).n
        if stale > 12:
            continue

        hist = g.iloc[:-1]["price_per_kg"].tail(BAND_MONTHS)
        q1, q3 = hist.quantile(0.25), hist.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0 or hist.std() == 0:
            continue

        # Flat KAMIS series give a band a few shillings wide, so a 5% wobble
        # reads as a CRASH. Every band spans at least +/-10% of the median.
        med = hist.median()
        lower = max(min(q1 - 1.5 * iqr, med * (1 - MIN_BAND)), 0.0)
        upper = max(q3 + 1.5 * iqr, med * (1 + MIN_BAND))
        latest = g.iloc[-1]
        price = float(latest["price_per_kg"])

        if price > upper:
            status, dev = "SPIKE", round((price - upper) / upper * 100, 1)
        elif price < lower and lower > 0:
            status, dev = "CRASH", round((lower - price) / lower * 100, 1)
        else:
            status, dev = "NORMAL", 0.0

        rows.append({
            "commodity": c, "market": mk, "pricetype": pt,
            "latest_date": latest["date"].date().isoformat(),
            "latest_price": round(price, 2),
            "normal_low": round(lower, 2),
            "normal_high": round(upper, 2),
            "status": status, "deviation_pct": dev,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(INPUT, parse_dates=["date"])
    out = build(df)
    print(f"[anomalies] {len(out)} rows, {out['market'].nunique()} markets, "
          f"status: {out['status'].value_counts().to_dict()}")

    EXPORTS.mkdir(parents=True, exist_ok=True)
    APP_DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(EXPORTS / "price_anomalies.csv", index=False)
    shutil.copy(EXPORTS / "price_anomalies.csv", APP_DATA / "price_anomalies.csv")
    print(f"[write] app/data/price_anomalies.csv")

    # smoke: check overlap with mapped markets
    hist = df.dropna(subset=["latitude", "longitude"])
    mapped = set(hist["market"].unique())
    hit = mapped & set(out["market"].unique())
    print(f"[smoke] {len(hit)}/{len(mapped)} mapped markets will colour")


if __name__ == "__main__":
    main()
