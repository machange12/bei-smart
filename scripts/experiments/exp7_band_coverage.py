"""Exp 7: do the forecast bands cover 80% of held-out actuals, before and after
rebuild_forecasts.calibrate_bands?

"After" is honest: the shift q is learned on half the series and scored on the
other half (both directions), so no series calibrates its own band.

    python scripts/experiments/exp7_band_coverage.py                 # chronos, every screened series (minutes)
    python scripts/experiments/exp7_band_coverage.py --all-models    # 40-series sample, prophet+sarima+chronos (slow)
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rebuild_forecasts as rf  # noqa: E402

OUT = rf.EXPORTS / "experiments"
ALL_MODELS = "--all-models" in sys.argv
models = ["prophet", "sarima", "chronos"] if ALL_MODELS else ["chronos"]

df = pd.read_csv(rf.INPUT, parse_dates=["date"])
scr = rf.screen(df, 36)
if ALL_MODELS:
    scr = scr.merge(pd.read_csv(OUT / "sample_series.csv")[["commodity", "market", "pricetype"]])

rows = []
for i, r in scr.reset_index(drop=True).iterrows():
    s = df[(df.commodity == r.commodity) & (df.market == r.market) & (df.pricetype == r.pricetype)].sort_values("date")
    res = rf._fit_one(r.commodity, r.market, r.pricetype, s["price_per_kg"].dropna().values,
                      s["date"].max(), 0.15 if r.commodity in {"Tomatoes", "Onions", "Kales", "Cabbages", "Spinach", "Carrots", "Bananas", "Beans (mixed)", "Beans (dry)", "Sorghum"} else 0.05, models)
    if res and res["test_mape"] <= 25:  # served series only
        rows += [{"series": i, "strategy": res["strategy"], "step": k + 1, "score": sc}
                 for k, sc in enumerate(res["cal_scores"])]
    if (i + 1) % 25 == 0:
        print(f"{i + 1}/{len(scr)}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / f"exp7_band_scores_{'sample' if ALL_MODELS else 'chronos'}.csv", index=False)
half = d.series % 2 == 0
after = np.concatenate([d.score[m] <= np.quantile(d.score[~m], rf.BAND_LEVEL) for m in (half, ~half)])
print(f"\nseries {d.series.nunique()}, test points {len(d)}")
print(f"coverage before: {np.mean(d.score <= 0):.1%}   after (held-out half): {after.mean():.1%}   target {rf.BAND_LEVEL:.0%}")
print("before, by strategy:\n", d.groupby("strategy").score.apply(lambda s: round(float(np.mean(s <= 0)), 3)))
print("before, by step:\n", d.groupby("step").score.apply(lambda s: round(float(np.mean(s <= 0)), 3)).to_string())
