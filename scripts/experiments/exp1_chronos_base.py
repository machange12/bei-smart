"""Exp 1: chronos-bolt-small vs chronos-bolt-base, zero-shot, 12-month test."""
import sys
import time

import pandas as pd

from common import METRICS, OUT, REFUGEE, build_sample, chronos_predict, iter_series, mape

# --all: chronos is cheap (<0.5s/series), so also run every non-refugee reliable series
ALL = "--all" in sys.argv
if ALL:
    m = pd.read_csv(METRICS)
    sample = m[m["reliable"] & ~m["market"].str.contains(REFUGEE, case=False)]
else:
    sample = build_sample()

MODELS = {"small": "amazon/chronos-bolt-small", "base": "amazon/chronos-bolt-base"}
for mid in MODELS.values():  # warm-load so load time isn't billed to series 1
    chronos_predict([1.0] * 24, 12, mid)

rows = []
for r, s, y in iter_series(sample):
    row = {"commodity": r.commodity, "market": r.market, "pricetype": r.pricetype, "n": len(y)}
    for name, mid in MODELS.items():
        t = time.perf_counter()
        p = chronos_predict(y[:-12], 12, mid)
        row[f"{name}_sec"] = round(time.perf_counter() - t, 3)
        row[f"{name}_mape"] = round(mape(y[-12:], p), 3)
    rows.append(row)
    print(row, flush=True) if not ALL else None

d = pd.DataFrame(rows)
d.to_csv(OUT / ("exp1_chronos_base_vs_small" + ("_all" if ALL else "") + ".csv"), index=False)
print(d[["small_mape", "base_mape", "small_sec", "base_sec"]].median())
print("base wins:", int((d.base_mape < d.small_mape).sum()), "/", len(d))
from scipy.stats import wilcoxon
print("mean", d.small_mape.mean(), d.base_mape.mean(), "wilcoxon p", wilcoxon(d.small_mape, d.base_mape).pvalue)
