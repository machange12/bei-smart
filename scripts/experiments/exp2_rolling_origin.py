"""Exp 2: rolling-origin (3 val windows) vs single-window model selection.

Val origins at -36/-30/-24 months (12-month horizon each, all before the
test window y[-12:]); the -24 window alone is what production uses.
Raw per-(series, model) scores append to exp2_raw.csv so a kill resumes.
Run `python exp2_rolling_origin.py` (fit), then `... --report`.
"""
import gc
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import (OUT, build_sample, chronos_predict, cps_for, iter_series, mape,
                    prophet_predict, sarima_predict)

RAW = OUT / "exp2_raw.csv"
ORIGINS = [36, 30, 24]  # val window = y[-k:-k+12]; test = y[-12:]
MODELS = ["prophet", "sarima", "chronos"]


def fit(model, y_train, c, last):
    if model == "prophet":
        return prophet_predict(y_train, 12, cps_for(c), last)
    if model == "sarima":
        return sarima_predict(y_train, 12)
    return chronos_predict(y_train, 12)


def run():
    done = set()
    if RAW.exists():
        done = set(map(tuple, pd.read_csv(RAW)[["commodity", "market", "pricetype"]].values))
    sample = build_sample()
    for r, s, y in iter_series(sample[sample["long"]]):
        key = (r.commodity, r.market, r.pricetype)
        if key in done:
            continue
        last = s["date"].max()
        rows = []
        for model in MODELS:
            row = {"commodity": r.commodity, "market": r.market, "pricetype": r.pricetype,
                   "n": len(y), "model": model}
            for k in ORIGINS + [12]:
                t = time.perf_counter()
                try:
                    p = fit(model, y[:-k], r.commodity, pd.Timestamp(last) - pd.DateOffset(months=k))
                except Exception:
                    p = None
                row[f"sec_{k}"] = round(time.perf_counter() - t, 3)
                end = len(y) - k + 12
                row[f"val_{k}" if k != 12 else "test"] = (
                    np.nan if p is None else round(mape(y[-k:end], p), 3))
                if k == 12 and p is not None:
                    row["test_pred"] = " ".join(f"{v:.4f}" for v in p)
            rows.append(row)
            gc.collect()
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
        print(key, {rr["model"]: rr["test"] for rr in rows}, flush=True)


def report():
    raw = pd.read_csv(RAW)
    raw["val_single"] = raw["val_24"]
    raw["val_roll"] = raw[[f"val_{k}" for k in ORIGINS]].mean(axis=1, skipna=False)
    sample = build_sample()
    tests = {(r.commodity, r.market, r.pricetype): y[-12:]
             for r, _, y in iter_series(sample[sample["long"]])}
    out, sel = [], []
    for key, g in raw.groupby(["commodity", "market", "pricetype"]):
        g = g.set_index("model")
        g = g[g["test"].notna()]
        test_y = tests[key]
        preds = {m: np.array(list(map(float, g.at[m, "test_pred"].split()))) for m in g.index}
        rec = {"commodity": key[0], "market": key[1], "pricetype": key[2], "n": int(g["n"].iloc[0]),
               **{f"{m}_test": g.at[m, "test"] for m in g.index},
               "oracle": g["test"].min()}
        for rule in ["single", "roll"]:
            v = g[f"val_{rule}"].dropna()
            if v.empty:
                continue
            w = v.idxmin()
            rec[f"{rule}_pick"], rec[f"{rule}_val"], rec[f"{rule}_test"] = w, v[w], g.at[w, "test"]
            top = v.sort_values().index[:2]
            wt = 1.0 / np.maximum(v[top].values, 1e-3)
            wt /= wt.sum()
            ens = sum(wi * preds[m] for wi, m in zip(wt, top))
            rec[f"{rule}_ens_test"] = mape(test_y, ens)
            if rule == "single":  # production rule incl. its test-window ensemble peek
                rec["prod_rule_test"] = (rec["single_ens_test"]
                                         if g.at[w, "test"] - rec["single_ens_test"] > 2
                                         else g.at[w, "test"])
            for m in v.index:
                sel.append({"rule": rule, "val": v[m], "test": g.at[m, "test"]})
        out.append(rec)
    d = pd.DataFrame(out)
    d.to_csv(OUT / "exp2_rolling_origin.csv", index=False)

    cols = ["prophet_test", "sarima_test", "chronos_test", "single_test", "roll_test",
            "single_ens_test", "roll_ens_test", "prod_rule_test", "oracle"]
    summ = pd.DataFrame({"median_test_mape": d[cols].median(), "mean_test_mape": d[cols].mean()})
    print(f"series: {len(d)}\n", summ.round(2))
    print("roll beats single (strict):", int((d.roll_test < d.single_test).sum()),
          " single beats roll:", int((d.single_test < d.roll_test).sum()),
          " same pick:", int((d.roll_pick == d.single_pick).sum()))
    print("picks single:", d.single_pick.value_counts().to_dict(), " roll:", d.roll_pick.value_counts().to_dict())
    cor = []
    for rule in ["single", "roll"]:
        dd = d[[f"{rule}_val", f"{rule}_test"]].dropna()
        aa = pd.DataFrame(sel).query("rule == @rule").dropna()
        cor.append({"rule": rule,
                    "pearson_chosen": pearsonr(dd.iloc[:, 0], dd.iloc[:, 1])[0],
                    "spearman_chosen": spearmanr(dd.iloc[:, 0], dd.iloc[:, 1])[0],
                    "pearson_all_pairs": pearsonr(aa.val, aa.test)[0],
                    "spearman_all_pairs": spearmanr(aa.val, aa.test)[0]})
    cor = pd.DataFrame(cor)
    print(cor.round(3))
    # runtime: production does val(-24)+test+forward = 3 fits/model; rolling adds 2 more val fits
    sec = raw[[f"sec_{k}" for k in ORIGINS + [12]]].groupby(raw["model"]).median()
    print("median sec per fit:\n", sec.round(2))
    per_series = raw.groupby(["commodity", "market"])[[f"sec_{k}" for k in ORIGINS + [12]]].sum().median()
    one = per_series["sec_24"] + 2 * per_series["sec_12"]  # val + test + forward (~test cost)
    roll = per_series[[f"sec_{k}" for k in ORIGINS]].sum() + 2 * per_series["sec_12"]
    print(f"per-series sec: current {one:.1f}  rolling {roll:.1f}  multiplier {roll / one:.2f}x")
    summ.to_csv(OUT / "exp2_summary.csv")
    cor.to_csv(OUT / "exp2_correlation.csv", index=False)


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
