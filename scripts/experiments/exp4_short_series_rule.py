"""Exp 4: honest model rule for short series (n < 84, no validation window).

Compares, on a fixed random sample of short series, test MAPE of: each single
model, old rule (pick best on test -- leaky), equal-weight mean, per-month median.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rebuild_forecasts as rf

OUT = rf.ROOT / "data" / "exports" / "experiments" / "exp4_short_series_rule.csv"
df = pd.read_csv(rf.INPUT, parse_dates=["date"])
scr = rf.screen(df, 36)
scr = scr[scr["n_records"] < 84].sample(n=40, random_state=7)
rows = []
for _, r in scr.iterrows():
    s = df[(df.commodity == r.commodity) & (df.market == r.market) & (df.pricetype == r.pricetype)].sort_values("date")
    y = s["price_per_kg"].dropna().values.astype(float)
    if len(y) < 24:
        continue
    tr, te = y[:-12], y[-12:]
    last = s["date"].max() - pd.DateOffset(months=12)
    preds = {}
    try: preds["prophet"] = np.asarray(rf._prophet_predict(tr, 12, 0.05, last)[0]).ravel()
    except Exception: pass
    try:
        p = rf._sarima_predict(tr, 12)
        if p is not None: preds["sarima"] = np.asarray(p[0]).ravel()
    except Exception: pass
    try: preds["chronos"] = np.asarray(rf._chronos_predict(tr, 12)[0]).ravel()
    except Exception: pass
    if len(preds) < 2:
        continue
    m = {k: rf._mape(te, v) for k, v in preds.items()}
    stack = np.vstack(list(preds.values()))
    rows.append({"commodity": r.commodity, "market": r.market, "pricetype": r.pricetype, "n": len(y),
                 **{f"{k}_mape": v for k, v in m.items()},
                 "old_pick_on_test": min(m.values()),
                 "mean_ens": rf._mape(te, stack.mean(0)),
                 "median_ens": rf._mape(te, np.median(stack, 0))})
    print(len(rows), r.commodity, r.market, {k: round(v, 1) for k, v in rows[-1].items() if k.endswith(("mape", "ens", "test"))}, flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
cols = [c for c in res.columns if c.endswith(("_mape", "_ens", "_test"))]
print("\nMEDIAN MAPE:", res[cols].median().round(2).to_dict())
print("MEAN MAPE:  ", res[cols].mean().round(2).to_dict())
print("pass <=25%: ", (res[cols] <= 25).sum().to_dict(), "of", len(res))
