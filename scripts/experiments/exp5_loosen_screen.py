"""Exp 5: what do we lose by serving series the gaps/volatility screen rejects?

Takes fresh series (>=36 obs, <=12 months stale, not a settlement) that fail
the missing-share (<25%) or CV (<60%) gate, scores Chronos zero-shot on the last
12 obs (the production short-series rule), and reports pass rate at the 25% gate.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rebuild_forecasts as rf

OUT = rf.ROOT / "data" / "exports" / "experiments" / "exp5_loosen_screen.csv"
df = pd.read_csv(rf.INPUT, parse_dates=["date"])
df = df[~df["market"].str.contains(rf.SETTLEMENT_RE)]
anchor = df["date"].max().to_period("M")
rows = []
for (c, mk, pt), g in df.groupby(["commodity", "market", "pricetype"]):
    g = g.sort_values("date"); n = len(g)
    if n < 36 or (anchor - g["date"].max().to_period("M")).n > 12:
        continue
    span = (g["date"].max().to_period("M") - g["date"].min().to_period("M")).n + 1
    miss = (span - n) / span * 100
    cv = g["price_per_kg"].std() / g["price_per_kg"].mean() * 100
    if miss < 25 and cv < 60:
        continue  # already passes the screen
    y = g["price_per_kg"].dropna().values.astype(float)
    mape = rf._mape(y[-12:], np.asarray(rf._chronos_predict(y[:-12], 12)[0]).ravel())
    rows.append({"commodity": c, "market": mk, "pricetype": pt, "n": n,
                 "missing_pct": round(miss, 1), "cv": round(cv, 1),
                 "fails": "both" if miss >= 25 and cv >= 60 else ("gaps" if miss >= 25 else "volatility"),
                 "chronos_mape": round(mape, 2)})
res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
res["pass"] = res.chronos_mape <= 25
print(f"rejected fresh series scored: {len(res)} ({res.market.nunique()} markets)")
print(res.groupby("fails").agg(series=("pass", "size"), pass_25=("pass", "sum"),
      median_mape=("chronos_mape", "median"), median_of_passers=("chronos_mape", lambda s: s[s <= 25].median())).round(2))
served = pd.read_csv(rf.ROOT / "app" / "data" / "model_metrics.csv"); served = served[served.reliable]
new = res[res["pass"]]
gain = set(new.market) - set(served.market)
both = pd.concat([served["mape"], new["chronos_mape"]])
print(f"\nwould add {len(new)} series, {len(gain)} NEW markets (served today: {served.market.nunique()})")
print(f"served median MAPE now {served.mape.median():.2f}% -> with additions {both.median():.2f}%")
print("confidence of additions:", pd.cut(new.chronos_mape, [0, 10, 20, 25], labels=["high", "medium", "low"]).value_counts().to_dict())
