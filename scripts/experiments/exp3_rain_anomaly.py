"""Exp 3: ERA5 county rainfall ANOMALY (lag 3/6) as a Prophet regressor.

anomaly(t) = rain(t) - mean rain for that calendar month over 1991-2020.
Leak-free design: the 12-month test window is cut into four 3-month blocks;
each block is forecast from an origin at its start, refitting on everything
before it. With horizon h <= 3 <= lag, the regressor value needed at t+h is
rain(t+h-lag), already observed at the origin. ERA5 file ends 2025-12, so
test months beyond 2025-12+lag have no regressor and are not scored; all arms
are scored on the same months (where every arm has a regressor value).
"""
import gc

import numpy as np
import pandas as pd

from common import INPUT, OUT, ROOT, build_sample, cps_for, iter_series

ERA5 = ROOT / "data" / "cleaned" / "era5_rainfall_long.csv"
ARMS = {"plain": [], "a3": ["a3"], "a6": ["a6"], "a3_a6": ["a3", "a6"]}


def norm(x):
    return str(x).lower().replace("-", " ").replace("'", "").strip()


def anomalies():
    e = pd.read_csv(ERA5, parse_dates=["date"])
    clim = e[e.date.dt.year.between(1991, 2020)].groupby(
        ["county", e.date.dt.month])["rainfall_mm"].mean()
    e["anom"] = e.rainfall_mm.values - clim.reindex(
        pd.MultiIndex.from_arrays([e.county, e.date.dt.month])).values
    out = {}
    for c, g in e.groupby("county"):
        a = g.set_index("date")["anom"].sort_index()
        out[norm(c)] = pd.DataFrame({"a3": a.shift(3, freq="MS"), "a6": a.shift(6, freq="MS")})
    return out


def prophet_block(train, future, cps, regs):
    from prophet import Prophet
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=cps)
    for r in regs:
        m.add_regressor(r)
    m.fit(train[["ds", "y"] + regs])
    p = m.predict(future[["ds"] + regs])["yhat"].values
    del m
    return p


def main():
    anom = anomalies()
    sample = build_sample()
    a1 = pd.read_csv(INPUT, usecols=["market", "admin1"]).dropna().groupby("market")["admin1"].agg(
        lambda s: s.mode().iloc[0])
    rows = []
    for r, s, y in iter_series(sample):
        a = anom.get(norm(a1.get(r.market)), anom.get(norm(r.market)))
        if a is None:
            print("no rainfall join:", r.market)
            continue
        d = pd.DataFrame({"ds": s.date.dt.to_period("M").dt.to_timestamp().values, "y": y})
        d = d.drop_duplicates("ds").merge(a, left_on="ds", right_index=True, how="left")
        n = len(d)
        for b in range(4):
            lo, hi = n - 12 + 3 * b, n - 12 + 3 * b + 3
            train, fut = d.iloc[:lo].dropna(), d.iloc[lo:hi]
            origin = fut.ds.iloc[0].to_period("M") - 1
            ok = fut[["a3", "a6"]].notna().all(axis=1)
            if not ok.any():
                continue
            fut = fut[ok]
            for arm, regs in ARMS.items():
                try:
                    p = prophet_block(train, fut, cps_for(r.commodity), regs)
                except Exception as ex:
                    print("fail", r.market, arm, ex)
                    continue
                for ds, yt, pt in zip(fut.ds, fut.y, p):
                    rows.append({"commodity": r.commodity, "market": r.market,
                                 "pricetype": r.pricetype, "arm": arm, "ds": ds,
                                 "h": (ds.to_period("M") - origin).n,
                                 "ape": abs(yt - pt) / yt * 100})
            gc.collect()
        print("done", r.commodity, r.market, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "exp3_raw.csv", index=False)
    # keep only (series, month) cells every arm scored
    w = raw.pivot_table(index=["commodity", "market", "pricetype", "ds"], columns="arm", values="ape").dropna()
    per = w.groupby(level=[0, 1, 2]).mean()
    per.to_csv(OUT / "exp3_rain_anomaly.csv")
    print(f"series {len(per)}, scored months {len(w)}")
    print("median per-series MAPE:\n", per.median().round(2))
    print("mean per-series MAPE:\n", per.mean().round(2))
    for arm in ["a3", "a6", "a3_a6"]:
        print(arm, "beats plain on", int((per[arm] < per["plain"]).sum()), "/", len(per))


if __name__ == "__main__":
    main()
