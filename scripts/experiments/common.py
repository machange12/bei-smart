"""Shared bits for the model experiments: fixed sample + model fns.

Model fns are copied (not imported) from scripts/rebuild_forecasts.py so
parallel edits to that file can't change experiment numbers mid-run.
"""
from __future__ import annotations

import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "exports" / "experiments"
# Frozen copy of data/cleaned/bei_smart_forecasting_chained_all.csv (taken
# 2026-10-29 after the chain rebuild that merged FEWS "Maize (white)" into
# "Maize") so parallel pipeline rebuilds can't shift experiment numbers.
INPUT = OUT / "chain_snapshot.csv"
METRICS = ROOT / "app" / "data" / "model_metrics.csv"
RENAMED = {"Maize (white)": "Maize"}  # metrics name -> current chain name
SAMPLE = OUT / "sample_series.csv"
REFUGEE = r"^(?:Kakuma|Kalobeyei|Dadaab|Daadab|Hagadera|Ifo|Dagahaley)"
FAST_CPS = {"Tomatoes", "Onions", "Kales", "Cabbages", "Spinach", "Carrots",
            "Bananas", "Beans (mixed)", "Beans (dry)", "Sorghum"}
LONG_N = 72  # enough history for 3 rolling val origins (-36/-30/-24) with >=36 train


def build_sample(seed: int = 42) -> pd.DataFrame:
    """40 series: per strategy 7 long (n>=72) + 3 short, commodities round-robin."""
    if SAMPLE.exists():
        return pd.read_csv(SAMPLE)
    m = pd.read_csv(METRICS)
    m = m[m["reliable"] & ~m["market"].str.contains(REFUGEE, case=False)]
    m = m.sample(frac=1, random_state=seed)
    m["long"] = m["n_records"] >= LONG_N
    # rank within (strategy, long, commodity) -> taking low ranks first spreads commodities
    m["rr"] = m.groupby(["strategy", "long", "commodity"]).cumcount()
    picks = []
    for (strat, long_), g in m.groupby(["strategy", "long"]):
        k = 7 if long_ else 3
        picks.append(g.sort_values("rr", kind="stable").head(k))
    s = pd.concat(picks)[["commodity", "market", "pricetype", "region", "n_records",
                          "strategy", "mape", "long"]].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE, index=False)
    return s


def iter_series(sample: pd.DataFrame):
    """Yield (row, dates, y) sequentially; same extraction as production."""
    df = pd.read_csv(INPUT, usecols=["commodity", "market", "pricetype", "date",
                                     "price_per_kg", "admin1"], parse_dates=["date"])
    df = df[df.market.isin(set(sample.market))]
    seen = set()
    for _, r in sample.iterrows():
        s = pd.DataFrame()
        for c in [r.commodity, RENAMED.get(r.commodity)]:
            if s.empty and c and (c, r.market, r.pricetype) not in seen:
                seen.add((c, r.market, r.pricetype))
                s = df[(df.commodity == c) & (df.market == r.market) &
                       (df.pricetype == r.pricetype)].sort_values("date").dropna(subset=["price_per_kg"])
        if len(s) < 36:  # production gate
            print("skip (missing/short/duplicate in chain):", r.commodity, r.market, r.pricetype, len(s))
            continue
        yield r, s, s["price_per_kg"].values.astype(float)


def cps_for(c):
    return 0.15 if c in FAST_CPS else 0.05


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = (y > 0) & np.isfinite(p)
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else np.nan


def prophet_predict(y_train, horizon, cps, last_date):
    from prophet import Prophet
    idx = pd.date_range(end=pd.Timestamp(last_date), periods=len(y_train), freq="MS")
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=cps)
    m.fit(pd.DataFrame({"ds": idx, "y": y_train}))
    return m.predict(m.make_future_dataframe(periods=horizon, freq="MS")).tail(horizon)["yhat"].values


def sarima_predict(y_train, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    for p, q in itertools.product([0, 1], [0, 1]):
        for P, Q in itertools.product([0, 1], [0, 1]):
            try:
                f = SARIMAX(y_train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if np.isfinite(f.aic) and (best is None or f.aic < best[0]):
                    best = (f.aic, f)
            except Exception:
                continue
    if best is None:
        return None
    pred = np.asarray(best[1].get_forecast(steps=horizon).predicted_mean)
    if not np.all(np.isfinite(pred)) or np.max(np.abs(pred)) > 10 * np.max(y_train):
        return None
    return pred


_PIPES = {}


def chronos_predict(y_train, horizon, model_id="amazon/chronos-bolt-small"):
    import torch
    from chronos import BaseChronosPipeline
    if model_id not in _PIPES:
        _PIPES[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32)
    q, _ = _PIPES[model_id].predict_quantiles(
        torch.tensor(np.asarray(y_train, dtype=np.float32)),
        prediction_length=horizon, quantile_levels=[0.5])
    return np.asarray(q[0].numpy())[:, 0]


if __name__ == "__main__":
    s = build_sample()
    print(s.groupby(["strategy", "long"]).size())
    print(s.commodity.value_counts())
    assert len(s) == 40 and not s.market.str.contains(REFUGEE, case=False).any()
