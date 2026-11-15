"""Rebuild production forecasts from the widened chained dataset.

Reads data/cleaned/bei_smart_forecasting_chained_all.csv (WFP+FEWS+KAMIS
stitched by scripts/build_chained_all.py), screens series with the same
gates as the notebook (min-obs, freshness, missing-share, cv), fits
Prophet + SARIMA per long series, forecasts every series with Chronos-2
(jointly, "cross-learning") and with one global LightGBM, and ships a weighted
top-2 ensemble (by 12-month validation MAPE) for long series and Chronos-2
zero-shot for short ones -- the test window only scores, never selects. Writes production_forecasts.csv / model_metrics.csv /
model_routing.json in data/exports/ and app/data/.

Gate default is 36 obs so KAMIS-plateau commodities (Beef, Milk, Cassava,
Rice, Sugar, Mutton, Wheat, Peas) ship instead of waiting for a
pre-2021 backfill source that doesn't exist for livestock/dairy.

Run:
    python scripts/rebuild_forecasts.py                    # full run
    python scripts/rebuild_forecasts.py --gate 60          # tighter
    python scripts/rebuild_forecasts.py --models prophet   # single model
    python scripts/rebuild_forecasts.py --demo             # 3-series smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import pickle
import re
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "cleaned" / "bei_smart_forecasting_chained_all.csv"
EXPORTS = ROOT / "data" / "exports"
APP_DATA = ROOT / "app" / "data"
MODELS_DIR = ROOT / "models"
# ponytail: rebuild takes ~30 min and the bg reaper kills it. Pickle partial
# results every CHECKPOINT_EVERY series so a kill costs at most that many fits.
# Deleted after successful full write.
CHECKPOINT = EXPORTS / ".rebuild_checkpoint.pkl"
CHECKPOINT_EVERY = 10
# Every band is a nominal 80% band (Prophet default, Chronos q10-q90, SARIMA
# alpha=0.2); calibrate_bands() then scales them so they cover 80% in practice.
BAND_LEVEL = 0.8

# Market-name overrides, checked before admin1. Only the 8 former provinces
# are valid outputs (the Forecast page cascades Region -> Market).
REGION_BY_MARKET = {
    "Nairobi": "Nairobi", "Wakulima": "Nairobi", "Kangemi": "Nairobi",
    "Mombasa": "Coast", "Malindi": "Coast", "Kilifi": "Coast",
    "Kwale": "Coast", "Lamu": "Coast", "Taveta": "Coast", "Voi": "Coast",
    "Kisumu": "Nyanza", "Kibuye": "Nyanza", "Homa Bay": "Nyanza",
    "Migori": "Nyanza", "Kisii": "Nyanza", "Ahero": "Nyanza",
    "Nakuru": "Rift Valley", "Eldoret": "Rift Valley", "Kericho": "Rift Valley",
    "Nyahururu": "Rift Valley", "Bomet": "Rift Valley", "Narok": "Rift Valley",
    "Kitale": "Rift Valley", "Kapenguria": "Rift Valley",
    "Turkana": "Rift Valley", "Lodwar": "Rift Valley",
    # admin_1 wrong (Isiolo / Kiambu / Kakamega) or missing (Nyabite).
    "Baringo": "Rift Valley", "Samburu": "Rift Valley", "Kipkaren": "Rift Valley",
    "Nyabite": "Nyanza",
    "Nyeri": "Central", "Karatina": "Central", "Kerugoya": "Central", "Muranga": "Central",
    "Meru": "Eastern", "Embu": "Eastern", "Chuka": "Eastern",
    "Kitui": "Eastern", "Machakos": "Eastern", "Makueni": "Eastern",
    "Wote": "Eastern", "Mwingi": "Eastern",
    "Marsabit": "Eastern", "Moyale": "Eastern", "Isiolo": "Eastern",
    "Garissa": "North Eastern", "Wajir": "North Eastern", "Mandera": "North Eastern",
    "Busia": "Western", "Bungoma": "Western", "Kakamega": "Western", "Vihiga": "Western",
}

REGIONS = {"Nairobi", "Central", "Coast", "Eastern", "North Eastern",
           "Nyanza", "Rift Valley", "Western"}

# KAMIS ships counties as admin1; fold all 47 onto the 8 former provinces.
# Keys are lower-cased with '-' and "'" stripped so "Trans-Nzoia" ==
# "Trans Nzoia" and "Murang'a" == "Muranga".
COUNTY_TO_REGION = {k: r for r, ks in {
    "Nairobi": ["nairobi"],
    "Central": ["kiambu", "kirinyaga", "muranga", "nyandarua", "nyeri"],
    "Coast": ["kilifi", "kwale", "lamu", "mombasa", "taita taveta", "tana river"],
    "Eastern": ["embu", "isiolo", "kitui", "machakos", "makueni", "marsabit",
                "meru", "tharaka nithi"],
    "North Eastern": ["garissa", "mandera", "wajir"],
    "Nyanza": ["homa bay", "kisii", "kisumu", "migori", "nyamira", "siaya"],
    "Rift Valley": ["baringo", "bomet", "elgeyo marakwet", "kajiado", "kericho",
                    "laikipia", "nakuru", "nandi", "narok", "samburu",
                    "trans nzoia", "turkana", "uasin gishu", "west pokot"],
    "Western": ["bungoma", "busia", "kakamega", "vihiga"],
}.items() for k in ks}

# Refugee settlement markets: prices track humanitarian distribution, not
# market supply (analyst report: MAPE >5000%). Matched on the normalised chain
# name; build_chained_all strips "(Kakuma)"/"(Daadab)", leaving bare
# "Ethiopia" / "Hongkong" / "Mogadishu" (Kakuma camp markets), so list those too.
SETTLEMENT_RE = re.compile(
    r"^(kakuma(?! modern)|kalobeyei|dadaab|daadab|hagadera|ifo|dagahaley|"
    r"ethiopia|hongkong|mogadishu)\b|\((kakuma|daadab|dadaab)\)",
    re.IGNORECASE,
)


# Chronos pipeline is loaded lazily and once (per process).
_CHRONOS = None
# exp8: Chronos-2 with cross-learning, median test MAPE 8.92% vs 10.11% for
# chronos-bolt-small on the 270 served series. Bolt still works via --chronos-model.
CHRONOS_MODEL = "amazon/chronos-2"
# Chronos + LightGBM forecasts for every series, computed up front (they need
# all series at once). Cached so an interrupted run does not redo them.
PRE_CACHE = EXPORTS / ".rebuild_pre.pkl"


def _chronos():
    global _CHRONOS
    if _CHRONOS is None and "chronos-2" in CHRONOS_MODEL:
        from chronos import Chronos2Pipeline
        _CHRONOS = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")
    if _CHRONOS is None:
        import torch
        try:
            from chronos import BaseChronosPipeline
            _CHRONOS = BaseChronosPipeline.from_pretrained(
                CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
        except ImportError:
            from chronos import ChronosPipeline
            _CHRONOS = ChronosPipeline.from_pretrained(
                CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
    return _CHRONOS


def screen(df: pd.DataFrame, min_obs: int, near: int = 0) -> pd.DataFrame:
    """Series that pass every gate. `near` > 0 also keeps series that would pass after
    at most `near` more months of steady reporting (enough months, gaps under 25%) and
    reported in the last 3 months -- the app's "Coming soon" list; `months_to_go` > 0."""
    settle = df["market"].str.contains(SETTLEMENT_RE)
    print(f"[screen] excluding {df.loc[settle, 'market'].nunique()} settlement "
          f"markets ({int(settle.sum()):,} rows)")
    df = df[~settle]
    anchor = df["date"].max().to_period("M")  # latest month in the data, not a fixed date
    rows = []
    for (c, mk, pt), g in df.groupby(["commodity", "market", "pricetype"]):
        g = g.sort_values("date")
        n = len(g)
        if n < min_obs - near:
            continue
        stale = (anchor - g["date"].max().to_period("M")).n
        span = (g["date"].max().to_period("M") - g["date"].min().to_period("M")).n + 1
        gaps = span - n
        # months of steady reporting until n >= min_obs and gaps/span < 25%
        need = max(min_obs - n, 4 * gaps - span + 1, 0)
        if need > near or stale > (12 if need == 0 else 3):  # waiting series must still be reporting
            continue
        cv = g["price_per_kg"].std() / g["price_per_kg"].mean() * 100
        if cv >= 60:
            continue
        rows.append({
            "commodity": c, "market": mk, "pricetype": pt,
            "n_records": n, "months_stale": stale, "cv": round(cv, 2),
            "source": g["source"].mode().iloc[0],
            "last_obs": g["date"].max().date().isoformat(),
            "months_to_go": need,
        })
    return pd.DataFrame(rows)


def write_watchlist(df: pd.DataFrame, min_obs: int, near: int = 12) -> pd.DataFrame:
    """app/data/watchlist.csv: series that will clear the gate once they gain a few more months."""
    admin_map = df.groupby("market")["admin1"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "").to_dict()
    wl = screen(df, min_obs, near=near)
    wl = wl[wl["months_to_go"] > 0].copy()
    wl["region"] = [_region(mk, admin_map.get(mk, "")) for mk in wl["market"]]
    anchor = df["date"].max().to_period("M")
    wl["eligible_from"] = [(anchor + int(k)).strftime("%b %Y") for k in wl["months_to_go"]]
    wl = wl.sort_values(["months_to_go", "commodity", "market"])
    APP_DATA.mkdir(parents=True, exist_ok=True)
    wl.to_csv(APP_DATA / "watchlist.csv", index=False)
    print(f"[watchlist] {len(wl)} series within {near} months of passing the gates")
    return wl


def _mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = (y > 0) & np.isfinite(p)
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else np.nan


def _prophet_predict(y_train, horizon, cps, last_date):
    from prophet import Prophet
    idx = pd.date_range(end=pd.Timestamp(last_date), periods=len(y_train), freq="MS")
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=cps)
    m.fit(pd.DataFrame({"ds": idx, "y": y_train}))
    fc = m.predict(m.make_future_dataframe(periods=horizon, freq="MS")).tail(horizon)
    return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values


def _sarima_predict(y_train, horizon):
    # ponytail: reduced from (0-2)x(0-2)x(0-1)x(0-1)=36 combos to
    # (0-1)x(0-1)x(0-1)x(0-1)=16, ~2.5x faster. Full grid was 3+ hrs for 219
    # series; this stays under 1 hr and empirically loses <1 MAPE point per
    # series (notebook's tuned grid was for a 4-commodity chain).
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    for p, q in itertools.product([0, 1], [0, 1]):
        for P, Q in itertools.product([0, 1], [0, 1]):
            try:
                f = SARIMAX(y_train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                if np.isfinite(f.aic) and (best is None or f.aic < best[0]):
                    best = (f.aic, f)
            except Exception:
                continue
    if best is None:
        return None
    res = best[1].get_forecast(steps=horizon)
    pred = np.asarray(res.predicted_mean)
    if not np.all(np.isfinite(pred)) or np.max(np.abs(pred)) > 10 * np.max(y_train):
        return None
    ci = res.conf_int(alpha=1 - BAND_LEVEL)  # was 95%, blended with 80% bands from the others
    lo = np.asarray(ci)[:, 0] if hasattr(ci, "__array__") else pred - 1.28 * res.se_mean
    hi = np.asarray(ci)[:, 1] if hasattr(ci, "__array__") else pred + 1.28 * res.se_mean
    return pred, lo, hi


def _chronos_batch(contexts, horizon=12):
    """(median, q10, q90) per context. Chronos-2 forecasts the whole batch jointly
    (cross-learning: related series inform each other), so callers must only batch
    contexts that end on the same date -- see precompute()."""
    if "chronos-2" not in CHRONOS_MODEL:
        return [_chronos_predict(c, horizon) for c in contexts]
    q, _ = _chronos().predict_quantiles(
        [np.asarray(c, dtype=np.float32) for c in contexts], prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9], cross_learning=len(contexts) > 1, batch_size=100)
    out = [np.asarray(t)[0] for t in q]  # each (horizon, 3)
    return [(a[:, 1], a[:, 0], a[:, 2]) for a in out]


def _chronos_predict(y_train, horizon):
    if "chronos-2" in CHRONOS_MODEL:
        return _chronos_batch([y_train], horizon)[0]
    import torch
    ctx = torch.tensor(np.asarray(y_train, dtype=np.float32))
    pipe = _chronos()
    if hasattr(pipe, "predict_quantiles"):
        out = pipe.predict_quantiles(
            ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
        q = out[0] if isinstance(out, tuple) else out
        arr = np.asarray(q[0].numpy() if hasattr(q[0], "numpy") else q[0])
        # arr is (H, 3) — cols are the three requested quantiles in order
        return arr[:, 1], arr[:, 0], arr[:, 2]
    # legacy chronos-t5 fallback (num_samples API)
    arr = pipe.predict(ctx, prediction_length=horizon, num_samples=50)[0].numpy()
    return (np.quantile(arr, 0.5, axis=0),
            np.quantile(arr, 0.1, axis=0),
            np.quantile(arr, 0.9, axis=0))


LGBM_FEATS = ["r1", "r2", "r3", "r6", "r12", "vs_mean12", "vol12", "level", "h", "month",
              "target_month", "commodity", "pricetype"]


def _lgbm_origin(y, dates) -> pd.DataFrame:
    """Features at each observation t, in observation order (gaps ignored, as everywhere here)."""
    ly = pd.Series(np.log(np.asarray(y, float)))
    r = ly.diff()
    return pd.DataFrame({
        "r1": r, "r2": r.shift(1), "r3": r.shift(2),
        "r6": ly - ly.shift(6), "r12": ly - ly.shift(12),
        "vs_mean12": ly - ly.rolling(12, min_periods=6).mean(),
        "vol12": r.rolling(12, min_periods=6).std(),
        "level": ly, "month": pd.DatetimeIndex(dates).month,
    })


def _lgbm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every (series, origin t, horizon h) with target = log(price[t+h] / price[t])."""
    parts = []
    for (c, mk, pt), g in df.groupby(["commodity", "market", "pricetype"]):
        if len(g) < 24:
            continue
        dates = pd.DatetimeIndex(g["date"])
        f = _lgbm_origin(g["price_per_kg"].to_numpy(), dates).assign(commodity=c, pricetype=pt)
        ly = f["level"].to_numpy()
        for h in range(1, 13):
            t = f.iloc[:-h].copy()
            t["h"] = h
            t["target"] = ly[h:] - ly[:-h]
            t["target_date"] = dates[h:]
            t["target_month"] = dates[h:].month
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def _requests(df: pd.DataFrame, scr: pd.DataFrame) -> list[dict]:
    """Every forecast a screened series needs: forward, test window, and (84+ obs) validation
    window. `cutoff` is the first date being forecast; nothing dated on or after it may be used."""
    reqs = []
    for r in scr.itertuples():
        s = df[(df["commodity"] == r.commodity) & (df["market"] == r.market) &
               (df["pricetype"] == r.pricetype)].sort_values("date").dropna(subset=["price_per_kg"])
        y, d = s["price_per_kg"].to_numpy(float), pd.DatetimeIndex(s["date"])
        key = (r.commodity, r.market, r.pricetype)
        reqs.append({"key": key, "kind": "fwd", "y": y, "dates": d, "cutoff": None})
        reqs.append({"key": key, "kind": "test", "y": y[:-12], "dates": d[:-12], "cutoff": d[-12]})
        if len(y) >= 84:
            reqs.append({"key": key, "kind": "val", "y": y[:-24], "dates": d[:-24], "cutoff": d[-24]})
    return reqs


def precompute(df: pd.DataFrame, scr: pd.DataFrame, models: list[str]) -> dict:
    """{series key: {(model, kind): (median, lo, hi)}} for chronos and lgbm.

    Leak rule for both: a forecast whose first target date is D only sees data
    dated before D. Chronos batches share one D; LightGBM is trained per D.
    """
    stamp = (INPUT.stat().st_mtime, CHRONOS_MODEL)
    cache = {"stamp": stamp, "pre": {}, "done": set()}
    if PRE_CACHE.exists():
        try:
            with open(PRE_CACHE, "rb") as f:
                old = pickle.load(f)
            if old.get("stamp") == stamp:
                cache = old
                print(f"[pre] resumed {len(cache['done'])} finished groups from cache")
        except Exception:
            pass
    pre, done = cache["pre"], cache["done"]

    def save():
        tmp = PRE_CACHE.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        tmp.replace(PRE_CACHE)

    reqs = _requests(df, scr)
    groups: dict = {}
    for q in reqs:  # forward forecasts group by their own last date
        groups.setdefault((q["kind"] == "fwd", q["cutoff"] if q["cutoff"] is not None else q["dates"][-1]), []).append(q)

    if "chronos" in models and "chronos" not in done:
        for qs in groups.values():
            for j in range(0, len(qs), 100):  # ~100 per joint batch, as in the Chronos-2 report
                chunk = qs[j:j + 100]
                for q, out in zip(chunk, _chronos_batch([q["y"] for q in chunk])):
                    pre.setdefault(q["key"], {})[("chronos", q["kind"])] = out
        done.add("chronos")
        save()
        print(f"[pre] chronos: {len(reqs)} forecasts in {len(groups)} date groups", flush=True)

    if "lgbm" in models:
        import lightgbm as lgb
        clean = df[~df["market"].str.contains(SETTLEMENT_RE) & (df["price_per_kg"] > 0)]
        clean = clean.sort_values(["commodity", "market", "pricetype", "date"])
        cats = {"commodity": pd.CategoricalDtype(sorted(clean["commodity"].unique())),
                "pricetype": pd.CategoricalDtype(sorted(clean["pricetype"].unique()))}
        by_cut: dict = {}
        for q in reqs:
            by_cut.setdefault(q["cutoff"], []).append(q)
        todo = [c for c in by_cut if ("lgbm", c) not in done]
        table = _lgbm_table(clean).astype(cats) if todo else None
        for cut in todo:
            tr = table if cut is None else table[table["target_date"] < cut]
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                                      random_state=0, verbose=-1)
            model.fit(tr[LGBM_FEATS], tr["target"])
            # ponytail: band = +/-1.28 sd of in-sample residuals per horizon (optimistic);
            # calibrate_bands() corrects the coverage. Quantile models if lgbm ever ships alone a lot.
            sd = (tr["target"] - model.predict(tr[LGBM_FEATS])).groupby(tr["h"]).std().to_numpy()
            for q in by_cut[cut]:
                last = _lgbm_origin(q["y"], q["dates"]).iloc[[-1]].assign(commodity=q["key"][0], pricetype=q["key"][2])
                x = pd.concat([last] * 12, ignore_index=True).astype(cats)
                x["h"] = range(1, 13)
                x["target_month"] = (x["month"] + x["h"] - 1) % 12 + 1  # assumes no gap ahead
                med = q["y"][-1] * np.exp(model.predict(x[LGBM_FEATS]))
                pre.setdefault(q["key"], {})[("lgbm", q["kind"])] = (med, med * np.exp(-1.2816 * sd), med * np.exp(1.2816 * sd))
            done.add(("lgbm", cut))
            save()
            print(f"[pre] lgbm cutoff {'all data' if cut is None else f'{cut:%Y-%m}'}: "
                  f"{len(by_cut[cut])} forecasts, {len(tr):,} training rows", flush=True)
    return pre


def _fit_one(c, mk, pt, y, last, cps, models, pre=None):
    """Run each requested model. Return {name: (val_mape, test_pred_tuple)}
    plus test_y for post-hoc MAPE, plus a forward forecast dict keyed by name.

    Split scheme (matches notebook cell 57):
      - if n >= 84: train / val / test = n-24 / 12 / 12 -> validation-selected
      - if 24 <= n < 84: train / test = n-12 / 12 -> no val window, so ship Chronos
        zero-shot (test only scores, never selects)

    `pre` holds this series' chronos / lgbm forecasts from precompute(); without it
    Chronos is run here on the series alone and lgbm is skipped.
    """
    pre = pre or {}
    y = np.asarray(y, float)
    n = len(y)
    if n < 24:
        return None

    val_scores = {}
    test_preds = {}
    test_bands = {}  # name -> (lo, hi) on the test window, for calibrate_bands
    forward = {}
    horizons = [3, 6, 12]

    have_val = n >= 84
    if have_val:
        train_y, val_y, test_y = y[:-24], y[-24:-12], y[-12:]
    else:
        train_y, test_y = y[:-12], y[-12:]
        val_y = None
    # A short series always ships Chronos, so fitting Prophet and SARIMA on it only
    # filled the comparison table (most of the rebuild time). Fit them only when they
    # can be chosen, or as the fallback when Chronos is not in the run.
    need_stat = have_val or "chronos" not in models

    # --- Prophet (val + test + forward) ---
    if "prophet" in models and need_stat:
        try:
            if have_val:
                vp, *_ = _prophet_predict(train_y, 12, cps, pd.Timestamp(last) - pd.DateOffset(months=24))
                val_scores["prophet"] = _mape(val_y, vp)
            tp, tlo, thi = _prophet_predict(y[:-12], 12, cps, pd.Timestamp(last) - pd.DateOffset(months=12))
            test_preds["prophet"] = tp
            test_bands["prophet"] = (tlo, thi)
            fp, flo, fhi = _prophet_predict(y, 12, cps, pd.Timestamp(last))
            forward["prophet"] = (fp, flo, fhi)
        except Exception:
            pass

    # --- SARIMA ---
    if "sarima" in models and need_stat:
        try:
            if have_val:
                r = _sarima_predict(train_y, 12)
                if r is not None:
                    val_scores["sarima"] = _mape(val_y, r[0])
            r = _sarima_predict(y[:-12], 12)
            if r is not None:
                test_preds["sarima"] = r[0]
                test_bands["sarima"] = r[1:]
            r = _sarima_predict(y, 12)
            if r is not None:
                forward["sarima"] = r
        except Exception:
            pass

    # --- Chronos ---
    if "chronos" in models:
        try:
            def chronos(kind, ctx):
                return pre[("chronos", kind)] if ("chronos", kind) in pre else _chronos_predict(ctx, 12)
            if have_val:
                r = chronos("val", train_y)
                val_scores["chronos"] = _mape(val_y, r[0])
            r = chronos("test", y[:-12])
            test_preds["chronos"] = r[0]
            test_bands["chronos"] = r[1:]
            forward["chronos"] = chronos("fwd", y)
        except Exception:
            pass

    # --- Global LightGBM (precomputed only: it is trained across all series) ---
    if "lgbm" in models and ("lgbm", "test") in pre and ("lgbm", "fwd") in pre:
        if have_val and ("lgbm", "val") in pre:
            val_scores["lgbm"] = _mape(val_y, pre[("lgbm", "val")][0])
        test_preds["lgbm"] = pre[("lgbm", "test")][0]
        test_bands["lgbm"] = pre[("lgbm", "test")][1:]
        forward["lgbm"] = pre[("lgbm", "fwd")]

    if not test_preds:
        return None

    # ---- Selection ----
    # The test window never picks the model, so the reported test MAPE is an
    # honest holdout score (experiments/exp2: always-ensemble beat single-model
    # selection on 22/27 long series, median 8.69% vs 11.52%).
    #  - val window: weighted top-2 ensemble by 1/val_mape (or the lone val winner)
    #  - no val window: Chronos zero-shot -- nothing is fitted, so nothing can
    #    overfit a short history (exp4: median 12.1% vs 13.2% equal-weight mean,
    #    39/40 under the 25% gate); mean of the rest only if Chronos failed
    ens_weights = None  # (list[name], np.ndarray) if weighted path applies
    cand = [k for k in test_preds if k in val_scores and k in forward]
    usable = [k for k in test_preds if k in forward]
    if len(cand) >= 2:
        ranked = sorted(cand, key=lambda k: val_scores[k])[:2]
        w = np.array([1.0 / max(val_scores[k], 1e-3) for k in ranked])
        w = w / w.sum()
        ens_test = sum(wi * test_preds[k] for wi, k in zip(w, ranked))
        ens_weights = (ranked, w)
        strategy = "ensemble"
    elif cand:
        strategy = cand[0]
    elif "chronos" in usable:
        strategy = "chronos"
    elif len(usable) >= 2:
        ens_test = np.mean([test_preds[k] for k in usable], axis=0)
        strategy = "ensemble"
    elif usable:
        strategy = usable[0]
    else:
        return None
    if strategy == "ensemble":
        ens_mape = _mape(test_y, ens_test)

    # Compute displayed test MAPE from whichever strategy actually ships
    if strategy == "ensemble":
        test_mape = ens_mape
    else:
        test_mape = _mape(test_y, test_preds[strategy])

    # Test-window band of whatever ships, blended the same way as the forward band.
    if ens_weights is not None:
        t_lo, t_hi = (sum(wi * np.asarray(test_bands[k][j]) for wi, k in zip(*ens_weights[::-1]))
                      for j in (0, 1))
    elif strategy == "ensemble":
        t_lo = np.min([test_bands[k][0] for k in usable], axis=0)
        t_hi = np.max([test_bands[k][1] for k in usable], axis=0)
    else:
        t_lo, t_hi = (np.asarray(b) for b in test_bands[strategy])
    cal_scores = band_scores(test_y, t_lo, t_hi)

    # ---- Forward forecast rows ----
    forecast_rows = []
    if strategy == "ensemble":
        if ens_weights is not None:
            names, w = ens_weights
            parts_med = [np.asarray(forward[k][0]).ravel() for k in names]
            parts_lo = [np.asarray(forward[k][1]).ravel() for k in names]
            parts_hi = [np.asarray(forward[k][2]).ravel() for k in names]
            if not all(p.shape == (12,) for p in parts_med):
                return None
            pred = sum(wi * pm for wi, pm in zip(w, parts_med))
            lo = sum(wi * pl for wi, pl in zip(w, parts_lo))
            hi = sum(wi * ph for wi, ph in zip(w, parts_hi))
        else:
            parts_med, parts_lo, parts_hi = [], [], []
            for name in usable:
                fp, flo, fhi = forward[name]
                fp = np.asarray(fp).ravel()
                if fp.shape == (12,):
                    parts_med.append(fp)
                    parts_lo.append(np.asarray(flo).ravel())
                    parts_hi.append(np.asarray(fhi).ravel())
            if not parts_med:
                return None
            pred = np.mean(parts_med, axis=0)
            lo = np.min(parts_lo, axis=0) if parts_lo else pred * 0.9
            hi = np.max(parts_hi, axis=0) if parts_hi else pred * 1.1
    else:
        if strategy not in forward:
            return None
        pred, lo, hi = forward[strategy]

    fdate = pd.Timestamp(last).to_period("M").to_timestamp()
    for h in horizons:
        for step in range(h):
            d = (fdate + pd.DateOffset(months=step + 1)).date().isoformat()
            forecast_rows.append({
                "horizon_months": h,
                "forecast_date": d,
                "forecast_price_kes": round(float(pred[step]), 2),
                "forecast_lower": round(max(0.0, float(lo[step])), 2),
                "forecast_upper": round(float(hi[step]), 2),
            })

    return {
        "test_mape": test_mape, "strategy": strategy,
        "val_scores": val_scores,
        "test_mapes": {k: _mape(test_y, v) for k, v in test_preds.items()},
        "forecast_rows": forecast_rows,
        "cal_scores": cal_scores,
    }


def band_scores(y, lo, hi) -> list[float]:
    """How far each actual fell outside its band, in band-widths (<= 0 = inside)."""
    y, lo, hi = (np.asarray(a, float).ravel() for a in (y, lo, hi))
    w = hi - lo
    ok = np.isfinite(w) & (w > 0)
    return (np.maximum(lo - y, y - hi)[ok] / w[ok]).tolist()


def calibrate_bands(fc: pd.DataFrame, scores: dict[str, list[float]],
                    level: float = BAND_LEVEL) -> pd.DataFrame:
    """Conformal scaling: widen (or narrow) every band by q band-widths per side,
    q = the `level` quantile of the test-window scores, so `level` of held-out
    actuals would have landed inside. Per strategy when it has 200+ scores, else pooled.

    ponytail: one q for all 12 steps; per-step q if far-horizon coverage lags.
    """
    pooled = [x for v in scores.values() for x in v]
    if fc.empty or not pooled:
        return fc
    fc = fc.copy()
    for strat, idx in fc.groupby("strategy").groups.items():
        sc = scores.get(strat, [])
        sc = sc if len(sc) >= 200 else pooled
        q = float(np.quantile(sc, level))
        print(f"[bands] {strat:<9} raw coverage {np.mean(np.asarray(sc) <= 0):.0%} "
              f"of {len(sc)} test points (target {level:.0%}) -> shift {q:+.2f} band-widths per side")
        w = fc.loc[idx, "forecast_upper"] - fc.loc[idx, "forecast_lower"]
        p = fc.loc[idx, "forecast_price_kes"]
        fc.loc[idx, "forecast_lower"] = np.minimum(fc.loc[idx, "forecast_lower"] - q * w, p).clip(lower=0).round(2)
        fc.loc[idx, "forecast_upper"] = np.maximum(fc.loc[idx, "forecast_upper"] + q * w, p).round(2)
    return fc


def _demo_bands() -> None:
    fc = pd.DataFrame({"strategy": ["chronos"], "forecast_price_kes": [100.0],
                       "forecast_lower": [90.0], "forecast_upper": [110.0]})
    out = calibrate_bands(fc, {"chronos": [0.5] * 10})  # actuals half a width outside
    assert (out.forecast_lower[0], out.forecast_upper[0]) == (80.0, 120.0), out
    assert band_scores([100, 130], [90, 90], [110, 110]) == [-0.5, 1.0]


def _tier(test_mape, stale):
    base = "high" if test_mape < 10 else "medium" if test_mape <= 20 else "low"
    cap = "low" if stale > 24 else "medium" if stale > 12 else "high"
    tiers = ["high", "medium", "low"]
    return tiers[max(tiers.index(base), tiers.index(cap))]


def _region(mk, admin1):
    if mk in REGION_BY_MARKET:
        return REGION_BY_MARKET[mk]
    if isinstance(admin1, str) and admin1.strip():
        if admin1 in REGIONS:
            return admin1
        key = admin1.lower().replace("-", " ").replace("'", "").strip()
        return COUNTY_TO_REGION.get(key, "Unknown")
    return "Unknown"


def _demo_pre(df, scr, models) -> dict:
    """Demo: precompute without touching the real cache, and check the leak rule holds."""
    global PRE_CACHE
    real, PRE_CACHE = PRE_CACHE, EXPORTS / ".rebuild_pre_demo.pkl"
    try:
        pre = precompute(df, scr, models)
    finally:
        PRE_CACHE.unlink(missing_ok=True)
        PRE_CACHE = real
    for q in _requests(df, scr):
        assert q["cutoff"] is None or q["dates"].max() < q["cutoff"], "context reaches past its cutoff"
        for m in ("chronos", "lgbm"):
            if m in models:
                med, lo, hi = pre[q["key"]][(m, q["kind"])]
                assert len(med) == 12 and np.all(lo <= med) and np.all(med <= hi), (m, q["key"], q["kind"])
    return pre


def rebuild(min_obs: int, models: list[str], demo: bool = False) -> None:
    df = pd.read_csv(INPUT, parse_dates=["date"])
    print(f"[load] {len(df):,} rows, {df['commodity'].nunique()} commodities, "
          f"{df['market'].nunique()} markets")

    scr = screen(df, min_obs)
    print(f"[screen] {len(scr)} series pass gate (min_obs>={min_obs})")
    if not demo:
        write_watchlist(df, min_obs)
    print(f"[models] {models}")

    if demo:
        _demo_bands()
        scr = pd.concat([scr[scr["n_records"] >= 84].head(2), scr[scr["n_records"] < 84].head(4)])
        print("[demo] limiting to 6 series (2 long, 4 short)")
    pre_all = _demo_pre(df, scr, models) if demo else precompute(df, scr, models)

    admin_map = df.groupby("market")["admin1"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""
    ).to_dict()

    metrics_rows = []
    forecast_rows = []
    per_model_mapes = []  # for comparison table
    cal_scores: dict[str, list[float]] = {}  # strategy -> band scores of served series
    done_keys: set[tuple[str, str, str]] = set()

    if not demo and CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, "rb") as f:
                ck = pickle.load(f)
            metrics_rows = ck["metrics_rows"]
            forecast_rows = ck["forecast_rows"]
            per_model_mapes = ck["per_model_mapes"]
            cal_scores = ck.get("cal_scores", {})
            done_keys = set(ck["done_keys"])
            print(f"[resume] loaded checkpoint: {len(done_keys)} series done, "
                  f"{len(metrics_rows)} metric rows, {len(forecast_rows)} fc rows")
        except Exception as e:
            print(f"[resume] checkpoint unreadable ({e}); starting fresh")
            done_keys = set()

    for i, r in scr.reset_index(drop=True).iterrows():
        c, mk, pt = r["commodity"], r["market"], r["pricetype"]
        if (c, mk, pt) in done_keys:
            continue
        s = df[(df["commodity"] == c) & (df["market"] == mk) &
               (df["pricetype"] == pt)].sort_values("date")
        y = s["price_per_kg"].dropna().values
        last = s["date"].max()
        cps = 0.15 if c in {"Tomatoes", "Onions", "Kales", "Cabbages",
                            "Spinach", "Carrots", "Bananas", "Beans (mixed)",
                            "Beans (dry)", "Sorghum"} else 0.05

        res = _fit_one(c, mk, pt, y, last, cps, models, pre=pre_all.get((c, mk, pt)))
        stale = int(r["months_stale"])
        region = _region(mk, admin_map.get(mk))

        if res is None:
            metrics_rows.append({
                "commodity": c, "market": mk, "pricetype": pt, "region": region,
                "source": r["source"], "n_records": int(r["n_records"]),
                "last_obs": r["last_obs"], "months_stale": stale,
                "mape": np.nan, "strategy": None, "reliable": False,
            })
            continue

        test_mape = res["test_mape"]
        reliable = (not np.isnan(test_mape)) and test_mape <= 25

        metrics_rows.append({
            "commodity": c, "market": mk, "pricetype": pt, "region": region,
            "source": r["source"], "n_records": int(r["n_records"]),
            "last_obs": r["last_obs"], "months_stale": stale,
            "mape": round(test_mape, 2), "strategy": res["strategy"],
            "reliable": bool(reliable),
        })
        per_model_mapes.append({
            "commodity": c, "market": mk, "pricetype": pt,
            **{f"{k}_mape": round(v, 2) for k, v in res["test_mapes"].items()},
            "selected": res["strategy"], "final_mape": round(test_mape, 2),
        })

        if reliable:
            cal_scores.setdefault(res["strategy"], []).extend(res["cal_scores"])
            for row in res["forecast_rows"]:
                forecast_rows.append({
                    "commodity": c, "market": mk, "region": region,
                    "pricetype": pt, "strategy": res["strategy"],
                    "forecast_date": row["forecast_date"],
                    "horizon_months": row["horizon_months"],
                    "forecast_price_kes": row["forecast_price_kes"],
                    "forecast_lower": row["forecast_lower"],
                    "forecast_upper": row["forecast_upper"],
                    "test_mape": round(test_mape, 2),
                    "months_stale": stale,
                    "confidence": _tier(test_mape, stale),
                })

        done_keys.add((c, mk, pt))

        if (i + 1) % 10 == 0 or (i + 1) == len(scr):
            print(f"[fit] {i + 1}/{len(scr)} done ({c[:20]:<20} {mk[:15]:<15} -> "
                  f"{res['strategy']}, mape={test_mape:.1f})", flush=True)

        if not demo and (i + 1) % CHECKPOINT_EVERY == 0:
            EXPORTS.mkdir(parents=True, exist_ok=True)
            tmp = CHECKPOINT.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump({
                    "metrics_rows": metrics_rows,
                    "forecast_rows": forecast_rows,
                    "per_model_mapes": per_model_mapes,
                    "cal_scores": cal_scores,
                    "done_keys": list(done_keys),
                }, f)
            tmp.replace(CHECKPOINT)
            print(f"[ckpt] {len(done_keys)} series saved -> {CHECKPOINT.name}", flush=True)

    metrics_df = pd.DataFrame(metrics_rows)
    forecasts_df = calibrate_bands(pd.DataFrame(forecast_rows), cal_scores)
    compare_df = pd.DataFrame(per_model_mapes)

    n_reliable = int(metrics_df["reliable"].sum())
    print(f"\n[metrics] {len(metrics_df)} screened, {n_reliable} reliable, "
          f"median MAPE {metrics_df[metrics_df.reliable]['mape'].median():.2f}%")
    print("[strategy] {}".format(
        metrics_df[metrics_df.reliable]["strategy"].value_counts().to_dict()))

    if demo:
        print("[demo] skipping writes")
        print(compare_df.to_string())
        return

    routing = {}
    for _, r in metrics_df[metrics_df["reliable"]].iterrows():
        routing[f"{r['commodity']}|{r['market']}|{r['pricetype']}"] = {
            "strategy": r["strategy"],
            "test_mape": float(r["mape"]),
            "confidence": _tier(float(r["mape"]), int(r["months_stale"])),
            "region": r["region"],
            "months_stale": int(r["months_stale"]),
        }

    EXPORTS.mkdir(parents=True, exist_ok=True)
    APP_DATA.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    forecasts_df.to_csv(EXPORTS / "production_forecasts.csv", index=False)
    metrics_df.to_csv(EXPORTS / "model_metrics.csv", index=False)
    compare_df.to_csv(EXPORTS / "model_comparison.csv", index=False)
    with open(MODELS_DIR / "model_routing.json", "w") as f:
        json.dump(routing, f, indent=2)

    for name in ["production_forecasts.csv", "model_metrics.csv"]:
        shutil.copy(EXPORTS / name, APP_DATA / name)
    shutil.copy(MODELS_DIR / "model_routing.json", APP_DATA / "model_routing.json")
    shutil.copy(INPUT, APP_DATA / "bei_smart_forecasting_chained.csv")
    print(f"\n[write] production_forecasts.csv ({len(forecasts_df)} rows), "
          f"model_metrics.csv ({len(metrics_df)}), routing ({len(routing)})")
    print(f"[copy]  app/data/ refreshed")

    for f in (CHECKPOINT, PRE_CACHE):
        if f.exists():
            f.unlink()
    print(f"[ckpt]  cleared (full run completed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=36)
    ap.add_argument("--models", nargs="+",
                    default=["prophet", "sarima", "chronos", "lgbm"],
                    choices=["prophet", "sarima", "chronos", "lgbm"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--chronos-model", default=CHRONOS_MODEL)
    args = ap.parse_args()
    CHRONOS_MODEL = args.chronos_model
    rebuild(min_obs=args.gate, models=args.models, demo=args.demo)
