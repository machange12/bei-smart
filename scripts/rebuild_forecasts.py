"""Rebuild production forecasts from the widened chained dataset.

Reads data/cleaned/bei_smart_forecasting_chained_all.csv (WFP+FEWS+KAMIS
stitched by scripts/build_chained_all.py), screens series with the same
gates as the notebook (min-obs, freshness, missing-share, cv), fits
Prophet + SARIMA + Chronos per series, and ships a weighted top-2 ensemble
(by 12-month validation MAPE) for long series and Chronos zero-shot for short
ones -- the test window only scores, never selects. Writes production_forecasts.csv / model_metrics.csv /
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
CHECKPOINT_EVERY = 25

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

ANCHOR = pd.Period("2026-09", "M")

# Chronos pipeline is loaded lazily and once (per process).
_CHRONOS = None
CHRONOS_MODEL = "amazon/chronos-bolt-small"  # overridden by --chronos-model


def _chronos():
    global _CHRONOS
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


def screen(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    settle = df["market"].str.contains(SETTLEMENT_RE)
    print(f"[screen] excluding {df.loc[settle, 'market'].nunique()} settlement "
          f"markets ({int(settle.sum()):,} rows)")
    df = df[~settle]
    rows = []
    for (c, mk, pt), g in df.groupby(["commodity", "market", "pricetype"]):
        g = g.sort_values("date")
        n = len(g)
        if n < min_obs:
            continue
        stale = (ANCHOR - g["date"].max().to_period("M")).n
        if stale > 12:
            continue
        span = (g["date"].max().to_period("M") - g["date"].min().to_period("M")).n + 1
        if (span - n) / span >= 0.25:
            continue
        cv = g["price_per_kg"].std() / g["price_per_kg"].mean() * 100
        if cv >= 60:
            continue
        rows.append({
            "commodity": c, "market": mk, "pricetype": pt,
            "n_records": n, "months_stale": stale, "cv": round(cv, 2),
            "source": g["source"].mode().iloc[0],
            "last_obs": g["date"].max().date().isoformat(),
        })
    return pd.DataFrame(rows)


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
    ci = res.conf_int(alpha=0.05)
    lo = np.asarray(ci)[:, 0] if hasattr(ci, "__array__") else pred - 1.96 * res.se_mean
    hi = np.asarray(ci)[:, 1] if hasattr(ci, "__array__") else pred + 1.96 * res.se_mean
    return pred, lo, hi


def _chronos_predict(y_train, horizon):
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


def _fit_one(c, mk, pt, y, last, cps, models):
    """Run each requested model. Return {name: (val_mape, test_pred_tuple)}
    plus test_y for post-hoc MAPE, plus a forward forecast dict keyed by name.

    Split scheme (matches notebook cell 57):
      - if n >= 84: train / val / test = n-24 / 12 / 12 -> validation-selected
      - if 24 <= n < 84: train / test = n-12 / 12 -> no val window, so ship Chronos
        zero-shot (test only scores, never selects)
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 24:
        return None

    val_scores = {}
    test_preds = {}
    forward = {}
    horizons = [3, 6, 12]

    have_val = n >= 84
    if have_val:
        train_y, val_y, test_y = y[:-24], y[-24:-12], y[-12:]
    else:
        train_y, test_y = y[:-12], y[-12:]
        val_y = None

    # --- Prophet (val + test + forward) ---
    if "prophet" in models:
        try:
            if have_val:
                vp, *_ = _prophet_predict(train_y, 12, cps, pd.Timestamp(last) - pd.DateOffset(months=24))
                val_scores["prophet"] = _mape(val_y, vp)
            tp, *_ = _prophet_predict(y[:-12], 12, cps, pd.Timestamp(last) - pd.DateOffset(months=12))
            test_preds["prophet"] = tp
            fp, flo, fhi = _prophet_predict(y, 12, cps, pd.Timestamp(last))
            forward["prophet"] = (fp, flo, fhi)
        except Exception:
            pass

    # --- SARIMA ---
    if "sarima" in models:
        try:
            if have_val:
                r = _sarima_predict(train_y, 12)
                if r is not None:
                    val_scores["sarima"] = _mape(val_y, r[0])
            r = _sarima_predict(y[:-12], 12)
            if r is not None:
                test_preds["sarima"] = r[0]
            r = _sarima_predict(y, 12)
            if r is not None:
                forward["sarima"] = r
        except Exception:
            pass

    # --- Chronos ---
    if "chronos" in models:
        try:
            if have_val:
                r = _chronos_predict(train_y, 12)
                val_scores["chronos"] = _mape(val_y, r[0])
            r = _chronos_predict(y[:-12], 12)
            test_preds["chronos"] = r[0]
            r = _chronos_predict(y, 12)
            forward["chronos"] = r
        except Exception:
            pass

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
    }


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


def rebuild(min_obs: int, models: list[str], demo: bool = False) -> None:
    df = pd.read_csv(INPUT, parse_dates=["date"])
    print(f"[load] {len(df):,} rows, {df['commodity'].nunique()} commodities, "
          f"{df['market'].nunique()} markets")

    scr = screen(df, min_obs)
    print(f"[screen] {len(scr)} series pass gate (min_obs>={min_obs})")
    print(f"[models] {models}")

    if demo:
        scr = scr.head(3)
        print("[demo] limiting to 3 series")

    admin_map = df.groupby("market")["admin1"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""
    ).to_dict()

    metrics_rows = []
    forecast_rows = []
    per_model_mapes = []  # for comparison table
    done_keys: set[tuple[str, str, str]] = set()

    if not demo and CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, "rb") as f:
                ck = pickle.load(f)
            metrics_rows = ck["metrics_rows"]
            forecast_rows = ck["forecast_rows"]
            per_model_mapes = ck["per_model_mapes"]
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

        res = _fit_one(c, mk, pt, y, last, cps, models)
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
                    "done_keys": list(done_keys),
                }, f)
            tmp.replace(CHECKPOINT)
            print(f"[ckpt] {len(done_keys)} series saved -> {CHECKPOINT.name}", flush=True)

    metrics_df = pd.DataFrame(metrics_rows)
    forecasts_df = pd.DataFrame(forecast_rows)
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

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"[ckpt]  cleared (full run completed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=36)
    ap.add_argument("--models", nargs="+",
                    default=["prophet", "sarima", "chronos"],
                    choices=["prophet", "sarima", "chronos"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--chronos-model", default=CHRONOS_MODEL)
    args = ap.parse_args()
    CHRONOS_MODEL = args.chronos_model
    rebuild(min_obs=args.gate, models=args.models, demo=args.demo)
