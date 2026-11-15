"""Exp 8: Chronos-2 vs Chronos-Bolt-small, and do covariates help Chronos-2?

Zero-shot, 12-month test window (last 12 obs), same split as production, every
served series. Arms:
  bolt       amazon/chronos-bolt-small          (what production runs for short series)
  c2         amazon/chronos-2, price history only
  c2_xl      c2 + cross-learning across series that share a test start month
             (grouped by date so no series sees another's future)
  c2_macro   c2 + ONI, corn and wheat futures, KES/USD   (already in the chain, unused)
  c2_ndvi    c2 + county NDVI as % of normal (WFP via HDX; national mean where the county is unknown)
  c2_border  c2 + maize prices in Arusha, Musoma (TZ), Kampala-Owino, Gulu (UG) (WFP via HDX, USD)
  c2_volume  c2 + KAMIS supply volume (only series with volume for 80%+ of months)
  c2_all     macro + ndvi + border
All covariates are past-only: the model never sees their values in the test window.

    python scripts/experiments/exp8_chronos2_covariates.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rebuild_forecasts as rf  # noqa: E402
from common import chronos_predict, mape  # noqa: E402

NEW = rf.ROOT / "data" / "raw" / "new_sources"
OUT = rf.EXPORTS / "experiments"
MACRO = ["oni_anom", "corn_usd_bushel", "wheat_usd_bushel", "kes_per_usd"]
BORDER = {  # name -> (file, market, commodity)
    "tz_arusha": ("wfp_tza.csv", "Arusha (urban)", "Maize"),
    "tz_musoma": ("wfp_tza.csv", "Musoma", "Maize"),
    "ug_owino": ("wfp_uga.csv", "Owino", "Maize flour"),
    "ug_gulu": ("wfp_uga.csv", "Gulu", "Maize (white)"),
}


def _county(s) -> str:
    return str(s).lower().replace("-", " ").replace("'", "").strip()


def load_ndvi() -> tuple[pd.DataFrame, pd.Series]:
    """(month x county) NDVI as % of the long-term average, plus the national mean."""
    d = pd.read_csv(NEW / "_ndvi_raw" / "ken-ndvi-subnat-full.csv", parse_dates=["date"])
    names = pd.read_excel(NEW / "_ndvi_raw" / "ken_admin_boundaries.xlsx", sheet_name="ken_admin1")
    # The file's level 1 is 8 coarse units; level 2 is sub-county, whose code starts with the county code.
    d = d[d.adm_level == 2].assign(adm1_pcode=lambda x: x.PCODE.str[:5]).merge(names[["adm1_pcode", "adm1_name"]])
    d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
    wide = d.assign(county=d.adm1_name.map(_county)).pivot_table(index="month", columns="county", values="viq")
    return wide, wide.mean(axis=1)


def load_border() -> pd.DataFrame:
    cols = {}
    for name, (f, market, commodity) in BORDER.items():
        d = pd.read_csv(NEW / "_fews_border" / f, skiprows=[1], low_memory=False, parse_dates=["date"])
        d = d[(d.market == market) & (d.commodity == commodity)]
        cols[name] = d.groupby(d.date.dt.to_period("M").dt.to_timestamp())["usdprice"].mean()
    return pd.DataFrame(cols)


def at(series: pd.Series, dates: pd.Series) -> np.ndarray:
    """Covariate values on the price series' own dates; gaps carried forward."""
    idx = pd.DatetimeIndex(dates)
    series = series[~series.index.duplicated()].sort_index()
    v = series.reindex(series.index.union(idx)).ffill().bfill().reindex(idx)
    return v.fillna(0.0).to_numpy(dtype=np.float32)


def c2_predict(pipe, tasks: list[dict], **kw) -> list[np.ndarray]:
    q, _ = pipe.predict_quantiles(tasks, prediction_length=12, quantile_levels=[0.5], **kw)
    out = [np.asarray(t).squeeze() for t in q]
    assert all(o.shape == (12,) for o in out), [o.shape for o in out][:3]
    return out


def main() -> None:
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    df = pd.read_csv(rf.INPUT, parse_dates=["date"])
    served = pd.read_csv(rf.APP_DATA / "model_metrics.csv").query("reliable")
    ndvi, ndvi_nat = load_ndvi()
    border = load_border()
    vol_path = NEW / "kamis_volume_monthly.csv"
    vol = pd.read_csv(vol_path, parse_dates=["date"]) if vol_path.exists() else pd.DataFrame(
        columns=["commodity", "market", "date", "volume"])

    S = []  # one dict per series
    for r in served.itertuples():
        s = df[(df.commodity == r.commodity) & (df.market == r.market) & (df.pricetype == r.pricetype)]
        s = s.sort_values("date").dropna(subset=["price_per_kg"])
        if len(s) < 36:
            continue
        ctx = s.iloc[:-12]
        county = _county(s.admin1.dropna().mode().iloc[0]) if s.admin1.notna().any() else ""
        v = vol[(vol.commodity == r.commodity) & (vol.market == r.market)].set_index("date")["volume"]
        cov = {
            "macro": {c: at(s.set_index("date")[c], ctx.date) for c in MACRO},
            "ndvi": {"ndvi": at(ndvi[county] if county in ndvi else ndvi_nat, ctx.date)},
            "border": {c: at(border[c].dropna(), ctx.date) for c in border},
            "volume": {"volume": at(v, ctx.date)} if ctx.date.isin(v.index).mean() >= 0.8 else None,
        }
        S.append({"commodity": r.commodity, "market": r.market, "pricetype": r.pricetype, "n": len(s),
                  "test_start": s.date.iloc[-12], "y": ctx.price_per_kg.to_numpy(np.float32),
                  "test": s.price_per_kg.to_numpy()[-12:], "cov": cov,
                  "ndvi_county": county in ndvi})
    print(f"{len(S)} series; volume for {sum(x['cov']['volume'] is not None for x in S)}; "
          f"own-county NDVI for {sum(x['ndvi_county'] for x in S)}", flush=True)

    def task(x, groups):
        pc = {k: v for g in groups for k, v in x["cov"][g].items()}
        return {"target": x["y"], "past_covariates": pc} if pc else {"target": x["y"]}

    preds = {"bolt": [chronos_predict(x["y"], 12) for x in S]}
    print("bolt done", flush=True)
    arms = {"c2": [], "c2_macro": ["macro"], "c2_ndvi": ["ndvi"], "c2_border": ["border"],
            "c2_all": ["macro", "ndvi", "border"]}
    for arm, groups in arms.items():
        preds[arm] = c2_predict(pipe, [task(x, groups) for x in S])
        print(arm, "done", flush=True)

    preds["c2_xl"] = [None] * len(S)
    by_start: dict = {}
    for i, x in enumerate(S):
        by_start.setdefault(x["test_start"], []).append(i)
    for idx in by_start.values():
        for j in range(0, len(idx), 100):  # ~100 per group, as in the Chronos-2 report
            chunk = idx[j:j + 100]
            out = c2_predict(pipe, [task(S[i], []) for i in chunk], cross_learning=True, batch_size=100)
            for i, p in zip(chunk, out):
                preds["c2_xl"][i] = p
    print("c2_xl done", flush=True)

    has_vol = [i for i, x in enumerate(S) if x["cov"]["volume"] is not None]
    preds["c2_volume"] = [None] * len(S)
    if has_vol:
        for i, p in zip(has_vol, c2_predict(pipe, [task(S[i], ["volume"]) for i in has_vol])):
            preds["c2_volume"][i] = p

    d = pd.DataFrame([{k: x[k] for k in ("commodity", "market", "pricetype", "n")} for x in S])
    for arm, ps in preds.items():
        d[arm] = [round(mape(x["test"], p), 3) if p is not None else np.nan for x, p in zip(S, ps)]
    d.to_csv(OUT / "exp8_chronos2_covariates.csv", index=False)

    from scipy.stats import wilcoxon
    arm_cols = list(preds)

    def report(sub: pd.DataFrame, title: str, base: str = "c2") -> None:
        print(f"\n== {title} ({len(sub)} series) ==")
        for a in arm_cols:
            m = sub[[a, base, "bolt"]].dropna() if a not in (base, "bolt") else sub[[base, "bolt"]].dropna()
            if m.empty:
                continue
            p = wilcoxon(m[a], m[base]).pvalue if a != base and (m[a] != m[base]).any() else np.nan
            print(f"{a:<10} median {m[a].median():6.2f}  mean {m[a].mean():6.2f}  "
                  f"beats {base} on {int((m[a] < m[base]).sum()):3d}/{len(m)}  p={p:.3f}"
                  f"   [bolt on same series: median {m['bolt'].median():.2f}]")

    report(d, "all served series")
    report(d[d.n < 84], "short series (< 84 obs, production = Chronos zero-shot)")
    report(d[d.commodity.str.contains("maize", case=False)], "maize only (border arm)")
    report(d[d.c2_volume.notna()], "series with KAMIS volume")


if __name__ == "__main__":
    main()
