"""Exp6: would WB RTFP (observed only) + FAO FPMA rows add servable series?

Read-only w.r.t. production: imports rebuild_forecasts / build_chained_all,
writes only to data/exports/experiments/new_sources/.

  1. augmented chain = existing chain + FPMA (KG) + RTFP (estimated==False, KG),
     new sources at LOWEST priority on (commodity, market, pricetype, date).
  2. candidates = pass rf.screen on augmented but not on existing.
  3. model each with rf._fit_one (production split/selection rule).
  4. goats track (KES/head): FEWS raw 'ea' rows + FPMA goat cache + RTFP observed.

Run:  python scripts/experiments/exp6_new_sources.py [--no-fit]
"""
from __future__ import annotations

import gc
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import rebuild_forecasts as rf  # noqa: E402
from build_chained_all import _norm_market  # noqa: E402

NS = ROOT / "data" / "raw" / "new_sources"
OUT = ROOT / "data" / "exports" / "experiments" / "new_sources"
KEY = ["commodity", "market", "pricetype"]
NEW_SRC = {"FAO_FPMA", "WB_RTFP"}
VOLATILE = {"Tomatoes", "Onions", "Kales", "Cabbages", "Spinach", "Carrots",
            "Bananas", "Beans (mixed)", "Beans (dry)", "Sorghum"}
COLS = KEY + ["date", "price_per_kg", "source", "admin1"]


def stitch(base: pd.DataFrame, extra: list[pd.DataFrame]) -> pd.DataFrame:
    """Existing rows win; among extras, earlier list entry wins."""
    parts = [base.assign(_p=0)] + [e.assign(_p=i + 1) for i, e in enumerate(extra)]
    a = pd.concat(parts, ignore_index=True)
    a = a[a["price_per_kg"].gt(0)].sort_values(KEY + ["date", "_p"])
    return a.drop_duplicates(KEY + ["date"], keep="first").drop(columns="_p").reset_index(drop=True)


def fails(g: pd.DataFrame | None, anchor) -> list[str]:
    """Which rf.screen gates a series fails (reporting only; pass/fail uses rf.screen)."""
    if g is None or g.empty:
        return ["absent"]
    n = len(g)
    p0, p1 = g["date"].min().to_period("M"), g["date"].max().to_period("M")
    span = (p1 - p0).n + 1
    out = []
    if n < 36: out.append("n<36")
    if (anchor - p1).n > 12: out.append("stale>12")
    if (span - n) / span >= 0.25: out.append("missing>=25%")
    if g["price_per_kg"].std() / g["price_per_kg"].mean() * 100 >= 60: out.append("cv>=60")
    return out


def stats(g: pd.DataFrame, anchor) -> dict:
    p0, p1 = g["date"].min().to_period("M"), g["date"].max().to_period("M")
    span = (p1 - p0).n + 1
    return {"n": len(g), "first": str(p0), "last": str(p1), "stale": (anchor - p1).n,
            "missing_pct": round((span - len(g)) / span * 100, 1),
            "cv": round(g["price_per_kg"].std() / g["price_per_kg"].mean() * 100, 1)}


def conf(m):
    return None if not np.isfinite(m) or m > 25 else "high" if m < 10 else "medium" if m <= 20 else "low"


def fit(g: pd.DataFrame, c, mk, pt, do_fit: bool) -> dict:
    g = g.sort_values("date")
    tail = g.tail(12)
    row = {"test_new_src": int(tail["source"].isin(NEW_SRC).sum()),
           "test_orig_src": int((~tail["source"].isin(NEW_SRC)).sum()),
           "rows_new_src": int(g["source"].isin(NEW_SRC).sum())}
    if not do_fit:
        return row
    res = rf._fit_one(c, mk, pt, g["price_per_kg"].values, g["date"].max(),
                      0.15 if c in VOLATILE else 0.05, ["prophet", "sarima", "chronos"])
    m = np.nan if res is None else res["test_mape"]
    row.update(strategy=None if res is None else res["strategy"],
               test_mape=round(m, 2) if np.isfinite(m) else np.nan,
               served=bool(np.isfinite(m) and m <= 25), confidence=conf(m))
    gc.collect()
    return row


def load_rtfp(unit: str) -> pd.DataFrame:
    r = pd.read_csv(NS / "rtfp_kenya_long.csv", parse_dates=["date"])
    r = r[(~r["estimated"]) & (r["unit"] == unit)]
    return r.rename(columns={"admin_1": "admin1", "price": "price_per_kg"})[COLS]


def load_fpma_goats() -> pd.DataFrame:
    meta = {s["uuid"]: s for s in json.loads((NS / "_fpma_raw" / "series_KEN.json")
                                             .read_text(encoding="utf-8"))["results"]}
    rows = []
    for fn in sorted(glob.glob(str(NS / "_fpma_raw" / "prices_KEN_*.json"))):
        for p in json.loads(pathlib.Path(fn).read_text(encoding="utf-8"))["results"]:
            s = meta[p["uuid"]]
            if not s["commodity_name"].lower().startswith("goat"):
                continue
            assert s["measure_unit_label"] == "1 unit" and s["currency"] == "KES"
            for d in p["datapoints"]:
                if d.get("price_value") is not None:
                    rows.append({"commodity": "Goats", "market": _norm_market(s["market_name"]),
                                 "pricetype": s["price_type"].title(),
                                 "date": pd.Timestamp(d["date"]).to_period("M").to_timestamp(),
                                 "price_per_kg": float(d["price_value"]), "source": "FAO_FPMA",
                                 "admin1": s.get("admin_unit") or ""})
    df = pd.DataFrame(rows)
    return df.groupby(KEY + ["date", "source", "admin1"], as_index=False)["price_per_kg"].mean()[COLS]


def load_fews_goats() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "data" / "raw" / "fews_kenya_full.csv", low_memory=False,
                    usecols=["product", "unit", "market", "price_type", "period_date", "value", "admin_1"])
    f = f[f["product"].eq("Goats (Local Quality)") & f["unit"].eq("ea")].copy()
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    f = f.dropna(subset=["value"])
    f["date"] = pd.to_datetime(f["period_date"]).dt.to_period("M").dt.to_timestamp()
    f["market"] = f["market"].map(_norm_market)
    f = f.rename(columns={"price_type": "pricetype", "admin_1": "admin1", "value": "price_per_kg"})
    f["commodity"], f["source"] = "Goats", "FEWS"
    f["admin1"] = f["admin1"].fillna("")  # groupby drops NaN keys (lost all of Kilifi)
    return f.groupby(KEY + ["date", "source", "admin1"], as_index=False)["price_per_kg"].mean()[COLS]


def run_track(base, aug, anchor, do_fit, only_new=True):
    pb = set(map(tuple, rf.screen(base, 36)[KEY].values)) if len(base) else set()
    pa = set(map(tuple, rf.screen(aug, 36)[KEY].values))
    keys = sorted(pa - pb) if only_new else sorted(pa)
    bg = dict(tuple(base.groupby(KEY))) if len(base) else {}
    ag = dict(tuple(aug.groupby(KEY)))
    rows = []
    for i, k in enumerate(keys):
        before = fails(bg.get(k), anchor)
        cls = ("a_new_series" if before == ["absent"]
               else "c_fresher" if before == ["stale>12"]
               else "b_gap_rescued" if set(before) <= {"n<36", "missing>=25%", "stale>12"}
               else "other(" + "+".join(before) + ")")
        g = ag[k]
        rescue = "+".join(sorted(set(g["source"]) & NEW_SRC))
        r = {"commodity": k[0], "market": k[1], "pricetype": k[2], "class": cls,
             "failed_before": "+".join(before), "rescue_source": rescue, **stats(g, anchor),
             **fit(g, *k, do_fit)}
        rows.append(r)
        print(f"[fit] {i + 1}/{len(keys)} {k} {cls} -> {r.get('strategy')} {r.get('test_mape')}", flush=True)
    return pd.DataFrame(rows)


def main(do_fit: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ROOT / "data" / "cleaned" / "bei_smart_forecasting_chained_all.csv",
                       usecols=COLS, parse_dates=["date"])
    fp = pd.read_csv(NS / "fpma_kenya_long.csv", parse_dates=["date"])
    fp = fp[fp["unit"].eq("KG") & fp["market"].ne("National Average")]
    fp = fp.rename(columns={"admin_1": "admin1", "price": "price_per_kg"})[COLS]
    rt = load_rtfp("KG")
    aug = stitch(base, [fp, rt])
    anchor = base["date"].max().to_period("M")
    assert aug["date"].max().to_period("M") == anchor
    added = aug[aug["source"].isin(NEW_SRC)]
    print(f"[aug] +{len(added):,} rows ({added['source'].value_counts().to_dict()}), anchor {anchor}")
    added.groupby(KEY + ["source"]).agg(rows=("date", "size"), first=("date", "min"), last=("date", "max")) \
         .reset_index().to_csv(OUT / "added_rows_by_series.csv", index=False)

    cand = run_track(base, aug, anchor, do_fit)
    served = pd.read_csv(ROOT / "app" / "data" / "model_metrics.csv")
    served = served[served["reliable"]]
    cand["new_market"] = ~cand["market"].isin(set(served["market"]))
    cand["new_commodity"] = ~cand["commodity"].isin(set(served["commodity"]))
    cand["track"] = "per_kg"
    del base, aug; gc.collect()

    # --- goats, KES per head: FEWS raw > FPMA > RTFP; alias RTFP town names to FEWS county names
    rtg = load_rtfp("Head")
    rtg["market"] = rtg["market"].replace({"Lodwar": "Turkana", "Hola": "Tana River"})
    rtg["commodity"] = "Goats"
    fw = load_fews_goats()
    goats = stitch(fw, [load_fpma_goats(), rtg])
    gt = run_track(pd.DataFrame(columns=COLS), goats, anchor, do_fit, only_new=False)
    gt["class"], gt["track"] = "goats_per_head", "per_head"
    gt["new_market"] = ~gt["market"].isin(set(served["market"]))
    gt["new_commodity"] = True
    # all screened goat series, incl. failures, for the report
    ga = [{"market": m, **stats(g, anchor), "fails": "+".join(fails(g, anchor)),
           "sources": "+".join(sorted(g["source"].unique()))}
          for (_, m, _), g in goats.groupby(KEY)]
    pd.DataFrame(ga).to_csv(OUT / "goats_series.csv", index=False)

    out = pd.concat([cand, gt], ignore_index=True)
    out.to_csv(OUT / "candidates.csv", index=False)
    print(out.drop(columns=["first"]).to_string())


def demo() -> None:
    d = pd.to_datetime(["2020-01-01", "2020-02-01"])
    b = pd.DataFrame({"commodity": "X", "market": "M", "pricetype": "Retail", "date": d[:1],
                      "price_per_kg": [1.0], "source": "FEWS", "admin1": ""})
    e = b.assign(price_per_kg=2.0, source="WB_RTFP")
    e = pd.concat([e, e.assign(date=d[1])])
    s = stitch(b, [e])
    assert len(s) == 2 and s.loc[0, "source"] == "FEWS" and s.loc[1, "source"] == "WB_RTFP"
    assert fails(None, pd.Period("2020-02", "M")) == ["absent"]
    print("[demo] OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main(do_fit="--no-fit" not in sys.argv)
