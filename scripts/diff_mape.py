"""Diff old vs new model_metrics on the shared (commodity, market, pricetype) set.

Old = git HEAD's app/data/model_metrics.csv (Prophet+SARIMA+Chronos ensemble,
5 commodities, 104 series).
New = current app/data/model_metrics.csv (rebuilt pipeline over widened chain).

Reports per-series delta, plus how many old series survived screening and
how much accuracy moved on the pre-existing set.
"""
from __future__ import annotations
import pathlib
import subprocess
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _old() -> pd.DataFrame:
    txt = subprocess.check_output(
        ["git", "show", "HEAD:app/data/model_metrics.csv"], cwd=ROOT, text=True
    )
    from io import StringIO
    return pd.read_csv(StringIO(txt))


def main() -> None:
    old = _old()
    new = pd.read_csv(ROOT / "app" / "data" / "model_metrics.csv")

    key = ["commodity", "market", "pricetype"]
    old_r = old[old["reliable"]].copy()
    new_r = new[new["reliable"]].copy()

    m = old_r.merge(new_r, on=key, suffixes=("_old", "_new"))
    m["mape_delta"] = m["mape_new"] - m["mape_old"]

    print(f"OLD deployed: {len(old_r)}  |  NEW deployed: {len(new_r)}")
    print(f"Shared (both deploy): {len(m)}")
    print(f"OLD-only dropped:     {len(old_r) - len(m)}")
    print(f"NEW-only added:       {len(new_r) - len(m)}")
    print()
    if len(m):
        print(f"On shared series:")
        print(f"  median MAPE old  : {m['mape_old'].median():.2f}%")
        print(f"  median MAPE new  : {m['mape_new'].median():.2f}%")
        print(f"  median delta     : {m['mape_delta'].median():+.2f} pts")
        print(f"  improved (< old) : {(m['mape_delta'] < 0).sum()}/{len(m)}")
        print(f"  regressed (>1pt) : {(m['mape_delta'] > 1).sum()}/{len(m)}")
        print()
        print("Per-commodity delta (median):")
        by_c = m.groupby("commodity")["mape_delta"].agg(
            n="count", median="median", worst="max", best="min"
        ).round(2).sort_values("median", ascending=False)
        print(by_c.to_string())
        print()
        print("Top 10 regressions:")
        cols = ["commodity", "market", "pricetype",
                "strategy_old", "mape_old", "strategy_new", "mape_new", "mape_delta"]
        cols = [c for c in cols if c in m.columns]
        worst = m.nlargest(10, "mape_delta")[cols]
        print(worst.to_string(index=False))

    print()
    print("Commodities NEW deploys that OLD didn't:")
    new_cs = set(new_r["commodity"]) - set(old_r["commodity"])
    for c in sorted(new_cs):
        n = (new_r["commodity"] == c).sum()
        print(f"  + {c:<25} {n} series")


if __name__ == "__main__":
    main()
