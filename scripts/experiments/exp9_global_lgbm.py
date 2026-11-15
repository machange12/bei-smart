"""Exp 9: one global LightGBM across every series vs Chronos-Bolt-small.

Production fits each series alone. A global model learns from all ~1,500 series
at once (lags of the price, calendar month, commodity, price type, horizon) and
predicts the log change h steps ahead, h = 1..12.

Same test as production: each served series' last 12 observations. No leakage:
for a series whose test starts at date D, the model is trained only on targets
dated before D (one model per distinct D).

    python scripts/experiments/exp9_global_lgbm.py
Arms: bolt, lgbm, blend (mean of the two).
"""
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rebuild_forecasts as rf  # noqa: E402
from common import chronos_predict, mape  # noqa: E402

OUT = rf.EXPORTS / "experiments"
KEY = ["commodity", "market", "pricetype"]
FEATS = ["r1", "r2", "r3", "r6", "r12", "vs_mean12", "vol12", "level", "h", "month", "target_month",
         "commodity", "pricetype"]


def origin_features(y: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per observation t (in observation order, gaps ignored like production)."""
    ly = pd.Series(np.log(y))
    r = ly.diff()
    return pd.DataFrame({
        "r1": r, "r2": r.shift(1), "r3": r.shift(2),
        "r6": ly - ly.shift(6), "r12": ly - ly.shift(12),
        "vs_mean12": ly - ly.rolling(12, min_periods=6).mean(),
        "vol12": r.rolling(12, min_periods=6).std(),
        "level": ly, "month": dates.month, "date": dates,
    })


def training_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (c, mk, pt), g in df.groupby(KEY):
        if len(g) < 24:
            continue
        y, dates = g["price_per_kg"].to_numpy(), pd.DatetimeIndex(g["date"])
        f = origin_features(y, dates).assign(commodity=c, pricetype=pt)
        ly = f["level"].to_numpy()
        for h in range(1, 13):
            t = f.iloc[:-h].copy()
            t["h"] = h
            t["target"] = ly[h:] - ly[:-h]
            t["target_date"] = dates[h:]
            t["target_month"] = dates[h:].month
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    df = pd.read_csv(rf.INPUT, usecols=KEY + ["date", "price_per_kg"], parse_dates=["date"])
    df = df[~df["market"].str.contains(rf.SETTLEMENT_RE) & (df["price_per_kg"] > 0)].sort_values(KEY + ["date"])
    cats = {"commodity": pd.CategoricalDtype(sorted(df.commodity.unique())),
            "pricetype": pd.CategoricalDtype(sorted(df.pricetype.unique()))}
    train = training_table(df).astype(cats)
    print(f"training table: {len(train):,} rows", flush=True)

    served = pd.read_csv(rf.APP_DATA / "model_metrics.csv").query("reliable")
    S = []
    for r in served.itertuples():
        g = df[(df.commodity == r.commodity) & (df.market == r.market) & (df.pricetype == r.pricetype)]
        if len(g) >= 36:
            S.append((r, g))
    by_start: dict = {}
    for r, g in S:
        by_start.setdefault(g["date"].iloc[-12], []).append((r, g))
    print(f"{len(S)} series, {len(by_start)} distinct test starts", flush=True)

    rows = []
    for d0, group in sorted(by_start.items()):
        tr = train[train["target_date"] < d0]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, verbose=-1)
        model.fit(tr[FEATS], tr["target"])
        for r, g in group:
            y, dates = g["price_per_kg"].to_numpy(), pd.DatetimeIndex(g["date"])
            ctx, test = y[:-12], y[-12:]
            last = origin_features(ctx, dates[:-12]).iloc[[-1]].assign(commodity=r.commodity, pricetype=r.pricetype)
            x = pd.concat([last] * 12, ignore_index=True).astype(cats)
            x["h"] = range(1, 13)
            x["target_month"] = (x["month"] + x["h"] - 1) % 12 + 1  # assumes no gap ahead
            p_lgbm = ctx[-1] * np.exp(model.predict(x[FEATS]))
            p_bolt = chronos_predict(ctx, 12)
            rows.append({"commodity": r.commodity, "market": r.market, "pricetype": r.pricetype, "n": len(y),
                         "bolt": mape(test, p_bolt), "lgbm": mape(test, p_lgbm),
                         "blend": mape(test, (p_bolt + p_lgbm) / 2)})
        print(f"{d0:%Y-%m}: {len(group)} series, trained on {len(tr):,} rows", flush=True)

    d = pd.DataFrame(rows).round(3)
    d.to_csv(OUT / "exp9_global_lgbm.csv", index=False)
    from scipy.stats import wilcoxon
    for title, sub in [("all served series", d), ("short (< 84 obs)", d[d.n < 84]), ("long (>= 84 obs)", d[d.n >= 84])]:
        print(f"\n== {title} ({len(sub)}) ==")
        for a in ("bolt", "lgbm", "blend"):
            extra = "" if a == "bolt" else (f"  beats bolt on {int((sub[a] < sub.bolt).sum())}/{len(sub)}"
                                            f"  p={wilcoxon(sub[a], sub.bolt).pvalue:.3f}")
            print(f"{a:<6} median {sub[a].median():6.2f}  mean {sub[a].mean():6.2f}{extra}")


if __name__ == "__main__":
    main()
