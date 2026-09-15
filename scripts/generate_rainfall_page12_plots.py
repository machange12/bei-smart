from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "bei_smart_forecasting.csv"
EXPORT_DIR = ROOT / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

df_fc = pd.read_csv(DATA_PATH, parse_dates=["date", "date_month"])

rain_price = df_fc.dropna(subset=["rainfall_mm", "price_per_kg"]).copy()
rain_price = rain_price[rain_price["price_per_kg"] > 0]
rain_price["log_price_per_kg"] = np.log1p(rain_price["price_per_kg"])

same_month_corr = rain_price[["rainfall_mm", "price_per_kg"]].corr().iloc[0, 1]
log_same_month_corr = rain_price[["rainfall_mm", "log_price_per_kg"]].corr().iloc[0, 1]

plt.figure(figsize=(11, 5))
sns.regplot(
    data=rain_price.sample(min(len(rain_price), 8000), random_state=42),
    x="rainfall_mm",
    y="log_price_per_kg",
    scatter_kws={"alpha": 0.12, "s": 18, "color": "teal"},
    line_kws={"color": "black", "linewidth": 2},
    lowess=True,
)
plt.title("Same-month Rainfall vs Food Price (log KES/KG)")
plt.xlabel("Rainfall (mm)")
plt.ylabel("log(Price per KG)")
plt.tight_layout()
plt.savefig(EXPORT_DIR / "eda_rainfall_same_month_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

monthly = (
    df_fc.dropna(subset=["date_month", "commodity", "pricetype", "price_per_kg", "rainfall_mm"])
    .groupby(["commodity", "pricetype", "date_month"], as_index=False)
    .agg(price_per_kg=("price_per_kg", "mean"), rainfall_mm=("rainfall_mm", "mean"))
    .sort_values(["commodity", "pricetype", "date_month"])
)

monthly["month"] = monthly["date_month"].dt.month
monthly["rain_climatology"] = monthly.groupby(["commodity", "pricetype", "month"])["rainfall_mm"].transform("mean")
monthly["rain_std"] = monthly.groupby(["commodity", "pricetype", "month"])["rainfall_mm"].transform("std")
monthly["rainfall_anomaly_mm"] = monthly["rainfall_mm"] - monthly["rain_climatology"]
monthly["rainfall_anomaly_z"] = monthly["rainfall_anomaly_mm"] / monthly["rain_std"].replace(0, np.nan)
monthly["price_yoy_change_pct"] = monthly.groupby(["commodity", "pricetype"])["price_per_kg"].pct_change(12) * 100

lag_rows = []
for (commodity, pricetype), grp in monthly.groupby(["commodity", "pricetype"]):
    grp = grp.sort_values("date_month").copy()
    if len(grp) < 36:
        continue
    for lag in range(0, 7):
        lagged_rain = grp["rainfall_anomaly_z"].shift(lag)
        corr = grp["price_yoy_change_pct"].corr(lagged_rain)
        lag_rows.append(
            {
                "commodity": commodity,
                "pricetype": pricetype,
                "lag_months": lag,
                "correlation": corr,
                "observations": pd.concat([grp["price_yoy_change_pct"], lagged_rain], axis=1).dropna().shape[0],
            }
        )

lag_corr = pd.DataFrame(lag_rows).dropna(subset=["correlation"])
lag_corr["series"] = lag_corr["commodity"] + " - " + lag_corr["pricetype"]
lag_corr.to_csv(EXPORT_DIR / "eda_rainfall_anomaly_lag_correlation.csv", index=False)

focus_series = lag_corr.groupby("series")["observations"].max().sort_values(ascending=False).head(8).index
lag_focus = lag_corr[lag_corr["series"].isin(focus_series)]

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=lag_focus,
    x="lag_months",
    y="correlation",
    hue="series",
    marker="o",
    linewidth=2,
)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Rainfall Anomaly Lag vs YoY Price Change")
plt.xlabel("Rainfall lag (months before price observation)")
plt.ylabel("Correlation with YoY price change")
plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(EXPORT_DIR / "eda_rainfall_anomaly_lag_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

bucketed = monthly.dropna(subset=["rainfall_anomaly_z", "price_yoy_change_pct"]).copy()
bucketed["rainfall_bucket"] = pd.cut(
    bucketed["rainfall_anomaly_z"],
    bins=[-np.inf, -1, 1, np.inf],
    labels=["Dry anomaly", "Normal rainfall", "Wet anomaly"],
)

bucket_rows = []
for lag in range(0, 7):
    temp = bucketed.copy()
    temp["lagged_bucket"] = temp.groupby(["commodity", "pricetype"])["rainfall_bucket"].shift(lag)
    grouped = (
        temp.dropna(subset=["lagged_bucket", "price_yoy_change_pct"])
        .groupby(["lagged_bucket"], observed=False)
        .agg(
            avg_yoy_price_change_pct=("price_yoy_change_pct", "mean"),
            median_yoy_price_change_pct=("price_yoy_change_pct", "median"),
            observations=("price_yoy_change_pct", "size"),
        )
        .reset_index()
    )
    grouped["lag_months"] = lag
    bucket_rows.append(grouped)

bucket_summary = pd.concat(bucket_rows, ignore_index=True)
bucket_summary.to_csv(EXPORT_DIR / "eda_rainfall_anomaly_price_response.csv", index=False)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=bucket_summary,
    x="lag_months",
    y="avg_yoy_price_change_pct",
    hue="lagged_bucket",
    marker="o",
    linewidth=2,
    palette={"Dry anomaly": "coral", "Normal rainfall": "steelblue", "Wet anomaly": "teal"},
)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Average YoY Price Change After Rainfall Anomalies")
plt.xlabel("Lag from rainfall month to price month")
plt.ylabel("Average YoY price change (%)")
plt.legend(title="Rainfall type")
plt.tight_layout()
plt.savefig(EXPORT_DIR / "eda_rainfall_anomaly_price_response.png", dpi=150, bbox_inches="tight")
plt.close()

example = (
    monthly[
        (monthly["commodity"].isin(["Maize", "Maize (white)"]))
        & (monthly["pricetype"] == "Retail")
    ]
    .groupby("date_month", as_index=False)
    .agg(
        price_per_kg=("price_per_kg", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
        rainfall_anomaly_z=("rainfall_anomaly_z", "mean"),
    )
    .sort_values("date_month")
)
example["price_yoy_change_pct"] = example["price_per_kg"].pct_change(12) * 100

fig, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(
    example["date_month"],
    example["price_yoy_change_pct"],
    color="steelblue",
    linewidth=2,
    label="YoY price change (%)",
)
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_ylabel("YoY price change (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.bar(
    example["date_month"],
    example["rainfall_anomaly_z"],
    width=25,
    color=np.where(example["rainfall_anomaly_z"] >= 0, "teal", "coral"),
    alpha=0.35,
    label="Rainfall anomaly z-score",
)
ax2.set_ylabel("Rainfall anomaly z-score", color="teal")
ax2.tick_params(axis="y", labelcolor="teal")

plt.title("Maize Retail Price Inflation vs Rainfall Anomalies")
fig.tight_layout()
plt.savefig(EXPORT_DIR / "eda_maize_rainfall_anomaly_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()

best_lags = (
    lag_corr.assign(abs_corr=lambda x: x["correlation"].abs())
    .sort_values("abs_corr", ascending=False)
    .groupby("series")
    .head(1)
    .sort_values("abs_corr", ascending=False)
    .head(10)
)

print("Generated rainfall page 12 plots:")
print(f"- Same-month raw correlation: {same_month_corr:.3f}")
print(f"- Same-month log correlation: {log_same_month_corr:.3f}")
print(best_lags[["series", "lag_months", "correlation", "observations"]].round(3).to_string(index=False))
