from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "bei_smart_forecasting.csv"
EXPORT_DIR = ROOT / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


BASE_FEATURES = [
    "price_lag_1",
    "price_lag_3",
    "price_lag_6",
    "price_lag_12",
    "rolling_3m_price",
    "month_sin",
    "month_cos",
]
RAIN_FEATURES = BASE_FEATURES + ["rainfall_lag_3", "rainfall_lag_6"]


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def prepare_monthly_features(df, commodity, pricetype):
    monthly = (
        df[(df["commodity"] == commodity) & (df["pricetype"] == pricetype)]
        .dropna(subset=["date_month", "price_per_kg", "rainfall_mm"])
        .groupby("date_month", as_index=False)
        .agg(price_per_kg=("price_per_kg", "mean"), rainfall_mm=("rainfall_mm", "mean"))
        .sort_values("date_month")
    )

    if monthly.empty:
        raise ValueError("No observations for this commodity/pricetype.")

    full_index = pd.date_range(monthly["date_month"].min(), monthly["date_month"].max(), freq="MS")
    monthly = monthly.set_index("date_month").reindex(full_index)
    monthly.index.name = "date_month"
    monthly["price_per_kg"] = monthly["price_per_kg"].interpolate(limit_direction="both")
    monthly["rainfall_mm"] = monthly["rainfall_mm"].interpolate(limit_direction="both")

    monthly["month"] = monthly.index.month
    rainfall_clim = monthly.groupby("month")["rainfall_mm"].transform("mean")
    rainfall_std = monthly.groupby("month")["rainfall_mm"].transform("std").replace(0, np.nan)
    monthly["rainfall_anomaly_z"] = ((monthly["rainfall_mm"] - rainfall_clim) / rainfall_std).fillna(0)

    for lag in [1, 3, 6, 12]:
        monthly[f"price_lag_{lag}"] = monthly["price_per_kg"].shift(lag)
    monthly["rolling_3m_price"] = monthly["price_per_kg"].shift(1).rolling(3).mean()
    monthly["rainfall_lag_3"] = monthly["rainfall_anomaly_z"].shift(3)
    monthly["rainfall_lag_6"] = monthly["rainfall_anomaly_z"].shift(6)
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

    return monthly.dropna()


def train_test_split(features, test_months=24):
    if len(features) < test_months + 48:
        raise ValueError("Series is too short for this comparison.")
    train = features.iloc[:-test_months]
    test = features.iloc[-test_months:]
    return train, test


def fit_ols(train, feature_cols):
    x_train = sm.add_constant(train[feature_cols], has_constant="add")
    y_train = train["price_per_kg"]
    return sm.OLS(y_train, x_train).fit()


def compare_models(features, test_months=24):
    train, test = train_test_split(features, test_months=test_months)

    baseline_model = fit_ols(train, BASE_FEATURES)
    rainfall_model = fit_ols(train, RAIN_FEATURES)

    baseline_pred = baseline_model.predict(sm.add_constant(test[BASE_FEATURES], has_constant="add"))
    rainfall_pred = rainfall_model.predict(sm.add_constant(test[RAIN_FEATURES], has_constant="add"))

    actual = test["price_per_kg"]
    base_mae = mean_absolute_error(actual, baseline_pred)
    rain_mae = mean_absolute_error(actual, rainfall_pred)
    base_rmse = mean_squared_error(actual, baseline_pred) ** 0.5
    rain_rmse = mean_squared_error(actual, rainfall_pred) ** 0.5
    base_mape = mape(actual, baseline_pred)
    rain_mape = mape(actual, rainfall_pred)

    f_test = rainfall_model.compare_f_test(baseline_model)

    metrics = {
        "n_months": len(features),
        "train_start": train.index.min().date(),
        "train_end": train.index.max().date(),
        "test_start": test.index.min().date(),
        "test_end": test.index.max().date(),
        "baseline_mae": base_mae,
        "rainfall_mae": rain_mae,
        "mae_improvement_pct": (base_mae - rain_mae) / base_mae * 100,
        "baseline_rmse": base_rmse,
        "rainfall_rmse": rain_rmse,
        "rmse_improvement_pct": (base_rmse - rain_rmse) / base_rmse * 100,
        "baseline_mape": base_mape,
        "rainfall_mape": rain_mape,
        "mape_improvement_pct": (base_mape - rain_mape) / base_mape * 100,
        "baseline_adj_r2": baseline_model.rsquared_adj,
        "rainfall_adj_r2": rainfall_model.rsquared_adj,
        "adj_r2_gain": rainfall_model.rsquared_adj - baseline_model.rsquared_adj,
        "rainfall_f_stat": f_test[0],
        "rainfall_f_pvalue": f_test[1],
        "rainfall_lag_3_coef": rainfall_model.params.get("rainfall_lag_3", np.nan),
        "rainfall_lag_3_pvalue": rainfall_model.pvalues.get("rainfall_lag_3", np.nan),
        "rainfall_lag_6_coef": rainfall_model.params.get("rainfall_lag_6", np.nan),
        "rainfall_lag_6_pvalue": rainfall_model.pvalues.get("rainfall_lag_6", np.nan),
    }

    predictions = pd.DataFrame(
        {
            "actual": actual,
            "without_rainfall": baseline_pred,
            "with_rainfall": rainfall_pred,
            "rainfall_lag_3": test["rainfall_lag_3"],
            "rainfall_lag_6": test["rainfall_lag_6"],
        }
    )
    return metrics, predictions


def plot_prediction_comparison(features, predictions, commodity, pricetype):
    fig, ax = plt.subplots(figsize=(13, 6))
    recent = features.loc[features.index >= predictions.index.min() - pd.DateOffset(months=30)]
    ax.plot(recent.index, recent["price_per_kg"], color="gray", linewidth=1.5, alpha=0.6, label="Historical actual")
    ax.plot(predictions.index, predictions["actual"], color="black", linewidth=2.5, label="Actual test price")
    ax.plot(
        predictions.index,
        predictions["without_rainfall"],
        color="coral",
        linewidth=2,
        linestyle="--",
        label="Without rainfall features",
    )
    ax.plot(
        predictions.index,
        predictions["with_rainfall"],
        color="teal",
        linewidth=2,
        linestyle="--",
        label="With lagged rainfall features",
    )
    ax.axvspan(predictions.index.min(), predictions.index.max(), color="steelblue", alpha=0.08, label="Test window")
    ax.set_title(f"{commodity} {pricetype}: Price Prediction With vs Without Rainfall")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (KES/KG)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "rainfall_timeseries_with_vs_without_maize_white_retail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(metrics):
    plot_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "MAPE"],
            "Without rainfall": [metrics["baseline_mae"], metrics["baseline_rmse"], metrics["baseline_mape"]],
            "With rainfall": [metrics["rainfall_mae"], metrics["rainfall_rmse"], metrics["rainfall_mape"]],
        }
    ).melt(id_vars="Metric", var_name="Model", value_name="Error")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=plot_df, x="Metric", y="Error", hue="Model", palette=["coral", "teal"], ax=ax)
    ax.set_title("Prediction Error: With vs Without Rainfall")
    ax.set_ylabel("Error value")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "rainfall_timeseries_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_significance(metrics):
    sig_df = pd.DataFrame(
        {
            "Rainfall feature": ["Lag 3 months", "Lag 6 months"],
            "Coefficient": [metrics["rainfall_lag_3_coef"], metrics["rainfall_lag_6_coef"]],
            "p_value": [metrics["rainfall_lag_3_pvalue"], metrics["rainfall_lag_6_pvalue"]],
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = np.where(sig_df["p_value"] < 0.05, "teal", "coral")
    ax.bar(sig_df["Rainfall feature"], sig_df["Coefficient"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rainfall Coefficient Significance")
    ax.set_ylabel("Effect on price prediction (KES/KG)")
    for idx, row in sig_df.iterrows():
        ax.text(idx, row["Coefficient"], f"p={row['p_value']:.3f}", ha="center", va="bottom" if row["Coefficient"] >= 0 else "top")
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "rainfall_feature_significance_maize_white_retail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_across_series(summary):
    ranked = summary.sort_values("mae_improvement_pct", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = np.where(ranked["mae_improvement_pct"] >= 0, "teal", "coral")
    ax.barh(ranked["series"], ranked["mae_improvement_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Rainfall Feature Impact Across Crop Price Series")
    ax.set_xlabel("MAE improvement from adding rainfall (%)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "rainfall_feature_impact_across_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date", "date_month"])

    commodity = "Maize (white)"
    pricetype = "Retail"
    features = prepare_monthly_features(df, commodity, pricetype)
    metrics, predictions = compare_models(features, test_months=24)
    metrics.update({"commodity": commodity, "pricetype": pricetype, "series": f"{commodity} - {pricetype}"})

    plot_prediction_comparison(features, predictions, commodity, pricetype)
    plot_error_comparison(metrics)
    plot_feature_significance(metrics)

    pd.DataFrame([metrics]).to_csv(EXPORT_DIR / "rainfall_timeseries_maize_white_retail_metrics.csv", index=False)
    predictions.to_csv(EXPORT_DIR / "rainfall_timeseries_maize_white_retail_predictions.csv")

    candidates = (
        df[df["category"].str.contains("cereals|pulses|vegetables|tubers", case=False, na=False)]
        .groupby(["commodity", "pricetype"])["date_month"]
        .nunique()
        .reset_index(name="months")
        .query("months >= 72")
        .sort_values("months", ascending=False)
        .head(16)
    )

    summary_rows = []
    for row in candidates.itertuples(index=False):
        try:
            series_features = prepare_monthly_features(df, row.commodity, row.pricetype)
            series_metrics, _ = compare_models(series_features, test_months=24)
            series_metrics.update(
                {
                    "commodity": row.commodity,
                    "pricetype": row.pricetype,
                    "series": f"{row.commodity} - {row.pricetype}",
                }
            )
            summary_rows.append(series_metrics)
        except Exception as exc:
            print(f"Skipped {row.commodity} - {row.pricetype}: {exc}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EXPORT_DIR / "rainfall_feature_significance_summary.csv", index=False)
    if not summary.empty:
        plot_across_series(summary)

    print("Main comparison: Maize (white) - Retail")
    print(f"Train: {metrics['train_start']} to {metrics['train_end']}")
    print(f"Test:  {metrics['test_start']} to {metrics['test_end']}")
    print(f"MAE without rainfall: {metrics['baseline_mae']:.2f}")
    print(f"MAE with rainfall:    {metrics['rainfall_mae']:.2f}")
    print(f"MAE improvement:      {metrics['mae_improvement_pct']:.1f}%")
    print(f"RMSE improvement:     {metrics['rmse_improvement_pct']:.1f}%")
    print(f"Adj R2 without rain:  {metrics['baseline_adj_r2']:.3f}")
    print(f"Adj R2 with rain:     {metrics['rainfall_adj_r2']:.3f}")
    print(f"Rainfall F-test p:    {metrics['rainfall_f_pvalue']:.4f}")
    print(f"Lag 3 rainfall p:     {metrics['rainfall_lag_3_pvalue']:.4f}")
    print(f"Lag 6 rainfall p:     {metrics['rainfall_lag_6_pvalue']:.4f}")
    print()
    print("Across-series summary:")
    cols = ["series", "mae_improvement_pct", "rainfall_f_pvalue", "rainfall_lag_3_pvalue", "rainfall_lag_6_pvalue"]
    print(summary[cols].sort_values("mae_improvement_pct", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
