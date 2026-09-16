from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "bei_smart_forecasting.csv"
EXPORT_DIR = ROOT / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_series(commodity="Maize (white)", pricetype="Retail"):
    df = pd.read_csv(DATA_PATH, parse_dates=["date_month"])
    monthly = (
        df[(df["commodity"] == commodity) & (df["pricetype"] == pricetype)]
        .dropna(subset=["date_month", "price_per_kg", "rainfall_mm"])
        .groupby("date_month", as_index=False)
        .agg(price_per_kg=("price_per_kg", "mean"), rainfall_mm=("rainfall_mm", "mean"))
        .sort_values("date_month")
    )

    idx = pd.date_range(monthly["date_month"].min(), monthly["date_month"].max(), freq="MS")
    monthly = monthly.set_index("date_month").reindex(idx)
    monthly.index.name = "date_month"
    monthly["price_per_kg"] = monthly["price_per_kg"].interpolate(limit_direction="both")
    monthly["rainfall_mm"] = monthly["rainfall_mm"].interpolate(limit_direction="both")

    monthly["month"] = monthly.index.month
    climatology = monthly.groupby("month")["rainfall_mm"].transform("mean")
    std = monthly.groupby("month")["rainfall_mm"].transform("std").replace(0, np.nan)
    monthly["rainfall_anomaly_z"] = ((monthly["rainfall_mm"] - climatology) / std).fillna(0)
    monthly["rainfall_level_lag3"] = monthly["rainfall_mm"].shift(3)
    monthly["rainfall_anomaly_lag3"] = monthly["rainfall_anomaly_z"].shift(3)
    return monthly.dropna()


def fit_forecast(train, test, exog_col=None):
    kwargs = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 0, 1, 12),
        "enforce_stationarity": False,
        "enforce_invertibility": False,
    }
    if exog_col is None:
        model = SARIMAX(train["price_per_kg"], **kwargs).fit(disp=False, maxiter=300)
        pred = model.get_forecast(steps=len(test)).predicted_mean
    else:
        model = SARIMAX(train["price_per_kg"], exog=train[[exog_col]], **kwargs).fit(disp=False, maxiter=300)
        pred = model.get_forecast(steps=len(test), exog=test[[exog_col]]).predicted_mean
    pred.index = test.index
    return model, pred


def score(actual, pred):
    return {
        "mae": mean_absolute_error(actual, pred),
        "rmse": mean_squared_error(actual, pred) ** 0.5,
        "mape": mape(actual, pred),
    }


def rolling_holdouts(series, windows=8, horizon=3):
    rows = []
    prediction_rows = []
    n = len(series)
    starts = list(range(n - windows * horizon, n, horizon))

    for start in starts:
        train = series.iloc[:start]
        test = series.iloc[start : start + horizon]
        if len(test) < horizon or len(train) < 60:
            continue

        variants = {
            "Price-only SARIMA": None,
            "SARIMAX + rainfall level": "rainfall_level_lag3",
            "SARIMAX + rainfall anomaly": "rainfall_anomaly_lag3",
        }

        for variant, exog_col in variants.items():
            try:
                model, pred = fit_forecast(train, test, exog_col)
                metrics = score(test["price_per_kg"], pred)
                rows.append(
                    {
                        "holdout_start": test.index.min(),
                        "holdout_label": test.index.min().strftime("%Y-%m"),
                        "model": variant,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                        "aic": model.aic,
                    }
                )
                for date, actual, yhat in zip(test.index, test["price_per_kg"], pred):
                    prediction_rows.append(
                        {
                            "date": date,
                            "holdout_label": test.index.min().strftime("%Y-%m"),
                            "model": variant,
                            "actual": actual,
                            "predicted": yhat,
                        }
                    )
            except Exception as exc:
                print(f"Skipped {variant} at {test.index.min().date()}: {exc}")

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def plot_rainfall_decomposition(series):
    result = seasonal_decompose(series["rainfall_mm"], model="additive", period=12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(result.trend, color="seagreen", linewidth=2)
    axes[0].set_title("Rainfall Decomposition: Trend, Seasonal Cycle, and Anomaly")
    axes[0].set_ylabel("Trend")

    axes[1].plot(result.seasonal, color="darkgoldenrod", linewidth=1.2)
    axes[1].set_ylabel("Seasonal")

    axes[2].plot(result.resid, color="slategray", linewidth=1)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Anomaly")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "rainfall_decomposition_trend_seasonal_anomaly.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_errors(metrics):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=metrics, x="holdout_label", y="mae", hue="model", ax=ax, palette=["gray", "coral", "teal"])
    ax.set_title("SARIMA Forecast Error With Rainfall Features")
    ax.set_xlabel("Holdout window")
    ax.set_ylabel("MAE (KES/KG)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "sarima_rainfall_holdout_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_example(series, predictions):
    last_holdout = predictions["holdout_label"].max()
    pred = predictions[predictions["holdout_label"] == last_holdout]
    start = pd.to_datetime(pred["date"].min()) - pd.DateOffset(months=30)
    recent = series.loc[series.index >= start]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent["price_per_kg"], color="black", linewidth=2, label="Actual price")
    for model, color in [
        ("Price-only SARIMA", "gray"),
        ("SARIMAX + rainfall level", "coral"),
        ("SARIMAX + rainfall anomaly", "teal"),
    ]:
        temp = pred[pred["model"] == model]
        ax.plot(pd.to_datetime(temp["date"]), temp["predicted"], marker="o", linestyle="--", color=color, label=model)

    ax.axvspan(pd.to_datetime(pred["date"].min()), pd.to_datetime(pred["date"].max()), color="steelblue", alpha=0.08)
    ax.set_title(f"Example Holdout Forecast: {last_holdout}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (KES/KG)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / "sarima_rainfall_example_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize(metrics):
    summary = (
        metrics.groupby("model")
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_mape=("mape", "mean"), wins=("mae", "count"))
        .reset_index()
    )
    best_by_window = metrics.loc[metrics.groupby("holdout_label")["mae"].idxmin()]
    win_counts = best_by_window["model"].value_counts()
    summary["holdout_wins"] = summary["model"].map(win_counts).fillna(0).astype(int)

    base_mae = summary.loc[summary["model"] == "Price-only SARIMA", "mean_mae"].iloc[0]
    summary["mae_improvement_vs_price_only_pct"] = (base_mae - summary["mean_mae"]) / base_mae * 100
    return summary.sort_values("mean_mae")


def main():
    series = load_series()
    plot_rainfall_decomposition(series)

    metrics, predictions = rolling_holdouts(series, windows=8, horizon=3)
    metrics.to_csv(EXPORT_DIR / "sarima_rainfall_holdout_metrics.csv", index=False)
    predictions.to_csv(EXPORT_DIR / "sarima_rainfall_holdout_predictions.csv", index=False)

    plot_holdout_errors(metrics)
    plot_forecast_example(series, predictions)

    summary = summarize(metrics)
    summary.to_csv(EXPORT_DIR / "sarima_rainfall_feature_summary.csv", index=False)

    print("SARIMA rainfall feature comparison: Maize (white) - Retail")
    print(summary.round(3).to_string(index=False))
    best = summary.iloc[0]
    print()
    print(f"Best average model: {best['model']}")
    print(f"Average MAE: {best['mean_mae']:.2f} KES/KG")
    print(f"Improvement vs price-only: {best['mae_improvement_vs_price_only_pct']:.1f}%")
    print(f"Holdout wins: {int(best['holdout_wins'])} of {metrics['holdout_label'].nunique()}")


if __name__ == "__main__":
    main()
