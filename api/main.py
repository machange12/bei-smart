"""AgriPulse FastAPI backend.

Serves precomputed forecasts and anomaly/volatility data from CSV exports,
and runs the XGBoost price-movement classifier on demand. All artifacts are
loaded once at startup (see `lifespan`) and reused across requests.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "data" / "exports"
HISTORICAL_PATH = BASE_DIR / "data" / "cleaned" / "bei_smart_forecasting_final.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.metadata = json.loads((MODELS_DIR / "classifier_metadata.json").read_text())
    app.state.encoders = joblib.load(MODELS_DIR / "classifier_encoders.joblib")
    app.state.label_encoder = joblib.load(MODELS_DIR / "classifier_label_encoder.joblib")
    app.state.xgb_model = joblib.load(MODELS_DIR / "classifier_xgb.joblib")
    app.state.routing = json.loads((MODELS_DIR / "model_routing.json").read_text())

    app.state.forecasts_df = pd.read_csv(
        EXPORTS_DIR / "production_forecasts.csv", parse_dates=["forecast_date"]
    )
    app.state.anomalies_df = pd.read_csv(
        EXPORTS_DIR / "price_anomalies.csv", parse_dates=["latest_date"]
    )

    historical = pd.read_csv(HISTORICAL_PATH, usecols=["market", "admin1"])
    app.state.market_region_map = (
        historical.dropna(subset=["admin1"])
        .drop_duplicates(subset=["market"])
        .set_index("market")["admin1"]
        .to_dict()
    )

    app.state.validation_summary_df = pd.read_csv(
        EXPORTS_DIR / "validation_summary.csv",
        parse_dates=["period_start", "period_end", "series_ended"],
    )
    app.state.validation_archive_df = pd.read_csv(
        EXPORTS_DIR / "validation_archive.csv",
        parse_dates=["date", "series_ended"],
    )

    yield


app = FastAPI(title="AgriPulse API", lifespan=lifespan)

# Defensive: Streamlit's /classify call is server-side (via `requests`), not a
# browser fetch, so CORS isn't strictly required today — but this keeps the
# door open for future browser-based clients without a code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool


class ForecastRow(BaseModel):
    commodity: str
    market: str
    region: str
    pricetype: str
    strategy: str
    forecast_date: date
    horizon_months: int
    forecast_price_kes: float
    forecast_lower: float
    forecast_upper: float
    test_mape: float
    confidence: str
    confidence_source: str


class ForecastResponse(BaseModel):
    count: int
    results: list[ForecastRow]


class ClassifyRequest(BaseModel):
    region: str
    county: str
    market: str
    category: str
    commodity: str
    season: str
    month: int = Field(ge=1, le=12)
    year: int
    rainfall_mm: float
    diesel_price_kes: float
    rolling_12m_avg: float
    latitude: float
    longitude: float


class ClassifyResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


class AnomalyRow(BaseModel):
    commodity: str
    market: str
    region: str | None
    pricetype: str
    latest_date: date
    latest_price: float
    normal_low: float
    normal_high: float
    status: str
    deviation_pct: float


class AlertsResponse(BaseModel):
    count: int
    results: list[AnomalyRow]


class MetadataResponse(BaseModel):
    model: str
    target: str
    classes: list[str]
    feature_order: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    valid_values: dict[str, list[str]]
    metrics: dict
    per_class: dict
    live_series_count: int
    archived_series_count: int


class ValidationSummaryRow(BaseModel):
    commodity: str
    market: str
    pricetype: str
    region: str
    n_months: int
    median_error: float
    mean_error: float
    period_start: date
    period_end: date
    series_ended: date


class ValidationSummaryResponse(BaseModel):
    count: int
    results: list[ValidationSummaryRow]


class ValidationPairRow(BaseModel):
    date: date
    actual_kes: float
    forecast_kes: float
    error_pct: float


class ValidationSeriesResponse(BaseModel):
    commodity: str
    market: str
    pricetype: str
    region: str
    n_months: int
    median_error: float
    mean_error: float
    pairs: list[ValidationPairRow]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "xgb_model", None) is not None
    return HealthResponse(status="ok", artifacts_loaded=loaded)


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata(request: Request) -> MetadataResponse:
    forecasts_df = request.app.state.forecasts_df
    live_series_count = len(
        forecasts_df[["commodity", "market", "pricetype"]].drop_duplicates()
    )
    archived_series_count = len(request.app.state.validation_summary_df)

    return MetadataResponse(
        **request.app.state.metadata,
        live_series_count=live_series_count,
        archived_series_count=archived_series_count,
    )


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    request: Request,
    commodity: str = Query(...),
    market: str = Query(...),
    pricetype: str = Query(...),
    horizon: int = Query(..., description="Forecast horizon in months (3, 6, or 12)"),
) -> ForecastResponse:
    df = request.app.state.forecasts_df
    matches = df[
        (df["commodity"] == commodity)
        & (df["market"] == market)
        & (df["pricetype"] == pricetype)
        & (df["horizon_months"] == horizon)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast for commodity={commodity!r}, market={market!r}, "
                f"pricetype={pricetype!r}, horizon={horizon!r}"
            ),
        )

    routing = request.app.state.routing
    results: list[ForecastRow] = []
    for row in matches.to_dict("records"):
        key = f"{row['commodity']}|{row['market']}|{row['pricetype']}"
        route = routing.get(key)
        if route is not None:
            confidence = route["confidence"]
            test_mape = route["test_mape"]
            confidence_source = "routing"
        else:
            confidence = row["confidence"]
            test_mape = row["test_mape"]
            confidence_source = "csv_fallback"

        results.append(
            ForecastRow(
                commodity=row["commodity"],
                market=row["market"],
                region=row["region"],
                pricetype=row["pricetype"],
                strategy=row["strategy"],
                forecast_date=row["forecast_date"].date(),
                horizon_months=row["horizon_months"],
                forecast_price_kes=row["forecast_price_kes"],
                forecast_lower=row["forecast_lower"],
                forecast_upper=row["forecast_upper"],
                test_mape=test_mape,
                confidence=confidence,
                confidence_source=confidence_source,
            )
        )

    return ForecastResponse(count=len(results), results=results)


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, payload: ClassifyRequest) -> ClassifyResponse:
    metadata = request.app.state.metadata
    valid_values = metadata["valid_values"]
    categorical_features = metadata["categorical_features"]

    errors = []
    for field in categorical_features:
        value = getattr(payload, field)
        allowed = valid_values[field]
        if value not in allowed:
            errors.append(
                {"field": field, "invalid_value": value, "allowed_values": allowed}
            )

    if errors:
        raise HTTPException(
            status_code=422,
            detail={"detail": "Invalid categorical value(s)", "errors": errors},
        )

    encoders = request.app.state.encoders
    xgb_model = request.app.state.xgb_model
    label_encoder = request.app.state.label_encoder

    row = payload.model_dump()
    X = pd.DataFrame([row])[metadata["feature_order"]]
    for col in categorical_features:
        X[col] = encoders[col].transform(X[col].astype(str))

    pred_idx = xgb_model.predict(X)
    predicted_class = label_encoder.inverse_transform(pred_idx)[0]
    proba = xgb_model.predict_proba(X)[0]
    probabilities = {
        cls: float(p) for cls, p in zip(label_encoder.classes_, proba)
    }

    return ClassifyResponse(predicted_class=predicted_class, probabilities=probabilities)


@app.get("/alerts", response_model=AlertsResponse)
def get_alerts(
    request: Request,
    region: str | None = Query(None),
    commodity: str | None = Query(None),
) -> AlertsResponse:
    df = request.app.state.anomalies_df.copy()
    df["region"] = df["market"].map(request.app.state.market_region_map)

    if commodity:
        df = df[df["commodity"] == commodity]
    if region:
        df = df[df["region"] == region]

    results = [
        AnomalyRow(
            commodity=row["commodity"],
            market=row["market"],
            region=row["region"] if pd.notna(row["region"]) else None,
            pricetype=row["pricetype"],
            latest_date=row["latest_date"].date(),
            latest_price=row["latest_price"],
            normal_low=row["normal_low"],
            normal_high=row["normal_high"],
            status=row["status"],
            deviation_pct=row["deviation_pct"],
        )
        for row in df.to_dict("records")
    ]

    return AlertsResponse(count=len(results), results=results)


@app.get("/validation", response_model=ValidationSummaryResponse)
def get_validation_summary(
    request: Request,
    market: str | None = Query(None),
    commodity: str | None = Query(None),
) -> ValidationSummaryResponse:
    df = request.app.state.validation_summary_df
    if commodity:
        df = df[df["commodity"] == commodity]
    if market:
        df = df[df["market"] == market]

    results = [
        ValidationSummaryRow(
            commodity=row["commodity"],
            market=row["market"],
            pricetype=row["pricetype"],
            region=row["region"],
            n_months=row["n_months"],
            median_error=row["median_error"],
            mean_error=row["mean_error"],
            period_start=row["period_start"].date(),
            period_end=row["period_end"].date(),
            series_ended=row["series_ended"].date(),
        )
        for row in df.to_dict("records")
    ]

    return ValidationSummaryResponse(count=len(results), results=results)


@app.get("/validation/{commodity}/{market}", response_model=ValidationSeriesResponse)
def get_validation_series(
    request: Request, commodity: str, market: str
) -> ValidationSeriesResponse:
    summary_df = request.app.state.validation_summary_df
    summary_matches = summary_df[
        (summary_df["commodity"] == commodity) & (summary_df["market"] == market)
    ]

    if summary_matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No archived validation series for commodity={commodity!r}, market={market!r}",
        )

    summary_row = summary_matches.iloc[0]

    archive_df = request.app.state.validation_archive_df
    pair_matches = archive_df[
        (archive_df["commodity"] == commodity) & (archive_df["market"] == market)
    ].sort_values("date")

    pairs = [
        ValidationPairRow(
            date=row["date"].date(),
            actual_kes=row["actual_kes"],
            forecast_kes=row["forecast_kes"],
            error_pct=row["error_pct"],
        )
        for row in pair_matches.to_dict("records")
    ]

    return ValidationSeriesResponse(
        commodity=summary_row["commodity"],
        market=summary_row["market"],
        pricetype=summary_row["pricetype"],
        region=summary_row["region"],
        n_months=int(summary_row["n_months"]),
        median_error=summary_row["median_error"],
        mean_error=summary_row["mean_error"],
        pairs=pairs,
    )
