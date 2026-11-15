# AgriPulse (Bei Smart): Kenya Food Price Intelligence

Forecasts of Kenyan food and livestock prices market by market, alerts when a price jumps or crashes, and a chatbot that answers in English or Kiswahili. Built as a Moringa School data science capstone.

**Live app:** https://bei-smart.streamlit.app/

## Live numbers

<!-- stats:start -->
_Updated automatically by the monthly refresh on 2026-11-15._

| | |
|---|---|
| Forecast series served | 283 (261 with data under 6 months old) |
| Markets / commodities | 84 / 23 |
| Regions covered | 8 of 8 |
| Median forecast error (12-month holdout) | 8.7% |
| Models serving | Chronos 170, blend of 2 models 113 |
| Price alerts | 34 spikes, 8 crashes across 788 monitored series |
| Prices observed up to | October 2026 |
<!-- stats:end -->

## What it does

| Page | Question it answers |
|---|---|
| **Home** | What matters this month: best place to sell, biggest price alerts, cheapest maize |
| **Forecast** | What will this commodity cost in this market in 3, 6 or 12 months? (shareable links, CSV download) |
| **Alerts** | Which prices are unusually high (spike) or low (crash) right now? |
| **Map** | Where are the alerts? |
| **Where to sell** | Which market pays most for my produce after transport? (adjustable cost per km) |
| **Classify** | Is a commodity likely cheap or expensive in a region? (a shortlist, not a verdict) |
| **Coming soon** | Which series are still reporting and will pass the data rules within a year, and when |
| **Ask AgriPulse** 💬 | Floating, draggable chatbot on every page, English and Kiswahili, answers cite the data |

## How it works

```
 KAMIS scraper ─┐
 FEWS NET API ──┤
 WFP (HDX) ─────┼─▶ build_chained_all ─▶ enrich_chain ─▶ rebuild_anomalies ─▶ rebuild_forecasts ─▶ app/data/ ─▶ Streamlit
 FAO FPMA API ──┤      (one monthly                         (IQR band,           (Prophet, SARIMA,
 KNBS, WFP supp ┘       price chain)                         last 36 months)      Chronos per series)
```

- **Runs itself:** `.github/workflows/refresh-data.yml` runs the whole chain on the 5th of each month, updates the numbers above, and commits. Streamlit Cloud redeploys on that commit.
- **Honest accuracy:** each series is scored on its last 12 months, which the model never saw, and the test window is never used to *pick* the model. Long series (84+ months) use a blend of the 2 best of four models (Prophet, SARIMA, Chronos-2, a LightGBM trained across all series), weighted on a separate validation window. Shorter series use Chronos-2 zero-shot, which forecasts related series together. A series is only served if its error is 25% or less.
- **Gates:** at least 36 monthly observations, updated within 12 months, under 25% of months missing, not erratic. Refugee-settlement markets are excluded, because their prices follow aid distribution, not the market.
- **Alerts:** a price outside its own 3-year normal range, after removing the series' usual seasonal pattern (the idea behind WFP's ALPS).
- **Bands:** every forecast band is an 80% range, rescaled each run so about 8 in 10 held-out prices fall inside.
- **Units:** prices are KES per kg, except goats, which are KES per head.

## Repository map

```
app/                     Streamlit dashboard (entry point: app/streamlit_app.py)
  home.py, pages/        the six pages
  src/ui.py              shared look, charts, floating chat
  src/rag/               chatbot: routing, data lookups, retrieval, answer templates
  src/units.py           kg vs per-head labels
  data/                  what the app serves (rewritten by the monthly refresh)
  data/kb/               31 short explainers the chatbot retrieves from
  models/                price-level classifier (XGBoost)
  tests/                 chatbot regression questions (python -m tests.test_rag from app/)
api/                     optional FastAPI backend over the same exports
scripts/                 the monthly pipeline, in run order:
  scrape_kamis.py          KAMIS prices (merges into history, --discover finds new products)
  kamis_local_push.ps1     monthly KAMIS pull + push from a Kenyan machine (scheduled task)
  ingest_fpma.py           FAO GIEWS FPMA prices
  fetch_external_features.py  ONI, grain futures, KES/USD, EPRA fuel
  build_chained_all.py     stitch every source into one monthly chain
  enrich_chain.py          join external features
  rebuild_anomalies.py     spike / crash alerts
  rebuild_forecasts.py     screen, model, score, write forecasts
  build_weekly.py          latest-week KAMIS prices for the Forecast page (run by the local KAMIS pull)
  send_alerts.py           SMS price alerts via Africa's Talking (dry run unless --send; needs secrets)
  update_readme_stats.py   refresh the numbers in this README
  diff_mape.py             compare two runs' accuracy
  experiments/             one-off studies behind design decisions (exp1-exp9 + rainfall work)
data/
  raw/                   source pulls (FEWS, WFP, KAMIS, supplements, new_sources/)
  cleaned/               the monthly chain + older modelling extracts
  exports/               pipeline outputs, EDA figures, experiments/ results
notebooks/
  eda/01_eda_master.ipynb          exploratory analysis
  eda/02_time_series_eda.ipynb     trend, seasonality, rainfall lags
  modeling/03_forecasting.ipynb    forecasting method development
  modeling/04_classification.ipynb price-level classifier
  archive/                         earlier drafts and per-member notebooks
reports/                 the CRISP-DM data report
PROGRESS.md              running log of decisions and next steps
```

## Run locally

```bash
python -m venv venv && venv/Scripts/pip install -r app/requirements.txt
venv/Scripts/python.exe -m streamlit run app/streamlit_app.py   # chat needs GROQ_API_KEY in .env (see .env.example)
```

Full pipeline (slow: the forecast rebuild takes hours; it resumes from a checkpoint if interrupted):

```bash
pip install -r requirements.txt
python scripts/scrape_kamis.py --discover --months-back 3
python scripts/build_chained_all.py && python scripts/enrich_chain.py
python scripts/rebuild_anomalies.py && python scripts/rebuild_forecasts.py
```

Most scripts have a `--demo` self-check.

**KAMIS is pulled from Kenya, not GitHub.** The KAMIS site refuses GitHub's runners, so a Windows scheduled task (`bei-smart KAMIS pull`, the 4th of each month at 10:00) scrapes the last 3 months and pushes them before the cloud run on the 5th. To run it by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\kamis_local_push.ps1
```

## Data sources

KAMIS (Kenya Ministry of Agriculture) · FEWS NET Data Warehouse · WFP Kenya Food Prices via HDX · FAO GIEWS FPMA · KNBS · ERA5 rainfall (World Bank CCKP) · NOAA ONI · World Bank commodity prices. All are public. Sources reviewed and not used, with the reasons, are in `PROGRESS.md`.

## Tech stack

Python, pandas · statsmodels (SARIMA), Prophet, Amazon Chronos-2, LightGBM · XGBoost (classifier) · ChromaDB + sentence-transformers (retrieval), Groq Llama 3.3 70B (answers) · Streamlit, Plotly, Folium · FastAPI · GitHub Actions

## Team

| Name |
|---|
| Mathews Machanje |
| Millicent Onchari |
| James Wakhu |
| Esther Muchoki |
| Mohamed Abdirahman |
| Marren Ojode |

## Deliverables

CRISP-DM data report (`reports/`) · notebooks · this repository · slide deck · deployment: [https://bei-smart.streamlit.app/](https://bei-smart.streamlit.app/)

## Acknowledgements

World Food Programme Kenya · Humanitarian Data Exchange · FEWS NET · Kenya Ministry of Agriculture (KAMIS) · FAO GIEWS · Moringa School Data Science Programme
