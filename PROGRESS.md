# bei-smart — Progress Log

Last updated: 2026-10-29

## Current state

**Data pipeline:**
- Chain: `data/cleaned/bei_smart_forecasting_chained_all.csv` — 114,632 rows, **49 commodities**, **468 markets** (settlement markets still in the chain; excluded downstream), 1,524 series ≥24 obs, data through 2026-09
- Sources stitched: WFP_SUPP, FEWS, WFP, KAMIS, KNBS (`SOURCE_PRIORITY` order). KAMIS is now the largest source (160 of 266 served series)
- **Refreshes automatically** on the 5th of each month via `.github/workflows/refresh-data.yml`
- External features merged into chain (oni, corn/wheat futures, KES/USD, EPRA fuel) — **present but unused by forecaster**

**Forecasts (`app/data/`, pre-maize-merge — see below):**
- **266 served series** (commodity+market+pricetype), **80 markets**, 22 commodity labels, **all 8 regions**; horizons 3/6/12 months. 284 screened, 18 fail the 25% MAPE gate
- Median test MAPE: **9.65%** (mean 10.36%)
- Strategy mix: chronos 145, ensemble 47, sarima 39, prophet 35 (model: `amazon/chronos-bolt-small`; ensemble = weighted top-2 by 1/val_mape)
- Confidence: high 141, medium 112, low 13
- Anomalies: 781 series, 135 markets, 31 SPIKE + 10 CRASH (36-month IQR band)
- Down from 318 because settlement markets are excluded again; the next full rebuild will also fold 50 `Maize (white)` series into `Maize`, so expect the count to move again

**Dashboard:** Streamlit at `app/streamlit_app.py`, points at `app/data/` CSVs. Chat is a floating "Ask AgriPulse" button on every page (no separate Chat page).

## This session (2026-10) — automation, data hygiene, demo UX

**Automation**
- `.github/workflows/refresh-data.yml` — cron `0 3 5 * *` + manual dispatch: fetch WFP (HDX), `fetch_external_features`, `scrape_kamis --discover` (last + current year) → `build_chained_all` → `enrich_chain` → `rebuild_anomalies` → `rebuild_forecasts` → commit `app/data`, exports, chain
- `scripts/scrape_kamis.py` merges into existing `kamis_prices.csv` instead of overwriting; `--discover` full-history-scrapes live product IDs not in `PRODUCTS`/`SKIPPED_IDS`

**Data / model hygiene**
- Refugee settlement markets (Kakuma, Kalobeyei, Dadaab/Daadab, Hagadera, Ifo, Dagahaley + Kakuma camp names) excluded from forecasts and anomalies — `SETTLEMENT_RE` in `rebuild_forecasts.py`, reused by `rebuild_anomalies.py`
- Region column normalised to the 8 former provinces (`REGIONS`, `COUNTY_TO_REGION` for KAMIS counties, market overrides; "Northern" retired)
- `build_chained_all.py`: `Maize (white)` variants fold onto `Maize`; `product_<id>` placeholders dropped (product_10 gone). Chain + anomalies already reflect this; **production forecasts still show both maize labels until the next full rebuild**
- `rebuild_anomalies.py`: IQR band from the 36 months before latest (was all history → flagged nothing), anchor = latest month in data
- Dead `_prophet_predict_log` / `_sarima_predict_log` wrappers deleted
- `--chronos-model` flag on `rebuild_forecasts.py` (default `amazon/chronos-bolt-small`) for the Bolt-base test
- RAG `lookup.get_forecast`: falls back to the variety a market tracks (e.g. Maize → Maize (white)); RAG suite 20 → 24 found

**UX (demo)**
- Floating "Ask AgriPulse" chat (st.dialog, history in session_state) replaces `6_Chat.py`; RAG stack warmed in a background thread
- Freshness banner on every page ("Data as of <month> · refreshes automatically on the 5th")
- Landing "This month" cards: best time & place to sell, price alerts, cheapest maize to buy
- Mobile CSS (≤640px padding, icon-only chat button)
- Forecast page: shareable links (selection mirrored into URL query params), CSV download, "Typical error ±x%" metric, plain-English model caption ("blend of the 2 best models", etc.)
- Alerts: CSV download of the monitored table
- Classify: "likely cheap/average/expensive — use as a shortlist" wording + per-class precision caption
- Map: CRASH markers in blue (previously fell through to NORMAL)
- KB (`app/data/kb/*.md`), README and this log refreshed to current numbers; robust phrasing so the chatbot doesn't drift monthly

## Earlier pass (2026-09) — A + B sequenced

### Part A — Data acquisition

1. **KAMIS product ID refresh** (`scripts/scrape_kamis.py`)
   - Added `dump_product_ids()` helper + `--dump-products` CLI flag — scrapes live `<select>` from search form
   - PRODUCTS dict expanded from 39 → 133 curated food items (grains, pulses, tubers, veg, fruits, meat, fish, dairy, oils, cash crops)
   - Site has ~193 total; skipped hides/fertilizer/cotton/obscure fish/numeric-placeholder rows
2. **Median monthly aggregation** (`scripts/scrape_kamis.py:235`)
   - `.mean()` → `.median()` at monthly aggregation to shrug off the 0.3% mislabeled-bag rows
3. **Rebuild**: `scrape_kamis` (cached most) → `build_chained_all` → `enrich_chain` → `rebuild_anomalies` → `rebuild_forecasts`

### Part B — Model lift

1. **Weighted top-2 ensemble** (`scripts/rebuild_forecasts.py::_fit_one`)
   - Ranks candidates by val MAPE, keeps top 2, weights by `1/mape`, applies to both point + interval
   - Only ensemble branch changed; single-model routing unchanged
2. **Log-transform for Prophet + SARIMA** — TRIED, REGRESSED, REVERTED
   - Caused Chronos to win val races it shouldn't; ensemble share dropped 54 → 34
   - Log wrapper functions left as dead code — deleted 2026-10
3. **Checkpointing** (`scripts/rebuild_forecasts.py`)
   - `CHECKPOINT = data/exports/.rebuild_checkpoint.pkl`, saved every 25 series (atomic rename), cleared on full run
   - Root-cause fix for the low-memory reaper killing rebuilds mid-flight (killed 3 runs previously)

## Verified findings from investigation

- **KAMIS "unit bug" was a red herring.** Cache is per-kg across all 5,616 xlsx files; the 168 inflated rows (0.3%) are upstream data-entry errors, not the systemic /Kg vs /Bag mislabel I first hypothesised. Median aggregation still helps for a different reason.
- **Source dedupe was not the leak.** Chain keeps 55,560/56,698 KAMIS rows (98%).
- **Real acquisition lever was dead product IDs.** Original `PRODUCTS` had 39 IDs, only 13 returned data. The rest were extinct or renumbered.

## Where we are on the ceiling question

**Not a hard ceiling.** Two remaining levers ranked by ROI:

1. ~~NAFIS scraper~~ — **dead**: nafis.go.ke no longer resolves (last archived 2021-01); it was a blog of image tables for 5 big markets already covered by FEWS.
2. **Chronos-Bolt-base swap** — 200MB download, 2× rebuild time, priors say +0.5-1 pt median MAPE. Cheap ceiling test once running.

Ruled out this pass:
- **Exogenous regressors into Prophet/SARIMA** — user's call: expected to hurt on this data
- **47 county portals** — 2-day dig for 0-100 mostly-redundant rows; bad ROI
- **KNBS bulk CPI** — national aggregates, not market-level, won't help per-market forecast
- **Log-transform target** — measured regression, reverted

## Files touched (2026-09 pass)

- `scripts/scrape_kamis.py` — PRODUCTS dict refreshed, `dump_product_ids()` helper + `--dump-products` flag, `.mean()` → `.median()`
- `scripts/rebuild_forecasts.py` — weighted top-2 ensemble, checkpointing, dead `_prophet_predict_log`/`_sarima_predict_log` wrappers to clean up
- `app/data/*.csv` — refreshed via rebuild
- `data/exports/*.csv` — refreshed
- `data/cleaned/bei_smart_forecasting_chained_all.csv` — widened

Uncommitted git state at time of writing: modified pipeline scripts + regenerated CSVs, untracked `scripts/scrape_kamis.py` (already tracked; the `??` was for a raw xlsx and `data/external/`).

## Next session — pick one

- **A. Full rebuild to land the maize merge** — `build_chained_all → enrich_chain → rebuild_anomalies → rebuild_forecasts` (or trigger the workflow); then re-check counts and commit. Also refreshes `app/data/bei_smart_forecasting_chained.csv`.
- **B. ~~NAFIS~~** — dead (see above). New-source review 2026-10-29 (scripts/experiments/exp6_new_sources.py): World Bank RTFP = copies of our FEWS/WFP + ML estimates (not integrated); FAO FPMA = FEWS mirror, integrated at lowest priority (adds Nairobi bread); NDMA = county averages upstream of FEWS (skip); KAMIS has nothing before 2021-05.
- **C. Try Chronos-Bolt-base** — `rebuild_forecasts.py --chronos-model amazon/chronos-bolt-base`, measure with `diff_mape.py`.
- Add the live Streamlit URL to README.

Done since last log: log-transform dead code deleted; mobile pass (CSS); monthly ingestion automated.

Parked from prior plans:
- Classifier swap
- Custom Prophet seasonalities
- Rolling-origin CV

## Key commands

```bash
# Refresh KAMIS product IDs (review + hand-edit PRODUCTS)
venv/Scripts/python.exe scripts/scrape_kamis.py --dump-products

# Full pipeline rebuild (CI does this monthly; manual: gh workflow run refresh-data.yml)
venv/Scripts/python.exe scripts/scrape_kamis.py --discover
venv/Scripts/python.exe scripts/build_chained_all.py
venv/Scripts/python.exe scripts/enrich_chain.py
venv/Scripts/python.exe scripts/rebuild_anomalies.py
venv/Scripts/python.exe scripts/rebuild_forecasts.py           # resumes from ckpt if killed
venv/Scripts/python.exe scripts/rebuild_forecasts.py --chronos-model amazon/chronos-bolt-base

# Measure lift
venv/Scripts/python.exe scripts/diff_mape.py

# Serve dashboard
venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

## 2026-10-29 (late) — new sources + goats
- **Goats (KES per head)**: FEWS `ea` rows kept (`PER_HEAD` in build_chained_all.py, `unit="HEAD"`); exp6: 20/22 county series served at median 6.2% MAPE. App/chatbot label via `app/src/units.py::unit_label` ("head"/"kichwa").
- **FEWS refresh**: workflow now re-downloads FEWS from fdw.fews.net each month (it was never refreshed; file had stopped at 2026-07).
- **FAO FPMA**: `scripts/ingest_fpma.py --refresh` in the workflow; `load_fpma()` at lowest SOURCE_PRIORITY.
- Workflow commit step does `git pull --rebase` before pushing so code pushes during a long rebuild don't reject the bot's push.
- Needs one full workflow run to put goats/bread/fresh FEWS into app/data.

## 2026-10-29 (evening) — pre-demo polish + repo tidy
- Front end: `st.navigation` router (`app/streamlit_app.py`) gives Home/Forecast/... titles and icons; landing moved to `app/home.py`. Regional coverage table counted forecast *rows* (Eastern "1,407") — now counts series (67). Alerts sorted by size, "x% above/below the normal range"; alert card shows spikes + crashes, distinct examples; dropped pointer to the empty Validation page.
- Anomaly bands span at least +/-10% of the median (`MIN_BAND`), so flat KAMIS series stop raising 5% "crashes" (takes effect next refresh).
- KAMIS CI scrape: `--months-back 3` + 60-min step cap (a clean runner re-scraped all of 2026, twice, for 4h40m+).
- Repo: four final notebooks numbered under `notebooks/eda|modeling/`, drafts and per-member notebooks in `notebooks/archive/`; one-off rainfall/RTFP scripts moved to `scripts/experiments/`; empty `src/` removed. README rewritten with a repo map; its "Live numbers" block is regenerated by `scripts/update_readme_stats.py` in every monthly run.

## 2026-11-15 — alerts, bands, sell page, weekly prices, SMS, new sources
- **Seasonal alert bands**: `rebuild_anomalies.py::seasonal_factors` removes a series' consistent calendar-month pattern before the IQR band (idea behind WFP ALPS). 42 flagged vs 39 without; 5 of 787 statuses change. `--demo` self-check.
- **80% forecast bands**: SARIMA band was 95% while Prophet/Chronos were 80%; all are 80% now (`BAND_LEVEL`). `calibrate_bands()` rescales bands from test-window scores. exp7: Chronos bands covered 91.3% (277 series); after calibration 79.8% on held-out series; all-model sample 90.4% -> 80.1%. **Takes effect at the next full rebuild.**
- **KAMIS cache bug fixed**: a month cached before it ended (or a future month) was never refetched (`_cache_final`). Local cache stopped at 2026-09-19; 297 cached months refetch on the next pull. KAMIS refused connections on 2026-11-15, so not re-pulled yet.
- **Where to sell** (`app/pages/6_Sell.py`, `app/src/sell.py`): latest wholesale price per market minus transport (slider, default KES 1.5/kg/100 km, an assumption). Most markets sit at county centres (`app/data/county_centroids.json`) — geocode towns if the page gets used.
- **Weekly prices**: `scripts/build_weekly.py` -> `app/data/kamis_weekly.csv`, "Latest week" line on Forecast; run by `kamis_local_push.ps1`. Only useful if that task runs weekly, and it shows nothing until a fresh pull.
- **SMS alerts**: `scripts/send_alerts.py` (Africa's Talking, EN/SW, dry run unless `--send`). Workflow step is a no-op until secrets `ALERT_SUBSCRIBERS`, `AT_USERNAME`, `AT_API_KEY` exist. No sign-up form, no WhatsApp.
- **KB**: `31_policy_events_timeline.md` (subsidies, duty waivers, import ban, fuel VAT; dates checked against press reports); 05 and 25 updated.
- **New sources reviewed**: FEWS has no current Uganda/Tanzania maize (UG ends 2015, TZ empty) -> WFP HDX `wfp_food_prices_uga/tza.csv` instead. KAMIS publishes daily Supply Volume (`experiments/build_kamis_volume.py`). County NDVI from HDX `ken-ndvi-subnational` (use level 2, county = first 5 chars of PCODE).
- **exp8 — Chronos-2 and covariates** (270 served series, zero-shot, last 12 obs; median / mean test MAPE; `exp8_chronos2_covariates.csv`):
  - Bolt-small 10.11 / 10.94 (baseline). Chronos-2 alone 9.60 / 11.17 — not a significant change (wins 132/270).
  - **Chronos-2 + cross-learning 8.92 / 10.01** (beats plain Chronos-2 on 166/270, p<0.001; short series 9.44 vs Bolt 10.44; maize 8.37 vs 10.22). Series grouped by test-start month so none sees another's future.
  - + county NDVI 9.50 / 10.52 (beats plain on 174/270, p<0.001). + macro (ONI, futures, FX) 9.46 / 10.71 (p=0.017). All three together 9.75 / 10.75 — no better than NDVI alone.
  - + Uganda/Tanzania maize prices: no effect (9.71 / 11.10, p=0.93; maize only p=0.17). + KAMIS supply volume (105 series): 11.55 vs 11.46 median, mean 11.76 vs 12.16, p=0.06 — not convincing.
  - Untested: cross-learning + NDVI together.
- **exp9 — global LightGBM** (`exp9_global_lgbm.csv`): alone it ties Bolt (10.10 / 11.25, wins 138/270). **Mean of Bolt + LightGBM 9.46 / 10.25**, beats Bolt on 179/270 (p<0.001); long series 8.39 vs 8.82, short 9.93 vs 10.44.
- **Wired into production and rebuilt (2026-11-15)**: `rebuild_forecasts.py` default Chronos is now `amazon/chronos-2` with cross-learning (batches share a forecast start date), and a global LightGBM is a fourth ensemble candidate (trained per cutoff date, so no leakage). Both are computed up front in `precompute()` and cached in `data/exports/.rebuild_pre.pkl`. Prophet/SARIMA are no longer fitted on short series (they could never be chosen), so the rebuild took about 45 minutes instead of hours. `--demo` asserts the leak rule.
  - Result: **283 served (was 270), median test MAPE 8.67% (was 9.7%)**; strategies chronos 170, ensemble 113. On the 267 series served before and after: median 9.72% -> 8.48%, 162 improved, 57 worse by more than 1 point (worst: Maize Kutus wholesale 12.7% -> 22.7%). 3 series dropped out, 16 added.
  - Bands: raw coverage 89% (chronos) / 93% (ensemble) on the test window, scaled to 80%.
  - CI installs lightgbm; first CI run downloads Chronos-2 (~480 MB) — watch that run's time and memory.
  - Not done: NDVI as a Chronos-2 covariate (helped in exp8; untested together with cross-learning).
  - Re-run after pulling the 2026-11-05 CI refresh (what is deployed): 283 served, median test MAPE 8.4%% (CI run on Bolt: 270, 9.8%%); on the 268 shared series 9.83%% -> 8.35%%, 160 improved, 53 worse by more than 1 point; chronos 169, ensemble 114.
