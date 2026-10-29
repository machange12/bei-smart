# bei-smart — Progress Log

Last updated: 2026-09-22

## Current state

**Data pipeline:**
- Chain: `data/cleaned/bei_smart_forecasting_chained_all.csv` — 123,998 rows, **51 commodities**, **468 markets**, 1,692 series ≥24 obs
- Sources stitched: WFP, WFP_SUPP, FEWS, KAMIS, KNBS (priority in that order)
- KAMIS scraper: 133 curated food products, 46 landing data, 250 markets
- External features merged into chain (oni, corn/wheat futures, KES/USD, EPRA fuel) — **present but unused by forecaster**

**Forecasts (`app/data/`):**
- **318 reliable series** (up from 248 pre-widen)
- Median MAPE: **9.50%** overall / **7.23% on series shared with prior baseline** (−2.19 pts, 33/41 improved)
- Strategy mix: chronos 170, ensemble 52, prophet 48, sarima 48
- Ensemble = **weighted top-2 by 1/val_mape** (replaced unweighted mean)
- Anomalies: 961 rows, 145 markets coloured, 45 SPIKE + 4 CRASH

**Dashboard:** Streamlit at `app/streamlit_app.py`, points at `app/data/` CSVs. Serves widened data.

## What was done this pass (A + B sequenced)

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
   - Log wrapper functions (`_prophet_predict_log`, `_sarima_predict_log`) left in the file as dead code (~10 lines) — could delete
3. **Checkpointing** (`scripts/rebuild_forecasts.py`)
   - `CHECKPOINT = data/exports/.rebuild_checkpoint.pkl`, saved every 25 series (atomic rename), cleared on full run
   - Root-cause fix for the low-memory reaper killing rebuilds mid-flight (killed 3 runs previously)

## Verified findings from investigation

- **KAMIS "unit bug" was a red herring.** Cache is per-kg across all 5,616 xlsx files; the 168 inflated rows (0.3%) are upstream data-entry errors, not the systemic /Kg vs /Bag mislabel I first hypothesised. Median aggregation still helps for a different reason.
- **Source dedupe was not the leak.** Chain keeps 55,560/56,698 KAMIS rows (98%).
- **Real acquisition lever was dead product IDs.** Original `PRODUCTS` had 39 IDs, only 13 returned data. The rest were extinct or renumbered.

## Where we are on the ceiling question

**Not a hard ceiling.** Two remaining levers ranked by ROI:

1. **NAFIS scraper** (nafis.go.ke) — half-day work, expected +20-50 new series (mostly fresh produce, some livestock). Structurally similar to KAMIS. **Best next dig.**
2. **Chronos-Bolt-base swap** — 200MB download, 2× rebuild time, priors say +0.5-1 pt median MAPE. Cheap ceiling test once running.

Ruled out this pass:
- **Exogenous regressors into Prophet/SARIMA** — user's call: expected to hurt on this data
- **47 county portals** — 2-day dig for 0-100 mostly-redundant rows; bad ROI
- **KNBS bulk CPI** — national aggregates, not market-level, won't help per-market forecast
- **Log-transform target** — measured regression, reverted

## Files touched this pass

- `scripts/scrape_kamis.py` — PRODUCTS dict refreshed, `dump_product_ids()` helper + `--dump-products` flag, `.mean()` → `.median()`
- `scripts/rebuild_forecasts.py` — weighted top-2 ensemble, checkpointing, dead `_prophet_predict_log`/`_sarima_predict_log` wrappers to clean up
- `app/data/*.csv` — refreshed via rebuild
- `data/exports/*.csv` — refreshed
- `data/cleaned/bei_smart_forecasting_chained_all.csv` — widened

Uncommitted git state at time of writing: modified pipeline scripts + regenerated CSVs, untracked `scripts/scrape_kamis.py` (already tracked; the `??` was for a raw xlsx and `data/external/`).

## Next session — pick one

- **A. Ship NAFIS scraper** — model on `scrape_kamis.py`, cache per (commodity, month), emit long-format rows in the chain schema. Then re-run `build_chained_all → enrich_chain → rebuild_anomalies → rebuild_forecasts`.
- **B. Try Chronos-Bolt-base** — swap the model ID in `_chronos_predict`, download weights, re-run rebuild once. Measure with `diff_mape.py`.
- **C. Delete the log-transform dead code** in `rebuild_forecasts.py` and commit the pass cleanly first.

Parked from prior plans (not this session's focus):
- Mobile pass (Streamlit phone-usable, plan drafted)
- Classifier swap
- Custom Prophet seasonalities
- Rolling-origin CV

## Key commands

```bash
# Refresh KAMIS product IDs (review + hand-edit PRODUCTS)
venv/Scripts/python.exe scripts/scrape_kamis.py --dump-products

# Full pipeline rebuild
venv/Scripts/python.exe scripts/scrape_kamis.py
venv/Scripts/python.exe scripts/build_chained_all.py
venv/Scripts/python.exe scripts/enrich_chain.py
venv/Scripts/python.exe scripts/rebuild_anomalies.py
venv/Scripts/python.exe scripts/rebuild_forecasts.py           # resumes from ckpt if killed

# Measure lift
venv/Scripts/python.exe scripts/diff_mape.py

# Serve dashboard
venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```
