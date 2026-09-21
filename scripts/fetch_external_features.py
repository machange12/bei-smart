#!/usr/bin/env python3
"""
fetch_external_features.py  (v2 — fixed sources)
─────────────────────────────────────────────────────────────────────────
Bei Smart — External Features Pipeline

Fixes from v1:
  - ONI parser hardened (shows raw file lines for debug)
  - Commodities: FRED direct CSV (no auth, replaces dead IMF endpoint)
  - KES/USD: Yahoo Finance, World Bank annual fallback (replaces dead IMF IFS)
  - EPRA: manual as before

Sources
-------
1. NOAA ONI          → oni.csv                   [auto]
2. FRED (St. Louis)  → world_bank_commodities.csv [auto - maize/wheat/palm oil]
3. Yahoo Finance     → fx_kesusd.csv              [auto - KESUSD=X monthly]
4. EPRA              → epra_fuel.csv              [MANUAL — see below]

Combined → data/external/combined_external.csv

EPRA Manual Step
─────────────────
1. https://www.epra.go.ke/pump-prices/
2. Download the Excel file
3. Save as: data/raw/epra_pump_prices.xlsx
   Script auto-parses it when found.
"""

import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data/external")
RAW_DIR    = Path("data/raw")
START_DATE = "2015-01-01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def to_monthly(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


# ── 1. NOAA ONI ──────────────────────────────────────────────────────────────
def fetch_oni() -> pd.DataFrame:
    season_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    urls = [
        "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "https://origin.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    ]
    for url in urls:
        try:
            log(f"ONI ── {url}")
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            log(f"ONI ── {len(lines)} lines received")
            log(f"ONI ── line[0]: {lines[0]}")
            log(f"ONI ── line[1]: {lines[1] if len(lines) > 1 else 'n/a'}")

            rows = []
            for line in lines:
                parts = line.split()
                # New NOAA format: SEAS YR TOTAL ANOM (4 cols)
                # Old NOAA format: SEAS YR TOTAL CLIM ANOM TOTAL CLIM ANOM (8 cols)
                if len(parts) < 4 or parts[0] not in season_map:
                    continue
                try:
                    year  = int(parts[1])
                    month = season_map[parts[0]]
                    # ANOM: last column in both formats; also try idx 4 for old format
                    anom = None
                    for idx in [-1, 4, 3]:
                        try:
                            v = float(parts[idx])
                            if -5 < v < 5:   # sanity: ONI anomaly is almost always -3 to +3
                                anom = v
                                break
                        except (ValueError, IndexError):
                            continue
                    if anom is None:
                        continue
                    rows.append({
                        "date": pd.Timestamp(year=year, month=month, day=1),
                        "oni_anom": anom
                    })
                except Exception:
                    continue

            if not rows:
                log(f"ONI ── zero rows parsed from {url}, trying next...")
                continue

            df = (pd.DataFrame(rows)
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True))
            df = df[df["date"] >= START_DATE].reset_index(drop=True)
            log(f"ONI ── {len(df)} rows  "
                f"({df['date'].min().date()} → {df['date'].max().date()})")
            return df

        except Exception as e:
            log(f"ONI ── failed: {e}")

    raise RuntimeError("ONI: all URLs failed")


# ── 2. Yahoo Finance — Commodity Prices ─────────────────────────────────────
def fetch_commodities() -> pd.DataFrame:
    """
    Commodity price proxies via Yahoo Finance (same endpoint that works for FX).

    Tickers:
      ZC=F   CBOT Corn futures       (USD/bushel) → converted to USD/mt
      ZW=F   CBOT Wheat futures      (USD/bushel) → converted to USD/mt
      FCPO.MY Crude Palm Oil futures (MYR/mt)     → kept in MYR (price signal only)

    Note: futures prices, not World Bank spot prices, but capture the same
    global commodity market signal and work well as model features.
    """
    log("Commodities ── Yahoo Finance futures (ZC=F, ZW=F, FCPO.MY)...")

    # (ticker, output_col, unit_multiplier, note)
    tickers = [
        ("ZC=F",    "corn_usd_bushel",   1.0,    "CBOT corn USD/bushel"),
        ("ZW=F",    "wheat_usd_bushel",  1.0,    "CBOT wheat USD/bushel"),
        ("FCPO.MY", "palm_oil_myr_mt",   1.0,    "Bursa palm oil MYR/mt"),
    ]

    dfs = []
    for ticker, col, mult, note in tickers:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1mo&range=12y&includeAdjustedClose=false"
        )
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            data   = r.json()
            result = data["chart"]["result"][0]
            ts     = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]

            df = pd.DataFrame({
                "date": pd.to_datetime(ts, unit="s"),
                col:    [c * mult if c else None for c in closes]
            })
            df["date"] = to_monthly(df["date"])
            df[col]    = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            dfs.append(df)
            log(f"  ✓ {ticker} ({note}): {len(df)} rows")
        except Exception as e:
            log(f"  ✗ {ticker}: {e}")

    if not dfs:
        log("Commodities ── all Yahoo Finance tickers failed")
        return pd.DataFrame(columns=["date","corn_usd_bushel","wheat_usd_bushel","palm_oil_myr_mt"])

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="date", how="outer")

    return (merged[merged["date"] >= START_DATE]
              .sort_values("date")
              .reset_index(drop=True))


# ── 3. Yahoo Finance — KES/USD ───────────────────────────────────────────────
def fetch_fx() -> pd.DataFrame:
    log("FX ── Yahoo Finance KESUSD=X (monthly)...")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/KESUSD=X"
        "?interval=1mo&range=12y&includeAdjustedClose=false"
    )
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        r.raise_for_status()
        data   = r.json()
        result = data["chart"]["result"][0]
        ts     = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]

        df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s"), "kes_per_usd": closes})
        df["date"]        = to_monthly(df["date"])
        df["kes_per_usd"] = pd.to_numeric(df["kes_per_usd"], errors="coerce")
        df = (df.dropna(subset=["kes_per_usd"])
                .query("date >= @START_DATE")
                .sort_values("date")
                .reset_index(drop=True))
        log(f"FX ── {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    except Exception as e:
        log(f"FX ── Yahoo Finance failed: {e}")
        return _fx_worldbank_fallback()


def _fx_worldbank_fallback() -> pd.DataFrame:
    """World Bank official exchange rate (annual) — interpolated to monthly."""
    log("FX ── World Bank annual rate fallback (interpolated to monthly)...")
    url = (
        "https://api.worldbank.org/v2/country/KE/indicator/PA.NUS.FCRF"
        "?format=json&per_page=50&mrv=50"
    )
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        r.raise_for_status()
        data = r.json()[1]
        rows = [
            {"date": pd.Timestamp(year=int(d["date"]), month=7, day=1),
             "kes_per_usd": float(d["value"])}
            for d in data if d["value"] is not None
        ]
        df = pd.DataFrame(rows).sort_values("date")
        monthly = pd.date_range(df["date"].min(), pd.Timestamp.today(), freq="MS")
        df = (df.set_index("date")
                .reindex(monthly)
                .interpolate("linear")
                .reset_index()
                .rename(columns={"index": "date"}))
        df = df[df["date"] >= START_DATE].reset_index(drop=True)
        log(f"FX ── World Bank annual → {len(df)} monthly rows (interpolated)")
        return df
    except Exception as e:
        log(f"FX ── World Bank fallback failed: {e}")
        # Last resort: check for manual CBK file
        cbk = RAW_DIR / "cbk_fx.csv"
        if cbk.exists():
            df = pd.read_csv(cbk, parse_dates=["date"])
            df["date"] = to_monthly(df["date"])
            log(f"FX ── loaded {len(df)} rows from manual CBK file")
            return df
        log("FX ── manual: https://www.centralbank.go.ke/forex-exchange-rates/")
        log(f"       Save as data/raw/cbk_fx.csv  columns: [date, kes_per_usd]")
        return pd.DataFrame(columns=["date", "kes_per_usd"])


# ── 4. EPRA Fuel (manual + parser) ──────────────────────────────────────────
def parse_epra() -> pd.DataFrame:
    path = RAW_DIR / "epra_pump_prices.xlsx"
    if not path.exists():
        log("EPRA ── not found (manual download needed)")
        log("  → https://www.epra.go.ke/pump-prices/")
        log(f"  → Save as: {path.resolve()}")
        return pd.DataFrame(columns=["date","petrol_nairobi_ksh_l","diesel_nairobi_ksh_l"])

    log("EPRA ── parsing pump prices Excel...")
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        log(f"  Sheets: {xl.sheet_names}")
        best = None
        for sheet in xl.sheet_names:
            raw = xl.parse(sheet)
            raw.columns = [str(c).lower().strip() for c in raw.columns]
            date_col   = next((c for c in raw.columns if any(x in c for x in
                               ["date","month","period","effective"])), None)
            petrol_col = next((c for c in raw.columns if any(x in c for x in
                               ["petrol","super","pms"])), None)
            diesel_col = next((c for c in raw.columns if any(x in c for x in
                               ["diesel","ago"])), None)
            if not date_col or not (petrol_col or diesel_col):
                continue
            keep = {date_col: "date"}
            if petrol_col: keep[petrol_col] = "petrol_nairobi_ksh_l"
            if diesel_col: keep[diesel_col] = "diesel_nairobi_ksh_l"
            df = raw[list(keep)].rename(columns=keep).copy()
            df["date"] = to_monthly(pd.to_datetime(df["date"], errors="coerce"))
            df = df.dropna(subset=["date"])
            if best is None or len(df) > len(best):
                best = df
        if best is not None:
            best = best.query("date >= @START_DATE").sort_values("date").reset_index(drop=True)
            log(f"EPRA ── {len(best)} rows")
            return best
        log("EPRA ── no matching column layout found")
    except Exception as e:
        log(f"EPRA ── parse error: {e}")
    return pd.DataFrame(columns=["date","petrol_nairobi_ksh_l","diesel_nairobi_ksh_l"])


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_all(dfs: dict) -> pd.DataFrame:
    combined = None
    for name, df in dfs.items():
        if df.empty:
            log(f"Merge ── skipping {name} (empty)")
            continue
        df = df.copy()
        df["date"] = to_monthly(df["date"])
        combined = df if combined is None else combined.merge(df, on="date", how="outer")
    return pd.DataFrame() if combined is None else combined.sort_values("date").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("═" * 60)
    log("Bei Smart — External Features Pipeline  (v2)")
    log(f"Output : {OUTPUT_DIR.resolve()}")
    log(f"Range  : {START_DATE} → today")
    log("═" * 60)

    tasks = [
        ("oni",         fetch_oni,         "oni.csv"),
        ("commodities", fetch_commodities, "world_bank_commodities.csv"),
        ("fx",          fetch_fx,          "fx_kesusd.csv"),
    ]

    results = {}
    for name, fn, fname in tasks:
        log("")
        try:
            df = fn()
            df.to_csv(OUTPUT_DIR / fname, index=False)
            results[name] = df
            log(f"✓ {fname}")
        except Exception as e:
            log(f"✗ {name} failed: {e}")
            results[name] = pd.DataFrame()

    log("")
    df = parse_epra()
    if not df.empty:
        df.to_csv(OUTPUT_DIR / "epra_fuel.csv", index=False)
        results["epra"] = df
        log("✓ epra_fuel.csv")
    else:
        results["epra"] = pd.DataFrame()

    log("")
    combined = merge_all(results)
    if not combined.empty:
        combined.to_csv(OUTPUT_DIR / "combined_external.csv", index=False)
        log("═" * 60)
        log(f"✓ combined_external.csv — {len(combined)} rows × {len(combined.columns)} cols")
        log(f"  Range   : {combined['date'].min().date()} → {combined['date'].max().date()}")
        log(f"  Columns : {list(combined.columns)}")
        log("\n  Coverage (non-null %):")
        for col in combined.columns:
            if col == "date": continue
            pct = combined[col].notna().mean() * 100
            log(f"    {'✓' if pct > 80 else '⚠'} {col}: {pct:.0f}%")
    else:
        log("✗ No data combined — check errors above")

    log("═" * 60)
    log("Done.")

if __name__ == "__main__":
    main()