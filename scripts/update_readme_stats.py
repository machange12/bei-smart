"""Rewrite the live-numbers block in README.md from the app's data files.

Runs in the monthly GitHub Actions refresh right after the rebuild, so the
README never quotes stale counts. Only the text between the two markers is
replaced; everything else in the README is left alone.

    <!-- stats:start -->  ...generated table...  <!-- stats:end -->

Run:
    python scripts/update_readme_stats.py          # rewrite README.md
    python scripts/update_readme_stats.py --demo   # offline self-check, no write
"""
from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
APP_DATA = ROOT / "app" / "data"
START, END = "<!-- stats:start -->", "<!-- stats:end -->"
STRATEGY_NAMES = {"chronos": "Chronos", "ensemble": "blend of 2 models", "sarima": "SARIMA", "prophet": "Prophet"}


def stats_block(metrics: pd.DataFrame, anomalies: pd.DataFrame, history: pd.DataFrame) -> str:
    served = metrics[metrics["reliable"]]
    live = served[served["months_stale"] <= 6]
    mix = ", ".join(f"{STRATEGY_NAMES.get(k, k)} {v}" for k, v in served["strategy"].value_counts().items())
    status = anomalies["status"].value_counts()
    rows = [
        ("Forecast series served", f"{len(served):,} ({len(live):,} with data under 6 months old)"),
        ("Markets / commodities", f"{served['market'].nunique()} / {served['commodity'].nunique()}"),
        ("Regions covered", f"{served['region'].nunique()} of 8"),
        ("Median forecast error (12-month holdout)", f"{served['mape'].median():.1f}%"),
        ("Models serving", mix),
        ("Price alerts", f"{status.get('SPIKE', 0)} spikes, {status.get('CRASH', 0)} crashes "
                         f"across {len(anomalies):,} monitored series"),
        ("Prices observed up to", f"{pd.to_datetime(history['date']).max():%B %Y}"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (f"{START}\n_Updated automatically by the monthly refresh on "
            f"{pd.Timestamp.now():%Y-%m-%d}._\n\n| | |\n|---|---|\n{table}\n{END}")


def splice(readme: str, block: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise SystemExit(f"README has no {START} ... {END} markers")
    return pattern.sub(lambda _: block, readme)


def demo() -> None:
    metrics = pd.DataFrame({
        "commodity": ["Maize", "Beans", "Goats"], "market": ["Eldoret", "Kisumu", "Garissa"],
        "region": ["Rift Valley", "Nyanza", "North Eastern"], "mape": [8.0, 12.0, 30.0],
        "strategy": ["chronos", "ensemble", "chronos"], "reliable": [True, True, False],
        "months_stale": [0, 9, 0],
    })
    anomalies = pd.DataFrame({"status": ["SPIKE", "NORMAL", "CRASH", "NORMAL"]})
    history = pd.DataFrame({"date": ["2026-08-01", "2026-09-01"]})
    block = stats_block(metrics, anomalies, history)
    assert "| Forecast series served | 2 (1 with data under 6 months old) |" in block, block
    assert "| Median forecast error (12-month holdout) | 10.0% |" in block
    assert "1 spikes, 1 crashes across 4" in block and "September 2026" in block
    out = splice(f"intro\n{START}\nSTALE-BLOCK\n{END}\noutro", block)
    assert out.startswith("intro\n") and out.endswith("\noutro") and "STALE-BLOCK" not in out
    print("[demo] OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        demo()
    else:
        block = stats_block(pd.read_csv(APP_DATA / "model_metrics.csv"),
                            pd.read_csv(APP_DATA / "price_anomalies.csv"),
                            pd.read_csv(APP_DATA / "bei_smart_forecasting_chained.csv", usecols=["date"]))
        README.write_text(splice(README.read_text(encoding="utf-8"), block), encoding="utf-8")
        print(f"[readme] stats block updated")
