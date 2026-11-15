"""Text this month's price alerts to subscribers by SMS (Africa's Talking).

Dry run by default: prints every message and sends nothing.

    python scripts/send_alerts.py                       # preview
    python scripts/send_alerts.py --send                # really send
    python scripts/send_alerts.py --demo                # self-check

Subscribers come from, in order: --subscribers <file>; the Supabase `subscribers`
table (people who signed up on the Alerts page) when SUPABASE_URL and
SUPABASE_SECRET_KEY are set; else the ALERT_SUBSCRIBERS environment variable.
Phone numbers never go in the repo. The file / variable is a JSON list:
    [{"phone": "+254712345678", "commodities": ["Maize"], "markets": ["Kitale"], "lang": "sw"}]
An empty or missing "commodities" / "markets" means all of them; "lang" is "en" or "sw".

Sending needs AT_USERNAME and AT_API_KEY (username "sandbox" uses the sandbox).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANOMALIES = ROOT / "app" / "data" / "price_anomalies.csv"
APP_URL = "bei-smart.streamlit.app"
PHONE_RE = re.compile(r"^\+254[17]\d{8}$")
MAX_ALERTS = 3       # per text, biggest first -- keeps it to one or two SMS segments
FRESH_MONTHS = 2     # an alert on a price older than this is not news

TEXT = {
    "en": {"SPIKE": "{c} {m} {pt} KES {p:.0f}/{u}, {d:.0f}% above normal",
           "CRASH": "{c} {m} {pt} KES {p:.0f}/{u}, {d:.0f}% below normal",
           "Wholesale": "wholesale", "Retail": "retail", "head": "head", "more": "More"},
    "sw": {"SPIKE": "{c} {m} {pt} KES {p:.0f}/{u}, {d:.0f}% juu ya kawaida",
           "CRASH": "{c} {m} {pt} KES {p:.0f}/{u}, {d:.0f}% chini ya kawaida",
           "Wholesale": "jumla", "Retail": "rejareja", "head": "kichwa", "more": "Zaidi"},
}


def active_alerts(anomalies: pd.DataFrame) -> pd.DataFrame:
    a = anomalies[anomalies["status"] != "NORMAL"].copy()
    a["latest_date"] = pd.to_datetime(a["latest_date"])
    newest = pd.to_datetime(anomalies["latest_date"]).max()
    a = a[a["latest_date"] >= newest - pd.DateOffset(months=FRESH_MONTHS)]
    return a.sort_values("deviation_pct", ascending=False)


def message_for(sub: dict, alerts: pd.DataFrame) -> str | None:
    """The text for one subscriber, or None when nothing they follow is flagged."""
    a = alerts
    if sub.get("commodities"):
        a = a[a["commodity"].str.lower().isin([c.lower() for c in sub["commodities"]])]
    if sub.get("markets"):
        a = a[a["market"].str.lower().isin([m.lower() for m in sub["markets"]])]
    if a.empty:
        return None
    t = TEXT.get(sub.get("lang", "en"), TEXT["en"])
    lines = [t[r.status].format(c=r.commodity, m=r.market, pt=t[r.pricetype], p=r.latest_price,
                                u=t["head"] if r.commodity == "Goats" else "kg", d=r.deviation_pct)
             for r in a.head(MAX_ALERTS).itertuples()]
    extra = f" (+{len(a) - MAX_ALERTS})" if len(a) > MAX_ALERTS else ""
    return f"AgriPulse: {'; '.join(lines)}{extra}. {t['more']}: {APP_URL}"


def send_sms(phone: str, text: str, username: str, api_key: str) -> str:
    import requests
    host = "api.sandbox.africastalking.com" if username == "sandbox" else "api.africastalking.com"
    r = requests.post(f"https://{host}/version1/messaging",
                      headers={"apiKey": api_key, "Accept": "application/json"},
                      data={"username": username, "to": phone, "message": text}, timeout=30)
    r.raise_for_status()
    return r.json()["SMSMessageData"]["Recipients"][0]["status"]


def load_subscribers(path: str | None) -> list[dict]:
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
    if path:
        return json.loads(pathlib.Path(path).read_text())
    if url and key:
        import requests
        r = requests.get(f"{url.rstrip('/')}/rest/v1/subscribers", timeout=30,
                         params={"active": "eq.true", "select": "phone,commodities,markets,lang"},
                         headers={"apikey": key, "Authorization": f"Bearer {key}"})
        r.raise_for_status()
        return r.json()
    raw = os.environ.get("ALERT_SUBSCRIBERS", "")
    return json.loads(raw) if raw.strip() else []


def main(args) -> int:
    subs = load_subscribers(args.subscribers)
    if len(subs) > args.max:  # a flood of sign-ups must not become a surprise SMS bill
        print(f"[alerts] {len(subs)} subscribers exceeds --max {args.max}; sending to the first {args.max}")
        subs = subs[:args.max]
    if not subs:
        print("[alerts] no subscribers configured; nothing to do")
        return 0
    username, api_key = os.environ.get("AT_USERNAME"), os.environ.get("AT_API_KEY")
    if args.send and not (username and api_key):
        print("[alerts] --send needs AT_USERNAME and AT_API_KEY", file=sys.stderr)
        return 1
    alerts = active_alerts(pd.read_csv(ANOMALIES))
    sent = failed = 0
    for sub in subs:
        phone = str(sub.get("phone", ""))
        tail = phone[-3:]  # logs are public in CI: never print a full number
        if not PHONE_RE.match(phone):
            print(f"[skip] ...{tail}: not a +254 mobile number")
            continue
        text = message_for(sub, alerts)
        if text is None:
            print(f"[quiet] ...{tail}: nothing flagged")
            continue
        if not args.send:
            print(f"[dry-run] ...{tail} ({len(text)} chars): {text}")
            continue
        try:
            print(f"[sent] ...{tail}: {send_sms(phone, text, username, api_key)}")
            sent += 1
        except Exception as e:  # one bad number must not stop the rest
            print(f"[fail] ...{tail}: {type(e).__name__}", file=sys.stderr)
            failed += 1
    print(f"[alerts] {len(alerts)} active alerts, {len(subs)} subscribers, sent {sent}, failed {failed}")
    return 1 if failed and not sent else 0


def demo() -> None:
    an = pd.DataFrame({
        "commodity": ["Maize", "Beans", "Maize", "Goats"], "market": ["Kitale", "Kitale", "Busia", "Garissa"],
        "pricetype": ["Wholesale", "Retail", "Retail", "Retail"],
        "latest_date": ["2026-10-01", "2026-10-01", "2026-10-01", "2025-01-01"],
        "latest_price": [62.0, 150.0, 55.0, 9000.0], "status": ["SPIKE", "NORMAL", "CRASH", "SPIKE"],
        "deviation_pct": [18.0, 0.0, 12.0, 40.0],
    })
    al = active_alerts(an)
    assert al["market"].tolist() == ["Kitale", "Busia"]          # NORMAL and the stale Goats row dropped
    en = message_for({"phone": "x", "markets": ["kitale"]}, al)
    assert en == f"AgriPulse: Maize Kitale wholesale KES 62/kg, 18% above normal. More: {APP_URL}", en
    assert "12% chini ya kawaida" in message_for({"lang": "sw", "commodities": ["Maize"]}, al)
    assert message_for({"commodities": ["Rice"]}, al) is None
    assert PHONE_RE.match("+254712345678") and not PHONE_RE.match("0712345678")
    print("demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually send (default: print only)")
    ap.add_argument("--subscribers", help="JSON file; default is the ALERT_SUBSCRIBERS env var")
    ap.add_argument("--max", type=int, default=500, help="most subscribers to text in one run")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        sys.exit(main(a))
