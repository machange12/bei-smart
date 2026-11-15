"""SMS alert sign-up: the app talks to two Supabase database functions and nothing else.

The public key the app holds can only call subscribe_sms / unsubscribe_sms; it
cannot read the subscribers table (row-level security, no policies), so one
visitor can never see another's number. scripts/send_alerts.py reads the table
with a separate secret key that never reaches the app.

Config: SUPABASE_URL and SUPABASE_KEY (publishable) default to the bei-smart project below;
Streamlit secrets or environment variables override them.
"""

from __future__ import annotations

import os
import re

import requests
import streamlit as st


DEFAULTS = {
    "SUPABASE_URL": "https://prphmchdrrqthqpzbnue.supabase.co",
    # Publishable (public) key: safe in the repo. It cannot read the subscribers table.
    "SUPABASE_KEY": "sb_publishable_MOEod6XpGd2dF4tGZQc7ww_5OFcdtot",
}


def _cfg(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file at all
        pass
    return os.environ.get(name, DEFAULTS.get(name, ""))


def configured() -> bool:
    return bool(_cfg("SUPABASE_URL") and _cfg("SUPABASE_KEY"))


def normalise_phone(raw: str) -> str | None:
    """'0712 345 678', '712345678', '+254 712-345-678' -> '+254712345678'; anything else -> None."""
    digits = re.sub(r"[\s\-()]", "", raw or "")
    m = re.fullmatch(r"(?:\+?254|0)?([17]\d{8})", digits)
    return f"+254{m.group(1)}" if m else None


def _rpc(fn: str, payload: dict) -> bool:
    key = _cfg("SUPABASE_KEY")
    try:
        r = requests.post(f"{_cfg('SUPABASE_URL').rstrip('/')}/rest/v1/rpc/{fn}", json=payload, timeout=15,
                          headers={"apikey": key, "Authorization": f"Bearer {key}"})
        return r.ok
    except requests.RequestException:
        return False


def subscribe(phone: str, commodities: list[str], markets: list[str], lang: str) -> bool:
    return _rpc("subscribe_sms", {"p_phone": phone, "p_commodities": commodities,
                                  "p_markets": markets, "p_lang": lang})


def unsubscribe(phone: str) -> bool:
    return _rpc("unsubscribe_sms", {"p_phone": phone})


if __name__ == "__main__":
    for raw in ("0712 345 678", "712345678", "+254 712-345-678", "254112345678"):
        assert normalise_phone(raw) in ("+254712345678", "+254112345678"), raw
    for raw in ("", "12345", "0812345678", "+255712345678", "07123456789"):
        assert normalise_phone(raw) is None, raw
    print("subscribe ok")
