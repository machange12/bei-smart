"""Stitch WFP + FEWS + KAMIS into one long-format monthly modelling table.

Mirrors the section-18 chain logic in `notebooks/modeling/forecast_model_clean.ipynb`
that produced `data/cleaned/bei_smart_forecasting_chained.csv` for four products,
and applies it across every commodity in every source.

Output schema matches the existing chained CSV so the pipeline notebook picks it
up with a single INPUT_PATH change:

    commodity, market, pricetype, date, price_per_kg, source, admin1,
    latitude, longitude, category, unit

Rainfall/GPR/diesel columns are omitted here — they get merged in downstream
(and they've proven neutral/negative for the forecast anyway, per the pipeline
notebook's regressor experiments).

Source priority on month conflicts (highest wins): FEWS > WFP > KAMIS.
Same ordering the existing chain uses.

Run:
    python scripts/build_chained_all.py           # writes bei_smart_forecasting_chained_all.csv
    python scripts/build_chained_all.py --demo    # tiny self-check, no write
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WFP = ROOT / "data" / "raw" / "wfp_food_prices_ken.csv"
FEWS = ROOT / "data" / "raw" / "fews_kenya_full.csv"
KAMIS = ROOT / "data" / "raw" / "kamis_prices.csv"
OUT = ROOT / "data" / "cleaned" / "bei_smart_forecasting_chained_all.csv"

SCHEMA = [
    "commodity", "market", "pricetype", "date", "price_per_kg",
    "source", "admin1", "latitude", "longitude", "category", "unit",
]

# Priority: earlier = wins on conflict. Matches existing chained CSV where
# a (commodity, market, pricetype, date) row's `source` reflects who won.
SOURCE_PRIORITY = ["WFP_SUPP", "FEWS", "WFP", "KAMIS", "KNBS"]

# WFP ships some prices as "per 90 KG bag" or "per 50 KG bag" — normalize to
# KES/KG. Litres left alone (oils/kerosene use volume and don't compare to
# solid staples; the pipeline gate filters them if under 60 obs anyway).
_UNIT_TO_KG = {
    "KG": 1.0, "Kg": 1.0, "kg": 1.0,
    "90 KG": 90.0, "50 KG": 50.0, "100 KG": 100.0, "110 KG": 110.0,
    "25 KG": 25.0, "5 KG": 5.0, "2 KG": 2.0,
    # Added: KG-denominated bags previously dropped (Cabbage 126 KG, Onions 13 KG,
    # scattered 64 KG entries). Straight numeric conversion.
    "126 KG": 126.0, "64 KG": 64.0, "13 KG": 13.0,
}

# Sources spell the same commodity many ways. This map is the single canonical
# collapse: WFP + FEWS + KAMIS + Wholesale supplement + KNBS retail all funnel
# into one string per commodity here. Ponytail: normalise the *input* once so
# every downstream group/gate/model works.
#
# Two prior bugs fixed here:
#  - `Maize flour` had duplicate keys mapping to different canonical names
#    (dict literal, later wins) which swapped Maize flour <-> Maize meal.
#  - Case-only variants (`Maize Flour`, `Sweet Potatoes`, `Millet (Finger)`)
#    passed through as separate commodities.
_COMMODITY_ALIAS = {
    # ---- Beans ----
    "Beans": "Beans",
    "Beans (dry)": "Beans (dry)",
    "Beans (mixed)": "Beans (mixed)",
    "Mixed beans": "Beans (mixed)",
    "Beans (rosecoco)": "Beans (rosecoco)",
    "Beans (Rosecoco)": "Beans (rosecoco)",
    "Beans Rosecoco": "Beans (rosecoco)",
    "Beans (yellow)": "Beans (yellow)",
    "Beans (Yellow-Green)": "Beans (yellow)",
    "Beans (kidney)": "Beans (kidney)",
    "Beans (dolichos)": "Beans (dolichos)",
    "Beans (mung)": "Beans (mung)",
    # Named varieties from the wholesale supplement — same basket the WFP
    # "Beans (mixed)" price already averages across, so fold them in.
    "Beans (Canadian wonder)": "Beans (mixed)",
    "Beans (Mwezi Moja)": "Beans (mixed)",
    "Beans (Mwitemania)": "Beans (mixed)",
    "Beans Red Haricot (Wairimu)": "Beans (mixed)",
    # ---- Cowpeas / peas / grams ----
    "Cowpeas": "Cowpeas",
    "Cowpeas (dry)": "Cowpeas",
    "Cowpeas (Red)": "Cowpeas",
    "Cowpea leaves": "Cowpea leaves",
    "Green grams": "Green grams",
    "Gram (Green)": "Green grams",
    "Dry peas": "Peas (dry)",
    "Pigeon peas": "Pigeon peas",
    "Pigeon peas (dry)": "Pigeon peas",
    # ---- Maize ----
    "Maize": "Maize",
    "Dry maize": "Maize",
    "Dry Maize": "Maize",
    "Maize (white)": "Maize (white)",
    "Maize (white, dry)": "Maize (white)",
    "Maize Grain (White)": "Maize (white)",
    # Flour vs meal: keep them SEPARATE. Meal = coarsely ground, flour = fine.
    "Maize flour": "Maize flour",
    "Maize Flour": "Maize flour",
    "Maize flour (white)": "Maize flour",
    "Maize meal": "Maize meal",
    "Maize Meal": "Maize meal",
    # ---- Sorghum / millet / wheat / rice ----
    "Sorghum": "Sorghum",
    "Sorghum (red)": "Sorghum",
    "Sorghum (Red)": "Sorghum",
    "Red sorghum": "Sorghum",
    "Red Sorghum": "Sorghum",
    "Sorghum (white)": "Sorghum (white)",
    "White Sorghum": "Sorghum (white)",
    "Millet": "Millet",
    "Millet (finger)": "Millet",
    "Millet (Finger)": "Millet",
    "Wheat": "Wheat",
    "Wheat Grain": "Wheat",
    "Wheat flour": "Wheat flour",
    "Rice": "Rice",
    "Rice (aromatic)": "Rice",
    "Rice (Low Grade)": "Rice",
    "Rice (imported)": "Rice (imported)",
    "Rice (imported, Pakistan)": "Rice (imported)",
    "Paddy rice": "Rice (paddy)",
    # ---- Tubers ----
    "Potatoes (Irish)": "Potatoes (Irish)",
    "Potatoes (Irish, white)": "Potatoes (Irish)",
    "Potatoes (Irish, red)": "Potatoes (Irish)",
    "Irish potatoes": "Potatoes (Irish)",
    "Potato (White)": "Potatoes (Irish)",
    "White Irish Potatoes": "Potatoes (Irish)",
    "Red Irish potato": "Potatoes (Irish)",
    "Cassava": "Cassava",
    "Sweet potatoes": "Sweet potatoes",
    "Sweet Potatoes": "Sweet potatoes",
    # ---- Vegetables / fruit ----
    "Tomatoes": "Tomatoes",
    "Onions": "Onions",
    "Onions (dry)": "Onions",
    "Onions (red)": "Onions",
    "Cabbages": "Cabbages",
    "Cabbage": "Cabbages",
    "Kales (sukuma)": "Kales",
    "Kale": "Kales",
    "Spinach": "Spinach",
    "Carrots": "Carrots",
    "Bananas": "Bananas",
    "Oranges": "Oranges",
    "Mangoes": "Mangoes",
    "Avocados": "Avocados",
    "Pineapples": "Pineapples",
    "Passion fruits": "Passion fruit",
    "Lemons": "Lemons",
    "Pawpaw": "Pawpaw",
    "Watermelon": "Watermelon",
    # ---- Groundnuts / sugar / oil ----
    "Groundnuts": "Groundnuts",
    "Sugar": "Sugar",
    "Refined sugar": "Sugar",
    "Oil (vegetable)": "Vegetable oil",
    "Oil (vegetable, fortified)": "Vegetable oil",
    "Refined Vegetable Oil": "Vegetable oil",
    "Cooking fat": "Vegetable oil",
    # ---- Livestock / dairy / eggs ----
    "Milk": "Milk",
    "Milk (cow, pasteurized)": "Milk",
    "Milk (UHT)": "Milk",
    "Milk (cow, fresh)": "Milk",
    "Cow's Milk (Fresh, Pasteurized)": "Milk",
    "Cow's milk (UHT)": "Milk",
    "Milk (camel, fresh)": "Milk (camel)",
    "Meat (beef)": "Beef",
    "Beef": "Beef",
    "Cattle (Male, 2-3 years old, Local Quality)": "Beef",
    "Meat (goat)": "Mutton",
    "Mutton": "Mutton",
    "Meat (camel)": "Camel meat",
    "Chicken": "Chicken",
    "Goats (Local Quality)": "Goats",
    "Eggs": "Eggs",
    "Eggs (Fresh)": "Eggs",
    # ---- Fish ----
    "Fish (tilapia)": "Fish (tilapia)",
    "Fish (omena, dry)": "Fish (omena)",
    "Fish (Dried, Salted, or In Brine)": "Fish (omena)",
}

# Non-food FEWS series (electricity, transport, uniforms) — filter out at load
# time; they have prices but no place in a commodity forecast pipeline.
_NON_FOOD_SKIP = {
    "Electricity", "Bus fare", "School uniform", "Charcoal",
    "Firewood", "Kerosene", "Diesel", "Petrol", "Fuel (kerosene)",
    "Fuel (diesel)", "Fuel (petrol)", "Water", "Salt",
}


def _canonical(name: str) -> str | None:
    """Alias-map lookup; returns None for non-food strings so callers can drop."""
    s = str(name).strip()
    if s in _NON_FOOD_SKIP:
        return None
    return _COMMODITY_ALIAS.get(s, s)


# KAMIS decorates market names with qualifiers WFP/FEWS don't use — strip
# them so "Nakuru Wakulima", "Nakuru Wholesale", "Nakuru Town" all fold onto
# WFP's plain "Nakuru". Conservative on purpose: only well-known suffixes.
_MARKET_STRIP_RE = re.compile(
    r"\s+(?:wholesale|retail|wakulima|town|market|open\s+air|crops?\s+market|"
    r"livestock\s+market|fish\s+landing\s+site|centre|center|main)\b",
    re.IGNORECASE,
)


def _norm_market(name: str) -> str:
    """Strip KAMIS decorators, county suffix after ' - ', collapse whitespace."""
    if not isinstance(name, str):
        return name
    s = name.split(" - ")[0]            # "Cheptiret - Uasin Gishu" -> "Cheptiret"
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # drop parenthetical "(Baringo)" etc.
    s = _MARKET_STRIP_RE.sub("", s)
    return " ".join(s.split()).title().strip()


def load_wfp() -> pd.DataFrame:
    df = pd.read_csv(WFP, low_memory=False)
    # First row of WFP HDX is a units/definitions row — drop if `price` isn't numeric there
    df = df[pd.to_numeric(df["price"], errors="coerce").notna()].copy()
    df["price"] = df["price"].astype(float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date", "market", "commodity", "pricetype"])
    df["kg_per_unit"] = df["unit"].map(_UNIT_TO_KG)
    df = df[df["kg_per_unit"].notna()].copy()
    df["price_per_kg"] = df["price"] / df["kg_per_unit"]
    df["commodity"] = df["commodity"].map(_canonical)
    df = df[df["commodity"].notna()].copy()
    df["market"] = df["market"].map(_norm_market)
    out = df.rename(columns={"admin1": "admin1"})[[
        "commodity", "market", "pricetype", "date", "price_per_kg",
        "admin1", "latitude", "longitude", "category",
    ]].copy()
    out["source"] = "WFP"
    out["unit"] = "KG"
    return out


def load_fews() -> pd.DataFrame:
    df = pd.read_csv(FEWS, low_memory=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "market", "product", "price_type", "period_date"])
    df["date"] = pd.to_datetime(df["period_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date"])
    # FEWS `unit` is a mess: 'kg', '90 kg', '90_kg', '98_kg', '99_kg', '110_kg', '2_kg'.
    # Regex parses "<n?><_ or space?>kg" -> KG value (defaulting to 1 when the number
    # group is absent). Recovers ~9.3k rows the old hardcoded dict dropped.
    unit_norm = df["unit"].astype(str).str.strip().str.lower()
    _kg_re = re.compile(r"^(\d+(?:\.\d+)?)?\s*[_ ]?\s*kg$")
    def _kg(u: str) -> float | None:
        m = _kg_re.match(u)
        if not m:
            return None
        return float(m.group(1)) if m.group(1) else 1.0
    df["kg_per_unit"] = unit_norm.map(_kg)
    df = df[df["kg_per_unit"].notna()].copy()
    df["price_per_kg"] = df["value"] / df["kg_per_unit"]
    df["commodity"] = df["product"].map(_canonical)
    df = df[df["commodity"].notna()].copy()
    df["market"] = df["market"].map(_norm_market)
    out = df.rename(columns={"admin_1": "admin1", "price_type": "pricetype"})[[
        "commodity", "market", "pricetype", "date", "price_per_kg",
        "admin1", "latitude", "longitude",
    ]].copy()
    out["source"] = "FEWS"
    out["unit"] = "KG"
    out["category"] = np.nan  # not present in FEWS
    return out


def load_kamis() -> pd.DataFrame:
    if not KAMIS.exists():
        print(f"[info] {KAMIS.relative_to(ROOT)} not found - run scripts/scrape_kamis.py first")
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(KAMIS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "market", "commodity", "pricetype", "price"])
    df["commodity"] = df["commodity"].map(_canonical)
    df = df[df["commodity"].notna()].copy()
    df["market"] = df["market"].map(_norm_market)
    out = df.rename(columns={"admin_1": "admin1", "price": "price_per_kg"})[[
        "commodity", "market", "pricetype", "date", "price_per_kg", "admin1",
    ]].copy()
    # KAMIS aggregation may leave duplicate (market, commodity, pricetype, date)
    # rows once market names collapse (e.g. Nakuru + Nakuru Wakulima -> Nakuru).
    # Average them before stitching so the chain's dedup gets a single row.
    out = out.groupby(
        ["commodity", "market", "pricetype", "date", "admin1"], as_index=False
    )["price_per_kg"].mean()
    out["source"] = "KAMIS"
    out["unit"] = "KG"
    out["latitude"] = np.nan
    out["longitude"] = np.nan
    out["category"] = np.nan
    return out


def _load_supplement(path: pathlib.Path, source_tag: str) -> pd.DataFrame:
    """Loader for the two on-disk supplement CSVs, both shipped in the same
    long format: [date, market, commodity, price, unit, pricetype, source].
    Prices already KES per <unit>; unit parsed by the same regex used for FEWS.
    """
    if not path.exists():
        print(f"[info] {path.relative_to(ROOT)} not found - skipping")
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "market", "commodity", "pricetype", "price"])
    unit_norm = df["unit"].astype(str).str.strip().str.lower()
    _kg_re = re.compile(r"^(\d+(?:\.\d+)?)?\s*[_ ]?\s*kg$")
    def _kg(u: str) -> float | None:
        m = _kg_re.match(u)
        return (float(m.group(1)) if m.group(1) else 1.0) if m else None
    df["kg_per_unit"] = unit_norm.map(_kg)
    df = df[df["kg_per_unit"].notna()].copy()
    df["price_per_kg"] = df["price"] / df["kg_per_unit"]
    df["commodity"] = df["commodity"].map(_canonical)
    df = df[df["commodity"].notna()].copy()
    df["market"] = df["market"].map(_norm_market)
    out = df[["commodity", "market", "pricetype", "date", "price_per_kg"]].copy()
    out["source"] = source_tag
    out["admin1"] = ""
    out["latitude"] = np.nan
    out["longitude"] = np.nan
    out["category"] = np.nan
    out["unit"] = "KG"
    return out[SCHEMA]


def load_wholesale_supplement() -> pd.DataFrame:
    return _load_supplement(
        ROOT / "data" / "raw" / "Wholesale supplement 2022–2026.csv", "WFP_SUPP"
    )


def load_knbs_retail() -> pd.DataFrame:
    return _load_supplement(
        ROOT / "data" / "raw" / "KNBS retail reference 2022–2026.csv", "KNBS"
    )


def stitch(wfp: pd.DataFrame, fews: pd.DataFrame, kamis: pd.DataFrame,
           wfp_supp: pd.DataFrame | None = None,
           knbs: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = [p for p in [wfp, fews, kamis, wfp_supp, knbs] if p is not None]
    all_ = pd.concat(parts, ignore_index=True)
    all_ = all_[all_["price_per_kg"].gt(0)]
    # Priority: FEWS first, WFP second, KAMIS third. Keep first per group.
    all_["_prio"] = all_["source"].map({s: i for i, s in enumerate(SOURCE_PRIORITY)})
    all_ = all_.sort_values(["commodity", "market", "pricetype", "date", "_prio"])
    picked = all_.drop_duplicates(
        subset=["commodity", "market", "pricetype", "date"], keep="first"
    ).drop(columns=["_prio"])
    return picked[SCHEMA].reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    print(f"\ntotal rows:  {len(df):>7,}")
    print(f"commodities: {df['commodity'].nunique():>7,}")
    print(f"markets:     {df['market'].nunique():>7,}")
    print(f"date range:  {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"sources:     {dict(df['source'].value_counts())}")
    counts = df.groupby(["commodity", "market", "pricetype"]).size()
    passing = counts.ge(60).groupby(level="commodity").sum().sort_values(ascending=False)
    print(f"\nseries with >=60 obs: {int(counts.ge(60).sum())} total")
    print("per commodity (top 25):")
    for name, n in passing.head(25).items():
        print(f"  {name:<28} {int(n):>4}")


def demo() -> None:
    wfp, fews, kamis = load_wfp(), load_fews(), load_kamis()
    supp, knbs = load_wholesale_supplement(), load_knbs_retail()
    for name, part in (("WFP", wfp), ("FEWS", fews), ("KAMIS", kamis),
                       ("WFP_SUPP", supp), ("KNBS", knbs)):
        assert list(part.columns) == SCHEMA if not part.empty else True, f"{name} schema drift"
    out = stitch(wfp, fews, kamis, supp, knbs)
    assert not out.empty, "stitcher produced empty output"
    assert list(out.columns) == SCHEMA, "output schema drift"
    assert out["price_per_kg"].gt(0).all()
    # Benchmark markets from the supplement should now be present.
    for mk in ["Nairobi", "Nakuru", "Mombasa", "Kisumu", "Eldoret"]:
        assert (out["market"] == mk).any(), f"benchmark market missing: {mk}"
    print(f"[demo] OK - stitched {len(out):,} rows, {out['commodity'].nunique()} commodities")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    wfp = load_wfp();   print(f"[load] WFP        {len(wfp):>7,} rows")
    fews = load_fews(); print(f"[load] FEWS       {len(fews):>7,} rows")
    kamis = load_kamis(); print(f"[load] KAMIS      {len(kamis):>7,} rows")
    supp = load_wholesale_supplement(); print(f"[load] WFP_SUPP   {len(supp):>7,} rows")
    knbs = load_knbs_retail();          print(f"[load] KNBS       {len(knbs):>7,} rows")

    out = stitch(wfp, fews, kamis, supp, knbs)
    report(out)

    if args.demo:
        sys.exit(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
