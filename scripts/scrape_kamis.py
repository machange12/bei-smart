"""Scrape KAMIS (kamis.kilimo.go.ke) historical wholesale + retail prices.

Emits long-format monthly rows matching the schema of
`data/cleaned/bei_smart_forecasting_chained.csv` so the stitcher in
`scripts/build_chained_all.py` can concat directly:

    [source, market, admin_1, commodity, pricetype, unit, currency, date, price]

KAMIS returns daily rows; we aggregate to monthly mean per (market, commodity,
pricetype). Prices arrive as "42.22/Kg" strings — we parse the number and keep
"/Kg" only (other units filtered out to keep KES/KG comparable with WFP/FEWS).

Uses the site's own Excel export behind the /site/market_search endpoint —
same params as the HTML search. `product[]` takes numeric IDs (scraped from
the search form's <select>); dates are Y-m-d; `per_page=1500` caps the sheet.
We chunk by (product, year) to stay under the cap — daily rows for popular
products push ~3000/year across all markets.

Cache: one .xlsx per (product_id, year) in data/raw/_kamis_cache/. Re-runs are
idempotent; delete the cache file to force refetch.

Run:
    python scripts/scrape_kamis.py                       # full pull (long)
    python scripts/scrape_kamis.py --years 2022 2023 2024 2025 2026
    python scripts/scrape_kamis.py --demo                # tiny self-check
"""
from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys
import time
from typing import Iterable

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "_kamis_cache"
OUT = ROOT / "data" / "raw" / "kamis_prices.csv"
BASE = "https://kamis.kilimo.go.ke"
SEARCH_URL = f"{BASE}/site/market_search"

# Product IDs refreshed 2026-09-21 via `--dump-products` (site has ~193 total).
# Curated to food commodities; skipped hides/fertilizer/cotton/obscure fish species
# and numeric-name placeholders (10, 50, 100, 500, 1500, 3000 — section headers).
# Cache in data/raw/_kamis_cache/ makes re-runs idempotent; new IDs get pulled,
# unchanged ones stay put. Names as they appear on kamis.kilimo.go.ke.
PRODUCTS = {
    # Grains & flours
    1: "Dry maize",           2: "Red sorghum",       3: "Wheat",
    4: "Rice",                51: "Pearl rush millet", 54: "Finger millet",
    56: "White sorghum",      149: "Green maize",     249: "Maize flour",
    265: "Wheat flour",       267: "Paddy rice",      220: "Wheat bran",
    221: "Maize bran",
    # Pulses & legumes
    64: "Beans rosecoco",     65: "Beans (mwitemania)", 66: "Beans (mwezi moja)",
    67: "Beans (canadian wonder)", 29: "Beans red haricot (wairimu)",
    30: "Beans (yellow-green)", 183: "Beans rosecoco (nyayo)",
    188: "Pigeon peas",       189: "Cowpeas",         245: "Dry peas",
    246: "Mixed beans",       269: "Beans (yellow)",  259: "Lentils",
    182: "Njugu mawe",
    # Tubers, roots
    57: "Red irish potato",   163: "White irish potatoes", 59: "Sweet potatoes",
    162: "Cassava fresh",     164: "Cassava chips (dry)", 131: "Yam",
    143: "Arrow root",
    # Vegetables
    58: "Cabbages",           60: "Carrots",          61: "Tomatoes",
    154: "Kales/sukuma wiki", 158: "Dry onions",      159: "Spring onions",
    160: "Fresh peas",        161: "Spinach",         165: "Chillies",
    170: "Pumpkin",           171: "Butternuts",      172: "Capsicums",
    173: "Cucumber",          174: "Egg plant (brinjals)", 175: "Cauliflower",
    177: "French beans",      178: "Ginger",          180: "Garlic",
    257: "Courgette",         258: "Broccoli",        166: "Lettuce",
    272: "Okra (lady's fingers or gumbo)",
    # Indigenous / traditional veg
    121: "Black nightshade (managu/ osuga)", 122: "Spider flower (saga)",
    123: "Amaranthus (terere)", 124: "Jute plant (murenda)",
    230: "Cowpea leaves (kunde)", 231: "Nderema- vine spinach",
    233: "Pumpkin leaves", 234: "Ethiopian kales -kanzira",
    235: "Indigenous crotolaria (mito/miro)", 242: "Coriander (dhania)",
    # Fruits
    142: "Avocado",           147: "Mangoes",         145: "Lemons",
    148: "Limes",             127: "Oranges",         151: "Pineapples",
    152: "Pawpaw",            150: "Water melon",     226: "Banana (ripening)",
    255: "Banana (cooking)",  256: "Banana (plantain)", 125: "Passion fruits",
    128: "Tree tomato",       129: "Pepino melon",    130: "Thorn melon",
    243: "Grapes",            244: "Apples",          262: "Tangerine (sandara)",
    270: "Coconut",           273: "Dragon fruit",
    # Meat, poultry, livestock
    73: "Meat beef",          74: "Meat mutton",      75: "Meat indiginous chicken",
    76: "Meat broiler",       208: "Camel meat",      209: "Meat chevon",
    210: "Rabbit meat",       140: "Cattle",          167: "Sheep",
    168: "Goat",              186: "Camel",           187: "Rabbit",
    211: "Pigs",              141: "Pork",            227: "Chicken",
    251: "Duck",              169: "Donkey",
    # Fish (common commercial species only)
    68: "Tilapia",            77: "Omena",            78: "Cat fish(mkizi/fume)",
    80: "Trout",              97: "Tuna",             98: "Mackerel",
    99: "Kingfish (nguru)",   101: "Sardines",        184: "Nile perch",
    104: "Prawns",            110: "Octopus (pweza)",
    # Dairy & eggs
    72: "Eggs",               133: "Cow milk(at collection point)",
    153: "Cow milk(processd)", 70: "Goat milk (at collection point)",
    139: "Goat milk (processed)", 134: "Camel milk(at collection point)",
    138: "Camel milk(processed)",
    # Oils, seeds, nuts
    12: "Ground nuts",        228: "Macadamia seed",  229: "Cashewnuts (korosho)",
    237: "Soybean oil",       238: "Coconut oil",     239: "Sunflower seeds",
    240: "Sunflower oil",     222: "Sunflower cake",  241: "Walnut seed",
    # Cash crops / beverages
    212: "Honey",             218: "Tea",             219: "Coffee",
}

# strings like "42.22/Kg" or "3,500.00/Bag(90 Kg)" — pull the leading number
# and the unit token after the slash.
_PRICE_RE = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*/\s*(.+?)\s*$")


def dump_product_ids() -> dict[int, str]:
    """Fetch the KAMIS search page and return {id: name} for every option.

    Used to refresh PRODUCTS when the site adds/renames/removes commodities.
    Prints a diff vs the current PRODUCTS dict so we can hand-edit.
    """
    html = requests.get(f"{BASE}/site/market_search", timeout=60).text
    pairs = re.findall(r'<option value="(\d+)">([^<]+)</option>', html)
    live = {int(pid): name.strip() for pid, name in pairs if int(pid) > 0}
    current = set(PRODUCTS)
    added = sorted(set(live) - current)
    removed = sorted(current - set(live))
    renamed = sorted(pid for pid in current & set(live) if PRODUCTS[pid] != live[pid])
    print(f"[live]    {len(live)} products on site")
    print(f"[current] {len(current)} in PRODUCTS")
    print(f"[added]   {len(added)}: " + ", ".join(f"{p}={live[p]}" for p in added[:20]))
    if len(added) > 20:
        print(f"          ... and {len(added)-20} more")
    print(f"[removed] {len(removed)}: " + ", ".join(f"{p}={PRODUCTS[p]}" for p in removed))
    print(f"[renamed] {len(renamed)}: " + ", ".join(
        f"{p}: {PRODUCTS[p]!r} -> {live[p]!r}" for p in renamed))
    print("\n--- fresh PRODUCTS dict (paste into scrape_kamis.py) ---")
    for pid in sorted(live):
        print(f"    {pid}: {live[pid]!r},")
    return live


def _parse_price(cell) -> tuple[float | None, str | None]:
    if cell is None or (isinstance(cell, float) and cell != cell):
        return None, None
    s = str(cell).strip()
    if s in ("-", "", "nan", "None"):
        return None, None
    m = _PRICE_RE.match(s)
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",", "")), m.group(2)
    except ValueError:
        return None, None


def _month_end(year: int, month: int) -> str:
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")


def _fetch_product_month(product_id: int, year: int, month: int, sleep: float = 1.0) -> pd.DataFrame:
    """One request per (product, year-month). Cached to disk.

    ponytail: the site's per_page cap is 1500 rows. Popular commodities push
    ~3000 daily rows/year across all markets, so chunk by month to stay safe.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"prod{product_id}_{year}-{month:02d}.xlsx"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return pd.read_excel(cache_file)
    # Empty marker file — already tried and got 500/empty
    if cache_file.exists():
        return pd.DataFrame()

    params = [
        ("product[]", str(product_id)),
        ("start", f"{year}-{month:02d}-01"),
        ("end", _month_end(year, month)),
        ("per_page", "1500"),
        ("export", "excel"),
    ]
    r = requests.get(SEARCH_URL, params=params, timeout=120)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        cache_file.write_bytes(b"")
        return pd.DataFrame()
    cache_file.write_bytes(r.content)
    time.sleep(sleep)
    try:
        return pd.read_excel(io.BytesIO(r.content))
    except Exception:
        return pd.DataFrame()


def _fetch_product_year(product_id: int, year: int, sleep: float = 1.0) -> pd.DataFrame:
    """Concat 12 monthly pulls for one (product, year)."""
    parts = []
    for m in range(1, 13):
        try:
            parts.append(_fetch_product_month(product_id, year, m, sleep=sleep))
        except requests.HTTPError as e:
            print(f"[warn]   {year}-{m:02d}: HTTP {e.response.status_code}", file=sys.stderr)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _to_long(daily: pd.DataFrame, canonical_name: str) -> pd.DataFrame:
    """Daily KAMIS rows → monthly long rows in the chained schema."""
    if daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Market", "County"])

    rows = []
    for label, col in (("Wholesale", "Wholesale"), ("Retail", "Retail")):
        parsed = df[col].map(_parse_price)
        prices = parsed.map(lambda t: t[0])
        units = parsed.map(lambda t: (t[1] or "").lower())
        mask = prices.notna() & units.str.contains("kg")  # KES/KG only
        sub = df[mask].assign(_price=prices[mask])
        if sub.empty:
            continue
        sub["date"] = sub["Date"].dt.to_period("M").dt.to_timestamp()
        # ponytail: median beats mean here — KAMIS occasionally mislabels a
        # wholesale-bag price as /Kg (0.3% of rows), and a monthly mean makes
        # one bad day dominate the reading. Median shrugs it off.
        monthly = (
            sub.groupby(["Market", "County", "date"], as_index=False)["_price"].median()
            .rename(columns={"Market": "market", "County": "admin_1", "_price": "price"})
        )
        monthly["source"] = "KAMIS"
        monthly["commodity"] = canonical_name
        monthly["pricetype"] = label
        monthly["unit"] = "KG"
        monthly["currency"] = "KES"
        rows.append(monthly[[
            "source", "market", "admin_1", "commodity",
            "pricetype", "unit", "currency", "date", "price",
        ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def scrape(product_ids: Iterable[int], years: Iterable[int], sleep: float = 1.0) -> pd.DataFrame:
    frames = []
    for pid in product_ids:
        name = PRODUCTS.get(pid, f"product_{pid}")
        for year in years:
            try:
                raw = _fetch_product_year(pid, year, sleep=sleep)
            except requests.HTTPError as e:
                print(f"[warn] {name} {year}: HTTP {e.response.status_code}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"[warn] {name} {year}: {e}", file=sys.stderr)
                continue
            long = _to_long(raw, name)
            if not long.empty:
                frames.append(long)
                print(f"[ok]   {name:<28} {year}: {len(raw):5d} daily -> {len(long):5d} monthly rows")
            else:
                print(f"[skip] {name:<28} {year}: empty")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Dedup — a (market, commodity, pricetype, month) should appear once
    out = out.sort_values("date").drop_duplicates(
        subset=["source", "market", "commodity", "pricetype", "date"],
        keep="last",
    )
    return out.reset_index(drop=True)


def demo() -> None:
    """Small self-check — pull one commodity, one year, assert schema + rows."""
    df = scrape([1], [2024], sleep=0.3)  # Dry maize, 2024
    assert not df.empty, "demo pull returned no rows"
    expected = {"source", "market", "admin_1", "commodity",
                "pricetype", "unit", "currency", "date", "price"}
    assert set(df.columns) == expected, f"schema drift: {set(df.columns) ^ expected}"
    assert df["price"].gt(0).all(), "non-positive prices leaked through"
    assert df["date"].dt.day.eq(1).all(), "dates not month-start"
    assert df["pricetype"].isin({"Wholesale", "Retail"}).all()
    n_markets = df["market"].nunique()
    print(f"\n[demo] OK - {len(df)} rows, {n_markets} markets, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--dump-products", action="store_true",
                    help="fetch live product IDs and print a refreshed PRODUCTS dict")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2027)))
    ap.add_argument("--products", type=int, nargs="+", default=list(PRODUCTS.keys()))
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    if args.demo:
        demo()
        sys.exit(0)

    if args.dump_products:
        dump_product_ids()
        sys.exit(0)

    df = scrape(args.products, args.years, sleep=args.sleep)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {len(df):,} rows -> {OUT.relative_to(ROOT)}")
    print(f"commodities: {df['commodity'].nunique()}, markets: {df['market'].nunique()}, "
          f"pairs >=60 obs: {df.groupby(['commodity','market','pricetype']).size().ge(60).sum()}")
