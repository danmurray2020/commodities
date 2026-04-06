#!/usr/bin/env python3
"""Fetch ACLED conflict data for West African cocoa-producing countries.

ACLED (Armed Conflict Location & Event Data) tracks political violence and
protest events worldwide.  For cocoa, we care about Ivory Coast, Ghana,
Cameroon, and Nigeria — the main West African cocoa belt.

API docs: https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2019/01/ACLED_API-User-Guide_2019FINAL.pdf
Register for a free key at https://acleddata.com/register/

Environment variables required:
    ACLED_API_KEY   – Your ACLED API key
    ACLED_EMAIL     – The email address used to register
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import requests

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.retry import retry_with_backoff

logger = logging.getLogger("commodities.fetch_acled")

ACLED_API_URL = "https://api.acleddata.com/acled/read"

# Countries in the West African cocoa belt
COUNTRIES = ["Ivory Coast", "Ghana", "Cameroon", "Nigeria"]

# Event types relevant to political / supply risk
EVENT_TYPES = ["Battles", "Protests", "Riots", "Strategic developments"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "chocolate" / "data"
OUTPUT_FILE = OUTPUT_DIR / "acled_conflict.csv"

PAGE_SIZE = 5000  # ACLED max results per page


def _check_credentials() -> tuple[str, str]:
    """Return (api_key, email) or exit with instructions."""
    api_key = os.environ.get("ACLED_API_KEY", "").strip()
    email = os.environ.get("ACLED_EMAIL", "").strip()
    if not api_key or not email:
        print(
            "=" * 68 + "\n"
            "  ACLED API credentials not found.\n\n"
            "  1. Register for a free account at:\n"
            "     https://acleddata.com/register/\n\n"
            "  2. Set environment variables:\n"
            "     export ACLED_API_KEY='your-key-here'\n"
            "     export ACLED_EMAIL='you@example.com'\n\n"
            "  3. Re-run this script.\n"
            "=" * 68
        )
        sys.exit(1)
    return api_key, email


@retry_with_backoff(max_retries=3, base_delay=3.0)
def _fetch_page(api_key: str, email: str, country: str, page: int = 1) -> list[dict]:
    """Fetch one page of ACLED data for a single country."""
    params = {
        "key": api_key,
        "email": email,
        "country": country,
        "event_date": "2015-01-01|",  # from 2015 onward
        "event_date_where": "BETWEEN",
        "event_type": "|".join(EVENT_TYPES),
        "fields": "event_date|country|event_type|fatalities|interaction",
        "limit": PAGE_SIZE,
        "page": page,
    }
    resp = requests.get(ACLED_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success", True):
        raise RuntimeError(f"ACLED API error: {payload.get('error', 'unknown')}")
    return payload.get("data", [])


def fetch_all_events(api_key: str, email: str) -> pd.DataFrame:
    """Fetch all relevant events for all target countries."""
    all_rows: list[dict] = []
    for country in COUNTRIES:
        logger.info(f"Fetching ACLED events for {country}...")
        page = 1
        while True:
            rows = _fetch_page(api_key, email, country, page=page)
            if not rows:
                break
            all_rows.extend(rows)
            logger.info(f"  {country} page {page}: {len(rows)} events")
            if len(rows) < PAGE_SIZE:
                break
            page += 1

    if not all_rows:
        logger.warning("No events returned from ACLED API.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw event data to weekly frequency.

    Columns produced:
        acled_event_count       – total events across all countries
        acled_fatalities        – total fatalities across all countries
        acled_events_civ        – events in Ivory Coast only
        acled_events_gha        – events in Ghana only
        acled_battles           – battle events (all countries)
        acled_protests_riots    – protests + riots (all countries)
    """
    df = df.copy()
    df = df.set_index("event_date").sort_index()

    weekly = pd.DataFrame()

    # Total events and fatalities
    weekly["acled_event_count"] = df.resample("W-FRI")["country"].count()
    weekly["acled_fatalities"] = df.resample("W-FRI")["fatalities"].sum()

    # Per-country breakdowns
    civ_mask = df["country"].str.contains("Ivory", case=False, na=False)
    gha_mask = df["country"].str.contains("Ghana", case=False, na=False)
    weekly["acled_events_civ"] = df.loc[civ_mask].resample("W-FRI")["country"].count()
    weekly["acled_events_gha"] = df.loc[gha_mask].resample("W-FRI")["country"].count()

    # Event type breakdowns
    battles = df[df["event_type"].str.contains("Battles", case=False, na=False)]
    protests_riots = df[df["event_type"].str.contains("Protests|Riots", case=False, na=False)]
    weekly["acled_battles"] = battles.resample("W-FRI")["country"].count()
    weekly["acled_protests_riots"] = protests_riots.resample("W-FRI")["country"].count()

    weekly = weekly.fillna(0).astype(int)
    weekly.index.name = "Date"
    return weekly


def main():
    api_key, email = _check_credentials()
    print(f"Fetching ACLED data for: {', '.join(COUNTRIES)}")
    raw = fetch_all_events(api_key, email)
    if raw.empty:
        print("No data fetched. Exiting.")
        return

    print(f"Total raw events: {len(raw)}")
    weekly = aggregate_weekly(raw)
    print(f"Weekly rows: {len(weekly)}  ({weekly.index.min()} to {weekly.index.max()})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
