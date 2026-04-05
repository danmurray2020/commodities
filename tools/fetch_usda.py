"""Fetch USDA Crop Progress data from the NASS QuickStats API.

Retrieves weekly crop progress reports for soybeans, wheat, and corn,
saving results to the relevant commodity data directories.

Usage:
    python3 tools/fetch_usda.py                  # fetch all crops
    python3 tools/fetch_usda.py --crops soybeans wheat
    python3 tools/fetch_usda.py --start-year 2020
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

# Add repo root so we can import the retry utility
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.retry import retry_with_backoff

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# Crop configurations: USDA commodity name -> (local directory, progress items)
CROP_CONFIG = {
    "SOYBEANS": {
        "dir": "soybeans",
        "progress_items": [
            "PROGRESS, MEASURED IN PCT PLANTED",
            "PROGRESS, MEASURED IN PCT EMERGED",
            "PROGRESS, MEASURED IN PCT BLOOMING",
        ],
        "condition": True,
    },
    "WHEAT": {
        "dir": "wheat",
        "progress_items": [
            "PROGRESS, MEASURED IN PCT PLANTED",
            "PROGRESS, MEASURED IN PCT EMERGED",
            "PROGRESS, MEASURED IN PCT HEADED",
        ],
        "condition": True,
    },
    "CORN": {
        "dir": None,  # no dedicated corn directory; save alongside soybeans
        "progress_items": [
            "PROGRESS, MEASURED IN PCT PLANTED",
            "PROGRESS, MEASURED IN PCT EMERGED",
        ],
        "condition": True,
    },
}

# Map USDA crop names to output directories (corn has no dir, save with soybeans)
CROP_OUTPUT_DIR = {
    "SOYBEANS": "soybeans",
    "WHEAT": "wheat",
    "CORN": "soybeans",
}


def get_api_key() -> str:
    """Read USDA_API_KEY from the environment."""
    key = os.environ.get("USDA_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: USDA_API_KEY environment variable not set.\n"
            "\n"
            "To get a free API key:\n"
            "  1. Visit https://quickstats.nass.usda.gov/api\n"
            "  2. Click 'Request API Key'\n"
            "  3. Enter your email and you will receive a key immediately\n"
            "  4. Export it:  export USDA_API_KEY=YOUR_KEY_HERE\n"
        )
        sys.exit(1)
    return key


@retry_with_backoff(max_retries=3, base_delay=3.0)
def fetch_nass(params: dict) -> list[dict]:
    """Make a single NASS QuickStats API request and return the data rows."""
    resp = requests.get(NASS_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"NASS API error: {payload['error']}")
    return payload.get("data", [])


def fetch_crop_progress(
    api_key: str,
    crop: str,
    start_year: int = 2015,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Fetch weekly crop progress data for a single crop.

    Returns a DataFrame with columns like: year, week_ending, short_desc, Value.
    """
    config = CROP_CONFIG[crop]
    all_rows: list[dict] = []

    # --- Fetch progress items ---
    for item in config["progress_items"]:
        print(f"  Fetching {crop} / {item} ...")
        params = {
            "key": api_key,
            "commodity_desc": crop,
            "statisticcat_desc": "PROGRESS",
            "short_desc": f"{crop} - {item}",
            "agg_level_desc": "NATIONAL",
            "state_alpha": "US",
            "year__GE": str(start_year),
            "year__LE": str(end_year),
            "freq_desc": "WEEKLY",
            "format": "JSON",
        }
        try:
            rows = fetch_nass(params)
            print(f"    -> {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"    WARNING: failed to fetch {item}: {e}")

    # --- Fetch condition (GOOD, EXCELLENT, etc.) ---
    if config["condition"]:
        print(f"  Fetching {crop} / CONDITION ...")
        params = {
            "key": api_key,
            "commodity_desc": crop,
            "statisticcat_desc": "CONDITION",
            "agg_level_desc": "NATIONAL",
            "state_alpha": "US",
            "year__GE": str(start_year),
            "year__LE": str(end_year),
            "freq_desc": "WEEKLY",
            "format": "JSON",
        }
        try:
            rows = fetch_nass(params)
            print(f"    -> {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"    WARNING: failed to fetch condition: {e}")

    if not all_rows:
        print(f"  WARNING: No data returned for {crop}")
        return pd.DataFrame()

    # Build a tidy DataFrame
    df = pd.DataFrame(all_rows)

    # Keep only useful columns
    keep_cols = [
        "commodity_desc",
        "short_desc",
        "year",
        "week_ending",
        "begin_code",
        "Value",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Clean up Value column (can contain commas, "(D)", etc.)
    df["Value"] = pd.to_numeric(
        df["Value"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )
    df.dropna(subset=["Value"], inplace=True)

    if "week_ending" in df.columns:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
        df.sort_values("week_ending", inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def save_crop_data(df: pd.DataFrame, crop: str) -> None:
    """Save a crop's progress DataFrame to the appropriate data directory."""
    out_dir_name = CROP_OUTPUT_DIR.get(crop)
    if out_dir_name is None:
        print(f"  Skipping save for {crop} (no output directory configured)")
        return

    out_dir = REPO_ROOT / out_dir_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"usda_crop_progress_{crop.lower()}.csv" if crop == "CORN" else "usda_crop_progress.csv"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch USDA Crop Progress data from NASS QuickStats API"
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=list(CROP_CONFIG.keys()),
        choices=[c.lower() for c in CROP_CONFIG] + list(CROP_CONFIG.keys()),
        help="Crops to fetch (default: all)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year (default: 2015)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="End year (default: 2026)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    crops = [c.upper() for c in args.crops]

    print(f"Fetching USDA crop progress for: {', '.join(crops)}")
    print(f"Year range: {args.start_year}-{args.end_year}\n")

    for crop in crops:
        if crop not in CROP_CONFIG:
            print(f"WARNING: Unknown crop '{crop}', skipping.")
            continue
        print(f"[{crop}]")
        df = fetch_crop_progress(api_key, crop, args.start_year, args.end_year)
        if not df.empty:
            save_crop_data(df, crop)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
