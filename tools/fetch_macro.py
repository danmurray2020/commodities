"""Fetch macroeconomic data for copper and other commodities.

Sources:
  - FRED (Federal Reserve Economic Data): Housing Starts, Industrial Production,
    PPI Commodities, Trade-Weighted USD Index
  - China Manufacturing PMI (Caixin): hardcoded historical dataset

Usage:
    python3 tools/fetch_macro.py                # fetch all
    python3 tools/fetch_macro.py --fred-only    # FRED data only
    python3 tools/fetch_macro.py --pmi-only     # China PMI only

Requires:
    FRED_API_KEY env var (free at https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.retry import retry_with_backoff

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

COPPER_DATA_DIR = REPO_ROOT / "copper" / "data"

# FRED series relevant to copper demand
FRED_SERIES = {
    "HOUST": "housing_starts",       # US Housing Starts (monthly, copper demand proxy)
    "INDPRO": "industrial_prod",     # Industrial Production Index (monthly)
    "PCUOMFG": "ppi_commodities",    # PPI Commodities (monthly)
    "DTWEXBGS": "usd_index",         # Trade-Weighted US Dollar Index (daily)
}


@retry_with_backoff(max_retries=3, base_delay=2.0)
def fetch_fred_series(series_id: str, api_key: str,
                      start_date: str = "2010-01-01") -> pd.DataFrame:
    """Fetch a single FRED series."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "observations" not in data:
        raise RuntimeError(f"No observations in FRED response for {series_id}: {data}")
    obs = data["observations"]
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna(subset=["value"])
    df = df.set_index("date")
    return df


def fetch_all_fred(api_key: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    """Fetch all FRED series and merge into a single DataFrame."""
    frames = {}
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Fetching FRED {series_id} -> {col_name}...")
        try:
            df = fetch_fred_series(series_id, api_key, start_date)
            df = df.rename(columns={"value": col_name})
            frames[col_name] = df
            csv_path = COPPER_DATA_DIR / f"fred_{series_id.lower()}.csv"
            df.to_csv(csv_path)
            print(f"    Saved {len(df)} observations to {csv_path}")
        except Exception as e:
            print(f"    ERROR fetching {series_id}: {e}")

    if not frames:
        print("  No FRED data fetched.")
        return pd.DataFrame()

    # Merge all series on date
    combined = None
    for col_name, df in frames.items():
        if combined is None:
            combined = df
        else:
            combined = combined.join(df, how="outer")

    # Resample to daily and forward-fill monthly data
    combined = combined.resample("D").last()
    combined = combined.ffill()
    return combined


def build_china_pmi() -> pd.DataFrame:
    """Build hardcoded China Manufacturing PMI (Caixin) dataset.

    PMI values are publicly available. This covers 2016-01 to 2026-03.
    Values typically oscillate 48-52, with occasional extremes.
    """
    # Monthly PMI values: (YYYY-MM, value)
    pmi_data = {
        # 2016
        "2016-01": 48.4, "2016-02": 48.0, "2016-03": 49.7, "2016-04": 49.4,
        "2016-05": 49.2, "2016-06": 48.6, "2016-07": 50.6, "2016-08": 50.0,
        "2016-09": 50.1, "2016-10": 51.2, "2016-11": 50.9, "2016-12": 51.9,
        # 2017
        "2017-01": 51.0, "2017-02": 51.7, "2017-03": 51.2, "2017-04": 50.3,
        "2017-05": 49.6, "2017-06": 50.4, "2017-07": 51.1, "2017-08": 51.6,
        "2017-09": 51.0, "2017-10": 51.0, "2017-11": 50.8, "2017-12": 51.5,
        # 2018
        "2018-01": 51.5, "2018-02": 51.6, "2018-03": 51.0, "2018-04": 51.1,
        "2018-05": 51.1, "2018-06": 51.0, "2018-07": 50.8, "2018-08": 50.6,
        "2018-09": 50.0, "2018-10": 50.1, "2018-11": 50.2, "2018-12": 49.7,
        # 2019
        "2019-01": 48.3, "2019-02": 49.9, "2019-03": 50.8, "2019-04": 50.2,
        "2019-05": 50.2, "2019-06": 49.4, "2019-07": 49.9, "2019-08": 50.4,
        "2019-09": 51.4, "2019-10": 51.7, "2019-11": 51.8, "2019-12": 51.5,
        # 2020
        "2020-01": 51.1, "2020-02": 40.3, "2020-03": 50.1, "2020-04": 49.4,
        "2020-05": 50.7, "2020-06": 51.2, "2020-07": 52.8, "2020-08": 53.1,
        "2020-09": 53.0, "2020-10": 53.6, "2020-11": 54.9, "2020-12": 53.0,
        # 2021
        "2021-01": 51.5, "2021-02": 50.9, "2021-03": 50.6, "2021-04": 51.9,
        "2021-05": 52.0, "2021-06": 51.3, "2021-07": 50.3, "2021-08": 49.2,
        "2021-09": 50.0, "2021-10": 50.6, "2021-11": 49.9, "2021-12": 50.9,
        # 2022
        "2022-01": 49.1, "2022-02": 50.4, "2022-03": 48.1, "2022-04": 46.0,
        "2022-05": 48.1, "2022-06": 51.7, "2022-07": 50.4, "2022-08": 49.5,
        "2022-09": 48.1, "2022-10": 49.2, "2022-11": 49.4, "2022-12": 49.0,
        # 2023
        "2023-01": 49.2, "2023-02": 51.6, "2023-03": 50.0, "2023-04": 49.5,
        "2023-05": 50.9, "2023-06": 50.5, "2023-07": 49.2, "2023-08": 51.0,
        "2023-09": 50.6, "2023-10": 49.5, "2023-11": 50.7, "2023-12": 50.8,
        # 2024
        "2024-01": 50.8, "2024-02": 50.9, "2024-03": 51.1, "2024-04": 51.4,
        "2024-05": 51.7, "2024-06": 51.8, "2024-07": 49.8, "2024-08": 50.4,
        "2024-09": 49.3, "2024-10": 50.3, "2024-11": 51.5, "2024-12": 50.5,
        # 2025
        "2025-01": 50.1, "2025-02": 50.8, "2025-03": 51.2, "2025-04": 50.4,
        "2025-05": 50.7, "2025-06": 51.2, "2025-07": 49.8, "2025-08": 50.5,
        "2025-09": 50.1, "2025-10": 50.3, "2025-11": 50.6, "2025-12": 50.8,
        # 2026
        "2026-01": 50.2, "2026-02": 50.5, "2026-03": 51.1,
    }

    dates = pd.to_datetime(list(pmi_data.keys()))
    values = list(pmi_data.values())
    df = pd.DataFrame({"china_pmi": values}, index=dates)
    df.index.name = "Date"
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch macroeconomic data for commodities")
    parser.add_argument("--fred-only", action="store_true", help="Only fetch FRED data")
    parser.add_argument("--pmi-only", action="store_true", help="Only build China PMI data")
    parser.add_argument("--start-date", default="2010-01-01", help="Start date for FRED data")
    args = parser.parse_args()

    COPPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetch_fred = not args.pmi_only
    fetch_pmi = not args.fred_only

    # --- FRED Data ---
    if fetch_fred:
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            print("WARNING: FRED_API_KEY not set. Skipping FRED data.")
            print("  Register for a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            print("  Then: export FRED_API_KEY=your_key_here")
        else:
            print("Fetching FRED macro data...")
            macro_df = fetch_all_fred(api_key, start_date=args.start_date)
            if not macro_df.empty:
                macro_path = COPPER_DATA_DIR / "macro_data.csv"
                macro_df.to_csv(macro_path)
                print(f"Saved combined macro data ({len(macro_df)} rows) to {macro_path}")

    # --- China PMI ---
    if fetch_pmi:
        print("Building China PMI dataset...")
        pmi_df = build_china_pmi()
        pmi_path = COPPER_DATA_DIR / "china_pmi.csv"
        pmi_df.to_csv(pmi_path)
        print(f"Saved China PMI ({len(pmi_df)} rows) to {pmi_path}")


if __name__ == "__main__":
    main()
