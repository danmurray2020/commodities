"""Fetch CFTC Commitment of Traders data for coffee futures."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# CFTC bulk CSV URLs (Legacy Futures Only, text format)
HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"
YEAR_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

COFFEE_CODE = 83  # CFTC Commodity Code for Coffee C


def download_cot_zip(url: str) -> pd.DataFrame:
    """Download and parse a CFTC COT ZIP file."""
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def extract_coffee_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean COT data for coffee futures."""
    coffee = df[df["CFTC Commodity Code"] == COFFEE_CODE].copy()

    # Parse date
    coffee["Date"] = pd.to_datetime(coffee["As of Date in Form YYYY-MM-DD"])
    coffee = coffee.sort_values("Date")

    # Extract key positioning columns
    cols = {
        "Date": "Date",
        "Open Interest (All)": "cot_open_interest",
        "Noncommercial Positions-Long (All)": "cot_noncomm_long",
        "Noncommercial Positions-Short (All)": "cot_noncomm_short",
        "Commercial Positions-Long (All)": "cot_comm_long",
        "Commercial Positions-Short (All)": "cot_comm_short",
        "Change in Open Interest (All)": "cot_oi_change",
        "Change in Noncommercial-Long (All)": "cot_noncomm_long_chg",
        "Change in Noncommercial-Short (All)": "cot_noncomm_short_chg",
    }
    result = coffee[list(cols.keys())].rename(columns=cols)

    # Derived features
    result["cot_net_spec"] = result["cot_noncomm_long"] - result["cot_noncomm_short"]
    result["cot_net_comm"] = result["cot_comm_long"] - result["cot_comm_short"]
    result["cot_spec_ratio"] = result["cot_noncomm_long"] / (
        result["cot_noncomm_long"] + result["cot_noncomm_short"]
    )
    result["cot_comm_net_pct_oi"] = result["cot_net_comm"] / result["cot_open_interest"]

    result = result.set_index("Date")
    return result


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Fetch historical archive + individual years 2017-2026
    all_dfs = [download_cot_zip(HISTORICAL_URL)]
    for year in range(2017, 2027):
        url = YEAR_URL_TEMPLATE.format(year=year)
        try:
            all_dfs.append(download_cot_zip(url))
        except Exception as e:
            print(f"  Skipping {year}: {e}")

    # Combine and deduplicate
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["As of Date in Form YYYY-MM-DD", "CFTC Commodity Code"], keep="last"
    )

    coffee_cot = extract_coffee_cot(combined)
    coffee_cot.to_csv(DATA_DIR / "coffee_cot.csv")
    print(f"\nSaved {len(coffee_cot)} COT records to {DATA_DIR / 'coffee_cot.csv'}")
    print(f"Date range: {coffee_cot.index.min()} to {coffee_cot.index.max()}")
    print(f"\nLatest positioning:")
    print(coffee_cot.iloc[-1][["cot_net_spec", "cot_net_comm", "cot_spec_ratio"]])


if __name__ == "__main__":
    main()
