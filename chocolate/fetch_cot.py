"""Fetch CFTC Commitment of Traders data for cocoa futures."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"
YEAR_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

COCOA_CODE = 73  # CFTC Commodity Code for Cocoa


def download_cot_zip(url: str) -> pd.DataFrame:
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def extract_cocoa_cot(df: pd.DataFrame) -> pd.DataFrame:
    cocoa = df[df["CFTC Commodity Code"] == COCOA_CODE].copy()
    cocoa["Date"] = pd.to_datetime(cocoa["As of Date in Form YYYY-MM-DD"])
    cocoa = cocoa.sort_values("Date")

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
    result = cocoa[list(cols.keys())].rename(columns=cols)

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

    all_dfs = [download_cot_zip(HISTORICAL_URL)]
    for year in range(2017, 2027):
        url = YEAR_URL_TEMPLATE.format(year=year)
        try:
            all_dfs.append(download_cot_zip(url))
        except Exception as e:
            print(f"  Skipping {year}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["As of Date in Form YYYY-MM-DD", "CFTC Commodity Code"], keep="last"
    )

    cocoa_cot = extract_cocoa_cot(combined)
    cocoa_cot.to_csv(DATA_DIR / "cocoa_cot.csv")
    print(f"\nSaved {len(cocoa_cot)} COT records to {DATA_DIR / 'cocoa_cot.csv'}")
    print(f"Date range: {cocoa_cot.index.min()} to {cocoa_cot.index.max()}")
    print(f"\nLatest positioning:")
    print(cocoa_cot.iloc[-1][["cot_net_spec", "cot_net_comm", "cot_spec_ratio"]])


if __name__ == "__main__":
    main()
