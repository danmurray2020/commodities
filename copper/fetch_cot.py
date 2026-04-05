"""Fetch CFTC COT data for copper futures."""

import io, zipfile
from pathlib import Path
import pandas as pd, requests

DATA_DIR = Path(__file__).parent / "data"
HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"
YEAR_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
COMMODITY_CODE = 85


def download_cot_zip(url):
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=120); resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            return pd.read_csv(f, low_memory=False)


def extract_cot(df):
    commodity = df[df["CFTC Commodity Code"] == COMMODITY_CODE].copy()
    commodity["Date"] = pd.to_datetime(commodity["As of Date in Form YYYY-MM-DD"])
    commodity = commodity.sort_values("Date")
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
    result = commodity[list(cols.keys())].rename(columns=cols)
    result["cot_net_spec"] = result["cot_noncomm_long"] - result["cot_noncomm_short"]
    result["cot_net_comm"] = result["cot_comm_long"] - result["cot_comm_short"]
    result["cot_spec_ratio"] = result["cot_noncomm_long"] / (result["cot_noncomm_long"] + result["cot_noncomm_short"])
    result["cot_comm_net_pct_oi"] = result["cot_net_comm"] / result["cot_open_interest"]
    return result.set_index("Date")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    all_dfs = [download_cot_zip(HISTORICAL_URL)]
    for year in range(2017, 2027):
        try: all_dfs.append(download_cot_zip(YEAR_URL_TEMPLATE.format(year=year)))
        except Exception as e: print(f"  Skipping {year}: {e}")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["As of Date in Form YYYY-MM-DD", "CFTC Commodity Code"], keep="last")
    cot = extract_cot(combined)
    cot.to_csv(DATA_DIR / "copper_cot.csv")
    print(f"Saved {len(cot)} COT records to {DATA_DIR / 'copper_cot.csv'}")


if __name__ == "__main__":
    main()
