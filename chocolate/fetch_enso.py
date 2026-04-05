"""Fetch ENSO (El Nino / La Nina) index data from NOAA."""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"


def fetch_oni() -> pd.DataFrame:
    print("Fetching ONI data from NOAA...")
    resp = requests.get(ONI_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    records = []
    season_months = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                season = parts[0]
                year = int(parts[1])
                anom = float(parts[3])
                month = season_months.get(season)
                if month:
                    records.append({"year": year, "month": month, "oni": anom})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.set_index("Date")[["oni"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_mei() -> pd.DataFrame:
    print("Fetching MEI.v2 data from NOAA...")
    resp = requests.get(MEI_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for i, val in enumerate(parts[1:13]):
            try:
                mei_val = float(val)
                if mei_val > -90:
                    records.append({"year": year, "month": i + 1, "mei": mei_val})
            except ValueError:
                continue
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.set_index("Date")[["mei"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    oni = fetch_oni()
    mei = fetch_mei()
    enso = oni.join(mei, how="outer")
    enso["enso_state"] = 0
    enso.loc[enso["oni"] >= 0.5, "enso_state"] = 1
    enso.loc[enso["oni"] <= -0.5, "enso_state"] = -1
    enso["oni_change_3m"] = enso["oni"].diff(3)
    enso["mei_change_3m"] = enso["mei"].diff(3)
    enso.to_csv(DATA_DIR / "enso.csv")
    print(f"Saved {len(enso)} ENSO records to {DATA_DIR / 'enso.csv'}")
    print(f"Date range: {enso.index.min()} to {enso.index.max()}")
    print(f"\nLatest ENSO state:")
    print(enso.dropna().iloc[-1])


if __name__ == "__main__":
    main()
