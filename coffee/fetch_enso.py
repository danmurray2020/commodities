"""Fetch ENSO (El Niño / La Niña) index data from NOAA."""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# NOAA Oceanic Niño Index (ONI) - the standard ENSO measure
# Monthly SST anomalies in the Niño 3.4 region
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Multivariate ENSO Index v2 (MEI.v2) - more comprehensive
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"


def fetch_oni() -> pd.DataFrame:
    """Fetch the Oceanic Niño Index from NOAA CPC."""
    print("Fetching ONI data from NOAA...")
    resp = requests.get(ONI_URL, timeout=30)
    resp.raise_for_status()

    # Parse the fixed-width text format
    lines = resp.text.strip().split("\n")
    records = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 4:
            try:
                season = parts[0]  # e.g., "DJF", "JFM", etc.
                year = int(parts[1])
                anom = float(parts[3])  # ANOM column
                # Map season to approximate month
                season_months = {
                    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
                    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
                    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
                }
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
    """Fetch the Multivariate ENSO Index v2 from NOAA PSL."""
    print("Fetching MEI.v2 data from NOAA...")
    resp = requests.get(MEI_URL, timeout=30)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    records = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        # 12 bimonthly values: DJ, JF, FM, MA, AM, MJ, JJ, JA, AS, SO, ON, ND
        for i, val in enumerate(parts[1:13]):
            try:
                mei_val = float(val)
                if mei_val > -90:  # filter missing values
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

    # Merge ONI and MEI
    enso = oni.join(mei, how="outer")

    # Add derived features
    # ENSO state classification
    enso["enso_state"] = 0  # neutral
    enso.loc[enso["oni"] >= 0.5, "enso_state"] = 1   # El Niño
    enso.loc[enso["oni"] <= -0.5, "enso_state"] = -1  # La Niña

    # Rate of change (ENSO transitioning)
    enso["oni_change_3m"] = enso["oni"].diff(3)
    enso["mei_change_3m"] = enso["mei"].diff(3)

    enso.to_csv(DATA_DIR / "enso.csv")
    print(f"Saved {len(enso)} ENSO records to {DATA_DIR / 'enso.csv'}")
    print(f"Date range: {enso.index.min()} to {enso.index.max()}")
    print(f"\nLatest ENSO state:")
    print(enso.dropna().iloc[-1])


if __name__ == "__main__":
    main()
