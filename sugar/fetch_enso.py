"""Fetch ENSO data — same as coffee/cocoa, sugar is also ENSO-sensitive."""

from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"


def fetch_oni():
    print("Fetching ONI...")
    resp = requests.get(ONI_URL, timeout=30); resp.raise_for_status()
    records = []
    season_months = {"DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,"JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12}
    for line in resp.text.strip().split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                month = season_months.get(parts[0])
                if month:
                    records.append({"year": int(parts[1]), "month": month, "oni": float(parts[3])})
            except (ValueError, IndexError): continue
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df[["year","month"]].assign(day=1))
    return df.set_index("Date")[["oni"]].sort_index().pipe(lambda d: d[~d.index.duplicated(keep="last")])


def fetch_mei():
    print("Fetching MEI.v2...")
    resp = requests.get(MEI_URL, timeout=30); resp.raise_for_status()
    records = []
    for line in resp.text.strip().split("\n")[1:]:
        parts = line.split()
        try: year = int(parts[0])
        except ValueError: continue
        for i, val in enumerate(parts[1:13]):
            try:
                v = float(val)
                if v > -90: records.append({"year": year, "month": i+1, "mei": v})
            except ValueError: continue
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df[["year","month"]].assign(day=1))
    return df.set_index("Date")[["mei"]].sort_index().pipe(lambda d: d[~d.index.duplicated(keep="last")])


def main():
    DATA_DIR.mkdir(exist_ok=True)
    enso = fetch_oni().join(fetch_mei(), how="outer")
    enso["enso_state"] = 0
    enso.loc[enso["oni"] >= 0.5, "enso_state"] = 1
    enso.loc[enso["oni"] <= -0.5, "enso_state"] = -1
    enso["oni_change_3m"] = enso["oni"].diff(3)
    enso["mei_change_3m"] = enso["mei"].diff(3)
    enso.to_csv(DATA_DIR / "enso.csv")
    print(f"Saved {len(enso)} records to {DATA_DIR / 'enso.csv'}")


if __name__ == "__main__":
    main()
