"""Fetch weather data for major sugar-producing regions."""

import time
from datetime import date
from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# Sugar-producing regions — Brazil dominates, India #2, Thailand #3
REGIONS = {
    "brazil_sp": {"lat": -21.20, "lon": -47.80},         # São Paulo state (sugarcane belt)
    "brazil_cerrado": {"lat": -19.00, "lon": -47.00},    # Cerrado (overlaps with coffee)
    "brazil_northeast": {"lat": -9.50, "lon": -36.00},   # Alagoas/Pernambuco
    "india_up": {"lat": 27.00, "lon": 80.00},            # Uttar Pradesh (#1 India state)
    "india_maharashtra": {"lat": 18.50, "lon": 74.00},   # Maharashtra (#2 India state)
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(name, lat, lon, start_date="2016-01-01", end_date=""):
    if not end_date:
        end_date = date.today().isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,et0_fao_evapotranspiration",
        "timezone": "auto",
    }
    print(f"  Fetching weather for {name} ({lat}, {lon})...")
    time.sleep(5)
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date")
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df


def compute_aggregates(df):
    brazil_temp = [c for c in df.columns if "brazil" in c and "temperature_2m_mean" in c]
    india_temp = [c for c in df.columns if "india" in c and "temperature_2m_mean" in c]
    brazil_precip = [c for c in df.columns if "brazil" in c and "precipitation_sum" in c]
    india_precip = [c for c in df.columns if "india" in c and "precipitation_sum" in c]

    df["brazil_avg_temp"] = df[brazil_temp].mean(axis=1) if brazil_temp else 0
    df["india_avg_temp"] = df[india_temp].mean(axis=1) if india_temp else 0
    df["brazil_total_precip"] = df[brazil_precip].sum(axis=1) if brazil_precip else 0
    df["india_total_precip"] = df[india_precip].sum(axis=1) if india_precip else 0

    for window in [7, 30]:
        df[f"brazil_temp_{window}d_avg"] = df["brazil_avg_temp"].rolling(window).mean()
        df[f"india_temp_{window}d_avg"] = df["india_avg_temp"].rolling(window).mean()
        df[f"brazil_precip_{window}d_sum"] = df["brazil_total_precip"].rolling(window).sum()
        df[f"india_precip_{window}d_sum"] = df["india_total_precip"].rolling(window).sum()

    df["brazil_temp_anomaly"] = df["brazil_avg_temp"] - df["brazil_temp_30d_avg"]
    df["india_temp_anomaly"] = df["india_avg_temp"] - df["india_temp_30d_avg"]
    df["brazil_drought_30d"] = (df["brazil_precip_30d_sum"] < 20).astype(int)
    df["india_drought_30d"] = (df["india_precip_30d_sum"] < 10).astype(int)

    # Frost risk for Brazil sugarcane
    brazil_min = [c for c in df.columns if "brazil" in c and "temperature_2m_min" in c]
    if brazil_min:
        df["brazil_frost_risk"] = (df[brazil_min].min(axis=1) < 3).astype(int)

    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    region_dfs = []
    for name, coords in REGIONS.items():
        try:
            df = fetch_region_weather(name, **coords)
            region_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Skipping {name}: {e}")
    if not region_dfs:
        print("ERROR: No weather data fetched.")
        return
    weather = region_dfs[0]
    for df in region_dfs[1:]:
        weather = weather.join(df, how="outer")
    weather = compute_aggregates(weather)
    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"\nSaved {len(weather)} days to {DATA_DIR / 'weather.csv'}")
    print(f"Columns: {len(weather.columns)}")


if __name__ == "__main__":
    main()
