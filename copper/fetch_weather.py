"""Fetch weather data for Copper producing/demand regions."""

import time
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np, requests

DATA_DIR = Path(__file__).parent / "data"
REGIONS = {
        "chile_santiago": {"lat": -33.45, "lon": -70.65},    # Chile (#1 producer, 28%)
        "chile_antofagasta": {"lat": -23.65, "lon": -70.40}, # Atacama mining region
        "peru_lima": {"lat": -12.05, "lon": -77.05},         # Peru (#2 producer)
        "drc_lubumbashi": {"lat": -11.66, "lon": 27.47},     # DRC (#3 producer, cobalt too)
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
    
    # Copper is less weather-sensitive than agriculture, but mining disruptions matter
    chile_precip = [c for c in df.columns if "chile" in c and "precipitation_sum" in c]
    peru_precip = [c for c in df.columns if "peru" in c and "precipitation_sum" in c]

    if chile_precip:
        df["chile_total_precip"] = df[chile_precip].sum(axis=1)
        for window in [7, 30]:
            df[f"chile_precip_{window}d_sum"] = df["chile_total_precip"].rolling(window).sum()
        # Heavy rain disrupts open-pit mining
        df["chile_rain_disruption"] = (df["chile_precip_7d_sum"] > 20).astype(int)
    if peru_precip:
        df["peru_total_precip"] = df[peru_precip].sum(axis=1)
        for window in [7, 30]:
            df[f"peru_precip_{window}d_sum"] = df["peru_total_precip"].rolling(window).sum()
    
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
        print("ERROR: No weather data fetched."); return
    weather = region_dfs[0]
    for d in region_dfs[1:]:
        weather = weather.join(d, how="outer")
    weather = compute_aggregates(weather)
    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"Saved {len(weather)} days to {DATA_DIR / 'weather.csv'}")


if __name__ == "__main__":
    main()
