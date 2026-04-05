"""Fetch weather data for Wheat producing/demand regions."""

import time
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np, requests

DATA_DIR = Path(__file__).parent / "data"
REGIONS = {
        "us_kansas": {"lat": 38.50, "lon": -98.50},         # Kansas (#1 US wheat state)
        "us_ndakota": {"lat": 47.50, "lon": -100.50},       # North Dakota (spring wheat)
        "france_beauce": {"lat": 48.10, "lon": 1.50},       # Beauce (EU #1 producer)
        "russia_kuban": {"lat": 45.30, "lon": 39.00},       # Krasnodar (Russia wheat belt)
        "australia_nsw": {"lat": -33.00, "lon": 148.00},    # New South Wales
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
    
    us_temp = [c for c in df.columns if ("kansas" in c or "ndakota" in c) and "temperature_2m_mean" in c]
    intl_temp = [c for c in df.columns if ("france" in c or "russia" in c or "australia" in c) and "temperature_2m_mean" in c]
    us_precip = [c for c in df.columns if ("kansas" in c or "ndakota" in c) and "precipitation_sum" in c]

    if us_temp: df["us_avg_temp"] = df[us_temp].mean(axis=1)
    if intl_temp: df["intl_avg_temp"] = df[intl_temp].mean(axis=1)
    if us_precip: df["us_total_precip"] = df[us_precip].sum(axis=1)

    for region in ["us", "intl"]:
        for window in [7, 30]:
            if f"{region}_avg_temp" in df.columns:
                df[f"{region}_temp_{window}d_avg"] = df[f"{region}_avg_temp"].rolling(window).mean()
        if f"{region}_avg_temp" in df.columns and f"{region}_temp_30d_avg" in df.columns:
            df[f"{region}_temp_anomaly"] = df[f"{region}_avg_temp"] - df[f"{region}_temp_30d_avg"]
    if "us_total_precip" in df.columns:
        for window in [7, 30]:
            df[f"us_precip_{window}d_sum"] = df["us_total_precip"].rolling(window).sum()
        df["us_drought_30d"] = (df["us_precip_30d_sum"] < 15).astype(int)

    # Frost risk for winter wheat
    us_min = [c for c in df.columns if ("kansas" in c) and "temperature_2m_min" in c]
    if us_min:
        df["us_frost_risk"] = (df[us_min].min(axis=1) < -15).astype(int)
    
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
