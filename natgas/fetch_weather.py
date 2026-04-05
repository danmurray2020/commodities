"""Fetch weather data for key US natural gas demand regions.

Natural gas demand is driven by heating (winter) and cooling (summer) degree days.
Key regions: US Northeast (heating), US South (cooling/power gen), US Midwest (heating).
"""

import time
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np, requests

DATA_DIR = Path(__file__).parent / "data"

# Population-weighted demand centers
REGIONS = {
    "us_northeast": {"lat": 40.70, "lon": -74.00},      # NYC metro (largest heating market)
    "us_chicago": {"lat": 41.88, "lon": -87.63},         # Chicago (Midwest heating)
    "us_houston": {"lat": 29.76, "lon": -95.37},         # Houston (gas production + cooling)
    "us_atlanta": {"lat": 33.75, "lon": -84.39},         # Southeast (growing demand)
    "us_denver": {"lat": 39.74, "lon": -104.99},         # Mountain West (heating)
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(name, lat, lon, start_date="2016-01-01", end_date=""):
    if not end_date:
        end_date = date.today().isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "America/New_York",
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
    # Average temp across all regions
    temp_cols = [c for c in df.columns if "temperature_2m_mean" in c]
    min_temp_cols = [c for c in df.columns if "temperature_2m_min" in c]
    max_temp_cols = [c for c in df.columns if "temperature_2m_max" in c]

    df["us_avg_temp"] = df[temp_cols].mean(axis=1)
    df["us_min_temp"] = df[min_temp_cols].min(axis=1)
    df["us_max_temp"] = df[max_temp_cols].max(axis=1)

    # Heating Degree Days (HDD) — base 65°F (18.3°C)
    # HDD = max(0, 18.3 - avg_temp) — higher = more heating demand
    df["hdd_daily"] = np.maximum(0, 18.3 - df["us_avg_temp"])
    df["cdd_daily"] = np.maximum(0, df["us_avg_temp"] - 18.3)  # Cooling Degree Days

    # Rolling HDD/CDD
    for window in [7, 14, 30]:
        df[f"hdd_{window}d_sum"] = df["hdd_daily"].rolling(window).sum()
        df[f"cdd_{window}d_sum"] = df["cdd_daily"].rolling(window).sum()
        df[f"us_temp_{window}d_avg"] = df["us_avg_temp"].rolling(window).mean()

    # Temperature anomaly (vs 30-day rolling mean)
    df["temp_anomaly"] = df["us_avg_temp"] - df["us_temp_30d_avg"]

    # Extreme cold/heat indicators
    df["extreme_cold"] = (df["us_min_temp"] < -10).astype(int)  # polar vortex risk
    df["extreme_heat"] = (df["us_max_temp"] > 35).astype(int)   # AC demand surge
    df["cold_snap_7d"] = df["extreme_cold"].rolling(7).sum()    # multi-day cold events

    # HDD anomaly (vs historical same-period norms)
    # Use deviation from 30-day rolling average as proxy
    df["hdd_anomaly_30d"] = df["hdd_daily"] - df["hdd_daily"].rolling(30).mean()

    # Wind chill effect on Northeast
    ne_wind = [c for c in df.columns if "northeast" in c and "wind" in c]
    ne_temp = [c for c in df.columns if "northeast" in c and "temperature_2m_min" in c]
    if ne_wind and ne_temp:
        df["northeast_windchill_proxy"] = df[ne_temp[0]] - df[ne_wind[0]] * 0.5

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
    for df in region_dfs[1:]:
        weather = weather.join(df, how="outer")
    weather = compute_aggregates(weather)
    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"\nSaved {len(weather)} days to {DATA_DIR / 'weather.csv'}, {len(weather.columns)} columns")


if __name__ == "__main__":
    main()
