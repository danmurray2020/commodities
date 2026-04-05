"""Fetch historical weather data for major coffee-producing regions via Open-Meteo."""

from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# Key coffee-producing regions with coordinates
REGIONS = {
    "brazil_sul_minas": {"lat": -21.75, "lon": -45.25},    # Largest arabica region
    "brazil_cerrado": {"lat": -19.00, "lon": -47.00},      # High-quality arabica
    "brazil_mogiana": {"lat": -21.20, "lon": -47.80},      # São Paulo belt
    "vietnam_daklak": {"lat": 12.67, "lon": 108.05},       # #1 robusta province
    "vietnam_lamdong": {"lat": 11.94, "lon": 108.44},      # Higher altitude
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(
    name: str,
    lat: float,
    lon: float,
    start_date: str = "2016-01-01",
    end_date: str = "",
) -> pd.DataFrame:
    """Fetch daily weather data for a single region from Open-Meteo."""
    if not end_date:
        end_date = date.today().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "auto",
    }
    print(f"  Fetching weather for {name} ({lat}, {lon})...")
    import time
    time.sleep(1)  # rate limit: avoid 429s from Open-Meteo
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date")

    # Prefix columns with region name
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df


def compute_aggregate_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate weather features across all regions."""
    df = weather_df.copy()

    # Average temperature across all Brazil regions
    brazil_temp_cols = [c for c in df.columns if "brazil" in c and "temperature_2m_mean" in c]
    vietnam_temp_cols = [c for c in df.columns if "vietnam" in c and "temperature_2m_mean" in c]
    brazil_precip_cols = [c for c in df.columns if "brazil" in c and "precipitation_sum" in c]
    vietnam_precip_cols = [c for c in df.columns if "vietnam" in c and "precipitation_sum" in c]

    df["brazil_avg_temp"] = df[brazil_temp_cols].mean(axis=1)
    df["vietnam_avg_temp"] = df[vietnam_temp_cols].mean(axis=1)
    df["brazil_total_precip"] = df[brazil_precip_cols].sum(axis=1)
    df["vietnam_total_precip"] = df[vietnam_precip_cols].sum(axis=1)

    # Rolling weather features (7-day, 30-day)
    for window in [7, 30]:
        df[f"brazil_temp_{window}d_avg"] = df["brazil_avg_temp"].rolling(window).mean()
        df[f"vietnam_temp_{window}d_avg"] = df["vietnam_avg_temp"].rolling(window).mean()
        df[f"brazil_precip_{window}d_sum"] = df["brazil_total_precip"].rolling(window).sum()
        df[f"vietnam_precip_{window}d_sum"] = df["vietnam_total_precip"].rolling(window).sum()

    # Temperature anomalies (deviation from 30-day rolling mean)
    df["brazil_temp_anomaly"] = df["brazil_avg_temp"] - df[f"brazil_temp_30d_avg"]
    df["vietnam_temp_anomaly"] = df["vietnam_avg_temp"] - df[f"vietnam_temp_30d_avg"]

    # Frost risk indicator for Brazil (min temp < 2°C in any region)
    brazil_min_cols = [c for c in df.columns if "brazil" in c and "temperature_2m_min" in c]
    df["brazil_frost_risk"] = (df[brazil_min_cols].min(axis=1) < 2).astype(int)

    # Drought indicator (low precipitation over 30 days)
    df["brazil_drought_30d"] = (df["brazil_precip_30d_sum"] < 20).astype(int)

    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Fetch weather for each region
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

    # Merge all regions on date
    weather = region_dfs[0]
    for df in region_dfs[1:]:
        weather = weather.join(df, how="outer")

    # Compute aggregate features
    weather = compute_aggregate_features(weather)

    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"\nSaved {len(weather)} days of weather data to {DATA_DIR / 'weather.csv'}")
    print(f"Date range: {weather.index.min()} to {weather.index.max()}")
    print(f"Columns: {len(weather.columns)}")

    # Summary stats
    print(f"\nBrazil frost risk days: {weather['brazil_frost_risk'].sum()}")
    print(f"Brazil drought days (30d): {weather['brazil_drought_30d'].sum()}")


if __name__ == "__main__":
    main()
