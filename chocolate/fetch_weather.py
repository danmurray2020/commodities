"""Fetch historical weather data for major cocoa-producing regions."""

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# Key cocoa-producing regions
# ~70% of world cocoa comes from Ivory Coast + Ghana
REGIONS = {
    "ivorycoast_east": {"lat": 6.50, "lon": -4.00},     # Eastern IC (biggest production zone)
    "ivorycoast_west": {"lat": 6.00, "lon": -6.50},     # Western IC
    "ghana_ashanti": {"lat": 6.70, "lon": -1.60},       # Ashanti region (Ghana's cocoa belt)
    "ghana_western": {"lat": 5.50, "lon": -2.30},       # Western Ghana
    "indonesia_sulawesi": {"lat": -1.50, "lon": 121.00}, # Sulawesi (#3 global producer)
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(
    name: str, lat: float, lon: float,
    start_date: str = "2016-01-01", end_date: str = "",
) -> pd.DataFrame:
    if not end_date:
        end_date = date.today().isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "rain_sum", "et0_fao_evapotranspiration",
        ]),
        "timezone": "auto",
    }
    print(f"  Fetching weather for {name} ({lat}, {lon})...")
    time.sleep(5)  # rate limit
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date")
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df


def compute_aggregate_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    westafrica_temp_cols = [c for c in df.columns if ("ivorycoast" in c or "ghana" in c) and "temperature_2m_mean" in c]
    indonesia_temp_cols = [c for c in df.columns if "indonesia" in c and "temperature_2m_mean" in c]
    westafrica_precip_cols = [c for c in df.columns if ("ivorycoast" in c or "ghana" in c) and "precipitation_sum" in c]
    indonesia_precip_cols = [c for c in df.columns if "indonesia" in c and "precipitation_sum" in c]

    df["westafrica_avg_temp"] = df[westafrica_temp_cols].mean(axis=1)
    df["indonesia_avg_temp"] = df[indonesia_temp_cols].mean(axis=1) if indonesia_temp_cols else 0
    df["westafrica_total_precip"] = df[westafrica_precip_cols].sum(axis=1)
    df["indonesia_total_precip"] = df[indonesia_precip_cols].sum(axis=1) if indonesia_precip_cols else 0

    for window in [7, 30]:
        df[f"westafrica_temp_{window}d_avg"] = df["westafrica_avg_temp"].rolling(window).mean()
        df[f"indonesia_temp_{window}d_avg"] = df["indonesia_avg_temp"].rolling(window).mean()
        df[f"westafrica_precip_{window}d_sum"] = df["westafrica_total_precip"].rolling(window).sum()
        df[f"indonesia_precip_{window}d_sum"] = df["indonesia_total_precip"].rolling(window).sum()

    df["westafrica_temp_anomaly"] = df["westafrica_avg_temp"] - df["westafrica_temp_30d_avg"]
    df["indonesia_temp_anomaly"] = df["indonesia_avg_temp"] - df["indonesia_temp_30d_avg"]

    # Harmattan dry season indicator (Dec-Feb, key risk for W. Africa cocoa)
    westafrica_min_cols = [c for c in df.columns if ("ivorycoast" in c or "ghana" in c) and "temperature_2m_min" in c]
    df["westafrica_heat_risk"] = (df[westafrica_temp_cols].max(axis=1) > 34).astype(int)

    # Drought: low precip over 30 days during growing season
    df["westafrica_drought_30d"] = (df["westafrica_precip_30d_sum"] < 30).astype(int)

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

    weather = compute_aggregate_features(weather)
    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"\nSaved {len(weather)} days of weather data to {DATA_DIR / 'weather.csv'}")
    print(f"Date range: {weather.index.min()} to {weather.index.max()}")
    print(f"Columns: {len(weather.columns)}")
    print(f"\nWest Africa heat risk days: {weather['westafrica_heat_risk'].sum()}")
    print(f"West Africa drought days (30d): {weather['westafrica_drought_30d'].sum()}")


if __name__ == "__main__":
    main()
