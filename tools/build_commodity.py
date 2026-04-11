"""Generate a complete commodity prediction project from a config.

Applies all lessons learned:
- Trend features (from sugar)
- scale_pos_weight tuning (from sugar)
- Enhanced seasonality (from natgas)
- Cross-commodity ratios where relevant
- Permutation importance feature selection
- Purged walk-forward CV
- Stability-penalized Optuna
"""

import os
from pathlib import Path

TEMPLATE_FETCH_DATA = '''"""Fetch {name_lower} futures data and supplementary market data."""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def fetch_prices(ticker: str = "{ticker}", period: str = "10y") -> pd.DataFrame:
    print(f"Fetching {{ticker}} data for the last {{period}}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df = df.dropna()
    print(f"Fetched {{len(df)}} rows from {{df.index.min()}} to {{df.index.max()}}")
    return df


def fetch_supplementary_data() -> dict[str, pd.DataFrame]:
    tickers = {supp_tickers}
    supplementary = {{}}
    for name, ticker in tickers.items():
        print(f"Fetching {{name}} ({{ticker}})...")
        df = yf.download(ticker, period="10y", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        if not df.empty:
            supplementary[name] = df[["Close"]].rename(columns={{"Close": name}})
    return supplementary


def main():
    DATA_DIR.mkdir(exist_ok=True)
    prices = fetch_prices()
    prices.to_csv(DATA_DIR / "{name_lower}_prices.csv")

    supplementary = fetch_supplementary_data()
    combined = prices[["Close"]].rename(columns={{"Close": "{price_col}"}})
    for name, df in supplementary.items():
        combined = combined.join(df, how="left")
    combined = combined.ffill()
    combined.to_csv(DATA_DIR / "combined_features.csv")
    print(f"Saved combined features to {{DATA_DIR / 'combined_features.csv'}}")


if __name__ == "__main__":
    main()
'''

TEMPLATE_FETCH_COT = '''"""Fetch CFTC COT data for {name_lower} futures."""

import io, zipfile
from pathlib import Path
import pandas as pd, requests

DATA_DIR = Path(__file__).parent / "data"
HISTORICAL_URL = "https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"
YEAR_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{{year}}.zip"
COMMODITY_CODE = {cftc_code}


def download_cot_zip(url):
    print(f"Downloading {{url}}...")
    resp = requests.get(url, timeout=120); resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            return pd.read_csv(f, low_memory=False)


def extract_cot(df):
    commodity = df[df["CFTC Commodity Code"] == COMMODITY_CODE].copy()
    commodity["Date"] = pd.to_datetime(commodity["As of Date in Form YYYY-MM-DD"])
    commodity = commodity.sort_values("Date")
    cols = {{
        "Date": "Date",
        "Open Interest (All)": "cot_open_interest",
        "Noncommercial Positions-Long (All)": "cot_noncomm_long",
        "Noncommercial Positions-Short (All)": "cot_noncomm_short",
        "Commercial Positions-Long (All)": "cot_comm_long",
        "Commercial Positions-Short (All)": "cot_comm_short",
        "Change in Open Interest (All)": "cot_oi_change",
        "Change in Noncommercial-Long (All)": "cot_noncomm_long_chg",
        "Change in Noncommercial-Short (All)": "cot_noncomm_short_chg",
    }}
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
        except Exception as e: print(f"  Skipping {{year}}: {{e}}")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["As of Date in Form YYYY-MM-DD", "CFTC Commodity Code"], keep="last")
    cot = extract_cot(combined)
    cot.to_csv(DATA_DIR / "{name_lower}_cot.csv")
    print(f"Saved {{len(cot)}} COT records to {{DATA_DIR / '{name_lower}_cot.csv'}}")


if __name__ == "__main__":
    main()
'''

TEMPLATE_FETCH_WEATHER = '''"""Fetch weather data for {name} producing/demand regions."""

import time
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np, requests

DATA_DIR = Path(__file__).parent / "data"
REGIONS = {weather_regions}
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(name, lat, lon, start_date="2016-01-01", end_date=""):
    if not end_date:
        end_date = date.today().isoformat()
    params = {{
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,et0_fao_evapotranspiration",
        "timezone": "auto",
    }}
    print(f"  Fetching weather for {{name}} ({{lat}}, {{lon}})...")
    time.sleep(5)
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date")
    df.columns = [f"{{name}}_{{col}}" for col in df.columns]
    return df


def compute_aggregates(df):
    {weather_aggregates}
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    region_dfs = []
    for name, coords in REGIONS.items():
        try:
            df = fetch_region_weather(name, **coords)
            region_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Skipping {{name}}: {{e}}")
    if not region_dfs:
        print("ERROR: No weather data fetched."); return
    weather = region_dfs[0]
    for d in region_dfs[1:]:
        weather = weather.join(d, how="outer")
    weather = compute_aggregates(weather)
    weather.to_csv(DATA_DIR / "weather.csv")
    print(f"Saved {{len(weather)}} days to {{DATA_DIR / 'weather.csv'}}")


if __name__ == "__main__":
    main()
'''

TEMPLATE_FEATURES = '''"""Feature engineering for {name_lower} price prediction."""

import pandas as pd
import numpy as np
from agents.indicators import rsi, MACD, BollingerBands, average_true_range


def add_price_features(df: pd.DataFrame, price_col: str = "{price_col}") -> pd.DataFrame:
    df = df.copy()
    price = df[price_col]

    for lag in [1, 5, 10, 21]:
        df[f"return_{{lag}}d"] = price.pct_change(lag)
    for window in [5, 10, 21, 50, 200]:
        df[f"sma_{{window}}"] = price.rolling(window).mean()
        df[f"price_vs_sma_{{window}}"] = price / df[f"sma_{{window}}"] - 1
    for window in [10, 21, 63]:
        df[f"volatility_{{window}}d"] = price.pct_change().rolling(window).std() * np.sqrt(252)

    df["rsi_14"] = rsi(price, window=14)
    macd = MACD(price)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(price, window=20)
    df["bb_pct"] = bb.bollinger_pband()

    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{{lag}}"] = price.shift(lag)

    if isinstance(df.index, pd.DatetimeIndex):
        df["month"] = df.index.month
        day_of_year = df.index.dayofyear
        for h in [1, 2]:
            df[f"season_sin_{{h}}"] = np.sin(2 * np.pi * h * day_of_year / 365.25)
            df[f"season_cos_{{h}}"] = np.cos(2 * np.pi * h * day_of_year / 365.25)
        {season_flags}

    # Mean reversion
    for window in [126, 252]:
        rm = price.rolling(window).mean()
        rs = price.rolling(window).std()
        df[f"zscore_{{window}}d"] = (price - rm) / rs
    if "zscore_252d" in df.columns:
        df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)
        df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
        df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

    # Trend features (learned from sugar)
    daily_ret = price.pct_change()
    for window in [21, 63, 126]:
        df[f"pct_up_days_{{window}}d"] = daily_ret.rolling(window).apply(lambda x: (x > 0).mean())
        df[f"trend_slope_{{window}}d"] = price.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0)
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["sma_50_200_gap"] = df["sma_50"] / df["sma_200"] - 1
    ret_21d = price.pct_change(21)
    df["momentum_rank_252d"] = ret_21d.rolling(252).rank(pct=True)

    {custom_features}

    return df


def build_target(df, price_col="{price_col}", horizon=63):
    df = df.copy()
    future = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def merge_cot_data(df, cot_path="data/{name_lower}_cot.csv"):
    try: cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    df = df.join(cot, how="left")
    cot_cols = [c for c in cot.columns if c in df.columns]
    df[cot_cols] = df[cot_cols].ffill()
    return df


def merge_weather_data(df, weather_path="data/weather.csv"):
    try: weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    agg_cols = [c for c in weather.columns if any(k in c for k in [
        "avg_temp", "total_precip", "_7d_", "_30d_", "anomaly", "drought", "frost", "hdd", "cdd"])]
    if agg_cols:
        weather = weather[agg_cols]
        df = df.join(weather, how="left")
        df[agg_cols] = df[agg_cols].ffill()
    return df


def merge_enso_data(df, enso_path="data/enso.csv"):
    try: enso = pd.read_csv(enso_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    df = df.join(enso, how="left")
    ecols = [c for c in enso.columns if c in df.columns]
    df[ecols] = df[ecols].ffill()
    return df


def prepare_dataset(csv_path="data/combined_features.csv", horizon=63,
                    use_cot=True, use_weather=True, use_enso=True):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    if use_cot: df = merge_cot_data(df)
    if use_weather: df = merge_weather_data(df)
    if use_enso: df = merge_enso_data(df)
    df = build_target(df, horizon=horizon)
    df = df.dropna()
    exclude = {{"{price_col}", "Open", "High", "Low", "Volume", "target_return", "target_direction"}}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols
'''


def build_commodity(config: dict, base_dir: Path):
    """Generate all files for a commodity project."""
    project_dir = base_dir / config["dir_name"]
    project_dir.mkdir(exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)

    name = config["name"]
    name_lower = config["name_lower"]
    price_col = config["price_col"]

    # fetch_data.py
    with open(project_dir / "fetch_data.py", "w") as f:
        f.write(TEMPLATE_FETCH_DATA.format(**config))

    # fetch_cot.py
    with open(project_dir / "fetch_cot.py", "w") as f:
        f.write(TEMPLATE_FETCH_COT.format(**config))

    # fetch_weather.py
    with open(project_dir / "fetch_weather.py", "w") as f:
        f.write(TEMPLATE_FETCH_WEATHER.format(**config))

    # features.py
    with open(project_dir / "features.py", "w") as f:
        f.write(TEMPLATE_FEATURES.format(**config))

    # fetch_enso.py — copy from natgas (same for all)
    enso_src = base_dir / "natgas" / "fetch_enso.py"
    if enso_src.exists():
        import shutil
        shutil.copy2(enso_src, project_dir / "fetch_enso.py")

    # train.py — copy from chocolate (all-in-one pipeline)
    train_src = base_dir / "chocolate" / "train.py"
    if train_src.exists():
        with open(train_src) as f:
            train_content = f.read()
        train_content = train_content.replace("cocoa_close", price_col)
        train_content = train_content.replace("Cocoa", name)
        train_content = train_content.replace("COCOA", name.upper())
        train_content = train_content.replace("CC=F", config["ticker"])
        # Use wider stops for volatile commodities
        if config.get("wide_stops"):
            train_content = train_content.replace("'stop_loss_pct': 0.10", "'stop_loss_pct': 0.15")
        with open(project_dir / "train.py", "w") as f:
            f.write(train_content)

    # strategy.py — copy from sugar (has scale_pos_weight)
    strat_src = base_dir / "sugar" / "strategy.py"
    if strat_src.exists():
        with open(strat_src) as f:
            strat_content = f.read()
        strat_content = strat_content.replace("sugar_close", price_col)
        with open(project_dir / "strategy.py", "w") as f:
            f.write(strat_content)

    # refresh.py
    with open(project_dir / "refresh.py", "w") as f:
        f.write(f'''"""Refresh all data sources."""
import subprocess, sys

def run(script):
    print(f"\\n{{"="*60}}\\nRunning {{script}}...\\n{{"="*60}}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {{script}} exited with code {{result.returncode}}")

def main():
    run("fetch_data.py")
    run("fetch_cot.py")
    run("fetch_weather.py")
    run("fetch_enso.py")
    print(f"\\nAll data refreshed. Run \\'python3 train.py\\' to train.")

if __name__ == "__main__":
    main()
''')

    # requirements.txt
    with open(project_dir / "requirements.txt", "w") as f:
        f.write("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\nxgboost>=2.0\nyfinance>=0.2\n"
                "matplotlib>=3.7\nta>=0.11\njoblib>=1.3\noptuna>=3.0\nrequests>=2.31\nflask>=3.0\n")

    print(f"Built {name} project at {project_dir}")


# Commodity configurations
SOYBEANS = {
    "name": "Soybeans", "name_lower": "soybeans", "dir_name": "soybeans",
    "ticker": "ZS=F", "price_col": "soybeans_close", "cftc_code": 5,
    "supp_tickers": """{
        "usd_index": "DX-Y.NYB",
        "crude_oil": "CL=F",
        "corn": "ZC=F",           # Competing crop (rotation)
        "wheat": "ZW=F",          # Grain complex
        "brl_usd": "BRLUSD=X",   # Brazil = #1 exporter
        "sp500": "^GSPC",
    }""",
    "weather_regions": """{
        "us_iowa": {"lat": 42.03, "lon": -93.47},           # Iowa (#1 US state)
        "us_illinois": {"lat": 40.63, "lon": -89.40},       # Illinois (#2)
        "brazil_matogrosso": {"lat": -12.60, "lon": -55.60}, # Mato Grosso (#1 Brazil state)
        "brazil_parana": {"lat": -24.00, "lon": -51.50},     # Paraná
        "argentina_buenosaires": {"lat": -35.00, "lon": -60.00}, # Pampas
    }""",
    "weather_aggregates": """
    us_temp = [c for c in df.columns if ("iowa" in c or "illinois" in c) and "temperature_2m_mean" in c]
    brazil_temp = [c for c in df.columns if "brazil" in c and "temperature_2m_mean" in c]
    us_precip = [c for c in df.columns if ("iowa" in c or "illinois" in c) and "precipitation_sum" in c]
    brazil_precip = [c for c in df.columns if "brazil" in c and "precipitation_sum" in c]

    if us_temp: df["us_avg_temp"] = df[us_temp].mean(axis=1)
    if brazil_temp: df["brazil_avg_temp"] = df[brazil_temp].mean(axis=1)
    if us_precip: df["us_total_precip"] = df[us_precip].sum(axis=1)
    if brazil_precip: df["brazil_total_precip"] = df[brazil_precip].sum(axis=1)

    for region in ["us", "brazil"]:
        for window in [7, 30]:
            if f"{region}_avg_temp" in df.columns:
                df[f"{region}_temp_{window}d_avg"] = df[f"{region}_avg_temp"].rolling(window).mean()
            if f"{region}_total_precip" in df.columns:
                df[f"{region}_precip_{window}d_sum"] = df[f"{region}_total_precip"].rolling(window).sum()
        if f"{region}_avg_temp" in df.columns and f"{region}_temp_30d_avg" in df.columns:
            df[f"{region}_temp_anomaly"] = df[f"{region}_avg_temp"] - df[f"{region}_temp_30d_avg"]
        if f"{region}_precip_30d_sum" in df.columns:
            df[f"{region}_drought_30d"] = (df[f"{region}_precip_30d_sum"] < 20).astype(int)
    """,
    "season_flags": """
        # US planting: April-June, US harvest: Sep-Nov
        df["us_planting"] = df.index.month.isin([4, 5, 6]).astype(int)
        df["us_harvest"] = df.index.month.isin([9, 10, 11]).astype(int)
        # Brazil planting: Oct-Dec, harvest: Feb-May
        df["brazil_planting"] = df.index.month.isin([10, 11, 12]).astype(int)
        df["brazil_harvest"] = df.index.month.isin([2, 3, 4, 5]).astype(int)""",
    "custom_features": """
    # BRL (Brazil is #1 soybean exporter)
    if "brl_usd" in df.columns:
        df["brl_return_21d"] = df["brl_usd"].pct_change(21)
        df["brl_vs_sma50"] = df["brl_usd"] / df["brl_usd"].rolling(50).mean() - 1
    # Soybean-corn ratio (key spread for planting decisions)
    if "corn" in df.columns:
        df["soy_corn_ratio"] = price / df["corn"]
        df["soy_corn_ratio_sma21"] = df["soy_corn_ratio"].rolling(21).mean()
    """,
}

WHEAT = {
    "name": "Wheat", "name_lower": "wheat", "dir_name": "wheat",
    "ticker": "ZW=F", "price_col": "wheat_close", "cftc_code": 1,
    "supp_tickers": """{
        "usd_index": "DX-Y.NYB",
        "crude_oil": "CL=F",
        "corn": "ZC=F",           # Competing grain
        "soybeans": "ZS=F",       # Grain complex
        "sp500": "^GSPC",
    }""",
    "weather_regions": """{
        "us_kansas": {"lat": 38.50, "lon": -98.50},         # Kansas (#1 US wheat state)
        "us_ndakota": {"lat": 47.50, "lon": -100.50},       # North Dakota (spring wheat)
        "france_beauce": {"lat": 48.10, "lon": 1.50},       # Beauce (EU #1 producer)
        "russia_kuban": {"lat": 45.30, "lon": 39.00},       # Krasnodar (Russia wheat belt)
        "australia_nsw": {"lat": -33.00, "lon": 148.00},    # New South Wales
    }""",
    "weather_aggregates": """
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
    """,
    "season_flags": """
        # Winter wheat: planted Sep-Oct, harvested Jun-Jul
        df["winter_wheat_growing"] = df.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5]).astype(int)
        df["wheat_harvest"] = df.index.month.isin([6, 7]).astype(int)
        # Spring wheat: planted Apr-May, harvested Aug-Sep
        df["spring_wheat_plant"] = df.index.month.isin([4, 5]).astype(int)""",
    "custom_features": """
    # Wheat-corn spread (substitution effect)
    if "corn" in df.columns:
        df["wheat_corn_ratio"] = price / df["corn"]
        df["wheat_corn_ratio_sma21"] = df["wheat_corn_ratio"].rolling(21).mean()
    """,
}

COPPER = {
    "name": "Copper", "name_lower": "copper", "dir_name": "copper",
    "ticker": "HG=F", "price_col": "copper_close", "cftc_code": 85,
    "wide_stops": True,
    "supp_tickers": """{
        "usd_index": "DX-Y.NYB",
        "crude_oil": "CL=F",
        "sp500": "^GSPC",
        "cny_usd": "CNYUSD=X",    # China = 50%+ of global copper demand
        "iron_ore": "TIO=F",       # Correlated industrial metal
    }""",
    "weather_regions": """{
        "chile_santiago": {"lat": -33.45, "lon": -70.65},    # Chile (#1 producer, 28%)
        "chile_antofagasta": {"lat": -23.65, "lon": -70.40}, # Atacama mining region
        "peru_lima": {"lat": -12.05, "lon": -77.05},         # Peru (#2 producer)
        "drc_lubumbashi": {"lat": -11.66, "lon": 27.47},     # DRC (#3 producer, cobalt too)
    }""",
    "weather_aggregates": """
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
    """,
    "season_flags": """
        # Copper is less seasonal but China construction season matters
        # China construction peak: Mar-Nov
        df["china_construction"] = df.index.month.isin([3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(int)""",
    "custom_features": """
    # CNY strength (China = 50%+ of demand)
    if "cny_usd" in df.columns:
        df["cny_return_21d"] = df["cny_usd"].pct_change(21)
        df["cny_vs_sma50"] = df["cny_usd"] / df["cny_usd"].rolling(50).mean() - 1
    # Copper-oil ratio (industrial activity proxy)
    if "crude_oil" in df.columns:
        df["copper_oil_ratio"] = price / df["crude_oil"]
        df["copper_oil_ratio_sma21"] = df["copper_oil_ratio"].rolling(21).mean()
    """,
}


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    for config in [SOYBEANS, WHEAT, COPPER]:
        build_commodity(config, base)
    print("\nAll three commodity projects built.")
    print("Next: cd into each and run 'python3 refresh.py && python3 train.py'")
