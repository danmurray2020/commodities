"""Feature engineering for natural gas price prediction.

Natural gas is uniquely seasonal — winter heating demand and summer cooling
demand dominate price action more than any other commodity.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import ta

DATA_DIR = Path(__file__).parent / "data"


def add_price_features(df: pd.DataFrame, price_col: str = "natgas_close") -> pd.DataFrame:
    df = df.copy()
    price = df[price_col]

    for lag in [1, 5, 10, 21]:
        df[f"return_{lag}d"] = price.pct_change(lag)
    for window in [5, 10, 21, 50, 200]:
        df[f"sma_{window}"] = price.rolling(window).mean()
        df[f"price_vs_sma_{window}"] = price / df[f"sma_{window}"] - 1
    for window in [10, 21, 63]:
        df[f"volatility_{window}d"] = price.pct_change().rolling(window).std() * np.sqrt(252)

    df["rsi_14"] = ta.momentum.rsi(price, window=14)
    macd = ta.trend.MACD(price)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(price, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()
    if all(c in df.columns for c in ["High", "Low"]):
        df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], price, window=14)

    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{lag}"] = price.shift(lag)

    # --- Enhanced seasonality (natgas is THE seasonal commodity) ---
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        day_of_year = df.index.dayofyear

        # Multiple harmonics to capture complex seasonal pattern
        for harmonic in [1, 2, 3, 4]:
            df[f"season_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
            df[f"season_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

        # Season flags
        df["winter"] = df.index.month.isin([12, 1, 2]).astype(int)
        df["shoulder"] = df.index.month.isin([3, 4, 10, 11]).astype(int)
        df["summer"] = df.index.month.isin([6, 7, 8]).astype(int)
        df["injection_season"] = df.index.month.isin([4, 5, 6, 7, 8, 9, 10]).astype(int)  # Apr-Oct
        df["withdrawal_season"] = df.index.month.isin([11, 12, 1, 2, 3]).astype(int)  # Nov-Mar

        # Days until winter (peak demand)
        # Approximate: count days until next December 15
        def days_to_winter(dt):
            # Find the next Dec 15 (winter peak demand)
            winter_peak = pd.Timestamp(dt.year, 12, 15)
            if dt > winter_peak:
                winter_peak = pd.Timestamp(dt.year + 1, 12, 15)
            return (winter_peak - dt).days
        df["days_to_winter"] = pd.Series([days_to_winter(d) for d in df.index], index=df.index)
        df["days_to_winter_norm"] = df["days_to_winter"] / 365.0

    # Mean reversion
    for window in [126, 252]:
        rm = price.rolling(window).mean()
        rs = price.rolling(window).std()
        df[f"zscore_{window}d"] = (price - rm) / rs
    if "zscore_252d" in df.columns:
        df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)
        df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
        df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

    # Trend features (learned from sugar)
    daily_ret = price.pct_change()
    for window in [21, 63, 126]:
        df[f"pct_up_days_{window}d"] = daily_ret.rolling(window).apply(lambda x: (x > 0).mean())
        df[f"trend_slope_{window}d"] = price.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0
        )
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["sma_50_200_gap"] = df["sma_50"] / df["sma_200"] - 1
    ret_21d = price.pct_change(21)
    df["momentum_rank_252d"] = ret_21d.rolling(252).rank(pct=True)

    # Natgas-crude oil spread (relative energy value)
    if "crude_oil" in df.columns:
        # Standard ratio: NG price * 6 vs crude (6:1 BTU equivalent)
        df["ng_oil_ratio"] = (price * 6) / df["crude_oil"]
        df["ng_oil_ratio_sma21"] = df["ng_oil_ratio"].rolling(21).mean()
        df["ng_oil_ratio_zscore"] = (
            (df["ng_oil_ratio"] - df["ng_oil_ratio"].rolling(252).mean())
            / df["ng_oil_ratio"].rolling(252).std()
        )

    # Heating oil spread
    if "heating_oil" in df.columns:
        df["ng_ho_ratio"] = price / df["heating_oil"]

    return df


def build_target(df, price_col="natgas_close", horizon=63):
    df = df.copy()
    future = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def merge_cot_data(df, cot_path=str(DATA_DIR / "natgas_cot.csv")):
    try: cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    df = df.join(cot, how="left")
    cot_cols = [c for c in cot.columns if c in df.columns]
    df[cot_cols] = df[cot_cols].apply(pd.to_numeric, errors="coerce").ffill(limit=7)
    return df


def merge_weather_data(df, weather_path=str(DATA_DIR / "weather.csv")):
    try: weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    agg_cols = [c for c in weather.columns if any(k in c for k in [
        "us_avg", "us_min", "us_max", "hdd", "cdd", "temp_anomaly",
        "extreme_cold", "extreme_heat", "cold_snap", "windchill"])]
    weather = weather[agg_cols]
    df = df.join(weather, how="left")
    df[agg_cols] = df[agg_cols].ffill(limit=5)
    return df


def merge_enso_data(df, enso_path=str(DATA_DIR / "enso.csv")):
    try: enso = pd.read_csv(enso_path, index_col=0, parse_dates=True)
    except FileNotFoundError: return df
    df = df.join(enso, how="left")
    ecols = [c for c in enso.columns if c in df.columns]
    df[ecols] = df[ecols].ffill(limit=30)
    return df


def prepare_dataset(csv_path=str(DATA_DIR / "combined_features.csv"), horizon=63,
                    use_cot=True, use_weather=True, use_enso=True):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    if use_cot: df = merge_cot_data(df)
    if use_weather: df = merge_weather_data(df)
    if use_enso: df = merge_enso_data(df)
    df = build_target(df, horizon=horizon)
    df = df.dropna()
    exclude = {"natgas_close", "Open", "High", "Low", "Volume", "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols
