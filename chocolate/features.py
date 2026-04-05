"""Feature engineering for cocoa price prediction."""

from pathlib import Path

import pandas as pd
import numpy as np
import ta

DATA_DIR = Path(__file__).parent / "data"


def add_price_features(df: pd.DataFrame, price_col: str = "cocoa_close") -> pd.DataFrame:
    df = df.copy()
    price = df[price_col]

    # Returns
    for lag in [1, 5, 10, 21]:
        df[f"return_{lag}d"] = price.pct_change(lag)

    # Moving averages
    for window in [5, 10, 21, 50, 200]:
        df[f"sma_{window}"] = price.rolling(window).mean()
        df[f"price_vs_sma_{window}"] = price / df[f"sma_{window}"] - 1

    # Volatility
    for window in [10, 21, 63]:
        df[f"volatility_{window}d"] = price.pct_change().rolling(window).std() * np.sqrt(252)

    # RSI
    df["rsi_14"] = ta.momentum.rsi(price, window=14)

    # MACD
    macd = ta.trend.MACD(price)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(price, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()

    # ATR
    if all(c in df.columns for c in ["High", "Low"]):
        df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], price, window=14)

    # Lagged prices
    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{lag}"] = price.shift(lag)

    # Seasonality
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # Fourier seasonal encoding
        day_of_year = df.index.dayofyear
        df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        df["season_sin_2"] = np.sin(4 * np.pi * day_of_year / 365.25)
        df["season_cos_2"] = np.cos(4 * np.pi * day_of_year / 365.25)

        # Cocoa harvest windows
        # West Africa main crop: Oct-Mar, mid-crop: May-Aug
        df["westafrica_main_harvest"] = df.index.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        df["westafrica_mid_harvest"] = df.index.month.isin([5, 6, 7, 8]).astype(int)
        # Indonesia: main Sep-Dec
        df["indonesia_harvest"] = df.index.month.isin([9, 10, 11, 12]).astype(int)

    # Mean-reversion z-scores
    for window in [126, 252]:
        rolling_mean = price.rolling(window).mean()
        rolling_std = price.rolling(window).std()
        df[f"zscore_{window}d"] = (price - rolling_mean) / rolling_std

    if "zscore_252d" in df.columns:
        df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)
        df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
        df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

    # GHS strength (if available)
    if "ghs_usd" in df.columns:
        df["ghs_return_21d"] = df["ghs_usd"].pct_change(21)
        df["ghs_vs_sma50"] = df["ghs_usd"] / df["ghs_usd"].rolling(50).mean() - 1

    # Coffee correlation (if available)
    if "coffee" in df.columns:
        df["cocoa_coffee_ratio"] = price / df["coffee"]
        df["cocoa_coffee_ratio_sma21"] = df["cocoa_coffee_ratio"].rolling(21).mean()

    # Trend strength features
    daily_ret = price.pct_change()
    for window in [21, 63, 126]:
        df[f"pct_up_days_{window}d"] = daily_ret.rolling(window).apply(lambda x: (x > 0).mean())
        df[f"trend_slope_{window}d"] = price.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0
        )

    # SMA crossovers
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["sma_50_200_gap"] = df["sma_50"] / df["sma_200"] - 1

    # Momentum rank (percentile of current 21d momentum vs last year)
    ret_21d = price.pct_change(21)
    df["momentum_rank_252d"] = ret_21d.rolling(252).rank(pct=True)

    return df


def build_target(df: pd.DataFrame, price_col: str = "cocoa_close", horizon: int = 63) -> pd.DataFrame:
    df = df.copy()
    future_price = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future_price / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def merge_cot_data(df: pd.DataFrame, cot_path: str = str(DATA_DIR / "cocoa_cot.csv")) -> pd.DataFrame:
    try:
        cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return df
    df = df.join(cot, how="left")
    cot_cols = [c for c in cot.columns if c in df.columns]
    df[cot_cols] = df[cot_cols].apply(pd.to_numeric, errors="coerce").ffill(limit=7)
    return df


def merge_weather_data(df: pd.DataFrame, weather_path: str = str(DATA_DIR / "weather.csv")) -> pd.DataFrame:
    try:
        weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return df
    agg_cols = [c for c in weather.columns if any(
        keyword in c for keyword in [
            "avg_temp", "total_precip", "_7d_", "_30d_",
            "anomaly", "heat_risk", "drought",
        ]
    )]
    weather = weather[agg_cols]
    df = df.join(weather, how="left")
    df[agg_cols] = df[agg_cols].ffill(limit=5)
    return df


def merge_enso_data(df: pd.DataFrame, enso_path: str = str(DATA_DIR / "enso.csv")) -> pd.DataFrame:
    try:
        enso = pd.read_csv(enso_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return df
    df = df.join(enso, how="left")
    enso_cols = [c for c in enso.columns if c in df.columns]
    df[enso_cols] = df[enso_cols].ffill(limit=30)
    return df


def prepare_dataset(
    csv_path: str = str(DATA_DIR / "combined_features.csv"),
    horizon: int = 63,
    use_cot: bool = True,
    use_weather: bool = True,
    use_enso: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    if use_cot:
        df = merge_cot_data(df)
    if use_weather:
        df = merge_weather_data(df)
    if use_enso:
        df = merge_enso_data(df)
    df = build_target(df, horizon=horizon)
    df = df.dropna()

    exclude = {"cocoa_close", "Open", "High", "Low", "Volume", "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols
