"""Feature engineering for Gold price prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.regime_features import add_regime_features

import pandas as pd
import numpy as np
from agents.indicators import rsi, MACD, BollingerBands, average_true_range

DATA_DIR = Path(__file__).parent / "data"


def add_price_features(df: pd.DataFrame, price_col: str = "gold_close") -> pd.DataFrame:
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
    df["rsi_14"] = rsi(price, window=14)

    # MACD
    macd = MACD(price)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(price, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()

    # ATR
    if all(c in df.columns for c in ["High", "Low"]):
        df["atr_14"] = average_true_range(df["High"], df["Low"], price, window=14)

    # Price lags
    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{lag}"] = price.shift(lag)

    # Seasonality
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        day_of_year = df.index.dayofyear
        for harmonic in [1, 2]:
            df[f"season_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
            df[f"season_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

    # Z-scores (mean reversion)
    for window in [126, 252]:
        rm = price.rolling(window).mean()
        rs = price.rolling(window).std()
        df[f"zscore_{window}d"] = (price - rm) / rs
    if "zscore_252d" in df.columns:
        df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)
        df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
        df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

    # Trend features
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

    return df


def build_target(df, price_col="gold_close", horizon=63):
    df = df.copy()
    future = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def prepare_dataset(csv_path=str(DATA_DIR / "combined_features.csv"), horizon=63):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    df = add_regime_features(df, price_col="gold_close")
    df = build_target(df, horizon=horizon)
    df = df.ffill()
    df = df.dropna()
    exclude = {"gold_close", "Open", "High", "Low", "Volume", "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols
