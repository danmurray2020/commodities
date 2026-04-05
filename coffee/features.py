"""Feature engineering for coffee price prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.regime_features import add_regime_features

import pandas as pd
import numpy as np
import ta

DATA_DIR = Path(__file__).parent / "data"


def add_price_features(df: pd.DataFrame, price_col: str = "coffee_close") -> pd.DataFrame:
    """Add technical indicator and lag features derived from price.

    Args:
        df: DataFrame with at least a price column.
        price_col: Name of the price column.

    Returns:
        DataFrame with new feature columns added.
    """
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

    # Average True Range (needs high/low/close)
    if all(c in df.columns for c in ["High", "Low"]):
        df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], price, window=14)

    # Lagged prices as features
    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{lag}"] = price.shift(lag)

    # Day of week / month seasonality
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # Seasonal crop cycle features (fourier encoding of day-of-year)
        day_of_year = df.index.dayofyear
        df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        df["season_sin_2"] = np.sin(4 * np.pi * day_of_year / 365.25)  # 2nd harmonic
        df["season_cos_2"] = np.cos(4 * np.pi * day_of_year / 365.25)

        # Harvest window flags
        # Brazil main harvest: April (month 4) - September (month 9)
        df["brazil_harvest"] = df.index.month.isin([4, 5, 6, 7, 8, 9]).astype(int)
        # Vietnam harvest: October - February
        df["vietnam_harvest"] = df.index.month.isin([10, 11, 12, 1, 2]).astype(int)

    # Mean-reversion features (z-scores relative to long-term distributions)
    for window in [126, 252]:  # 6mo, 1yr
        rolling_mean = price.rolling(window).mean()
        rolling_std = price.rolling(window).std()
        df[f"zscore_{window}d"] = (price - rolling_mean) / rolling_std

    # Rate of mean-reversion: how fast z-score is normalizing
    if "zscore_252d" in df.columns:
        df["zscore_252d_change_21d"] = df["zscore_252d"].diff(21)

    # Extreme indicator: price > 2 std devs from 1yr mean
    if "zscore_252d" in df.columns:
        df["extreme_high"] = (df["zscore_252d"] > 2).astype(int)
        df["extreme_low"] = (df["zscore_252d"] < -2).astype(int)

    # Arabica-Robusta spread (if available)
    if "robusta" in df.columns:
        df["arabica_robusta_ratio"] = price / df["robusta"]
        df["arabica_robusta_ratio_sma21"] = df["arabica_robusta_ratio"].rolling(21).mean()
        df["arabica_robusta_spread_zscore"] = (
            (df["arabica_robusta_ratio"] - df["arabica_robusta_ratio"].rolling(252).mean())
            / df["arabica_robusta_ratio"].rolling(252).std()
        )

    # BRL strength (if available) — weaker BRL is bearish for coffee prices
    if "brl_usd" in df.columns:
        df["brl_return_21d"] = df["brl_usd"].pct_change(21)
        df["brl_vs_sma50"] = df["brl_usd"] / df["brl_usd"].rolling(50).mean() - 1

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


def build_target(df: pd.DataFrame, price_col: str = "coffee_close", horizon: int = 5) -> pd.DataFrame:
    """Create prediction target: future return over the given horizon.

    Args:
        df: DataFrame with price column.
        price_col: Name of the price column.
        horizon: Number of trading days ahead to predict.

    Returns:
        DataFrame with target columns added.
    """
    df = df.copy()
    future_price = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future_price / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def merge_cot_data(df: pd.DataFrame, cot_path: str = str(DATA_DIR / "coffee_cot.csv")) -> pd.DataFrame:
    """Merge COT positioning data into the main dataset.

    COT is released weekly (Tuesday snapshot, Friday release), so we
    forward-fill to align with daily price data.
    """
    try:
        cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return df
    df = df.join(cot, how="left")
    cot_cols = [c for c in cot.columns if c in df.columns]
    df[cot_cols] = df[cot_cols].apply(pd.to_numeric, errors="coerce").ffill(limit=7)
    return df


def merge_weather_data(df: pd.DataFrame, weather_path: str = str(DATA_DIR / "weather.csv")) -> pd.DataFrame:
    """Merge weather data into the main dataset.

    Uses only the aggregate/derived weather features to keep dimensionality
    manageable. Forward-fills weekends/holidays.
    """
    try:
        weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return df
    # Use only aggregate features to avoid bloating feature space
    agg_cols = [c for c in weather.columns if any(
        keyword in c for keyword in [
            "avg_temp", "total_precip", "_7d_", "_30d_",
            "anomaly", "frost_risk", "drought",
        ]
    )]
    weather = weather[agg_cols]
    df = df.join(weather, how="left")
    df[agg_cols] = df[agg_cols].ffill(limit=5)
    return df


def merge_enso_data(df: pd.DataFrame, enso_path: str = str(DATA_DIR / "enso.csv")) -> pd.DataFrame:
    """Merge ENSO (El Niño/La Niña) index data.

    ENSO is monthly, so we forward-fill to daily frequency.
    """
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
    horizon: int = 5,
    use_cot: bool = True,
    use_weather: bool = True,
    use_enso: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data, engineer features, and return a clean dataset.

    Args:
        csv_path: Path to the combined features CSV.
        horizon: Prediction horizon in trading days.
        use_cot: Whether to include COT positioning data.
        use_weather: Whether to include weather data.

    Returns:
        Tuple of (DataFrame with features and targets, list of feature column names).
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Add technical features
    df = add_price_features(df)
    df = add_regime_features(df, price_col="coffee_close")

    # Add fundamental data
    if use_cot:
        df = merge_cot_data(df)
    if use_weather:
        df = merge_weather_data(df)
    if use_enso:
        df = merge_enso_data(df)

    # Add target
    df = build_target(df, horizon=horizon)

    # Drop rows with NaNs from rolling calculations
    df = df.ffill()
    df = df.dropna()

    # Identify feature columns (everything except targets and raw prices)
    exclude = {"coffee_close", "Open", "High", "Low", "Volume", "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols
