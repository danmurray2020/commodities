"""Shared regime detection features for all commodities.

These features help models condition on market regime rather than
learning fixed directional biases that break during regime changes.

Usage (from any commodity's features.py):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.regime_features import add_regime_features
    df = add_regime_features(df, price_col="coffee_close")
"""

import numpy as np
import pandas as pd


def add_regime_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add regime-detection features to help models adapt to market state.

    Features added:
    - vol_regime: Rolling 63-day volatility quintile (1-5)
    - vol_regime_change: Change in vol regime over 63 days
    - trend_regime: Current trend state based on price vs SMAs
    - drawdown: Current drawdown from rolling 252-day high
    - drawdown_regime: Drawdown severity quintile (1-5)
    - mean_reversion_pressure: Z-score of 63-day return (extreme → reversion)
    - regime_uncertainty: Disagreement between short/long trend signals
    """
    price = df[price_col]

    # Volatility regime (quintile of rolling 63-day realized vol)
    returns = price.pct_change()
    vol_63 = returns.rolling(63).std() * np.sqrt(252)
    vol_rank = vol_63.rolling(504, min_periods=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["vol_regime"] = (vol_rank * 5).clip(1, 5)
    df["vol_regime_change"] = df["vol_regime"] - df["vol_regime"].shift(63)

    # Trend regime: composite of price vs 50/200 SMA
    sma_50 = price.rolling(50).mean()
    sma_200 = price.rolling(200).mean()
    above_50 = (price > sma_50).astype(float)
    above_200 = (price > sma_200).astype(float)
    sma_50_above_200 = (sma_50 > sma_200).astype(float)
    # 0=bear, 1=weak, 2=recovering, 3=bull
    df["trend_regime"] = above_50 + above_200 + sma_50_above_200

    # Drawdown from rolling high
    rolling_high = price.rolling(252, min_periods=63).max()
    df["drawdown"] = (price - rolling_high) / rolling_high
    # Quintile: 1=shallow, 5=deep
    dd_rank = df["drawdown"].rolling(504, min_periods=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["drawdown_regime"] = ((1 - dd_rank) * 5).clip(1, 5)

    # Mean reversion pressure: z-score of recent 63-day return
    ret_63 = price.pct_change(63)
    ret_mean = ret_63.rolling(504, min_periods=252).mean()
    ret_std = ret_63.rolling(504, min_periods=252).std()
    df["mean_reversion_pressure"] = ((ret_63 - ret_mean) / ret_std).clip(-3, 3)

    # Regime uncertainty: disagreement between short and long signals
    sma_10 = price.rolling(10).mean()
    short_trend = (price > sma_10).astype(float) + (price > sma_50).astype(float)
    long_trend = (price > sma_200).astype(float) + sma_50_above_200
    df["regime_uncertainty"] = (short_trend - long_trend).abs()

    return df
