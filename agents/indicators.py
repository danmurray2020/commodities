"""Native pandas implementations of the technical indicators we need.

Replaces the `ta` library, which has a transitive `multitasking` dependency
that fails to build wheels in some environments (notably the remote agent
sandbox). Same numerical behaviour, same API surface used in features.py.
"""

import numpy as np
import pandas as pd


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (matches ta.momentum.rsi)."""
    diff = price.diff(1)
    up = diff.where(diff > 0, 0.0)
    dn = -diff.where(diff < 0, 0.0)
    emaup = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    emadn = dn.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = emaup / emadn
    return pd.Series(np.where(emadn == 0, 100.0, 100 - (100 / (1 + rs))), index=price.index)


class MACD:
    """Moving Average Convergence Divergence (matches ta.trend.MACD defaults)."""

    def __init__(
        self,
        price: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
    ):
        ema_fast = price.ewm(span=window_fast, adjust=False, min_periods=window_fast).mean()
        ema_slow = price.ewm(span=window_slow, adjust=False, min_periods=window_slow).mean()
        self._macd = ema_fast - ema_slow
        self._signal = self._macd.ewm(span=window_sign, adjust=False, min_periods=window_sign).mean()

    def macd(self) -> pd.Series:
        return self._macd

    def macd_signal(self) -> pd.Series:
        return self._signal

    def macd_diff(self) -> pd.Series:
        return self._macd - self._signal


class BollingerBands:
    """Bollinger Bands (matches ta.volatility.BollingerBands defaults)."""

    def __init__(self, price: pd.Series, window: int = 20, window_dev: int = 2):
        self._sma = price.rolling(window=window, min_periods=window).mean()
        self._std = price.rolling(window=window, min_periods=window).std(ddof=0)
        self._k = window_dev
        self._price = price

    def bollinger_hband(self) -> pd.Series:
        return self._sma + self._k * self._std

    def bollinger_lband(self) -> pd.Series:
        return self._sma - self._k * self._std

    def bollinger_pband(self) -> pd.Series:
        upper = self.bollinger_hband()
        lower = self.bollinger_lband()
        return (self._price - lower) / (upper - lower)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range using Wilder's smoothing (matches ta.volatility.average_true_range).

    `ta` seeds the recursion with an SMA of the first `window` true-range values,
    then applies Wilder's recursive update. Pandas `ewm` doesn't seed with an SMA,
    so we compute the recursion explicitly.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    arr = tr.to_numpy(dtype=float)
    out = np.zeros(len(arr))
    if len(arr) >= window:
        # ta seeds at index window-1 with the NaN-skipping mean of TR[0:window]
        out[window - 1] = np.nanmean(arr[:window])
        for i in range(window, len(arr)):
            out[i] = (out[i - 1] * (window - 1) + arr[i]) / window
    return pd.Series(out, index=tr.index)
