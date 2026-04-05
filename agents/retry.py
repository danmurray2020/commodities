"""Retry utility with exponential backoff for network calls."""

import logging
import time
from functools import wraps


logger = logging.getLogger("commodities")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 30.0):
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        max_delay: Maximum delay between retries.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


def download_with_retry(ticker: str, period: str = "10y", **kwargs) -> "pd.DataFrame":
    """Download Yahoo Finance data with retry logic.

    Args:
        ticker: Yahoo Finance ticker symbol.
        period: How far back to fetch.
        **kwargs: Additional arguments passed to yf.download().

    Returns:
        DataFrame with OHLCV data.

    Raises:
        RuntimeError: If download fails after all retries or returns empty data.
    """
    import pandas as pd
    import yfinance as yf

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _download():
        df = yf.download(ticker, period=period, auto_adjust=True, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        if df.empty:
            raise RuntimeError(f"Empty data returned for {ticker}")
        return df

    return _download()
