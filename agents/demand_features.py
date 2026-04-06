"""Demand-side features based on buyer stocks, consumer sentiment, and earnings cycles.

Major commodity buyers serve as demand proxies — when Starbucks stock falls,
coffee demand expectations are weakening. This module adds features derived
from buyer stock prices for coffee, sugar, and cocoa.
"""

import numpy as np
import pandas as pd


# ── Earnings calendars (approximate quarterly report dates) ────────────────

# Starbucks: late Jan, late Apr, late Jul, late Oct
_SBUX_EARNINGS_MD = [(1, 28), (4, 28), (7, 28), (10, 28)]

# Nestle: mid Feb, mid Jul (semi-annual, but add Q1/Q3 trading updates)
_NSRGY_EARNINGS_MD = [(2, 15), (7, 15)]

# Coca-Cola: mid Feb, late Apr, late Jul, late Oct
_KO_EARNINGS_MD = [(2, 14), (4, 25), (7, 25), (10, 23)]

# PepsiCo: mid Feb, late Apr, mid Jul, early Oct
_PEP_EARNINGS_MD = [(2, 9), (4, 25), (7, 13), (10, 8)]

# Mondelez: late Jan, late Apr, late Jul, late Oct
_MDLZ_EARNINGS_MD = [(1, 30), (4, 30), (7, 28), (10, 29)]

# Hershey: early Feb, late Apr, late Jul, late Oct
_HSY_EARNINGS_MD = [(2, 5), (4, 27), (7, 27), (10, 26)]


def _days_to_next_earnings(dates: pd.DatetimeIndex, earnings_md: list[tuple[int, int]]) -> np.ndarray:
    """Compute days until next earnings date for each date in the index."""
    result = np.full(len(dates), np.nan)
    for i, dt in enumerate(dates):
        min_days = 366
        for month, day in earnings_md:
            # Check current year and next year
            for year in [dt.year, dt.year + 1]:
                try:
                    earn_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                delta = (earn_date - dt).days
                if 0 <= delta < min_days:
                    min_days = delta
        result[i] = min_days
    return result


def _post_earnings_window(dates: pd.DatetimeIndex, earnings_md: list[tuple[int, int]],
                          window: int = 5) -> np.ndarray:
    """Binary flag: 1 if within `window` trading days after an earnings date."""
    result = np.zeros(len(dates), dtype=int)
    for i, dt in enumerate(dates):
        for month, day in earnings_md:
            for year in [dt.year, dt.year - 1]:
                try:
                    earn_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                delta = (dt - earn_date).days
                if 0 <= delta <= window + 2:  # +2 for weekends
                    result[i] = 1
                    break
    return result


def _safe_return(series: pd.Series, periods: int) -> pd.Series:
    """Compute pct_change only if sufficient non-NaN data exists."""
    if series.notna().sum() < periods + 10:
        return pd.Series(np.nan, index=series.index)
    return series.pct_change(periods)


def _safe_vs_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute price vs SMA ratio safely."""
    if series.notna().sum() < window + 10:
        return pd.Series(np.nan, index=series.index)
    sma = series.rolling(window).mean()
    return series / sma - 1


def add_demand_features(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Add demand-side features based on buyer stocks, consumer sentiment, and earnings cycles.

    Args:
        df: DataFrame with DatetimeIndex. Should already have buyer stock columns
            merged in (e.g. sbux, nsrgy for coffee).
        commodity: One of 'coffee', 'sugar', 'cocoa' (or 'chocolate').

    Returns:
        DataFrame with demand features added.
    """
    df = df.copy()
    commodity = commodity.lower()
    if commodity == "chocolate":
        commodity = "cocoa"

    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

    # ── Coffee demand features ─────────────────────────────────────────
    if commodity == "coffee":
        if "sbux" in df.columns:
            df["sbux_return_21d"] = _safe_return(df["sbux"], 21)
            df["sbux_vs_sma50"] = _safe_vs_sma(df["sbux"], 50)

        if "nsrgy" in df.columns:
            nsrgy_return_21d = _safe_return(df["nsrgy"], 21)
            df["nsrgy_return_21d"] = nsrgy_return_21d

        # Buyer momentum: average of both buyer returns
        if "sbux" in df.columns and "nsrgy" in df.columns:
            sbux_r = _safe_return(df["sbux"], 21)
            nsrgy_r = _safe_return(df["nsrgy"], 21)
            df["buyer_momentum"] = (sbux_r + nsrgy_r) / 2
        elif "sbux" in df.columns:
            df["buyer_momentum"] = _safe_return(df["sbux"], 21)

        # Coffee demand divergence: coffee price up but buyer stocks down
        price_col = "coffee_close"
        if price_col in df.columns and "sbux" in df.columns:
            coffee_ret = _safe_return(df[price_col], 21)
            sbux_ret = _safe_return(df["sbux"], 21)
            # Positive = coffee up & buyers down (bearish divergence)
            df["coffee_demand_divergence"] = coffee_ret - sbux_ret

        # Earnings calendar features
        if has_datetime_index:
            # Use Starbucks as primary buyer earnings
            df["days_to_buyer_earnings"] = _days_to_next_earnings(df.index, _SBUX_EARNINGS_MD)
            df["post_earnings_window"] = _post_earnings_window(df.index, _SBUX_EARNINGS_MD)

    # ── Sugar demand features ──────────────────────────────────────────
    elif commodity == "sugar":
        ko_ret = None
        pep_ret = None

        if "ko" in df.columns:
            ko_ret = _safe_return(df["ko"], 21)
        if "pep" in df.columns:
            pep_ret = _safe_return(df["pep"], 21)

        # Beverage demand: average return of KO + PEP
        if ko_ret is not None and pep_ret is not None:
            df["beverage_demand"] = (ko_ret + pep_ret) / 2
        elif ko_ret is not None:
            df["beverage_demand"] = ko_ret
        elif pep_ret is not None:
            df["beverage_demand"] = pep_ret

        # Beverage vs SMA50
        ko_sma = None
        pep_sma = None
        if "ko" in df.columns:
            ko_sma = _safe_vs_sma(df["ko"], 50)
        if "pep" in df.columns:
            pep_sma = _safe_vs_sma(df["pep"], 50)

        if ko_sma is not None and pep_sma is not None:
            df["beverage_vs_sma50"] = (ko_sma + pep_sma) / 2
        elif ko_sma is not None:
            df["beverage_vs_sma50"] = ko_sma
        elif pep_sma is not None:
            df["beverage_vs_sma50"] = pep_sma

        # Earnings calendar
        if has_datetime_index:
            df["days_to_buyer_earnings"] = _days_to_next_earnings(df.index, _KO_EARNINGS_MD)
            df["post_earnings_window"] = _post_earnings_window(df.index, _KO_EARNINGS_MD)

    # ── Cocoa/Chocolate demand features ────────────────────────────────
    elif commodity == "cocoa":
        mdlz_ret = None
        hsy_ret = None

        if "mdlz" in df.columns:
            mdlz_ret = _safe_return(df["mdlz"], 21)
        if "hsy" in df.columns:
            hsy_ret = _safe_return(df["hsy"], 21)

        # Chocolate demand: average return of MDLZ + HSY
        if mdlz_ret is not None and hsy_ret is not None:
            df["chocolate_demand"] = (mdlz_ret + hsy_ret) / 2
        elif mdlz_ret is not None:
            df["chocolate_demand"] = mdlz_ret
        elif hsy_ret is not None:
            df["chocolate_demand"] = hsy_ret

        # Buyer divergence: cocoa price up but chocolate stocks down
        price_col = "cocoa_close"
        if price_col in df.columns:
            cocoa_ret = _safe_return(df[price_col], 21)
            buyer_ret = None
            if mdlz_ret is not None and hsy_ret is not None:
                buyer_ret = (mdlz_ret + hsy_ret) / 2
            elif mdlz_ret is not None:
                buyer_ret = mdlz_ret
            elif hsy_ret is not None:
                buyer_ret = hsy_ret

            if buyer_ret is not None:
                df["buyer_divergence"] = cocoa_ret - buyer_ret

        # Earnings calendar
        if has_datetime_index:
            df["days_to_buyer_earnings"] = _days_to_next_earnings(df.index, _MDLZ_EARNINGS_MD)
            df["post_earnings_window"] = _post_earnings_window(df.index, _MDLZ_EARNINGS_MD)

    # ── Common features for all commodities ────────────────────────────

    # Consumer confidence (if available from FRED or other source)
    for col_name in ["consumer_confidence", "umcsent"]:
        if col_name in df.columns:
            series = df[col_name].ffill()
            df["consumer_confidence_level"] = series
            df["consumer_confidence_change"] = series.pct_change(1)
            break  # use first available

    # Retail sales (if available)
    for col_name in ["retail_sales", "rsxfs"]:
        if col_name in df.columns:
            series = df[col_name].ffill()
            df["retail_sales_level"] = series
            df["retail_sales_change"] = series.pct_change(1)
            break

    return df
