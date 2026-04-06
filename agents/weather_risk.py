"""Seasonal weather risk features for commodities.

Forward-looking weather risk signals based on seasonal patterns and
current conditions. These capture what moves prices: not what the
weather WAS, but when weather RISK is elevated.

Usage (from any commodity's features.py):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.weather_risk import add_weather_risk_features
    df = add_weather_risk_features(df, price_col="coffee_close", commodity="coffee")
"""

import numpy as np
import pandas as pd


def _coffee_weather_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add coffee-specific weather risk features.

    Key risks:
    - Brazil frost (May-Aug): can destroy arabica crops overnight
    - Brazil drought (Oct-Mar growing season): reduces yields
    - Vietnam flooding (Sep-Nov monsoon): disrupts robusta harvest
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    month = df.index.month

    # --- Brazil frost season (May-Aug) ---
    df["brazil_frost_season"] = month.isin([5, 6, 7, 8]).astype(int)

    # Frost risk score: combine season flag with temperature anomaly
    # When temps are below average during frost season, risk is elevated
    temp_anomaly_cols = [c for c in df.columns if "brazil" in c.lower() and "temp" in c.lower() and "anomaly" in c.lower()]
    if temp_anomaly_cols:
        # Use the first matching anomaly column
        temp_anomaly = df[temp_anomaly_cols[0]]
        # Negative anomaly = colder than normal = higher frost risk
        # Clip to [0, 1] range: 0 = no risk, 1 = high risk
        cold_signal = (-temp_anomaly).clip(lower=0)
        # Normalize to 0-1 using expanding percentile rank
        cold_score = cold_signal.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        df["brazil_frost_risk_score"] = df["brazil_frost_season"] * cold_score
    else:
        # Fallback: use just the seasonal flag (still useful as a prior)
        df["brazil_frost_risk_score"] = df["brazil_frost_season"] * 0.5

    # --- Brazil drought risk (growing season Oct-Mar) ---
    growing_season = month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    precip_cols = [c for c in df.columns if "brazil" in c.lower() and "precip" in c.lower() and "30d" in c.lower()]
    if precip_cols:
        precip = df[precip_cols[0]]
        # Below 50th percentile = drier than normal
        precip_pctile = precip.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        df["brazil_drought_risk"] = growing_season * (1 - precip_pctile)
    else:
        df["brazil_drought_risk"] = growing_season * 0.3  # mild prior

    # --- Vietnam flood risk (monsoon Sep-Nov) ---
    monsoon = month.isin([9, 10, 11]).astype(int)
    vn_precip_cols = [c for c in df.columns if "vietnam" in c.lower() and "precip" in c.lower() and "30d" in c.lower()]
    if vn_precip_cols:
        vn_precip = df[vn_precip_cols[0]]
        # Above 90th percentile = flood risk
        vn_pctile = vn_precip.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        df["vietnam_flood_risk"] = monsoon * (vn_pctile > 0.9).astype(float)
    else:
        df["vietnam_flood_risk"] = monsoon * 0.1  # low prior

    return df


def _sugar_weather_risk(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add sugar-specific weather risk features.

    Key risks:
    - Brazil harvest moisture (Apr-Nov): rain during harvest lowers sucrose
    - India monsoon (Jun-Sep): weak monsoon = lower production
    - Ethanol parity: high crude/sugar ratio diverts cane to ethanol
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    month = df.index.month

    # --- Brazil harvest moisture (Apr-Nov) ---
    harvest = month.isin([4, 5, 6, 7, 8, 9, 10, 11]).astype(int)
    precip_cols = [c for c in df.columns if "brazil" in c.lower() and "precip" in c.lower() and "30d" in c.lower()]
    if precip_cols:
        precip = df[precip_cols[0]]
        # High rain during harvest = lower sucrose content = bearish
        precip_pctile = precip.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        df["brazil_harvest_moisture"] = harvest * precip_pctile
    else:
        df["brazil_harvest_moisture"] = harvest * 0.5

    # --- India monsoon risk (Jun-Sep) ---
    india_monsoon = month.isin([6, 7, 8, 9]).astype(int)
    india_precip_cols = [c for c in df.columns if "india" in c.lower() and "precip" in c.lower() and "30d" in c.lower()]
    if india_precip_cols:
        india_precip = df[india_precip_cols[0]]
        # Low monsoon rainfall = lower production = bullish sugar
        india_pctile = india_precip.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        # Risk = inverse of rainfall percentile during monsoon
        df["india_monsoon_risk"] = india_monsoon * (1 - india_pctile)
    else:
        df["india_monsoon_risk"] = india_monsoon * 0.3

    # --- Ethanol parity (crude_oil / sugar ratio) ---
    if "crude_oil" in df.columns and price_col in df.columns:
        crude_sugar_ratio = df["crude_oil"] / df[price_col]
        # When ratio is high, ethanol from sugar becomes more attractive
        # → more cane diverted to ethanol → less sugar supply → bullish
        ratio_pctile = crude_sugar_ratio.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
        df["ethanol_parity"] = (ratio_pctile > 0.8).astype(float)
        # Also add the continuous ratio percentile as a feature
        df["crude_sugar_ratio_pctile"] = ratio_pctile

    return df


def add_weather_risk_features(
    df: pd.DataFrame,
    price_col: str,
    commodity: str,
) -> pd.DataFrame:
    """Add weather risk features based on seasonal patterns and current conditions.

    These are forward-looking risk signals: they flag WHEN weather risk is
    elevated based on (a) the calendar and (b) current weather conditions.
    This matters because commodity prices move on anticipated supply shocks,
    not just realized weather.

    Args:
        df: DataFrame with DatetimeIndex and weather/price columns.
        price_col: Name of the commodity's price column.
        commodity: One of "coffee", "sugar".

    Returns:
        DataFrame with weather risk feature columns added.
    """
    df = df.copy()

    if commodity == "coffee":
        df = _coffee_weather_risk(df)
    elif commodity == "sugar":
        df = _sugar_weather_risk(df, price_col)
    else:
        # No weather risk features defined for this commodity
        pass

    return df
