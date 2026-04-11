"""Feature engineering for copper price prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.regime_features import add_regime_features

import pandas as pd
import numpy as np
from agents.indicators import rsi, MACD, BollingerBands, average_true_range

DATA_DIR = Path(__file__).parent / "data"


def add_price_features(df: pd.DataFrame, price_col: str = "copper_close") -> pd.DataFrame:
    df = df.copy()
    price = df[price_col]

    for lag in [1, 5, 10, 21]:
        df[f"return_{lag}d"] = price.pct_change(lag)
    for window in [5, 10, 21, 50, 200]:
        df[f"sma_{window}"] = price.rolling(window).mean()
        df[f"price_vs_sma_{window}"] = price / df[f"sma_{window}"] - 1
    for window in [10, 21, 63]:
        df[f"volatility_{window}d"] = price.pct_change().rolling(window).std() * np.sqrt(252)

    df["rsi_14"] = rsi(price, window=14)
    macd = MACD(price)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(price, window=20)
    df["bb_pct"] = bb.bollinger_pband()

    for lag in [1, 2, 3, 5, 10]:
        df[f"price_lag_{lag}"] = price.shift(lag)

    if isinstance(df.index, pd.DatetimeIndex):
        df["month"] = df.index.month
        day_of_year = df.index.dayofyear
        for h in [1, 2]:
            df[f"season_sin_{h}"] = np.sin(2 * np.pi * h * day_of_year / 365.25)
            df[f"season_cos_{h}"] = np.cos(2 * np.pi * h * day_of_year / 365.25)
        
        # Copper is less seasonal but China construction season matters
        # China construction peak: Mar-Nov
        df["china_construction"] = df.index.month.isin([3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(int)

    # Mean reversion
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
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0)
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["sma_50_200_gap"] = df["sma_50"] / df["sma_200"] - 1
    ret_21d = price.pct_change(21)
    df["momentum_rank_252d"] = ret_21d.rolling(252).rank(pct=True)

    
    # CNY strength (China = 50%+ of demand)
    if "cny_usd" in df.columns:
        df["cny_return_21d"] = df["cny_usd"].pct_change(21)
        df["cny_vs_sma50"] = df["cny_usd"] / df["cny_usd"].rolling(50).mean() - 1
    # Copper-oil ratio (industrial activity proxy)
    if "crude_oil" in df.columns:
        df["copper_oil_ratio"] = price / df["crude_oil"]
        df["copper_oil_ratio_sma21"] = df["copper_oil_ratio"].rolling(21).mean()
        # Z-score of copper/crude ratio over 1 year — captures divergence
        # from the typical industrial-energy relationship
        ratio_mean = df["copper_oil_ratio"].rolling(252, min_periods=63).mean()
        ratio_std = df["copper_oil_ratio"].rolling(252, min_periods=63).std()
        df["copper_crude_ratio_zscore"] = (df["copper_oil_ratio"] - ratio_mean) / ratio_std
    

    return df


def build_target(df, price_col="copper_close", horizon=63):
    df = df.copy()
    future = df[price_col].shift(-horizon)
    df["target_return"] = np.log(future / df[price_col])
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def merge_cot_data(df, cot_path=str(DATA_DIR / "copper_cot.csv")):
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
        "avg_temp", "total_precip", "_7d_", "_30d_", "anomaly", "drought", "frost", "hdd", "cdd"])]
    if agg_cols:
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


def merge_macro_data(df,
                     macro_path=str(DATA_DIR / "macro_data.csv"),
                     pmi_path=str(DATA_DIR / "china_pmi.csv")):
    """Merge FRED macro data and China PMI into the feature DataFrame."""
    # --- FRED macro data ---
    try:
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        # Drop columns already present in df (e.g. usd_index is sometimes
        # fetched as supplementary data and stored in combined_features.csv).
        # Without this dedup the join raises:
        #   "columns overlap but no suffix specified: Index(['usd_index'])"
        overlap = [c for c in macro.columns if c in df.columns]
        if overlap:
            macro = macro.drop(columns=overlap)
        df = df.join(macro, how="left")
        macro_cols = [c for c in macro.columns if c in df.columns]
        df[macro_cols] = df[macro_cols].ffill()
    except FileNotFoundError:
        pass

    # --- China PMI ---
    try:
        pmi = pd.read_csv(pmi_path, index_col=0, parse_dates=True)
        overlap = [c for c in pmi.columns if c in df.columns]
        if overlap:
            pmi = pmi.drop(columns=overlap)
        df = df.join(pmi, how="left")
        if "china_pmi" in df.columns:
            df["china_pmi"] = df["china_pmi"].ffill()
    except FileNotFoundError:
        pass

    # --- Derived features ---
    # Year-over-year changes for monthly macro indicators
    if "housing_starts" in df.columns:
        df["housing_starts_yoy"] = df["housing_starts"].pct_change(252)
    if "industrial_prod" in df.columns:
        df["industrial_prod_yoy"] = df["industrial_prod"].pct_change(252)

    # China PMI derived features
    if "china_pmi" in df.columns:
        pmi_roll_mean = df["china_pmi"].rolling(252, min_periods=63).mean()
        pmi_roll_std = df["china_pmi"].rolling(252, min_periods=63).std()
        df["china_pmi_zscore"] = (df["china_pmi"] - pmi_roll_mean) / pmi_roll_std
        df["china_pmi_above_50"] = (df["china_pmi"] > 50).astype(int)
        # 3-month momentum (~63 trading days)
        df["china_pmi_momentum"] = df["china_pmi"].diff(63)

    return df


def prepare_dataset(csv_path=str(DATA_DIR / "combined_features.csv"), horizon=63,
                    use_cot=True, use_weather=True, use_enso=True):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    df = add_regime_features(df, price_col="copper_close")
    if use_cot: df = merge_cot_data(df)
    if use_weather: df = merge_weather_data(df)
    if use_enso: df = merge_enso_data(df)
    df = merge_macro_data(df)
    df = build_target(df, horizon=horizon)
    df = df.ffill()
    df = df.dropna()
    exclude = {"copper_close", "Open", "High", "Low", "Volume", "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols
