"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add sugar project to path so we can import features
sys.path.insert(0, str(Path("/Users/danielmurray/dev2/sugar")))
from features import add_price_features, build_target


@pytest.fixture
def synthetic_price_df():
    """Create a synthetic price DataFrame with 1000 trading days."""
    np.random.seed(42)
    n = 1000
    dates = pd.bdate_range("2019-01-01", periods=n)
    prices = 20 + np.random.randn(n).cumsum() * 0.5
    prices = np.abs(prices) + 5  # Keep prices positive

    df = pd.DataFrame({
        "sugar_close": prices,
        "High": prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        "Low": prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        "Volume": np.random.randint(1000, 50000, n),
    }, index=dates)
    return df


@pytest.fixture
def featured_df(synthetic_price_df):
    """Price DF with features added."""
    return add_price_features(synthetic_price_df)


@pytest.fixture
def target_df(synthetic_price_df):
    """Price DF with targets built."""
    return build_target(synthetic_price_df, price_col="sugar_close", horizon=63)


class TestBuildTarget:
    """Tests for build_target()."""

    def test_target_return_formula(self, synthetic_price_df):
        """target_return = log(future_price / current_price)."""
        df = build_target(synthetic_price_df, price_col="sugar_close", horizon=63)
        # Check a specific row where target should be non-NaN
        for i in range(0, len(df) - 63):
            expected = np.log(df["sugar_close"].iloc[i + 63] / df["sugar_close"].iloc[i])
            actual = df["target_return"].iloc[i]
            assert abs(actual - expected) < 1e-10, (
                f"Row {i}: expected target_return={expected}, got {actual}"
            )

    def test_last_horizon_rows_nan(self, synthetic_price_df):
        """Last `horizon` rows should have NaN targets."""
        horizon = 63
        df = build_target(synthetic_price_df, price_col="sugar_close", horizon=horizon)
        last_targets = df["target_return"].iloc[-horizon:]
        assert last_targets.isna().all(), (
            f"Expected last {horizon} target_return values to be NaN, "
            f"but {last_targets.notna().sum()} are non-NaN"
        )

    def test_non_last_rows_not_nan(self, synthetic_price_df):
        """Rows before the last `horizon` should not have NaN targets."""
        horizon = 63
        df = build_target(synthetic_price_df, price_col="sugar_close", horizon=horizon)
        non_last = df["target_return"].iloc[:-horizon]
        assert non_last.notna().all(), (
            f"{non_last.isna().sum()} NaN values found before the last {horizon} rows"
        )

    def test_target_direction(self, synthetic_price_df):
        """target_direction should be 1 when target_return > 0, else 0."""
        df = build_target(synthetic_price_df, price_col="sugar_close", horizon=63)
        valid = df.dropna(subset=["target_return"])
        expected_dir = (valid["target_return"] > 0).astype(int)
        pd.testing.assert_series_equal(
            valid["target_direction"], expected_dir, check_names=False
        )

    def test_target_columns_exist(self, target_df):
        assert "target_return" in target_df.columns
        assert "target_direction" in target_df.columns

    def test_different_horizons(self, synthetic_price_df):
        """Different horizons should produce different NaN counts."""
        df_21 = build_target(synthetic_price_df, horizon=21)
        df_63 = build_target(synthetic_price_df, horizon=63)
        assert df_21["target_return"].isna().sum() == 21
        assert df_63["target_return"].isna().sum() == 63


class TestAddPriceFeatures:
    """Tests for add_price_features()."""

    def test_no_nan_after_dropna(self, featured_df):
        """After dropping NaN rows, first 252 rows of remaining data should be clean."""
        clean = featured_df.dropna()
        if len(clean) > 252:
            first_252 = clean.iloc[:252]
            assert first_252.isna().sum().sum() == 0, (
                "Found NaN values in first 252 rows after dropna"
            )

    def test_expected_return_columns(self, featured_df):
        for lag in [1, 5, 10, 21]:
            col = f"return_{lag}d"
            assert col in featured_df.columns, f"Missing column: {col}"

    def test_expected_sma_columns(self, featured_df):
        for window in [5, 10, 21, 50, 200]:
            assert f"sma_{window}" in featured_df.columns, f"Missing sma_{window}"
            assert f"price_vs_sma_{window}" in featured_df.columns

    def test_expected_volatility_columns(self, featured_df):
        for window in [10, 21, 63]:
            assert f"volatility_{window}d" in featured_df.columns

    def test_rsi_column(self, featured_df):
        assert "rsi_14" in featured_df.columns
        clean = featured_df["rsi_14"].dropna()
        assert (clean >= 0).all() and (clean <= 100).all(), "RSI should be between 0 and 100"

    def test_macd_columns(self, featured_df):
        assert "macd" in featured_df.columns
        assert "macd_signal" in featured_df.columns
        assert "macd_diff" in featured_df.columns

    def test_bollinger_columns(self, featured_df):
        assert "bb_high" in featured_df.columns
        assert "bb_low" in featured_df.columns
        assert "bb_pct" in featured_df.columns

    def test_atr_with_high_low(self, featured_df):
        """ATR should be computed when High and Low columns exist."""
        assert "atr_14" in featured_df.columns

    def test_seasonal_features(self, featured_df):
        """DatetimeIndex should produce seasonal features."""
        assert "day_of_week" in featured_df.columns
        assert "month" in featured_df.columns
        assert "season_sin" in featured_df.columns
        assert "season_cos" in featured_df.columns

    def test_zscore_columns(self, featured_df):
        assert "zscore_126d" in featured_df.columns
        assert "zscore_252d" in featured_df.columns

    def test_trend_columns(self, featured_df):
        for window in [21, 63, 126]:
            assert f"pct_up_days_{window}d" in featured_df.columns
            assert f"trend_slope_{window}d" in featured_df.columns

    def test_price_lag_columns(self, featured_df):
        for lag in [1, 2, 3, 5, 10]:
            assert f"price_lag_{lag}" in featured_df.columns

    def test_original_columns_preserved(self, featured_df):
        assert "sugar_close" in featured_df.columns


class TestNoLookAheadBias:
    """Features must only use past data -- no look-ahead bias."""

    def test_rolling_sma_uses_past_only(self, synthetic_price_df):
        """SMA at index i should only depend on data at indices <= i."""
        df = add_price_features(synthetic_price_df)
        # Check SMA_5: the value at index i should equal mean of prices [i-4..i]
        prices = synthetic_price_df["sugar_close"]
        for i in [10, 50, 200, 500]:
            expected = prices.iloc[i - 4:i + 1].mean()
            actual = df["sma_5"].iloc[i]
            assert abs(actual - expected) < 1e-10, (
                f"SMA_5 at index {i}: expected {expected}, got {actual}"
            )

    def test_return_uses_past_only(self, synthetic_price_df):
        """return_1d at index i should be (price[i] - price[i-1]) / price[i-1]."""
        df = add_price_features(synthetic_price_df)
        prices = synthetic_price_df["sugar_close"]
        for i in [5, 100, 500]:
            expected = (prices.iloc[i] / prices.iloc[i - 1]) - 1
            actual = df["return_1d"].iloc[i]
            assert abs(actual - expected) < 1e-10, (
                f"return_1d at index {i}: expected {expected}, got {actual}"
            )

    def test_price_lag_uses_past(self, synthetic_price_df):
        """price_lag_k at index i should equal price at index i-k."""
        df = add_price_features(synthetic_price_df)
        prices = synthetic_price_df["sugar_close"]
        for lag in [1, 5, 10]:
            for i in [20, 100, 500]:
                expected = prices.iloc[i - lag]
                actual = df[f"price_lag_{lag}"].iloc[i]
                assert abs(actual - expected) < 1e-10, (
                    f"price_lag_{lag} at index {i}: expected {expected}, got {actual}"
                )

    def test_modifying_future_doesnt_change_past_features(self, synthetic_price_df):
        """Changing a future price should not affect features computed at earlier indices."""
        df1 = add_price_features(synthetic_price_df.copy())
        modified = synthetic_price_df.copy()
        modified.iloc[600:, modified.columns.get_loc("sugar_close")] *= 2.0
        df2 = add_price_features(modified)

        # Features at index 500 should be identical
        check_idx = 500
        for col in ["sma_5", "sma_200", "return_1d", "rsi_14", "volatility_21d"]:
            v1 = df1[col].iloc[check_idx]
            v2 = df2[col].iloc[check_idx]
            if pd.notna(v1) and pd.notna(v2):
                assert abs(v1 - v2) < 1e-10, (
                    f"Modifying future data changed {col} at index {check_idx}: "
                    f"{v1} vs {v2}"
                )


class TestDropnaReasonable:
    """dropna() should not remove an unreasonable fraction of rows."""

    def test_dropna_keeps_majority(self, featured_df):
        """After adding features, dropna should not remove more than 40% of rows."""
        original_len = len(featured_df)
        clean_len = len(featured_df.dropna())
        drop_pct = 1 - clean_len / original_len
        assert drop_pct < 0.40, (
            f"dropna removed {drop_pct:.1%} of rows (max 40% allowed). "
            f"Original: {original_len}, after dropna: {clean_len}"
        )
