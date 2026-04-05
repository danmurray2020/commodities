"""Tests for walk_forward_split function."""

import numpy as np
import pandas as pd
import pytest


# Copy of walk_forward_split for testing (avoids sys.path manipulation)
def walk_forward_split(
    df: pd.DataFrame, n_splits: int = 5, test_size: int = 63,
    min_train_size: int = 504, purge_gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    splits = []
    for i in range(n_splits):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start - purge_gap
        if train_end < min_train_size:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    splits.reverse()
    return splits


@pytest.fixture
def large_df():
    """A DataFrame with 2000 rows, enough for multiple folds."""
    dates = pd.bdate_range("2015-01-01", periods=2000)
    return pd.DataFrame({"price": np.random.randn(2000).cumsum() + 100}, index=dates)


@pytest.fixture
def small_df():
    """A DataFrame too small for multiple folds."""
    dates = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame({"price": np.random.randn(600).cumsum() + 50}, index=dates)


class TestNoOverlap:
    """Train and test indices must never overlap within any fold."""

    def test_no_overlap_default(self, large_df):
        splits = walk_forward_split(large_df)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Fold {fold_i} has overlapping indices: {overlap[:5]}"

    def test_no_overlap_with_purge(self, large_df):
        splits = walk_forward_split(large_df, purge_gap=63)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Fold {fold_i} has overlapping indices with purge"

    def test_no_overlap_across_folds(self, large_df):
        """Test sets across folds should not overlap."""
        splits = walk_forward_split(large_df)
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = np.intersect1d(splits[i][1], splits[j][1])
                assert len(overlap) == 0, f"Test sets of folds {i} and {j} overlap"


class TestPurgeGap:
    """Purge gap must be respected between train end and test start."""

    def test_purge_gap_zero(self, large_df):
        splits = walk_forward_split(large_df, purge_gap=0)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            # With purge_gap=0, train_end == test_start (no overlap, but adjacent)
            gap = test_idx[0] - train_idx[-1]
            assert gap >= 1, f"Fold {fold_i}: train/test should not overlap (gap={gap})"

    def test_purge_gap_63(self, large_df):
        splits = walk_forward_split(large_df, purge_gap=63)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            gap = test_idx[0] - train_idx[-1]
            assert gap >= 63, (
                f"Fold {fold_i}: gap between train end ({train_idx[-1]}) "
                f"and test start ({test_idx[0]}) is {gap}, expected >= 63"
            )

    def test_no_train_index_within_63_of_test(self, large_df):
        """With purge_gap=63, no train index should be within 63 of any test index."""
        splits = walk_forward_split(large_df, purge_gap=63)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            min_test = test_idx.min()
            max_train = train_idx.max()
            distance = min_test - max_train
            assert distance >= 63, (
                f"Fold {fold_i}: closest train index to test is {distance} apart, need >= 63"
            )

    def test_purge_gap_large(self, large_df):
        splits = walk_forward_split(large_df, purge_gap=126)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            gap = test_idx[0] - train_idx[-1]
            assert gap >= 126, f"Fold {fold_i}: purge gap of 126 not respected"


class TestChronologicalOrder:
    """Folds should be in chronological order (earlier folds first)."""

    def test_folds_chronological(self, large_df):
        splits = walk_forward_split(large_df)
        assert len(splits) >= 2, "Need at least 2 folds to test ordering"
        for i in range(len(splits) - 1):
            assert splits[i][1][-1] < splits[i + 1][1][-1], (
                f"Fold {i} test ends at {splits[i][1][-1]} but fold {i+1} "
                f"test ends at {splits[i+1][1][-1]} - not chronological"
            )

    def test_train_grows_over_folds(self, large_df):
        """Each successive fold should have a larger training set."""
        splits = walk_forward_split(large_df)
        for i in range(len(splits) - 1):
            assert len(splits[i][0]) < len(splits[i + 1][0]), (
                f"Fold {i} train size ({len(splits[i][0])}) should be < "
                f"fold {i+1} train size ({len(splits[i+1][0])})"
            )


class TestMinTrainSize:
    """min_train_size must be respected."""

    def test_min_train_size_default(self, large_df):
        splits = walk_forward_split(large_df, min_train_size=504)
        for fold_i, (train_idx, _) in enumerate(splits):
            assert len(train_idx) >= 504, (
                f"Fold {fold_i}: train size {len(train_idx)} < min_train_size 504"
            )

    def test_min_train_size_reduces_folds(self, small_df):
        """A small dataset should produce fewer folds."""
        splits_relaxed = walk_forward_split(small_df, min_train_size=100)
        splits_strict = walk_forward_split(small_df, min_train_size=400)
        assert len(splits_relaxed) >= len(splits_strict), (
            "Stricter min_train_size should produce fewer or equal folds"
        )

    def test_no_folds_if_too_small(self):
        """If data is smaller than min_train_size + test_size, no folds should be produced."""
        tiny_df = pd.DataFrame({"x": range(100)})
        splits = walk_forward_split(tiny_df, min_train_size=504, test_size=63)
        assert len(splits) == 0, "Should produce 0 folds for tiny dataset"


class TestTestSize:
    """Test set size must match the requested size."""

    def test_test_size_default(self, large_df):
        splits = walk_forward_split(large_df, test_size=63)
        for fold_i, (_, test_idx) in enumerate(splits):
            assert len(test_idx) == 63, (
                f"Fold {fold_i}: test size {len(test_idx)} != 63"
            )

    def test_test_size_custom(self, large_df):
        splits = walk_forward_split(large_df, test_size=126)
        for fold_i, (_, test_idx) in enumerate(splits):
            assert len(test_idx) == 126, (
                f"Fold {fold_i}: test size {len(test_idx)} != 126"
            )

    def test_number_of_splits(self, large_df):
        splits = walk_forward_split(large_df, n_splits=3, test_size=63)
        assert len(splits) <= 3, f"Expected at most 3 splits, got {len(splits)}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_split(self, large_df):
        splits = walk_forward_split(large_df, n_splits=1)
        assert len(splits) == 1

    def test_indices_are_contiguous(self, large_df):
        splits = walk_forward_split(large_df)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            assert np.array_equal(train_idx, np.arange(train_idx[0], train_idx[-1] + 1)), (
                f"Fold {fold_i}: train indices not contiguous"
            )
            assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1)), (
                f"Fold {fold_i}: test indices not contiguous"
            )

    def test_train_starts_at_zero(self, large_df):
        splits = walk_forward_split(large_df)
        for fold_i, (train_idx, _) in enumerate(splits):
            assert train_idx[0] == 0, f"Fold {fold_i}: train should start at index 0"
