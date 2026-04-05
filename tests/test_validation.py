"""Tests for the validation module."""

import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path("/Users/danielmurray/dev2/commodities")))

from agents.config import CommodityConfig
from agents.validation import check_fold_variance, validate_features


@pytest.fixture
def mock_cfg(tmp_path):
    """Create a CommodityConfig that points to a temp directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cfg = MagicMock(spec=CommodityConfig)
    cfg.name = "TestCommodity"
    cfg.models_dir = models_dir
    cfg.metadata_path = models_dir / "production_metadata.json"
    return cfg


def write_metadata(cfg, metadata: dict):
    """Helper to write metadata JSON for a mock config."""
    with open(cfg.metadata_path, "w") as f:
        json.dump(metadata, f)


class TestCheckFoldVariance:
    """Tests for check_fold_variance()."""

    def test_flags_high_std(self, mock_cfg):
        """Should flag when fold accuracy std > 0.15."""
        # Accuracies with std > 0.15
        accs = [0.50, 0.55, 0.90, 0.52, 0.88]
        write_metadata(mock_cfg, {
            "regression": {"fold_accuracies": accs},
        })
        result = check_fold_variance(mock_cfg, max_std=0.15)
        assert "regression" in result
        assert result["regression"]["high_variance"] is True
        assert result["regression"]["std"] > 0.15

    def test_does_not_flag_low_std(self, mock_cfg):
        """Should not flag when fold accuracy std < 0.15."""
        accs = [0.60, 0.62, 0.58, 0.61, 0.59]
        write_metadata(mock_cfg, {
            "classification": {"fold_accuracies": accs},
        })
        result = check_fold_variance(mock_cfg, max_std=0.15)
        assert "classification" in result
        assert result["classification"]["high_variance"] is False
        assert result["classification"]["std"] < 0.15

    def test_flags_perfect_folds(self, mock_cfg):
        """Should flag when any fold accuracy >= 0.99."""
        accs = [0.60, 0.99, 0.62, 0.58, 0.61]
        write_metadata(mock_cfg, {
            "classification": {"fold_accuracies": accs},
        })
        result = check_fold_variance(mock_cfg)
        assert result["classification"]["has_perfect_fold"] is True
        assert result["classification"]["suspicious"] is True

    def test_flags_all_perfect(self, mock_cfg):
        """All folds at 1.0 should flag as perfect and suspicious."""
        accs = [1.0, 1.0, 1.0, 1.0, 1.0]
        write_metadata(mock_cfg, {
            "regression": {"fold_accuracies": accs},
        })
        result = check_fold_variance(mock_cfg)
        assert result["regression"]["has_perfect_fold"] is True
        assert result["regression"]["suspicious"] is True

    def test_normal_folds_not_suspicious(self, mock_cfg):
        """Normal fold accuracies should not be flagged."""
        accs = [0.55, 0.58, 0.52, 0.57, 0.54]
        write_metadata(mock_cfg, {
            "classification": {"fold_accuracies": accs},
        })
        result = check_fold_variance(mock_cfg)
        assert result["classification"]["suspicious"] is False
        assert result["classification"]["has_perfect_fold"] is False
        assert result["classification"]["high_variance"] is False

    def test_no_metadata_returns_status(self, mock_cfg):
        """Should return no_metadata status when file missing."""
        # Don't write metadata
        result = check_fold_variance(mock_cfg)
        assert result == {"status": "no_metadata"}

    def test_both_model_types(self, mock_cfg):
        """Should check both regression and classification if present."""
        write_metadata(mock_cfg, {
            "regression": {"fold_accuracies": [0.50, 0.55, 0.90, 0.52, 0.88]},
            "classification": {"fold_accuracies": [0.55, 0.58, 0.52, 0.57, 0.54]},
        })
        result = check_fold_variance(mock_cfg)
        assert "regression" in result
        assert "classification" in result
        assert result["regression"]["high_variance"] is True
        assert result["classification"]["high_variance"] is False

    def test_custom_max_std(self, mock_cfg):
        """Custom max_std threshold should be respected."""
        accs = [0.50, 0.55, 0.60, 0.52, 0.58]
        write_metadata(mock_cfg, {
            "regression": {"fold_accuracies": accs},
        })
        # With a very low threshold, should flag
        result_strict = check_fold_variance(mock_cfg, max_std=0.01)
        assert result_strict["regression"]["high_variance"] is True

        # With a high threshold, should not flag
        result_relaxed = check_fold_variance(mock_cfg, max_std=0.50)
        assert result_relaxed["regression"]["high_variance"] is False


class TestValidateFeatures:
    """Tests for validate_features()."""

    def test_all_features_present(self):
        """When all expected features exist and are non-NaN, should report ok."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "feat_c": [7.0, 8.0, 9.0],
        })
        result = validate_features(df, ["feat_a", "feat_b", "feat_c"])
        assert result["ok"] is True
        assert result["missing"] == []
        assert result["nan_in_latest"] == []
        assert result["expected"] == 3
        assert result["available"] == 3

    def test_missing_features_reported(self):
        """Should report features that are not in the DataFrame."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
        })
        result = validate_features(df, ["feat_a", "feat_b", "feat_missing", "feat_gone"])
        assert result["ok"] is False
        assert "feat_missing" in result["missing"]
        assert "feat_gone" in result["missing"]
        assert len(result["missing"]) == 2

    def test_nan_features_reported(self):
        """Should report features that have NaN in the latest row."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, np.nan],  # NaN in last row
            "feat_c": [7.0, np.nan, 9.0],  # NaN in middle (ok)
        })
        result = validate_features(df, ["feat_a", "feat_b", "feat_c"])
        assert result["ok"] is False
        assert "feat_b" in result["nan_in_latest"]
        assert "feat_c" not in result["nan_in_latest"]

    def test_both_missing_and_nan(self):
        """Should report both missing features and NaN features."""
        df = pd.DataFrame({
            "feat_a": [1.0, np.nan],
            "feat_b": [3.0, 4.0],
        })
        result = validate_features(df, ["feat_a", "feat_b", "feat_missing"])
        assert result["ok"] is False
        assert "feat_missing" in result["missing"]
        assert "feat_a" in result["nan_in_latest"]

    def test_empty_expected_features(self):
        """With no expected features, should be ok."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = validate_features(df, [])
        assert result["ok"] is True
        assert result["expected"] == 0
