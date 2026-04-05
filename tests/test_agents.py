"""Integration tests for the agent pipeline."""

import json
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.config import COMMODITIES, SIZING, TRADE_DEFAULTS, CommodityConfig
from agents.kelly import compute_kelly_size
from agents.validation import (
    check_data_freshness,
    check_model_files,
    check_fold_variance,
    validate_features,
)
from agents.strategy import (
    get_base_size,
    apply_correlation_adjustment,
    generate_trade_plan,
)


# ── Kelly criterion tests ────────────────────────────────────────────────

class TestKellyCriterion:
    """Tests for the shared Kelly sizing function."""

    def test_basic_kelly(self):
        """Standard Kelly with positive edge."""
        size = compute_kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05, fraction=1.0)
        # b = 0.10/0.05 = 2, kelly = (0.6*2 - 0.4)/2 = 0.4
        assert abs(size - 0.4) < 1e-10

    def test_half_kelly(self):
        """Half-Kelly should be half of full Kelly."""
        full = compute_kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05, fraction=1.0)
        half = compute_kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05, fraction=0.5)
        assert abs(half - full * 0.5) < 1e-10

    def test_no_edge(self):
        """50/50 win rate with equal win/loss should return 0."""
        size = compute_kelly_size(win_rate=0.5, avg_win=0.10, avg_loss=0.10)
        assert size == 0.0

    def test_negative_edge(self):
        """Losing edge should return 0 (clamped)."""
        size = compute_kelly_size(win_rate=0.3, avg_win=0.05, avg_loss=0.10)
        assert size == 0.0

    def test_zero_avg_loss(self):
        """Zero avg_loss should return 0 (avoid division by zero)."""
        size = compute_kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.0)
        assert size == 0.0

    def test_zero_win_rate(self):
        size = compute_kelly_size(win_rate=0.0, avg_win=0.10, avg_loss=0.05)
        assert size == 0.0

    def test_capped_at_one(self):
        """Result should never exceed 1.0."""
        size = compute_kelly_size(win_rate=0.99, avg_win=10.0, avg_loss=0.01, fraction=1.0)
        assert size <= 1.0


# ── Validation tests ─────────────────────────────────────────────────────

class TestValidation:
    """Tests for validation utilities."""

    def test_validate_features_all_present(self):
        """All expected features present and non-NaN."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        result = validate_features(df, ["feat_a", "feat_b"])
        assert result["ok"] is True
        assert result["missing"] == []
        assert result["nan_in_latest"] == []

    def test_validate_features_missing(self):
        """Missing features should be flagged."""
        df = pd.DataFrame({"feat_a": [1.0]})
        result = validate_features(df, ["feat_a", "feat_b"])
        assert result["ok"] is False
        assert "feat_b" in result["missing"]

    def test_validate_features_nan_in_latest(self):
        """NaN in latest row should be flagged."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, float("nan")],
            "feat_b": [4.0, 5.0, 6.0],
        })
        result = validate_features(df, ["feat_a", "feat_b"])
        assert result["ok"] is False
        assert "feat_a" in result["nan_in_latest"]

    def test_check_fold_variance_high_std(self):
        """High fold variance should be flagged as suspicious."""
        cfg = MagicMock()
        cfg.metadata_path = Path("/tmp/_test_metadata.json")
        meta = {
            "regression": {
                "fold_accuracies": [0.95, 0.30, 0.80, 0.70, 0.25],
            },
            "classification": {
                "fold_accuracies": [0.80, 0.82, 0.78, 0.81, 0.79],
            },
        }
        cfg.metadata_path.write_text(json.dumps(meta))
        try:
            result = check_fold_variance(cfg)
            assert result["regression"]["high_variance"] is True
            assert result["classification"]["high_variance"] is False
        finally:
            cfg.metadata_path.unlink(missing_ok=True)

    def test_check_fold_variance_perfect_fold(self):
        """A perfect fold (1.0) should be flagged as suspicious."""
        cfg = MagicMock()
        cfg.metadata_path = Path("/tmp/_test_metadata_perfect.json")
        meta = {
            "regression": {
                "fold_accuracies": [1.0, 0.80, 0.85, 0.82, 0.78],
            },
        }
        cfg.metadata_path.write_text(json.dumps(meta))
        try:
            result = check_fold_variance(cfg)
            assert result["regression"]["has_perfect_fold"] is True
            assert result["regression"]["suspicious"] is True
        finally:
            cfg.metadata_path.unlink(missing_ok=True)


# ── Strategy agent tests ─────────────────────────────────────────────────

class TestStrategyAgent:
    """Tests for the strategy/trade plan generation."""

    def test_get_base_size_tiers(self):
        """Base size should increase with confidence."""
        assert get_base_size(0.70) == 0.0  # below threshold
        assert get_base_size(0.76) == SIZING.base_sizes["75-80"]
        assert get_base_size(0.82) == SIZING.base_sizes["80-85"]
        assert get_base_size(0.87) == SIZING.base_sizes["85-90"]
        assert get_base_size(0.95) == SIZING.base_sizes["90+"]

    def test_generate_trade_plan_no_signals(self):
        """Predictions below threshold should produce no signals."""
        predictions = {
            "coffee": {
                "direction": "UP",
                "confidence": 0.50,
                "signal": False,
                "price": 200.0,
                "pred_return": 0.05,
            },
        }
        plan = generate_trade_plan(predictions)
        assert plan["n_signals"] == 0
        assert "coffee" in plan["no_trade"]

    def test_generate_trade_plan_with_signal(self):
        """High-confidence prediction should generate a signal."""
        predictions = {
            "coffee": {
                "direction": "UP",
                "confidence": 0.90,
                "signal": True,
                "price": 200.0,
                "pred_return": 0.10,
            },
        }
        plan = generate_trade_plan(predictions)
        assert plan["n_signals"] == 1
        assert "coffee" in plan["signals"]
        sig = plan["signals"]["coffee"]
        assert sig["direction"] == "LONG"
        assert sig["tp_price"] > sig["price"]
        assert sig["sl_price"] < sig["price"]

    def test_generate_trade_plan_short(self):
        """DOWN prediction should generate SHORT signal."""
        predictions = {
            "wheat": {
                "direction": "DOWN",
                "confidence": 0.85,
                "signal": True,
                "price": 600.0,
                "pred_return": -0.08,
            },
        }
        plan = generate_trade_plan(predictions)
        assert plan["signals"]["wheat"]["direction"] == "SHORT"

    def test_correlation_adjustment(self):
        """Correlated signals should have reduced sizing."""
        signals = {
            "coffee": {"base_size": 0.12},
            "sugar": {"base_size": 0.12},
        }
        adjusted = apply_correlation_adjustment(signals)
        assert adjusted["coffee"]["size_multiplier"] < 1.0
        assert adjusted["sugar"]["size_multiplier"] < 1.0

    def test_portfolio_exposure_cap(self):
        """Total exposure should not exceed max_portfolio_exposure."""
        predictions = {}
        for key in ["coffee", "cocoa", "sugar", "wheat", "soybeans", "natgas", "copper"]:
            predictions[key] = {
                "direction": "UP",
                "confidence": 0.95,
                "signal": True,
                "price": 100.0,
                "pred_return": 0.20,
            }
        plan = generate_trade_plan(predictions)
        assert plan["total_exposure"] <= SIZING.max_portfolio_exposure + 0.001

    def test_none_predictions_handled(self):
        """None predictions should be skipped gracefully."""
        predictions = {"coffee": None, "wheat": None}
        plan = generate_trade_plan(predictions)
        assert plan["n_signals"] == 0


# ── Config consistency tests ─────────────────────────────────────────────

class TestConfigConsistency:
    """Verify config matches actual project structure."""

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_commodity_project_dir_exists(self, key):
        cfg = COMMODITIES[key]
        assert cfg.project_dir.exists(), f"{cfg.name} project dir missing: {cfg.project_dir}"

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_commodity_has_production_metadata(self, key):
        cfg = COMMODITIES[key]
        assert cfg.metadata_path.exists(), f"{cfg.name} missing production_metadata.json"

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_commodity_has_model_files(self, key):
        cfg = COMMODITIES[key]
        reg = cfg.models_dir / "production_regressor.joblib"
        clf = cfg.models_dir / "production_classifier.joblib"
        assert reg.exists(), f"{cfg.name} missing production_regressor.joblib"
        assert clf.exists(), f"{cfg.name} missing production_classifier.joblib"

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_metadata_has_required_keys(self, key):
        cfg = COMMODITIES[key]
        with open(cfg.metadata_path) as f:
            meta = json.load(f)
        for required in ["features", "horizon", "regression", "classification"]:
            assert required in meta, f"{cfg.name} metadata missing '{required}'"
        assert meta["horizon"] in (10, 21, 42, 63), f"{cfg.name} unexpected horizon: {meta['horizon']}"

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_metadata_features_nonempty(self, key):
        cfg = COMMODITIES[key]
        with open(cfg.metadata_path) as f:
            meta = json.load(f)
        assert len(meta["features"]) > 0, f"{cfg.name} has 0 features in metadata"

    @pytest.mark.parametrize("key", list(COMMODITIES.keys()))
    def test_commodity_has_data(self, key):
        cfg = COMMODITIES[key]
        csv = cfg.data_dir / "combined_features.csv"
        assert csv.exists(), f"{cfg.name} missing combined_features.csv"


# ── Retry utility tests ──────────────────────────────────────────────────

class TestRetryUtility:
    """Tests for the retry decorator."""

    def test_retry_succeeds_on_third_attempt(self):
        from agents.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max(self):
        from agents.retry import retry_with_backoff

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            always_fails()

    def test_retry_no_retry_on_success(self):
        from agents.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def works_first_time():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = works_first_time()
        assert result == "ok"
        assert call_count == 1
