"""Tests for commodities configuration."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/danielmurray/dev2/commodities")))

from agents.config import COMMODITIES, CORRELATION_GROUPS, CommodityConfig


class TestCommodityConfigs:
    """All commodity configs should have valid settings."""

    def test_all_have_valid_project_dirs(self):
        """Every commodity's project_dir should exist on disk."""
        for key, cfg in COMMODITIES.items():
            assert cfg.project_dir.exists(), (
                f"Commodity '{key}': project_dir {cfg.project_dir} does not exist"
            )

    def test_all_have_name(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.name, f"Commodity '{key}' has no name"
            assert isinstance(cfg.name, str)

    def test_all_have_ticker(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.ticker, f"Commodity '{key}' has no ticker"
            assert isinstance(cfg.ticker, str)

    def test_all_have_price_col(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.price_col, f"Commodity '{key}' has no price_col"
            assert isinstance(cfg.price_col, str)
            assert cfg.price_col.endswith("_close"), (
                f"Commodity '{key}': price_col '{cfg.price_col}' should end with '_close'"
            )

    def test_all_have_dir_name(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.dir_name, f"Commodity '{key}' has no dir_name"

    def test_all_have_positive_horizon(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.horizon > 0, f"Commodity '{key}' has non-positive horizon"

    def test_confidence_threshold_in_range(self):
        for key, cfg in COMMODITIES.items():
            assert 0 < cfg.confidence_threshold <= 1.0, (
                f"Commodity '{key}': confidence_threshold {cfg.confidence_threshold} "
                f"not in (0, 1]"
            )

    def test_expected_commodities_present(self):
        """All 7 commodities should be configured."""
        expected = {"coffee", "cocoa", "sugar", "natgas", "soybeans", "wheat", "copper"}
        actual = set(COMMODITIES.keys())
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"


class TestCorrelationGroups:
    """CORRELATION_GROUPS should cover all commodities."""

    def test_all_commodities_in_groups(self):
        """Every commodity should appear in exactly one correlation group."""
        all_in_groups = []
        for group_name, members in CORRELATION_GROUPS.items():
            all_in_groups.extend(members)

        commodity_keys = set(COMMODITIES.keys())
        grouped_keys = set(all_in_groups)

        missing = commodity_keys - grouped_keys
        assert len(missing) == 0, (
            f"Commodities not in any correlation group: {missing}"
        )

    def test_no_duplicates_across_groups(self):
        """No commodity should appear in multiple correlation groups."""
        all_in_groups = []
        for members in CORRELATION_GROUPS.values():
            all_in_groups.extend(members)
        assert len(all_in_groups) == len(set(all_in_groups)), (
            "Some commodities appear in multiple correlation groups"
        )

    def test_group_members_are_valid_commodities(self):
        """All members of correlation groups should be valid commodity keys."""
        for group_name, members in CORRELATION_GROUPS.items():
            for member in members:
                assert member in COMMODITIES, (
                    f"Correlation group '{group_name}' member '{member}' "
                    f"is not a valid commodity"
                )

    def test_groups_are_non_empty(self):
        for group_name, members in CORRELATION_GROUPS.items():
            assert len(members) > 0, f"Correlation group '{group_name}' is empty"


class TestCommodityConfigProperties:
    """Test computed properties of CommodityConfig."""

    def test_models_dir_is_under_project_dir(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.models_dir.parent == cfg.project_dir, (
                f"Commodity '{key}': models_dir should be under project_dir"
            )

    def test_data_dir_is_under_project_dir(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.data_dir.parent == cfg.project_dir, (
                f"Commodity '{key}': data_dir should be under project_dir"
            )

    def test_metadata_path_is_under_models_dir(self):
        for key, cfg in COMMODITIES.items():
            assert cfg.metadata_path.parent == cfg.models_dir, (
                f"Commodity '{key}': metadata_path should be under models_dir"
            )
