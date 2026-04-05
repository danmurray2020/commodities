"""Shared validation utilities for data quality, model health, and pipeline integrity."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, DATA_STALENESS_WARN_DAYS, DATA_STALENESS_FAIL_DAYS


logger = logging.getLogger("commodities")


# ── Data validation ────────────────────────────────────────────────────

def check_data_freshness(cfg: CommodityConfig) -> dict:
    """Check how stale the data is for a commodity."""
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "missing", "message": f"{csv_path} not found"}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.empty:
        return {"status": "empty", "message": f"{csv_path} has no rows"}

    latest_date = df.index[-1]
    age_days = (datetime.now() - latest_date).days

    if age_days > DATA_STALENESS_FAIL_DAYS:
        status = "stale"
    elif age_days > DATA_STALENESS_WARN_DAYS:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "age_days": age_days,
        "rows": len(df),
    }


def check_supplementary_data(cfg: CommodityConfig) -> dict:
    """Check freshness and completeness of supplementary data files."""
    results = {}
    for name in ["combined_features.csv", f"{cfg.dir_name}_cot.csv", "weather.csv", "enso.csv"]:
        # Handle special naming: coffee uses coffee_cot.csv, etc.
        if name.startswith(cfg.dir_name):
            path = cfg.data_dir / name
        else:
            cot_name = f"{cfg.price_col.replace('_close', '')}_cot.csv"
            if name == f"{cfg.dir_name}_cot.csv":
                path = cfg.data_dir / cot_name
            else:
                path = cfg.data_dir / name

        if not path.exists():
            results[name] = {"status": "missing", "rows": 0}
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            results[name] = {"status": "ok", "rows": len(df)}
            if df.empty:
                results[name]["status"] = "empty"
        except Exception as e:
            results[name] = {"status": "error", "message": str(e)}

    return results


def validate_features(df: pd.DataFrame, expected_features: list[str]) -> dict:
    """Validate that expected features exist and have no NaN issues."""
    available = [f for f in expected_features if f in df.columns]
    missing = [f for f in expected_features if f not in df.columns]

    latest = df.iloc[-1]
    nan_features = [f for f in available if pd.isna(latest[f])]

    return {
        "expected": len(expected_features),
        "available": len(available),
        "missing": missing,
        "nan_in_latest": nan_features,
        "ok": len(missing) == 0 and len(nan_features) == 0,
    }


# ── Model validation ──────────────────────────────────────────────────

def check_model_files(cfg: CommodityConfig) -> dict:
    """Verify model files exist and metadata is consistent."""
    results = {"status": "ok", "issues": []}

    # Check metadata
    if not cfg.metadata_path.exists():
        results["status"] = "missing"
        results["issues"].append("production_metadata.json not found")
        return results

    with open(cfg.metadata_path) as f:
        meta = json.load(f)

    # Check model files
    reg_path = cfg.models_dir / "production_regressor.joblib"
    clf_path = cfg.models_dir / "production_classifier.joblib"

    if not reg_path.exists():
        results["status"] = "missing"
        results["issues"].append("production_regressor.joblib not found")
    if not clf_path.exists():
        results["status"] = "missing"
        results["issues"].append("production_classifier.joblib not found")

    # Check metadata completeness
    for key in ["features", "horizon", "regression", "classification"]:
        if key not in meta:
            results["issues"].append(f"metadata missing key: {key}")
            results["status"] = "incomplete"

    if "features" in meta:
        results["n_features"] = len(meta["features"])
    if "regression" in meta:
        results["reg_accuracy"] = meta["regression"].get("avg_accuracy")
    if "classification" in meta:
        results["clf_accuracy"] = meta["classification"].get("avg_accuracy")

    return results


def check_fold_variance(cfg: CommodityConfig, max_std: float = 0.15) -> dict:
    """Flag models with high fold-to-fold variance (overfitting indicator)."""
    if not cfg.metadata_path.exists():
        return {"status": "no_metadata"}

    with open(cfg.metadata_path) as f:
        meta = json.load(f)

    results = {}
    for model_type in ["regression", "classification"]:
        if model_type not in meta:
            continue
        accs = meta[model_type].get("fold_accuracies", [])
        if not accs:
            continue

        std = float(np.std(accs))
        mean = float(np.mean(accs))
        has_perfect = any(a >= 0.99 for a in accs)

        results[model_type] = {
            "fold_accuracies": accs,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "range": round(max(accs) - min(accs), 4),
            "has_perfect_fold": has_perfect,
            "high_variance": std > max_std,
            "suspicious": has_perfect or std > max_std,
        }

    return results


# ── Pipeline validation ───────────────────────────────────────────────

def run_full_health_check(cfg: CommodityConfig) -> dict:
    """Run all validation checks for a single commodity."""
    return {
        "commodity": cfg.name,
        "data_freshness": check_data_freshness(cfg),
        "model_files": check_model_files(cfg),
        "fold_variance": check_fold_variance(cfg),
    }


def run_system_health_check() -> dict:
    """Run health checks across all commodities."""
    results = {}
    for key, cfg in COMMODITIES.items():
        results[key] = run_full_health_check(cfg)
    return results
