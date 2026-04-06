"""Feature Drift Agent — monitors whether feature distributions have shifted.

When features drift, the model is extrapolating into unfamiliar territory
and predictions become unreliable. This agent compares recent feature
distributions against the training window and flags significant shifts.

Usage:
    python -m agents drift                     # all commodities
    python -m agents drift coffee natgas       # specific commodities
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig
from .design_log import log_observation, log_challenge
from .log import setup_logging, log_event
from .signals import emit_signal

logger = setup_logging("feature_drift")

REGIME_FEATURES = {"vol_regime", "trend_regime", "drawdown"}

# Thresholds
ZSCORE_DRIFT_THRESHOLD = 2.0
ZSCORE_HIGH_THRESHOLD = 3.0
DRIFT_LOW_MAX = 2
DRIFT_MEDIUM_MAX = 5


# ── Per-commodity drift check ────────────────────────────────────────


def check_drift_commodity(cfg: CommodityConfig) -> dict:
    """Analyse feature drift for a single commodity.

    Returns a dict with:
        commodity, status, severity, drifted_features, out_of_range_features,
        regime_drift, feature_details, summary
    """
    result = {
        "commodity": cfg.name,
        "key": cfg.dir_name,
        "status": "ok",
        "severity": None,
        "drifted_features": [],
        "out_of_range_features": [],
        "regime_drift": False,
        "feature_details": {},
        "summary": "",
    }

    # Load feature data
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        result["status"] = "no_data"
        result["summary"] = f"No combined_features.csv for {cfg.name}"
        logger.warning(result["summary"])
        return result

    # Load feature list from metadata
    metadata_path = cfg.metadata_path
    if not metadata_path.exists():
        result["status"] = "no_metadata"
        result["summary"] = f"No production_metadata.json for {cfg.name}"
        logger.warning(result["summary"])
        return result

    with open(metadata_path) as f:
        metadata = json.load(f)

    feature_list = metadata.get("features", [])
    if not feature_list:
        result["status"] = "no_features"
        result["summary"] = f"Empty feature list in metadata for {cfg.name}"
        logger.warning(result["summary"])
        return result

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Only keep features that exist in the dataframe
    available_features = [f for f in feature_list if f in df.columns]
    if not available_features:
        result["status"] = "features_missing"
        result["summary"] = f"None of the metadata features found in CSV for {cfg.name}"
        logger.warning(result["summary"])
        return result

    # Split into training and recent windows
    # Training: all data except last 63 days (contiguous with recent window)
    # Recent: last 63 days
    if len(df) < 126:
        result["status"] = "insufficient_data"
        result["summary"] = f"Not enough data rows ({len(df)}) for drift analysis"
        logger.warning(result["summary"])
        return result

    train_df = df.iloc[:-63]
    recent_df = df.iloc[-63:]

    drifted = []
    out_of_range = []
    max_zscore = 0.0
    regime_drifted = []

    for feat in available_features:
        train_vals = train_df[feat].dropna()
        recent_vals = recent_df[feat].dropna()

        if len(train_vals) < 30 or len(recent_vals) < 5:
            continue

        train_mean = train_vals.mean()
        train_std = train_vals.std()

        recent_mean = recent_vals.mean()
        recent_std = recent_vals.std()

        # Z-score of recent mean relative to training distribution
        if train_std > 0:
            zscore = abs(recent_mean - train_mean) / train_std
        else:
            zscore = 0.0

        # Range check: any recent value outside training [min, max]
        train_min = train_vals.min()
        train_max = train_vals.max()
        oor_count = int(((recent_vals < train_min) | (recent_vals > train_max)).sum())

        detail = {
            "train_mean": round(float(train_mean), 6),
            "train_std": round(float(train_std), 6),
            "recent_mean": round(float(recent_mean), 6),
            "recent_std": round(float(recent_std), 6),
            "zscore": round(float(zscore), 3),
            "out_of_range_count": oor_count,
        }
        result["feature_details"][feat] = detail

        if zscore > max_zscore:
            max_zscore = zscore

        if zscore > ZSCORE_DRIFT_THRESHOLD:
            drifted.append(feat)
            if feat in REGIME_FEATURES:
                regime_drifted.append(feat)

        if oor_count > 0:
            out_of_range.append(feat)

    result["drifted_features"] = drifted
    result["out_of_range_features"] = out_of_range

    # Determine severity
    n_drifted = len(drifted)
    if n_drifted == 0:
        result["severity"] = None
        result["status"] = "ok"
        result["summary"] = f"{cfg.name}: no feature drift detected"
    elif n_drifted <= DRIFT_LOW_MAX and max_zscore <= ZSCORE_HIGH_THRESHOLD:
        result["severity"] = "low"
        result["status"] = "drift"
        result["summary"] = (
            f"{cfg.name}: low drift — {n_drifted} feature(s) shifted: "
            f"{', '.join(drifted)}"
        )
    elif n_drifted <= DRIFT_MEDIUM_MAX and max_zscore <= ZSCORE_HIGH_THRESHOLD:
        result["severity"] = "medium"
        result["status"] = "drift"
        result["summary"] = (
            f"{cfg.name}: medium drift — {n_drifted} feature(s) shifted: "
            f"{', '.join(drifted)}"
        )
    else:
        result["severity"] = "high"
        result["status"] = "drift"
        reason = []
        if n_drifted > DRIFT_MEDIUM_MAX:
            reason.append(f"{n_drifted} features drifted")
        if max_zscore > ZSCORE_HIGH_THRESHOLD:
            reason.append(f"max z-score {max_zscore:.1f}")
        result["summary"] = (
            f"{cfg.name}: HIGH drift — {'; '.join(reason)}. "
            f"Features: {', '.join(drifted)}"
        )

    # Regime drift flag
    if regime_drifted:
        result["regime_drift"] = True
        result["summary"] += (
            f" | MODEL OPERATING IN UNFAMILIAR REGIME "
            f"(drifted regime features: {', '.join(regime_drifted)})"
        )

    if out_of_range:
        result["summary"] += (
            f" | {len(out_of_range)} feature(s) with values outside training range"
        )

    return result


# ── Main entry points ────────────────────────────────────────────────


def run_drift_check(commodity_keys: list[str] = None) -> list[dict]:
    """Run drift check across commodities and emit signals.

    Args:
        commodity_keys: List of commodity keys to check. None = all.

    Returns:
        List of per-commodity drift analysis dicts.
    """
    if commodity_keys is None:
        commodity_keys = list(COMMODITIES.keys())

    results = []
    for key in commodity_keys:
        cfg = COMMODITIES.get(key)
        if cfg is None:
            logger.warning(f"Unknown commodity key: {key}")
            continue

        logger.info(f"Checking feature drift for {cfg.name}...")
        analysis = check_drift_commodity(cfg)
        results.append(analysis)

        # Log and emit signals
        if analysis["severity"]:
            logger.warning(analysis["summary"])
            log_observation("feature_drift", analysis["summary"], commodity=key)

            emit_signal(
                source_agent="feature_drift",
                signal_type="feature_drift",
                commodity=key,
                severity=analysis["severity"],
                detail=analysis["summary"],
                metadata={
                    "drifted_features": analysis["drifted_features"],
                    "out_of_range_features": analysis["out_of_range_features"],
                    "regime_drift": analysis["regime_drift"],
                },
            )

            # Challenge model validity if drift is medium or high
            if analysis["severity"] in ("medium", "high"):
                log_challenge(
                    agent="feature_drift",
                    decision="current model validity",
                    evidence=(
                        f"{cfg.name} has {len(analysis['drifted_features'])} "
                        f"drifted features — model may be extrapolating into "
                        f"unfamiliar territory. Consider retraining or reducing "
                        f"position size."
                    ),
                    commodity=key,
                )

            if analysis["regime_drift"]:
                log_challenge(
                    agent="feature_drift",
                    decision="current model validity",
                    evidence=(
                        f"{cfg.name} regime features have drifted — model is "
                        f"operating in an unfamiliar regime. Predictions are "
                        f"less trustworthy."
                    ),
                    commodity=key,
                )
        else:
            logger.info(analysis["summary"])

    # Print summary table
    print("\n" + "=" * 70)
    print("FEATURE DRIFT REPORT")
    print("=" * 70)
    for r in results:
        sev = r["severity"] or "none"
        drift_count = len(r["drifted_features"])
        oor_count = len(r["out_of_range_features"])
        regime = " [REGIME]" if r["regime_drift"] else ""
        print(
            f"  {r['commodity']:<15} severity={sev:<8} "
            f"drifted={drift_count}  out-of-range={oor_count}{regime}"
        )
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Feature Drift Agent — detect feature distribution shifts"
    )
    parser.add_argument(
        "commodities",
        nargs="*",
        default=None,
        help="Commodity keys to check (default: all)",
    )
    args = parser.parse_args()

    keys = args.commodities if args.commodities else None
    run_drift_check(commodity_keys=keys)


if __name__ == "__main__":
    main()
