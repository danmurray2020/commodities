"""Calibration Agent — checks whether classifier confidence matches reality.

Verifies that predicted probabilities (confidence) are well-calibrated:
a model that says 80% confident should be right ~80% of the time.
Flags overconfident commodities and emits signals when miscalibrated.

Usage:
    python -m agents calibration                # all commodities
    python -m agents calibration coffee sugar    # specific commodities
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, COMMODITIES_DIR, CommodityConfig, LOGS_DIR
from .design_log import log_observation
from .log import setup_logging, log_event
from .signals import emit_signal

logger = setup_logging("calibration")

# Confidence buckets: (lower_bound_inclusive, upper_bound_exclusive, label)
BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]

MIN_PREDICTIONS_PER_BUCKET = 10
OVERCONFIDENCE_THRESHOLD = 0.15  # 15% gap between confidence and accuracy


# ── Helpers ──────────────────────────────────────────────────────────


def _load_predictions(cfg: CommodityConfig) -> list[dict]:
    """Load predictions for a commodity from logs/predictions.jsonl."""
    predictions_log = LOGS_DIR / "predictions.jsonl"
    if not predictions_log.exists():
        return []

    entries = []
    with open(predictions_log) as f:
        for line in f:
            try:
                entry = json.loads(line)
                commodity_field = entry.get("commodity", "")
                if (commodity_field.lower().replace(" ", "") == cfg.name.lower().replace(" ", "")
                        or commodity_field == cfg.name):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries


def _load_prices(cfg: CommodityConfig) -> pd.Series | None:
    """Load price series from combined_features.csv."""
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return None

    return df[cfg.price_col].dropna()


# ── Core calibration logic ───────────────────────────────────────────


def calibrate_commodity(key: str, cfg: CommodityConfig) -> dict:
    """Compute calibration results for a single commodity.

    Returns a dict with per-bucket accuracy, overconfidence flags,
    and overall calibration metrics.
    """
    predictions = _load_predictions(cfg)
    if not predictions:
        logger.info(f"{cfg.name}: no predictions found")
        return {"commodity": key, "status": "no_predictions"}

    price = _load_prices(cfg)
    if price is None:
        logger.info(f"{cfg.name}: no price data found")
        return {"commodity": key, "status": "no_data"}

    # Evaluate each prediction against actual outcome
    evaluated = []
    for pred in predictions:
        confidence = pred.get("confidence")
        direction = pred.get("direction")
        pred_date_str = pred.get("date")

        if confidence is None or direction is None or pred_date_str is None:
            continue

        pred_date = pd.Timestamp(pred_date_str)
        horizon = pred.get("horizon", cfg.horizon)
        target_date = pred_date + pd.Timedelta(days=horizon)

        # Need price at prediction date and after horizon
        if target_date > price.index[-1]:
            continue

        # Find entry price (on or after prediction date)
        entry_mask = price.index >= pred_date
        if not entry_mask.any():
            continue
        entry_price = float(price.loc[entry_mask].iloc[0])

        # Find realized price (on or after target date)
        target_mask = price.index >= target_date
        if not target_mask.any():
            continue
        realized_price = float(price.loc[target_mask].iloc[0])

        actual_direction = "UP" if realized_price > entry_price else "DOWN"
        correct = actual_direction == direction

        evaluated.append({
            "date": str(pred_date.date()),
            "confidence": float(confidence),
            "direction": direction,
            "actual_direction": actual_direction,
            "correct": correct,
        })

    if not evaluated:
        logger.info(f"{cfg.name}: no predictions with realized outcomes yet")
        return {"commodity": key, "status": "no_realized_outcomes", "n_predictions": len(predictions)}

    # Bucket predictions by confidence
    bucket_results = []
    overconfident_buckets = []

    for lower, upper, label in BUCKETS:
        in_bucket = [e for e in evaluated if lower <= e["confidence"] < upper]
        n = len(in_bucket)

        if n < MIN_PREDICTIONS_PER_BUCKET:
            bucket_results.append({
                "bucket": label,
                "n_predictions": n,
                "status": "insufficient_data",
                "min_required": MIN_PREDICTIONS_PER_BUCKET,
            })
            continue

        n_correct = sum(1 for e in in_bucket if e["correct"])
        realized_accuracy = n_correct / n
        midpoint_confidence = (lower + min(upper, 1.0)) / 2.0
        gap = midpoint_confidence - realized_accuracy

        bucket_info = {
            "bucket": label,
            "n_predictions": n,
            "n_correct": n_correct,
            "realized_accuracy": round(realized_accuracy, 4),
            "midpoint_confidence": round(midpoint_confidence, 4),
            "gap": round(gap, 4),
            "overconfident": gap > OVERCONFIDENCE_THRESHOLD,
        }
        bucket_results.append(bucket_info)

        if gap > OVERCONFIDENCE_THRESHOLD:
            overconfident_buckets.append(bucket_info)

    # Overall metrics
    total_evaluated = len(evaluated)
    total_correct = sum(1 for e in evaluated if e["correct"])
    overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
    mean_confidence = np.mean([e["confidence"] for e in evaluated])
    overall_gap = float(mean_confidence) - overall_accuracy

    # Determine calibration status
    if overconfident_buckets:
        worst_gap = max(b["gap"] for b in overconfident_buckets)
        if worst_gap > 0.25:
            calibration_status = "badly_miscalibrated"
            severity = "high"
        else:
            calibration_status = "overconfident"
            severity = "medium"
    else:
        calibration_status = "well_calibrated"
        severity = None

    result = {
        "commodity": key,
        "status": "ok",
        "calibration_status": calibration_status,
        "n_evaluated": total_evaluated,
        "overall_accuracy": round(overall_accuracy, 4),
        "mean_confidence": round(float(mean_confidence), 4),
        "overall_gap": round(overall_gap, 4),
        "buckets": bucket_results,
        "n_overconfident_buckets": len(overconfident_buckets),
    }

    # Emit signal if badly miscalibrated
    if calibration_status in ("overconfident", "badly_miscalibrated"):
        worst = max(overconfident_buckets, key=lambda b: b["gap"])
        detail = (
            f"{cfg.name} overconfident in {worst['bucket']} bucket: "
            f"predicted ~{worst['midpoint_confidence']:.0%} confidence but "
            f"realized {worst['realized_accuracy']:.0%} accuracy "
            f"(gap={worst['gap']:.0%}, n={worst['n_predictions']})"
        )
        emit_signal(
            "calibration", "calibration_off", key,
            severity=severity,
            detail=detail,
            metadata={
                "worst_bucket": worst["bucket"],
                "gap": worst["gap"],
                "n_overconfident_buckets": len(overconfident_buckets),
            },
        )
        log_observation(
            "calibration", detail, commodity=key,
        )
        logger.warning(detail)
    else:
        logger.info(
            f"{cfg.name}: well calibrated "
            f"(overall accuracy={overall_accuracy:.1%}, "
            f"mean confidence={mean_confidence:.1%})"
        )

    return result


# ── Run across commodities ───────────────────────────────────────────


def run_calibration(commodity_keys: list[str] | None = None) -> dict:
    """Run calibration analysis across specified (or all) commodities.

    Args:
        commodity_keys: List of commodity keys to check, or None for all.

    Returns:
        Report dict with per-commodity calibration results and summary.
    """
    keys = commodity_keys or list(COMMODITIES.keys())
    report = {
        "timestamp": datetime.now().isoformat(),
        "commodities": {},
        "alerts": [],
    }

    for key in keys:
        if key not in COMMODITIES:
            logger.warning(f"Unknown commodity: {key}")
            continue

        cfg = COMMODITIES[key]
        logger.info(f"Calibrating {cfg.name}...")
        result = calibrate_commodity(key, cfg)
        report["commodities"][key] = result

        if result.get("calibration_status") in ("overconfident", "badly_miscalibrated"):
            report["alerts"].append({
                "commodity": key,
                "name": cfg.name,
                "calibration_status": result["calibration_status"],
                "overall_gap": result.get("overall_gap"),
                "n_overconfident_buckets": result.get("n_overconfident_buckets"),
            })

    # Save report
    report_dir = LOGS_DIR / "calibration"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"calibration_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    return report


# ── CLI entry point ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Calibration Agent — check classifier confidence vs reality")
    parser.add_argument("commodities", nargs="*", help="Specific commodities to check (default: all)")
    args = parser.parse_args()

    keys = args.commodities if args.commodities else None
    report = run_calibration(keys)

    # Print summary
    print(f"\n{'='*60}")
    print("CALIBRATION REPORT")
    print(f"{'='*60}")

    for key, result in report["commodities"].items():
        cfg = COMMODITIES.get(key)
        name = cfg.name if cfg else key

        status = result.get("status", "unknown")
        if status != "ok":
            print(f"\n  {name}: {status}")
            continue

        cal_status = result.get("calibration_status", "unknown")
        marker = "OK" if cal_status == "well_calibrated" else "WARNING"
        print(f"\n  {name}: [{marker}] {cal_status}")
        print(f"    Overall: accuracy={result['overall_accuracy']:.1%}, "
              f"mean_confidence={result['mean_confidence']:.1%}, "
              f"gap={result['overall_gap']:+.1%}")
        print(f"    Evaluated: {result['n_evaluated']} predictions")

        # Per-bucket breakdown
        for bucket in result.get("buckets", []):
            if bucket.get("status") == "insufficient_data":
                print(f"    {bucket['bucket']:>8}: n={bucket['n_predictions']} "
                      f"(need >= {MIN_PREDICTIONS_PER_BUCKET})")
            else:
                flag = " << OVERCONFIDENT" if bucket.get("overconfident") else ""
                print(f"    {bucket['bucket']:>8}: "
                      f"accuracy={bucket['realized_accuracy']:.1%} "
                      f"(n={bucket['n_predictions']}, "
                      f"gap={bucket['gap']:+.1%}){flag}")

    # Alerts
    if report.get("alerts"):
        print(f"\n  ALERTS:")
        for alert in report["alerts"]:
            print(f"    {alert['name']}: {alert['calibration_status']} "
                  f"(gap={alert['overall_gap']:+.1%}, "
                  f"{alert['n_overconfident_buckets']} bad buckets)")
    else:
        print(f"\n  No calibration alerts.")

    print()


if __name__ == "__main__":
    main()
