"""Monitoring / Audit Agent — ongoing model health and system integrity.

Responsibilities:
- Track prediction accuracy over time (realized vs predicted)
- Detect distribution shift or regime changes
- Monitor data staleness across all sources
- Generate health reports

Usage:
    python -m agents.monitoring             # full health check
    python -m agents.monitoring --accuracy  # prediction accuracy audit
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .design_log import log_observation, log_challenge
from .validation import (
    run_system_health_check,
    check_data_freshness,
    check_fold_variance,
)
from .log import setup_logging


logger = setup_logging("monitoring")


def audit_prediction_accuracy(lookback_days: int = 90) -> dict:
    """Compare past predictions against realized outcomes.

    Reads the predictions log and checks whether predicted direction
    matched actual price movement over the horizon.
    """
    predictions_log = COMMODITIES_DIR / "logs" / "predictions.jsonl"
    if not predictions_log.exists():
        return {"status": "no_log", "message": "No predictions log found"}

    entries = []
    with open(predictions_log) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return {"status": "empty", "message": "Predictions log is empty"}

    # Group by commodity and check realized outcomes
    results = {}
    cutoff = datetime.now() - timedelta(days=lookback_days)

    for key, cfg in COMMODITIES.items():
        commodity_preds = [
            e for e in entries
            if e.get("commodity", "").lower().replace(" ", "") == key
            or e.get("commodity") == cfg.name
        ]

        if not commodity_preds:
            continue

        # Load price data
        csv_path = cfg.data_dir / "combined_features.csv"
        if not csv_path.exists():
            continue

        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)[cfg.price_col]

        verified = []
        for pred in commodity_preds:
            pred_date = pd.Timestamp(pred["date"])
            horizon = pred.get("horizon", 63)
            target_date = pred_date + pd.Timedelta(days=horizon)

            # Skip if we don't have enough future data yet
            if target_date > prices.index[-1]:
                continue

            # Find closest available price
            mask = prices.index >= target_date
            if not mask.any():
                continue

            realized_price = float(prices.loc[mask].iloc[0])
            entry_price = pred["price"]
            realized_return = (realized_price / entry_price) - 1
            predicted_direction = pred["direction"]
            actual_direction = "UP" if realized_return > 0 else "DOWN"
            correct = predicted_direction == actual_direction

            verified.append({
                "date": pred["date"],
                "predicted": predicted_direction,
                "actual": actual_direction,
                "correct": correct,
                "confidence": pred["confidence"],
                "pred_return": pred["pred_return"],
                "realized_return": round(realized_return, 6),
            })

        if verified:
            accuracy = np.mean([v["correct"] for v in verified])
            high_conf = [v for v in verified if v["confidence"] >= cfg.confidence_threshold]
            hc_accuracy = np.mean([v["correct"] for v in high_conf]) if high_conf else None

            results[key] = {
                "total_predictions": len(verified),
                "accuracy": round(accuracy, 4),
                "high_confidence_count": len(high_conf),
                "high_confidence_accuracy": round(hc_accuracy, 4) if hc_accuracy else None,
            }

    return results


def check_regime_shift(cfg: CommodityConfig, window: int = 63) -> dict:
    """Detect potential market regime changes using volatility and trend."""
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data"}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    price = df[cfg.price_col]

    if len(price) < window * 4:
        return {"status": "insufficient_data"}

    # Current vs historical volatility
    current_vol = float(price.pct_change().tail(window).std() * np.sqrt(252))
    historical_vol = float(price.pct_change().std() * np.sqrt(252))
    vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

    # Current trend
    current_return = float((price.iloc[-1] / price.iloc[-window] - 1))
    annual_return = float((price.iloc[-1] / price.iloc[-252] - 1)) if len(price) >= 252 else None

    regime = "normal"
    if vol_ratio > 2.0:
        regime = "extreme_volatility"
    elif vol_ratio > 1.5:
        regime = "high_volatility"

    return {
        "regime": regime,
        "current_vol_annualized": round(current_vol, 4),
        "historical_vol_annualized": round(historical_vol, 4),
        "vol_ratio": round(vol_ratio, 2),
        "current_63d_return": round(current_return, 4),
        "annual_return": round(annual_return, 4) if annual_return else None,
    }


def generate_health_report() -> dict:
    """Generate comprehensive system health report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_health": run_system_health_check(),
        "regime_analysis": {},
        "prediction_audit": audit_prediction_accuracy(),
    }

    for key, cfg in COMMODITIES.items():
        report["regime_analysis"][key] = check_regime_shift(cfg)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="System monitoring")
    parser.add_argument("--accuracy", action="store_true", help="Run prediction accuracy audit only")
    args = parser.parse_args()

    if args.accuracy:
        results = audit_prediction_accuracy()
        print(f"\n{'='*60}")
        print("PREDICTION ACCURACY AUDIT")
        print(f"{'='*60}")
        for key, stats in results.items():
            cfg = COMMODITIES[key]
            print(f"\n  {cfg.name}:")
            print(f"    Total verified: {stats['total_predictions']}")
            print(f"    Direction accuracy: {stats['accuracy']:.1%}")
            if stats.get("high_confidence_accuracy") is not None:
                print(f"    High-conf accuracy: {stats['high_confidence_accuracy']:.1%} "
                      f"({stats['high_confidence_count']} trades)")
        return

    report = generate_health_report()

    print(f"\n{'='*60}")
    print("SYSTEM HEALTH REPORT")
    print(f"{'='*60}")

    # Data freshness
    print("\nDATA FRESHNESS:")
    for key, health in report["system_health"].items():
        cfg = COMMODITIES[key]
        fresh = health["data_freshness"]
        status_icon = {"ok": "OK", "warning": "WARN", "stale": "STALE"}.get(fresh.get("status"), "??")
        print(f"  {cfg.name:<15} [{status_icon}] age={fresh.get('age_days', '?')}d  rows={fresh.get('rows', '?')}")

    # Model health
    print("\nMODEL HEALTH:")
    for key, health in report["system_health"].items():
        cfg = COMMODITIES[key]
        model = health["model_files"]
        variance = health["fold_variance"]
        issues = model.get("issues", [])
        suspicious = any(
            isinstance(v, dict) and v.get("suspicious")
            for v in variance.values()
        )
        status = "WARN" if suspicious or issues else "OK"
        print(f"  {cfg.name:<15} [{status}]  "
              f"reg={model.get('reg_accuracy', '?')}  clf={model.get('clf_accuracy', '?')}")
        if suspicious:
            print(f"    ^ High fold variance — possible overfitting")

    # Regime
    print("\nMARKET REGIME:")
    for key, regime in report["regime_analysis"].items():
        cfg = COMMODITIES[key]
        if regime.get("status"):
            print(f"  {cfg.name:<15} {regime['status']}")
        else:
            r = regime["regime"].upper()
            vol = regime["vol_ratio"]
            ret = regime["current_63d_return"]
            print(f"  {cfg.name:<15} [{r}]  vol_ratio={vol:.1f}x  63d_return={ret:+.1%}")


if __name__ == "__main__":
    main()
