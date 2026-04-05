"""Orchestrator — runs the full agent pipeline in the correct order.

Dependency chain:
    Data Pipeline → Prediction → Strategy/Risk
                 → Monitoring (parallel)

Usage:
    python -m agents.orchestrator                  # full weekly run
    python -m agents.orchestrator --predict-only   # skip data refresh
    python -m agents.orchestrator --retrain        # include retraining
    python -m agents.orchestrator coffee sugar      # specific commodities
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from .config import COMMODITIES, COMMODITIES_DIR
from .data_pipeline import refresh_all
from .prediction import predict_all
from .strategy import generate_trade_plan
from .monitoring import generate_health_report
from .training import retrain_all
from .log import setup_logging


logger = setup_logging("orchestrator")

REPORTS_DIR = COMMODITIES_DIR / "logs" / "reports"


def save_report(name: str, data: dict):
    """Save a timestamped report to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{name}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Report saved: {path}")


def run_weekly(
    commodity_keys: list[str] = None,
    skip_refresh: bool = False,
    include_retrain: bool = False,
) -> dict:
    """Execute the full weekly workflow.

    Steps:
        1. Data Pipeline Agent — refresh all data
        2. (Optional) Training Agent — retrain models
        3. Prediction Agent — generate predictions
        4. Strategy Agent — size positions and generate trade plan
        5. Monitoring Agent — health check
    """
    results = {"timestamp": datetime.now().isoformat()}

    # Log agent run to database
    run_id = None
    try:
        from db import get_db
        run_id = get_db().start_agent_run("orchestrator", commodity_keys)
    except Exception:
        pass

    # ── Step 1: Data refresh ──────────────────────────────────
    if not skip_refresh:
        logger.info("=" * 60)
        logger.info("STEP 1: DATA PIPELINE")
        logger.info("=" * 60)
        data_report = refresh_all(commodity_keys)
        results["data_pipeline"] = data_report
        save_report("data_pipeline", data_report)

        if not data_report["all_ok"]:
            logger.warning("Some data fetches failed — continuing with available data")
    else:
        logger.info("Skipping data refresh (--predict-only)")

    # ── Step 2: Retraining (optional) ────────────────────────
    if include_retrain:
        logger.info("=" * 60)
        logger.info("STEP 2: TRAINING")
        logger.info("=" * 60)
        train_reports = retrain_all(commodity_keys)
        results["training"] = train_reports
        save_report("training", train_reports)

    # ── Step 3: Predictions ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: PREDICTIONS")
    logger.info("=" * 60)
    predictions = predict_all(commodity_keys)
    results["predictions"] = predictions
    save_report("predictions", predictions)

    # ── Step 4: Strategy ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: STRATEGY")
    logger.info("=" * 60)
    trade_plan = generate_trade_plan(predictions)
    results["trade_plan"] = trade_plan
    save_report("trade_plan", trade_plan)

    # ── Step 5: Monitoring ──��────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: MONITORING")
    logger.info("=" * 60)
    health = generate_health_report()
    results["health"] = health
    save_report("health", health)

    # ── Summary ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WEEKLY RUN COMPLETE")
    logger.info("=" * 60)

    print(f"\n{'='*60}")
    print(f"WEEKLY SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Signals
    signals = trade_plan.get("signals", {})
    if signals:
        print(f"\nACTIVE SIGNALS ({len(signals)}):")
        for key, sig in signals.items():
            print(f"  {sig['direction']:>5} {sig['commodity']:<15} "
                  f"conf={sig['confidence']:.0%}  size={sig['final_size']:.1%}  "
                  f"ret={sig['pred_return']:+.1%}")
        print(f"  Total exposure: {trade_plan['total_exposure']:.1%}")
    else:
        print("\nNo active signals this week.")

    # Health warnings
    warnings = []
    for key, h in health.get("system_health", {}).items():
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue
        fresh = h.get("data_freshness", {})
        if fresh.get("status") in ("stale", "warning"):
            warnings.append(f"{cfg.name}: data {fresh.get('age_days')}d old")
        variance = h.get("fold_variance", {})
        for mt, v in variance.items():
            if isinstance(v, dict) and v.get("suspicious"):
                warnings.append(f"{cfg.name} {mt}: high fold variance")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    print(f"\nReports saved to: {REPORTS_DIR}")

    # Finish agent run in database
    if run_id:
        try:
            n_signals = len(trade_plan.get("signals", {}))
            summary = f"{n_signals} signals, exposure={trade_plan.get('total_exposure', 0):.1%}"
            if warnings:
                summary += f", {len(warnings)} warnings"
            get_db().finish_agent_run(run_id, "ok", summary=summary, report=results)
        except Exception:
            pass

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run commodities agent pipeline")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--predict-only", action="store_true", help="Skip data refresh")
    parser.add_argument("--retrain", action="store_true", help="Include model retraining")
    args = parser.parse_args()

    run_weekly(
        commodity_keys=args.commodities or None,
        skip_refresh=args.predict_only,
        include_retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
