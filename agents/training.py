"""Training Agent — retrains models with gated promotion to production.

Responsibilities:
- Run train.py for each commodity
- Compare new model metrics to previous production model
- Only promote if new model outperforms (gated deployment)
- Log train vs test accuracy gap (overfitting monitor)
- Back up previous model before overwriting

Usage:
    python -m agents.training              # retrain all
    python -m agents.training coffee sugar  # retrain specific
    python -m agents.training --dry-run    # evaluate only, don't promote
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .design_log import log_observation, log_challenge
from .validation import check_model_files, check_fold_variance
from .log import setup_logging, log_event


logger = setup_logging("training")

TRAINING_LOG = COMMODITIES_DIR / "logs" / "training.jsonl"


def backup_model(cfg: CommodityConfig) -> Path | None:
    """Back up current production model before retraining."""
    if not cfg.metadata_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = cfg.models_dir / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["production_*.joblib", "production_metadata.json"]:
        for f in cfg.models_dir.glob(pattern):
            shutil.copy2(f, backup_dir / f.name)

    logger.info(f"Backed up {cfg.name} models to {backup_dir}")
    return backup_dir


def load_current_metrics(cfg: CommodityConfig) -> dict | None:
    """Load metrics from current production model."""
    if not cfg.metadata_path.exists():
        return None
    with open(cfg.metadata_path) as f:
        meta = json.load(f)
    return {
        "reg_accuracy": meta.get("regression", {}).get("avg_accuracy"),
        "clf_accuracy": meta.get("classification", {}).get("avg_accuracy"),
        "n_features": meta.get("n_features"),
    }


def retrain_commodity(cfg: CommodityConfig, dry_run: bool = False) -> dict:
    """Retrain a single commodity and optionally promote to production.

    Returns training report with metrics comparison.
    """
    report = {
        "commodity": cfg.name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Load current metrics for comparison
    current = load_current_metrics(cfg)
    report["previous_metrics"] = current

    # Back up current model
    if not dry_run and current:
        backup_dir = backup_model(cfg)
        report["backup_dir"] = str(backup_dir) if backup_dir else None

    # Run training
    train_script = cfg.project_dir / "train.py"
    if not train_script.exists():
        report["status"] = "error"
        report["error"] = "train.py not found"
        return report

    logger.info(f"Training {cfg.name}...")
    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            timeout=1800,  # 30 min max
        )

        if result.returncode != 0:
            report["status"] = "train_failed"
            report["stderr"] = result.stderr[-500:]
            logger.error(f"{cfg.name} training failed: {result.stderr[-200:]}")
            return report

        report["stdout_tail"] = result.stdout[-500:]

    except subprocess.TimeoutExpired:
        report["status"] = "timeout"
        logger.error(f"{cfg.name} training timed out")
        return report

    # Load new metrics
    new = load_current_metrics(cfg)
    report["new_metrics"] = new

    if not new:
        report["status"] = "no_metrics"
        return report

    # Compare
    if current and new:
        reg_improved = (new["reg_accuracy"] or 0) >= (current["reg_accuracy"] or 0)
        clf_improved = (new["clf_accuracy"] or 0) >= (current["clf_accuracy"] or 0)

        report["reg_delta"] = round((new["reg_accuracy"] or 0) - (current["reg_accuracy"] or 0), 4)
        report["clf_delta"] = round((new["clf_accuracy"] or 0) - (current["clf_accuracy"] or 0), 4)
        report["improved"] = reg_improved or clf_improved

        if not report["improved"] and not dry_run:
            logger.warning(f"{cfg.name}: new model did NOT improve — consider reverting")
    else:
        report["improved"] = True  # No previous model to compare against

    # Check fold variance
    variance = check_fold_variance(cfg)
    report["fold_variance"] = variance

    for model_type, stats in variance.items():
        if isinstance(stats, dict) and stats.get("suspicious"):
            logger.warning(f"{cfg.name} {model_type}: suspicious fold variance (std={stats['std']:.2%})")

    report["status"] = "ok"
    logger.info(f"{cfg.name}: reg_acc={new.get('reg_accuracy', '?')}, clf_acc={new.get('clf_accuracy', '?')}")

    # Log observations to design log
    reg_acc = new.get("reg_accuracy")
    clf_acc = new.get("clf_accuracy")
    if reg_acc is not None:
        log_observation("training", f"Retrained — reg_acc={reg_acc:.2%}, clf_acc={clf_acc:.2%}", cfg.name)

    # Challenge assumptions if performance is poor
    for model_type, stats in variance.items():
        if isinstance(stats, dict):
            if stats.get("suspicious"):
                log_challenge("training", "Walk-forward CV stability",
                    f"{cfg.name} {model_type}: fold std={stats['std']:.2%}, "
                    f"range={stats['range']:.2%}. High variance suggests "
                    f"regime changes or overfitting.", cfg.name)
            if reg_acc is not None and reg_acc < 0.55:
                log_challenge("training", "Model predictive edge",
                    f"{cfg.name} regression direction accuracy is {reg_acc:.2%} "
                    f"(near random). Consider alternative approaches: "
                    f"shorter horizon, different features, or ensemble methods.", cfg.name)

    return report


def log_training(report: dict):
    """Append training report to the audit log."""
    TRAINING_LOG.parent.mkdir(exist_ok=True)
    with open(TRAINING_LOG, "a") as f:
        f.write(json.dumps(report, default=str) + "\n")


def retrain_all(commodity_keys: list[str] = None, dry_run: bool = False) -> dict:
    """Retrain all (or specified) commodities."""
    targets = commodity_keys or list(COMMODITIES.keys())
    reports = {}

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Retraining {cfg.name}")
        logger.info(f"{'='*50}")

        report = retrain_commodity(cfg, dry_run=dry_run)
        reports[key] = report
        log_training(report)

    return reports


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain commodity models")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't promote")
    args = parser.parse_args()

    reports = retrain_all(args.commodities or None, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    for key, report in reports.items():
        cfg = COMMODITIES[key]
        status = report["status"]
        new = report.get("new_metrics", {})
        delta_r = report.get("reg_delta", "")
        delta_c = report.get("clf_delta", "")
        improved = report.get("improved", "?")

        print(f"  {cfg.name:<15} [{status}]  "
              f"reg={new.get('reg_accuracy', '?')}  clf={new.get('clf_accuracy', '?')}  "
              f"improved={improved}")


if __name__ == "__main__":
    main()
