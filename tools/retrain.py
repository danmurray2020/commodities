"""Monthly retraining pipeline — retrains both models and validates before swapping.

Run monthly (e.g., first Saturday of each month).
Retrains on all available data, compares to existing model, and only
swaps if the new model is better or within tolerance.
"""

import json
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"


def retrain_commodity(project_dir: Path, name: str):
    """Retrain a commodity model and validate against the old one."""
    models_dir = project_dir / "models"
    timestamp = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print(f"RETRAINING {name.upper()}")
    print(f"{'='*60}")

    # Step 1: Refresh data
    print(f"\n1. Refreshing data...")
    result = subprocess.run(
        [sys.executable, "refresh.py"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=600,
    )
    if result.returncode != 0:
        print(f"  WARNING: refresh had errors (continuing anyway)")

    # Step 2: Backup current models
    print(f"\n2. Backing up current models...")
    backup_dir = models_dir / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    for f in models_dir.glob("production_*"):
        shutil.copy2(f, backup_dir / f.name)
    for f in models_dir.glob("v2_production_*"):
        shutil.copy2(f, backup_dir / f.name)
    print(f"  Backed up to {backup_dir}")

    # Step 3: Load old model metrics
    old_metrics = None
    for meta_file in ["v2_production_metadata.json", "production_metadata.json"]:
        path = models_dir / meta_file
        if path.exists():
            with open(path) as f:
                old_metrics = json.load(f)
            break

    old_clf_acc = old_metrics["classification"]["avg_accuracy"] if old_metrics else 0
    old_reg_acc = old_metrics["regression"]["avg_accuracy"] if old_metrics else 0
    print(f"  Old model: clf={old_clf_acc:.1%}, reg={old_reg_acc:.1%}")

    # Step 4: Retrain
    print(f"\n3. Retraining model...")
    train_script = "train.py"
    if (project_dir / "train_final_v2.py").exists():
        # Coffee has separate scripts; cocoa has all-in-one train.py
        # For coffee, run select_features then train_final_v2
        if name.lower() == "coffee":
            print("  Running feature selection...")
            subprocess.run(
                [sys.executable, "select_features.py"],
                capture_output=True, text=True, cwd=str(project_dir), timeout=600,
            )
            train_script = "train_final_v2.py"

    result = subprocess.run(
        [sys.executable, train_script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=1200,
    )
    if result.returncode != 0:
        print(f"  ERROR: Training failed!")
        print(f"  {result.stderr[-500:]}")
        print(f"  Restoring backup...")
        for f in backup_dir.glob("*"):
            shutil.copy2(f, models_dir / f.name)
        return False

    # Step 5: Load new model metrics
    new_metrics = None
    for meta_file in ["v2_production_metadata.json", "production_metadata.json"]:
        path = models_dir / meta_file
        if path.exists():
            with open(path) as f:
                new_metrics = json.load(f)
            break

    if new_metrics is None:
        print(f"  ERROR: No metadata after training")
        return False

    new_clf_acc = new_metrics["classification"]["avg_accuracy"]
    new_reg_acc = new_metrics["regression"]["avg_accuracy"]

    print(f"\n4. Comparing models...")
    print(f"  Old: clf={old_clf_acc:.1%}, reg={old_reg_acc:.1%}")
    print(f"  New: clf={new_clf_acc:.1%}, reg={new_reg_acc:.1%}")

    # Accept if new model is within 5% of old, or better
    clf_delta = new_clf_acc - old_clf_acc
    tolerance = -0.05  # allow up to 5% degradation

    if clf_delta >= tolerance:
        print(f"  ACCEPTED: clf delta = {clf_delta:+.1%} (threshold: {tolerance:+.0%})")
        # Also run strategy backtest
        if (project_dir / "strategy.py").exists():
            print(f"\n5. Running strategy backtest...")
            subprocess.run(
                [sys.executable, "-c",
                 "from strategy import run_strategy_backtest, TradeConfig; "
                 "run_strategy_backtest(TradeConfig(confidence_threshold=0.70, "
                 "stop_loss_pct=0.10, allow_short=True))"],
                capture_output=True, text=True, cwd=str(project_dir), timeout=600,
            )
            print(f"  Strategy backtest complete")
        return True
    else:
        print(f"  REJECTED: clf delta = {clf_delta:+.1%} (worse than {tolerance:+.0%})")
        print(f"  Restoring backup...")
        for f in backup_dir.glob("*"):
            shutil.copy2(f, models_dir / f.name)
        return False


def main():
    print("=" * 60)
    print(f"MONTHLY RETRAIN — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

    results = {}
    for project_dir, name in [(COFFEE_DIR, "Coffee"), (COCOA_DIR, "Cocoa")]:
        success = retrain_commodity(project_dir, name)
        results[name] = "UPDATED" if success else "KEPT OLD"

    print(f"\n{'='*60}")
    print("RETRAIN SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
