"""Model Quality Agent — diagnoses model issues and applies per-commodity remediation.

Unlike other agents that observe, this one acts. It reads current model metrics,
diagnoses specific problems, generates per-commodity training configs, retrains
with those configs, and logs every decision.

Runs weekly after the regular training agent. Can also be run standalone.

Usage:
    python -m agents quality                    # diagnose + remediate all
    python -m agents quality coffee cocoa       # specific commodities
    python -m agents quality --diagnose-only    # diagnose without retraining
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .design_log import log_observation, log_challenge
from .log import setup_logging
from .signals import get_active_signals, emit_signal, resolve_signal

logger = setup_logging("model_quality")

QUALITY_LOG = COMMODITIES_DIR / "logs" / "model_quality.jsonl"
COMMODITY_CONFIGS_DIR = COMMODITIES_DIR / "configs"


# ── History tracking ─────────────────────────────────────────────────────

def load_history(key: str) -> list[dict]:
    """Load past quality agent runs for a commodity.

    Returns list of past {config, diagnosis, retrain} dicts, newest first.
    """
    if not QUALITY_LOG.exists():
        return []

    entries = []
    with open(QUALITY_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("key") == key:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return list(reversed(entries))  # newest first


def get_tried_actions(key: str) -> list[dict]:
    """Get list of previously tried actions and their outcomes.

    Returns list of {action, horizon, reg_acc_before, reg_acc_after, improved}.
    """
    history = load_history(key)
    tried = []
    for entry in history:
        config = entry.get("config", {})
        diag = entry.get("diagnosis", {})
        retrain = entry.get("retrain", {})

        old_acc = diag.get("metrics", {}).get("reg_acc")
        new_acc = retrain.get("new_metrics", {}).get("reg_acc")

        actions = [a["action"] for a in diag.get("actions", [])]
        tried.append({
            "actions": actions,
            "horizon": config.get("horizon"),
            "old_acc": old_acc,
            "new_acc": new_acc,
            "improved": (new_acc or 0) > (old_acc or 0) if old_acc and new_acc else None,
            "timestamp": entry.get("diagnosis", {}).get("timestamp"),
        })

    return tried


# ── Diagnosis ────────────────────────────────────────────────────────────

def load_metrics(cfg: CommodityConfig) -> dict | None:
    """Load current production metadata for a commodity."""
    if not cfg.metadata_path.exists():
        return None
    with open(cfg.metadata_path) as f:
        return json.load(f)


def diagnose_commodity(cfg: CommodityConfig) -> dict:
    """Analyze a commodity's model health and identify specific issues.

    Returns a diagnosis dict with issues found and recommended actions.
    """
    meta = load_metrics(cfg)
    if not meta:
        return {"status": "no_model", "issues": ["No production metadata found"], "actions": []}

    diag = {
        "commodity": cfg.name,
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "issues": [],
        "actions": [],
        "metrics": {},
    }

    reg = meta.get("regression", {})
    clf = meta.get("classification", {})
    holdout = meta.get("holdout", {})
    horizon = meta.get("horizon", 63)
    n_features = meta.get("n_features", 0)

    # Extract key metrics
    reg_acc = reg.get("avg_accuracy") or reg.get("avg_dir_acc_independent")
    reg_std = reg.get("std_accuracy", 0)
    reg_spearman = reg.get("avg_spearman")
    clf_acc = clf.get("avg_accuracy") or clf.get("avg_acc_independent")
    ho_reg = holdout.get("reg_direction_accuracy") or holdout.get("reg_dir_acc_independent")
    ho_spearman = holdout.get("reg_spearman")
    fold_accs = reg.get("fold_accuracies", [])

    diag["metrics"] = {
        "reg_acc": reg_acc, "reg_std": reg_std, "reg_spearman": reg_spearman,
        "clf_acc": clf_acc, "holdout_reg": ho_reg, "holdout_spearman": ho_spearman,
        "horizon": horizon, "n_features": n_features,
    }

    # ── Load history to avoid repeating failed strategies ──

    key = cfg.dir_name if cfg.dir_name != "chocolate" else "cocoa"
    # Try matching on commodity name in lowercase
    for k, c in COMMODITIES.items():
        if c.name == cfg.name:
            key = k
            break

    tried = get_tried_actions(key)
    tried_horizons = {t["horizon"] for t in tried if t.get("horizon")}
    failed_actions = set()
    for t in tried:
        if t.get("improved") is False:
            for a in t.get("actions", []):
                failed_actions.add((a, t.get("horizon")))

    diag["history"] = {
        "n_previous_runs": len(tried),
        "tried_horizons": sorted(tried_horizons),
        "failed_action_count": len(failed_actions),
    }

    # Helper: determine next horizon to try based on history
    def pick_horizon():
        """Pick the next horizon to try, avoiding ones that already failed."""
        candidates = [21, 42, 10, 63]
        for h in candidates:
            if h != horizon and h not in tried_horizons:
                return h
        # All tried — pick the one that performed best historically
        best_h, best_acc = horizon, reg_acc or 0
        for t in tried:
            if t.get("new_acc") and t["new_acc"] > best_acc:
                best_h = t["horizon"]
                best_acc = t["new_acc"]
        return best_h if best_h != horizon else None

    # ── Issue detection ──

    # 1. Below-random regression direction
    if reg_acc is not None and reg_acc < 0.50:
        diag["issues"].append(f"Regression direction accuracy ({reg_acc:.1%}) below random")
        diag["status"] = "poor"

        next_h = pick_horizon()
        if next_h and next_h != horizon:
            diag["actions"].append({
                "action": "try_shorter_horizon",
                "param": next_h,
                "reason": f"Trying horizon={next_h}d (current {horizon}d below random, "
                          f"previously tried: {sorted(tried_horizons) or 'none'})",
            })
        diag["actions"].append({
            "action": "increase_regularization",
            "reason": "Model may be overfitting to training regime",
        })

    # 2. High fold variance
    if reg_std is not None and reg_std > 0.15:
        diag["issues"].append(f"High fold variance (std={reg_std:.1%})")
        if "poor" not in diag["status"]:
            diag["status"] = "unstable"
        diag["actions"].append({
            "action": "increase_regularization",
            "reason": f"Fold std={reg_std:.1%} suggests overfitting to specific regimes",
        })

    # 3. Perfect folds (overfitting signal)
    if fold_accs and any(a >= 0.99 for a in fold_accs):
        diag["issues"].append("Perfect fold detected (100%) — likely overfitting")
        diag["actions"].append({
            "action": "increase_regularization",
            "reason": "Perfect fold is a strong overfitting indicator",
        })

    # 4. Collapsed folds (regime failure)
    if fold_accs and any(a <= 0.10 for a in fold_accs):
        diag["issues"].append("Collapsed fold (<10%) — model failed in one regime")
        next_h = pick_horizon()
        if next_h and next_h != horizon:
            diag["actions"].append({
                "action": "try_shorter_horizon",
                "param": next_h,
                "reason": f"Trying horizon={next_h}d — collapse suggests current horizon too long "
                          f"(tried: {sorted(tried_horizons) or 'none'})",
            })

    # 5. Poor holdout despite decent CV
    if ho_reg is not None and reg_acc is not None:
        if ho_reg < 0.45 and reg_acc > 0.55:
            diag["issues"].append(f"Holdout ({ho_reg:.1%}) much worse than CV ({reg_acc:.1%})")
            diag["actions"].append({
                "action": "increase_regularization",
                "reason": "CV/holdout gap suggests overfitting",
            })

    # 6. Negative Spearman (predictions inversely correlated with reality)
    if reg_spearman is not None and reg_spearman < -0.1:
        diag["issues"].append(f"Negative Spearman correlation ({reg_spearman:.3f})")
        diag["status"] = "poor"
        diag["actions"].append({
            "action": "try_shorter_horizon",
            "param": 21,
            "reason": "Negative Spearman means predictions are inverted — fundamental approach problem",
        })

    # 7. Too many features
    if n_features > 30:
        diag["issues"].append(f"High feature count ({n_features}) — overfitting risk")
        diag["actions"].append({
            "action": "stricter_feature_selection",
            "reason": f"{n_features} features is high for the data size",
        })

    # 8. Too few features
    if n_features < 5 and reg_acc is not None and reg_acc < 0.65:
        diag["issues"].append(f"Very few features ({n_features}) with mediocre accuracy")
        diag["actions"].append({
            "action": "relax_feature_selection",
            "reason": f"Only {n_features} features — may be discarding useful signal",
        })

    # ── Consume signals from other agents ──

    active_signals = get_active_signals(commodity=cfg.name)
    for sig in active_signals:
        sig_type = sig.get("type")
        sig_detail = sig.get("detail", "")

        if sig_type == "data_anomaly" and sig.get("severity") in ("high", "critical"):
            diag["issues"].append(f"Data anomaly (from {sig.get('source')}): {sig_detail[:80]}")
            diag["status"] = "poor"

        elif sig_type == "feature_drift" and sig.get("severity") in ("high", "critical"):
            diag["issues"].append(f"Feature drift detected: {sig_detail[:80]}")
            diag["actions"].append({
                "action": "increase_regularization",
                "reason": f"Feature drift: {sig_detail[:60]}",
            })

        elif sig_type == "baseline_beaten":
            diag["issues"].append(f"Model underperforms baselines: {sig_detail[:80]}")
            diag["status"] = "poor"
            diag["actions"].append({
                "action": "try_shorter_horizon",
                "param": 21,
                "reason": f"Can't beat simple baselines — fundamental approach may be wrong",
            })

        elif sig_type == "calibration_off":
            diag["issues"].append(f"Calibration issue: {sig_detail[:80]}")
            # Calibration doesn't change training, but we log it

    if active_signals:
        diag["consumed_signals"] = len(active_signals)

    # Deduplicate actions
    seen = set()
    unique_actions = []
    for a in diag["actions"]:
        key = a["action"]
        if key not in seen:
            seen.add(key)
            unique_actions.append(a)
    diag["actions"] = unique_actions

    return diag


# ── Per-commodity config generation ──────────────────────────────────────

def generate_training_config(cfg: CommodityConfig, diagnosis: dict) -> dict:
    """Generate a per-commodity training config based on diagnosis.

    This config is written to configs/<commodity>.json and read by train.py.
    """
    actions = {a["action"] for a in diagnosis.get("actions", [])}

    # Start with defaults
    config = {
        "commodity": cfg.name,
        "generated_by": "model_quality_agent",
        "generated_at": datetime.now().isoformat(),
        "horizon": 63,
        "optuna_trials": 200,
        "test_size": 252,
        "n_splits": 5,
        "min_gamma": 0.0,
        "min_reg_alpha": 1e-8,
        "min_reg_lambda": 1e-8,
        "max_depth_range": [2, 10],
        "feature_selection_min": 3,
        "diagnosis_status": diagnosis.get("status", "unknown"),
        "diagnosis_issues": diagnosis.get("issues", []),
    }

    # Apply remediation actions
    if "try_shorter_horizon" in actions:
        param = next((a["param"] for a in diagnosis["actions"]
                      if a["action"] == "try_shorter_horizon"), 21)
        config["horizon"] = param
        config["test_size"] = 126  # shorter horizon → can use shorter test folds
        config["_reason_horizon"] = f"Shortened from 63 to {param} due to regime instability"

    if "increase_regularization" in actions:
        config["min_gamma"] = 2.0
        config["min_reg_alpha"] = 0.01
        config["min_reg_lambda"] = 0.1
        config["max_depth_range"] = [2, 6]
        config["_reason_regularization"] = "Increased regularization floor to combat overfitting"

    if "stricter_feature_selection" in actions:
        config["feature_selection_min"] = 5
        config["optuna_trials"] = 250  # more trials to compensate for fewer features
        config["_reason_features"] = "Stricter selection to reduce overfitting from excess features"

    if "relax_feature_selection" in actions:
        config["feature_selection_min"] = 3  # Keep the fallback low
        config["_reason_features"] = "Relaxed selection — too few features were being kept"

    return config


def write_training_config(key: str, config: dict):
    """Write per-commodity training config to disk."""
    COMMODITY_CONFIGS_DIR.mkdir(exist_ok=True)
    path = COMMODITY_CONFIGS_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


def load_training_config(key: str) -> dict | None:
    """Load per-commodity training config if it exists."""
    path = COMMODITY_CONFIGS_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Remediation (retrain with per-commodity config) ──────────────────────

def retrain_with_config(cfg: CommodityConfig, config: dict) -> dict:
    """Retrain a commodity using its per-commodity config.

    Passes config as environment variables to the train.py subprocess.
    """
    report = {
        "commodity": cfg.name,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    train_script = cfg.project_dir / "train.py"
    if not train_script.exists():
        report["status"] = "error"
        report["error"] = "train.py not found"
        return report

    # Pass config via environment variables that train.py can read
    import os
    env = os.environ.copy()
    env["QUALITY_HORIZON"] = str(config.get("horizon", 63))
    env["QUALITY_OPTUNA_TRIALS"] = str(config.get("optuna_trials", 200))
    env["QUALITY_TEST_SIZE"] = str(config.get("test_size", 252))
    env["QUALITY_N_SPLITS"] = str(config.get("n_splits", 5))
    env["QUALITY_MIN_GAMMA"] = str(config.get("min_gamma", 0.0))
    env["QUALITY_MIN_REG_ALPHA"] = str(config.get("min_reg_alpha", 1e-8))
    env["QUALITY_MIN_REG_LAMBDA"] = str(config.get("min_reg_lambda", 1e-8))
    env["QUALITY_MAX_DEPTH_MIN"] = str(config.get("max_depth_range", [2, 10])[0])
    env["QUALITY_MAX_DEPTH_MAX"] = str(config.get("max_depth_range", [2, 10])[1])

    logger.info(f"Retraining {cfg.name} with quality config: "
                f"horizon={config['horizon']}, "
                f"min_gamma={config.get('min_gamma', 0)}, "
                f"max_depth={config.get('max_depth_range', [2,10])}")

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            env=env,
            timeout=2400,  # 40 min max
        )

        if result.returncode != 0:
            report["status"] = "train_failed"
            report["stderr"] = result.stderr[-500:]
            logger.error(f"{cfg.name} quality retrain failed: {result.stderr[-200:]}")
            return report

        report["stdout_tail"] = result.stdout[-500:]
        report["status"] = "ok"

    except subprocess.TimeoutExpired:
        report["status"] = "timeout"
        logger.error(f"{cfg.name} quality retrain timed out")
        return report

    # Load new metrics
    new_meta = load_metrics(cfg)
    if new_meta:
        reg = new_meta.get("regression", {})
        ho = new_meta.get("holdout", {})
        report["new_metrics"] = {
            "reg_acc": reg.get("avg_accuracy"),
            "reg_spearman": reg.get("avg_spearman"),
            "holdout_reg": ho.get("reg_direction_accuracy"),
            "holdout_spearman": ho.get("reg_spearman"),
            "n_features": new_meta.get("n_features"),
            "horizon": new_meta.get("horizon"),
        }

    return report


# ── Logging ──────────────────────────────────────────────────────────────

def log_quality_report(report: dict):
    """Append quality report to JSONL log."""
    QUALITY_LOG.parent.mkdir(exist_ok=True)
    with open(QUALITY_LOG, "a") as f:
        f.write(json.dumps(report, default=str) + "\n")


# ── Main pipeline ────────────────────────────────────────────────────────

def run_quality_agent(commodity_keys: list[str] = None, diagnose_only: bool = False) -> dict:
    """Run the full quality diagnosis and remediation pipeline."""
    targets = commodity_keys or list(COMMODITIES.keys())
    results = {}

    logger.info("=" * 60)
    logger.info("MODEL QUALITY AGENT")
    logger.info("=" * 60)

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"Diagnosing {cfg.name}...")

        # Step 1: Diagnose
        diag = diagnose_commodity(cfg)
        results[key] = {"diagnosis": diag}

        if diag["issues"]:
            logger.warning(f"  {cfg.name} status: {diag['status']}")
            for issue in diag["issues"]:
                logger.warning(f"    - {issue}")
            for action in diag["actions"]:
                logger.info(f"    Action: {action['action']} — {action['reason']}")
        else:
            logger.info(f"  {cfg.name}: OK — no issues detected")

        # Log diagnosis to design log
        if diag["issues"]:
            log_observation("model_quality",
                f"Diagnosed {len(diag['issues'])} issues: {'; '.join(diag['issues'])}",
                cfg.name)
        else:
            log_observation("model_quality",
                f"Model health OK — no issues detected", cfg.name)

        if diagnose_only:
            continue

        # Step 2: Generate per-commodity config
        if diag["actions"]:
            config = generate_training_config(cfg, diag)
            config_path = write_training_config(key, config)
            results[key]["config"] = config
            logger.info(f"  Config written to {config_path}")

            # Log config decisions
            for action in diag["actions"]:
                log_observation("model_quality",
                    f"Applying {action['action']}: {action['reason']}", cfg.name)

            # Step 3: Retrain with new config
            logger.info(f"  Retraining {cfg.name}...")
            report = retrain_with_config(cfg, config)
            results[key]["retrain"] = report
            log_quality_report({"key": key, "diagnosis": diag, "config": config, "retrain": report})

            if report["status"] == "ok":
                # Resolve any retraining_needed signals
                resolve_signal("model_quality", "retraining_needed", cfg.name,
                    resolution="Retrained with quality config")
                resolve_signal("model_quality", "model_degraded", cfg.name,
                    resolution="Retrained with quality config")

                new = report.get("new_metrics", {})
                old_acc = diag["metrics"].get("reg_acc")
                new_acc = new.get("reg_acc")

                if old_acc is not None and new_acc is not None:
                    delta = new_acc - old_acc
                    improved = "improved" if delta > 0 else "declined"
                    logger.info(f"  {cfg.name}: {improved} ({old_acc:.1%} → {new_acc:.1%}, delta={delta:+.1%})")

                    log_observation("model_quality",
                        f"Retrained: reg_acc {old_acc:.1%} → {new_acc:.1%} "
                        f"(horizon={config['horizon']}, "
                        f"spearman={new.get('reg_spearman', '?')})",
                        cfg.name)

                    if delta < -0.05:
                        log_challenge("model_quality",
                            f"{cfg.name} remediation strategy",
                            f"Accuracy declined by {delta:.1%} after remediation. "
                            f"Actions taken: {[a['action'] for a in diag['actions']]}. "
                            f"May need different approach.", cfg.name)
                else:
                    logger.info(f"  {cfg.name}: retrained (no comparison available)")
            else:
                logger.error(f"  {cfg.name}: retrain failed — {report['status']}")
                log_observation("model_quality",
                    f"Retrain failed: {report['status']}", cfg.name)
        else:
            logger.info(f"  {cfg.name}: no action needed")
            # Still write a config marking it as healthy
            config = generate_training_config(cfg, diag)
            write_training_config(key, config)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model quality diagnosis and remediation")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--diagnose-only", action="store_true", help="Diagnose without retraining")
    args = parser.parse_args()

    results = run_quality_agent(args.commodities or None, diagnose_only=args.diagnose_only)

    print(f"\n{'='*60}")
    print(f"MODEL QUALITY SUMMARY")
    print(f"{'='*60}")

    for key, result in results.items():
        cfg = COMMODITIES[key]
        diag = result["diagnosis"]
        status = diag["status"]
        n_issues = len(diag["issues"])
        n_actions = len(diag["actions"])

        retrain = result.get("retrain", {})
        new_metrics = retrain.get("new_metrics", {})

        old_acc = diag["metrics"].get("reg_acc")
        new_acc = new_metrics.get("reg_acc")
        old_str = f"{old_acc:.1%}" if old_acc is not None else "?"
        new_str = f"{new_acc:.1%}" if new_acc is not None else "—"

        config = result.get("config", {})
        horizon = config.get("horizon", "—")

        print(f"  {cfg.name:<15} [{status:<8}] "
              f"{n_issues} issues, {n_actions} actions  "
              f"acc: {old_str} → {new_str}  "
              f"horizon: {horizon}")

        if diag["actions"]:
            for action in diag["actions"]:
                print(f"    └─ {action['action']}: {action['reason'][:60]}")


if __name__ == "__main__":
    main()
