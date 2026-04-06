"""Conditional Confidence Calibration — meta-model that predicts WHEN primary models are reliable.

Builds a "meta-model" per commodity that learns to predict P(prediction_correct | market_conditions).
This replaces raw model confidence with a calibrated confidence that accounts for regime,
volatility, drawdown, and other market state variables.

High calibrated confidence => trade aggressively.
Low calibrated confidence => sit out.

Usage:
    python -m agents confidence              # train all commodities
    python -m agents confidence natgas wheat  # specific commodities
"""

import argparse
import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR, LOGS_DIR
from .design_log import log_observation
from .log import setup_logging, log_event
from .train_utils import walk_forward_split

logger = setup_logging("confidence_model")

# Commodities with ensemble models available
ENSEMBLE_COMMODITIES = ["coffee", "cocoa", "sugar", "natgas", "soybeans", "wheat", "copper"]

# Meta-model feature columns (derived from market state + primary model output)
META_FEATURES = [
    "vol_regime",
    "trend_regime",
    "drawdown",
    "mean_reversion_pressure",
    "regime_uncertainty",
    "model_confidence",
    "model_agreement",
    "recent_accuracy_10",
]

MIN_PREDICTIONS = 50  # Minimum data points needed to train meta-model


# ── Step 1: Build meta-model training data ──────────────────────────────

def _load_commodity_data(cfg: CommodityConfig) -> dict | None:
    """Load data and ensemble metadata for a commodity via subprocess.

    Runs in the commodity's project directory so `from features import prepare_dataset`
    works correctly. Returns the data needed to build meta-model training examples.
    """
    meta_path = cfg.models_dir / "ensemble_metadata.json"
    if not meta_path.exists():
        logger.warning(f"{cfg.name}: No ensemble_metadata.json found, skipping")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    best_horizon = meta.get("horizon", cfg.horizon)
    features = meta.get("features", [])

    if not features:
        logger.warning(f"{cfg.name}: No features in ensemble metadata, skipping")
        return None

    # Use subprocess to load data from commodity directory (same pattern as research agent)
    script = f"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from features import prepare_dataset

df, all_cols = prepare_dataset(horizon={best_horizon})

# Output shape info and the data as JSON
result = {{
    "n_rows": len(df),
    "columns": list(df.columns),
    "index": [str(d) for d in df.index],
}}
print(json.dumps(result))

# Save the dataframe for the parent process
df.to_csv('/tmp/_confidence_meta_{cfg.dir_name}.csv')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir), timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"{cfg.name}: Failed to load data: {result.stderr[-500:]}")
            return None

        info = json.loads(result.stdout.strip().split("\n")[-1])
        df = pd.read_csv(f"/tmp/_confidence_meta_{cfg.dir_name}.csv",
                         index_col=0, parse_dates=True)

        return {
            "df": df,
            "meta": meta,
            "horizon": best_horizon,
            "features": features,
            "info": info,
        }
    except Exception as e:
        logger.error(f"{cfg.name}: Exception loading data: {e}")
        return None


def _build_meta_training_data(
    df: pd.DataFrame,
    meta: dict,
    cfg: CommodityConfig,
    horizon: int,
    features: list[str],
) -> pd.DataFrame | None:
    """Walk through data in walk-forward fashion, recording when predictions are correct.

    For each test fold:
    - Train primary models on training data
    - Make predictions on test data
    - Record market conditions + whether prediction was correct

    This creates the (market_conditions -> was_prediction_correct) dataset.
    """
    from xgboost import XGBClassifier, XGBRegressor

    price_col = cfg.price_col

    # Verify required columns exist
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 3:
        logger.warning(f"{cfg.name}: Only {len(available_features)} features available, need >= 3")
        return None

    # Regime feature columns (these become meta-model inputs)
    regime_cols = ["vol_regime", "trend_regime", "drawdown",
                   "mean_reversion_pressure", "regime_uncertainty"]
    missing_regime = [c for c in regime_cols if c not in df.columns]
    if missing_regime:
        logger.warning(f"{cfg.name}: Missing regime columns {missing_regime}, will fill with 0")

    # Build targets
    target_return = np.log(
        df[price_col].shift(-horizon) / df[price_col]
    )
    target_direction = (target_return > 0).astype(int)

    df_work = df.copy()
    df_work["target_return"] = target_return
    df_work["target_direction"] = target_direction
    df_work = df_work.dropna(subset=["target_return"])

    # Walk-forward splits -- use same logic as ensemble training
    test_size = max(horizon * 10, 126)
    splits = walk_forward_split(df_work, n_splits=5, test_size=test_size, purge_gap=horizon)

    if len(splits) < 2:
        logger.warning(f"{cfg.name}: Not enough data for walk-forward splits")
        return None

    X = df_work[available_features].values
    y_ret = df_work["target_return"].values
    y_dir = df_work["target_direction"].values

    meta_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        val_size = max(5, min(horizon * 2, len(train_idx) // 5))
        fit_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]

        # Train primary regressor on this fold's training data
        try:
            reg = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.6,
                min_child_weight=10, gamma=1.0,
                early_stopping_rounds=20, random_state=42,
            )
            reg.fit(X[fit_idx], y_ret[fit_idx],
                    eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
        except Exception as e:
            logger.warning(f"{cfg.name}: Regressor training failed fold {fold_idx}: {e}")
            continue

        # Train primary classifier on this fold's training data
        try:
            neg = np.sum(y_dir[fit_idx] == 0)
            pos = np.sum(y_dir[fit_idx] == 1)
            spw = neg / pos if pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.6,
                min_child_weight=10, gamma=1.0, scale_pos_weight=spw,
                eval_metric="logloss", early_stopping_rounds=20, random_state=42,
            )
            clf.fit(X[fit_idx], y_dir[fit_idx],
                    eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
        except Exception as e:
            logger.warning(f"{cfg.name}: Classifier training failed fold {fold_idx}: {e}")
            continue

        # Make predictions on test fold
        reg_preds = reg.predict(X[test_idx])
        clf_preds = clf.predict(X[test_idx])
        clf_proba = clf.predict_proba(X[test_idx])

        # Determine correctness: did the direction prediction match reality?
        pred_direction = (reg_preds > 0).astype(int)
        actual_direction = y_dir[test_idx]
        correct = (pred_direction == actual_direction).astype(int)

        # Model confidence: probability assigned to predicted class
        model_confidence = np.array([
            clf_proba[i, int(clf_preds[i])] for i in range(len(clf_preds))
        ])

        # Model agreement: do regressor and classifier agree on direction?
        reg_direction = (reg_preds > 0).astype(int)
        model_agreement = (reg_direction == clf_preds).astype(float)

        # Collect regime features for each test point
        for i, t_idx in enumerate(test_idx):
            row = {}

            # Regime features
            for col in regime_cols:
                if col in df_work.columns:
                    val = df_work.iloc[t_idx][col]
                    row[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    row[col] = 0.0

            # Model output features
            row["model_confidence"] = float(model_confidence[i])
            row["model_agreement"] = float(model_agreement[i])

            # Rolling accuracy of recent predictions (within this fold)
            if i >= 10:
                row["recent_accuracy_10"] = float(np.mean(correct[max(0, i - 10):i]))
            else:
                # Use fold-level accuracy for early predictions
                row["recent_accuracy_10"] = 0.5

            # Target: was this prediction correct?
            row["correct"] = int(correct[i])
            row["fold"] = fold_idx
            row["date"] = str(df_work.index[t_idx])

            meta_rows.append(row)

    if not meta_rows:
        return None

    meta_df = pd.DataFrame(meta_rows)
    return meta_df


# ── Step 2: Train meta-classifier ────────────────────────────────────────

def _train_meta_model(
    meta_df: pd.DataFrame,
    cfg: CommodityConfig,
) -> dict | None:
    """Train a LogisticRegression meta-model predicting P(correct | conditions).

    Uses very heavy regularization (C=0.01) for generalization.
    Returns dict with model, metrics, and feature insights.
    """
    feature_cols = [c for c in META_FEATURES if c in meta_df.columns]

    if len(meta_df) < MIN_PREDICTIONS:
        logger.warning(
            f"{cfg.name}: Only {len(meta_df)} meta-training points, "
            f"need {MIN_PREDICTIONS}. Skipping."
        )
        return None

    X = meta_df[feature_cols].values
    y = meta_df["correct"].values

    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Check class balance
    pos_rate = np.mean(y)
    logger.info(f"{cfg.name}: Base accuracy rate = {pos_rate:.1%} "
                f"({len(meta_df)} predictions)")

    # Use last fold as holdout for evaluation
    folds = sorted(meta_df["fold"].unique())
    if len(folds) < 2:
        logger.warning(f"{cfg.name}: Only 1 fold, cannot evaluate meta-model")
        return None

    last_fold = folds[-1]
    train_mask = meta_df["fold"] != last_fold
    test_mask = meta_df["fold"] == last_fold

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) < 30 or len(X_test) < 10:
        logger.warning(f"{cfg.name}: Not enough data in train/test split for meta-model")
        return None

    # Train with heavy regularization
    meta_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=0.01,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    meta_pipeline.fit(X_train, y_train)

    # Evaluate
    train_proba = meta_pipeline.predict_proba(X_train)[:, 1]
    test_proba = meta_pipeline.predict_proba(X_test)[:, 1]

    train_brier = brier_score_loss(y_train, train_proba)
    test_brier = brier_score_loss(y_test, test_proba)

    # Calibration check: in high-confidence windows, is accuracy actually higher?
    high_conf_mask = test_proba >= 0.6
    low_conf_mask = test_proba < 0.4

    high_conf_acc = float(np.mean(y_test[high_conf_mask])) if high_conf_mask.sum() > 5 else None
    low_conf_acc = float(np.mean(y_test[low_conf_mask])) if low_conf_mask.sum() > 5 else None
    overall_acc = float(np.mean(y_test))

    # Feature coefficients (after scaling)
    coefs = meta_pipeline.named_steps["logistic"].coef_[0]
    feature_importance = sorted(
        zip(feature_cols, coefs.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Retrain on ALL data for the final model
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=0.01,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    final_pipeline.fit(X, y)

    metrics = {
        "n_predictions": len(meta_df),
        "base_accuracy": round(pos_rate, 4),
        "train_brier": round(train_brier, 4),
        "test_brier": round(test_brier, 4),
        "overall_test_acc": round(overall_acc, 4),
        "high_conf_acc": round(high_conf_acc, 4) if high_conf_acc is not None else None,
        "low_conf_acc": round(low_conf_acc, 4) if low_conf_acc is not None else None,
        "high_conf_n": int(high_conf_mask.sum()),
        "low_conf_n": int(low_conf_mask.sum()),
        "feature_importance": feature_importance,
    }

    return {
        "model": final_pipeline,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }


# ── Step 3: Calibrated confidence function ──────────────────────────────

def get_calibrated_confidence(
    df: pd.DataFrame,
    raw_prediction: float,
    model_confidence: float,
    agreement: float,
    meta_model_path: Path | None = None,
    meta_model: Pipeline | None = None,
) -> dict:
    """Return calibrated confidence using the meta-model.

    Looks at current market conditions (from the last row of df) and adjusts
    the raw model confidence based on whether these conditions historically
    lead to accurate predictions.

    Args:
        df: DataFrame with regime features (uses last row for current conditions).
        raw_prediction: The primary model's predicted return.
        model_confidence: The primary model's raw confidence (0-1).
        agreement: Agreement across ensemble models (0-1).
        meta_model_path: Path to the saved meta-model .joblib file.
        meta_model: Pre-loaded meta-model pipeline (alternative to path).

    Returns:
        Dict with calibrated_confidence, raw_confidence, conditions, etc.
    """
    # Load meta-model if needed
    if meta_model is None and meta_model_path is not None:
        if meta_model_path.exists():
            meta_model = joblib.load(meta_model_path)
        else:
            # No meta-model available, return raw confidence
            return {
                "calibrated_confidence": model_confidence,
                "raw_confidence": model_confidence,
                "meta_model_available": False,
                "conditions": {},
            }

    if meta_model is None:
        return {
            "calibrated_confidence": model_confidence,
            "raw_confidence": model_confidence,
            "meta_model_available": False,
            "conditions": {},
        }

    # Extract current market conditions from last row
    latest = df.iloc[-1]
    conditions = {}
    regime_cols = ["vol_regime", "trend_regime", "drawdown",
                   "mean_reversion_pressure", "regime_uncertainty"]

    for col in regime_cols:
        if col in df.columns:
            val = latest[col]
            conditions[col] = float(val) if not pd.isna(val) else 0.0
        else:
            conditions[col] = 0.0

    conditions["model_confidence"] = float(model_confidence)
    conditions["model_agreement"] = float(agreement)
    conditions["recent_accuracy_10"] = 0.5  # Default; caller can override

    # Build feature vector in correct order
    feature_cols = META_FEATURES
    X = np.array([[conditions.get(f, 0.0) for f in feature_cols]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Get calibrated probability
    calibrated = float(meta_model.predict_proba(X)[0, 1])

    return {
        "calibrated_confidence": round(calibrated, 4),
        "raw_confidence": round(model_confidence, 4),
        "meta_model_available": True,
        "adjustment": round(calibrated - model_confidence, 4),
        "conditions": {k: round(v, 4) for k, v in conditions.items()},
    }


# ── Step 4: Train and save ──────────────────────────────────────────────

def train_confidence_model(key: str) -> dict:
    """Build the meta-model for a single commodity.

    Steps:
    1. Load historical data and ensemble metadata
    2. Walk-forward: make predictions, record correctness + conditions
    3. Train LogisticRegression meta-classifier on (conditions -> correct)
    4. Save to {commodity}/models/confidence_meta_model.joblib

    Returns dict with status and metrics.
    """
    if key not in COMMODITIES:
        return {"status": "error", "message": f"Unknown commodity: {key}"}

    cfg = COMMODITIES[key]
    logger.info(f"{'=' * 60}")
    logger.info(f"Training confidence meta-model for {cfg.name}")
    logger.info(f"{'=' * 60}")

    # Step 1: Load data
    data = _load_commodity_data(cfg)
    if data is None:
        return {"status": "skipped", "message": "Could not load data"}

    df = data["df"]
    meta = data["meta"]
    horizon = data["horizon"]
    features = data["features"]

    logger.info(f"  Loaded {len(df)} rows, horizon={horizon}, {len(features)} features")

    # Step 2: Build meta-training data
    meta_df = _build_meta_training_data(df, meta, cfg, horizon, features)
    if meta_df is None or len(meta_df) < MIN_PREDICTIONS:
        n = len(meta_df) if meta_df is not None else 0
        msg = f"Insufficient meta-training data: {n} < {MIN_PREDICTIONS}"
        logger.warning(f"  {msg}")
        return {"status": "skipped", "message": msg}

    logger.info(f"  Built {len(meta_df)} meta-training examples across "
                f"{meta_df['fold'].nunique()} folds")
    logger.info(f"  Base accuracy: {meta_df['correct'].mean():.1%}")

    # Step 3: Train meta-model
    result = _train_meta_model(meta_df, cfg)
    if result is None:
        return {"status": "skipped", "message": "Meta-model training failed"}

    metrics = result["metrics"]
    model = result["model"]
    feature_cols = result["feature_cols"]

    # Step 4: Save
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.models_dir / "confidence_meta_model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    meta_info = {
        "commodity": cfg.name,
        "key": key,
        "trained_at": datetime.now().isoformat(),
        "horizon": horizon,
        "feature_cols": feature_cols,
        "n_training_predictions": metrics["n_predictions"],
        "base_accuracy": metrics["base_accuracy"],
        "train_brier": metrics["train_brier"],
        "test_brier": metrics["test_brier"],
        "high_conf_acc": metrics["high_conf_acc"],
        "low_conf_acc": metrics["low_conf_acc"],
        "high_conf_n": metrics["high_conf_n"],
        "low_conf_n": metrics["low_conf_n"],
        "feature_importance": metrics["feature_importance"],
    }

    meta_path = cfg.models_dir / "confidence_meta_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    # Report
    logger.info(f"  Meta-model saved to {model_path}")
    logger.info(f"  Train Brier: {metrics['train_brier']:.4f}")
    logger.info(f"  Test Brier:  {metrics['test_brier']:.4f}")

    if metrics["high_conf_acc"] is not None:
        logger.info(
            f"  High-confidence windows (n={metrics['high_conf_n']}): "
            f"accuracy={metrics['high_conf_acc']:.1%}"
        )
    if metrics["low_conf_acc"] is not None:
        logger.info(
            f"  Low-confidence windows (n={metrics['low_conf_n']}): "
            f"accuracy={metrics['low_conf_acc']:.1%}"
        )

    # Feature importance insights
    logger.info("  Feature importance (absolute coefficient):")
    for feat, coef in metrics["feature_importance"]:
        direction = "+" if coef > 0 else "-"
        logger.info(f"    {direction} {feat}: {abs(coef):.4f}")

    # Log observations to design log
    _log_insights(key, cfg, metrics)

    return {
        "status": "ok",
        "commodity": cfg.name,
        "model_path": str(model_path),
        "metrics": {k: v for k, v in metrics.items() if k != "feature_importance"},
    }


def _log_insights(key: str, cfg: CommodityConfig, metrics: dict):
    """Log interesting findings about what predicts model accuracy."""
    observations = []

    # High vs low confidence accuracy spread
    if metrics["high_conf_acc"] is not None and metrics["low_conf_acc"] is not None:
        spread = metrics["high_conf_acc"] - metrics["low_conf_acc"]
        if spread > 0.1:
            observations.append(
                f"Confidence calibration shows {spread:.0%} accuracy spread between "
                f"high-confidence ({metrics['high_conf_acc']:.0%}) and low-confidence "
                f"({metrics['low_conf_acc']:.0%}) windows — meta-model is finding "
                f"exploitable patterns in when predictions work"
            )
        elif spread < -0.05:
            observations.append(
                f"Counterintuitive: low-confidence windows ({metrics['low_conf_acc']:.0%}) "
                f"outperform high-confidence ({metrics['high_conf_acc']:.0%}) — "
                f"possible overfitting in primary model's confidence"
            )

    # Top predictors of accuracy
    if metrics["feature_importance"]:
        top_feat, top_coef = metrics["feature_importance"][0]
        direction = "increases" if top_coef > 0 else "decreases"
        observations.append(
            f"Top predictor of model accuracy: {top_feat} "
            f"({direction} P(correct), coef={top_coef:.4f})"
        )

    # Brier score comparison
    if metrics["test_brier"] < 0.24:
        observations.append(
            f"Meta-model Brier score {metrics['test_brier']:.4f} beats naive "
            f"baseline (0.25) — calibration is adding value"
        )
    else:
        observations.append(
            f"Meta-model Brier score {metrics['test_brier']:.4f} is near/above "
            f"naive baseline — limited ability to predict when model is correct"
        )

    for obs in observations:
        log_observation("confidence_model", obs, commodity=key)
        logger.info(f"  [insight] {obs}")


def train_all_confidence_models() -> dict:
    """Train confidence meta-models for all commodities with ensemble models.

    Returns summary dict with results per commodity.
    """
    logger.info("=" * 60)
    logger.info("Training confidence meta-models for all commodities")
    logger.info("=" * 60)

    results = {}

    for key in ENSEMBLE_COMMODITIES:
        if key not in COMMODITIES:
            continue
        try:
            result = train_confidence_model(key)
            results[key] = result
            status = result["status"]
            logger.info(f"  {COMMODITIES[key].name}: {status}")
        except Exception as e:
            logger.error(f"  {COMMODITIES[key].name}: FAILED ({e})")
            results[key] = {"status": "error", "message": str(e)}

    # Summary
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    errors = sum(1 for r in results.values() if r["status"] == "error")

    logger.info(f"\nSummary: {ok} trained, {skipped} skipped, {errors} errors")

    return results


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train conditional confidence calibration meta-models"
    )
    parser.add_argument(
        "commodities", nargs="*",
        help="Specific commodities to train (default: all with ensemble models)"
    )
    args = parser.parse_args()

    if args.commodities:
        results = {}
        for key in args.commodities:
            if key not in COMMODITIES:
                logger.error(f"Unknown commodity: {key}")
                continue
            results[key] = train_confidence_model(key)
    else:
        results = train_all_confidence_models()

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Commodity':<15} {'Status':<10} {'Base Acc':>10} {'Brier':>10} "
          f"{'Hi-Conf':>10} {'Lo-Conf':>10}")
    print("-" * 70)

    for key, result in results.items():
        name = COMMODITIES[key].name if key in COMMODITIES else key
        status = result["status"]
        if status == "ok":
            m = result.get("metrics", {})
            base = f"{m.get('base_accuracy', 0):.1%}"
            brier = f"{m.get('test_brier', 0):.4f}"
            hi = f"{m['high_conf_acc']:.1%}" if m.get("high_conf_acc") is not None else "n/a"
            lo = f"{m['low_conf_acc']:.1%}" if m.get("low_conf_acc") is not None else "n/a"
        else:
            base = brier = hi = lo = "-"

        print(f"{name:<15} {status:<10} {base:>10} {brier:>10} {hi:>10} {lo:>10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
