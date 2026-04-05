"""Train and evaluate coffee price prediction models."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import joblib

from features import prepare_dataset


MODELS_DIR = Path(__file__).parent / "models"


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 63,  # ~3 months of trading days
    min_train_size: int = 504,  # ~2 years
    purge_gap: int = 5,  # gap between train and test to prevent label leakage
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward (expanding window) train/test indices.

    Args:
        df: Full dataset (must be time-sorted).
        n_splits: Number of CV folds.
        test_size: Number of rows in each test fold.
        min_train_size: Minimum rows in the training set.
        purge_gap: Number of rows to skip between train and test sets.
            Should equal the prediction horizon to prevent label leakage.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    n = len(df)
    splits = []
    for i in range(n_splits):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start - purge_gap  # purge gap prevents label leakage
        if train_end < min_train_size:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    splits.reverse()  # chronological order
    return splits


def train_regression_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_return",
) -> dict:
    """Train an XGBoost regressor with walk-forward CV.

    Returns:
        Dict with model, metrics, and feature importances.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    splits = walk_forward_split(df, purge_gap=5)

    fold_metrics = []
    models = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            random_state=42,
        )
        val_size = min(63, len(X_train) // 5)
        X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
        y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        direction_acc = np.mean((preds > 0) == (y_test > 0))

        fold_metrics.append({"fold": fold_i, "rmse": rmse, "mae": mae, "direction_acc": direction_acc})
        models.append(model)
        print(f"  Fold {fold_i}: RMSE={rmse:.6f}, MAE={mae:.6f}, Direction Acc={direction_acc:.2%}")

    # Select the model from the fold with best direction accuracy
    best_fold_idx = int(np.argmax([m["direction_acc"] for m in fold_metrics]))
    best_model = models[best_fold_idx]
    print(f"  Selected model from fold {best_fold_idx} (best direction_acc={fold_metrics[best_fold_idx]['direction_acc']:.2%})")

    # Feature importance
    importances = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    avg_metrics = {
        "avg_rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "avg_mae": np.mean([m["mae"] for m in fold_metrics]),
        "avg_direction_acc": np.mean([m["direction_acc"] for m in fold_metrics]),
    }
    print(f"\n  Average: RMSE={avg_metrics['avg_rmse']:.6f}, "
          f"Direction Acc={avg_metrics['avg_direction_acc']:.2%}")

    return {
        "model": best_model,
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "feature_importances": importances,
    }


def train_classification_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction",
) -> dict:
    """Train an XGBoost classifier for direction prediction with walk-forward CV.

    Returns:
        Dict with model, metrics, and feature importances.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    splits = walk_forward_split(df, purge_gap=5)

    fold_metrics = []
    models = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        val_size = min(63, len(X_train) // 5)
        X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
        y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        fold_metrics.append({"fold": fold_i, "accuracy": acc})
        models.append(model)
        print(f"  Fold {fold_i}: Accuracy={acc:.2%}")

    best_fold_idx = int(np.argmax([m["accuracy"] for m in fold_metrics]))
    best_model = models[best_fold_idx]
    print(f"  Selected model from fold {best_fold_idx} (best accuracy={fold_metrics[best_fold_idx]['accuracy']:.2%})")
    importances = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    print(f"\n  Average Accuracy: {avg_acc:.2%}")

    return {
        "model": best_model,
        "fold_metrics": fold_metrics,
        "avg_accuracy": avg_acc,
        "feature_importances": importances,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Preparing dataset...")
    df, feature_cols = prepare_dataset(horizon=5)
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features\n")

    # Train regression model (predict return magnitude)
    print("Training regression model (predict 5-day return)...")
    reg_results = train_regression_model(df, feature_cols)

    # Train classification model (predict direction)
    print("\nTraining classification model (predict 5-day direction)...")
    clf_results = train_classification_model(df, feature_cols)

    # Save models
    joblib.dump(reg_results["model"], MODELS_DIR / "xgb_regressor.joblib")
    joblib.dump(clf_results["model"], MODELS_DIR / "xgb_classifier.joblib")

    # Save feature list and metrics
    metadata = {
        "feature_cols": feature_cols,
        "horizon": 5,
        "regression_metrics": reg_results["avg_metrics"],
        "classification_accuracy": clf_results["avg_accuracy"],
        "top_features_regression": dict(list(reg_results["feature_importances"].items())[:15]),
        "top_features_classification": dict(list(clf_results["feature_importances"].items())[:15]),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModels saved to {MODELS_DIR}/")
    print(f"Top 10 features (regression):")
    for feat, imp in list(reg_results["feature_importances"].items())[:10]:
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
