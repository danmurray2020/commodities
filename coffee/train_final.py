"""Train the final production model: 63-day horizon with Optuna-tuned hyperparameters."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import joblib

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63

# Best params from Optuna (tuned on classification)
TUNED_PARAMS = {
    "n_estimators": 477,
    "max_depth": 8,
    "learning_rate": 0.298,
    "subsample": 0.81,
    "colsample_bytree": 0.533,
    "min_child_weight": 16,
    "gamma": 0.728,
    "reg_alpha": 0.074,
    "reg_lambda": 0.000885,
    "early_stopping_rounds": 30,
    "random_state": 42,
}


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Preparing dataset (horizon={HORIZON} days)...")
    df, feature_cols = prepare_dataset(horizon=HORIZON)
    X = df[feature_cols].values
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features\n")

    splits = walk_forward_split(df, purge_gap=HORIZON)

    # --- Regression model (tuned params adapted for regression) ---
    print("Training regression model...")
    reg_metrics = []
    reg_models = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(**TUNED_PARAMS)
        model.fit(
            X[fit_idx], df["target_return"].values[fit_idx],
            eval_set=[(X[val_idx], df["target_return"].values[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        actuals = df["target_return"].values[test_idx]
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        direction_acc = np.mean((preds > 0) == (actuals > 0))
        reg_metrics.append({"fold": fold_i, "rmse": rmse, "mae": mae, "direction_acc": direction_acc})
        reg_models.append(model)
        print(f"  Fold {fold_i}: RMSE={rmse:.6f}, MAE={mae:.6f}, Direction Acc={direction_acc:.2%}")

    avg_reg = {
        "avg_rmse": np.mean([m["rmse"] for m in reg_metrics]),
        "avg_mae": np.mean([m["mae"] for m in reg_metrics]),
        "avg_direction_acc": np.mean([m["direction_acc"] for m in reg_metrics]),
    }
    print(f"\n  Regression Average: RMSE={avg_reg['avg_rmse']:.6f}, Direction Acc={avg_reg['avg_direction_acc']:.2%}")

    # --- Classification model ---
    print("\nTraining classification model...")
    clf_metrics = []
    clf_models = []
    clf_params = {**TUNED_PARAMS, "eval_metric": "logloss"}
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBClassifier(**clf_params)
        model.fit(
            X[fit_idx], df["target_direction"].values[fit_idx],
            eval_set=[(X[val_idx], df["target_direction"].values[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        acc = accuracy_score(df["target_direction"].values[test_idx], preds)
        clf_metrics.append({"fold": fold_i, "accuracy": acc})
        clf_models.append(model)
        print(f"  Fold {fold_i}: Accuracy={acc:.2%}")

    avg_clf_acc = np.mean([m["accuracy"] for m in clf_metrics])
    print(f"\n  Classification Average: {avg_clf_acc:.2%}")

    # --- Save best models (best performing fold) ---
    best_reg_idx = int(np.argmax([m["direction_acc"] for m in reg_metrics]))
    best_clf_idx = int(np.argmax([m["accuracy"] for m in clf_metrics]))
    best_reg = reg_models[best_reg_idx]
    best_clf = clf_models[best_clf_idx]
    print(f"  Selected regressor from fold {best_reg_idx}, classifier from fold {best_clf_idx}")

    joblib.dump(best_reg, MODELS_DIR / "final_regressor_63d.joblib")
    joblib.dump(best_clf, MODELS_DIR / "final_classifier_63d.joblib")

    # Feature importances
    reg_importances = dict(zip(feature_cols, best_reg.feature_importances_.tolist()))
    reg_importances = dict(sorted(reg_importances.items(), key=lambda x: x[1], reverse=True))
    clf_importances = dict(zip(feature_cols, best_clf.feature_importances_.tolist()))
    clf_importances = dict(sorted(clf_importances.items(), key=lambda x: x[1], reverse=True))

    metadata = {
        "horizon": HORIZON,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_rows": len(df),
        "tuned_params": TUNED_PARAMS,
        "regression_metrics": avg_reg,
        "classification_accuracy": avg_clf_acc,
        "fold_metrics_regression": reg_metrics,
        "fold_metrics_classification": clf_metrics,
        "top_features_regression": dict(list(reg_importances.items())[:20]),
        "top_features_classification": dict(list(clf_importances.items())[:20]),
    }
    with open(MODELS_DIR / "final_metadata_63d.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nFinal models saved to {MODELS_DIR}/")
    print(f"\nTop 15 features (regression):")
    for feat, imp in list(reg_importances.items())[:15]:
        print(f"  {feat}: {imp:.4f}")
    print(f"\nTop 15 features (classification):")
    for feat, imp in list(clf_importances.items())[:15]:
        print(f"  {feat}: {imp:.4f}")

    # --- Generate current prediction ---
    print("\n" + "=" * 60)
    print("CURRENT 63-DAY PREDICTION")
    print("=" * 60)
    latest = df.iloc[[-1]]
    X_latest = latest[feature_cols].values

    pred_return = best_reg.predict(X_latest)[0]
    pred_direction = best_clf.predict(X_latest)[0]
    pred_proba = best_clf.predict_proba(X_latest)[0]

    current_price = latest["coffee_close"].values[0]
    predicted_price = current_price * (1 + pred_return)

    print(f"Date: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted 63-day return: {pred_return:+.2%}")
    print(f"Predicted price in 63 trading days: ${predicted_price:.2f}")
    print(f"Direction: {'UP' if pred_direction == 1 else 'DOWN'} "
          f"(confidence: {max(pred_proba):.1%})")


if __name__ == "__main__":
    main()
