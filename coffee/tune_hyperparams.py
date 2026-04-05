"""Hyperparameter tuning with Optuna for XGBoost coffee price model."""

import json
import numpy as np
from pathlib import Path

import optuna
from xgboost import XGBClassifier

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"


def objective(trial: optuna.Trial, X, y, splits) -> float:
    """Optuna objective: maximize walk-forward direction accuracy."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "early_stopping_rounds": 30,
        "random_state": 42,
    }

    fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBClassifier(**params)
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        fold_accs.append(np.mean(preds == y[test_idx]))

    return np.mean(fold_accs)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Preparing dataset...")
    df, feature_cols = prepare_dataset(horizon=5)
    X = df[feature_cols].values
    y = df["target_direction"].values
    splits = walk_forward_split(df, purge_gap=5)
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(splits)} CV folds\n")

    study = optuna.create_study(direction="maximize", study_name="coffee_xgb")
    study.optimize(lambda trial: objective(trial, X, y, splits), n_trials=100, show_progress_bar=True)

    print(f"\nBest accuracy: {study.best_value:.2%}")
    print(f"Best params: {study.best_params}")

    # Save results
    results = {
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open(MODELS_DIR / "optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {MODELS_DIR / 'optuna_results.json'}")

    # Retrain best model on all data and save
    best_params = study.best_params
    best_params["eval_metric"] = "logloss"
    best_params["early_stopping_rounds"] = 30
    best_params["random_state"] = 42

    # Train on all but the last test fold
    last_train_idx = splits[-1][0]
    last_test_idx = splits[-1][1]
    val_size = min(63, len(last_train_idx) // 5)
    fit_idx, val_idx = last_train_idx[:-val_size], last_train_idx[-val_size:]
    model = XGBClassifier(**best_params)
    model.fit(
        X[fit_idx], y[fit_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False,
    )
    import joblib
    joblib.dump(model, MODELS_DIR / "xgb_classifier_tuned.joblib")
    print(f"Tuned model saved to {MODELS_DIR / 'xgb_classifier_tuned.joblib'}")


if __name__ == "__main__":
    main()
