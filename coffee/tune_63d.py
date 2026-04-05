"""Optuna hyperparameter tuning specifically for the 63-day horizon model."""

import json
import numpy as np
from pathlib import Path

import optuna
import joblib
from xgboost import XGBClassifier, XGBRegressor

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63


def clf_objective(trial: optuna.Trial, X, y, splits) -> float:
    """Maximize walk-forward direction accuracy for classification."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 5),
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

    # Penalize high variance across folds to encourage stability
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    return mean_acc - 0.5 * std_acc  # reward accuracy, penalize instability


def reg_objective(trial: optuna.Trial, X, y, splits) -> float:
    """Maximize walk-forward direction accuracy for regression."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 5),
        "early_stopping_rounds": 30,
        "random_state": 42,
    }

    fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(**params)
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        actuals = y[test_idx]
        fold_accs.append(np.mean((preds > 0) == (actuals > 0)))

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    return mean_acc - 0.5 * std_acc


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Preparing dataset (horizon={HORIZON})...")
    df, feature_cols = prepare_dataset(horizon=HORIZON)
    X = df[feature_cols].values
    splits = walk_forward_split(df, purge_gap=HORIZON)
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(splits)} folds\n")

    # Tune regression model
    print("=" * 60)
    print("Tuning REGRESSION model (150 trials)...")
    print("=" * 60)
    reg_study = optuna.create_study(direction="maximize", study_name="coffee_63d_reg")
    reg_study.optimize(
        lambda trial: reg_objective(trial, X, df["target_return"].values, splits),
        n_trials=150, show_progress_bar=True,
    )
    print(f"\nBest regression score (acc - 0.5*std): {reg_study.best_value:.4f}")
    print(f"Best params: {reg_study.best_params}")

    # Tune classification model
    print("\n" + "=" * 60)
    print("Tuning CLASSIFICATION model (150 trials)...")
    print("=" * 60)
    clf_study = optuna.create_study(direction="maximize", study_name="coffee_63d_clf")
    clf_study.optimize(
        lambda trial: clf_objective(trial, X, df["target_direction"].values, splits),
        n_trials=150, show_progress_bar=True,
    )
    print(f"\nBest classification score (acc - 0.5*std): {clf_study.best_value:.4f}")
    print(f"Best params: {clf_study.best_params}")

    # Retrain best models on full walk-forward and evaluate
    print("\n" + "=" * 60)
    print("Retraining best models and evaluating...")
    print("=" * 60)

    # Regression
    reg_params = {**reg_study.best_params, "early_stopping_rounds": 30, "random_state": 42}
    reg_fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(**reg_params)
        model.fit(X[fit_idx], df["target_return"].values[fit_idx],
                  eval_set=[(X[val_idx], df["target_return"].values[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])
        actuals = df["target_return"].values[test_idx]
        acc = np.mean((preds > 0) == (actuals > 0))
        reg_fold_accs.append(acc)
        print(f"  Reg Fold {len(reg_fold_accs)-1}: Direction Acc={acc:.2%}")
    print(f"  Reg Average: {np.mean(reg_fold_accs):.2%} (std={np.std(reg_fold_accs):.2%})")

    # Classification
    clf_params = {**clf_study.best_params, "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}
    clf_fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBClassifier(**clf_params)
        model.fit(X[fit_idx], df["target_direction"].values[fit_idx],
                  eval_set=[(X[val_idx], df["target_direction"].values[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])
        acc = np.mean(preds == df["target_direction"].values[test_idx])
        clf_fold_accs.append(acc)
        print(f"  Clf Fold {len(clf_fold_accs)-1}: Accuracy={acc:.2%}")
    print(f"  Clf Average: {np.mean(clf_fold_accs):.2%} (std={np.std(clf_fold_accs):.2%})")

    # Save final tuned models (train on largest window)
    last_train, last_test = splits[-1]

    val_size = min(63, len(last_train) // 5)
    fit_idx, val_idx = last_train[:-val_size], last_train[-val_size:]
    final_reg = XGBRegressor(**reg_params)
    final_reg.fit(X[fit_idx], df["target_return"].values[fit_idx],
                  eval_set=[(X[val_idx], df["target_return"].values[val_idx])], verbose=False)
    joblib.dump(final_reg, MODELS_DIR / "final_regressor_63d_tuned.joblib")

    val_size = min(63, len(last_train) // 5)
    fit_idx, val_idx = last_train[:-val_size], last_train[-val_size:]
    final_clf = XGBClassifier(**clf_params)
    final_clf.fit(X[fit_idx], df["target_direction"].values[fit_idx],
                  eval_set=[(X[val_idx], df["target_direction"].values[val_idx])], verbose=False)
    joblib.dump(final_clf, MODELS_DIR / "final_classifier_63d_tuned.joblib")

    # Save metadata
    results = {
        "horizon": HORIZON,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "regression": {
            "best_params": reg_study.best_params,
            "fold_accuracies": reg_fold_accs,
            "avg_accuracy": float(np.mean(reg_fold_accs)),
            "std_accuracy": float(np.std(reg_fold_accs)),
        },
        "classification": {
            "best_params": clf_study.best_params,
            "fold_accuracies": clf_fold_accs,
            "avg_accuracy": float(np.mean(clf_fold_accs)),
            "std_accuracy": float(np.std(clf_fold_accs)),
        },
    }
    with open(MODELS_DIR / "final_tuned_63d_metadata.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Current prediction
    print("\n" + "=" * 60)
    print("CURRENT 63-DAY PREDICTION (TUNED)")
    print("=" * 60)
    latest = df.iloc[[-1]]
    X_latest = latest[feature_cols].values
    pred_return = final_reg.predict(X_latest)[0]
    pred_dir = final_clf.predict(X_latest)[0]
    pred_proba = final_clf.predict_proba(X_latest)[0]
    current_price = latest["coffee_close"].values[0]
    predicted_price = current_price * (1 + pred_return)

    print(f"Date: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted 63-day return: {pred_return:+.2%}")
    print(f"Predicted price in ~3 months: ${predicted_price:.2f}")
    print(f"Direction: {'UP' if pred_dir == 1 else 'DOWN'} (confidence: {max(pred_proba):.1%})")


if __name__ == "__main__":
    main()
