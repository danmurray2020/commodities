"""Final production model: 63-day horizon, selected features, purged CV, tuned hyperparameters."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import optuna
import joblib

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63

# Load features from existing metadata if available, else use defaults
_DEFAULT_FEATURES = [
    "price_vs_sma_200", "price_vs_sma_50", "macd_signal", "mei_change_3m",
    "cot_noncomm_long", "return_21d", "vietnam_temp_30d_avg", "cot_oi_change",
    "price_lag_2", "season_sin", "sma_10", "volatility_10d", "bb_pct",
    "brazil_temp_30d_avg", "sma_5", "price_lag_3", "rsi_14",
]
try:
    import json as _json
    with open(MODELS_DIR / "production_metadata.json") as _f:
        SELECTED_FEATURES = _json.load(_f)["features"]
    print(f"Loaded {len(SELECTED_FEATURES)} features from metadata")
except (FileNotFoundError, KeyError):
    SELECTED_FEATURES = _DEFAULT_FEATURES
    print("Using default feature list (no metadata found)")


def clf_objective(trial, X, y, splits):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
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
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])
        fold_accs.append(np.mean(preds == y[test_idx]))
    return np.mean(fold_accs) - 0.5 * np.std(fold_accs)


def reg_objective(trial, X, y, splits):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
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
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])
        actuals = y[test_idx]
        fold_accs.append(np.mean((preds > 0) == (actuals > 0)))
    return np.mean(fold_accs) - 0.5 * np.std(fold_accs)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("PRODUCTION MODEL: 63-day, 17 features, purged CV")
    print("=" * 60)

    print("\nPreparing dataset...")
    df, all_feature_cols = prepare_dataset(horizon=HORIZON)

    # Use only selected features (verify they exist)
    feature_cols = [f for f in SELECTED_FEATURES if f in all_feature_cols]
    missing = set(SELECTED_FEATURES) - set(feature_cols)
    if missing:
        print(f"Warning: missing features: {missing}")
    print(f"Using {len(feature_cols)} selected features (of {len(all_feature_cols)} available)")

    X = df[feature_cols].values
    y_ret = df["target_return"].values
    y_dir = df["target_direction"].values

    # Purged walk-forward CV (63-day gap prevents label leakage)
    splits = walk_forward_split(df, purge_gap=HORIZON)
    print(f"Dataset: {len(df)} rows, {len(splits)} purged CV folds\n")

    # --- Optuna tuning (200 trials each) ---
    print("Tuning regression model (200 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    reg_study = optuna.create_study(direction="maximize", study_name="prod_reg")
    reg_study.optimize(lambda t: reg_objective(t, X, y_ret, splits), n_trials=200)
    print(f"Best regression score: {reg_study.best_value:.4f}")

    print("\nTuning classification model (200 trials)...")
    clf_study = optuna.create_study(direction="maximize", study_name="prod_clf")
    clf_study.optimize(lambda t: clf_objective(t, X, y_dir, splits), n_trials=200)
    print(f"Best classification score: {clf_study.best_value:.4f}")

    # --- Evaluate best params ---
    print("\n" + "=" * 60)
    print("EVALUATION (purged walk-forward CV)")
    print("=" * 60)

    reg_params = {**reg_study.best_params, "early_stopping_rounds": 30, "random_state": 42}
    clf_params = {**clf_study.best_params, "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}

    reg_fold_accs = []
    clf_fold_accs = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

        # Regression
        reg = XGBRegressor(**reg_params)
        reg.fit(X[fit_idx], y_ret[fit_idx],
                eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
        reg_preds = reg.predict(X[test_idx])
        reg_acc = np.mean((reg_preds > 0) == (y_ret[test_idx] > 0))
        reg_fold_accs.append(reg_acc)

        # Classification
        clf = XGBClassifier(**clf_params)
        clf.fit(X[fit_idx], y_dir[fit_idx],
                eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
        clf_preds = clf.predict(X[test_idx])
        clf_acc = accuracy_score(y_dir[test_idx], clf_preds)
        clf_fold_accs.append(clf_acc)

        print(f"  Fold {fold_i}: Reg Dir Acc={reg_acc:.2%}, Clf Acc={clf_acc:.2%}")

    print(f"\n  REGRESSION  — Avg: {np.mean(reg_fold_accs):.2%}, Std: {np.std(reg_fold_accs):.2%}")
    print(f"  CLASSIFIER  — Avg: {np.mean(clf_fold_accs):.2%}, Std: {np.std(clf_fold_accs):.2%}")

    # --- Train final models on maximum data ---
    print("\nTraining final models on maximum data...")
    last_train, last_test = splits[-1]

    val_size = min(63, len(last_train) // 5)
    fit_idx, val_idx = last_train[:-val_size], last_train[-val_size:]

    final_reg = XGBRegressor(**reg_params)
    final_reg.fit(X[fit_idx], y_ret[fit_idx],
                  eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)

    final_clf = XGBClassifier(**clf_params)
    final_clf.fit(X[fit_idx], y_dir[fit_idx],
                  eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)

    # Save
    joblib.dump(final_reg, MODELS_DIR / "production_regressor.joblib")
    joblib.dump(final_clf, MODELS_DIR / "production_classifier.joblib")

    metadata = {
        "horizon": HORIZON,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "purge_gap": HORIZON,
        "regression": {
            "params": reg_study.best_params,
            "fold_accuracies": reg_fold_accs,
            "avg_accuracy": float(np.mean(reg_fold_accs)),
            "std_accuracy": float(np.std(reg_fold_accs)),
        },
        "classification": {
            "params": clf_study.best_params,
            "fold_accuracies": clf_fold_accs,
            "avg_accuracy": float(np.mean(clf_fold_accs)),
            "std_accuracy": float(np.std(clf_fold_accs)),
        },
    }
    with open(MODELS_DIR / "production_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # --- Current prediction ---
    print("\n" + "=" * 60)
    print("PRODUCTION 63-DAY PREDICTION")
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
    print(f"Direction: {'UP' if pred_dir == 1 else 'DOWN'} "
          f"(confidence: {max(pred_proba):.1%})")
    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
