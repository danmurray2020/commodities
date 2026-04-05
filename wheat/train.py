"""Train and evaluate wheat price prediction models."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier
import optuna
import joblib

from features import prepare_dataset

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63


def walk_forward_split(
    df: pd.DataFrame, n_splits: int = 5, test_size: int = 63,
    min_train_size: int = 504, purge_gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    splits = []
    for i in range(n_splits):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start - purge_gap
        if train_end < min_train_size:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    splits.reverse()
    return splits


def select_features(df, feature_cols, splits):
    """Permutation importance feature selection."""
    X = df[feature_cols].values
    y = df["target_direction"].values
    all_importances = np.zeros((len(splits), len(feature_cols)))

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Feature selection fold {fold_i}...")
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        neg = np.sum(y[fit_idx] == 0)
        pos = np.sum(y[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=10, gamma=1.0,
            scale_pos_weight=spw,
            eval_metric="logloss", early_stopping_rounds=30, random_state=42,
        )
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        result = permutation_importance(
            model, X[test_idx], y[test_idx],
            n_repeats=10, random_state=42, scoring="accuracy",
        )
        all_importances[fold_i] = result.importances_mean

    mean_imp = all_importances.mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_imp), key=lambda x: x[1], reverse=True)
    selected = [f for f, imp in ranking if imp > 0]

    print(f"\n  Selected {len(selected)} features (from {len(feature_cols)}):")
    for f, imp in ranking[:20]:
        marker = " *" if imp > 0 else ""
        print(f"    {f:<40} {imp:.4f}{marker}")
    return selected


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
        "eval_metric": "logloss", "early_stopping_rounds": 30, "random_state": 42,
    }
    fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        neg = np.sum(y[fit_idx] == 0)
        pos = np.sum(y[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        params["scale_pos_weight"] = spw
        model = XGBClassifier(**params)
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        fold_accs.append(np.mean(model.predict(X[test_idx]) == y[test_idx]))
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
        "early_stopping_rounds": 30, "random_state": 42,
    }
    fold_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(**params)
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])
        fold_accs.append(np.mean((preds > 0) == (y[test_idx] > 0)))
    return np.mean(fold_accs) - 0.5 * np.std(fold_accs)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("WHEAT FUTURES PREDICTION MODEL")
    print("=" * 60)

    print("\nPreparing dataset...")
    df, all_feature_cols = prepare_dataset(horizon=HORIZON)

    # Separate CV for feature selection (excludes last 2 folds to prevent leakage)
    selection_splits = walk_forward_split(df, n_splits=3, purge_gap=HORIZON)
    print(f"Dataset: {len(df)} rows, {len(all_feature_cols)} features")
    print(f"Feature selection: {len(selection_splits)} folds (separate from training)\n")

    # Feature selection on separate folds
    print("Running feature selection...")
    selected = select_features(df, all_feature_cols, selection_splits)

    # Training/evaluation uses full 5-fold CV
    splits = walk_forward_split(df, purge_gap=HORIZON)
    print(f"Training: {len(splits)} folds (full walk-forward CV)")

    feature_cols = selected
    X = df[feature_cols].values
    y_ret = df["target_return"].values
    y_dir = df["target_direction"].values

    # Optuna tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nTuning regression model (200 trials)...")
    reg_study = optuna.create_study(direction="maximize", study_name="wheat_reg")
    reg_study.optimize(lambda t: reg_objective(t, X, y_ret, splits), n_trials=200)
    print(f"Best regression score: {reg_study.best_value:.4f}")

    print(f"\nTuning classification model (200 trials)...")
    clf_study = optuna.create_study(direction="maximize", study_name="wheat_clf")
    clf_study.optimize(lambda t: clf_objective(t, X, y_dir, splits), n_trials=200)
    print(f"Best classification score: {clf_study.best_value:.4f}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION (purged walk-forward CV)")
    print("=" * 60)
    reg_params = {**reg_study.best_params, "early_stopping_rounds": 30, "random_state": 42}
    clf_params = {**clf_study.best_params, "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}

    reg_fold_accs = []
    reg_fold_maes = []
    reg_fold_rmses = []
    clf_fold_accs = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

        reg = XGBRegressor(**reg_params)
        reg.fit(X[fit_idx], y_ret[fit_idx],
                eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
        reg_preds = reg.predict(X[test_idx])
        reg_acc = np.mean((reg_preds > 0) == (y_ret[test_idx] > 0))
        reg_mae = float(np.mean(np.abs(reg_preds - y_ret[test_idx])))
        reg_rmse = float(np.sqrt(np.mean((reg_preds - y_ret[test_idx]) ** 2)))
        reg_fold_accs.append(reg_acc)
        reg_fold_maes.append(reg_mae)
        reg_fold_rmses.append(reg_rmse)

        neg = np.sum(y_dir[fit_idx] == 0)
        pos = np.sum(y_dir[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
        clf.fit(X[fit_idx], y_dir[fit_idx],
                eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
        clf_acc = accuracy_score(y_dir[test_idx], clf.predict(X[test_idx]))
        clf_fold_accs.append(clf_acc)
        print(f"  Fold {fold_i}: Reg Dir={reg_acc:.2%} MAE={reg_mae:.4f} RMSE={reg_rmse:.4f}, Clf={clf_acc:.2%}")

    print(f"\n  REGRESSION  — Avg: {np.mean(reg_fold_accs):.2%}, Std: {np.std(reg_fold_accs):.2%}, MAE: {np.mean(reg_fold_maes):.4f}, RMSE: {np.mean(reg_fold_rmses):.4f}")
    print(f"  CLASSIFIER  — Avg: {np.mean(clf_fold_accs):.2%}, Std: {np.std(clf_fold_accs):.2%}")

    # Train final models
    last_train, last_test = splits[-1]
    val_size = min(63, len(last_train) // 5)
    fit_idx, val_idx = last_train[:-val_size], last_train[-val_size:]
    final_reg = XGBRegressor(**reg_params)
    final_reg.fit(X[fit_idx], y_ret[fit_idx],
                  eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
    neg = np.sum(y_dir[fit_idx] == 0)
    pos = np.sum(y_dir[fit_idx] == 1)
    spw = neg / pos if pos > 0 else 1.0
    final_clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
    final_clf.fit(X[fit_idx], y_dir[fit_idx],
                  eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)

    # Held-out evaluation (last 126 rows never seen during CV)
    holdout_size = 126
    if len(df) > holdout_size + 504:  # Only if we have enough data
        holdout_idx = np.arange(len(df) - holdout_size, len(df))
        X_holdout = X[holdout_idx]
        y_ret_holdout = y_ret[holdout_idx]
        y_dir_holdout = y_dir[holdout_idx]

        holdout_reg_preds = final_reg.predict(X_holdout)
        holdout_reg_acc = float(np.mean((holdout_reg_preds > 0) == (y_ret_holdout > 0)))
        holdout_reg_mae = float(np.mean(np.abs(holdout_reg_preds - y_ret_holdout)))

        holdout_clf_preds = final_clf.predict(X_holdout)
        holdout_clf_acc = float(accuracy_score(y_dir_holdout, holdout_clf_preds))

        print(f"\n  HELD-OUT TEST ({holdout_size} days, never seen during CV):")
        print(f"    Regression direction: {holdout_reg_acc:.2%}, MAE: {holdout_reg_mae:.4f}")
        print(f"    Classification:       {holdout_clf_acc:.2%}")
    else:
        holdout_reg_acc = None
        holdout_clf_acc = None
        holdout_reg_mae = None
        print("\n  HELD-OUT TEST: Skipped (insufficient data)")

    joblib.dump(final_reg, MODELS_DIR / "production_regressor.joblib")
    joblib.dump(final_clf, MODELS_DIR / "production_classifier.joblib")

    strategy_config = {
        "confidence_threshold": 0.70, "stop_loss_pct": 0.10,
        "take_profit_multiplier": 1.0, "max_hold_days": 63,
        "allow_short": True,
    }

    metadata = {
        "commodity": "wheat",
        "ticker": "ZW=F",
        "horizon": HORIZON,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "purge_gap": HORIZON,
        "strategy": strategy_config,
        "regression": {
            "params": reg_study.best_params,
            "fold_accuracies": reg_fold_accs,
            "avg_accuracy": float(np.mean(reg_fold_accs)),
            "std_accuracy": float(np.std(reg_fold_accs)),
            "fold_maes": reg_fold_maes,
            "fold_rmses": reg_fold_rmses,
            "avg_mae": float(np.mean(reg_fold_maes)),
            "avg_rmse": float(np.mean(reg_fold_rmses)),
        },
        "classification": {
            "params": clf_study.best_params,
            "fold_accuracies": clf_fold_accs,
            "avg_accuracy": float(np.mean(clf_fold_accs)),
            "std_accuracy": float(np.std(clf_fold_accs)),
        },
        "holdout": {
            "size": holdout_size if len(df) > holdout_size + 504 else 0,
            "reg_direction_accuracy": holdout_reg_acc,
            "reg_mae": holdout_reg_mae,
            "clf_accuracy": holdout_clf_acc,
        },
    }
    with open(MODELS_DIR / "production_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Prediction
    print("\n" + "=" * 60)
    print("WHEAT 63-DAY PREDICTION")
    print("=" * 60)
    latest = df.iloc[[-1]]
    X_latest = latest[feature_cols].values
    pred_return = float(final_reg.predict(X_latest)[0])
    pred_dir = int(final_clf.predict(X_latest)[0])
    pred_proba = final_clf.predict_proba(X_latest)[0]
    confidence = float(pred_proba[pred_dir])
    current_price = float(latest["wheat_close"].values[0])
    predicted_price = current_price * (1 + pred_return)

    print(f"Date: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted 63-day return: {pred_return:+.2%}")
    print(f"Predicted price: ${predicted_price:.2f}")
    print(f"Direction: {'UP' if pred_dir == 1 else 'DOWN'} (confidence: {confidence:.1%})")

    if confidence >= strategy_config["confidence_threshold"]:
        direction = "LONG" if pred_dir == 1 else "SHORT"
        tp = abs(pred_return)
        sl = strategy_config["stop_loss_pct"]
        tp_price = current_price * (1 + tp) if direction == "LONG" else current_price * (1 - tp)
        sl_price = current_price * (1 - sl) if direction == "LONG" else current_price * (1 + sl)
        print(f"\nStrategy: {direction}")
        print(f"  Entry: ${current_price:.2f}, TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")
    else:
        print(f"\nStrategy: NO TRADE (confidence {confidence:.1%} < 70%)")

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
