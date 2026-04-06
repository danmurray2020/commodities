"""Multi-horizon, multi-model ensemble.

Trains 3 model types (XGBoost, LightGBM, Ridge) at 3 horizons (5d, 10d, 21d)
= 9 models per commodity. Combines with weighted voting.

Different model types have decorrelated errors — a tree model and a linear
model make different mistakes, so the ensemble is more robust than any single
model.

When all models agree on direction, confidence is genuinely high.
When they disagree, we stay out.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import optuna
import joblib

from .train_utils import (
    walk_forward_split, evaluate_predictions, evaluate_classification,
    filter_stable_features,
)


HORIZONS = [5, 10, 21]
MODEL_TYPES = ["xgboost", "lightgbm", "ridge"]
OPTUNA_TRIALS_PER_MODEL = 60  # 60 trials x 3 models x 3 horizons = 540 total


def train_single_horizon(
    df_full: pd.DataFrame,
    all_feature_cols: list[str],
    horizon: int,
    price_col: str,
    optuna_trials: int = OPTUNA_TRIALS_PER_MODEL,
) -> dict:
    """Train regressor + classifier for a single horizon.

    Returns dict with models, metrics, selected features, and params.
    """
    # Build features at this horizon
    from importlib import import_module

    # Rebuild targets for this horizon
    target_return = np.log(
        df_full[price_col].shift(-horizon) / df_full[price_col]
    )
    target_direction = (target_return > 0).astype(int)

    df = df_full.copy()
    df["target_return"] = target_return
    df["target_direction"] = target_direction
    df = df.dropna(subset=["target_return"])

    # Non-overlapping test size: enough for ~8 independent obs per fold
    test_size = max(horizon * 10, 126)  # at least 10 non-overlapping per fold

    # Feature selection via permutation importance
    selection_splits = walk_forward_split(
        df, n_splits=3, test_size=test_size, purge_gap=horizon
    )

    # Feature stability filter (use only training data to avoid leakage)
    train_end = int(selection_splits[0][1][0]) if selection_splits else None
    stable, _ = filter_stable_features(df, all_feature_cols, train_end=train_end)
    if len(stable) < 5:
        stable = all_feature_cols[:max(15, len(all_feature_cols) // 3)]

    if len(selection_splits) < 2:
        # Not enough data for feature selection at this horizon
        feature_cols = stable[:20]
    else:
        feature_cols = _select_features(df, stable, selection_splits, horizon)

    if len(feature_cols) < 3:
        feature_cols = stable[:10]

    # Training splits
    splits = walk_forward_split(
        df, n_splits=5, test_size=test_size, purge_gap=horizon
    )

    if len(splits) < 2:
        return {"status": "insufficient_data", "horizon": horizon}

    X = df[feature_cols].values
    y_ret = df["target_return"].values
    y_dir = df["target_direction"].values

    # Optuna tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def reg_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.5, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "early_stopping_rounds": 30, "random_state": 42,
        }
        scores = []
        for train_idx, test_idx in splits:
            val_size = min(horizon * 2, len(train_idx) // 5)
            if val_size < 5:
                val_size = len(train_idx) // 5
            fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
            model = XGBRegressor(**params)
            model.fit(X[fit_idx], y_ret[fit_idx],
                      eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
            preds = model.predict(X[test_idx])
            # Use direction accuracy (more reliable with small samples than Spearman)
            dir_acc = float(np.mean((preds > 0) == (y_ret[test_idx] > 0)))
            scores.append(dir_acc)
        return np.mean(scores) - 0.3 * np.std(scores)

    def clf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.5, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "eval_metric": "logloss", "early_stopping_rounds": 30, "random_state": 42,
        }
        scores = []
        for train_idx, test_idx in splits:
            val_size = min(horizon * 2, len(train_idx) // 5)
            if val_size < 5:
                val_size = len(train_idx) // 5
            fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
            neg = np.sum(y_dir[fit_idx] == 0)
            pos = np.sum(y_dir[fit_idx] == 1)
            params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
            model = XGBClassifier(**params)
            model.fit(X[fit_idx], y_dir[fit_idx],
                      eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
            preds = model.predict(X[test_idx])
            acc = float(np.mean(preds == y_dir[test_idx]))
            scores.append(acc)
        return np.mean(scores) - 0.3 * np.std(scores)

    reg_study = optuna.create_study(direction="maximize")
    reg_study.optimize(reg_objective, n_trials=optuna_trials)

    clf_study = optuna.create_study(direction="maximize")
    clf_study.optimize(clf_objective, n_trials=optuna_trials)

    # Evaluate with best params
    reg_params = {**reg_study.best_params, "early_stopping_rounds": 30, "random_state": 42}
    clf_params = {**clf_study.best_params, "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}

    # Higher regularization floor
    for p in [reg_params, clf_params]:
        p["gamma"] = max(p.get("gamma", 0), 0.5)
        p["reg_alpha"] = max(p.get("reg_alpha", 0), 0.01)
        p["reg_lambda"] = max(p.get("reg_lambda", 0), 0.01)

    reg_spearman = []
    reg_dir_acc = []
    clf_acc_ind = []

    for train_idx, test_idx in splits:
        val_size = min(horizon * 2, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

        reg = XGBRegressor(**reg_params)
        reg.fit(X[fit_idx], y_ret[fit_idx],
                eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
        reg_preds = reg.predict(X[test_idx])
        rm = evaluate_predictions(y_ret[test_idx], reg_preds, horizon=horizon)
        reg_spearman.append(rm["spearman"])
        reg_dir_acc.append(rm["dir_acc_independent"])

        neg = np.sum(y_dir[fit_idx] == 0)
        pos = np.sum(y_dir[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
        clf.fit(X[fit_idx], y_dir[fit_idx],
                eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
        cm = evaluate_classification(y_dir[test_idx], clf.predict(X[test_idx]), horizon=horizon)
        clf_acc_ind.append(cm["acc_independent"])

    # Train final models on all data
    val_size = min(horizon * 2, len(X) // 5)
    fit_idx = np.arange(0, len(X) - val_size)
    val_idx = np.arange(len(X) - val_size, len(X))

    final_reg = XGBRegressor(**reg_params)
    final_reg.fit(X[fit_idx], y_ret[fit_idx],
                  eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)

    neg = np.sum(y_dir[fit_idx] == 0)
    pos = np.sum(y_dir[fit_idx] == 1)
    spw = neg / pos if pos > 0 else 1.0
    final_clf = XGBClassifier(**clf_params, scale_pos_weight=spw)
    final_clf.fit(X[fit_idx], y_dir[fit_idx],
                  eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)

    return {
        "status": "ok",
        "horizon": horizon,
        "regressor": final_reg,
        "classifier": final_clf,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "reg_params": reg_study.best_params,
        "clf_params": clf_study.best_params,
        "metrics": {
            "reg_spearman": reg_spearman,
            "avg_spearman": float(np.mean(reg_spearman)),
            "reg_dir_acc": reg_dir_acc,
            "avg_dir_acc": float(np.mean(reg_dir_acc)),
            "clf_acc_ind": clf_acc_ind,
            "avg_clf_acc": float(np.mean(clf_acc_ind)),
        },
    }


def train_multi_model_horizon(
    df_full: pd.DataFrame,
    all_feature_cols: list[str],
    horizon: int,
    price_col: str,
    optuna_trials: int = OPTUNA_TRIALS_PER_MODEL,
) -> list[dict]:
    """Train XGBoost + LightGBM + Ridge for a single horizon.

    Returns a list of model results (one per model type that succeeded).
    """
    # Rebuild targets
    target_return = np.log(
        df_full[price_col].shift(-horizon) / df_full[price_col]
    )
    target_direction = (target_return > 0).astype(int)

    df = df_full.copy()
    df["target_return"] = target_return
    df["target_direction"] = target_direction
    df = df.dropna(subset=["target_return"])

    test_size = max(horizon * 10, 126)

    # Feature selection
    selection_splits = walk_forward_split(df, n_splits=3, test_size=test_size, purge_gap=horizon)

    # Feature stability filter (use only training data to avoid leakage)
    train_end = int(selection_splits[0][1][0]) if selection_splits else None
    stable, _ = filter_stable_features(df, all_feature_cols, train_end=train_end)
    if len(stable) < 5:
        stable = all_feature_cols[:max(15, len(all_feature_cols) // 3)]
    if len(selection_splits) < 2:
        feature_cols = stable[:20]
    else:
        feature_cols = _select_features(df, stable, selection_splits, horizon)
    if len(feature_cols) < 3:
        feature_cols = stable[:10]

    splits = walk_forward_split(df, n_splits=5, test_size=test_size, purge_gap=horizon)
    if len(splits) < 2:
        return []

    X = df[feature_cols].values
    y_ret = df["target_return"].values
    y_dir = df["target_direction"].values

    results = []

    # ── XGBoost ──
    try:
        xgb_result = _train_xgboost(X, y_ret, y_dir, splits, horizon, optuna_trials)
        xgb_result["model_type"] = "xgboost"
        xgb_result["features"] = feature_cols
        xgb_result["horizon"] = horizon
        results.append(xgb_result)
        print(f"    xgboost:  dir={xgb_result['avg_dir_acc']:.1%}, clf={xgb_result['avg_clf_acc']:.1%}")
    except Exception as e:
        print(f"    xgboost:  FAILED ({e})")

    # ── LightGBM ──
    try:
        lgb_result = _train_lightgbm(X, y_ret, y_dir, splits, horizon, optuna_trials)
        lgb_result["model_type"] = "lightgbm"
        lgb_result["features"] = feature_cols
        lgb_result["horizon"] = horizon
        results.append(lgb_result)
        print(f"    lightgbm: dir={lgb_result['avg_dir_acc']:.1%}, clf={lgb_result['avg_clf_acc']:.1%}")
    except Exception as e:
        print(f"    lightgbm: FAILED ({e})")

    # ── Ridge / Logistic (no Optuna needed — fast) ──
    try:
        ridge_result = _train_ridge(X, y_ret, y_dir, splits, horizon)
        ridge_result["model_type"] = "ridge"
        ridge_result["features"] = feature_cols
        ridge_result["horizon"] = horizon
        results.append(ridge_result)
        print(f"    ridge:    dir={ridge_result['avg_dir_acc']:.1%}, clf={ridge_result['avg_clf_acc']:.1%}")
    except Exception as e:
        print(f"    ridge:    FAILED ({e})")

    return results


def _train_xgboost(X, y_ret, y_dir, splits, horizon, optuna_trials):
    """Train XGBoost regressor + classifier with Optuna tuning."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.5, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "early_stopping_rounds": 30, "random_state": 42,
        }
        scores = []
        for train_idx, test_idx in splits:
            val_size = max(5, min(horizon * 2, len(train_idx) // 5))
            fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
            model = XGBRegressor(**params)
            model.fit(X[fit_idx], y_ret[fit_idx],
                      eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
            preds = model.predict(X[test_idx])
            scores.append(float(np.mean((preds > 0) == (y_ret[test_idx] > 0))))
        return np.mean(scores) - 0.3 * np.std(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)
    best = {**study.best_params, "early_stopping_rounds": 30, "random_state": 42}

    # Evaluate + train final
    dir_accs, clf_accs = [], []
    for train_idx, test_idx in splits:
        val_size = max(5, min(horizon * 2, len(train_idx) // 5))
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        reg = XGBRegressor(**best)
        reg.fit(X[fit_idx], y_ret[fit_idx], eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
        preds = reg.predict(X[test_idx])
        m = evaluate_predictions(y_ret[test_idx], preds, horizon=horizon)
        dir_accs.append(m["dir_acc_independent"])

        neg, pos = np.sum(y_dir[fit_idx] == 0), np.sum(y_dir[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        clf = XGBClassifier(**{**study.best_params, "eval_metric": "logloss",
                               "early_stopping_rounds": 30, "random_state": 42,
                               "scale_pos_weight": spw})
        clf.fit(X[fit_idx], y_dir[fit_idx], eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)
        cm = evaluate_classification(y_dir[test_idx], clf.predict(X[test_idx]), horizon=horizon)
        clf_accs.append(cm["acc_independent"])

    # Final models
    val_size = max(5, len(X) // 5)
    fit_idx, val_idx = np.arange(len(X) - val_size), np.arange(len(X) - val_size, len(X))
    final_reg = XGBRegressor(**best)
    final_reg.fit(X[fit_idx], y_ret[fit_idx], eval_set=[(X[val_idx], y_ret[val_idx])], verbose=False)
    neg, pos = np.sum(y_dir[fit_idx] == 0), np.sum(y_dir[fit_idx] == 1)
    final_clf = XGBClassifier(**{**study.best_params, "eval_metric": "logloss",
                                 "early_stopping_rounds": 30, "random_state": 42,
                                 "scale_pos_weight": neg / pos if pos > 0 else 1.0})
    final_clf.fit(X[fit_idx], y_dir[fit_idx], eval_set=[(X[val_idx], y_dir[val_idx])], verbose=False)

    return {
        "regressor": final_reg, "classifier": final_clf,
        "avg_dir_acc": float(np.mean(dir_accs)), "avg_clf_acc": float(np.mean(clf_accs)),
        "params": study.best_params,
    }


def _train_lightgbm(X, y_ret, y_dir, splits, horizon, optuna_trials):
    """Train LightGBM regressor + classifier with Optuna tuning."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "random_state": 42, "verbose": -1,
        }
        scores = []
        for train_idx, test_idx in splits:
            val_size = max(5, min(horizon * 2, len(train_idx) // 5))
            fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
            model = lgb.LGBMRegressor(**params)
            model.fit(X[fit_idx], y_ret[fit_idx],
                      eval_set=[(X[val_idx], y_ret[val_idx])],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            preds = model.predict(X[test_idx])
            scores.append(float(np.mean((preds > 0) == (y_ret[test_idx] > 0))))
        return np.mean(scores) - 0.3 * np.std(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)
    best = {**study.best_params, "random_state": 42, "verbose": -1}

    dir_accs, clf_accs = [], []
    for train_idx, test_idx in splits:
        val_size = max(5, min(horizon * 2, len(train_idx) // 5))
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        reg = lgb.LGBMRegressor(**best)
        reg.fit(X[fit_idx], y_ret[fit_idx], eval_set=[(X[val_idx], y_ret[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        preds = reg.predict(X[test_idx])
        m = evaluate_predictions(y_ret[test_idx], preds, horizon=horizon)
        dir_accs.append(m["dir_acc_independent"])

        clf = lgb.LGBMClassifier(**{**study.best_params, "random_state": 42, "verbose": -1})
        clf.fit(X[fit_idx], y_dir[fit_idx], eval_set=[(X[val_idx], y_dir[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        cm = evaluate_classification(y_dir[test_idx], clf.predict(X[test_idx]), horizon=horizon)
        clf_accs.append(cm["acc_independent"])

    val_size = max(5, len(X) // 5)
    fit_idx, val_idx = np.arange(len(X) - val_size), np.arange(len(X) - val_size, len(X))
    final_reg = lgb.LGBMRegressor(**best)
    final_reg.fit(X[fit_idx], y_ret[fit_idx], eval_set=[(X[val_idx], y_ret[val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    final_clf = lgb.LGBMClassifier(**{**study.best_params, "random_state": 42, "verbose": -1})
    final_clf.fit(X[fit_idx], y_dir[fit_idx], eval_set=[(X[val_idx], y_dir[val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    return {
        "regressor": final_reg, "classifier": final_clf,
        "avg_dir_acc": float(np.mean(dir_accs)), "avg_clf_acc": float(np.mean(clf_accs)),
        "params": study.best_params,
    }


def _train_ridge(X, y_ret, y_dir, splits, horizon):
    """Train Ridge regressor + Logistic classifier (no Optuna — fast)."""
    dir_accs, clf_accs = [], []

    for train_idx, test_idx in splits:
        # Ridge regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_s, y_ret[train_idx])
        preds = reg.predict(X_test_s)
        m = evaluate_predictions(y_ret[test_idx], preds, horizon=horizon)
        dir_accs.append(m["dir_acc_independent"])

        # Logistic regression
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_dir[train_idx])
        cm = evaluate_classification(y_dir[test_idx], clf.predict(X_test_s), horizon=horizon)
        clf_accs.append(cm["acc_independent"])

    # Final models (wrapped in Pipeline for proper scaling)
    final_reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    final_reg.fit(X, y_ret)

    final_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])
    final_clf.fit(X, y_dir)

    return {
        "regressor": final_reg, "classifier": final_clf,
        "avg_dir_acc": float(np.mean(dir_accs)), "avg_clf_acc": float(np.mean(clf_accs)),
        "params": {"alpha": 1.0, "C": 0.1},
    }


def _select_features(df, feature_cols, splits, horizon):
    """Quick permutation importance feature selection."""
    X = df[feature_cols].values
    y = df["target_direction"].values
    all_importances = np.zeros((len(splits), len(feature_cols)))

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        val_size = min(horizon * 2, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        neg = np.sum(y[fit_idx] == 0)
        pos = np.sum(y[fit_idx] == 1)
        spw = neg / pos if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=10, gamma=1.0, scale_pos_weight=spw,
            eval_metric="logloss", early_stopping_rounds=20, random_state=42,
        )
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        result = permutation_importance(
            model, X[test_idx], y[test_idx],
            n_repeats=5, random_state=42, scoring="accuracy",
        )
        all_importances[fold_i] = result.importances_mean

    mean_imp = all_importances.mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_imp), key=lambda x: x[1], reverse=True)
    selected = [f for f, imp in ranking if imp > 0]
    if len(selected) < 5:
        selected = [f for f, _ in ranking[:max(10, len(ranking) // 4)]]
    return selected


# ── Ensemble prediction ──────────────────────────────────────────────────

def ensemble_predict(models_dir: Path, df: pd.DataFrame, price_col: str) -> dict:
    """Generate ensemble prediction from multi-horizon models.

    Loads models at each horizon, generates predictions, and combines them
    with weighted voting. Agreement between horizons = high confidence.
    """
    meta_path = models_dir / "ensemble_metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    horizon_results = []
    weights = []

    for h_meta in meta.get("horizons", []):
        horizon = h_meta["horizon"]
        features = h_meta["features"]
        weight = h_meta.get("weight", 1.0)

        reg_path = models_dir / f"ensemble_reg_{horizon}d.joblib"
        clf_path = models_dir / f"ensemble_clf_{horizon}d.joblib"

        if not reg_path.exists() or not clf_path.exists():
            continue

        # Check features exist
        available = [f for f in features if f in df.columns]
        if len(available) < len(features):
            continue

        reg = joblib.load(reg_path)
        clf = joblib.load(clf_path)

        latest = df.iloc[[-1]]
        X = latest[features].values

        pred_return = float(reg.predict(X)[0])
        pred_dir = int(clf.predict(X)[0])
        pred_proba = clf.predict_proba(X)[0]
        confidence = float(pred_proba[pred_dir])

        horizon_results.append({
            "horizon": horizon,
            "pred_return": pred_return,
            "direction": "UP" if pred_dir == 1 else "DOWN",
            "confidence": confidence,
            "weight": weight,
        })
        weights.append(weight)

    if not horizon_results:
        return None

    # Combine: weighted vote on direction
    total_weight = sum(weights)
    up_weight = sum(
        h["weight"] for h in horizon_results if h["direction"] == "UP"
    )
    down_weight = total_weight - up_weight

    # Ensemble direction
    if up_weight > down_weight:
        ensemble_dir = "UP"
        agreement = up_weight / total_weight
    else:
        ensemble_dir = "DOWN"
        agreement = down_weight / total_weight

    # Ensemble return: weighted average of predicted returns
    ensemble_return = sum(
        h["pred_return"] * h["weight"] for h in horizon_results
    ) / total_weight

    # Ensemble confidence: combines individual confidences + agreement bonus
    avg_confidence = sum(
        h["confidence"] * h["weight"] for h in horizon_results
    ) / total_weight

    # Agreement bonus: if all horizons agree, boost confidence
    all_agree = all(h["direction"] == ensemble_dir for h in horizon_results)
    if all_agree:
        ensemble_confidence = min(avg_confidence * 1.1, 0.99)
    else:
        ensemble_confidence = avg_confidence * agreement

    current_price = float(df.iloc[-1][price_col])

    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "price": round(current_price, 4),
        "direction": ensemble_dir,
        "pred_return": round(ensemble_return, 6),
        "pred_price": round(current_price * (1 + ensemble_return), 4),
        "confidence": round(ensemble_confidence, 4),
        "agreement": round(agreement, 4),
        "all_agree": all_agree,
        "n_horizons": len(horizon_results),
        "horizon_details": horizon_results,
    }
