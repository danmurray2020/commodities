"""V2 model: XGBoost + CatBoost + LightGBM ensemble with expanded features."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
import joblib

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63


def evaluate_ensemble(df, feature_cols, splits, reg_models, clf_models):
    """Evaluate a set of models as an ensemble on walk-forward splits."""
    X = df[feature_cols].values
    y_ret = df["target_return"].values
    y_dir = df["target_direction"].values

    reg_fold_accs = []
    clf_fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Train all models
        reg_preds_all = []
        clf_preds_all = []
        clf_proba_all = []

        for name, RegClass, ClfClass, reg_params, clf_params in zip(
            [m[0] for m in reg_models],
            [m[1] for m in reg_models],
            [m[1] for m in clf_models],
            [m[2] for m in reg_models],
            [m[2] for m in clf_models],
        ):
            # Regression
            reg = RegClass(**reg_params)
            if name == "catboost":
                reg.fit(X[train_idx], y_ret[train_idx],
                        eval_set=(X[test_idx], y_ret[test_idx]), verbose=False)
            elif name == "lightgbm":
                reg.fit(X[train_idx], y_ret[train_idx],
                        eval_set=[(X[test_idx], y_ret[test_idx])])
            else:
                reg.fit(X[train_idx], y_ret[train_idx],
                        eval_set=[(X[test_idx], y_ret[test_idx])], verbose=False)
            reg_preds_all.append(reg.predict(X[test_idx]))

            # Classification
            clf = ClfClass(**clf_params)
            if name == "catboost":
                clf.fit(X[train_idx], y_dir[train_idx],
                        eval_set=(X[test_idx], y_dir[test_idx]), verbose=False)
            elif name == "lightgbm":
                clf.fit(X[train_idx], y_dir[train_idx],
                        eval_set=[(X[test_idx], y_dir[test_idx])])
            else:
                clf.fit(X[train_idx], y_dir[train_idx],
                        eval_set=[(X[test_idx], y_dir[test_idx])], verbose=False)
            clf_preds_all.append(clf.predict(X[test_idx]))
            clf_proba_all.append(clf.predict_proba(X[test_idx])[:, 1])

        # Ensemble: average predictions
        reg_ensemble = np.mean(reg_preds_all, axis=0)
        clf_ensemble_proba = np.mean(clf_proba_all, axis=0)
        clf_ensemble = (clf_ensemble_proba > 0.5).astype(int)

        reg_acc = np.mean((reg_ensemble > 0) == (y_ret[test_idx] > 0))
        clf_acc = accuracy_score(y_dir[test_idx], clf_ensemble)
        reg_fold_accs.append(reg_acc)
        clf_fold_accs.append(clf_acc)

        print(f"  Fold {fold_i}: Reg Dir={reg_acc:.2%}, Clf={clf_acc:.2%} "
              f"(XGB={np.mean(clf_preds_all[0]==y_dir[test_idx]):.0%}, "
              f"Cat={np.mean(clf_preds_all[1]==y_dir[test_idx]):.0%}, "
              f"LGB={np.mean(clf_preds_all[2]==y_dir[test_idx]):.0%})")

    return reg_fold_accs, clf_fold_accs


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("V2 MODEL: 3-Model Ensemble + Expanded Features")
    print("=" * 60)

    # Re-fetch data with Robusta
    print("\nRefreshing data (adding Robusta)...")
    import subprocess, sys
    subprocess.run([sys.executable, "fetch_data.py"], capture_output=True)

    print("Preparing dataset...")
    df, feature_cols = prepare_dataset(horizon=HORIZON)
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features")
    splits = walk_forward_split(df, purge_gap=HORIZON)
    print(f"Purged CV: {len(splits)} folds\n")

    # Show new features
    new_feats = [f for f in feature_cols if any(k in f for k in ["zscore", "extreme", "robusta", "brl_"])]
    print(f"New features added: {new_feats}\n")

    # --- Define model configs ---
    # XGBoost (use production-tuned params)
    with open(MODELS_DIR / "production_metadata.json") as f:
        meta = json.load(f)

    xgb_reg_params = {**meta["regression"]["params"], "early_stopping_rounds": 30, "random_state": 42}
    xgb_clf_params = {**meta["classification"]["params"], "eval_metric": "logloss",
                      "early_stopping_rounds": 30, "random_state": 42}

    # CatBoost
    cat_reg_params = {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 5, "random_seed": 42,
        "early_stopping_rounds": 30, "verbose": False,
    }
    cat_clf_params = {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 5, "random_seed": 42,
        "early_stopping_rounds": 30, "verbose": False,
        "eval_metric": "Logloss",
    }

    # LightGBM
    lgb_reg_params = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "verbose": -1,
        "callbacks": [lambda env: None],  # suppress warnings
    }
    lgb_clf_params = {
        **lgb_reg_params, "objective": "binary",
    }

    # Fix LightGBM callbacks for early stopping
    from lightgbm import early_stopping, log_evaluation
    lgb_reg_params["callbacks"] = [early_stopping(30), log_evaluation(-1)]
    lgb_clf_params["callbacks"] = [early_stopping(30), log_evaluation(-1)]

    reg_models = [
        ("xgboost", XGBRegressor, xgb_reg_params),
        ("catboost", CatBoostRegressor, cat_reg_params),
        ("lightgbm", LGBMRegressor, lgb_reg_params),
    ]
    clf_models = [
        ("xgboost", XGBClassifier, xgb_clf_params),
        ("catboost", CatBoostClassifier, cat_clf_params),
        ("lightgbm", LGBMClassifier, lgb_clf_params),
    ]

    print("Evaluating 3-model ensemble (XGBoost + CatBoost + LightGBM)...")
    reg_accs, clf_accs = evaluate_ensemble(df, feature_cols, splits, reg_models, clf_models)

    print(f"\n  ENSEMBLE REGRESSION  — Avg: {np.mean(reg_accs):.2%}, Std: {np.std(reg_accs):.2%}")
    print(f"  ENSEMBLE CLASSIFIER  — Avg: {np.mean(clf_accs):.2%}, Std: {np.std(clf_accs):.2%}")

    # --- Train final ensemble on maximum data ---
    print("\nTraining final ensemble models...")
    X = df[feature_cols].values
    last_train, last_test = splits[-1]

    final_models = {}
    for name, RegClass, reg_params in reg_models:
        reg = RegClass(**reg_params)
        if name == "catboost":
            reg.fit(X[last_train], df["target_return"].values[last_train],
                    eval_set=(X[last_test], df["target_return"].values[last_test]), verbose=False)
        elif name == "lightgbm":
            reg.fit(X[last_train], df["target_return"].values[last_train],
                    eval_set=[(X[last_test], df["target_return"].values[last_test])])
        else:
            reg.fit(X[last_train], df["target_return"].values[last_train],
                    eval_set=[(X[last_test], df["target_return"].values[last_test])], verbose=False)
        final_models[f"reg_{name}"] = reg

    for name, ClfClass, clf_params in clf_models:
        clf = ClfClass(**clf_params)
        if name == "catboost":
            clf.fit(X[last_train], df["target_direction"].values[last_train],
                    eval_set=(X[last_test], df["target_direction"].values[last_test]), verbose=False)
        elif name == "lightgbm":
            clf.fit(X[last_train], df["target_direction"].values[last_train],
                    eval_set=[(X[last_test], df["target_direction"].values[last_test])])
        else:
            clf.fit(X[last_train], df["target_direction"].values[last_train],
                    eval_set=[(X[last_test], df["target_direction"].values[last_test])], verbose=False)
        final_models[f"clf_{name}"] = clf

    # Save all models
    for name, model in final_models.items():
        if "catboost" in name:
            model.save_model(str(MODELS_DIR / f"v2_{name}.cbm"))
        else:
            joblib.dump(model, MODELS_DIR / f"v2_{name}.joblib")

    # Save metadata
    v2_meta = {
        "horizon": HORIZON,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "models": ["xgboost", "catboost", "lightgbm"],
        "regression_fold_accs": reg_accs,
        "classification_fold_accs": clf_accs,
        "regression_avg": float(np.mean(reg_accs)),
        "classification_avg": float(np.mean(clf_accs)),
    }
    with open(MODELS_DIR / "v2_metadata.json", "w") as f:
        json.dump(v2_meta, f, indent=2, default=str)

    # --- Current prediction ---
    print("\n" + "=" * 60)
    print("V2 ENSEMBLE 63-DAY PREDICTION")
    print("=" * 60)
    latest = df.iloc[[-1]]
    X_latest = latest[feature_cols].values

    reg_preds = [final_models[f"reg_{name}"].predict(X_latest)[0] for name in ["xgboost", "catboost", "lightgbm"]]
    clf_probas = [final_models[f"clf_{name}"].predict_proba(X_latest)[0][1] for name in ["xgboost", "catboost", "lightgbm"]]

    pred_return = float(np.mean(reg_preds))
    pred_proba_up = float(np.mean(clf_probas))
    pred_dir = 1 if pred_proba_up > 0.5 else 0
    confidence = max(pred_proba_up, 1 - pred_proba_up)
    current_price = float(latest["coffee_close"].values[0])
    predicted_price = current_price * (1 + pred_return)

    print(f"Date: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"")
    print(f"Individual model predictions:")
    for name, rp, cp in zip(["XGBoost", "CatBoost", "LightGBM"], reg_preds, clf_probas):
        print(f"  {name:<10}: return={rp:+.2%}, P(up)={cp:.1%}")
    print(f"")
    print(f"ENSEMBLE predicted return: {pred_return:+.2%}")
    print(f"ENSEMBLE predicted price:  ${predicted_price:.2f}")
    print(f"ENSEMBLE direction: {'UP' if pred_dir == 1 else 'DOWN'} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()
