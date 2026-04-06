"""Shared training utilities — walk-forward CV, evaluation metrics, feature filtering.

Centralizes methodology decisions so all commodity train.py files stay consistent.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── Volatility-adjusted target ─────────────────────────────────────────

def add_volatility_adjusted_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Add target_return_vol_adj = target_return / rolling_volatility.

    This normalizes returns by recent volatility, making predictions
    comparable across different volatility regimes. A 10% return during
    a 40-vol period is less surprising than 10% during a 10-vol period.

    Args:
        df: DataFrame with a 'target_return' column already computed.
        price_col: Name of the price column used to compute rolling vol.
        horizon: Prediction horizon in trading days.
        vol_window: Lookback window for rolling volatility (default 63 days).

    Returns:
        DataFrame with 'target_vol_adj' column added (original target unchanged).
    """
    df = df.copy()

    if "target_return" not in df.columns:
        raise ValueError("DataFrame must have 'target_return' column. Call build_target first.")

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    # Annualized rolling volatility
    daily_returns = df[price_col].pct_change()
    rolling_vol = daily_returns.rolling(vol_window).std() * np.sqrt(252)

    # Scale factor: vol * sqrt(horizon/252) gives expected vol over the horizon
    horizon_vol = rolling_vol * np.sqrt(horizon / 252)

    # Avoid division by zero — use NaN where vol is too small
    horizon_vol = horizon_vol.replace(0, np.nan)
    min_vol = 0.01  # floor at 1% annualized
    horizon_vol = horizon_vol.clip(lower=min_vol * np.sqrt(horizon / 252))

    df["target_vol_adj"] = df["target_return"] / horizon_vol

    return df


# ── Walk-forward CV with non-overlapping evaluation ──────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 252,     # ~1 year of trading days (was 63)
    min_train_size: int = 504,  # ~2 years minimum training
    purge_gap: int = 63,      # gap = prediction horizon
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward (expanding window) train/test indices.

    Uses 252-day test folds (vs previous 63) for more independent observations
    per fold. With horizon=63, each fold has ~4 non-overlapping observations.
    """
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


def subsample_non_overlapping(indices: np.ndarray, horizon: int = 63) -> np.ndarray:
    """Subsample indices to get non-overlapping observations.

    With horizon=63, consecutive targets share 62/63 days.
    This returns every `horizon`-th index to get independent observations.
    """
    return indices[::horizon]


# ── Evaluation metrics ───────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 63) -> dict:
    """Evaluate regression predictions with proper non-overlapping metrics.

    Returns both overlapping (all data) and non-overlapping (subsampled) metrics.
    The non-overlapping metrics are the trustworthy ones.
    """
    # Non-overlapping subsample for honest evaluation
    step = max(1, horizon)
    y_true_ind = y_true[::step]
    y_pred_ind = y_pred[::step]
    n_independent = len(y_true_ind)

    # Spearman rank correlation (primary metric)
    if n_independent >= 5:
        spearman_corr, spearman_p = spearmanr(y_true_ind, y_pred_ind)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = 0.0, 1.0
    else:
        spearman_corr, spearman_p = 0.0, 1.0

    # Direction accuracy on non-overlapping samples
    if n_independent > 0:
        dir_acc_ind = float(np.mean((y_pred_ind > 0) == (y_true_ind > 0)))
    else:
        dir_acc_ind = 0.5

    # Traditional metrics on all data (for backwards compat, but flagged as overlapping)
    dir_acc_all = float(np.mean((y_pred > 0) == (y_true > 0)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return {
        "spearman": round(spearman_corr, 4),
        "spearman_p": round(spearman_p, 4),
        "dir_acc_independent": round(dir_acc_ind, 4),
        "n_independent": n_independent,
        "dir_acc_overlapping": round(dir_acc_all, 4),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 63) -> dict:
    """Evaluate classification predictions with non-overlapping metrics."""
    step = max(1, horizon)
    y_true_ind = y_true[::step]
    y_pred_ind = y_pred[::step]
    n_independent = len(y_true_ind)

    if n_independent > 0:
        acc_ind = float(np.mean(y_pred_ind == y_true_ind))
    else:
        acc_ind = 0.5

    acc_all = float(np.mean(y_pred == y_true))

    return {
        "acc_independent": round(acc_ind, 4),
        "n_independent": n_independent,
        "acc_overlapping": round(acc_all, 4),
    }


# ── Feature stability filter ────────────────────────────────────────────

def filter_stable_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_return",
    min_correlation_consistency: float = 0.0,
) -> tuple[list[str], list[dict]]:
    """Remove features whose correlation with target flips sign between halves.

    A feature whose correlation goes from +0.3 in H1 to -0.2 in H2 is learning
    a non-stationary relationship that will hurt out-of-sample performance.

    Args:
        df: Full dataset with features and target.
        feature_cols: Candidate features to filter.
        target_col: Target column name.
        min_correlation_consistency: Minimum threshold — features where
            corr_h1 * corr_h2 < this value are dropped.

    Returns:
        Tuple of (stable_features, diagnostics).
    """
    midpoint = len(df) // 2
    h1 = df.iloc[:midpoint]
    h2 = df.iloc[midpoint:]
    target_h1 = h1[target_col]
    target_h2 = h2[target_col]

    stable = []
    unstable = []
    diagnostics = []

    for feat in feature_cols:
        if feat not in df.columns:
            continue

        corr_h1 = h1[feat].corr(target_h1)
        corr_h2 = h2[feat].corr(target_h2)

        # Check if correlation flips sign
        sign_consistent = (corr_h1 * corr_h2) >= min_correlation_consistency
        delta = abs(corr_h1 - corr_h2)

        diag = {
            "feature": feat,
            "corr_h1": round(corr_h1, 4) if not np.isnan(corr_h1) else 0,
            "corr_h2": round(corr_h2, 4) if not np.isnan(corr_h2) else 0,
            "delta": round(delta, 4),
            "sign_flip": not sign_consistent,
            "stable": sign_consistent,
        }
        diagnostics.append(diag)

        if sign_consistent:
            stable.append(feat)
        else:
            unstable.append(feat)

    return stable, diagnostics


# ── Optuna objective with Spearman ───────────────────────────────────────

def reg_objective_spearman(trial, X, y, splits, horizon=63):
    """Optuna objective for regressor using Spearman correlation."""
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

    from xgboost import XGBRegressor

    fold_scores = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(**params)
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        preds = model.predict(X[test_idx])

        # Evaluate on non-overlapping samples
        metrics = evaluate_predictions(y[test_idx], preds, horizon=horizon)
        fold_scores.append(metrics["spearman"])

    return np.mean(fold_scores) - 0.5 * np.std(fold_scores)


def clf_objective_spearman(trial, X, y, splits, horizon=63):
    """Optuna objective for classifier using non-overlapping accuracy."""
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

    from xgboost import XGBClassifier

    fold_scores = []
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
        preds = model.predict(X[test_idx])

        metrics = evaluate_classification(y[test_idx], preds, horizon=horizon)
        fold_scores.append(metrics["acc_independent"])

    return np.mean(fold_scores) - 0.5 * np.std(fold_scores)
