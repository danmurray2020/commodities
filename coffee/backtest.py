"""Simple backtesting to evaluate model performance over time."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from features import prepare_dataset
from train import walk_forward_split
from xgboost import XGBRegressor


OUTPUT_DIR = Path(__file__).parent / "data"


def run_backtest(horizon: int = 5):
    """Run a walk-forward backtest and plot results.

    Trains on expanding windows, predicts forward, and simulates a
    simple long/flat strategy based on predicted direction.

    Args:
        horizon: Trading days ahead to predict.
    """
    df, feature_cols = prepare_dataset(horizon=horizon)
    X = df[feature_cols].values
    y = df["target_return"].values
    prices = df["coffee_close"].values
    dates = df.index

    # Use more granular splits for backtesting
    splits = walk_forward_split(df, n_splits=10, test_size=63, min_train_size=252, purge_gap=horizon)

    all_preds = []
    all_actuals = []
    all_dates = []

    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, random_state=42,
        )
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        all_preds.extend(preds)
        all_actuals.extend(y[test_idx])
        all_dates.extend(dates[test_idx])

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_dates = np.array(all_dates)

    # Metrics
    direction_acc = np.mean((all_preds > 0) == (all_actuals > 0))
    correlation = np.corrcoef(all_preds, all_actuals)[0, 1]
    print(f"Backtest Results ({len(all_preds)} predictions):")
    print(f"  Direction accuracy: {direction_acc:.2%}")
    print(f"  Pred/actual correlation: {correlation:.4f}")

    # Simple strategy: go long when model predicts positive return, else flat
    strategy_returns = np.where(all_preds > 0, all_actuals, 0)
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_buyhold = np.cumprod(1 + all_actuals)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Cumulative returns
    axes[0].plot(all_dates, cumulative_strategy, label="Model Strategy (long/flat)", linewidth=1.5)
    axes[0].plot(all_dates, cumulative_buyhold, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    axes[0].set_title(f"Backtest: Model vs Buy & Hold ({horizon}-day horizon)")
    axes[0].legend()
    axes[0].set_ylabel("Cumulative Return")
    axes[0].grid(True, alpha=0.3)

    # Predicted vs actual scatter
    axes[1].scatter(all_actuals, all_preds, alpha=0.3, s=10)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Actual Return")
    axes[1].set_ylabel("Predicted Return")
    axes[1].set_title(f"Predicted vs Actual (corr={correlation:.3f})")
    axes[1].grid(True, alpha=0.3)

    # Rolling direction accuracy
    window = 63
    rolling_acc = pd.Series(
        ((all_preds > 0) == (all_actuals > 0)).astype(float)
    ).rolling(window).mean()
    axes[2].plot(all_dates, rolling_acc, linewidth=1.5)
    axes[2].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[2].set_title(f"Rolling {window}-day Direction Accuracy")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_results.png", dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR / 'backtest_results.png'}")
    plt.show()


if __name__ == "__main__":
    run_backtest()
