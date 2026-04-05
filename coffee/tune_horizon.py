"""Evaluate model performance across multiple prediction horizons."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier

from features import prepare_dataset
from train import walk_forward_split

OUTPUT_DIR = Path(__file__).parent / "data"


def evaluate_horizon(horizon: int) -> dict:
    """Train and evaluate models for a given prediction horizon."""
    df, feature_cols = prepare_dataset(horizon=horizon)
    X = df[feature_cols].values
    splits = walk_forward_split(df, purge_gap=horizon)

    # Regression
    reg_direction_accs = []
    reg_rmses = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, random_state=42,
        )
        model.fit(
            X[fit_idx], df["target_return"].values[fit_idx],
            eval_set=[(X[val_idx], df["target_return"].values[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        actuals = df["target_return"].values[test_idx]
        reg_direction_accs.append(np.mean((preds > 0) == (actuals > 0)))
        reg_rmses.append(np.sqrt(np.mean((preds - actuals) ** 2)))

    # Classification
    clf_accs = []
    for train_idx, test_idx in splits:
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, random_state=42,
            eval_metric="logloss",
        )
        model.fit(
            X[fit_idx], df["target_direction"].values[fit_idx],
            eval_set=[(X[val_idx], df["target_direction"].values[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[test_idx])
        clf_accs.append(np.mean(preds == df["target_direction"].values[test_idx]))

    return {
        "horizon": horizon,
        "reg_direction_acc": np.mean(reg_direction_accs),
        "reg_rmse": np.mean(reg_rmses),
        "clf_accuracy": np.mean(clf_accs),
    }


def main():
    horizons = [1, 2, 3, 5, 10, 21, 42, 63]
    results = []

    for h in horizons:
        print(f"Evaluating horizon={h} days...")
        r = evaluate_horizon(h)
        results.append(r)
        print(f"  Regression direction acc: {r['reg_direction_acc']:.2%}, "
              f"Classification acc: {r['clf_accuracy']:.2%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "horizon_tuning.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df["horizon"], results_df["reg_direction_acc"], "o-", label="Regression direction acc")
    axes[0].plot(results_df["horizon"], results_df["clf_accuracy"], "s-", label="Classification accuracy")
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[0].set_xlabel("Prediction Horizon (trading days)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Direction Accuracy by Horizon")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df["horizon"], results_df["reg_rmse"], "o-", color="orange")
    axes[1].set_xlabel("Prediction Horizon (trading days)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Regression RMSE by Horizon")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "horizon_tuning.png", dpi=150)
    print(f"\nResults saved to {OUTPUT_DIR / 'horizon_tuning.csv'}")
    print(f"Plot saved to {OUTPUT_DIR / 'horizon_tuning.png'}")
    plt.show()


if __name__ == "__main__":
    main()
