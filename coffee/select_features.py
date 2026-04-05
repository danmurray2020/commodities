"""Feature selection using permutation importance on walk-forward CV."""

import json
import numpy as np
from pathlib import Path

from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from features import prepare_dataset
from train import walk_forward_split

MODELS_DIR = Path(__file__).parent / "models"
HORIZON = 63


def run_selection(output_suffix: str = ""):
    """Run permutation importance feature selection.

    Args:
        output_suffix: Suffix for output file (e.g., '_v2' for expanded features).

    Returns:
        List of selected feature names.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Preparing dataset (horizon={HORIZON})...")
    df, feature_cols = prepare_dataset(horizon=HORIZON)
    X = df[feature_cols].values
    y = df["target_direction"].values
    splits = walk_forward_split(df, purge_gap=HORIZON)
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(splits)} folds\n")

    # Compute permutation importance on each test fold
    all_importances = np.zeros((len(splits), len(feature_cols)))

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold_i}...")
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=10, gamma=1.0,
            eval_metric="logloss", early_stopping_rounds=30, random_state=42,
        )
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )

        # Permutation importance on the TEST set (out-of-sample)
        result = permutation_importance(
            model, X[test_idx], y[test_idx],
            n_repeats=10, random_state=42, scoring="accuracy",
        )
        all_importances[fold_i] = result.importances_mean

    # Average across folds
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    # Rank features
    ranking = sorted(
        zip(feature_cols, mean_importance, std_importance),
        key=lambda x: x[1], reverse=True,
    )

    print(f"\n{'Feature':<40} {'Mean Imp':>10} {'Std':>10}")
    print("-" * 62)
    selected = []
    for feat, imp, std in ranking:
        marker = "  *" if imp > 0.0 else ""
        print(f"{feat:<40} {imp:>10.4f} {std:>10.4f}{marker}")
        if imp > 0.0:
            selected.append(feat)

    print(f"\n{len(selected)} features with positive permutation importance (marked with *)")
    print(f"Dropped: {len(feature_cols) - len(selected)} features")

    # Save selected features
    fname = f"feature_selection{output_suffix}.json"
    results = {
        "horizon": HORIZON,
        "total_features": len(feature_cols),
        "selected_features": selected,
        "n_selected": len(selected),
        "ranking": [{"feature": f, "importance": float(i), "std": float(s)} for f, i, s in ranking],
    }
    with open(MODELS_DIR / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {MODELS_DIR / fname}")
    return selected


def main():
    run_selection(output_suffix="_v2")


if __name__ == "__main__":
    main()
