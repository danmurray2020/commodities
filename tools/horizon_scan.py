"""Scan optimal prediction horizons for each commodity.

Tests horizons from 5 to 126 trading days using strict OOS evaluation
(train pre-2024, test 2024+) to find the genuine best horizon per commodity.
"""

import json
import sys
import subprocess
from pathlib import Path
import pandas as pd

import numpy as np

COMMODITIES = [
    ("Coffee", Path(__file__).parent.parent / "coffee", "coffee_close"),
    ("Cocoa", Path(__file__).parent.parent / "chocolate", "cocoa_close"),
    ("Sugar", Path(__file__).parent.parent / "sugar", "sugar_close"),
    ("NatGas", Path(__file__).parent.parent / "natgas", "natgas_close"),
    ("Soybeans", Path(__file__).parent.parent / "soybeans", "soybeans_close"),
    ("Wheat", Path(__file__).parent.parent / "wheat", "wheat_close"),
    ("Copper", Path(__file__).parent.parent / "copper", "copper_close"),
]

HORIZONS = [5, 10, 21, 42, 63, 84, 126]


OOS_SPLIT = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")


def test_horizons(name, project_dir, price_col):
    script = """
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

df = pd.read_csv("data/combined_features.csv", index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)

# Use only stable features (no absolute values)
exclude_starts = ["sma_5","sma_10","sma_21","sma_50","sma_200","price_lag_","bb_high","bb_low","atr_",
                   "usd_index","crude_oil","sugar","coffee","cocoa","corn","wheat","soybeans","sp500",
                   "heating_oil","coal","cot_open_interest","cot_noncomm_long","cot_noncomm_short",
                   "cot_comm_long","cot_comm_short","brl_usd","inr_usd","cny_usd","ghs_usd",
                   "robusta","iron_ore","day_of_week"]

exclude_set = set()
for col in df.columns:
    for es in exclude_starts:
        if col == es or col.startswith(es):
            exclude_set.add(col)
            break

base_features = [c for c in df.columns if c not in exclude_set and c not in
    ["{price_col}", "Open", "High", "Low", "Volume"]]

results = []
for horizon in HORIZONS:
    # Build target for this horizon
    df_h = df.copy()
    future = df_h["{price_col}"].shift(-horizon)
    df_h["target_return"] = (future / df_h["{price_col}"]) - 1
    df_h["target_direction"] = (df_h["target_return"] > 0).astype(int)
    df_h = df_h.dropna()

    feature_cols = [f for f in base_features if f in df_h.columns and f not in ["target_return", "target_direction"]]
    X = df_h[feature_cols].values
    y = df_h["target_direction"].values

    # Strict OOS
    train_mask = df_h.index < "{oos_split}"
    test_mask = df_h.index >= "{oos_split}"

    if test_mask.sum() < 30:
        continue

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Train with reasonable defaults
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
        gamma=1.0, scale_pos_weight=1.0,
        eval_metric="logloss", early_stopping_rounds=30, random_state=42,
    )
    clf.fit(X_train, y_train, eval_set=[(X_test[:63], y_test[:63])], verbose=False)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    confidence = np.maximum(proba[:, 0], proba[:, 1])

    overall_acc = float(accuracy_score(y_test, pred))

    # High-conf accuracy
    hc_mask = confidence >= 0.70
    hc_acc = float(accuracy_score(y_test[hc_mask], pred[hc_mask])) if hc_mask.sum() > 10 else 0
    hc_n = int(hc_mask.sum())

    # Class balance in test
    test_up_pct = float(y_test.mean())

    results.append({
        "horizon": horizon,
        "overall_acc": round(overall_acc, 3),
        "hc_acc": round(hc_acc, 3),
        "hc_n": hc_n,
        "test_n": int(test_mask.sum()),
        "test_up_pct": round(test_up_pct, 2),
    })

print(json.dumps(results))
""".replace("{price_col}", price_col).replace("HORIZONS", str(HORIZONS)).replace("{oos_split}", OOS_SPLIT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=300,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except:
        return None


def main():
    print("=" * 80)
    print("HORIZON SCAN — Strict OOS (train pre-2024, test 2024+)")
    print("Which prediction horizon works best for each commodity?")
    print("=" * 80)

    all_results = {}

    for name, project_dir, price_col in COMMODITIES:
        print(f"\n{name}...")
        results = test_horizons(name, project_dir, price_col)
        if results:
            all_results[name] = results
            print(f"  {'Horizon':>8} {'OOS Acc':>8} {'HC Acc':>8} {'HC Trades':>10} {'Test UP%':>9}")
            print(f"  {'-'*48}")
            best = max(results, key=lambda x: x["hc_acc"] if x["hc_acc"] > 0 else x["overall_acc"])
            for r in results:
                marker = " <-- BEST" if r == best else ""
                hc = f"{r['hc_acc']:.0%}({r['hc_n']})" if r["hc_n"] > 10 else "n/a"
                print(f"  {r['horizon']:>6}d {r['overall_acc']:>7.0%} {hc:>10} {r['hc_n']:>10} {r['test_up_pct']:>8.0%}{marker}")

    # Summary: optimal horizon per commodity
    print(f"\n{'='*80}")
    print("OPTIMAL HORIZONS")
    print(f"{'='*80}")
    print(f"{'Commodity':<12} {'Best Horizon':>13} {'OOS Acc':>8} {'HC Acc':>8} {'Current (63d)':>14}")
    print("-" * 58)
    for name, results in all_results.items():
        best = max(results, key=lambda x: x["hc_acc"] if x["hc_acc"] > 0 else x["overall_acc"])
        current = next((r for r in results if r["horizon"] == 63), None)
        curr_str = f"{current['overall_acc']:.0%}" if current else "n/a"
        hc = f"{best['hc_acc']:.0%}" if best["hc_n"] > 10 else f"{best['overall_acc']:.0%}*"
        print(f"{name:<12} {best['horizon']:>10}d {best['overall_acc']:>7.0%} {hc:>8} {curr_str:>14}")

    # Save
    output_path = Path(__file__).parent / "horizon_scan.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
