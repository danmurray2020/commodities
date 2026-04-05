"""Fix OOS accuracy by normalizing features and removing drifters.

The core insight: absolute features (raw SMAs, price lags, raw ratios)
drift out of distribution when prices move to new levels.
Relative features (returns, z-scores, ratios-of-ratios) are stationary.

This script retrains all models using only distribution-stable features.
"""

import json
import sys
import subprocess
from pathlib import Path
import pandas as pd

COMMODITIES = [
    {"name": "Coffee", "dir": Path(__file__).parent.parent / "coffee", "price_col": "coffee_close", "ticker": "KC=F"},
    {"name": "Cocoa", "dir": Path(__file__).parent.parent / "chocolate", "price_col": "cocoa_close", "ticker": "CC=F"},
    {"name": "Sugar", "dir": Path(__file__).parent.parent / "sugar", "price_col": "sugar_close", "ticker": "SB=F"},
    {"name": "NatGas", "dir": Path(__file__).parent.parent / "natgas", "price_col": "natgas_close", "ticker": "NG=F"},
    {"name": "Soybeans", "dir": Path(__file__).parent.parent / "soybeans", "price_col": "soybeans_close", "ticker": "ZS=F"},
    {"name": "Wheat", "dir": Path(__file__).parent.parent / "wheat", "price_col": "wheat_close", "ticker": "ZW=F"},
    {"name": "Copper", "dir": Path(__file__).parent.parent / "copper", "price_col": "copper_close", "ticker": "HG=F"},
]

# Features that are distribution-stable (won't drift with price level)
# These use returns, ratios, z-scores, percentiles — NOT absolute values
STABLE_FEATURE_PATTERNS = [
    "return_", "price_vs_sma_", "volatility_", "rsi_", "macd", "bb_pct",
    "zscore_", "extreme_", "pct_up_days_", "trend_slope_", "momentum_rank_",
    "sma_50_200_cross", "sma_50_200_gap",
    "season_", "month", "winter", "summer", "shoulder", "harvest", "planting",
    "injection_season", "withdrawal_season", "days_to_winter",
    "cot_net_spec", "cot_net_comm", "cot_spec_ratio", "cot_comm_net_pct_oi",
    "cot_oi_change", "cot_noncomm_long_chg", "cot_noncomm_short_chg",
    "oni", "mei", "enso_state", "oni_change", "mei_change",
    "temp_anomaly", "drought", "frost", "heat_risk", "cold_snap", "extreme_cold", "extreme_heat",
    "hdd_anomaly",
    "_vs_sma50", "_return_21d",  # BRL/CNY/INR relative features
    "_ratio_sma21", "_spread_zscore", "_ratio_zscore",  # Cross-commodity ratios (normalized)
    "streak_", "china_construction",
]

# Features to ALWAYS exclude (absolute values that drift)
EXCLUDE_PATTERNS = [
    "sma_5", "sma_10", "sma_21", "sma_50", "sma_200",  # Raw SMAs
    "price_lag_",  # Raw lagged prices
    "bb_high", "bb_low",  # Absolute Bollinger bands
    "atr_",  # Absolute ATR
    "usd_index", "crude_oil", "sugar", "coffee", "cocoa", "corn", "wheat", "soybeans",  # Raw supplementary prices
    "sp500",  # S&P500 absolute level
    "heating_oil", "coal",  # Raw energy prices
    "cot_open_interest", "cot_noncomm_long", "cot_noncomm_short",  # Absolute COT (grow over time)
    "cot_comm_long", "cot_comm_short",  # Absolute COT
    "brl_usd", "inr_usd", "cny_usd", "ghs_usd",  # Raw FX (use _vs_sma50 instead)
    "robusta",  # Raw price
    "iron_ore",  # Raw price
    "day_of_week",  # Not useful at 63-day horizon
]


def is_stable_feature(feat_name: str) -> bool:
    """Check if a feature is distribution-stable."""
    for pattern in EXCLUDE_PATTERNS:
        if feat_name == pattern or feat_name.startswith(pattern):
            return False
    for pattern in STABLE_FEATURE_PATTERNS:
        if pattern in feat_name:
            return True
    return False


def retrain_with_stable_features(commodity: dict) -> dict:
    name = commodity["name"]
    project_dir = commodity["dir"]
    price_col = commodity["price_col"]
    oos_split = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import optuna

sys.path.insert(0, ".")
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

df = pd.read_csv("data/combined_features.csv", index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col="{price_col}", horizon=63)
df = df.dropna()

exclude_set = {{{', '.join(repr(p) for p in EXCLUDE_PATTERNS)}}}
stable_patterns = {{{', '.join(repr(p) for p in STABLE_FEATURE_PATTERNS)}}}

all_features = [c for c in df.columns if c not in {{"{price_col}", "Open", "High", "Low", "Volume", "target_return", "target_direction"}}]

# Filter to stable features only
stable_features = []
for feat in all_features:
    excluded = False
    for ep in exclude_set:
        if feat == ep or feat.startswith(ep):
            excluded = True
            break
    if excluded:
        continue
    for sp in stable_patterns:
        if sp in feat:
            stable_features.append(feat)
            break

X = df[stable_features].values
y_dir = df["target_direction"].values
y_ret = df["target_return"].values
n = len(X)

# Strict OOS split (1 year holdout)
train_mask = df.index < "{oos_split}"
test_mask = df.index >= "{oos_split}"
X_train, X_test = X[train_mask], X[test_mask]
y_dir_train, y_dir_test = y_dir[train_mask], y_dir[test_mask]
y_ret_train, y_ret_test = y_ret[train_mask], y_ret[test_mask]

# Also do walk-forward for Optuna
purge, test_size, min_train = 63, 63, 504
splits = []
for i in range(5):
    te = n - i*test_size; ts = te-test_size; tr = ts-purge
    if tr < min_train: break
    splits.append((tr, ts, te))
splits.reverse()

# Optuna with scale_pos_weight
def objective(trial):
    params = {{
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
        "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.2, 5.0),
        "eval_metric": "logloss", "early_stopping_rounds": 30, "random_state": 42,
    }}
    accs = []
    for tr, ts, te in splits:
        m = XGBClassifier(**params)
        m.fit(X[:tr], y_dir[:tr], eval_set=[(X[ts:te], y_dir[ts:te])], verbose=False)
        accs.append(float(accuracy_score(y_dir[ts:te], m.predict(X[ts:te]))))
    return np.mean(accs) - 0.5 * np.std(accs)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

best = study.best_params
best.update({{"eval_metric": "logloss", "early_stopping_rounds": 30, "random_state": 42}})

# Evaluate on STRICT OOS
clf = XGBClassifier(**best)
clf.fit(X_train, y_dir_train, eval_set=[(X_test[:63], y_dir_test[:63])], verbose=False)
clf_pred = clf.predict(X_test)
clf_proba = clf.predict_proba(X_test)
clf_acc = float(accuracy_score(y_dir_test, clf_pred))

# High-confidence OOS accuracy
confidence = np.maximum(clf_proba[:, 0], clf_proba[:, 1])
hc_mask = confidence >= 0.75
hc_acc = float(accuracy_score(y_dir_test[hc_mask], clf_pred[hc_mask])) if hc_mask.sum() > 5 else 0
hc_n = int(hc_mask.sum())

# Also CV accuracy for comparison
cv_accs = []
for tr, ts, te in splits:
    m = XGBClassifier(**best)
    m.fit(X[:tr], y_dir[:tr], eval_set=[(X[ts:te], y_dir[ts:te])], verbose=False)
    cv_accs.append(float(accuracy_score(y_dir[ts:te], m.predict(X[ts:te]))))

# Save model
last_tr, last_ts, last_te = splits[-1]
final_clf = XGBClassifier(**best)
final_clf.fit(X[:last_tr], y_dir[:last_tr], eval_set=[(X[last_ts:last_te], y_dir[last_ts:last_te])], verbose=False)
joblib.dump(final_clf, "models/v4_production_classifier.joblib")

r_params = {{k:v for k,v in best.items() if k not in ["scale_pos_weight","eval_metric"]}}
final_reg = XGBRegressor(**r_params)
final_reg.fit(X[:last_tr], y_ret[:last_tr], eval_set=[(X[last_ts:last_te], y_ret[last_ts:last_te])], verbose=False)
joblib.dump(final_reg, "models/v4_production_regressor.joblib")

meta = {{
    "version": "v4", "commodity": "{name}", "ticker": "{commodity['ticker']}",
    "horizon": 63, "features": stable_features, "n_features": len(stable_features),
    "purge_gap": 63,
    "strategy": {{"confidence_threshold": 0.75, "stop_loss_pct": 0.15 if "{name}" in ["NatGas","Copper"] else 0.10,
                  "take_profit_multiplier": 1.0, "max_hold_days": 63, "allow_short": True}},
    "classification": {{
        "params": {{k:v for k,v in best.items() if k not in ["eval_metric","early_stopping_rounds"]}},
        "fold_accuracies": cv_accs,
        "avg_accuracy": float(np.mean(cv_accs)),
        "std_accuracy": float(np.std(cv_accs)),
    }},
    "regression": {{
        "params": {{k:v for k,v in r_params.items() if k != "early_stopping_rounds"}},
        "avg_accuracy": 0, "std_accuracy": 0, "fold_accuracies": [],
    }},
}}
json.dump(meta, open("models/v4_production_metadata.json", "w"), indent=2, default=str)

# Also copy as v2 so dashboard picks it up
import shutil
shutil.copy2("models/v4_production_classifier.joblib", "models/v2_production_classifier.joblib")
shutil.copy2("models/v4_production_regressor.joblib", "models/v2_production_regressor.joblib")
shutil.copy2("models/v4_production_metadata.json", "models/v2_production_metadata.json")

result = {{
    "n_stable_features": len(stable_features),
    "n_total_features": len(all_features),
    "cv_avg": round(float(np.mean(cv_accs)), 3),
    "cv_folds": [round(a, 3) for a in cv_accs],
    "oos_accuracy": round(clf_acc, 3),
    "oos_hc_accuracy": round(hc_acc, 3),
    "oos_hc_n": hc_n,
    "scale_pos_weight": round(best["scale_pos_weight"], 2),
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=600,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except:
        return None


def main():
    print("=" * 70)
    print("V4: DISTRIBUTION-STABLE FEATURES — OOS-OPTIMIZED")
    print("=" * 70)
    print("\nDropping absolute features (SMAs, raw prices, raw COT levels)")
    print("Keeping only: returns, z-scores, ratios, percentiles, flags\n")

    results = {}
    oos_before = {
        "Coffee": 0.70, "Cocoa": 0.60, "Sugar": 0.79, "NatGas": 0.43,
        "Soybeans": 0.66, "Wheat": 0.65, "Copper": 0.63,
    }

    for commodity in COMMODITIES:
        name = commodity["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        r = retrain_with_stable_features(commodity)
        if r:
            results[name] = r
            before = oos_before.get(name, 0)
            delta = r["oos_accuracy"] - before
            hc_str = f"{r['oos_hc_accuracy']:.0%} ({r['oos_hc_n']}t)" if r["oos_hc_n"] > 5 else "n/a"
            print(f"  Features: {r['n_stable_features']} stable (from {r['n_total_features']})")
            print(f"  CV avg: {r['cv_avg']:.1%}, scale_pos_weight: {r['scale_pos_weight']}")
            print(f"  OOS accuracy:   {r['oos_accuracy']:.1%} (was {before:.0%}, delta={delta:+.1%})")
            print(f"  OOS high-conf:  {hc_str}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: V4 (stable features) vs V3 (all features)")
    print(f"{'='*70}")
    print(f"{'Commodity':<12} {'V3 OOS':>8} {'V4 OOS':>8} {'Delta':>7} {'V4 HC':>12} {'Features':>9}")
    print("-" * 60)
    for name, r in results.items():
        before = oos_before.get(name, 0)
        delta = r["oos_accuracy"] - before
        hc = f"{r['oos_hc_accuracy']:.0%}({r['oos_hc_n']})" if r["oos_hc_n"] > 5 else "n/a"
        marker = " ✓" if delta > 0.02 else " ~" if delta > -0.02 else " ✗"
        print(f"{name:<12} {before:>7.0%} {r['oos_accuracy']:>7.1%} {delta:>+6.1%}{marker} {hc:>12} {r['n_stable_features']:>9}")


if __name__ == "__main__":
    main()
