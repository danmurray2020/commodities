"""V5: Per-commodity optimal horizon + best feature strategy.

Combines:
- Horizon scan results (126d for coffee/natgas/copper, 42d for sugar, 63d for soy/wheat)
- V3 features for coffee/cocoa/sugar (absolute features helped)
- V4 stable features for natgas/wheat (removing absolute features helped)
- scale_pos_weight tuning for all
- Strict OOS validation
"""

import json, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

COMMODITIES = [
    {"name": "Coffee", "dir": Path(__file__).parent.parent / "coffee",
     "price_col": "coffee_close", "ticker": "KC=F",
     "horizon": 126, "use_stable_only": False, "stop_loss": 0.10},
    {"name": "Cocoa", "dir": Path(__file__).parent.parent / "chocolate",
     "price_col": "cocoa_close", "ticker": "CC=F",
     "horizon": 63, "use_stable_only": False, "stop_loss": 0.15},
    {"name": "Sugar", "dir": Path(__file__).parent.parent / "sugar",
     "price_col": "sugar_close", "ticker": "SB=F",
     "horizon": 42, "use_stable_only": False, "stop_loss": 0.10},
    {"name": "NatGas", "dir": Path(__file__).parent.parent / "natgas",
     "price_col": "natgas_close", "ticker": "NG=F",
     "horizon": 126, "use_stable_only": True, "stop_loss": 0.15},
    {"name": "Soybeans", "dir": Path(__file__).parent.parent / "soybeans",
     "price_col": "soybeans_close", "ticker": "ZS=F",
     "horizon": 63, "use_stable_only": False, "stop_loss": 0.10},
    {"name": "Wheat", "dir": Path(__file__).parent.parent / "wheat",
     "price_col": "wheat_close", "ticker": "ZW=F",
     "horizon": 63, "use_stable_only": True, "stop_loss": 0.10},
    {"name": "Copper", "dir": Path(__file__).parent.parent / "copper",
     "price_col": "copper_close", "ticker": "HG=F",
     "horizon": 126, "use_stable_only": False, "stop_loss": 0.15},
]


def retrain(commodity):
    name = commodity["name"]
    price_col = commodity["price_col"]
    horizon = commodity["horizon"]
    stable_only = commodity["use_stable_only"]
    sl = commodity["stop_loss"]
    oos_split = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    stable_filter = """
exclude_starts = ["sma_5","sma_10","sma_21","sma_50","sma_200","price_lag_","bb_high","bb_low","atr_",
    "usd_index","crude_oil","sugar","coffee","cocoa","corn","wheat","soybeans","sp500",
    "heating_oil","coal","cot_open_interest","cot_noncomm_long","cot_noncomm_short",
    "cot_comm_long","cot_comm_short","brl_usd","inr_usd","cny_usd","ghs_usd",
    "robusta","iron_ore","day_of_week"]
feature_cols = []
for feat in all_features:
    excluded = any(feat == es or feat.startswith(es) for es in exclude_starts)
    if not excluded:
        feature_cols.append(feat)
""" if stable_only else "feature_cols = all_features"

    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor
import optuna

sys.path.insert(0, ".")
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

df = pd.read_csv("data/combined_features.csv", index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col="{price_col}", horizon={horizon})
df = df.dropna()

all_features = [c for c in df.columns if c not in
    ["{price_col}","Open","High","Low","Volume","target_return","target_direction"]]

{stable_filter}

X_full = df[feature_cols].values
y_dir = df["target_direction"].values
y_ret = df["target_return"].values
n = len(X_full)

# Feature selection
purge = {horizon}
test_size = 63
min_train = 504
splits = []
for i in range(5):
    te = n - i*test_size; ts = te-test_size; tr = ts-purge
    if tr < min_train: break
    splits.append((tr, ts, te))
splits.reverse()

print(f"Feature selection on {{len(feature_cols)}} features...", file=sys.stderr)
all_imp = np.zeros((len(splits), len(feature_cols)))
for fi, (tr, ts, te) in enumerate(splits):
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=10, gamma=1.0,
        eval_metric="logloss", early_stopping_rounds=30, random_state=42)
    m.fit(X_full[:tr], y_dir[:tr], eval_set=[(X_full[ts:te], y_dir[ts:te])], verbose=False)
    r = permutation_importance(m, X_full[ts:te], y_dir[ts:te], n_repeats=10, random_state=42, scoring="accuracy")
    all_imp[fi] = r.importances_mean

mean_imp = all_imp.mean(axis=0)
ranking = sorted(zip(feature_cols, mean_imp), key=lambda x: x[1], reverse=True)
selected = [f for f, imp in ranking if imp > 0]
if len(selected) < 6:
    selected = [f for f, _ in ranking[:15]]

print(f"Selected {{len(selected)}} features", file=sys.stderr)
for f, imp in ranking[:5]:
    print(f"  {{f}}: {{imp:.4f}}", file=sys.stderr)

X = df[selected].values

# Optuna
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

# Strict OOS evaluation
train_mask = df.index < "{oos_split}"
test_mask = df.index >= "{oos_split}"
X_train, X_test = X[train_mask], X[test_mask]
y_dir_train, y_dir_test = y_dir[train_mask], y_dir[test_mask]

clf = XGBClassifier(**best)
clf.fit(X_train, y_dir_train, eval_set=[(X_test[:63], y_dir_test[:63])], verbose=False)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
confidence = np.maximum(proba[:, 0], proba[:, 1])

oos_acc = float(accuracy_score(y_dir_test, pred))
hc_mask = confidence >= 0.75
hc_acc = float(accuracy_score(y_dir_test[hc_mask], pred[hc_mask])) if hc_mask.sum() > 5 else 0
hc_n = int(hc_mask.sum())

# CV accuracy
cv_accs = []
for tr, ts, te in splits:
    m = XGBClassifier(**best)
    m.fit(X[:tr], y_dir[:tr], eval_set=[(X[ts:te], y_dir[ts:te])], verbose=False)
    cv_accs.append(float(accuracy_score(y_dir[ts:te], m.predict(X[ts:te]))))

# Save
last_tr, last_ts, last_te = splits[-1]
final_clf = XGBClassifier(**best)
final_clf.fit(X[:last_tr], y_dir[:last_tr], eval_set=[(X[last_ts:last_te], y_dir[last_ts:last_te])], verbose=False)
joblib.dump(final_clf, "models/v5_production_classifier.joblib")

r_p = {{k:v for k,v in best.items() if k not in ["scale_pos_weight","eval_metric"]}}
final_reg = XGBRegressor(**r_p)
final_reg.fit(X[:last_tr], df["target_return"].values[:last_tr],
    eval_set=[(X[last_ts:last_te], df["target_return"].values[last_ts:last_te])], verbose=False)
joblib.dump(final_reg, "models/v5_production_regressor.joblib")

meta = {{
    "version": "v5", "commodity": "{name}", "ticker": "{commodity['ticker']}",
    "horizon": {horizon}, "features": selected, "n_features": len(selected),
    "purge_gap": {horizon},
    "strategy": {{"confidence_threshold": 0.75, "stop_loss_pct": {sl},
                  "take_profit_multiplier": 1.0, "max_hold_days": {horizon}, "allow_short": True}},
    "classification": {{
        "params": {{k:v for k,v in best.items() if k not in ["eval_metric","early_stopping_rounds"]}},
        "fold_accuracies": cv_accs, "avg_accuracy": float(np.mean(cv_accs)), "std_accuracy": float(np.std(cv_accs)),
    }},
    "regression": {{"params": {{k:v for k,v in r_p.items() if k != "early_stopping_rounds"}},
        "avg_accuracy": 0, "std_accuracy": 0, "fold_accuracies": []}},
}}
json.dump(meta, open("models/v5_production_metadata.json", "w"), indent=2, default=str)
import shutil
for sfx in ["classifier.joblib", "regressor.joblib", "metadata.json"]:
    shutil.copy2(f"models/v5_production_{{sfx}}", f"models/v2_production_{{sfx}}")

result = {{
    "horizon": {horizon},
    "n_features": len(selected),
    "cv_avg": round(float(np.mean(cv_accs)), 3),
    "oos_acc": round(oos_acc, 3),
    "oos_hc_acc": round(hc_acc, 3),
    "oos_hc_n": hc_n,
    "spw": round(best["scale_pos_weight"], 2),
    "top_features": [f for f, _ in ranking[:5]],
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(commodity["dir"]), timeout=900,
    )
    for line in result.stderr.strip().split("\n"):
        if line.strip():
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except:
        return None


def main():
    print("=" * 75)
    print("V5: OPTIMAL HORIZON + BEST FEATURE STRATEGY PER COMMODITY")
    print("=" * 75)

    oos_v3 = {"Coffee": 0.70, "Cocoa": 0.60, "Sugar": 0.79, "NatGas": 0.43,
              "Soybeans": 0.66, "Wheat": 0.65, "Copper": 0.63}

    results = {}
    for c in COMMODITIES:
        print(f"\n{'='*60}")
        print(f"  {c['name']} — horizon={c['horizon']}d, stable_only={c['use_stable_only']}")
        print(f"{'='*60}")
        r = retrain(c)
        if r:
            results[c["name"]] = r

    print(f"\n{'='*75}")
    print("V5 RESULTS (Strict OOS)")
    print(f"{'='*75}")
    print(f"{'Commodity':<12} {'Horizon':>8} {'V3 OOS':>8} {'V5 OOS':>8} {'V5 HC':>12} {'Feats':>6} {'Delta':>7}")
    print("-" * 65)
    for name, r in results.items():
        v3 = oos_v3.get(name, 0)
        delta = r["oos_acc"] - v3
        hc = f"{r['oos_hc_acc']:.0%}({r['oos_hc_n']})" if r["oos_hc_n"] > 5 else "n/a"
        marker = " +" if delta > 0.03 else " =" if delta > -0.03 else " -"
        print(f"{name:<12} {r['horizon']:>6}d {v3:>7.0%} {r['oos_acc']:>7.1%} {hc:>12} {r['n_features']:>6} {delta:>+6.1%}{marker}")


if __name__ == "__main__":
    main()
