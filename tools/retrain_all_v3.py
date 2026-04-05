"""V3 retrain: apply all cross-commodity lessons to every model.

Lessons applied:
1. scale_pos_weight in Optuna search (from sugar — biggest single improvement)
2. All trend features available (from sugar/natgas)
3. Permutation importance feature selection with purged CV
4. Stability-penalized Optuna objective (acc - 0.5*std)
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

COMMODITIES = [
    {"name": "Coffee", "dir": Path(__file__).parent.parent / "coffee", "price_col": "coffee_close", "ticker": "KC=F"},
    {"name": "Cocoa", "dir": Path(__file__).parent.parent / "chocolate", "price_col": "cocoa_close", "ticker": "CC=F"},
    {"name": "Sugar", "dir": Path(__file__).parent.parent / "sugar", "price_col": "sugar_close", "ticker": "SB=F"},
    {"name": "Natural Gas", "dir": Path(__file__).parent.parent / "natgas", "price_col": "natgas_close", "ticker": "NG=F"},
]

HORIZON = 63


def retrain_commodity(commodity: dict) -> dict | None:
    """Retrain a single commodity with all V3 improvements."""
    name = commodity["name"]
    project_dir = commodity["dir"]
    price_col = commodity["price_col"]

    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor
import optuna

sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

# Load data
df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col='{price_col}', horizon={HORIZON})
df = df.dropna()

exclude = {{'{price_col}', 'Open', 'High', 'Low', 'Volume', 'target_return', 'target_direction'}}
all_features = [c for c in df.columns if c not in exclude]
X_all = df[all_features].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values
n = len(X_all)

# Walk-forward splits with purge
purge, test_size, min_train = {HORIZON}, 63, 504
splits = []
for i in range(5):
    te = n - i*test_size; ts = te-test_size; tr = ts-purge
    if tr < min_train: break
    splits.append((tr, ts, te))
splits.reverse()

print(f'Dataset: {{n}} rows, {{len(all_features)}} features, {{len(splits)}} folds', file=sys.stderr)

# --- Step 1: Feature selection ---
print('Feature selection...', file=sys.stderr)
all_imp = np.zeros((len(splits), len(all_features)))
for fi, (tr, ts, te) in enumerate(splits):
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.6, min_child_weight=10, gamma=1.0,
                       eval_metric='logloss', early_stopping_rounds=30, random_state=42)
    m.fit(X_all[:tr], y_dir[:tr], eval_set=[(X_all[ts:te], y_dir[ts:te])], verbose=False)
    r = permutation_importance(m, X_all[ts:te], y_dir[ts:te], n_repeats=10, random_state=42, scoring='accuracy')
    all_imp[fi] = r.importances_mean

mean_imp = all_imp.mean(axis=0)
ranking = sorted(zip(all_features, mean_imp), key=lambda x: x[1], reverse=True)
selected = [f for f, imp in ranking if imp > 0]

# If too few selected, take top 15
if len(selected) < 8:
    selected = [f for f, _ in ranking[:15]]

print(f'Selected {{len(selected)}} features', file=sys.stderr)
for f, imp in ranking[:10]:
    print(f'  {{f}}: {{imp:.4f}}', file=sys.stderr)

feature_cols = selected
X = df[feature_cols].values

# --- Step 2: Optuna with scale_pos_weight (key sugar lesson) ---
print('Optuna tuning (200 trials)...', file=sys.stderr)

def objective(trial):
    params = {{
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.2, 5.0),
        'eval_metric': 'logloss', 'early_stopping_rounds': 30, 'random_state': 42,
    }}
    accs = []
    for tr, ts, te in splits:
        m = XGBClassifier(**params)
        m.fit(X[:tr], y_dir[:tr], eval_set=[(X[ts:te], y_dir[ts:te])], verbose=False)
        accs.append(float(accuracy_score(y_dir[ts:te], m.predict(X[ts:te]))))
    return np.mean(accs) - 0.5 * np.std(accs)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

best = study.best_params
best.update({{'eval_metric': 'logloss', 'early_stopping_rounds': 30, 'random_state': 42}})
print(f'Best score: {{study.best_value:.4f}}, scale_pos_weight={{best["scale_pos_weight"]:.2f}}', file=sys.stderr)

# --- Step 3: Evaluate ---
clf_accs = []
reg_accs = []
for fi, (tr, ts, te) in enumerate(splits):
    clf = XGBClassifier(**best)
    clf.fit(X[:tr], y_dir[:tr], eval_set=[(X[ts:te], y_dir[ts:te])], verbose=False)
    clf_acc = float(accuracy_score(y_dir[ts:te], clf.predict(X[ts:te])))
    clf_accs.append(clf_acc)

    reg_params = {{k:v for k,v in best.items() if k not in ['scale_pos_weight','eval_metric']}}
    reg = XGBRegressor(**reg_params)
    reg.fit(X[:tr], y_ret[:tr], eval_set=[(X[ts:te], y_ret[ts:te])], verbose=False)
    reg_acc = float(np.mean((reg.predict(X[ts:te]) > 0) == (y_ret[ts:te] > 0)))
    reg_accs.append(reg_acc)
    print(f'  Fold {{fi}}: Clf={{clf_acc:.0%}}, Reg={{reg_acc:.0%}}', file=sys.stderr)

print(f'  Clf avg: {{np.mean(clf_accs):.1%}} (std={{np.std(clf_accs):.1%}})', file=sys.stderr)
print(f'  Reg avg: {{np.mean(reg_accs):.1%}} (std={{np.std(reg_accs):.1%}})', file=sys.stderr)

# --- Step 4: Save ---
last_tr, last_ts, last_te = splits[-1]
final_clf = XGBClassifier(**best)
final_clf.fit(X[:last_tr], y_dir[:last_tr], eval_set=[(X[last_ts:last_te], y_dir[last_ts:last_te])], verbose=False)
joblib.dump(final_clf, 'models/v3_production_classifier.joblib')

reg_params = {{k:v for k,v in best.items() if k not in ['scale_pos_weight','eval_metric']}}
final_reg = XGBRegressor(**reg_params)
final_reg.fit(X[:last_tr], y_ret[:last_tr], eval_set=[(X[last_ts:last_te], y_ret[last_ts:last_te])], verbose=False)
joblib.dump(final_reg, 'models/v3_production_regressor.joblib')

meta = {{
    'version': 'v3',
    'commodity': '{name}',
    'ticker': '{commodity["ticker"]}',
    'horizon': {HORIZON},
    'features': feature_cols,
    'n_features': len(feature_cols),
    'purge_gap': {HORIZON},
    'strategy': {{'confidence_threshold': 0.75, 'stop_loss_pct': 0.15 if '{name}' == 'Natural Gas' else 0.10,
                  'take_profit_multiplier': 1.0, 'max_hold_days': 63, 'allow_short': True}},
    'classification': {{
        'params': {{k:v for k,v in best.items() if k not in ['eval_metric','early_stopping_rounds']}},
        'fold_accuracies': clf_accs,
        'avg_accuracy': float(np.mean(clf_accs)),
        'std_accuracy': float(np.std(clf_accs)),
    }},
    'regression': {{
        'params': {{k:v for k,v in reg_params.items() if k != 'early_stopping_rounds'}},
        'fold_accuracies': reg_accs,
        'avg_accuracy': float(np.mean(reg_accs)),
        'std_accuracy': float(np.std(reg_accs)),
    }},
}}
json.dump(meta, open('models/v3_production_metadata.json','w'), indent=2, default=str)

# Output summary as JSON
result = {{
    'clf_avg': float(np.mean(clf_accs)),
    'clf_std': float(np.std(clf_accs)),
    'clf_folds': clf_accs,
    'reg_avg': float(np.mean(reg_accs)),
    'scale_pos_weight': best['scale_pos_weight'],
    'n_features': len(feature_cols),
    'top_features': [f for f, _ in ranking[:5]],
}}
print(json.dumps(result))
"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=1200,
    )

    # Print stderr (progress)
    for line in result.stderr.strip().split("\n"):
        if line.strip():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None

    try:
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return data
    except (json.JSONDecodeError, IndexError):
        print(f"  ERROR parsing output")
        return None


def update_model_links(commodity: dict):
    """Update production model symlinks to point to v3."""
    models_dir = commodity["dir"] / "models"
    # The dashboard looks for v2_production or production metadata
    # Copy v3 as the new v2 (since dashboard loads v2 first)
    import shutil
    for suffix in ["classifier.joblib", "regressor.joblib", "metadata.json"]:
        v3_file = models_dir / f"v3_production_{suffix}"
        v2_file = models_dir / f"v2_production_{suffix}"
        if v3_file.exists():
            shutil.copy2(v3_file, v2_file)


def main():
    print("=" * 60)
    print(f"V3 RETRAIN ALL — Applying Cross-Commodity Lessons")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print("\nLessons being applied:")
    print("  1. scale_pos_weight in Optuna (from sugar)")
    print("  2. Trend features present (from sugar/natgas)")
    print("  3. Stability-penalized objective (acc - 0.5*std)")
    print("  4. Permutation importance feature selection")

    results = {}
    for commodity in COMMODITIES:
        data = retrain_commodity(commodity)
        if data:
            results[commodity["name"]] = data
            update_model_links(commodity)

    # Summary
    print(f"\n{'='*60}")
    print("V3 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Commodity':<15} {'Clf Avg':>8} {'Std':>6} {'Feats':>6} {'SPW':>6} {'Folds':>30}")
    print("-" * 75)
    for name, r in results.items():
        folds = ", ".join(f"{a:.0%}" for a in r["clf_folds"])
        print(f"{name:<15} {r['clf_avg']:>7.1%} {r['clf_std']:>5.1%} {r['n_features']:>6} "
              f"{r['scale_pos_weight']:>5.2f} {folds:>30}")

    print(f"\nTop features per commodity:")
    for name, r in results.items():
        print(f"  {name}: {', '.join(r['top_features'][:3])}")


if __name__ == "__main__":
    main()
