"""Confidence calibration analysis and dynamic threshold optimization.

Answers: Are the model's probability estimates honest? What's the optimal
confidence threshold? How should position sizing scale with confidence?
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"


def run_calibration(project_dir: Path, name: str, price_col: str) -> dict | None:
    """Run calibration analysis via subprocess."""
    script = f"""
import json, sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

# Load data with targets for calibration
df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col='{price_col}', horizon=63)
df = df.dropna()

# Load model metadata and models
models_dir = 'models'
for meta_file in ['v2_production_metadata.json', 'production_metadata.json']:
    try:
        with open(f'{{models_dir}}/{{meta_file}}') as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue

version = 'v2' if 'v2' in meta_file else 'v1'
reg_file = f'{{models_dir}}/v2_production_regressor.joblib' if version == 'v2' else f'{{models_dir}}/production_regressor.joblib'
clf_file = f'{{models_dir}}/v2_production_classifier.joblib' if version == 'v2' else f'{{models_dir}}/production_classifier.joblib'

reg = joblib.load(reg_file)
clf = joblib.load(clf_file)
feature_cols = [f for f in meta['features'] if f in df.columns]

X = df[feature_cols].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values

# Walk-forward predictions (expanding window, purged)
n = len(df)
min_train = 504
purge = 63
step = 63  # predict every 63 days

all_clf_proba = []
all_clf_pred = []
all_reg_pred = []
all_actual_dir = []
all_actual_ret = []
all_dates = []

i = min_train
while i + purge + 63 < n:
    train_end = i
    test_start = i + purge
    test_end = min(test_start + step, n)

    if test_end > n:
        break

    X_train = X[:train_end]
    y_train_dir = y_dir[:train_end]
    y_train_ret = y_ret[:train_end]

    X_test = X[test_start:test_end]
    y_test_dir = y_dir[test_start:test_end]
    y_test_ret = y_ret[test_start:test_end]

    # Train
    from xgboost import XGBClassifier, XGBRegressor
    clf_params = meta['classification'].get('params', {{}})
    reg_params = meta['regression'].get('params', {{}})

    c = XGBClassifier(**clf_params, eval_metric='logloss', early_stopping_rounds=30, random_state=42)
    c.fit(X_train, y_train_dir, eval_set=[(X_test, y_test_dir)], verbose=False)

    r = XGBRegressor(**reg_params, early_stopping_rounds=30, random_state=42)
    r.fit(X_train, y_train_ret, eval_set=[(X_test, y_test_ret)], verbose=False)

    # Predict
    clf_proba = c.predict_proba(X_test)[:, 1]  # P(up)
    clf_pred = c.predict(X_test)
    reg_pred = r.predict(X_test)

    all_clf_proba.extend(clf_proba.tolist())
    all_clf_pred.extend(clf_pred.tolist())
    all_reg_pred.extend(reg_pred.tolist())
    all_actual_dir.extend(y_test_dir.tolist())
    all_actual_ret.extend(y_test_ret.tolist())
    all_dates.extend([d.strftime('%Y-%m-%d') for d in df.index[test_start:test_end]])

    i += step

all_clf_proba = np.array(all_clf_proba)
all_clf_pred = np.array(all_clf_pred)
all_reg_pred = np.array(all_reg_pred)
all_actual_dir = np.array(all_actual_dir)
all_actual_ret = np.array(all_actual_ret)

# --- 1. Calibration bins ---
bins = [(0.40, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
        (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85),
        (0.85, 0.90), (0.90, 1.01)]
calibration = []
for lo, hi in bins:
    # Confidence = max(p, 1-p), direction from clf_pred
    confidence = np.maximum(all_clf_proba, 1 - all_clf_proba)
    mask = (confidence >= lo) & (confidence < hi)
    if mask.sum() >= 3:
        correct = (all_clf_pred[mask] == all_actual_dir[mask])
        avg_conf = float(confidence[mask].mean())
        actual_acc = float(correct.mean())
        avg_ret = float(all_actual_ret[mask].mean())
        n_samples = int(mask.sum())

        # Model-directed returns (positive if model was right about direction)
        directed_ret = np.where(
            all_clf_pred[mask] == 1,
            all_actual_ret[mask],
            -all_actual_ret[mask]
        )
        avg_directed_ret = float(directed_ret.mean())

        calibration.append({{
            'bin': f'{{lo:.0%}}-{{hi:.0%}}',
            'avg_confidence': round(avg_conf, 3),
            'actual_accuracy': round(actual_acc, 3),
            'n': n_samples,
            'avg_return': round(avg_ret, 4),
            'avg_directed_return': round(avg_directed_ret, 4),
            'calibration_error': round(actual_acc - avg_conf, 3),
        }})

# --- 2. Optimal threshold search ---
thresholds = np.arange(0.50, 0.95, 0.05)
threshold_results = []
confidence = np.maximum(all_clf_proba, 1 - all_clf_proba)

for thresh in thresholds:
    mask = confidence >= thresh
    if mask.sum() < 3:
        continue
    correct = (all_clf_pred[mask] == all_actual_dir[mask])
    win_rate = float(correct.mean())
    n_trades = int(mask.sum())

    directed_ret = np.where(
        all_clf_pred[mask] == 1,
        all_actual_ret[mask],
        -all_actual_ret[mask]
    )
    avg_pnl = float(directed_ret.mean())
    total_pnl = float(directed_ret.sum())

    wins = directed_ret[directed_ret > 0]
    losses = directed_ret[directed_ret <= 0]
    pf = float(abs(wins.sum() / losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 999

    # Sharpe-like metric
    if len(directed_ret) > 1 and np.std(directed_ret) > 0:
        sharpe_like = float(np.mean(directed_ret) / np.std(directed_ret) * np.sqrt(4))  # ~4 trades/yr
    else:
        sharpe_like = 0

    threshold_results.append({{
        'threshold': round(float(thresh), 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 3),
        'avg_pnl': round(avg_pnl, 4),
        'total_pnl': round(total_pnl, 4),
        'profit_factor': round(pf, 2),
        'sharpe_like': round(sharpe_like, 2),
    }})

# Find optimal threshold (maximize sharpe_like)
best_threshold = max(threshold_results, key=lambda x: x['sharpe_like']) if threshold_results else None

# --- 3. Model agreement analysis ---
# When regressor and classifier agree on direction
reg_direction = (all_reg_pred > 0).astype(int)
agree_mask = all_clf_pred == reg_direction
disagree_mask = ~agree_mask

agree_acc = float((all_clf_pred[agree_mask] == all_actual_dir[agree_mask]).mean()) if agree_mask.sum() > 0 else 0
disagree_acc = float((all_clf_pred[disagree_mask] == all_actual_dir[disagree_mask]).mean()) if disagree_mask.sum() > 0 else 0

agree_directed = np.where(all_clf_pred[agree_mask] == 1, all_actual_ret[agree_mask], -all_actual_ret[agree_mask])
disagree_directed = np.where(all_clf_pred[disagree_mask] == 1, all_actual_ret[disagree_mask], -all_actual_ret[disagree_mask])

agreement = {{
    'agree_pct': float(agree_mask.mean()),
    'agree_accuracy': round(agree_acc, 3),
    'agree_avg_pnl': round(float(agree_directed.mean()), 4) if len(agree_directed) > 0 else 0,
    'agree_n': int(agree_mask.sum()),
    'disagree_accuracy': round(disagree_acc, 3),
    'disagree_avg_pnl': round(float(disagree_directed.mean()), 4) if len(disagree_directed) > 0 else 0,
    'disagree_n': int(disagree_mask.sum()),
}}

# --- 4. Suggested position sizing ---
sizing = []
for cal in calibration:
    if cal['actual_accuracy'] > 0.5 and cal['avg_directed_return'] > 0:
        # Kelly: f = (p*b - q) / b where p=win_rate, b=avg_win/avg_loss, q=1-p
        # Simplified: use directed return as edge
        edge = cal['actual_accuracy'] - 0.5
        suggested_pct = min(0.25, max(0.05, edge * 2))  # 5-25% range
        sizing.append({{
            'confidence_range': cal['bin'],
            'actual_accuracy': cal['actual_accuracy'],
            'suggested_position_pct': round(suggested_pct, 2),
            'avg_directed_return': cal['avg_directed_return'],
        }})

result = {{
    'n_predictions': len(all_clf_proba),
    'calibration': calibration,
    'threshold_analysis': threshold_results,
    'optimal_threshold': best_threshold,
    'model_agreement': agreement,
    'position_sizing': sizing,
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=600,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-400:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  ERROR parsing output")
        return None


def print_calibration(name: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  {name} CONFIDENCE CALIBRATION")
    print(f"  ({r['n_predictions']} out-of-sample predictions)")
    print(f"{'='*60}")

    # Calibration table
    print(f"\n  --- Calibration: Is confidence honest? ---")
    print(f"  {'Bin':<12} {'Avg Conf':>9} {'Actual Acc':>11} {'Error':>7} {'N':>5} {'Avg PnL':>9}")
    print(f"  {'-'*57}")
    for c in r["calibration"]:
        err_marker = ""
        if c["calibration_error"] > 0.10:
            err_marker = " UNDER-CONF"
        elif c["calibration_error"] < -0.10:
            err_marker = " OVER-CONF"
        print(f"  {c['bin']:<12} {c['avg_confidence']:>8.0%} {c['actual_accuracy']:>10.0%} "
              f"{c['calibration_error']:>+6.0%} {c['n']:>5} {c['avg_directed_return']:>+8.2%}{err_marker}")

    # Threshold analysis
    print(f"\n  --- Optimal Threshold ---")
    print(f"  {'Threshold':>9} {'Trades':>7} {'Win Rate':>9} {'Avg PnL':>9} {'PF':>6} {'Sharpe':>7}")
    print(f"  {'-'*50}")
    for t in r["threshold_analysis"]:
        marker = " <-- BEST" if r["optimal_threshold"] and t["threshold"] == r["optimal_threshold"]["threshold"] else ""
        print(f"  {t['threshold']:>8.0%} {t['n_trades']:>7} {t['win_rate']:>8.0%} "
              f"{t['avg_pnl']:>+8.2%} {t['profit_factor']:>6.2f} {t['sharpe_like']:>6.2f}{marker}")

    if r["optimal_threshold"]:
        opt = r["optimal_threshold"]
        print(f"\n  RECOMMENDED THRESHOLD: {opt['threshold']:.0%} "
              f"(win rate: {opt['win_rate']:.0%}, Sharpe: {opt['sharpe_like']:.2f}, "
              f"{opt['n_trades']} trades)")

    # Model agreement
    print(f"\n  --- Model Agreement (Regressor + Classifier) ---")
    a = r["model_agreement"]
    print(f"  Models agree:    {a['agree_pct']:.0%} of the time")
    print(f"  When agreeing:   {a['agree_accuracy']:.0%} accuracy, {a['agree_avg_pnl']:+.2%} avg PnL ({a['agree_n']} samples)")
    print(f"  When disagreeing: {a['disagree_accuracy']:.0%} accuracy, {a['disagree_avg_pnl']:+.2%} avg PnL ({a['disagree_n']} samples)")

    if a["agree_accuracy"] > a["disagree_accuracy"] + 0.05:
        print(f"  -> Agreement is a useful meta-signal (+{a['agree_accuracy'] - a['disagree_accuracy']:.0%} accuracy boost)")
    else:
        print(f"  -> Agreement provides limited additional signal")

    # Position sizing
    if r["position_sizing"]:
        print(f"\n  --- Suggested Position Sizing ---")
        print(f"  {'Confidence':>14} {'Actual Acc':>11} {'Position %':>11} {'Avg PnL':>9}")
        print(f"  {'-'*48}")
        for s in r["position_sizing"]:
            print(f"  {s['confidence_range']:>14} {s['actual_accuracy']:>10.0%} "
                  f"{s['suggested_position_pct']:>10.0%} {s['avg_directed_return']:>+8.2%}")


def save_optimal_config(coffee_result: dict, cocoa_result: dict):
    """Save recommended configuration based on calibration."""
    config = {"generated": str(pd.Timestamp.now()), "commodities": {}}

    for name, r in [("coffee", coffee_result), ("cocoa", cocoa_result)]:
        if r is None:
            continue
        opt = r.get("optimal_threshold")
        agreement = r.get("model_agreement", {})

        commodity_config = {
            "optimal_threshold": opt["threshold"] if opt else 0.70,
            "require_model_agreement": agreement.get("agree_accuracy", 0) > agreement.get("disagree_accuracy", 0) + 0.10,
            "position_sizing": {},
        }
        for s in r.get("position_sizing", []):
            commodity_config["position_sizing"][s["confidence_range"]] = s["suggested_position_pct"]

        config["commodities"][name] = commodity_config

    config_path = Path(__file__).parent / "optimal_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nOptimal config saved to {config_path}")
    return config


def main():
    print("=" * 60)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 60)

    coffee_result = None
    cocoa_result = None

    for project_dir, name, price_col in [
        (COFFEE_DIR, "Coffee", "coffee_close"),
        (COCOA_DIR, "Cocoa", "cocoa_close"),
    ]:
        print(f"\nAnalyzing {name}...")
        r = run_calibration(project_dir, name, price_col)
        if r:
            print_calibration(name, r)
            if name == "Coffee":
                coffee_result = r
            else:
                cocoa_result = r
        else:
            print(f"  {name}: Failed to analyze")

    # Save optimal config
    if coffee_result or cocoa_result:
        config = save_optimal_config(coffee_result, cocoa_result)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, cfg in config.get("commodities", {}).items():
            print(f"  {name}: threshold={cfg['optimal_threshold']:.0%}, "
                  f"require_agreement={cfg['require_model_agreement']}")


if __name__ == "__main__":
    main()
