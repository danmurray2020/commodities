"""Platt scaling for probability calibration + quantile regression for prediction intervals.

Platt scaling fits a logistic regression on top of the raw XGBoost probabilities,
transforming overconfident/underconfident outputs into well-calibrated probabilities.

Quantile regression gives prediction intervals (e.g., 80% chance price is between $X and $Y).
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"


def run_calibration_and_intervals(project_dir: Path, name: str, price_col: str) -> dict | None:
    """Fit Platt scaling + quantile regression via subprocess."""
    script = f"""
import json, sys, joblib, numpy as np, pandas as pd, pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

# Load data
df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col='{price_col}', horizon=63)
df = df.dropna()

# Load model config
models_dir = 'models'
for meta_file in ['v2_production_metadata.json', 'production_metadata.json']:
    try:
        with open(f'{{models_dir}}/{{meta_file}}') as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue

feature_cols = [f for f in meta['features'] if f in df.columns]
X = df[feature_cols].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values

clf_params = meta['classification'].get('params', {{}})
reg_params = meta['regression'].get('params', {{}})

# --- Walk-forward to collect out-of-sample probabilities ---
min_train = 504
purge = 63
step = 21  # finer granularity for calibration

raw_probas = []
actual_dirs = []
actual_rets = []
reg_preds_all = []
indices = []

i = min_train
while i + purge < len(X):
    test_end = min(i + purge + step, len(X))
    test_start = i + purge

    if test_start >= len(X):
        break

    X_train = X[:i]
    y_dir_train = y_dir[:i]
    y_ret_train = y_ret[:i]

    X_test = X[test_start:test_end]
    y_dir_test = y_dir[test_start:test_end]
    y_ret_test = y_ret[test_start:test_end]

    # Train classifier
    c = XGBClassifier(**clf_params, eval_metric='logloss', early_stopping_rounds=30, random_state=42)
    c.fit(X_train, y_dir_train, eval_set=[(X_test, y_dir_test)], verbose=False)

    # Train regressor
    r = XGBRegressor(**reg_params, early_stopping_rounds=30, random_state=42)
    r.fit(X_train, y_ret_train, eval_set=[(X_test, y_ret_test)], verbose=False)

    probas = c.predict_proba(X_test)[:, 1]
    reg_preds = r.predict(X_test)

    raw_probas.extend(probas.tolist())
    actual_dirs.extend(y_dir_test.tolist())
    actual_rets.extend(y_ret_test.tolist())
    reg_preds_all.extend(reg_preds.tolist())
    indices.extend(range(test_start, test_end))

    i += step

raw_probas = np.array(raw_probas)
actual_dirs = np.array(actual_dirs)
actual_rets = np.array(actual_rets)
reg_preds_all = np.array(reg_preds_all)

# --- 1. Platt Scaling ---
# Fit logistic regression: raw_proba -> calibrated_proba
platt = LogisticRegression(C=1.0, solver='lbfgs')
platt.fit(raw_probas.reshape(-1, 1), actual_dirs)
calibrated_probas = platt.predict_proba(raw_probas.reshape(-1, 1))[:, 1]

# Compare calibration before/after
bins = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
cal_before = []
cal_after = []
for lo, hi in bins:
    # Before (raw)
    conf_raw = np.maximum(raw_probas, 1 - raw_probas)
    mask = (conf_raw >= lo) & (conf_raw < hi)
    if mask.sum() >= 5:
        pred_raw = (raw_probas[mask] > 0.5).astype(int)
        acc_raw = float((pred_raw == actual_dirs[mask]).mean())
        cal_before.append({{'bin': f'{{lo:.0%}}-{{hi:.0%}}', 'avg_conf': float(conf_raw[mask].mean()), 'accuracy': round(acc_raw, 3), 'n': int(mask.sum())}})

    # After (calibrated)
    conf_cal = np.maximum(calibrated_probas, 1 - calibrated_probas)
    mask = (conf_cal >= lo) & (conf_cal < hi)
    if mask.sum() >= 5:
        pred_cal = (calibrated_probas[mask] > 0.5).astype(int)
        acc_cal = float((pred_cal == actual_dirs[mask]).mean())
        cal_after.append({{'bin': f'{{lo:.0%}}-{{hi:.0%}}', 'avg_conf': float(conf_cal[mask].mean()), 'accuracy': round(acc_cal, 3), 'n': int(mask.sum())}})

# Save Platt scaler
pickle.dump(platt, open(f'{{models_dir}}/platt_scaler.pkl', 'wb'))

# Calibration error (ECE)
ece_before = np.mean([abs(b['accuracy'] - b['avg_conf']) for b in cal_before]) if cal_before else 0
ece_after = np.mean([abs(b['accuracy'] - b['avg_conf']) for b in cal_after]) if cal_after else 0

# --- 2. Quantile Regression for Prediction Intervals ---
# Train quantile regressors for 10th, 25th, 50th, 75th, 90th percentiles
quantile_predictions = {{}}
for alpha in [0.10, 0.25, 0.50, 0.75, 0.90]:
    qr = XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=alpha,
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        random_state=42,
    )
    # Train on all data except last 63 days
    train_end = len(X) - 63
    qr.fit(X[:train_end], y_ret[:train_end])
    joblib.dump(qr, f'{{models_dir}}/quantile_{{int(alpha*100)}}.joblib')

    # Predict on last available point
    latest_X = X[[-1]]
    quantile_predictions[f'q{{int(alpha*100)}}'] = float(qr.predict(latest_X)[0])

# Also predict with Platt-calibrated probability for the latest point
version = 'v2' if 'v2' in meta_file else 'v1'
reg_file = f'{{models_dir}}/v2_production_regressor.joblib' if version == 'v2' else f'{{models_dir}}/production_regressor.joblib'
clf_file = f'{{models_dir}}/v2_production_classifier.joblib' if version == 'v2' else f'{{models_dir}}/production_classifier.joblib'

final_reg = joblib.load(reg_file)
final_clf = joblib.load(clf_file)
latest_X = X[[-1]]

raw_proba_latest = final_clf.predict_proba(latest_X)[0][1]
calibrated_latest = float(platt.predict_proba(np.array([[raw_proba_latest]]))[0][1])
calibrated_conf = max(calibrated_latest, 1 - calibrated_latest)
calibrated_dir = 'UP' if calibrated_latest > 0.5 else 'DOWN'

pred_return = float(final_reg.predict(latest_X)[0])
current_price = float(df['{price_col}'].iloc[-1])

# Prediction intervals
price_q10 = current_price * (1 + quantile_predictions['q10'])
price_q25 = current_price * (1 + quantile_predictions['q25'])
price_q50 = current_price * (1 + quantile_predictions['q50'])
price_q75 = current_price * (1 + quantile_predictions['q75'])
price_q90 = current_price * (1 + quantile_predictions['q90'])

result = {{
    'date': df.index[-1].strftime('%Y-%m-%d'),
    'current_price': current_price,
    'platt_scaling': {{
        'raw_proba_up': float(raw_proba_latest),
        'calibrated_proba_up': calibrated_latest,
        'raw_confidence': float(max(raw_proba_latest, 1 - raw_proba_latest)),
        'calibrated_confidence': calibrated_conf,
        'calibrated_direction': calibrated_dir,
        'ece_before': round(ece_before, 3),
        'ece_after': round(ece_after, 3),
    }},
    'calibration_before': cal_before,
    'calibration_after': cal_after,
    'prediction_intervals': {{
        'point_estimate': round(pred_return, 4),
        'q10': round(quantile_predictions['q10'], 4),
        'q25': round(quantile_predictions['q25'], 4),
        'q50': round(quantile_predictions['q50'], 4),
        'q75': round(quantile_predictions['q75'], 4),
        'q90': round(quantile_predictions['q90'], 4),
        'price_q10': round(price_q10, 2),
        'price_q25': round(price_q25, 2),
        'price_q50': round(price_q50, 2),
        'price_q75': round(price_q75, 2),
        'price_q90': round(price_q90, 2),
    }},
    'quantile_predictions': quantile_predictions,
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


def print_results(name: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  {name} — CALIBRATED PROBABILITIES + PREDICTION INTERVALS")
    print(f"  {r['date']} | ${r['current_price']:.2f}")
    print(f"{'='*60}")

    ps = r["platt_scaling"]
    print(f"\n  --- Platt Scaling ---")
    print(f"  Raw P(up):        {ps['raw_proba_up']:.1%} (confidence: {ps['raw_confidence']:.1%})")
    print(f"  Calibrated P(up): {ps['calibrated_proba_up']:.1%} (confidence: {ps['calibrated_confidence']:.1%})")
    print(f"  Direction:        {ps['calibrated_direction']}")
    print(f"  ECE before:       {ps['ece_before']:.1%}")
    print(f"  ECE after:        {ps['ece_after']:.1%}")

    if ps['ece_after'] < ps['ece_before']:
        improvement = (1 - ps['ece_after'] / ps['ece_before']) * 100
        print(f"  Calibration improved by {improvement:.0f}%")

    # Calibration comparison
    print(f"\n  --- Calibration Before vs After ---")
    print(f"  {'Bin':<12} {'Before':>16} {'After':>16}")
    print(f"  {'':12} {'Conf→Acc':>16} {'Conf→Acc':>16}")
    print(f"  {'-'*46}")
    before = {b['bin']: b for b in r['calibration_before']}
    after = {a['bin']: a for a in r['calibration_after']}
    all_bins = sorted(set(list(before.keys()) + list(after.keys())))
    for b in all_bins:
        b_str = f"{before[b]['avg_conf']:.0%}→{before[b]['accuracy']:.0%}" if b in before else "—"
        a_str = f"{after[b]['avg_conf']:.0%}→{after[b]['accuracy']:.0%}" if b in after else "—"
        print(f"  {b:<12} {b_str:>16} {a_str:>16}")

    # Prediction intervals
    pi = r["prediction_intervals"]
    print(f"\n  --- 63-Day Prediction Intervals ---")
    print(f"  Point estimate:     {pi['point_estimate']:+.2%} (${r['current_price'] * (1 + pi['point_estimate']):.2f})")
    print(f"")
    print(f"  80% interval:       {pi['q10']:+.2%} to {pi['q90']:+.2%}")
    print(f"                      ${pi['price_q10']:.2f} to ${pi['price_q90']:.2f}")
    print(f"")
    print(f"  50% interval:       {pi['q25']:+.2%} to {pi['q75']:+.2%}")
    print(f"                      ${pi['price_q25']:.2f} to ${pi['price_q75']:.2f}")
    print(f"")
    print(f"  Median estimate:    {pi['q50']:+.2%} (${pi['price_q50']:.2f})")

    # Trading implication
    print(f"\n  --- Trading Implication ---")
    threshold = 0.75
    if ps['calibrated_confidence'] >= threshold:
        print(f"  CALIBRATED SIGNAL: {ps['calibrated_direction']} (calibrated confidence: {ps['calibrated_confidence']:.0%})")
        print(f"  Range: ${pi['price_q25']:.2f} - ${pi['price_q75']:.2f} (50% interval)")
    else:
        print(f"  NO SIGNAL — calibrated confidence {ps['calibrated_confidence']:.0%} < {threshold:.0%}")
        if ps['raw_confidence'] >= 0.70 and ps['calibrated_confidence'] < threshold:
            print(f"  Note: raw model said {ps['raw_confidence']:.0%} but Platt scaling corrected to {ps['calibrated_confidence']:.0%}")
            print(f"  This means the raw probability was overconfident — good that we're filtering it out")


def main():
    print("=" * 60)
    print("PLATT SCALING + PREDICTION INTERVALS")
    print("=" * 60)

    for project_dir, name, price_col in [
        (COFFEE_DIR, "Coffee", "coffee_close"),
        (COCOA_DIR, "Cocoa", "cocoa_close"),
    ]:
        print(f"\nProcessing {name}...")
        r = run_calibration_and_intervals(project_dir, name, price_col)
        if r:
            print_results(name, r)
        else:
            print(f"  {name}: Failed")


if __name__ == "__main__":
    main()
