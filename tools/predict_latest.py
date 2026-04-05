"""Get predictions from the absolute latest data point (no target needed)."""

import json
import sys
import subprocess
from pathlib import Path

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"
SOYBEANS_DIR = Path(__file__).parent.parent / "soybeans"
WHEAT_DIR = Path(__file__).parent.parent / "wheat"
COPPER_DIR = Path(__file__).parent.parent / "copper"


def predict_commodity(project_dir: Path, name: str, ticker: str, price_col: str):
    """Get prediction from the latest available data point."""
    script = f"""
import json, sys, os, joblib, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

# Load raw data without target so we can use all rows
df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = df.dropna()

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

latest = df.iloc[[-1]]
X = latest[feature_cols].values

pred_return = float(reg.predict(X)[0])
pred_dir = int(clf.predict(X)[0])
pred_proba = clf.predict_proba(X)[0]
confidence = float(pred_proba[pred_dir])
current_price = float(latest['{price_col}'].values[0])

result = {{
    'date': latest.index[0].strftime('%Y-%m-%d'),
    'price': current_price,
    'pred_return': pred_return,
    'pred_price': current_price * (1 + pred_return),
    'direction': 'UP' if pred_dir == 1 else 'DOWN',
    'confidence': confidence,
    'version': version,
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def main():
    print("=" * 60)
    print("LATEST PREDICTIONS (all available data)")
    print("=" * 60)

    for project_dir, name, ticker, price_col in [
        (COFFEE_DIR, "Coffee", "KC=F", "coffee_close"),
        (COCOA_DIR, "Cocoa", "CC=F", "cocoa_close"),
    ]:
        print(f"\n{name} ({ticker}):")
        r = predict_commodity(project_dir, name, ticker, price_col)
        if r:
            threshold = 0.70
            action = ("LONG" if r["direction"] == "UP" else "SHORT") if r["confidence"] >= threshold else "NO TRADE"

            print(f"  Data as of:    {r['date']}")
            print(f"  Current price: ${r['price']:.2f}")
            print(f"  Predicted:     ${r['pred_price']:.2f} ({r['pred_return']:+.2%})")
            print(f"  Direction:     {r['direction']} (confidence: {r['confidence']:.1%})")
            print(f"  Strategy:      {action}")

            if action != "NO TRADE":
                sl = 0.10
                tp = abs(r["pred_return"])
                if action == "LONG":
                    print(f"  Entry:         ${r['price']:.2f}")
                    print(f"  Take profit:   ${r['price'] * (1 + tp):.2f} ({tp:+.1%})")
                    print(f"  Stop loss:     ${r['price'] * (1 - sl):.2f} (-{sl:.0%})")
                else:
                    print(f"  Entry:         ${r['price']:.2f}")
                    print(f"  Take profit:   ${r['price'] * (1 - tp):.2f} ({-tp:+.1%})")
                    print(f"  Stop loss:     ${r['price'] * (1 + sl):.2f} (+{sl:.0%})")
        else:
            print("  Failed to load")


if __name__ == "__main__":
    main()
