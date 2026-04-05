"""Prediction Agent — generates daily predictions for all commodities.

Responsibilities:
- Load production models and latest data
- Generate predictions with confidence scores
- Validate feature availability before predicting
- Log all predictions for audit and drift detection

Usage:
    python -m agents.prediction              # predict all
    python -m agents.prediction coffee sugar  # predict specific
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .validation import check_data_freshness, check_model_files
from .log import setup_logging, log_event


logger = setup_logging("prediction")

PREDICTIONS_LOG = COMMODITIES_DIR / "logs" / "predictions.jsonl"


def predict_commodity(cfg: CommodityConfig) -> dict | None:
    """Generate prediction for a single commodity using subprocess isolation.

    Runs in a subprocess to avoid module import conflicts between commodities.
    """
    # Pre-flight checks
    freshness = check_data_freshness(cfg)
    if freshness["status"] == "stale":
        logger.warning(f"{cfg.name}: data is {freshness['age_days']} days old — prediction may be unreliable")

    model_check = check_model_files(cfg)
    if model_check["status"] == "missing":
        logger.error(f"{cfg.name}: model files missing — {model_check['issues']}")
        return None

    script = f"""
import json, sys, joblib, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = df.dropna()

with open('models/production_metadata.json') as f:
    meta = json.load(f)

reg = joblib.load('models/production_regressor.joblib')
clf = joblib.load('models/production_classifier.joblib')
feature_cols = [f for f in meta['features'] if f in df.columns]

missing = set(meta['features']) - set(feature_cols)
if missing:
    print(json.dumps({{"error": f"missing features: {{missing}}"}}))
    sys.exit(1)

latest = df.iloc[[-1]]
X = latest[feature_cols].values

pred_return = float(reg.predict(X)[0])
pred_dir = int(clf.predict(X)[0])
pred_proba = clf.predict_proba(X)[0]
confidence = float(pred_proba[pred_dir])
current_price = float(latest['{cfg.price_col}'].values[0])

result = {{
    'date': latest.index[0].strftime('%Y-%m-%d'),
    'price': round(current_price, 4),
    'pred_return': round(pred_return, 6),
    'pred_price': round(current_price * (1 + pred_return), 4),
    'direction': 'UP' if pred_dir == 1 else 'DOWN',
    'confidence': round(confidence, 4),
    'n_features': len(feature_cols),
    'horizon': meta.get('horizon', 63),
}}
print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"{cfg.name} prediction failed: {result.stderr[-300:]}")
            return None

        prediction = json.loads(result.stdout.strip().split("\n")[-1])
        prediction["commodity"] = cfg.name
        prediction["ticker"] = cfg.ticker
        prediction["threshold"] = cfg.confidence_threshold
        prediction["signal"] = prediction["confidence"] >= cfg.confidence_threshold

        return prediction

    except subprocess.TimeoutExpired:
        logger.error(f"{cfg.name} prediction timed out")
        return None
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"{cfg.name} prediction parse error: {e}")
        return None


def log_prediction(prediction: dict):
    """Append prediction to the audit log."""
    PREDICTIONS_LOG.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        **prediction,
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def predict_all(commodity_keys: list[str] = None) -> dict:
    """Generate predictions for all (or specified) commodities."""
    targets = commodity_keys or list(COMMODITIES.keys())
    predictions = {}

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            logger.warning(f"Unknown commodity: {key}")
            continue

        logger.info(f"Predicting {cfg.name}...")
        pred = predict_commodity(cfg)

        if pred:
            predictions[key] = pred
            log_prediction(pred)

            direction = pred["direction"]
            conf = pred["confidence"]
            signal = "SIGNAL" if pred["signal"] else "no trade"
            logger.info(f"  {cfg.name}: {direction} ({conf:.1%}) — {signal}")
        else:
            predictions[key] = None
            logger.warning(f"  {cfg.name}: prediction failed")

    return predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate commodity predictions")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    args = parser.parse_args()

    predictions = predict_all(args.commodities or None)

    print(f"\n{'='*60}")
    print("PREDICTIONS SUMMARY")
    print(f"{'='*60}")

    signals = []
    for key, pred in predictions.items():
        if pred is None:
            print(f"  {key:<12} FAILED")
            continue

        cfg = COMMODITIES[key]
        action = "NO TRADE"
        if pred["signal"]:
            action = "LONG" if pred["direction"] == "UP" else "SHORT"
            signals.append((cfg.name, action, pred["confidence"], pred["pred_return"]))

        print(f"  {cfg.name:<15} {pred['direction']:>4} {pred['confidence']:>6.1%}  "
              f"ret={pred['pred_return']:+.2%}  -> {action}")

    if signals:
        print(f"\nACTIVE SIGNALS ({len(signals)}):")
        for name, action, conf, ret in signals:
            print(f"  {action} {name} (confidence: {conf:.1%}, expected return: {ret:+.2%})")
    else:
        print("\nNo active signals.")


if __name__ == "__main__":
    main()
