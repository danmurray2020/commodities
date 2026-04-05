"""Refresh all data sources and generate a fresh prediction."""

import subprocess
import sys


def run(script: str):
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")


def main():
    # Refresh all data
    run("fetch_data.py")
    run("fetch_cot.py")
    run("fetch_weather.py")
    run("fetch_enso.py")

    print(f"\n{'='*60}")
    print("All data refreshed. Generating prediction...")
    print(f"{'='*60}")

    # Generate prediction with existing model
    from features import prepare_dataset
    from pathlib import Path
    import json
    import joblib

    MODELS_DIR = Path(__file__).parent / "models"

    # Load feature list from metadata (single source of truth)
    with open(MODELS_DIR / "production_metadata.json") as f:
        metadata = json.load(f)
    selected_features = metadata["features"]

    reg = joblib.load(MODELS_DIR / "production_regressor.joblib")
    clf = joblib.load(MODELS_DIR / "production_classifier.joblib")

    df, all_cols = prepare_dataset(horizon=63)
    feature_cols = [f for f in selected_features if f in all_cols]
    latest = df.iloc[[-1]]
    X = latest[feature_cols].values

    pred_return = float(reg.predict(X)[0])
    pred_dir = int(clf.predict(X)[0])
    pred_proba = clf.predict_proba(X)[0]
    current_price = float(latest["coffee_close"].values[0])
    predicted_price = current_price * (1 + pred_return)

    print(f"\nDate: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted 63-day return: {pred_return:+.2%}")
    print(f"Predicted price in ~3 months: ${predicted_price:.2f}")
    print(f"Direction: {'UP' if pred_dir == 1 else 'DOWN'} (confidence: {pred_proba[pred_dir]:.1%})")
    print(f"\nRun 'python app.py' to view the dashboard at http://localhost:5000")


if __name__ == "__main__":
    main()
