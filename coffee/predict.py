"""Generate predictions using trained models."""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from features import add_price_features


MODELS_DIR = Path(__file__).parent / "models"


def load_models() -> tuple:
    """Load trained models and metadata."""
    regressor = joblib.load(MODELS_DIR / "xgb_regressor.joblib")
    classifier = joblib.load(MODELS_DIR / "xgb_classifier.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return regressor, classifier, metadata


def predict_latest(csv_path: str = "data/combined_features.csv"):
    """Make a prediction for the most recent data point.

    Args:
        csv_path: Path to the combined features CSV.
    """
    regressor, classifier, metadata = load_models()
    feature_cols = metadata["feature_cols"]
    horizon = metadata["horizon"]

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_price_features(df)
    df = df.dropna()

    latest = df.iloc[[-1]]
    X = latest[feature_cols].values

    predicted_return = regressor.predict(X)[0]
    predicted_direction = classifier.predict(X)[0]
    direction_proba = classifier.predict_proba(X)[0]

    current_price = latest["coffee_close"].values[0]
    predicted_price = current_price * (1 + predicted_return)

    print(f"Date: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted {horizon}-day return: {predicted_return:+.2%}")
    print(f"Predicted price in {horizon} days: ${predicted_price:.2f}")
    print(f"Direction: {'UP' if predicted_direction == 1 else 'DOWN'} "
          f"(confidence: {max(direction_proba):.1%})")


if __name__ == "__main__":
    predict_latest()
