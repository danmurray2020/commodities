"""Ensemble model combining ARIMA + XGBoost for coffee price prediction."""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from features import prepare_dataset
from train import walk_forward_split

OUTPUT_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

warnings.filterwarnings("ignore", category=UserWarning)


def fit_arima_forecast(train_prices: np.ndarray, horizon: int) -> float:
    """Fit ARIMA on price series and return predicted return over horizon."""
    try:
        model = ARIMA(train_prices, order=(2, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        predicted_price = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
        current_price = train_prices[-1]
        return (predicted_price / current_price) - 1
    except Exception:
        return 0.0  # fallback if ARIMA fails to converge


def run_ensemble_backtest(horizon: int = 5):
    """Run walk-forward backtest comparing XGBoost, ARIMA, and their ensemble."""
    df, feature_cols = prepare_dataset(horizon=horizon)
    X = df[feature_cols].values
    y = df["target_return"].values
    prices = df["coffee_close"].values
    dates = df.index

    splits = walk_forward_split(df, n_splits=8, test_size=63, min_train_size=252, purge_gap=horizon)

    results = {"date": [], "actual": [], "xgb_pred": [], "arima_pred": [], "ensemble_pred": []}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold_i + 1}/{len(splits)}...")

        # XGBoost
        val_size = min(63, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        xgb = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, random_state=42,
        )
        xgb.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        xgb_preds = xgb.predict(X[test_idx])

        # ARIMA - predict each test point
        arima_preds = []
        for i in test_idx:
            # Use all prices up to this point
            train_prices = prices[:i]
            arima_pred = fit_arima_forecast(train_prices, horizon)
            arima_preds.append(arima_pred)
        arima_preds = np.array(arima_preds)

        # Ensemble: weighted average (optimize weights on training fold)
        # Simple equal weight to start
        ensemble_preds = 0.6 * xgb_preds + 0.4 * arima_preds

        results["date"].extend(dates[test_idx])
        results["actual"].extend(y[test_idx])
        results["xgb_pred"].extend(xgb_preds)
        results["arima_pred"].extend(arima_preds)
        results["ensemble_pred"].extend(ensemble_preds)

    results_df = pd.DataFrame(results).set_index("date")

    # Compute metrics
    metrics = {}
    for name, col in [("xgb", "xgb_pred"), ("arima", "arima_pred"), ("ensemble", "ensemble_pred")]:
        preds = results_df[col].values
        actuals = results_df["actual"].values
        direction_acc = np.mean((preds > 0) == (actuals > 0))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        corr = np.corrcoef(preds, actuals)[0, 1]
        metrics[name] = {"direction_acc": direction_acc, "rmse": rmse, "correlation": corr}
        print(f"\n{name.upper()}: Direction Acc={direction_acc:.2%}, RMSE={rmse:.6f}, Corr={corr:.4f}")

    # Strategy comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for name, col in [("XGBoost", "xgb_pred"), ("ARIMA", "arima_pred"), ("Ensemble", "ensemble_pred")]:
        strategy_returns = np.where(results_df[col].values > 0, results_df["actual"].values, 0)
        cumulative = np.cumprod(1 + strategy_returns)
        axes[0].plot(results_df.index, cumulative, label=f"{name}", linewidth=1.5)

    buyhold = np.cumprod(1 + results_df["actual"].values)
    axes[0].plot(results_df.index, buyhold, label="Buy & Hold", linewidth=1.5, alpha=0.5, linestyle="--")
    axes[0].set_title(f"Ensemble Backtest: Cumulative Returns ({horizon}-day horizon)")
    axes[0].legend()
    axes[0].set_ylabel("Cumulative Return")
    axes[0].grid(True, alpha=0.3)

    # Rolling direction accuracy
    window = 63
    for name, col in [("XGBoost", "xgb_pred"), ("ARIMA", "arima_pred"), ("Ensemble", "ensemble_pred")]:
        rolling_acc = pd.Series(
            ((results_df[col].values > 0) == (results_df["actual"].values > 0)).astype(float)
        ).rolling(window).mean()
        axes[1].plot(results_df.index, rolling_acc.values, label=f"{name}", linewidth=1.5)

    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[1].set_title(f"Rolling {window}-day Direction Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_backtest.png", dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR / 'ensemble_backtest.png'}")

    # Save metrics
    with open(MODELS_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    plt.show()


if __name__ == "__main__":
    run_ensemble_backtest()
