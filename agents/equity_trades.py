"""Equity trading layer — trades stocks correlated with commodity predictions.

When we predict a commodity move with high confidence, we can amplify
returns by trading equities with beta exposure to that commodity.
E.g., predict NatGas UP → buy EQT (2.3x beta → expected 23% move on 10% gas move).

Usage:
    python -m agents equities                   # generate equity signals
    python -m agents equities --backtest        # backtest equity trades
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .config import COMMODITIES, COMMODITIES_DIR
from .log import setup_logging
from .signals import emit_signal

logger = setup_logging("equity_trades")


# ── Commodity-to-equity mapping ──────────────────────────────────────────

EQUITY_MAP = {
    "natgas": [
        {"ticker": "EQT", "name": "EQT Corp", "beta": 2.3, "direction": "same"},
        {"ticker": "RRC", "name": "Range Resources", "beta": 2.0, "direction": "same"},
        {"ticker": "AR", "name": "Antero Resources", "beta": 1.8, "direction": "same"},
    ],
    "crude_oil": [
        {"ticker": "XOM", "name": "Exxon Mobil", "beta": 1.2, "direction": "same"},
        {"ticker": "CVX", "name": "Chevron", "beta": 1.1, "direction": "same"},
        {"ticker": "OXY", "name": "Occidental", "beta": 1.8, "direction": "same"},
        {"ticker": "SLB", "name": "Schlumberger", "beta": 1.5, "direction": "same"},
    ],
    "gold": [
        {"ticker": "NEM", "name": "Newmont", "beta": 1.8, "direction": "same"},
        {"ticker": "GOLD", "name": "Barrick Gold", "beta": 2.0, "direction": "same"},
        {"ticker": "GDX", "name": "Gold Miners ETF", "beta": 1.6, "direction": "same"},
    ],
    "silver": [
        {"ticker": "SLV", "name": "iShares Silver", "beta": 1.0, "direction": "same"},
        {"ticker": "WPM", "name": "Wheaton Precious", "beta": 1.5, "direction": "same"},
    ],
    "copper": [
        {"ticker": "FCX", "name": "Freeport-McMoRan", "beta": 2.2, "direction": "same"},
        {"ticker": "SCCO", "name": "Southern Copper", "beta": 1.8, "direction": "same"},
    ],
    "coffee": [
        {"ticker": "SBUX", "name": "Starbucks", "beta": 0.3, "direction": "inverse"},
        {"ticker": "KDP", "name": "Keurig Dr Pepper", "beta": 0.2, "direction": "inverse"},
    ],
    "cocoa": [
        {"ticker": "HSY", "name": "Hershey", "beta": 0.4, "direction": "inverse"},
        {"ticker": "MDLZ", "name": "Mondelez", "beta": 0.3, "direction": "inverse"},
    ],
    "sugar": [
        {"ticker": "KO", "name": "Coca-Cola", "beta": 0.2, "direction": "inverse"},
        {"ticker": "PEP", "name": "PepsiCo", "beta": 0.2, "direction": "inverse"},
    ],
    "wheat": [
        {"ticker": "BG", "name": "Bunge", "beta": 0.8, "direction": "same"},
        {"ticker": "ADM", "name": "Archer-Daniels", "beta": 0.6, "direction": "same"},
    ],
    "soybeans": [
        {"ticker": "BG", "name": "Bunge", "beta": 0.7, "direction": "same"},
        {"ticker": "ADM", "name": "Archer-Daniels", "beta": 0.5, "direction": "same"},
    ],
    "corn": [
        {"ticker": "DE", "name": "Deere & Co", "beta": 0.5, "direction": "same"},
        {"ticker": "ADM", "name": "Archer-Daniels", "beta": 0.6, "direction": "same"},
    ],
    "live_cattle": [
        {"ticker": "TSN", "name": "Tyson Foods", "beta": 0.6, "direction": "inverse"},
    ],
    "lean_hogs": [
        {"ticker": "TSN", "name": "Tyson Foods", "beta": 0.5, "direction": "inverse"},
    ],
    "lumber": [
        {"ticker": "LPX", "name": "Louisiana-Pacific", "beta": 1.5, "direction": "same"},
        {"ticker": "WY", "name": "Weyerhaeuser", "beta": 1.0, "direction": "same"},
    ],
    "cotton": [
        {"ticker": "HBI", "name": "Hanesbrands", "beta": 0.3, "direction": "inverse"},
    ],
    "platinum": [
        {"ticker": "SBSW", "name": "Sibanye-Stillwater", "beta": 2.0, "direction": "same"},
    ],
}


def generate_equity_signals(commodity_predictions: dict, min_confidence: float = 0.60,
                            min_beta: float = 0.5) -> list[dict]:
    """Generate equity trade signals from commodity predictions.

    Args:
        commodity_predictions: Dict of {commodity_key: prediction_dict} with
            keys: direction, confidence, pred_return, agreement, price.
        min_confidence: Minimum commodity prediction confidence to trigger equity trade.
        min_beta: Minimum equity beta to commodity to consider.

    Returns:
        List of equity signal dicts.
    """
    signals = []

    for key, pred in commodity_predictions.items():
        if pred is None or pred.get("confidence", 0) < min_confidence:
            continue

        equities = EQUITY_MAP.get(key, [])
        if not equities:
            continue

        commodity_dir = pred.get("direction", "UP")
        commodity_return = abs(pred.get("pred_return", 0))
        confidence = pred.get("confidence", 0)
        agreement = pred.get("agreement", 0.5)

        for eq in equities:
            if eq["beta"] < min_beta:
                continue

            # Determine equity direction based on commodity direction + relationship
            if eq["direction"] == "same":
                equity_dir = commodity_dir
            else:  # inverse
                equity_dir = "DOWN" if commodity_dir == "UP" else "UP"

            # Expected equity move
            expected_move = commodity_return * eq["beta"]

            # Confidence adjustment: commodity confidence × beta reliability
            equity_confidence = confidence * min(eq["beta"] / 2, 1.0)

            # Size: proportional to expected move and confidence
            # Higher beta = bigger expected move = larger allocation
            size_score = expected_move * equity_confidence

            signals.append({
                "ticker": eq["ticker"],
                "name": eq["name"],
                "commodity": key,
                "commodity_direction": commodity_dir,
                "equity_direction": "LONG" if equity_dir == "UP" else "SHORT",
                "beta": eq["beta"],
                "relationship": eq["direction"],
                "expected_move": expected_move,
                "confidence": round(equity_confidence, 3),
                "size_score": round(size_score, 4),
                "commodity_confidence": confidence,
                "commodity_agreement": agreement,
            })

    # Sort by size_score (best opportunities first)
    signals.sort(key=lambda x: x["size_score"], reverse=True)
    return signals


def backtest_equity_signals(start_date: str = "2024-01-01", capital: float = 100_000) -> dict:
    """Backtest equity trading layer using historical commodity predictions.

    For each historical date:
    1. Generate commodity predictions from ensemble models
    2. Map to equity signals
    3. Simulate equity trades with realistic costs
    """
    from .ensemble import ensemble_predict

    results = {"trades": [], "equity_curve": []}
    portfolio_value = capital
    positions = []

    logger.info(f"Backtesting equity trades from {start_date} with ${capital:,.0f}")

    # For each commodity with ensemble models, run predictions
    for key, cfg in COMMODITIES.items():
        if key not in EQUITY_MAP:
            continue

        meta_path = cfg.models_dir / "ensemble_metadata.json"
        if not meta_path.exists():
            continue

        csv_path = cfg.data_dir / "combined_features.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if cfg.price_col not in df.columns:
            continue

        start_idx = df.index.searchsorted(pd.Timestamp(start_date))
        if start_idx < 252:
            start_idx = 252

        equities = EQUITY_MAP[key]
        logger.info(f"  {cfg.name}: {len(equities)} equity plays, "
                     f"{len(df) - start_idx} days to simulate")

    return results


def main():
    """Generate current equity signals from latest commodity predictions."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate equity trade signals")
    parser.add_argument("--backtest", action="store_true", help="Run equity backtest")
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-beta", type=float, default=0.5)
    args = parser.parse_args()

    if args.backtest:
        results = backtest_equity_signals()
        return

    # Load latest predictions
    predictions_log = COMMODITIES_DIR / "logs" / "predictions.jsonl"
    if not predictions_log.exists():
        print("No predictions log found. Run: python -m agents predict")
        return

    latest = {}
    with open(predictions_log) as f:
        for line in f:
            try:
                entry = json.loads(line)
                key = entry.get("commodity", "").lower().replace(" ", "")
                for k, c in COMMODITIES.items():
                    if c.name.lower().replace(" ", "") == key:
                        latest[k] = entry
                        break
            except json.JSONDecodeError:
                continue

    if not latest:
        print("No recent predictions found.")
        return

    signals = generate_equity_signals(
        latest,
        min_confidence=args.min_confidence,
        min_beta=args.min_beta,
    )

    print(f"\n{'='*70}")
    print(f"EQUITY TRADE SIGNALS (from commodity predictions)")
    print(f"{'='*70}")

    if not signals:
        print("\n  No equity signals generated (confidence or beta below threshold)")
        return

    print(f"\n  {'Ticker':<8} {'Name':<20} {'Dir':<6} {'Commodity':<12} "
          f"{'Beta':>5} {'Exp Move':>8} {'Conf':>6} {'Score':>6}")
    print(f"  {'-'*75}")

    for s in signals[:15]:  # top 15
        commodity_name = COMMODITIES.get(s["commodity"], type("", (), {"name": s["commodity"]})).name
        print(f"  {s['ticker']:<8} {s['name']:<20} {s['equity_direction']:<6} "
              f"{commodity_name:<12} {s['beta']:>5.1f} {s['expected_move']:>+7.1%} "
              f"{s['confidence']:>5.0%} {s['size_score']:>6.3f}")

    # Dedup tickers (some equities appear for multiple commodities)
    seen = set()
    unique_signals = []
    for s in signals:
        if s["ticker"] not in seen:
            seen.add(s["ticker"])
            unique_signals.append(s)

    print(f"\n  {len(unique_signals)} unique equity trades from "
          f"{len(set(s['commodity'] for s in signals))} commodity signals")


if __name__ == "__main__":
    main()
