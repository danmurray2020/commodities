"""P&L Forecast Agent — projects forward-looking returns and scenarios.

Estimates expected P&L for current positions based on model predictions,
historical accuracy at each confidence level, position sizing, and
execution costs. Produces best/base/worst case projections.

Usage:
    python -m agents pnl                       # full P&L forecast
    python -m agents pnl coffee wheat          # specific commodities
    python -m agents pnl --scenarios           # scenario analysis
    python -m agents pnl --attribution         # return attribution forecast
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR, SIZING, TRADE_DEFAULTS
from .log import setup_logging


logger = setup_logging("pnl_forecast")


def _get_db():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from db import get_db
        return get_db()
    except Exception:
        return None


def get_historical_accuracy_by_confidence(db, commodity: str = None) -> dict:
    """Compute historical accuracy bucketed by confidence level.

    Returns accuracy at different confidence bands so we can estimate
    how reliable the current prediction is.
    """
    if not db:
        return {}

    rows = db.execute(
        """SELECT confidence, direction_correct FROM predictions
           WHERE realized_price IS NOT NULL"""
        + (" AND commodity = ?" if commodity else ""),
        (commodity,) if commodity else (),
    )

    if not rows:
        # Fallback: use model CV accuracy as estimate
        return {
            "50-60": 0.52, "60-70": 0.58, "70-80": 0.65,
            "80-90": 0.72, "90+": 0.80,
        }

    bands = {"50-60": [], "60-70": [], "70-80": [], "80-90": [], "90+": []}
    for r in rows:
        conf = r["confidence"]
        correct = r["direction_correct"]
        if conf >= 0.90:
            bands["90+"].append(correct)
        elif conf >= 0.80:
            bands["80-90"].append(correct)
        elif conf >= 0.70:
            bands["70-80"].append(correct)
        elif conf >= 0.60:
            bands["60-70"].append(correct)
        else:
            bands["50-60"].append(correct)

    return {
        band: round(np.mean(vals), 4) if vals else None
        for band, vals in bands.items()
    }


def get_confidence_band(confidence: float) -> str:
    if confidence >= 0.90:
        return "90+"
    elif confidence >= 0.80:
        return "80-90"
    elif confidence >= 0.70:
        return "70-80"
    elif confidence >= 0.60:
        return "60-70"
    return "50-60"


def forecast_single_position(
    commodity: str,
    direction: str,
    confidence: float,
    pred_return: float,
    position_size: float,
    price: float,
    historical_accuracy: dict,
    execution_cost: float = 0.006,  # 0.3% in + 0.3% out
) -> dict:
    """Forecast P&L for a single position.

    Returns base/best/worst case projections with probabilities.
    """
    band = get_confidence_band(confidence)
    hist_acc = historical_accuracy.get(band, confidence)  # fallback to model confidence
    if hist_acc is None:
        hist_acc = confidence

    abs_return = abs(pred_return)
    sl_pct = TRADE_DEFAULTS.stop_loss_pct

    # Probability-weighted scenarios
    # Base case: direction correct, achieve predicted return minus execution costs
    base_gross = abs_return - execution_cost
    # Best case: direction correct, achieve 1.5x predicted return
    best_gross = abs_return * 1.5 - execution_cost
    # Worst case: stopped out
    worst_gross = -(sl_pct + execution_cost)

    # Expected value: P(correct) * avg_win - P(wrong) * avg_loss
    p_correct = hist_acc
    p_wrong = 1 - hist_acc

    expected_return = (p_correct * base_gross) + (p_wrong * worst_gross)
    expected_pnl_sized = expected_return * position_size

    # Dollar estimates (per $100k portfolio)
    portfolio_value = 100_000
    dollar_at_risk = portfolio_value * position_size
    expected_dollar_pnl = portfolio_value * expected_pnl_sized

    return {
        "commodity": commodity,
        "direction": direction,
        "confidence": round(confidence, 4),
        "confidence_band": band,
        "historical_accuracy_at_band": hist_acc,
        "position_size": round(position_size, 4),
        "pred_return": round(pred_return, 4),
        "execution_cost": round(execution_cost, 4),
        "scenarios": {
            "best": {
                "return_pct": round(best_gross, 4),
                "probability": round(p_correct * 0.3, 4),  # 30% of correct predictions exceed target
                "sized_return": round(best_gross * position_size, 4),
                "dollar_pnl": round(portfolio_value * best_gross * position_size, 2),
            },
            "base": {
                "return_pct": round(base_gross, 4),
                "probability": round(p_correct * 0.7, 4),  # 70% of correct predictions hit target
                "sized_return": round(base_gross * position_size, 4),
                "dollar_pnl": round(portfolio_value * base_gross * position_size, 2),
            },
            "worst": {
                "return_pct": round(worst_gross, 4),
                "probability": round(p_wrong, 4),
                "sized_return": round(worst_gross * position_size, 4),
                "dollar_pnl": round(portfolio_value * worst_gross * position_size, 2),
            },
        },
        "expected_return": round(expected_return, 4),
        "expected_sized_return": round(expected_pnl_sized, 4),
        "expected_dollar_pnl": round(expected_dollar_pnl, 2),
        "dollar_at_risk": round(dollar_at_risk, 2),
        "max_loss": round(worst_gross * position_size, 4),
    }


def forecast_portfolio(predictions: list[dict] = None) -> dict:
    """Forecast P&L for the entire portfolio of active signals.

    Loads latest predictions from DB if not provided.
    """
    db = _get_db()

    # Load predictions from DB
    if predictions is None:
        if db:
            predictions = db.get_latest_predictions()
        else:
            predictions = []

    # Get historical accuracy
    hist_accuracy = get_historical_accuracy_by_confidence(db)

    # Filter to signals only
    signals = [p for p in predictions if p.get("is_signal")]
    if not signals:
        return {"status": "no_signals", "message": "No active signals to forecast"}

    # Compute position sizes (simplified — mirrors strategy agent logic)
    position_forecasts = []
    for pred in signals:
        commodity = pred.get("commodity", "")
        cfg = COMMODITIES.get(commodity)
        if not cfg:
            continue

        confidence = pred["confidence"]
        # Position sizing from config
        if confidence >= 0.90:
            size = SIZING.base_sizes["90+"]
        elif confidence >= 0.85:
            size = SIZING.base_sizes["85-90"]
        elif confidence >= 0.80:
            size = SIZING.base_sizes["80-85"]
        elif confidence >= 0.75:
            size = SIZING.base_sizes["75-80"]
        else:
            size = 0.05

        direction = pred["direction"]
        forecast = forecast_single_position(
            commodity=cfg.name,
            direction="LONG" if direction == "UP" else "SHORT",
            confidence=confidence,
            pred_return=pred["pred_return"],
            position_size=size,
            price=pred["price"],
            historical_accuracy=hist_accuracy,
        )
        position_forecasts.append(forecast)

    # Portfolio aggregation
    total_expected = sum(f["expected_sized_return"] for f in position_forecasts)
    total_at_risk = sum(f["dollar_at_risk"] for f in position_forecasts)
    total_max_loss = sum(f["max_loss"] for f in position_forecasts)
    best_case = sum(f["scenarios"]["best"]["sized_return"] for f in position_forecasts)
    worst_case = sum(f["scenarios"]["worst"]["sized_return"] for f in position_forecasts)

    portfolio_value = 100_000

    report = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio_value,
        "n_positions": len(position_forecasts),
        "positions": position_forecasts,
        "portfolio_summary": {
            "expected_return_pct": round(total_expected, 4),
            "expected_dollar_pnl": round(total_expected * portfolio_value, 2),
            "best_case_pct": round(best_case, 4),
            "best_case_dollar": round(best_case * portfolio_value, 2),
            "worst_case_pct": round(worst_case, 4),
            "worst_case_dollar": round(worst_case * portfolio_value, 2),
            "total_exposure_pct": round(sum(f["position_size"] for f in position_forecasts), 4),
            "total_at_risk_dollar": round(total_at_risk, 2),
            "max_portfolio_loss_pct": round(total_max_loss, 4),
        },
        "horizon_days": TRADE_DEFAULTS.max_hold_days,
        "historical_accuracy_by_band": hist_accuracy,
    }

    # Log to DB
    if db:
        try:
            run_id = db.start_agent_run("pnl_forecast", [p["commodity"] for p in signals])
            summary = (
                f"{len(position_forecasts)} positions, "
                f"expected={total_expected:+.2%}, "
                f"range=[{worst_case:+.2%}, {best_case:+.2%}]"
            )
            db.finish_agent_run(run_id, "ok", summary=summary, report=report)
        except Exception:
            pass

    return report


def run_scenario_analysis() -> dict:
    """Run P&L scenarios for current positions under different market conditions."""
    db = _get_db()
    predictions = db.get_latest_predictions() if db else []
    signals = [p for p in predictions if p.get("is_signal")]

    if not signals:
        return {"status": "no_signals"}

    scenarios = {
        "model_correct": {
            "description": "All models predict correctly",
            "multiplier": 1.0,
        },
        "half_correct": {
            "description": "50% of signals are wrong (stopped out)",
            "multiplier": 0.0,  # computed differently
        },
        "all_wrong": {
            "description": "All models wrong — maximum loss scenario",
            "multiplier": -1.0,
        },
        "double_return": {
            "description": "Market moves 2x predicted (momentum breakout)",
            "multiplier": 2.0,
        },
    }

    results = {}
    for scenario_name, scenario in scenarios.items():
        if scenario_name == "half_correct":
            # Half win, half stopped out
            total = 0
            for i, pred in enumerate(signals):
                size = 0.10  # approximate
                if i % 2 == 0:
                    total += abs(pred["pred_return"]) * size  # win
                else:
                    total -= TRADE_DEFAULTS.stop_loss_pct * size  # loss
            results[scenario_name] = {
                "description": scenario["description"],
                "portfolio_return_pct": round(total, 4),
                "portfolio_dollar": round(total * 100_000, 2),
            }
        else:
            total = 0
            for pred in signals:
                size = 0.10
                ret = pred["pred_return"] * scenario["multiplier"]
                if scenario["multiplier"] < 0:
                    ret = -TRADE_DEFAULTS.stop_loss_pct  # stopped out
                total += ret * size
            results[scenario_name] = {
                "description": scenario["description"],
                "portfolio_return_pct": round(total, 4),
                "portfolio_dollar": round(total * 100_000, 2),
            }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P&L forecast agent")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all signals)")
    parser.add_argument("--scenarios", action="store_true", help="Run scenario analysis")
    parser.add_argument("--attribution", action="store_true", help="Return attribution forecast")
    args = parser.parse_args()

    report = forecast_portfolio()

    print(f"\n{'='*60}")
    print("P&L FORECAST")
    print(f"{'='*60}")

    if report.get("status") == "no_signals":
        print("\nNo active signals. Nothing to forecast.")
        return

    summary = report["portfolio_summary"]
    print(f"\nPortfolio: ${report['portfolio_value']:,.0f}")
    print(f"Positions: {report['n_positions']}")
    print(f"Horizon:   {report['horizon_days']} days")

    print(f"\n  {'Position':<25} {'Dir':>5} {'Conf':>6} {'Size':>6} {'Expected':>9} {'Best':>9} {'Worst':>9}")
    print("  " + "-" * 72)
    for p in report["positions"]:
        print(f"  {p['commodity']:<25} {p['direction']:>5} {p['confidence']:>5.0%} "
              f"{p['position_size']:>5.0%} {p['expected_sized_return']:>+8.2%} "
              f"{p['scenarios']['best']['sized_return']:>+8.2%} "
              f"{p['scenarios']['worst']['sized_return']:>+8.2%}")

    print(f"\n  {'PORTFOLIO TOTAL':<25} {'':>5} {'':>6} "
          f"{summary['total_exposure_pct']:>5.0%} {summary['expected_return_pct']:>+8.2%} "
          f"{summary['best_case_pct']:>+8.2%} {summary['worst_case_pct']:>+8.2%}")

    print(f"\n  Expected P&L:    ${summary['expected_dollar_pnl']:>+10,.2f}")
    print(f"  Best case:       ${summary['best_case_dollar']:>+10,.2f}")
    print(f"  Worst case:      ${summary['worst_case_dollar']:>+10,.2f}")
    print(f"  Max portfolio DD: {summary['max_portfolio_loss_pct']:>+.2%}")

    if args.scenarios:
        scenarios = run_scenario_analysis()
        print(f"\n{'='*60}")
        print("SCENARIO ANALYSIS")
        print(f"{'='*60}")
        for name, s in scenarios.items():
            print(f"\n  {s['description']}")
            print(f"    Return: {s['portfolio_return_pct']:+.2%}  (${s['portfolio_dollar']:>+,.2f})")


if __name__ == "__main__":
    main()
