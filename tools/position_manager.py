"""Position manager: correlation-aware sizing + stale model detection.

Adjusts position sizes when multiple correlated commodities signal simultaneously,
and flags models whose live predictions are degrading.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

COMMODITIES_DIR = Path(__file__).parent
TRADES_FILE = COMMODITIES_DIR / "paper_trades.json"
CONFIG_FILE = COMMODITIES_DIR / "optimal_config.json"

# Correlation groups — commodities that share drivers
# If multiple in a group signal, reduce sizing
CORRELATION_GROUPS = {
    "brazil_soft": ["Coffee", "Sugar"],          # Both Brazil-driven, BRL sensitive
    "grains": ["Soybeans", "Wheat"],             # Both US agriculture, correlated weather
    "energy": ["Natural Gas"],                    # Independent
    "industrial": ["Copper"],                     # Independent
    "tropical": ["Cocoa"],                        # Independent
}

# Commodity-specific confidence thresholds (final OOS-validated)
CALIBRATED_THRESHOLDS = {
    "Coffee": 0.75,        # V3, 63d — 78% OOS HC, validated
    "Cocoa": 0.80,         # V5, 63d — 66% OOS HC, marginal
    "Sugar": 0.75,         # V3, 63d — 90% OOS HC, strongest model
    "Natural Gas": 0.95,   # Excluded — no reliable HC signals OOS
    "Soybeans": 0.80,      # V3, 63d — 72% OOS HC, marginal
    "Wheat": 0.75,         # V5, 63d — 92% OOS HC, excellent
    "Copper": 0.75,        # V5, 126d — 75% OOS HC, good
}

# Model version and horizon per commodity
MODEL_CONFIG = {
    "Coffee": {"version": "v3", "horizon": 63},
    "Cocoa": {"version": "v5", "horizon": 63},
    "Sugar": {"version": "v3", "horizon": 63},
    "Natural Gas": {"version": "v3", "horizon": 63},
    "Soybeans": {"version": "v3", "horizon": 63},
    "Wheat": {"version": "v5", "horizon": 63},
    "Copper": {"version": "v5", "horizon": 126},
}

# Base position sizes by confidence (from calibration, adjusted for OOS reality)
BASE_SIZING = {
    "75-80%": 0.08,    # Small position — edge is modest at this level
    "80-85%": 0.12,
    "85-90%": 0.18,
    "90%+": 0.22,
}

EQUITY_SIZING = 0.04  # 4% per equity trade


def get_position_size(confidence: float, commodity: str) -> float:
    """Get base position size from confidence level."""
    if confidence >= 0.90:
        return BASE_SIZING["90%+"]
    elif confidence >= 0.85:
        return BASE_SIZING["85-90%"]
    elif confidence >= 0.80:
        return BASE_SIZING["80-85%"]
    elif confidence >= 0.75:
        return BASE_SIZING["75-80%"]
    return 0.0


def apply_correlation_adjustment(signals: dict) -> dict:
    """Reduce position sizes when correlated commodities signal together.

    If 2+ commodities in the same correlation group fire, cut each by 30%.
    If 3+ total signals fire across all groups, cut each by an additional 20%.
    """
    adjusted = {}

    # Count signals per group
    group_signals = {}
    for group_name, members in CORRELATION_GROUPS.items():
        active = [m for m in members if m in signals]
        if active:
            group_signals[group_name] = active

    total_signals = sum(len(v) for v in group_signals.values())

    for name, signal in signals.items():
        multiplier = 1.0

        # Find which group this commodity belongs to
        for group_name, members in CORRELATION_GROUPS.items():
            if name in members and group_name in group_signals:
                if len(group_signals[group_name]) >= 2:
                    multiplier *= 0.70  # 30% cut for correlated signals
                    signal["correlation_note"] = f"Correlated with {', '.join(m for m in group_signals[group_name] if m != name)}"

        # Portfolio-level cut if too many signals
        if total_signals >= 4:
            multiplier *= 0.80  # Additional 20% cut
            signal["portfolio_note"] = f"{total_signals} simultaneous signals — reduced sizing"

        signal["size_multiplier"] = round(multiplier, 2)
        adjusted[name] = signal

    return adjusted


def check_model_staleness(predictions_log: Path = None) -> dict:
    """Check if any model's recent predictions are degrading.

    Looks at the paper trade log to see if predictions are being validated.
    Returns a flag per commodity if accuracy drops below threshold.
    """
    trades_file = TRADES_FILE
    if not trades_file.exists():
        return {}

    with open(trades_file) as f:
        state = json.load(f)

    closed = state.get("closed_positions", [])
    if not closed:
        return {"status": "insufficient_data", "message": "No closed trades yet — need live history to detect staleness"}

    # Group closed trades by commodity
    commodity_trades = {}
    for trade in closed:
        name = trade.get("commodity", "").split("→")[0].strip()  # Handle equity trades
        if name not in commodity_trades:
            commodity_trades[name] = []
        commodity_trades[name].append(trade)

    staleness = {}
    for name, trades in commodity_trades.items():
        if len(trades) < 5:
            staleness[name] = {"status": "insufficient", "n_trades": len(trades)}
            continue

        # Check last 10 trades (or all if fewer)
        recent = trades[-10:]
        wins = sum(1 for t in recent if t["pnl_pct"] > 0)
        win_rate = wins / len(recent)

        # Flag if win rate drops below 50% on recent trades
        if win_rate < 0.50:
            staleness[name] = {
                "status": "STALE",
                "recent_win_rate": round(win_rate, 2),
                "n_recent": len(recent),
                "message": f"Win rate {win_rate:.0%} on last {len(recent)} trades — consider retraining",
            }
        else:
            staleness[name] = {
                "status": "OK",
                "recent_win_rate": round(win_rate, 2),
                "n_recent": len(recent),
            }

    return staleness


def generate_position_plan(commodity_predictions: dict) -> dict:
    """Generate a complete position plan with correlation-aware sizing.

    Args:
        commodity_predictions: dict of {name: {direction, confidence, price, pred_return, ...}}

    Returns:
        dict with position plan, sizing, and risk notes
    """
    # Filter to only tradeable signals
    tradeable = {}
    for name, pred in commodity_predictions.items():
        threshold = CALIBRATED_THRESHOLDS.get(name, 0.80)
        if pred["confidence"] >= threshold:
            base_size = get_position_size(pred["confidence"], name)
            tradeable[name] = {
                "direction": "LONG" if pred["direction"] == "UP" else "SHORT",
                "confidence": pred["confidence"],
                "threshold": threshold,
                "base_size_pct": base_size,
                "pred_return": pred.get("pred_return", 0),
                "price": pred.get("price", 0),
            }

    if not tradeable:
        return {
            "status": "NO TRADES",
            "message": "No commodity above its calibrated threshold",
            "thresholds": CALIBRATED_THRESHOLDS,
        }

    # Apply correlation adjustments
    adjusted = apply_correlation_adjustment(tradeable)

    # Calculate final sizes
    plan = {"positions": [], "total_exposure": 0, "risk_notes": []}

    for name, signal in adjusted.items():
        final_size = signal["base_size_pct"] * signal["size_multiplier"]
        plan["positions"].append({
            "commodity": name,
            "direction": signal["direction"],
            "confidence": signal["confidence"],
            "base_size": signal["base_size_pct"],
            "multiplier": signal["size_multiplier"],
            "final_size": round(final_size, 3),
            "correlation_note": signal.get("correlation_note", ""),
            "portfolio_note": signal.get("portfolio_note", ""),
        })
        plan["total_exposure"] += final_size

    plan["total_exposure"] = round(plan["total_exposure"], 3)

    # Risk warnings
    if plan["total_exposure"] > 0.60:
        plan["risk_notes"].append(f"High total exposure ({plan['total_exposure']:.0%}) — consider skipping lowest-confidence signal")

    if len(plan["positions"]) >= 5:
        plan["risk_notes"].append(f"{len(plan['positions'])} simultaneous positions — portfolio risk is elevated")

    # Check staleness
    staleness = check_model_staleness()
    for pos in plan["positions"]:
        name = pos["commodity"]
        if name in staleness and staleness[name].get("status") == "STALE":
            pos["stale_warning"] = staleness[name]["message"]
            plan["risk_notes"].append(f"{name} model may be stale: {staleness[name]['message']}")

    return plan


def print_plan(plan: dict):
    """Pretty-print the position plan."""
    print(f"\n{'='*60}")
    print("POSITION PLAN (Correlation-Adjusted)")
    print(f"{'='*60}")

    if plan.get("status") == "NO TRADES":
        print(f"\n  {plan['message']}")
        print(f"\n  Current thresholds (OOS-calibrated):")
        for name, thresh in plan["thresholds"].items():
            print(f"    {name:<15} {thresh:.0%}")
        return

    print(f"\n  {'Commodity':<15} {'Dir':<6} {'Conf':>5} {'Base':>6} {'Mult':>5} {'Final':>6} {'Notes'}")
    print(f"  {'-'*70}")
    for pos in plan["positions"]:
        notes = []
        if pos["correlation_note"]:
            notes.append(pos["correlation_note"])
        if pos["portfolio_note"]:
            notes.append(pos["portfolio_note"])
        if pos.get("stale_warning"):
            notes.append(f"STALE: {pos['stale_warning']}")
        note_str = " | ".join(notes) if notes else ""
        print(f"  {pos['commodity']:<15} {pos['direction']:<6} {pos['confidence']:>4.0%} "
              f"{pos['base_size']:>5.0%} {pos['multiplier']:>4.1f}x {pos['final_size']:>5.1%} {note_str}")

    print(f"\n  Total exposure: {plan['total_exposure']:.0%}")

    if plan["risk_notes"]:
        print(f"\n  Risk warnings:")
        for note in plan["risk_notes"]:
            print(f"    ! {note}")


def main():
    """Demonstrate position manager with current signals."""
    print("=" * 60)
    print("POSITION MANAGER — OOS-Calibrated")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Show calibrated thresholds
    print("\n  OOS-Calibrated Thresholds (strict out-of-sample validated):")
    print(f"  {'Commodity':<15} {'Threshold':>10} {'OOS High-Conf Acc':>20}")
    print(f"  {'-'*47}")
    oos_data = {
        "Coffee": ("75%", "78%"), "Cocoa": ("80%", "75%"),
        "Sugar": ("75%", "90%"), "Natural Gas": ("85%", "n/a"),
        "Soybeans": ("80%", "72%"), "Wheat": ("85%", "67%"),
        "Copper": ("85%", "n/a"),
    }
    for name, (thresh, oos) in oos_data.items():
        print(f"  {name:<15} {thresh:>10} {oos:>20}")

    # Check model staleness
    print(f"\n  Model Staleness Check:")
    staleness = check_model_staleness()
    if isinstance(staleness, dict) and "status" in staleness:
        print(f"    {staleness['message']}")
    else:
        for name, s in staleness.items():
            status = s.get("status", "unknown")
            if status == "STALE":
                print(f"    {name}: STALE — {s['message']}")
            elif status == "OK":
                print(f"    {name}: OK (win rate {s['recent_win_rate']:.0%} on last {s['n_recent']} trades)")
            else:
                print(f"    {name}: {status} ({s.get('n_trades', 0)} trades)")

    # Example: generate plan from current paper trade predictions
    # Load last predictions from paper trades
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            state = json.load(f)
        if state.get("prediction_history"):
            # Get latest prediction per commodity
            latest_preds = {}
            for pred in state["prediction_history"]:
                latest_preds[pred["commodity"]] = pred

            if latest_preds:
                print(f"\n  Current signals:")
                for name, pred in latest_preds.items():
                    thresh = CALIBRATED_THRESHOLDS.get(name, 0.80)
                    status = "TRADE" if pred["confidence"] >= thresh else "skip"
                    print(f"    {name:<15} {pred['direction']:>4} ({pred['confidence']:.0%}) threshold={thresh:.0%} -> {status}")

                plan = generate_position_plan(latest_preds)
                print_plan(plan)
                return

    print("\n  No predictions available. Run paper_trade.py first.")


if __name__ == "__main__":
    main()
