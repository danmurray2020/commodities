"""Strategy / Risk Agent — trade signals, position sizing, and risk management.

Responsibilities:
- Apply confidence thresholds and Kelly sizing to predictions
- Adjust for cross-commodity correlation
- Enforce portfolio-level exposure limits
- Generate actionable trade recommendations

Usage:
    python -m agents.strategy                 # from latest predictions
    python -m agents.strategy --from-file predictions.json
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (
    COMMODITIES, CORRELATION_GROUPS, SIZING, TRADE_DEFAULTS, COMMODITIES_DIR,
)
from .kelly import compute_kelly_size
from .log import setup_logging


logger = setup_logging("strategy")


def get_base_size(confidence: float) -> float:
    """Map confidence to base position size."""
    if confidence >= 0.90:
        return SIZING.base_sizes["90+"]
    elif confidence >= 0.85:
        return SIZING.base_sizes["85-90"]
    elif confidence >= 0.80:
        return SIZING.base_sizes["80-85"]
    elif confidence >= 0.75:
        return SIZING.base_sizes["75-80"]
    return 0.0


def apply_correlation_adjustment(signals: dict) -> dict:
    """Reduce sizing when correlated commodities signal simultaneously."""
    # Count active signals per correlation group
    group_counts = {}
    for group_name, members in CORRELATION_GROUPS.items():
        active = [m for m in members if m in signals]
        if active:
            group_counts[group_name] = active

    total_signals = sum(len(v) for v in group_counts.values())

    for key, signal in signals.items():
        multiplier = 1.0

        # Correlated group cut
        for group_name, members in CORRELATION_GROUPS.items():
            if key in members and group_name in group_counts:
                if len(group_counts[group_name]) >= 2:
                    multiplier *= SIZING.correlated_cut
                    signal["correlation_note"] = (
                        f"Correlated with {', '.join(m for m in group_counts[group_name] if m != key)}"
                    )

        # Portfolio overload cut
        if total_signals >= 4:
            multiplier *= SIZING.portfolio_overload_cut
            signal["portfolio_note"] = f"{total_signals} simultaneous signals"

        signal["size_multiplier"] = round(multiplier, 3)

    return signals


def generate_trade_plan(predictions: dict) -> dict:
    """Convert predictions into sized trade recommendations.

    Args:
        predictions: Dict of {commodity_key: prediction_dict} from Prediction Agent.

    Returns:
        Trade plan with sized positions and risk parameters.
    """
    signals = {}
    no_trade = {}

    for key, pred in predictions.items():
        if pred is None:
            continue

        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        # Apply threshold
        if not pred.get("signal", False):
            no_trade[key] = {
                "reason": f"confidence {pred['confidence']:.1%} < threshold {cfg.confidence_threshold:.0%}",
            }
            continue

        # Build signal
        direction = "LONG" if pred["direction"] == "UP" else "SHORT"
        base_size = get_base_size(pred["confidence"])
        price = pred["price"]
        pred_return = abs(pred["pred_return"])

        # TP/SL
        if direction == "LONG":
            tp_price = price * (1 + pred_return * TRADE_DEFAULTS.take_profit_multiplier)
            sl_price = price * (1 - TRADE_DEFAULTS.stop_loss_pct)
        else:
            tp_price = price * (1 - pred_return * TRADE_DEFAULTS.take_profit_multiplier)
            sl_price = price * (1 + TRADE_DEFAULTS.stop_loss_pct)

        signals[key] = {
            "commodity": cfg.name,
            "direction": direction,
            "confidence": pred["confidence"],
            "pred_return": pred["pred_return"],
            "price": price,
            "tp_price": round(tp_price, 4),
            "sl_price": round(sl_price, 4),
            "risk_reward": round(pred_return / TRADE_DEFAULTS.stop_loss_pct, 2),
            "base_size": base_size,
            "max_hold_days": TRADE_DEFAULTS.max_hold_days,
        }

    # Apply correlation adjustments
    if signals:
        signals = apply_correlation_adjustment(signals)

    # Compute final sizes and check portfolio limit
    total_exposure = 0.0
    for key, sig in signals.items():
        final_size = sig["base_size"] * sig.get("size_multiplier", 1.0)
        final_size = min(final_size, SIZING.max_portfolio_exposure - total_exposure)
        final_size = max(final_size, 0.0)
        sig["final_size"] = round(final_size, 4)
        total_exposure += final_size

    plan = {
        "timestamp": datetime.now().isoformat(),
        "signals": signals,
        "no_trade": no_trade,
        "total_exposure": round(total_exposure, 4),
        "n_signals": len(signals),
    }

    return plan


def main():
    """Generate trade plan from latest predictions log."""
    predictions_log = COMMODITIES_DIR / "logs" / "predictions.jsonl"

    if not predictions_log.exists():
        print("No predictions log found. Run the prediction agent first.")
        return

    # Build name-to-key mapping once (handles "Natural Gas" -> "natgas", etc.)
    name_to_key = {}
    for k, cfg in COMMODITIES.items():
        name_to_key[k] = k  # key itself
        name_to_key[cfg.name.lower().replace(" ", "")] = k
        name_to_key[cfg.dir_name.lower().replace(" ", "")] = k

    # Load latest prediction per commodity
    latest = {}
    with open(predictions_log) as f:
        for line in f:
            try:
                entry = json.loads(line)
                raw = entry.get("commodity", "").lower().replace(" ", "")
                resolved = name_to_key.get(raw, raw)
                latest[resolved] = entry
            except json.JSONDecodeError:
                continue

    if not latest:
        print("No predictions found in log.")
        return

    plan = generate_trade_plan(latest)

    print(f"\n{'='*60}")
    print("TRADE PLAN")
    print(f"{'='*60}")

    if plan["signals"]:
        print(f"\nACTIVE SIGNALS ({plan['n_signals']}):")
        for key, sig in plan["signals"].items():
            print(f"\n  {sig['direction']} {sig['commodity']}")
            print(f"    Confidence:  {sig['confidence']:.1%}")
            print(f"    Entry:       ${sig['price']:.2f}")
            print(f"    Take Profit: ${sig['tp_price']:.2f} (R:R = {sig['risk_reward']:.1f})")
            print(f"    Stop Loss:   ${sig['sl_price']:.2f}")
            print(f"    Position:    {sig['final_size']:.1%} of portfolio")
            if "correlation_note" in sig:
                print(f"    Note:        {sig['correlation_note']}")
        print(f"\n  Total exposure: {plan['total_exposure']:.1%}")
    else:
        print("\nNo active signals.")

    if plan["no_trade"]:
        print(f"\nPASSED ({len(plan['no_trade'])}):")
        for key, info in plan["no_trade"].items():
            cfg = COMMODITIES.get(key)
            name = cfg.name if cfg else key
            print(f"  {name}: {info['reason']}")


if __name__ == "__main__":
    main()
