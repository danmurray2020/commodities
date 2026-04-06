"""Simulate trading using ensemble models on historical data.

Walks through history day by day, generates ensemble predictions,
applies position sizing and risk rules, and tracks P&L with
realistic transaction costs.

Usage:
    python tools/simulate_trading.py                    # all commodities
    python tools/simulate_trading.py natgas wheat       # specific
    python tools/simulate_trading.py --start 2024-01-01 # custom start
"""

import json
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import COMMODITIES


@dataclass
class SimConfig:
    """Simulation parameters."""
    initial_capital: float = 100_000
    max_position_pct: float = 0.15       # max 15% of capital per position
    confidence_threshold: float = 0.60   # min confidence to trade
    agreement_threshold: float = 0.55    # min model agreement to trade
    stop_loss_pct: float = 0.08          # 8% stop loss
    take_profit_pct: float = None        # None = use predicted return
    max_hold_days: int = None            # None = use model horizon
    slippage_pct: float = 0.002          # 0.2% slippage per side
    commission_pct: float = 0.001        # 0.1% commission per side


@dataclass
class Position:
    commodity: str
    direction: str  # "LONG" or "SHORT"
    entry_date: str
    entry_price: float
    size_dollars: float
    horizon: int
    pred_return: float
    confidence: float
    stop_loss: float
    take_profit: float
    hold_days: int = 0
    exit_date: str = None
    exit_price: float = None
    exit_reason: str = None
    pnl_dollars: float = 0
    pnl_pct: float = 0


def load_ensemble_for_simulation(key: str) -> dict:
    """Load ensemble metadata and models for a commodity."""
    cfg = COMMODITIES[key]
    meta_path = cfg.models_dir / "ensemble_metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "config": cfg,
        "meta": meta,
        "models_dir": cfg.models_dir,
    }


def get_prediction_at_date(cfg, meta, models_dir, df, date_idx, price_col):
    """Generate ensemble prediction for a specific date in the historical data."""
    models_list = meta.get("models", [])
    if not models_list:
        return None

    row = df.iloc[[date_idx]]
    predictions = []

    for m_meta in models_list:
        features = m_meta["features"]
        available = [f for f in features if f in df.columns]
        if len(available) < len(features) * 0.8:  # allow 20% missing
            continue

        horizon = m_meta["horizon"]
        model_type = m_meta["model_type"]

        reg_path = models_dir / f"ensemble_reg_{horizon}d_{model_type}.joblib"
        clf_path = models_dir / f"ensemble_clf_{horizon}d_{model_type}.joblib"

        if not reg_path.exists() or not clf_path.exists():
            continue

        try:
            import joblib
            reg = joblib.load(reg_path)
            clf = joblib.load(clf_path)

            X = row[available].values
            pred_return = float(reg.predict(X)[0])
            pred_dir = int(clf.predict(X)[0])
            pred_proba = clf.predict_proba(X)[0]
            confidence = float(pred_proba[pred_dir])

            predictions.append({
                "horizon": horizon,
                "model_type": model_type,
                "pred_return": pred_return,
                "direction": 1 if pred_dir == 1 else -1,
                "confidence": confidence,
                "weight": m_meta.get("weight", 1.0),
            })
        except Exception:
            continue

    if not predictions:
        return None

    # Ensemble combination
    total_weight = sum(p["weight"] for p in predictions)
    up_weight = sum(p["weight"] for p in predictions if p["direction"] == 1)
    down_weight = total_weight - up_weight

    if up_weight > down_weight:
        direction = "LONG"
        agreement = up_weight / total_weight
    else:
        direction = "SHORT"
        agreement = down_weight / total_weight

    ensemble_return = sum(p["pred_return"] * p["weight"] for p in predictions) / total_weight
    avg_confidence = sum(p["confidence"] * p["weight"] for p in predictions) / total_weight

    # Best model's horizon for hold period
    best = max(predictions, key=lambda x: x["confidence"])
    horizon = best["horizon"]

    return {
        "direction": direction,
        "pred_return": ensemble_return,
        "confidence": avg_confidence,
        "agreement": agreement,
        "horizon": horizon,
        "n_models": len(predictions),
    }


def simulate_commodity(key: str, start_date: str = None, config: SimConfig = None) -> dict:
    """Run historical simulation for a single commodity."""
    if config is None:
        config = SimConfig()

    ensemble = load_ensemble_for_simulation(key)
    if not ensemble:
        return {"commodity": key, "status": "no_ensemble", "trades": []}

    cfg = ensemble["config"]
    meta = ensemble["meta"]
    models_dir = ensemble["models_dir"]
    price_col = cfg.price_col

    # Load data via subprocess for proper feature engineering
    script = f"""
import json, sys, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from features import prepare_dataset
df, cols = prepare_dataset(horizon=5)
for col in ['target_return', 'target_direction']:
    if col in df.columns:
        df = df.drop(columns=[col])
df.to_csv('/tmp/_sim_data_{key}.csv')
print(json.dumps({{"rows": len(df), "cols": len(df.columns)}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        cwd=str(cfg.project_dir), timeout=120,
    )
    if result.returncode != 0:
        return {"commodity": key, "status": "data_error", "error": result.stderr[-200:], "trades": []}

    df = pd.read_csv(f"/tmp/_sim_data_{key}.csv", index_col=0, parse_dates=True)

    # Determine simulation window
    if start_date:
        start_idx = df.index.searchsorted(pd.Timestamp(start_date))
    else:
        start_idx = len(df) - 504  # last 2 years

    if start_idx < 252:
        start_idx = 252  # need at least 1 year of history

    prices = df[price_col].values
    dates = df.index

    # Simulation state
    capital = config.initial_capital
    positions = []
    closed_trades = []
    equity_curve = []
    daily_dates = []

    print(f"  Simulating {cfg.name} from {dates[start_idx].date()} to {dates[-1].date()}...")

    for i in range(start_idx, len(df)):
        current_price = prices[i]
        current_date = dates[i].strftime("%Y-%m-%d")

        # Check existing positions for exit
        active_positions = []
        for pos in positions:
            pos.hold_days += 1
            current_val = current_price

            # Calculate unrealized P&L
            if pos.direction == "LONG":
                pnl_pct = (current_val - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - current_val) / pos.entry_price

            # Exit conditions
            exit_reason = None
            max_hold = pos.horizon if config.max_hold_days is None else config.max_hold_days

            if pnl_pct <= -config.stop_loss_pct:
                exit_reason = "stop_loss"
            elif pos.take_profit and pnl_pct >= pos.take_profit:
                exit_reason = "take_profit"
            elif pos.hold_days >= max_hold:
                exit_reason = "time_exit"

            if exit_reason:
                # Close position
                exit_price = current_val * (1 + config.slippage_pct * (-1 if pos.direction == "LONG" else 1))
                if pos.direction == "LONG":
                    realized_pnl = (exit_price - pos.entry_price) / pos.entry_price
                else:
                    realized_pnl = (pos.entry_price - exit_price) / pos.entry_price

                realized_pnl -= config.commission_pct  # exit commission
                pnl_dollars = pos.size_dollars * realized_pnl

                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.pnl_pct = realized_pnl
                pos.pnl_dollars = pnl_dollars
                capital += pos.size_dollars + pnl_dollars
                closed_trades.append(pos)
            else:
                active_positions.append(pos)

        positions = active_positions

        # Check for new entry (only if we have capacity)
        total_invested = sum(p.size_dollars for p in positions)
        available_capital = capital - total_invested

        if available_capital > config.initial_capital * 0.10:  # need at least 10% free
            # Only check every N days to avoid overtrading
            # (use the best model's horizon as recheck interval)
            best_horizon = meta.get("models", [{}])[0].get("horizon", 10) if meta.get("models") else 10
            if (i - start_idx) % max(best_horizon // 2, 3) == 0:
                pred = get_prediction_at_date(cfg, meta, models_dir, df, i, price_col)

                if pred and pred["confidence"] >= config.confidence_threshold and pred["agreement"] >= config.agreement_threshold:
                    # Size position
                    position_size = min(
                        available_capital * config.max_position_pct,
                        capital * config.max_position_pct,
                    )

                    # Adjust by confidence
                    confidence_mult = (pred["confidence"] - 0.5) * 2  # 0.5→0, 1.0→1
                    position_size *= max(confidence_mult, 0.3)

                    entry_price = current_price * (1 + config.slippage_pct * (1 if pred["direction"] == "LONG" else -1))

                    tp_pct = abs(pred["pred_return"]) if config.take_profit_pct is None else config.take_profit_pct

                    pos = Position(
                        commodity=cfg.name,
                        direction=pred["direction"],
                        entry_date=current_date,
                        entry_price=entry_price,
                        size_dollars=position_size,
                        horizon=pred["horizon"],
                        pred_return=pred["pred_return"],
                        confidence=pred["confidence"],
                        stop_loss=config.stop_loss_pct,
                        take_profit=tp_pct if tp_pct > 0.01 else 0.05,
                    )
                    capital -= position_size + (position_size * config.commission_pct)
                    positions.append(pos)

        # Track equity
        unrealized = sum(
            p.size_dollars * ((prices[i] - p.entry_price) / p.entry_price if p.direction == "LONG"
                              else (p.entry_price - prices[i]) / p.entry_price)
            for p in positions
        )
        total_equity = capital + sum(p.size_dollars for p in positions) + unrealized
        equity_curve.append(total_equity)
        daily_dates.append(current_date)

    # Force close remaining positions (with slippage, same as normal exits)
    for pos in positions:
        raw_price = prices[-1]
        final_price = raw_price * (1 + config.slippage_pct * (-1 if pos.direction == "LONG" else 1))
        if pos.direction == "LONG":
            pnl = (final_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - final_price) / pos.entry_price
        pnl -= config.commission_pct
        pos.exit_date = dates[-1].strftime("%Y-%m-%d")
        pos.exit_price = final_price
        pos.exit_reason = "sim_end"
        pos.pnl_pct = pnl
        pos.pnl_dollars = pos.size_dollars * pnl
        closed_trades.append(pos)

    # Compute metrics
    if not closed_trades:
        return {"commodity": cfg.name, "status": "no_trades", "trades": []}

    wins = [t for t in closed_trades if t.pnl_pct > 0]
    losses = [t for t in closed_trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    total_pnl = sum(t.pnl_dollars for t in closed_trades)
    total_return = total_pnl / config.initial_capital

    # Equity curve stats
    equity = np.array(equity_curve)
    if len(equity) > 1:
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.maximum(peak, 1e-10)
        max_dd = float(np.min(drawdown))
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
    else:
        max_dd = 0
        sharpe = 0

    gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    return {
        "commodity": cfg.name,
        "status": "ok",
        "start_date": daily_dates[0] if daily_dates else "?",
        "end_date": daily_dates[-1] if daily_dates else "?",
        "n_trades": len(closed_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "final_equity": equity_curve[-1] if equity_curve else config.initial_capital,
        "exit_reasons": {
            "stop_loss": sum(1 for t in closed_trades if t.exit_reason == "stop_loss"),
            "take_profit": sum(1 for t in closed_trades if t.exit_reason == "take_profit"),
            "time_exit": sum(1 for t in closed_trades if t.exit_reason == "time_exit"),
            "sim_end": sum(1 for t in closed_trades if t.exit_reason == "sim_end"),
        },
        "trades": [{
            "entry": t.entry_date, "exit": t.exit_date,
            "dir": t.direction, "pnl": f"{t.pnl_pct:+.2%}",
            "reason": t.exit_reason, "conf": f"{t.confidence:.0%}",
            "hold": t.hold_days,
        } for t in closed_trades[-20:]],  # last 20 trades
        "equity_curve": {
            "dates": daily_dates[::5],  # sample every 5 days
            "values": [round(v, 2) for v in equity_curve[::5]],
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simulate trading with ensemble models")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    args = parser.parse_args()

    targets = args.commodities or list(COMMODITIES.keys())
    config = SimConfig(initial_capital=args.capital)

    results = {}
    total_pnl = 0

    for key in targets:
        if key not in COMMODITIES:
            # Try resolving
            for k, c in COMMODITIES.items():
                if c.dir_name == key or c.name.lower() == key.lower():
                    key = k
                    break
            else:
                print(f"Unknown commodity: {key}")
                continue

        r = simulate_commodity(key, start_date=args.start, config=config)
        results[key] = r
        if r["status"] == "ok":
            total_pnl += r["total_pnl"]

    print(f"\n{'='*80}")
    print(f"TRADING SIMULATION RESULTS (${config.initial_capital:,.0f} initial capital per commodity)")
    print(f"{'='*80}")

    print(f"\n{'Commodity':<15} {'Trades':>6} {'Win%':>6} {'P&L':>12} {'Return':>8} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'PF':>5} {'Exits'}")
    print("-" * 80)

    for key, r in results.items():
        if r["status"] != "ok":
            print(f"  {r['commodity']:<15} {r['status']}")
            continue

        exits = r["exit_reasons"]
        exit_str = f"TP:{exits['take_profit']} SL:{exits['stop_loss']} T:{exits['time_exit']}"

        print(f"  {r['commodity']:<13} {r['n_trades']:>6} {r['win_rate']:>5.0%} "
              f"${r['total_pnl']:>+10,.0f} {r['total_return']:>+7.1%} "
              f"{r['sharpe']:>6.2f} {r['max_drawdown']:>6.1%} "
              f"{r['profit_factor']:>5.2f} {exit_str}")

    portfolio_return = total_pnl / (config.initial_capital * len([r for r in results.values() if r["status"] == "ok"]))
    print(f"\n  {'PORTFOLIO':<13} {'':>6} {'':>5} "
          f"${total_pnl:>+10,.0f} {portfolio_return:>+7.1%}")

    # Show recent trades for best performer
    best = max([r for r in results.values() if r["status"] == "ok"],
               key=lambda x: x["total_return"], default=None)
    if best and best["trades"]:
        print(f"\n  Recent trades ({best['commodity']}):")
        for t in best["trades"][-10:]:
            print(f"    {t['entry']} → {t['exit']}  {t['dir']:<5} {t['pnl']:>7}  "
                  f"conf={t['conf']} hold={t['hold']}d  [{t['reason']}]")


if __name__ == "__main__":
    main()
