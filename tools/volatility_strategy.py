"""Volatility harvesting strategies for high-vol commodities.

Three strategies that profit from volatility without requiring high
directional accuracy:

1. Breakout: Follow price breakouts from consolidation ranges
2. Mean-reversion: Fade extreme moves when vol is elevated
3. Straddle: Enter both directions, let stops/profits sort it out

Each can be backtested independently or combined into a portfolio.

Usage:
    python tools/volatility_strategy.py                         # all commodities, all strategies
    python tools/volatility_strategy.py coffee cocoa            # specific commodities
    python tools/volatility_strategy.py --strategy breakout     # specific strategy
    python tools/volatility_strategy.py --start 2023-01-01      # custom start
"""

import json
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import COMMODITIES


@dataclass
class VolConfig:
    """Configuration for volatility strategies."""
    initial_capital: float = 100_000
    position_pct: float = 0.10           # 10% per trade
    slippage_pct: float = 0.002          # 0.2% per side
    commission_pct: float = 0.001        # 0.1% per side


# ── Strategy 1: Breakout ──────────────────────────────────────────────────

def backtest_breakout(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int = 20,
    hold_days: int = 10,
    stop_loss_pct: float = 0.05,
    take_profit_mult: float = 2.0,
    config: VolConfig = None,
) -> dict:
    """Breakout strategy: enter when price breaks N-day high/low.

    When price breaks above the 20-day high, go long.
    When price breaks below the 20-day low, go short.
    Exit on stop loss, take profit (2x the breakout range), or time.
    """
    if config is None:
        config = VolConfig()

    capital = config.initial_capital
    trades = []
    equity = []
    in_trade = False
    entry_price = 0
    direction = 0
    entry_idx = 0
    position_size = 0

    for i in range(lookback, len(prices)):
        current = prices[i]
        high_n = np.max(prices[i - lookback:i])
        low_n = np.min(prices[i - lookback:i])
        range_n = high_n - low_n

        if in_trade:
            hold = i - entry_idx
            if direction == 1:
                pnl_pct = (current - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current) / entry_price

            exit_reason = None
            if pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif range_n > 0 and pnl_pct >= (stop_loss_pct * take_profit_mult):
                exit_reason = "take_profit"
            elif hold >= hold_days:
                exit_reason = "time_exit"

            if exit_reason:
                pnl_pct -= config.slippage_pct + config.commission_pct
                pnl_dollars = position_size * pnl_pct
                capital += position_size + pnl_dollars
                trades.append({
                    "entry": dates[entry_idx].strftime("%Y-%m-%d"),
                    "exit": dates[i].strftime("%Y-%m-%d"),
                    "dir": "LONG" if direction == 1 else "SHORT",
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": pnl_dollars,
                    "hold": hold,
                    "reason": exit_reason,
                })
                in_trade = False

        if not in_trade and range_n > 0:
            # Breakout above high
            if current > high_n and (current - high_n) / range_n > 0.1:
                direction = 1
                entry_price = current * (1 + config.slippage_pct)
                position_size = capital * config.position_pct
                capital -= position_size + (position_size * config.commission_pct)
                entry_idx = i
                in_trade = True

            # Breakout below low
            elif current < low_n and (low_n - current) / range_n > 0.1:
                direction = -1
                entry_price = current * (1 - config.slippage_pct)
                position_size = capital * config.position_pct
                capital -= position_size + (position_size * config.commission_pct)
                entry_idx = i
                in_trade = True

        total_eq = capital + (position_size if in_trade else 0)
        if in_trade:
            if direction == 1:
                total_eq += position_size * ((current - entry_price) / entry_price)
            else:
                total_eq += position_size * ((entry_price - current) / entry_price)
        equity.append(total_eq)

    return _compute_metrics("breakout", trades, equity, config)


# ── Strategy 2: Mean Reversion ────────────────────────────────────────────

def backtest_mean_reversion(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    zscore_window: int = 63,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    stop_loss_pct: float = 0.08,
    max_hold: int = 21,
    config: VolConfig = None,
) -> dict:
    """Mean reversion: fade extreme moves.

    When price z-score exceeds ±2, enter opposite direction (expect reversion).
    Exit when z-score returns to ±0.5 or stop loss / time.

    Works best in high-vol, range-bound markets.
    """
    if config is None:
        config = VolConfig()

    capital = config.initial_capital
    trades = []
    equity = []
    in_trade = False
    entry_price = 0
    direction = 0
    entry_idx = 0
    position_size = 0

    # Compute rolling z-score
    ret = pd.Series(prices).pct_change(zscore_window)
    ret_mean = ret.rolling(252, min_periods=63).mean()
    ret_std = ret.rolling(252, min_periods=63).std()
    zscore = ((ret - ret_mean) / ret_std).values

    for i in range(zscore_window + 252, len(prices)):
        current = prices[i]
        z = zscore[i] if not np.isnan(zscore[i]) else 0

        if in_trade:
            hold = i - entry_idx
            if direction == 1:
                pnl_pct = (current - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current) / entry_price

            exit_reason = None
            if pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif (direction == 1 and z >= -exit_zscore) or (direction == -1 and z <= exit_zscore):
                exit_reason = "mean_revert"
            elif hold >= max_hold:
                exit_reason = "time_exit"

            if exit_reason:
                pnl_pct -= config.slippage_pct + config.commission_pct
                pnl_dollars = position_size * pnl_pct
                capital += position_size + pnl_dollars
                trades.append({
                    "entry": dates[entry_idx].strftime("%Y-%m-%d"),
                    "exit": dates[i].strftime("%Y-%m-%d"),
                    "dir": "LONG" if direction == 1 else "SHORT",
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": pnl_dollars,
                    "hold": hold,
                    "reason": exit_reason,
                })
                in_trade = False

        if not in_trade:
            if z < -entry_zscore:  # Oversold → buy
                direction = 1
                entry_price = current * (1 + config.slippage_pct)
                position_size = capital * config.position_pct
                capital -= position_size + (position_size * config.commission_pct)
                entry_idx = i
                in_trade = True
            elif z > entry_zscore:  # Overbought → sell
                direction = -1
                entry_price = current * (1 - config.slippage_pct)
                position_size = capital * config.position_pct
                capital -= position_size + (position_size * config.commission_pct)
                entry_idx = i
                in_trade = True

        total_eq = capital + (position_size if in_trade else 0)
        if in_trade:
            if direction == 1:
                total_eq += position_size * ((current - entry_price) / entry_price)
            else:
                total_eq += position_size * ((entry_price - current) / entry_price)
        equity.append(total_eq)

    return _compute_metrics("mean_reversion", trades, equity, config)


# ── Strategy 3: Straddle ──────────────────────────────────────────────────

def backtest_straddle(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    vol_window: int = 21,
    vol_threshold_pctile: float = 80,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    cooldown: int = 5,
    config: VolConfig = None,
) -> dict:
    """Straddle: enter both directions when vol is spiking.

    When short-term vol exceeds its 80th percentile, expect a big move.
    Enter both long and short positions with tight stops.
    One leg gets stopped out, the other (hopefully) runs to profit.
    Net P&L = take_profit - stop_loss if the move is big enough.

    This is the purest volatility strategy — doesn't need direction prediction.
    """
    if config is None:
        config = VolConfig()

    capital = config.initial_capital
    trades = []
    equity = []
    half_size = config.position_pct / 2  # Split between long and short

    # Compute vol and its percentile
    returns = pd.Series(prices).pct_change()
    vol = returns.rolling(vol_window).std() * np.sqrt(252)
    vol_pctile = vol.rolling(252, min_periods=63).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).values

    last_entry = -cooldown - 1  # Allow first trade

    # Track paired legs
    long_pos = None
    short_pos = None

    for i in range(vol_window + 252, len(prices)):
        current = prices[i]
        vp = vol_pctile[i] if not np.isnan(vol_pctile[i]) else 0

        # Check long leg
        if long_pos is not None:
            pnl = (current - long_pos["entry"]) / long_pos["entry"]
            exit_reason = None
            if pnl <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif pnl >= take_profit_pct:
                exit_reason = "take_profit"
            elif i - long_pos["idx"] >= 15:
                exit_reason = "time_exit"

            if exit_reason:
                pnl -= config.slippage_pct + config.commission_pct
                pnl_d = long_pos["size"] * pnl
                capital += long_pos["size"] + pnl_d
                trades.append({
                    "entry": dates[long_pos["idx"]].strftime("%Y-%m-%d"),
                    "exit": dates[i].strftime("%Y-%m-%d"),
                    "dir": "LONG", "pnl_pct": pnl, "pnl_dollars": pnl_d,
                    "hold": i - long_pos["idx"], "reason": exit_reason,
                })
                long_pos = None

        # Check short leg
        if short_pos is not None:
            pnl = (short_pos["entry"] - current) / short_pos["entry"]
            exit_reason = None
            if pnl <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif pnl >= take_profit_pct:
                exit_reason = "take_profit"
            elif i - short_pos["idx"] >= 15:
                exit_reason = "time_exit"

            if exit_reason:
                pnl -= config.slippage_pct + config.commission_pct
                pnl_d = short_pos["size"] * pnl
                capital += short_pos["size"] + pnl_d
                trades.append({
                    "entry": dates[short_pos["idx"]].strftime("%Y-%m-%d"),
                    "exit": dates[i].strftime("%Y-%m-%d"),
                    "dir": "SHORT", "pnl_pct": pnl, "pnl_dollars": pnl_d,
                    "hold": i - short_pos["idx"], "reason": exit_reason,
                })
                short_pos = None

        # Enter straddle when vol is high and no open positions
        if long_pos is None and short_pos is None and (i - last_entry) >= cooldown:
            if vp >= vol_threshold_pctile / 100:
                pos_size = capital * half_size
                long_pos = {"entry": current * (1 + config.slippage_pct), "idx": i, "size": pos_size}
                short_pos = {"entry": current * (1 - config.slippage_pct), "idx": i, "size": pos_size}
                capital -= 2 * pos_size + 2 * (pos_size * config.commission_pct)
                last_entry = i

        # Equity tracking
        total_eq = capital
        if long_pos:
            total_eq += long_pos["size"] * (1 + (current - long_pos["entry"]) / long_pos["entry"])
        if short_pos:
            total_eq += short_pos["size"] * (1 + (short_pos["entry"] - current) / short_pos["entry"])
        equity.append(total_eq)

    return _compute_metrics("straddle", trades, equity, config)


# ── Shared metrics ────────────────────────────────────────────────────────

def _compute_metrics(strategy_name: str, trades: list, equity: list, config: VolConfig) -> dict:
    """Compute standard performance metrics from trade list."""
    if not trades:
        return {"strategy": strategy_name, "status": "no_trades", "n_trades": 0}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    total_pnl = sum(t["pnl_dollars"] for t in trades)
    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["hold"] for t in trades])

    gross_profit = sum(t["pnl_dollars"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    eq = np.array(equity) if equity else np.array([config.initial_capital])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-10)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    if len(eq) > 1:
        daily_ret = np.diff(eq) / eq[:-1]
        sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0
    else:
        sharpe = 0

    exit_counts = {}
    for t in trades:
        r = t["reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        "strategy": strategy_name,
        "status": "ok",
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_hold": avg_hold,
        "total_pnl": total_pnl,
        "total_return": total_pnl / config.initial_capital,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "final_equity": eq[-1] if len(eq) > 0 else config.initial_capital,
        "exit_reasons": exit_counts,
        "trades": trades[-10:],
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_commodity(key: str, strategies: list, start_date: str = None, config: VolConfig = None) -> dict:
    """Run all volatility strategies for a commodity."""
    if config is None:
        config = VolConfig()

    cfg = COMMODITIES[key]
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"commodity": cfg.name, "status": "no_data"}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return {"commodity": cfg.name, "status": "no_price_col"}

    prices = df[cfg.price_col].values
    dates = df.index

    if start_date:
        start_idx = dates.searchsorted(pd.Timestamp(start_date))
    else:
        start_idx = max(0, len(df) - 756)  # last 3 years

    prices = prices[start_idx:]
    dates = dates[start_idx:]

    # Compute annualized vol for context
    ret = pd.Series(prices).pct_change()
    annual_vol = float(ret.std() * np.sqrt(252))

    results = {"commodity": cfg.name, "annual_vol": annual_vol, "strategies": {}}

    for strat in strategies:
        if strat == "breakout":
            r = backtest_breakout(prices, dates, config=config)
        elif strat == "mean_reversion":
            r = backtest_mean_reversion(prices, dates, config=config)
        elif strat == "straddle":
            r = backtest_straddle(prices, dates, config=config)
        else:
            continue
        results["strategies"][strat] = r

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest volatility strategies")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--strategy", choices=["breakout", "mean_reversion", "straddle", "all"],
                        default="all", help="Which strategy to run")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()

    strategies = ["breakout", "mean_reversion", "straddle"] if args.strategy == "all" else [args.strategy]
    config = VolConfig(initial_capital=args.capital)

    targets = args.commodities or list(COMMODITIES.keys())
    all_results = {}

    for key in targets:
        # Resolve name
        if key not in COMMODITIES:
            for k, c in COMMODITIES.items():
                if c.dir_name == key or c.name.lower() == key.lower():
                    key = k
                    break
            else:
                continue

        print(f"  Running {COMMODITIES[key].name}...", end=" ", flush=True)
        r = run_commodity(key, strategies, start_date=args.start, config=config)
        all_results[key] = r
        print(f"vol={r.get('annual_vol', 0):.0%}")

    # Summary
    print(f"\n{'='*90}")
    print(f"VOLATILITY STRATEGY BACKTEST (${config.initial_capital:,.0f} capital)")
    print(f"{'='*90}")

    for strat in strategies:
        print(f"\n  Strategy: {strat.upper()}")
        print(f"  {'Commodity':<15} {'Vol':>5} {'Trades':>6} {'Win%':>6} {'P&L':>12} "
              f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'PF':>6}")
        print(f"  {'-'*80}")

        strat_total = 0
        strat_count = 0
        for key, r in all_results.items():
            s = r.get("strategies", {}).get(strat, {})
            if s.get("status") != "ok":
                print(f"  {r['commodity']:<15} {r.get('annual_vol', 0):>4.0%} {'no trades':>6}")
                continue

            strat_total += s["total_pnl"]
            strat_count += 1
            exits = s.get("exit_reasons", {})
            print(f"  {r['commodity']:<13} {r['annual_vol']:>5.0%} {s['n_trades']:>6} "
                  f"{s['win_rate']:>5.0%} ${s['total_pnl']:>+10,.0f} "
                  f"{s['total_return']:>+7.1%} {s['sharpe']:>6.2f} "
                  f"{s['max_drawdown']:>6.1%} {s['profit_factor']:>6.2f}")

        if strat_count > 0:
            avg_ret = strat_total / (config.initial_capital * strat_count)
            print(f"  {'TOTAL':<13} {'':>5} {'':>6} {'':>5} "
                  f"${strat_total:>+10,.0f} {avg_ret:>+7.1%}")

    # Combined best strategy per commodity
    print(f"\n{'='*90}")
    print(f"BEST STRATEGY PER COMMODITY")
    print(f"{'='*90}")
    combined_pnl = 0
    for key, r in all_results.items():
        best_name = ""
        best_ret = -999
        for sname, s in r.get("strategies", {}).items():
            if s.get("status") == "ok" and s["total_return"] > best_ret:
                best_ret = s["total_return"]
                best_name = sname
        if best_name:
            s = r["strategies"][best_name]
            combined_pnl += s["total_pnl"]
            print(f"  {r['commodity']:<15} {best_name:<18} {best_ret:>+7.1%} "
                  f"(win={s['win_rate']:.0%}, sharpe={s['sharpe']:.2f})")
        else:
            print(f"  {r['commodity']:<15} no profitable strategy")

    n_commodities = len([r for r in all_results.values() if r.get("strategies")])
    if n_commodities > 0:
        print(f"\n  Combined P&L: ${combined_pnl:>+,.0f} "
              f"({combined_pnl / (config.initial_capital * n_commodities):>+.1%} avg)")


if __name__ == "__main__":
    main()
