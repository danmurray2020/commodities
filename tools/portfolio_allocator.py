"""Portfolio allocation strategy — concentrates capital on highest-edge commodities.

Combines directional + volatility strategies with dynamic sizing based on
model accuracy, and earns treasury yield on idle cash.

Usage:
    python tools/portfolio_allocator.py                    # default
    python tools/portfolio_allocator.py --capital 500000   # custom capital
    python tools/portfolio_allocator.py --start 2023-01-01 # custom start
    python tools/portfolio_allocator.py --treasury-rate 0.045  # current T-bill rate
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import COMMODITIES
from tools.simulate_trading import SimConfig, get_prediction_at_date, load_ensemble_for_simulation
from tools.volatility_strategy import VolConfig, backtest_mean_reversion
import subprocess


@dataclass
class PortfolioConfig:
    """Portfolio-level configuration."""
    total_capital: float = 500_000
    treasury_rate: float = 0.045          # annual T-bill yield on idle cash
    max_commodity_allocation: float = 0.30  # max 30% in any single commodity
    max_total_invested: float = 0.70       # max 70% deployed, 30% always in cash/treasuries
    min_model_accuracy: float = 0.60       # only trade commodities above this accuracy
    rebalance_frequency: int = 5           # rebalance every N trading days
    # Directional strategy params
    dir_confidence_threshold: float = 0.55
    dir_agreement_threshold: float = 0.50
    dir_stop_loss: float = 0.10
    # Vol strategy params
    vol_position_pct: float = 0.05         # smaller vol positions


def load_model_accuracies() -> dict:
    """Load best model accuracy for each commodity from ensemble metadata."""
    accuracies = {}
    for key, cfg in COMMODITIES.items():
        meta_path = cfg.models_dir / "ensemble_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            models = meta.get("models", [])
            if models and "model_type" in models[0]:
                best = max(models, key=lambda x: x.get("avg_dir_acc", 0))
                accuracies[key] = {
                    "accuracy": best["avg_dir_acc"],
                    "horizon": best["horizon"],
                    "model_type": best["model_type"],
                }
            else:
                accuracies[key] = {"accuracy": meta.get("regression", {}).get("avg_accuracy", 0.5),
                                   "horizon": meta.get("horizon", 21), "model_type": "unknown"}
        else:
            accuracies[key] = {"accuracy": 0.5, "horizon": 21, "model_type": "none"}
    return accuracies


def compute_allocation_weights(accuracies: dict, config: PortfolioConfig) -> dict:
    """Compute capital allocation weights based on model accuracy.

    Higher accuracy = more capital. Below min_accuracy = zero.
    Uses accuracy-minus-baseline as the edge measure.
    """
    edges = {}
    for key, info in accuracies.items():
        acc = info["accuracy"]
        if acc >= config.min_model_accuracy:
            # Edge = how much better than random (50%)
            edge = acc - 0.50
            edges[key] = edge
        else:
            edges[key] = 0

    total_edge = sum(edges.values())
    if total_edge <= 0:
        return {k: 0 for k in accuracies}

    weights = {}
    for key, edge in edges.items():
        raw_weight = edge / total_edge
        # Cap at max_commodity_allocation
        weights[key] = min(raw_weight, config.max_commodity_allocation)

    # Renormalize to sum to max_total_invested
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        scale = min(config.max_total_invested, weight_sum) / weight_sum
        weights = {k: v * scale for k, v in weights.items()}

    return weights


def simulate_portfolio(config: PortfolioConfig = None, start_date: str = None) -> dict:
    """Run full portfolio simulation with dynamic allocation."""
    if config is None:
        config = PortfolioConfig()

    accuracies = load_model_accuracies()
    weights = compute_allocation_weights(accuracies, config)

    print(f"\n{'='*70}")
    print(f"PORTFOLIO ALLOCATION (${config.total_capital:,.0f} total)")
    print(f"{'='*70}")
    print(f"\n  Capital allocation by model edge:")

    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for key, weight in ranked:
        cfg = COMMODITIES[key]
        info = accuracies[key]
        alloc = config.total_capital * weight
        if weight > 0:
            print(f"    {cfg.name:<15} {weight:>5.1%} (${alloc:>8,.0f}) "
                  f"— acc={info['accuracy']:.0%} {info['horizon']}d {info['model_type']}")
        else:
            print(f"    {cfg.name:<15}   0.0% — below {config.min_model_accuracy:.0%} threshold")

    cash_weight = 1.0 - sum(weights.values())
    cash_alloc = config.total_capital * cash_weight
    print(f"    {'Cash/T-bills':<15} {cash_weight:>5.1%} (${cash_alloc:>8,.0f}) — earning {config.treasury_rate:.1%}/yr")

    # Run simulation per commodity with allocated capital
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")

    total_dir_pnl = 0
    total_vol_pnl = 0
    all_results = []
    worst_dd = 0

    for key, weight in ranked:
        if weight <= 0:
            continue

        cfg = COMMODITIES[key]
        commodity_capital = config.total_capital * weight
        info = accuracies[key]

        # Scale position size by accuracy edge
        edge_mult = min((info["accuracy"] - 0.50) * 4, 1.5)  # 60%→0.4x, 70%→0.8x, 75%→1.0x
        position_pct = min(0.50 * edge_mult, 0.50)

        # Directional strategy
        dir_config = SimConfig(
            initial_capital=commodity_capital,
            max_position_pct=position_pct,
            confidence_threshold=config.dir_confidence_threshold,
            agreement_threshold=config.dir_agreement_threshold,
            stop_loss_pct=config.dir_stop_loss,
        )
        dr = simulate_commodity_silent(key, start_date=start_date or "2024-01-01", config=dir_config)
        dir_pnl = dr.get("total_pnl", 0)
        dir_trades = dr.get("n_trades", 0)
        dir_wr = dr.get("win_rate", 0)
        dir_dd = dr.get("max_drawdown", 0)

        # Mean reversion on volatile commodities
        vol_pnl = 0
        vol_trades = 0
        csv_path = cfg.data_dir / "combined_features.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if cfg.price_col in df.columns:
                prices = df[cfg.price_col].values
                dates = df.index
                annual_vol = float(pd.Series(prices).pct_change().std() * np.sqrt(252))

                if annual_vol > 0.25:  # only on volatile commodities
                    vol_c = VolConfig(
                        initial_capital=commodity_capital * 0.3,
                        position_pct=config.vol_position_pct,
                    )
                    vr = backtest_mean_reversion(prices, dates, config=vol_c)
                    vol_pnl = vr.get("total_pnl", 0)
                    vol_trades = vr.get("n_trades", 0)

        total_pnl = dir_pnl + vol_pnl
        total_dir_pnl += dir_pnl
        total_vol_pnl += vol_pnl
        worst_dd = min(worst_dd, dir_dd)

        ret = total_pnl / commodity_capital if commodity_capital > 0 else 0
        all_results.append({
            "commodity": cfg.name,
            "capital": commodity_capital,
            "weight": weight,
            "dir_pnl": dir_pnl,
            "vol_pnl": vol_pnl,
            "total_pnl": total_pnl,
            "return": ret,
            "dir_trades": dir_trades,
            "vol_trades": vol_trades,
            "win_rate": dir_wr,
            "max_dd": dir_dd,
            "accuracy": info["accuracy"],
        })

    # Treasury yield on idle cash
    sim_years = 2.25  # approx period
    treasury_income = cash_alloc * config.treasury_rate * sim_years

    # Print results
    print(f"\n  {'Commodity':<15} {'Capital':>10} {'Dir P&L':>10} {'Vol P&L':>10} "
          f"{'Total':>10} {'Return':>8} {'Trades':>7} {'Win%':>6} {'MaxDD':>7}")
    print(f"  {'-'*90}")

    for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
        print(f"  {r['commodity']:<13} ${r['capital']:>9,.0f} ${r['dir_pnl']:>+9,.0f} "
              f"${r['vol_pnl']:>+9,.0f} ${r['total_pnl']:>+9,.0f} "
              f"{r['return']:>+7.1%} {r['dir_trades']+r['vol_trades']:>6} "
              f"{r['win_rate']:>5.0%} {r['max_dd']:>6.1%}")

    # Portfolio totals
    total_invested_pnl = total_dir_pnl + total_vol_pnl
    grand_total = total_invested_pnl + treasury_income
    total_return = grand_total / config.total_capital
    annual_return = (1 + total_return) ** (1 / sim_years) - 1

    print(f"\n  {'-'*90}")
    print(f"  {'Commodities':<13} {'':>10} ${total_dir_pnl:>+9,.0f} "
          f"${total_vol_pnl:>+9,.0f} ${total_invested_pnl:>+9,.0f}")
    print(f"  {'T-bill yield':<13} ${cash_alloc:>9,.0f} {'':>10} {'':>10} "
          f"${treasury_income:>+9,.0f}")
    print(f"  {'PORTFOLIO':<13} ${config.total_capital:>9,.0f} {'':>10} {'':>10} "
          f"${grand_total:>+9,.0f} {total_return:>+7.1%}")

    print(f"\n  Total return:     {total_return:>+.1%}")
    print(f"  Annualized:       {annual_return:>+.1%}")
    print(f"  Worst drawdown:   {worst_dd:>.1%}")
    print(f"  T-bill income:    ${treasury_income:>+,.0f} ({config.treasury_rate:.1%} on ${cash_alloc:,.0f})")
    print(f"  S&P 500 benchmark: ~10% annualized")
    print(f"  Alpha vs S&P:     {annual_return - 0.10:>+.1%}")

    # Risk-adjusted comparison
    print(f"\n  {'='*50}")
    print(f"  RISK-ADJUSTED COMPARISON")
    print(f"  {'='*50}")
    print(f"  {'':20} {'Return':>8} {'MaxDD':>8} {'Ret/DD':>8}")
    print(f"  {'This portfolio':<20} {annual_return:>+7.1%} {worst_dd:>7.1%} {annual_return/abs(worst_dd) if worst_dd != 0 else 0:>7.2f}x")
    print(f"  {'S&P 500 (typical)':<20} {'~+10%':>8} {'~-25%':>8} {'~0.40x':>8}")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": worst_dd,
        "treasury_income": treasury_income,
        "commodity_pnl": total_invested_pnl,
        "grand_total": grand_total,
        "results": all_results,
    }


def simulate_commodity_silent(key, start_date, config):
    """Run simulate_commodity without print output."""
    import io
    from contextlib import redirect_stdout
    from tools.simulate_trading import simulate_commodity
    f = io.StringIO()
    with redirect_stdout(f):
        return simulate_commodity(key, start_date=start_date, config=config)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio allocation strategy backtest")
    parser.add_argument("--capital", type=float, default=500_000, help="Total portfolio capital")
    parser.add_argument("--start", default="2024-01-01", help="Backtest start date")
    parser.add_argument("--treasury-rate", type=float, default=0.045, help="T-bill yield (annual)")
    parser.add_argument("--max-commodity", type=float, default=0.30, help="Max allocation per commodity")
    parser.add_argument("--max-invested", type=float, default=0.70, help="Max total invested (rest in T-bills)")
    args = parser.parse_args()

    config = PortfolioConfig(
        total_capital=args.capital,
        treasury_rate=args.treasury_rate,
        max_commodity_allocation=args.max_commodity,
        max_total_invested=args.max_invested,
    )

    simulate_portfolio(config=config, start_date=args.start)


if __name__ == "__main__":
    main()
