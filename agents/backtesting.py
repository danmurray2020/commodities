"""Backtesting Agent — systematic strategy validation across all commodities.

Responsibilities:
- Run walk-forward backtests for each commodity
- Compare strategy variants (confidence thresholds, position sizing, hold periods)
- Track backtest results over time to detect degradation
- Generate portfolio-level performance metrics
- Log all results to database

Usage:
    python -m agents backtest                          # backtest all
    python -m agents backtest coffee sugar             # specific commodities
    python -m agents backtest --sweep                  # parameter sweep
    python -m agents backtest --portfolio              # portfolio-level analysis
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR, CORRELATION_GROUPS
from .log import setup_logging


logger = setup_logging("backtesting")


def run_commodity_backtest(cfg: CommodityConfig) -> dict | None:
    """Run walk-forward backtest for a single commodity via subprocess."""
    script = f"""
import json, sys, numpy as np
sys.path.insert(0, '.')
from strategy import run_strategy_backtest, TradeConfig

config = TradeConfig(
    confidence_threshold={cfg.confidence_threshold},
    stop_loss_pct=0.10,
    take_profit_multiplier=1.0,
    max_hold_days=63,
    allow_short=True,
    slippage_pct=0.003,
)

def _get(t, k, default=None):
    # Trade may be a dict or a dataclass instance
    if isinstance(t, dict):
        return t.get(k, default)
    return getattr(t, k, default)

try:
    result = run_strategy_backtest(config)
    if result is None:
        # Strategy returned None (typically: no trades generated)
        output = {{
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe": 0, "max_drawdown": 0, "total_return": 0,
            "avg_hold_days": 0, "avg_win": 0, "avg_loss": 0,
            "n_long": 0, "n_short": 0, "win_trades": [],
            "note": "no_trades",
        }}
    else:
        # Extract key metrics
        metrics = result.get("metrics", {{}}) if isinstance(result, dict) else {{}}
        trades = result.get("trades", []) if isinstance(result, dict) else []

        def _num(x):
            try:
                import numpy as _np
                if hasattr(x, 'item'):
                    return x.item()
            except Exception:
                pass
            return x

        output = {{
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": _num(metrics.get("win_rate", 0)),
            "profit_factor": _num(metrics.get("profit_factor", 0)),
            "sharpe": _num(metrics.get("sharpe", 0)),
            "max_drawdown": _num(metrics.get("max_drawdown", 0)),
            "total_return": _num(metrics.get("total_return", 0)),
            "avg_hold_days": _num(metrics.get("avg_hold_days", 0)),
            "avg_win": _num(metrics.get("avg_win", 0)),
            "avg_loss": _num(metrics.get("avg_loss", 0)),
            "n_long": sum(1 for t in trades if _get(t, "direction") == "LONG"),
            "n_short": sum(1 for t in trades if _get(t, "direction") == "SHORT"),
            "win_trades": [{{
                "direction": _get(t, "direction"),
                "entry_date": _get(t, "entry_date"),
                "exit_date": _get(t, "exit_date"),
                "pnl_pct": round(float(_get(t, "pnl_pct", 0) or 0), 4),
                "exit_reason": _get(t, "exit_reason"),
                "hold_days": _get(t, "hold_days"),
            }} for t in trades],
        }}
    print(json.dumps(output, default=float))
except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "tb": traceback.format_exc()[-500:]}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"{cfg.name} backtest failed: {result.stderr[-300:]}")
            return None
        data = json.loads(result.stdout.strip().split("\n")[-1])
        if "error" in data:
            logger.error(f"{cfg.name} backtest error: {data['error']}")
            return None
        data["commodity"] = cfg.name
        return data
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"{cfg.name} backtest output parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"{cfg.name} backtest exception: {e}")
        return None


def run_parameter_sweep(cfg: CommodityConfig) -> dict | None:
    """Run backtest with multiple parameter configurations to find optimal settings."""
    script = f"""
import json, sys, numpy as np
sys.path.insert(0, '.')
from strategy import _quick_backtest, TradeConfig

thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
results = []

for thresh in thresholds:
    config = TradeConfig(
        confidence_threshold=thresh,
        stop_loss_pct=0.10,
        take_profit_multiplier=1.0,
        max_hold_days=63,
        allow_short=True,
        slippage_pct=0.003,
    )
    try:
        metrics = _quick_backtest(config)
        if metrics:
            results.append({{
                "threshold": thresh,
                "total_trades": metrics.get("total_trades", 0),
                "win_rate": metrics.get("win_rate", 0),
                "sharpe": metrics.get("sharpe", 0),
                "total_return": metrics.get("total_return", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
            }})
    except Exception:
        pass

print(json.dumps(results))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            timeout=600,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return {"commodity": cfg.name, "sweep_results": data}
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"{cfg.name} sweep output parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"{cfg.name} sweep failed: {e}")
        return None


def compute_portfolio_metrics(backtest_results: dict) -> dict:
    """Compute portfolio-level metrics from individual commodity backtests."""
    all_trades = []
    commodity_returns = {}

    for key, result in backtest_results.items():
        if result is None:
            continue
        trades = result.get("win_trades", [])
        for t in trades:
            t["commodity"] = key
            all_trades.append(t)

        commodity_returns[key] = {
            "total_return": result.get("total_return", 0),
            "sharpe": result.get("sharpe", 0),
            "win_rate": result.get("win_rate", 0),
            "max_drawdown": result.get("max_drawdown", 0),
        }

    if not all_trades:
        return {"status": "no_trades"}

    # Sort by date
    all_trades.sort(key=lambda t: t.get("entry_date", ""))

    # Portfolio stats
    all_pnls = [t["pnl_pct"] for t in all_trades]
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]

    # Correlation check: how many trades overlap in correlated groups?
    overlap_count = 0
    for i, t1 in enumerate(all_trades):
        for t2 in all_trades[i+1:]:
            if t1.get("entry_date") == t2.get("entry_date"):
                # Check if same correlation group
                for group, members in CORRELATION_GROUPS.items():
                    if t1["commodity"] in members and t2["commodity"] in members:
                        overlap_count += 1

    return {
        "total_trades": len(all_trades),
        "portfolio_win_rate": round(len(wins) / len(all_trades), 4) if all_trades else 0,
        "portfolio_avg_pnl": round(np.mean(all_pnls), 4),
        "portfolio_total_pnl": round(sum(all_pnls), 4),
        "best_commodity": max(commodity_returns, key=lambda k: commodity_returns[k]["total_return"]) if commodity_returns else None,
        "worst_commodity": min(commodity_returns, key=lambda k: commodity_returns[k]["total_return"]) if commodity_returns else None,
        "correlated_overlap_trades": overlap_count,
        "per_commodity": commodity_returns,
    }


def backtest_all(
    commodity_keys: list[str] = None,
    sweep: bool = False,
    portfolio: bool = True,
) -> dict:
    """Run backtests for all (or specified) commodities."""
    targets = commodity_keys or list(COMMODITIES.keys())
    report = {
        "timestamp": datetime.now().isoformat(),
        "backtests": {},
        "sweeps": {},
    }

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        logger.info(f"Backtesting {cfg.name}...")
        result = run_commodity_backtest(cfg)
        report["backtests"][key] = result

        if result:
            wr = float(result.get("win_rate") or 0)
            sharpe = float(result.get("sharpe") or 0)
            trades = result.get("total_trades", 0)
            logger.info(f"  {cfg.name}: {trades} trades, WR={wr:.0%}, Sharpe={sharpe:.2f}")

        if sweep:
            logger.info(f"  Running parameter sweep for {cfg.name}...")
            sweep_result = run_parameter_sweep(cfg)
            report["sweeps"][key] = sweep_result

    if portfolio:
        logger.info("Computing portfolio metrics...")
        report["portfolio"] = compute_portfolio_metrics(report["backtests"])

    # Log to DB
    try:
        sys.path.insert(0, str(COMMODITIES_DIR.parent))
        from db import get_db
        db = get_db()
        run_id = db.start_agent_run("backtesting", targets)
        n_trades = sum(
            r.get("total_trades", 0) for r in report["backtests"].values() if r
        )
        db.finish_agent_run(run_id, "ok", summary=f"{n_trades} total trades across {len(targets)} commodities")
    except Exception:
        pass

    # Save report
    log_dir = COMMODITIES_DIR / "logs" / "backtests"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"backtest_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Backtest report saved to {report_path}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtesting agent")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--portfolio", action="store_true", help="Portfolio-level analysis only")
    args = parser.parse_args()

    report = backtest_all(
        commodity_keys=args.commodities or None,
        sweep=args.sweep,
        portfolio=True,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\n{'Commodity':<15} {'Trades':>6} {'Win%':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7}")
    print("-" * 55)
    for key, result in report["backtests"].items():
        cfg = COMMODITIES.get(key)
        if result is None:
            print(f"  {cfg.name:<15} FAILED")
            continue
        wr = float(result.get("win_rate") or 0)
        sh = float(result.get("sharpe") or 0)
        tr = float(result.get("total_return") or 0)
        dd = float(result.get("max_drawdown") or 0)
        print(f"  {cfg.name:<13} {result.get('total_trades', 0):>6} "
              f"{wr:>5.0%} {sh:>7.2f} "
              f"{tr:>7.1%} {dd:>6.1%}")

    port = report.get("portfolio", {})
    if port and port.get("total_trades", 0) > 0:
        print(f"\n--- Portfolio ---")
        print(f"  Total trades:    {port['total_trades']}")
        print(f"  Win rate:        {port['portfolio_win_rate']:.0%}")
        print(f"  Avg trade P&L:   {port['portfolio_avg_pnl']:+.2%}")
        print(f"  Total P&L:       {port['portfolio_total_pnl']:+.1%}")
        print(f"  Best commodity:  {port['best_commodity']}")
        print(f"  Worst commodity: {port['worst_commodity']}")
        if port.get("correlated_overlap_trades", 0) > 0:
            print(f"  Correlated overlaps: {port['correlated_overlap_trades']} (risk!)")

    if report.get("sweeps"):
        print(f"\n{'='*60}")
        print("PARAMETER SWEEP")
        print(f"{'='*60}")
        for key, sweep in report["sweeps"].items():
            if sweep and "sweep_results" in sweep:
                cfg = COMMODITIES.get(key)
                print(f"\n  {cfg.name}:")
                print(f"  {'Threshold':>10} {'Trades':>7} {'Win%':>6} {'Sharpe':>7} {'Return':>8}")
                for r in sweep["sweep_results"]:
                    print(f"  {r['threshold']:>10.0%} {r['total_trades']:>7} "
                          f"{r['win_rate']:>5.0%} {r['sharpe']:>7.2f} {r['total_return']:>7.1%}")


if __name__ == "__main__":
    main()
