"""Portfolio-level analysis — combined equity, correlation check, capital allocation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"


def load_backtest(project_dir: Path) -> list | None:
    strat_file = project_dir / "models" / "strategy_results.json"
    if not strat_file.exists():
        return None
    with open(strat_file) as f:
        return json.load(f)["trades"]


def simulate_combined_portfolio(
    coffee_trades: list,
    cocoa_trades: list,
    starting_equity: float = 10000.0,
    allocation: str = "equal",  # "equal", "sharpe_weighted", "inverse_vol"
):
    """Simulate a combined portfolio trading both commodities."""

    # Convert to DataFrames
    all_trades = []
    for t in (coffee_trades or []):
        all_trades.append({**t, "commodity": "Coffee"})
    for t in (cocoa_trades or []):
        all_trades.append({**t, "commodity": "Cocoa"})

    if not all_trades:
        return None

    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("entry_date")

    # Position sizing
    if allocation == "equal":
        coffee_pct = 0.50
        cocoa_pct = 0.50
    elif allocation == "sharpe_weighted":
        # Weight by backtest Sharpe ratio
        coffee_sharpe = 0.86  # from backtest
        cocoa_sharpe = 0.93
        total = coffee_sharpe + cocoa_sharpe
        coffee_pct = coffee_sharpe / total
        cocoa_pct = cocoa_sharpe / total
    else:
        coffee_pct = 0.50
        cocoa_pct = 0.50

    # Simulate
    equity = starting_equity
    equity_curve = [{"date": df["entry_date"].min(), "equity": equity}]
    trade_results = []

    for _, t in df.iterrows():
        commodity_pct = coffee_pct if t["commodity"] == "Coffee" else cocoa_pct
        position_size = equity * commodity_pct * 0.30  # 30% of commodity allocation per trade
        pnl = position_size * t["pnl_pct"]
        equity += pnl

        trade_results.append({
            "date": t["exit_date"],
            "commodity": t["commodity"],
            "direction": t["direction"],
            "pnl_pct": t["pnl_pct"],
            "pnl_dollar": round(pnl, 2),
            "equity_after": round(equity, 2),
        })
        equity_curve.append({"date": t["exit_date"], "equity": round(equity, 2)})

    # Metrics
    pnls = [t["pnl_pct"] for t in trade_results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = equity / starting_equity - 1
    n_years = max((df["exit_date"].max() - df["entry_date"].min()).days / 365.25, 0.01)
    cagr = (equity / starting_equity) ** (1 / n_years) - 1

    peak = starting_equity
    max_dd = 0
    for point in equity_curve:
        if point["equity"] > peak:
            peak = point["equity"]
        dd = (peak - point["equity"]) / peak
        if dd > max_dd:
            max_dd = dd

    trades_per_yr = len(pnls) / n_years
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(trades_per_yr)) if len(pnls) > 1 and np.std(pnls) > 0 else 0

    return {
        "allocation": f"Coffee {coffee_pct:.0%} / Cocoa {cocoa_pct:.0%}",
        "starting_equity": starting_equity,
        "final_equity": round(equity, 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 2),
        "total_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
        "avg_win": round(np.mean(wins), 4) if wins else 0,
        "avg_loss": round(np.mean(losses), 4) if losses else 0,
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        "coffee_trades": len([t for t in trade_results if t["commodity"] == "Coffee"]),
        "cocoa_trades": len([t for t in trade_results if t["commodity"] == "Cocoa"]),
        "equity_curve": equity_curve,
        "trades": trade_results,
    }


def check_correlation_risk(coffee_dir: Path, cocoa_dir: Path) -> dict:
    """Check if opening positions in both is too correlated."""
    try:
        coffee_df = pd.read_csv(coffee_dir / "data" / "combined_features.csv", index_col=0, parse_dates=True)
        cocoa_df = pd.read_csv(cocoa_dir / "data" / "combined_features.csv", index_col=0, parse_dates=True)

        coffee_ret = coffee_df["coffee_close"].pct_change()
        cocoa_ret = cocoa_df["cocoa_close"].pct_change()

        combined = pd.DataFrame({"coffee": coffee_ret, "cocoa": cocoa_ret}).dropna()
        corr_63d = combined["coffee"].rolling(63).corr(combined["cocoa"]).iloc[-1]
        corr_21d = combined["coffee"].rolling(21).corr(combined["cocoa"]).iloc[-1]
        corr_overall = combined["coffee"].corr(combined["cocoa"])

        risk_level = "LOW"
        if abs(corr_63d) > 0.6:
            risk_level = "HIGH"
        elif abs(corr_63d) > 0.3:
            risk_level = "MODERATE"

        return {
            "corr_21d": round(float(corr_21d), 3),
            "corr_63d": round(float(corr_63d), 3),
            "corr_overall": round(float(corr_overall), 3),
            "risk_level": risk_level,
            "recommendation": (
                "Reduce position sizes if going same direction in both"
                if risk_level == "HIGH" else
                "Monitor — some diversification benefit"
                if risk_level == "MODERATE" else
                "Good diversification — positions are relatively independent"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("PORTFOLIO ANALYSIS")
    print("=" * 60)

    # Load backtests
    coffee_trades = load_backtest(COFFEE_DIR)
    cocoa_trades = load_backtest(COCOA_DIR)

    print(f"\nLoaded: Coffee={len(coffee_trades or [])} trades, Cocoa={len(cocoa_trades or [])} trades")

    # Run simulations with different allocations
    for alloc in ["equal", "sharpe_weighted"]:
        result = simulate_combined_portfolio(coffee_trades, cocoa_trades, allocation=alloc)
        if result:
            print(f"\n--- {result['allocation']} ---")
            print(f"  Trades:     {result['total_trades']} (Coffee: {result['coffee_trades']}, Cocoa: {result['cocoa_trades']})")
            print(f"  Win rate:   {result['win_rate']:.0%}")
            print(f"  Profit fac: {result['profit_factor']:.2f}")
            print(f"  CAGR:       {result['cagr']:+.1%}")
            print(f"  Max DD:     {result['max_drawdown']:.1%}")
            print(f"  Sharpe:     {result['sharpe']:.2f}")
            print(f"  Equity:     ${result['starting_equity']:.0f} -> ${result['final_equity']:.0f}")

    # Correlation risk
    print(f"\n--- Correlation Risk ---")
    corr = check_correlation_risk(COFFEE_DIR, COCOA_DIR)
    if "error" not in corr:
        print(f"  21-day corr:  {corr['corr_21d']:.3f}")
        print(f"  63-day corr:  {corr['corr_63d']:.3f}")
        print(f"  Overall corr: {corr['corr_overall']:.3f}")
        print(f"  Risk level:   {corr['risk_level']}")
        print(f"  {corr['recommendation']}")


if __name__ == "__main__":
    main()
