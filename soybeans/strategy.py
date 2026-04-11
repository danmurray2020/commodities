"""Trading strategy backtester with entry/exit rules, position sizing, and risk metrics."""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.kelly import compute_kelly_size

from features import prepare_dataset
from train import walk_forward_split

OUTPUT_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

def _load_features():
    """Load feature list from production metadata."""
    for meta_file in ["v2_production_metadata.json", "production_metadata.json"]:
        meta_path = MODELS_DIR / meta_file
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if "features" in meta:
                return meta["features"]
    return []


SELECTED_FEATURES = _load_features()


@dataclass
class TradeConfig:
    """Trading strategy parameters."""
    confidence_threshold: float = 0.60   # min confidence to enter
    stop_loss_pct: float = 0.10          # max loss before exit (10%)
    take_profit_pct: float = None        # exit early if target hit (None = use model prediction)
    take_profit_multiplier: float = 1.0  # multiply predicted return for TP level
    max_hold_days: int = 63              # time exit (trading days)
    position_size: float = 1.0           # fraction of equity per trade (1.0 = full Kelly)
    kelly_fraction: float = 0.5          # half-Kelly by default
    allow_short: bool = False            # long-only by default
    scale_in: bool = False               # whether to scale into positions
    scale_in_confirm_days: int = 10      # days to wait for confirmation
    scale_in_threshold: float = 0.02     # price must move 2% in predicted direction
    slippage_pct: float = 0.003          # 0.3% slippage per trade (entry + exit)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: str
    exit_date: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    predicted_return: float
    confidence: float
    exit_reason: str  # "time", "stop_loss", "take_profit"
    pnl_pct: float = 0.0
    hold_days: int = 0
    position_size: float = 1.0


def run_strategy_backtest(config: TradeConfig = None):
    """Run a full walk-forward strategy backtest.

    For each CV fold:
    1. Train model on training data
    2. Generate signals on test data
    3. Simulate trades with entry/exit rules
    """
    if config is None:
        config = TradeConfig()

    print("=" * 60)
    print("STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Confidence threshold: {config.confidence_threshold:.0%}")
    print(f"Stop loss: {config.stop_loss_pct:.0%}")
    print(f"Max hold: {config.max_hold_days} days")
    print(f"Kelly fraction: {config.kelly_fraction}")
    print(f"Allow short: {config.allow_short}")
    print(f"Scale-in: {config.scale_in}")

    # Prepare data
    df, all_cols = prepare_dataset(horizon=63)
    feature_cols = [f for f in SELECTED_FEATURES if f in all_cols]
    X = df[feature_cols].values
    prices = df["soybeans_close"].values
    dates = df.index

    # Load tuned params
    with open(MODELS_DIR / "production_metadata.json") as f:
        meta = json.load(f)
    reg_params = {**meta["regression"]["params"], "early_stopping_rounds": 30, "random_state": 42}
    clf_params = {**meta["classification"]["params"], "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}

    # Walk-forward with more granular test windows for realistic simulation
    splits = walk_forward_split(df, n_splits=12, test_size=63, min_train_size=504, purge_gap=63)
    print(f"\n{len(splits)} walk-forward folds\n")

    all_trades: list[Trade] = []
    equity_curve = [100.0]  # start with $100
    equity_dates = [dates[0]]

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Train models
        reg = XGBRegressor(**reg_params)
        reg.fit(X[train_idx], df["target_return"].values[train_idx],
                eval_set=[(X[test_idx], df["target_return"].values[test_idx])], verbose=False)

        clf = XGBClassifier(**clf_params)
        clf.fit(X[train_idx], df["target_direction"].values[train_idx],
                eval_set=[(X[test_idx], df["target_direction"].values[test_idx])], verbose=False)

        # Generate signals for test period
        test_start = test_idx[0]
        test_end = test_idx[-1]

        i = test_start
        while i <= test_end:
            # Check if we're too close to end of data for a full trade
            if i + config.max_hold_days >= len(prices):
                break

            X_i = X[[i]]
            pred_return = float(reg.predict(X_i)[0])
            pred_dir = int(clf.predict(X_i)[0])
            pred_proba = clf.predict_proba(X_i)[0]
            confidence = float(pred_proba[pred_dir])

            # Entry decision
            if confidence < config.confidence_threshold:
                i += 1
                continue

            if pred_dir == 0 and not config.allow_short:
                i += 1
                continue

            direction = "LONG" if pred_dir == 1 else "SHORT"
            # Apply slippage to entry
            raw_entry = prices[i]
            if direction == "LONG":
                entry_price = raw_entry * (1 + config.slippage_pct)
            else:
                entry_price = raw_entry * (1 - config.slippage_pct)
            entry_date = dates[i]

            # Determine take-profit level
            if config.take_profit_pct is not None:
                tp_pct = config.take_profit_pct
            else:
                tp_pct = abs(pred_return) * config.take_profit_multiplier

            # Scale-in logic
            initial_size = 0.33 if config.scale_in else 1.0
            current_size = initial_size
            scaled_in = False

            # Simulate the trade day by day
            exit_reason = "time"
            exit_price = entry_price
            hold_days = 0

            for day in range(1, config.max_hold_days + 1):
                if i + day >= len(prices):
                    hold_days = day - 1
                    exit_price = prices[i + hold_days]
                    exit_reason = "data_end"
                    break

                current_price = prices[i + day]
                hold_days = day

                if direction == "LONG":
                    current_return = (current_price / entry_price) - 1
                else:
                    current_return = (entry_price / current_price) - 1

                # Scale-in check
                if config.scale_in and not scaled_in and day <= config.scale_in_confirm_days:
                    if current_return >= config.scale_in_threshold:
                        current_size = 0.66
                        scaled_in = True

                # Stop loss
                if current_return <= -config.stop_loss_pct:
                    exit_price = current_price
                    exit_reason = "stop_loss"
                    break

                # Take profit
                if current_return >= tp_pct:
                    exit_price = current_price
                    exit_reason = "take_profit"
                    break

                # Time exit
                if day == config.max_hold_days:
                    exit_price = current_price
                    exit_reason = "time"
                    break

            # Apply slippage to exit
            if direction == "LONG":
                exit_price = exit_price * (1 - config.slippage_pct)
            else:
                exit_price = exit_price * (1 + config.slippage_pct)

            # Calculate PnL
            if direction == "LONG":
                pnl_pct = (exit_price / entry_price) - 1
            else:
                pnl_pct = (entry_price / exit_price) - 1

            trade = Trade(
                entry_date=str(entry_date.date()),
                exit_date=str(dates[i + hold_days].date()) if i + hold_days < len(dates) else "N/A",
                direction=direction,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                predicted_return=round(pred_return, 4),
                confidence=round(confidence, 4),
                exit_reason=exit_reason,
                pnl_pct=round(pnl_pct, 4),
                hold_days=hold_days,
                position_size=round(current_size, 2),
            )
            all_trades.append(trade)

            # Update equity
            sized_pnl = pnl_pct * current_size * config.position_size
            new_equity = equity_curve[-1] * (1 + sized_pnl)
            equity_curve.append(new_equity)
            equity_dates.append(dates[i + hold_days] if i + hold_days < len(dates) else dates[-1])

            # Skip ahead past this trade
            i += hold_days + 1
            continue

        # If no trades in this fold, advance
        i = test_end + 1

    # Compute metrics
    if not all_trades:
        print("\nNo trades generated. Try lowering confidence_threshold.")
        return

    pnls = [t.pnl_pct for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    n_years = (equity_dates[-1] - equity_dates[0]).days / 365.25
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized)
    trade_returns = [t.pnl_pct * t.position_size for t in all_trades]
    if len(trade_returns) > 1:
        trades_per_year = len(all_trades) / n_years if n_years > 0 else 4
        sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    kelly = compute_kelly_size(win_rate, avg_win, abs(avg_loss), config.kelly_fraction)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total trades:     {len(all_trades)}")
    print(f"Win rate:         {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win:          {avg_win:+.2%}")
    print(f"Avg loss:         {np.mean(losses):+.2%}" if losses else "Avg loss:         N/A")
    print(f"Profit factor:    {profit_factor:.2f}")
    print(f"Kelly size:       {kelly:.1%}")
    print(f"")
    print(f"Total return:     {total_return:+.1%}")
    print(f"CAGR:             {cagr:+.1%}")
    print(f"Max drawdown:     {max_dd:.1%}")
    print(f"Sharpe ratio:     {sharpe:.2f}")
    print(f"")
    print(f"Avg hold (days):  {np.mean([t.hold_days for t in all_trades]):.0f}")
    print(f"Trades/year:      {len(all_trades)/n_years:.1f}" if n_years > 0 else "")

    # Exit reason breakdown
    reasons = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\nExit reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        reason_trades = [t for t in all_trades if t.exit_reason == reason]
        reason_wr = np.mean([t.pnl_pct > 0 for t in reason_trades])
        print(f"  {reason:15s}: {count:3d} trades, win rate {reason_wr:.0%}")

    # Trade log
    print(f"\nTrade Log:")
    print(f"{'Entry':<12} {'Exit':<12} {'Dir':<6} {'Entry$':>8} {'Exit$':>8} {'PnL':>8} {'Conf':>6} {'Days':>5} {'Reason':<12}")
    print("-" * 85)
    for t in all_trades:
        pnl_color = "+" if t.pnl_pct > 0 else ""
        print(f"{t.entry_date:<12} {t.exit_date:<12} {t.direction:<6} "
              f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
              f"{pnl_color}{t.pnl_pct:>7.2%} {t.confidence:>5.1%} {t.hold_days:>5d} {t.exit_reason:<12}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Equity curve
    axes[0].plot(equity_dates, equity_curve, linewidth=2, color="#a78bfa")
    axes[0].set_title(f"Equity Curve (${equity_curve[0]:.0f} → ${equity_curve[-1]:.0f}, "
                      f"CAGR={cagr:+.1%}, Sharpe={sharpe:.2f})")
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(100, color="gray", linestyle="--", alpha=0.3)

    # Trade PnL distribution
    axes[1].bar(range(len(pnls)), pnls,
                color=["#22c55e" if p > 0 else "#ef4444" for p in pnls], alpha=0.8)
    axes[1].set_title(f"Trade PnL (Win Rate: {win_rate:.0%}, Profit Factor: {profit_factor:.2f})")
    axes[1].set_ylabel("Return (%)")
    axes[1].set_xlabel("Trade #")
    axes[1].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    # Price chart with entry/exit markers
    all_prices_df = pd.read_csv(str(OUTPUT_DIR / "combined_features.csv"), index_col=0, parse_dates=True)
    test_period = all_prices_df.loc[all_prices_df.index >= dates[splits[0][1][0]]]
    axes[2].plot(test_period.index, test_period["soybeans_close"], linewidth=1, color="#71717a", alpha=0.7)
    for t in all_trades:
        entry_d = pd.Timestamp(t.entry_date)
        exit_d = pd.Timestamp(t.exit_date)
        color = "#22c55e" if t.pnl_pct > 0 else "#ef4444"
        marker = "^" if t.direction == "LONG" else "v"
        axes[2].plot(entry_d, t.entry_price, marker=marker, color=color, markersize=10, zorder=5)
        axes[2].plot(exit_d, t.exit_price, "x", color=color, markersize=8, zorder=5)
        axes[2].plot([entry_d, exit_d], [t.entry_price, t.exit_price],
                     linestyle="--", color=color, alpha=0.4, linewidth=1)
    axes[2].set_title("Trade Entries (triangles) and Exits (x)")
    axes[2].set_ylabel("Soybeans Price ($/bu)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_backtest.png", dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR / 'strategy_backtest.png'}")

    # Save results
    results = {
        "config": {
            "confidence_threshold": config.confidence_threshold,
            "stop_loss_pct": config.stop_loss_pct,
            "max_hold_days": config.max_hold_days,
            "kelly_fraction": config.kelly_fraction,
            "allow_short": config.allow_short,
            "scale_in": config.scale_in,
        },
        "metrics": {
            "total_trades": len(all_trades),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(float(np.mean(losses)), 4) if losses else None,
            "profit_factor": round(profit_factor, 2),
            "kelly_size": round(kelly, 4),
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe": round(sharpe, 2),
        },
        "trades": [
            {
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "direction": t.direction, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "pnl_pct": t.pnl_pct,
                "confidence": t.confidence, "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
            }
            for t in all_trades
        ],
    }
    with open(MODELS_DIR / "strategy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {MODELS_DIR / 'strategy_results.json'}")

    plt.show()

    return results


def run_parameter_sweep():
    """Sweep key strategy parameters to find optimal configuration."""
    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)

    configs = [
        ("Baseline (60% conf, 10% SL)", TradeConfig(confidence_threshold=0.60, stop_loss_pct=0.10)),
        ("High conf (70%, 10% SL)", TradeConfig(confidence_threshold=0.70, stop_loss_pct=0.10)),
        ("Tight SL (60% conf, 5% SL)", TradeConfig(confidence_threshold=0.60, stop_loss_pct=0.05)),
        ("Wide SL (60% conf, 15% SL)", TradeConfig(confidence_threshold=0.60, stop_loss_pct=0.15)),
        ("Scale-in (60% conf, 10% SL)", TradeConfig(confidence_threshold=0.60, stop_loss_pct=0.10, scale_in=True)),
        ("With shorts (60% conf, 10% SL)", TradeConfig(confidence_threshold=0.60, stop_loss_pct=0.10, allow_short=True)),
        ("Conservative (70% conf, 5% SL, 0.25 Kelly)", TradeConfig(confidence_threshold=0.70, stop_loss_pct=0.05, kelly_fraction=0.25)),
    ]

    # Suppress output during sweep — just collect results
    import io
    import sys

    summary = []
    for name, config in configs:
        print(f"\nTesting: {name}...")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # We need a simplified version that just returns metrics
            result = _quick_backtest(config)
            summary.append({"name": name, **result})
        finally:
            sys.stdout = old_stdout

    # Print comparison
    print(f"\n{'Config':<45} {'Trades':>6} {'WinR':>6} {'CAGR':>8} {'MaxDD':>7} {'Sharpe':>7} {'PF':>6}")
    print("-" * 95)
    for s in summary:
        print(f"{s['name']:<45} {s['n_trades']:>6d} {s['win_rate']:>5.0%} "
              f"{s['cagr']:>+7.1%} {s['max_dd']:>6.1%} {s['sharpe']:>7.2f} {s['pf']:>6.2f}")


def _quick_backtest(config: TradeConfig) -> dict:
    """Simplified backtest that returns just metrics (no plots/prints)."""
    df, all_cols = prepare_dataset(horizon=63)
    feature_cols = [f for f in SELECTED_FEATURES if f in all_cols]
    X = df[feature_cols].values
    prices = df["soybeans_close"].values
    dates = df.index

    with open(MODELS_DIR / "production_metadata.json") as f:
        meta = json.load(f)
    reg_params = {**meta["regression"]["params"], "early_stopping_rounds": 30, "random_state": 42}
    clf_params = {**meta["classification"]["params"], "eval_metric": "logloss",
                  "early_stopping_rounds": 30, "random_state": 42}

    splits = walk_forward_split(df, n_splits=12, test_size=63, min_train_size=504, purge_gap=63)
    pnls = []
    equity = [100.0]

    for train_idx, test_idx in splits:
        reg = XGBRegressor(**reg_params)
        reg.fit(X[train_idx], df["target_return"].values[train_idx],
                eval_set=[(X[test_idx], df["target_return"].values[test_idx])], verbose=False)
        clf = XGBClassifier(**clf_params)
        clf.fit(X[train_idx], df["target_direction"].values[train_idx],
                eval_set=[(X[test_idx], df["target_direction"].values[test_idx])], verbose=False)

        i = test_idx[0]
        while i <= test_idx[-1]:
            if i + config.max_hold_days >= len(prices):
                break
            X_i = X[[i]]
            pred_return = float(reg.predict(X_i)[0])
            pred_dir = int(clf.predict(X_i)[0])
            confidence = float(clf.predict_proba(X_i)[0][pred_dir])

            if confidence < config.confidence_threshold:
                i += 1
                continue
            if pred_dir == 0 and not config.allow_short:
                i += 1
                continue

            direction = "LONG" if pred_dir == 1 else "SHORT"
            entry_price = prices[i]
            entry_price = entry_price * (1 + config.slippage_pct) if direction == "LONG" else entry_price * (1 - config.slippage_pct)
            tp_pct = abs(pred_return) * config.take_profit_multiplier

            exit_price = entry_price
            hold_days = 0
            for day in range(1, config.max_hold_days + 1):
                if i + day >= len(prices):
                    break
                current_price = prices[i + day]
                hold_days = day
                ret = (current_price / entry_price - 1) if direction == "LONG" else (entry_price / current_price - 1)
                if ret <= -config.stop_loss_pct:
                    exit_price = current_price
                    break
                if ret >= tp_pct:
                    exit_price = current_price
                    break
                if day == config.max_hold_days:
                    exit_price = current_price

            exit_price = exit_price * (1 - config.slippage_pct) if direction == "LONG" else exit_price * (1 + config.slippage_pct)
            pnl = (exit_price / entry_price - 1) if direction == "LONG" else (entry_price / exit_price - 1)
            pnls.append(pnl)
            equity.append(equity[-1] * (1 + pnl * config.kelly_fraction))
            i += hold_days + 1

    if not pnls:
        return {"n_trades": 0, "win_rate": 0, "cagr": 0, "max_dd": 0, "sharpe": 0, "pf": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n_years = max((dates[splits[-1][1][-1]] - dates[splits[0][1][0]]).days / 365.25, 0.01)
    total_ret = equity[-1] / equity[0]
    cagr = total_ret ** (1 / n_years) - 1
    peak = equity[0]
    max_dd = 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd
    trades_per_yr = len(pnls) / n_years
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(trades_per_yr)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    return {
        "n_trades": len(pnls),
        "win_rate": len(wins) / len(pnls),
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "pf": pf,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        run_parameter_sweep()
    else:
        run_strategy_backtest()
