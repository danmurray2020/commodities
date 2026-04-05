"""Backtest: trade equities based on commodity model signals.

Uses commodity model predictions to trade the most sensitive equities.
Only trades when commodity confidence exceeds calibrated threshold.
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

COMMODITIES_DIR = Path(__file__).parent

# Best equity plays per commodity (from scanner)
EQUITY_TRADES = {
    "copper": [
        {"ticker": "FCX", "role": "producer", "beta": 0.89, "direction": "same"},
        {"ticker": "SCCO", "role": "producer", "beta": 0.80, "direction": "same"},
    ],
    "natgas": [
        {"ticker": "EQT", "role": "producer", "beta": 0.23, "direction": "same"},
        {"ticker": "RRC", "role": "producer", "beta": 0.22, "direction": "same"},
    ],
    "wheat": [
        {"ticker": "BG", "role": "producer", "beta": 0.10, "direction": "same"},
    ],
    "soybeans": [
        {"ticker": "BG", "role": "producer", "beta": 0.23, "direction": "same"},
        {"ticker": "ADM", "role": "producer", "beta": 0.21, "direction": "same"},
    ],
    "cocoa": [
        {"ticker": "HSY", "role": "consumer", "beta": -0.07, "direction": "inverse"},
    ],
}

COMMODITY_CONFIGS = {
    "coffee": {"dir": "coffee", "price_col": "coffee_close", "ticker": "KC=F"},
    "cocoa": {"dir": "chocolate", "price_col": "cocoa_close", "ticker": "CC=F"},
    "sugar": {"dir": "sugar", "price_col": "sugar_close", "ticker": "SB=F"},
    "natgas": {"dir": "natgas", "price_col": "natgas_close", "ticker": "NG=F"},
    "soybeans": {"dir": "soybeans", "price_col": "soybeans_close", "ticker": "ZS=F"},
    "wheat": {"dir": "wheat", "price_col": "wheat_close", "ticker": "ZW=F"},
    "copper": {"dir": "copper", "price_col": "copper_close", "ticker": "HG=F"},
}


def get_commodity_signals(commodity: str, config: dict) -> pd.DataFrame | None:
    """Generate walk-forward commodity signals via subprocess."""
    project_dir = COMMODITIES_DIR.parent / config["dir"]
    price_col = config["price_col"]

    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col='{price_col}', horizon=63)
df = df.dropna()

models_dir = 'models'
for meta_file in ['v3_production_metadata.json', 'v2_production_metadata.json', 'production_metadata.json']:
    try:
        with open(f'{{models_dir}}/{{meta_file}}') as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue

feature_cols = [f for f in meta['features'] if f in df.columns]
clf_params = meta['classification'].get('params', {{}})
reg_params = meta['regression'].get('params', {{}})

X = df[feature_cols].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values
n = len(X)

# Walk-forward: generate out-of-sample signals
min_train = 504
purge = 63
step = 21

signals = []
i = min_train
while i + purge < n:
    test_end = min(i + purge + step, n)
    test_start = i + purge

    if test_start >= n:
        break

    # Train
    c_params = {{k:v for k,v in clf_params.items() if k not in ['eval_metric','early_stopping_rounds']}}
    c_params['eval_metric'] = 'logloss'
    c_params['early_stopping_rounds'] = 30
    c_params['random_state'] = 42

    clf = XGBClassifier(**c_params)
    clf.fit(X[:i], y_dir[:i], eval_set=[(X[test_start:test_end], y_dir[test_start:test_end])], verbose=False)

    r_params = {{k:v for k,v in reg_params.items() if k not in ['early_stopping_rounds','eval_metric','scale_pos_weight']}}
    r_params['early_stopping_rounds'] = 30
    r_params['random_state'] = 42
    reg = XGBRegressor(**r_params)
    reg.fit(X[:i], y_ret[:i], eval_set=[(X[test_start:test_end], y_ret[test_start:test_end])], verbose=False)

    # Predict
    proba = clf.predict_proba(X[test_start:test_end])[:, 1]
    pred_dir = clf.predict(X[test_start:test_end])
    pred_ret = reg.predict(X[test_start:test_end])

    for j in range(test_end - test_start):
        idx = test_start + j
        confidence = float(max(proba[j], 1 - proba[j]))
        signals.append({{
            'date': df.index[idx].strftime('%Y-%m-%d'),
            'commodity_price': float(df['{price_col}'].iloc[idx]),
            'pred_direction': 'UP' if pred_dir[j] == 1 else 'DOWN',
            'pred_return': float(pred_ret[j]),
            'confidence': confidence,
            'actual_direction': 'UP' if y_dir[idx] == 1 else 'DOWN',
            'actual_return': float(y_ret[idx]),
        }})

    i += step

print(json.dumps(signals))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=600,
    )
    if result.returncode != 0:
        print(f"  ERROR generating signals for {commodity}: {result.stderr[-200:]}")
        return None
    try:
        signals = json.loads(result.stdout.strip().split("\n")[-1])
        return pd.DataFrame(signals).set_index("date")
    except (json.JSONDecodeError, IndexError):
        return None


def fetch_equity_prices(tickers: list) -> pd.DataFrame:
    """Fetch historical equity prices."""
    print(f"  Fetching equity prices for {tickers}...")
    all_prices = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                all_prices[ticker] = df["Close"]
        except Exception:
            pass
    return pd.DataFrame(all_prices)


def simulate_equity_trades(
    signals: pd.DataFrame,
    equity_prices: pd.DataFrame,
    equity_config: dict,
    confidence_threshold: float = 0.75,
    hold_days: int = 63,
    stop_loss_pct: float = 0.15,
) -> list:
    """Simulate equity trades based on commodity signals."""
    ticker = equity_config["ticker"]
    direction_type = equity_config["direction"]  # "same" or "inverse"

    if ticker not in equity_prices.columns:
        return []

    eq_prices = equity_prices[ticker].dropna()
    trades = []
    last_exit_date = None

    for date_str, row in signals.iterrows():
        date = pd.Timestamp(date_str)

        # Skip if we're still in a position
        if last_exit_date and date <= last_exit_date:
            continue

        # Check confidence threshold
        if row["confidence"] < confidence_threshold:
            continue

        # Determine equity trade direction
        commodity_direction = row["pred_direction"]
        if direction_type == "same":
            equity_direction = commodity_direction
        else:
            equity_direction = "DOWN" if commodity_direction == "UP" else "UP"

        # Find entry price (next available trading day)
        future_prices = eq_prices[eq_prices.index >= date]
        if len(future_prices) < 2:
            continue

        entry_date = future_prices.index[0]
        entry_price = float(future_prices.iloc[0])

        # Simulate hold period
        exit_reason = "time"
        exit_price = entry_price
        exit_date = entry_date
        hold = 0

        for day_idx in range(1, min(hold_days + 1, len(future_prices))):
            current_price = float(future_prices.iloc[day_idx])
            current_date = future_prices.index[day_idx]
            hold = day_idx

            if equity_direction == "UP":
                current_return = (current_price / entry_price) - 1
            else:
                current_return = (entry_price / current_price) - 1

            # Stop loss
            if current_return <= -stop_loss_pct:
                exit_price = current_price
                exit_date = current_date
                exit_reason = "stop_loss"
                break

            # Take profit (use predicted commodity return as guide)
            tp_pct = abs(row["pred_return"]) * equity_config["beta"]
            if tp_pct > 0.02 and current_return >= tp_pct:
                exit_price = current_price
                exit_date = current_date
                exit_reason = "take_profit"
                break

            if day_idx == hold_days or day_idx == len(future_prices) - 1:
                exit_price = current_price
                exit_date = current_date
                exit_reason = "time"

        # Calculate PnL
        if equity_direction == "UP":
            pnl_pct = (exit_price / entry_price) - 1
        else:
            pnl_pct = (entry_price / exit_price) - 1

        trades.append({
            "commodity_signal_date": date_str,
            "commodity_direction": commodity_direction,
            "commodity_confidence": round(row["confidence"], 3),
            "equity_ticker": ticker,
            "equity_direction": equity_direction,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 4),
            "hold_days": hold,
            "exit_reason": exit_reason,
        })

        last_exit_date = exit_date

    return trades


def main():
    print("=" * 60)
    print("EQUITY BACKTEST — Commodity Signals → Equity Trades")
    print("=" * 60)

    # Collect all unique equity tickers
    all_tickers = set()
    for equities in EQUITY_TRADES.values():
        for eq in equities:
            all_tickers.add(eq["ticker"])

    # Fetch all equity prices once
    print(f"\nFetching {len(all_tickers)} equity prices...")
    equity_prices = fetch_equity_prices(list(all_tickers))
    print(f"  Got data for: {list(equity_prices.columns)}")

    all_trades = []
    commodity_results = {}

    for commodity, equities in EQUITY_TRADES.items():
        config = COMMODITY_CONFIGS.get(commodity)
        if not config:
            continue

        print(f"\n{'='*60}")
        print(f"  {commodity.upper()}")
        print(f"{'='*60}")

        # Generate commodity signals
        print(f"  Generating walk-forward signals...")
        signals = get_commodity_signals(commodity, config)
        if signals is None or len(signals) == 0:
            print(f"  No signals generated")
            continue

        high_conf = (signals["confidence"] >= 0.75).sum()
        print(f"  {len(signals)} total predictions, {high_conf} above 75% confidence")

        # Trade each equity
        for eq_config in equities:
            ticker = eq_config["ticker"]
            print(f"\n  Trading {ticker} (beta={eq_config['beta']:+.2f}, {eq_config['direction']})...")

            trades = simulate_equity_trades(
                signals, equity_prices, eq_config,
                confidence_threshold=0.75,
                hold_days=63,
                stop_loss_pct=0.15,
            )

            if not trades:
                print(f"    No trades")
                continue

            pnls = [t["pnl_pct"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

            print(f"    Trades: {len(trades)}, Win rate: {win_rate:.0%}")
            print(f"    Avg win: {avg_win:+.2%}, Avg loss: {avg_loss:+.2%}, PF: {pf:.2f}")

            for t in trades:
                marker = "W" if t["pnl_pct"] > 0 else "L"
                print(f"    [{marker}] {t['entry_date']} → {t['exit_date']}: "
                      f"{t['equity_direction']} {ticker} ${t['entry_price']} → ${t['exit_price']} "
                      f"({t['pnl_pct']:+.2%}, {t['exit_reason']}, comm_conf={t['commodity_confidence']:.0%})")

            all_trades.extend(trades)
            commodity_results[f"{commodity}→{ticker}"] = {
                "trades": len(trades), "win_rate": round(win_rate, 3),
                "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
                "profit_factor": round(pf, 2),
            }

    # Portfolio summary
    if all_trades:
        print(f"\n{'='*60}")
        print("PORTFOLIO SUMMARY — All Equity Trades")
        print(f"{'='*60}")

        all_pnls = [t["pnl_pct"] for t in all_trades]
        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]
        win_rate = len(wins) / len(all_pnls)
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

        # Equity curve
        equity = 10000.0
        for t in sorted(all_trades, key=lambda x: x["entry_date"]):
            pnl = equity * 0.10 * t["pnl_pct"]  # 10% position sizing
            equity += pnl

        dates = sorted(set(t["entry_date"] for t in all_trades))
        n_years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 0.01)
        cagr = (equity / 10000) ** (1 / n_years) - 1

        print(f"  Total trades:    {len(all_trades)}")
        print(f"  Win rate:        {win_rate:.0%} ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg win:         {np.mean(wins):+.2%}")
        print(f"  Avg loss:        {np.mean(losses):+.2%}")
        print(f"  Profit factor:   {pf:.2f}")
        print(f"  Equity:          $10,000 → ${equity:,.0f}")
        print(f"  CAGR:            {cagr:+.1%}")

        print(f"\n  By commodity→equity:")
        print(f"  {'Pair':<25} {'Trades':>7} {'WinR':>6} {'PF':>6}")
        print(f"  {'-'*46}")
        for pair, stats in sorted(commodity_results.items(), key=lambda x: -x[1]["profit_factor"]):
            print(f"  {pair:<25} {stats['trades']:>7} {stats['win_rate']:>5.0%} {stats['profit_factor']:>6.2f}")

    # Save
    output = {
        "scan_date": str(datetime.now().date()),
        "total_trades": len(all_trades),
        "commodity_results": commodity_results,
        "trades": all_trades,
    }
    output_path = COMMODITIES_DIR / "equity_data" / "equity_backtest.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
