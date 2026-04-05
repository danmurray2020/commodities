"""Paper trading system — logs predictions and tracks hypothetical P&L.

Run weekly alongside the alert system. Tracks:
- Every prediction made (whether traded or not)
- Open positions with entry/TP/SL levels
- Closed positions with realized P&L
- Running equity curve
"""

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"
SOYBEANS_DIR = Path(__file__).parent.parent / "soybeans"
WHEAT_DIR = Path(__file__).parent.parent / "wheat"
COPPER_DIR = Path(__file__).parent.parent / "copper"
TRADES_FILE = Path(__file__).parent / "paper_trades.json"
PREDICTIONS_LOG = Path(__file__).parent / "predictions.log"

STARTING_EQUITY = 10000.0
CONFIG_FILE = Path(__file__).parent / "optimal_config.json"

# Equity trades triggered by commodity signals (from backtest validation)
EQUITY_TRADES = {
    "Natural Gas": [
        {"ticker": "EQT", "beta": 0.23, "direction": "same"},
        {"ticker": "RRC", "beta": 0.22, "direction": "same"},
    ],
    "Wheat": [
        {"ticker": "BG", "beta": 0.10, "direction": "same"},
    ],
}
EQUITY_POSITION_PCT = 0.05  # 5% per equity trade (smaller than commodity trades)


def load_config() -> dict:
    """Load calibrated trading config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def get_position_size(confidence: float, commodity: str, config: dict) -> float:
    """Get position size based on calibrated confidence bands."""
    commodity_cfg = config.get(commodity.lower(), {})
    sizing = commodity_cfg.get("position_sizing", {})

    if confidence >= 0.90:
        return sizing.get("90%+", 0.25)
    elif confidence >= 0.85:
        return sizing.get("85-90%", 0.25)
    elif confidence >= 0.80:
        return sizing.get("80-85%", 0.20)
    elif confidence >= 0.75:
        return sizing.get("75-80%", 0.15)
    elif confidence >= 0.70:
        return sizing.get("70-75%", 0.08)
    return 0.05


def load_trades() -> dict:
    """Load existing paper trade state."""
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return {
        "equity": STARTING_EQUITY,
        "open_positions": [],
        "closed_positions": [],
        "prediction_history": [],
    }


def save_trades(state: dict):
    with open(TRADES_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_latest_prediction(project_dir: Path, price_col: str) -> dict | None:
    """Get prediction from the latest data point via subprocess."""
    script = f"""
import json, sys, joblib, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = df.dropna()

models_dir = 'models'
for meta_file in ['v2_production_metadata.json', 'production_metadata.json']:
    try:
        with open(f'{{models_dir}}/{{meta_file}}') as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue
else:
    sys.exit(1)

version = 'v2' if 'v2' in meta_file else 'v1'
reg_file = f'{{models_dir}}/v2_production_regressor.joblib' if version == 'v2' else f'{{models_dir}}/production_regressor.joblib'
clf_file = f'{{models_dir}}/v2_production_classifier.joblib' if version == 'v2' else f'{{models_dir}}/production_classifier.joblib'

reg = joblib.load(reg_file)
clf = joblib.load(clf_file)
available = [f for f in meta['features'] if f in df.columns]

latest = df.iloc[[-1]]
X = latest[available].values

pred_return = float(reg.predict(X)[0])
pred_dir = int(clf.predict(X)[0])
pred_proba = clf.predict_proba(X)[0]
confidence = float(pred_proba[pred_dir])
current_price = float(latest['{price_col}'].values[0])

strategy = meta.get('strategy', {{'confidence_threshold': 0.70, 'stop_loss_pct': 0.10}})

result = {{
    'date': latest.index[0].strftime('%Y-%m-%d'),
    'price': current_price,
    'pred_return': pred_return,
    'pred_price': current_price * (1 + pred_return),
    'direction': 'UP' if pred_dir == 1 else 'DOWN',
    'confidence': confidence,
    'threshold': strategy.get('confidence_threshold', 0.70),
    'stop_loss_pct': strategy.get('stop_loss_pct', 0.10),
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=120,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def check_open_positions(state: dict, commodities: dict):
    """Check if any open positions hit TP, SL, or time exit."""
    closed = []
    still_open = []

    for pos in state["open_positions"]:
        name = pos["commodity"]
        current_price = commodities.get(name, {}).get("price")
        if current_price is None:
            still_open.append(pos)
            continue

        entry_price = pos["entry_price"]
        direction = pos["direction"]

        if direction == "LONG":
            current_return = (current_price / entry_price) - 1
        else:
            current_return = (entry_price / current_price) - 1

        # Check exits
        days_held = (datetime.now() - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days
        exit_reason = None

        if current_return <= -pos["stop_loss_pct"]:
            exit_reason = "stop_loss"
        elif current_return >= pos["take_profit_pct"]:
            exit_reason = "take_profit"
        elif days_held >= 90:  # ~63 trading days
            exit_reason = "time"

        if exit_reason:
            pnl_pct = current_return
            pnl_dollar = pos["position_size"] * pnl_pct
            state["equity"] += pnl_dollar

            closed_pos = {
                **pos,
                "exit_date": datetime.now().strftime("%Y-%m-%d"),
                "exit_price": current_price,
                "exit_reason": exit_reason,
                "pnl_pct": round(pnl_pct, 4),
                "pnl_dollar": round(pnl_dollar, 2),
                "days_held": days_held,
            }
            state["closed_positions"].append(closed_pos)
            closed.append(closed_pos)
            print(f"  CLOSED: {name} {direction} — {exit_reason} — PnL: {pnl_pct:+.2%} (${pnl_dollar:+.2f})")
        else:
            pos["current_price"] = current_price
            pos["current_return"] = round(current_return, 4)
            pos["days_held"] = days_held
            still_open.append(pos)

    state["open_positions"] = still_open
    return closed


def open_new_positions(state: dict, commodities: dict):
    """Open new positions using OOS-calibrated thresholds and correlation-aware sizing."""
    from position_manager import generate_position_plan

    open_names = {p["commodity"] for p in state["open_positions"]}

    plan = generate_position_plan(commodities)
    if plan.get("status") == "NO TRADES":
        print("  No signals above OOS-calibrated thresholds")
        return

    for pos in plan.get("positions", []):
        name = pos["commodity"]
        if name in open_names:
            continue

        pred = commodities[name]
        direction = pos["direction"]
        final_size_pct = pos["final_size"]
        position_size = state["equity"] * final_size_pct
        tp_pct = abs(pred.get("pred_return", 0))
        sl_pct = pred.get("stop_loss_pct", 0.10)

        position = {
            "commodity": name,
            "direction": direction,
            "entry_date": pred["date"],
            "entry_price": pred["price"],
            "take_profit_pct": round(tp_pct, 4),
            "stop_loss_pct": sl_pct,
            "position_size": round(position_size, 2),
            "confidence": pred["confidence"],
            "predicted_return": pred.get("pred_return", 0),
        }
        state["open_positions"].append(position)

        notes = f" [{pos['correlation_note']}]" if pos.get("correlation_note") else ""
        print(f"  OPENED: {name} {direction} @ ${pred['price']:.2f} "
              f"(conf={pred['confidence']:.0%}, size={final_size_pct:.0%}→${position_size:.0f}){notes}")


def get_equity_price(ticker: str) -> float | None:
    """Get current equity price."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def open_equity_positions(state: dict, commodities: dict):
    """Open equity positions when commodity signals fire (using OOS thresholds)."""
    from position_manager import CALIBRATED_THRESHOLDS

    open_tickers = {p.get("ticker", p.get("commodity")) for p in state["open_positions"]}

    for commodity_name, equity_list in EQUITY_TRADES.items():
        pred = commodities.get(commodity_name)
        if pred is None:
            continue

        # Use OOS-calibrated threshold
        threshold = CALIBRATED_THRESHOLDS.get(commodity_name, 0.80)

        if pred["confidence"] < threshold:
            continue

        commodity_direction = "UP" if pred["direction"] == "UP" else "DOWN"

        for eq in equity_list:
            ticker = eq["ticker"]
            if ticker in open_tickers:
                continue

            # Determine equity direction
            if eq["direction"] == "same":
                eq_direction = "LONG" if commodity_direction == "UP" else "SHORT"
            else:
                eq_direction = "SHORT" if commodity_direction == "UP" else "LONG"

            # Get current equity price
            eq_price = get_equity_price(ticker)
            if eq_price is None:
                print(f"  Could not fetch {ticker} price")
                continue

            position_size = state["equity"] * EQUITY_POSITION_PCT
            tp_pct = abs(pred["pred_return"]) * abs(eq["beta"])
            sl_pct = 0.15

            position = {
                "commodity": f"{commodity_name}→{ticker}",
                "ticker": ticker,
                "type": "equity",
                "direction": eq_direction,
                "entry_date": pred["date"],
                "entry_price": eq_price,
                "take_profit_pct": round(max(tp_pct, 0.03), 4),
                "stop_loss_pct": sl_pct,
                "position_size": round(position_size, 2),
                "confidence": pred["confidence"],
                "commodity_signal": commodity_direction,
                "beta": eq["beta"],
            }
            state["open_positions"].append(position)
            print(f"  OPENED EQUITY: {eq_direction} {ticker} @ ${eq_price:.2f} "
                  f"(triggered by {commodity_name} {commodity_direction}, "
                  f"conf={pred['confidence']:.0%}, size=${position_size:.2f})")


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*60}")
    print(f"PAPER TRADING UPDATE — {now}")
    print(f"{'='*60}")

    state = load_trades()
    print(f"\nEquity: ${state['equity']:.2f}")
    print(f"Open positions: {len(state['open_positions'])}")
    print(f"Closed positions: {len(state['closed_positions'])}")

    # Get current predictions
    print("\nFetching predictions...")
    commodities = {}
    for project_dir, name, price_col in [
        (COFFEE_DIR, "Coffee", "coffee_close"),
        (COCOA_DIR, "Cocoa", "cocoa_close"),
        (SUGAR_DIR, "Sugar", "sugar_close"),
        (NATGAS_DIR, "Natural Gas", "natgas_close"),
        (SOYBEANS_DIR, "Soybeans", "soybeans_close"),
        (WHEAT_DIR, "Wheat", "wheat_close"),
        (COPPER_DIR, "Copper", "copper_close"),
    ]:
        pred = get_latest_prediction(project_dir, price_col)
        if pred:
            commodities[name] = pred
            action = ("LONG" if pred["direction"] == "UP" else "SHORT") if pred["confidence"] >= pred["threshold"] else "NO TRADE"
            print(f"  {name}: ${pred['price']:.2f}, {pred['direction']} ({pred['confidence']:.0%}), -> {action}")

    # Log predictions
    for name, pred in commodities.items():
        state["prediction_history"].append({
            "timestamp": now,
            "commodity": name,
            **pred,
        })

    # Check open positions for exits
    print("\nChecking open positions...")
    closed = check_open_positions(state, commodities)
    if not closed and state["open_positions"]:
        for pos in state["open_positions"]:
            print(f"  HOLDING: {pos['commodity']} {pos['direction']} "
                  f"@ ${pos['entry_price']:.2f} -> ${pos.get('current_price', 0):.2f} "
                  f"({pos.get('current_return', 0):+.2%}, {pos.get('days_held', 0)}d)")

    # Open new positions (commodities)
    print("\nChecking for new commodity entries...")
    open_new_positions(state, commodities)

    # Open equity positions triggered by commodity signals
    print("\nChecking for equity entries...")
    open_equity_positions(state, commodities)

    # Summary
    print(f"\n{'='*60}")
    print(f"PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    print(f"  Equity:          ${state['equity']:.2f} ({(state['equity']/STARTING_EQUITY - 1):+.2%} from start)")
    print(f"  Open positions:  {len(state['open_positions'])}")
    print(f"  Total trades:    {len(state['closed_positions'])}")

    if state["closed_positions"]:
        wins = [p for p in state["closed_positions"] if p["pnl_pct"] > 0]
        losses = [p for p in state["closed_positions"] if p["pnl_pct"] <= 0]
        win_rate = len(wins) / len(state["closed_positions"])
        total_pnl = sum(p["pnl_dollar"] for p in state["closed_positions"])
        print(f"  Win rate:        {win_rate:.0%} ({len(wins)}W / {len(losses)}L)")
        print(f"  Total P&L:       ${total_pnl:+.2f}")

    save_trades(state)
    print(f"\nState saved to {TRADES_FILE}")

    # Log to predictions file
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(f"\n{now}\n")
        for name, pred in commodities.items():
            action = ("LONG" if pred["direction"] == "UP" else "SHORT") if pred["confidence"] >= pred["threshold"] else "HOLD"
            f.write(f"  {name}: ${pred['price']:.2f} -> ${pred['pred_price']:.2f} "
                    f"({pred['pred_return']:+.2%}, {pred['confidence']:.0%}) -> {action}\n")


if __name__ == "__main__":
    main()
