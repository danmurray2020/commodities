"""Web dashboard for sugar price predictions."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string

from features import prepare_dataset

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"

app = Flask(__name__)


def _log_prediction(data: dict):
    """Append prediction to JSON-lines log for audit."""
    log_path = Path(__file__).parent / "data" / "prediction_log.jsonl"
    try:
        from datetime import datetime as _dt
        entry = {"logged_at": _dt.now().isoformat(), **data}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass  # Don't let logging break the app


def get_prediction():
    with open(MODELS_DIR / "production_metadata.json") as f:
        meta = json.load(f)
    reg = joblib.load(MODELS_DIR / "production_regressor.joblib")
    clf = joblib.load(MODELS_DIR / "production_classifier.joblib")
    feature_cols = meta["features"]

    df, all_cols = prepare_dataset(horizon=63)
    available = [f for f in feature_cols if f in all_cols]

    latest = df.iloc[[-1]]
    X = latest[available].values

    pred_return = float(reg.predict(X)[0])
    pred_dir = int(clf.predict(X)[0])
    pred_proba = clf.predict_proba(X)[0]
    confidence = float(pred_proba[pred_dir])
    current_price = float(latest["sugar_close"].values[0])
    as_of_date = latest.index[0].strftime("%Y-%m-%d")

    _log_prediction({
        "date": as_of_date,
        "price": current_price,
        "pred_return": pred_return,
        "direction": "UP" if pred_dir == 1 else "DOWN",
        "confidence": confidence,
    })
    predicted_price = current_price * (1 + pred_return)
    target_date = (latest.index[0] + timedelta(days=90)).strftime("%Y-%m-%d")

    feature_values = {feat: float(latest[feat].values[0]) for feat in available}

    prices_df = pd.read_csv(DATA_DIR / "combined_features.csv", index_col=0, parse_dates=True)
    cutoff = prices_df.index.max() - pd.DateOffset(years=2)
    prices_2y = prices_df.loc[prices_df.index >= cutoff, "sugar_close"]
    price_dates = [d.strftime("%Y-%m-%d") for d in prices_2y.index]
    price_values = [round(float(v), 2) for v in prices_2y.values]

    strategy_cfg = meta.get("strategy", {})
    threshold = strategy_cfg.get("confidence_threshold", 0.70)
    sl_pct = strategy_cfg.get("stop_loss_pct", 0.10)

    if confidence >= threshold:
        direction = "LONG" if pred_dir == 1 else "SHORT"
        tp_pct = abs(pred_return)
        tp_price = current_price * (1 + tp_pct) if direction == "LONG" else current_price * (1 - tp_pct)
        sl_price = current_price * (1 - sl_pct) if direction == "LONG" else current_price * (1 + sl_pct)
        strategy = {
            "action": direction, "entry": current_price,
            "take_profit": tp_price, "take_profit_pct": tp_pct,
            "stop_loss": sl_price, "stop_loss_pct": sl_pct,
            "risk_reward": tp_pct / sl_pct if sl_pct > 0 else 0,
            "max_hold": 63, "threshold": threshold,
        }
    else:
        strategy = {
            "action": "NO TRADE",
            "reason": f"Confidence {confidence:.1%} below threshold {threshold:.0%}",
            "threshold": threshold, "max_hold": 63,
        }

    return {
        "commodity": "Sugar",
        "ticker": "SB=F",
        "as_of_date": as_of_date, "target_date": target_date,
        "current_price": current_price, "predicted_price": predicted_price,
        "predicted_return": pred_return,
        "direction": "UP" if pred_dir == 1 else "DOWN",
        "confidence": confidence,
        "reg_accuracy": meta["regression"]["avg_accuracy"],
        "clf_accuracy": meta["classification"]["avg_accuracy"],
        "fold_accuracies": meta["classification"]["fold_accuracies"],
        "feature_values": feature_values,
        "price_dates": price_dates, "price_values": price_values,
        "n_features": len(available),
        "strategy": strategy,
    }


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sugar Futures Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e4e4e7; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 20px 0; border-bottom: 1px solid #27272a; margin-bottom: 24px; }
        header h1 { font-size: 24px; font-weight: 600; }
        .tag { background: #18181b; border: 1px solid #3f3f46; padding: 4px 10px; border-radius: 6px; font-size: 12px; color: #a1a1aa; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; }
        .card-label { font-size: 13px; color: #71717a; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        .card-value { font-size: 28px; font-weight: 700; }
        .card-sub { font-size: 13px; color: #71717a; margin-top: 4px; }
        .up { color: #22c55e; } .down { color: #ef4444; } .neutral { color: #a1a1aa; }
        .full-width { grid-column: 1 / -1; }
        .chart-wrapper { height: 320px; margin-top: 12px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
        @media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { text-align: left; padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #27272a; }
        th { color: #71717a; font-weight: 500; }
        td.mono { font-family: 'SF Mono', 'Menlo', monospace; text-align: right; }
        .signal-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; margin-top: 8px; }
        .signal-up { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
        .signal-down { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
        .signal-none { background: rgba(161,161,170,0.15); color: #a1a1aa; border: 1px solid rgba(161,161,170,0.3); }
        .bar-container { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
        .bar-label { font-size: 13px; width: 50px; text-align: right; }
        .bar-bg { flex: 1; height: 24px; background: #27272a; border-radius: 4px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 12px; font-weight: 500; }
        .strategy-label { color: #71717a; font-size: 13px; }
        .strategy-value { font-weight: 600; font-size: 14px; }
        .disclaimer { margin-top: 32px; padding: 16px; background: #1c1917; border: 1px solid #44403c; border-radius: 8px; font-size: 12px; color: #a8a29e; line-height: 1.5; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Sugar Futures Dashboard</h1>
        <span class="tag">{{ data.ticker }}</span>
        <span class="tag">63-Day Horizon</span>
        <span class="tag">{{ data.n_features }} Features</span>
    </header>
    <div class="grid">
        <div class="card">
            <div class="card-label">Current Price</div>
            <div class="card-value">${{ "%.2f"|format(data.current_price) }}</div>
            <div class="card-sub">As of {{ data.as_of_date }}</div>
        </div>
        <div class="card">
            <div class="card-label">Predicted Price (~3 Months)</div>
            <div class="card-value {{ 'up' if data.direction == 'UP' else 'down' }}">${{ "%.2f"|format(data.predicted_price) }}</div>
            <div class="card-sub">Target: {{ data.target_date }}</div>
        </div>
        <div class="card">
            <div class="card-label">Signal</div>
            <div class="signal-badge {{ 'signal-up' if data.direction == 'UP' else 'signal-down' }}">{{ data.direction }} {{ "%+.1f"|format(data.predicted_return * 100) }}%</div>
            <div class="card-sub" style="margin-top:12px">Confidence: {{ "%.1f"|format(data.confidence * 100) }}%</div>
        </div>
    </div>
    <div class="grid">
        <div class="card full-width">
            <div class="card-label">Strategy Recommendation</div>
            {% if data.strategy.action == "NO TRADE" %}
            <div class="signal-badge signal-none" style="margin-top:12px">NO TRADE</div>
            <div class="card-sub" style="margin-top:8px">{{ data.strategy.reason }}</div>
            {% else %}
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-top: 16px;">
                <div><div class="strategy-label">Action</div><div class="signal-badge {{ 'signal-up' if data.strategy.action == 'LONG' else 'signal-down' }}">{{ data.strategy.action }}</div></div>
                <div><div class="strategy-label">Entry Price</div><div class="strategy-value">${{ "%.2f"|format(data.strategy.entry) }}</div></div>
                <div><div class="strategy-label">Take Profit</div><div class="strategy-value up">${{ "%.2f"|format(data.strategy.take_profit) }} <span style="font-size:12px">({{ "%+.1f"|format(data.strategy.take_profit_pct * 100) }}%)</span></div></div>
                <div><div class="strategy-label">Stop Loss</div><div class="strategy-value down">${{ "%.2f"|format(data.strategy.stop_loss) }} <span style="font-size:12px">(-{{ "%.0f"|format(data.strategy.stop_loss_pct * 100) }}%)</span></div></div>
                <div><div class="strategy-label">Risk/Reward</div><div class="strategy-value">{{ "%.1f"|format(data.strategy.risk_reward) }}x</div></div>
                <div><div class="strategy-label">Max Hold</div><div class="strategy-value">63 days</div></div>
            </div>
            {% endif %}
        </div>
    </div>
    <div class="grid"><div class="card full-width"><div class="card-label">Price History (2 Years) + Prediction</div><div class="chart-wrapper"><canvas id="priceChart"></canvas></div></div></div>
    <div class="two-col">
        <div class="card">
            <div class="card-label">Model Accuracy (Purged Walk-Forward CV)</div>
            <div style="margin-top:12px">
                <div style="font-size:14px; margin-bottom:12px">Classification: <strong>{{ "%.1f"|format(data.clf_accuracy * 100) }}%</strong> | Regression Dir: <strong>{{ "%.1f"|format(data.reg_accuracy * 100) }}%</strong></div>
                {% for acc in data.fold_accuracies %}
                <div class="bar-container"><div class="bar-label">Fold {{ loop.index0 }}</div><div class="bar-bg"><div class="bar-fill {{ 'up' if acc > 0.55 else ('down' if acc < 0.5 else 'neutral') }}" style="width:{{ [acc * 100, 100]|min }}%; background:{{ 'rgba(34,197,94,0.3)' if acc > 0.55 else ('rgba(239,68,68,0.3)' if acc < 0.5 else 'rgba(161,161,170,0.2)') }}">{{ "%.0f"|format(acc * 100) }}%</div></div></div>
                {% endfor %}
            </div>
        </div>
        <div class="card">
            <div class="card-label">Feature Values (Current)</div>
            <div style="overflow-y:auto; max-height:320px">
            <table><tr><th>Feature</th><th style="text-align:right">Value</th></tr>
            {% for feat, val in data.feature_values.items() %}<tr><td>{{ feat }}</td><td class="mono">{{ "%.4f"|format(val) if val|abs < 100 else "%.0f"|format(val) }}</td></tr>{% endfor %}
            </table></div>
        </div>
    </div>
    <div class="disclaimer"><strong>Disclaimer:</strong> This model is for educational and research purposes only. It is not financial advice. Sugar futures are volatile and past performance does not guarantee future results.</div>
</div>
<script>
    const ctx = document.getElementById('priceChart').getContext('2d');
    const dates = {{ data.price_dates | tojson }};
    const prices = {{ data.price_values | tojson }};
    const predPrice = {{ "%.2f"|format(data.predicted_price) }};
    const predDate = '{{ data.target_date }}';
    const allDates = [...dates, predDate];
    const actualPrices = [...prices, null];
    const predLine = new Array(prices.length - 1).fill(null);
    predLine.push(prices[prices.length - 1]); predLine.push(predPrice);
    {% if data.strategy.action != "NO TRADE" and data.strategy.take_profit %}
    const tpLine = new Array(prices.length-1).fill(null); tpLine.push(prices[prices.length-1]); tpLine.push({{ "%.2f"|format(data.strategy.take_profit) }});
    const slLine = new Array(prices.length-1).fill(null); slLine.push(prices[prices.length-1]); slLine.push({{ "%.2f"|format(data.strategy.stop_loss) }});
    {% endif %}
    new Chart(ctx, { type: 'line', data: { labels: allDates, datasets: [
        { label: 'Sugar Price (USD/mt)', data: actualPrices, borderColor: '#10b981', backgroundColor: 'rgba(245,158,11,0.08)', fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2 },
        { label: 'Prediction', data: predLine, borderColor: '{{ "#22c55e" if data.direction == "UP" else "#ef4444" }}', borderDash: [6,4], pointRadius: [0,6], pointBackgroundColor: '{{ "#22c55e" if data.direction == "UP" else "#ef4444" }}', borderWidth: 2, fill: false },
        {% if data.strategy.action != "NO TRADE" and data.strategy.take_profit %}
        { label: 'Take Profit', data: tpLine, borderColor: 'rgba(34,197,94,0.5)', borderDash: [3,3], pointRadius: 0, borderWidth: 1, fill: false },
        { label: 'Stop Loss', data: slLine, borderColor: 'rgba(239,68,68,0.5)', borderDash: [3,3], pointRadius: 0, borderWidth: 1, fill: false },
        {% endif %}
    ]}, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#a1a1aa', font: { size: 11 } } } }, scales: { x: { ticks: { color: '#71717a', maxTicksLimit: 12, font: { size: 11 } }, grid: { color: 'rgba(39,39,42,0.5)' } }, y: { ticks: { color: '#71717a', font: { size: 11 }, callback: v => '$' + v }, grid: { color: 'rgba(39,39,42,0.5)' } } } } });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    data = get_prediction()
    return render_template_string(TEMPLATE, data=data)


@app.route("/api/prediction")
def api_prediction():
    data = get_prediction()
    return {k: v for k, v in data.items() if k not in ("price_dates", "price_values")}


if __name__ == "__main__":
    print("Starting Sugar Futures Dashboard...")
    print("Open http://localhost:8052 in your browser")
    app.run(host="127.0.0.1", debug=False, port=8052)
