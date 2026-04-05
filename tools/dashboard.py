"""Combined commodities dashboard — coffee + cocoa side by side."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string

# Add both project dirs to path
COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"
SOYBEANS_DIR = Path(__file__).parent.parent / "soybeans"
WHEAT_DIR = Path(__file__).parent.parent / "wheat"
COPPER_DIR = Path(__file__).parent.parent / "copper"

app = Flask(__name__)


def _run_prediction_subprocess(project_dir: Path, price_col: str) -> dict:
    """Run prediction in a subprocess to isolate module imports."""
    import subprocess
    script = f"""
import json, sys, joblib, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

# Build features without target so we can use the very latest row
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
    print(json.dumps({{"error": "no metadata"}}))
    sys.exit(0)

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
as_of = latest.index[0].strftime('%Y-%m-%d')

result = {{
    'version': version, 'as_of': as_of,
    'current_price': current_price,
    'pred_return': pred_return, 'pred_dir': pred_dir,
    'confidence': confidence,
    'clf_accuracy': meta['classification']['avg_accuracy'],
    'reg_accuracy': meta['regression']['avg_accuracy'],
    'fold_accuracies': meta['classification']['fold_accuracies'],
    'n_features': len(available),
    'strategy_cfg': meta.get('strategy', {{'confidence_threshold': 0.70, 'stop_loss_pct': 0.10}}),
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


def load_commodity(project_dir: Path, commodity_name: str, ticker: str, price_col: str, color: str):
    """Load prediction data for a single commodity."""
    data_dir = project_dir / "data"

    raw = _run_prediction_subprocess(project_dir, price_col)
    if raw is None or "error" in raw:
        return None

    version = raw["version"]
    current_price = raw["current_price"]
    pred_return = raw["pred_return"]
    pred_dir = raw["pred_dir"]
    confidence = raw["confidence"]
    predicted_price = current_price * (1 + pred_return)
    as_of_date = raw["as_of"]
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    target_date = (as_of_dt + timedelta(days=90)).strftime("%Y-%m-%d")

    # Data staleness
    data_age_days = (datetime.now() - as_of_dt).days
    is_stale = data_age_days > 7

    # Price history
    prices_df = pd.read_csv(data_dir / "combined_features.csv", index_col=0, parse_dates=True)
    cutoff = prices_df.index.max() - pd.DateOffset(years=2)
    if price_col in prices_df.columns:
        prices_2y = prices_df.loc[prices_df.index >= cutoff, price_col]
    else:
        prices_2y = prices_df.iloc[:, 0].loc[prices_df.index >= cutoff]
    price_dates = [str(d.strftime("%Y-%m-%d")) for d in prices_2y.index]
    price_values = [round(float(v), 2) for v in prices_2y.values]

    # Strategy
    strategy_cfg = raw["strategy_cfg"]
    threshold = strategy_cfg.get("confidence_threshold", 0.70)
    sl_pct = strategy_cfg.get("stop_loss_pct", 0.10)

    if confidence >= threshold:
        direction = "LONG" if pred_dir == 1 else "SHORT"
        tp_pct = abs(pred_return)
        if direction == "LONG":
            tp_price = current_price * (1 + tp_pct)
            sl_price = current_price * (1 - sl_pct)
        else:
            tp_price = current_price * (1 - tp_pct)
            sl_price = current_price * (1 + sl_pct)
        strategy = {
            "action": direction, "entry": current_price,
            "take_profit": tp_price, "take_profit_pct": tp_pct,
            "stop_loss": sl_price, "stop_loss_pct": sl_pct,
            "risk_reward": tp_pct / sl_pct if sl_pct > 0 else 0,
        }
    else:
        strategy = {"action": "NO TRADE", "reason": f"Confidence {confidence:.1%} < {threshold:.0%}"}

    # Equity plays for this commodity
    equity_map = {
        "Natural Gas": [{"ticker": "EQT", "beta": 0.23, "wr": "86%", "pf": "14.0"},
                        {"ticker": "RRC", "beta": 0.22, "wr": "82%", "pf": "2.7"}],
        "Wheat": [{"ticker": "BG", "beta": 0.10, "wr": "71%", "pf": "3.8"}],
    }
    equity_plays = equity_map.get(commodity_name, [])

    # Backtest metrics
    models_dir = project_dir / "models"
    strat_file = models_dir / "strategy_results.json"
    backtest = None
    if strat_file.exists():
        with open(strat_file) as f:
            backtest = json.load(f).get("metrics")

    return {
        "name": commodity_name, "ticker": ticker, "color": color, "version": version,
        "as_of_date": as_of_date, "is_stale": is_stale, "data_age_days": data_age_days,
        "current_price": current_price, "predicted_price": predicted_price,
        "predicted_return": pred_return,
        "direction": "UP" if pred_dir == 1 else "DOWN",
        "confidence": confidence,
        "clf_accuracy": raw["clf_accuracy"],
        "reg_accuracy": raw["reg_accuracy"],
        "fold_accuracies": raw["fold_accuracies"],
        "n_features": raw["n_features"],
        "strategy": strategy,
        "backtest": backtest,
        "equity_plays": equity_plays,
        "price_dates": price_dates, "price_values": price_values,
    }


def compute_correlation(coffee_dir: Path, cocoa_dir: Path):
    """Compute rolling correlation between coffee and cocoa returns."""
    try:
        coffee_df = pd.read_csv(coffee_dir / "data" / "combined_features.csv", index_col=0, parse_dates=True)
        cocoa_df = pd.read_csv(cocoa_dir / "data" / "combined_features.csv", index_col=0, parse_dates=True)

        coffee_ret = coffee_df["coffee_close"].pct_change()
        cocoa_ret = cocoa_df["cocoa_close"].pct_change()

        combined = pd.DataFrame({"coffee": coffee_ret, "cocoa": cocoa_ret}).dropna()
        corr_63d = combined["coffee"].rolling(63).corr(combined["cocoa"])

        current_corr = float(corr_63d.iloc[-1]) if len(corr_63d) > 0 else 0
        avg_corr = float(corr_63d.mean()) if len(corr_63d) > 0 else 0

        # Dates and values for chart (last 2 years)
        cutoff = combined.index.max() - pd.DateOffset(years=2)
        recent = corr_63d[corr_63d.index >= cutoff].dropna()
        corr_dates = [d.strftime("%Y-%m-%d") for d in recent.index]
        corr_values = [round(float(v), 3) for v in recent.values]

        return {
            "current": current_corr, "average": avg_corr,
            "dates": corr_dates, "corr_vals": corr_values,
            "diverging": abs(current_corr - avg_corr) > 0.2,
        }
    except Exception:
        return None


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Commodities Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e4e4e7; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        header { display: flex; align-items: center; gap: 12px; padding: 20px 0; border-bottom: 1px solid #27272a; margin-bottom: 24px; }
        header h1 { font-size: 24px; font-weight: 600; }
        .tag { background: #18181b; border: 1px solid #3f3f46; padding: 4px 10px; border-radius: 6px; font-size: 12px; color: #a1a1aa; }
        .stale-warning { background: #44403c; border: 1px solid #78716c; color: #fbbf24; padding: 8px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 16px; }
        .commodity-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        @media (max-width: 900px) { .commodity-grid { grid-template-columns: 1fr; } }
        .commodity-panel { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; }
        .panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #27272a; }
        .panel-title { font-size: 18px; font-weight: 600; }
        .price-row { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px; }
        .current-price { font-size: 32px; font-weight: 700; }
        .pred-price { font-size: 20px; font-weight: 600; }
        .up { color: #22c55e; } .down { color: #ef4444; } .neutral { color: #a1a1aa; }
        .signal-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; }
        .signal-up { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
        .signal-down { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
        .signal-none { background: rgba(161,161,170,0.15); color: #a1a1aa; border: 1px solid rgba(161,161,170,0.3); }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
        .stat { text-align: center; padding: 10px; background: #0f1117; border-radius: 8px; }
        .stat-val { font-size: 18px; font-weight: 700; }
        .stat-label { font-size: 10px; color: #71717a; text-transform: uppercase; margin-top: 2px; }
        .strategy-box { margin-top: 12px; padding: 12px; background: #0f1117; border-radius: 8px; }
        .strategy-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
        .strategy-key { color: #71717a; }
        .chart-wrapper { height: 200px; margin-top: 12px; }
        .section-card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; margin-bottom: 24px; }
        .section-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
        .corr-info { display: flex; gap: 24px; margin-bottom: 12px; }
        .corr-stat { text-align: center; }
        .corr-val { font-size: 24px; font-weight: 700; }
        .corr-label { font-size: 11px; color: #71717a; text-transform: uppercase; }
        .disclaimer { margin-top: 24px; padding: 16px; background: #1c1917; border: 1px solid #44403c; border-radius: 8px; font-size: 12px; color: #a8a29e; line-height: 1.5; }
        .backtest-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px; }
        .bt-stat { text-align: center; padding: 8px; background: #0f1117; border-radius: 6px; }
        .bt-val { font-size: 15px; font-weight: 600; }
        .bt-label { font-size: 10px; color: #71717a; text-transform: uppercase; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Commodities Dashboard</h1>
        <span class="tag">63-Day Horizon</span>
        <span class="tag">Weekly Check</span>
    </header>

    {% for c in commodities %}
    {% if c and c.is_stale %}
    <div class="stale-warning">
        {{ c.name }} data is {{ c.data_age_days }} days old (last: {{ c.as_of_date }}). Run <code>python3 refresh.py</code> in the {{ c.name | lower }} directory.
    </div>
    {% endif %}
    {% endfor %}

    <div class="commodity-grid">
        {% for c in commodities %}
        {% if c %}
        <div class="commodity-panel">
            <div class="panel-header">
                <div>
                    <span class="panel-title">{{ c.name }}</span>
                    <span class="tag" style="margin-left:8px">{{ c.ticker }}</span>
                    <span class="tag">{{ c.version }}</span>
                </div>
                <span class="signal-badge {{ 'signal-up' if c.strategy.action == 'LONG' else ('signal-down' if c.strategy.action == 'SHORT' else 'signal-none') }}">
                    {{ c.strategy.action }}
                </span>
            </div>

            <div class="price-row">
                <div>
                    <div style="font-size:12px; color:#71717a">Current</div>
                    <div class="current-price">${{ "%.2f"|format(c.current_price) }}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:12px; color:#71717a">Predicted (~3mo)</div>
                    <div class="pred-price {{ 'up' if c.direction == 'UP' else 'down' }}">
                        ${{ "%.2f"|format(c.predicted_price) }}
                        <span style="font-size:13px">({{ "%+.1f"|format(c.predicted_return * 100) }}%)</span>
                    </div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-val">{{ "%.0f"|format(c.confidence * 100) }}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat">
                    <div class="stat-val {{ 'up' if c.clf_accuracy > 0.6 else 'neutral' }}">{{ "%.0f"|format(c.clf_accuracy * 100) }}%</div>
                    <div class="stat-label">Model Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-val">{{ c.n_features }}</div>
                    <div class="stat-label">Features</div>
                </div>
            </div>

            {% if c.strategy.action != "NO TRADE" %}
            <div class="strategy-box">
                <div style="font-size:12px; color:#71717a; text-transform:uppercase; margin-bottom:8px">Trade Setup</div>
                <div class="strategy-row"><span class="strategy-key">Entry</span><span>${{ "%.2f"|format(c.strategy.entry) }}</span></div>
                <div class="strategy-row"><span class="strategy-key">Take Profit</span><span class="up">${{ "%.2f"|format(c.strategy.take_profit) }} ({{ "%+.1f"|format(c.strategy.take_profit_pct * 100) }}%)</span></div>
                <div class="strategy-row"><span class="strategy-key">Stop Loss</span><span class="down">${{ "%.2f"|format(c.strategy.stop_loss) }} (-{{ "%.0f"|format(c.strategy.stop_loss_pct * 100) }}%)</span></div>
                <div class="strategy-row"><span class="strategy-key">Risk/Reward</span><span>{{ "%.1f"|format(c.strategy.risk_reward) }}x</span></div>
            </div>
            {% else %}
            <div class="strategy-box">
                <div style="color:#71717a; font-size:13px">{{ c.strategy.reason }}</div>
            </div>
            {% endif %}

            {% if c.equity_plays %}
            <div style="margin-top:8px; padding:8px; background:#0f1117; border-radius:6px;">
                <div style="font-size:11px; color:#71717a; text-transform:uppercase; margin-bottom:4px">Equity Plays</div>
                {% for eq in c.equity_plays %}
                <div style="font-size:13px; display:flex; justify-content:space-between; padding:2px 0">
                    <span>{{ eq.ticker }} (β={{ eq.beta }})</span>
                    <span style="color:#71717a">WR: {{ eq.wr }}, PF: {{ eq.pf }}</span>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            {% if c.backtest %}
            <div class="backtest-row">
                <div class="bt-stat"><div class="bt-val">{{ "%.0f"|format(c.backtest.win_rate * 100) }}%</div><div class="bt-label">Win Rate</div></div>
                <div class="bt-stat"><div class="bt-val">{{ "%.2f"|format(c.backtest.profit_factor) }}</div><div class="bt-label">Profit Factor</div></div>
                <div class="bt-stat"><div class="bt-val">{{ "%.2f"|format(c.backtest.sharpe) }}</div><div class="bt-label">Sharpe</div></div>
                <div class="bt-stat"><div class="bt-val">{{ "%.0f"|format(c.backtest.max_drawdown * 100) }}%</div><div class="bt-label">Max DD</div></div>
            </div>
            {% endif %}

            <div class="chart-wrapper"><canvas id="chart_{{ loop.index0 }}"></canvas></div>
        </div>
        {% endif %}
        {% endfor %}
    </div>

    {% if correlation %}
    <div class="section-card">
        <div class="section-title">Coffee-Cocoa Correlation</div>
        <div class="corr-info">
            <div class="corr-stat">
                <div class="corr-val {{ 'up' if correlation.current > 0.3 else ('down' if correlation.current < -0.1 else 'neutral') }}">{{ "%.2f"|format(correlation.current) }}</div>
                <div class="corr-label">Current (63d)</div>
            </div>
            <div class="corr-stat">
                <div class="corr-val neutral">{{ "%.2f"|format(correlation.average) }}</div>
                <div class="corr-label">Historical Avg</div>
            </div>
            {% if correlation.diverging %}
            <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); border-radius:8px; padding:8px 16px; display:flex; align-items:center; font-size:13px; color:#fbbf24;">
                Correlation diverging from average — potential mean-reversion opportunity
            </div>
            {% endif %}
        </div>
        <div style="height:180px"><canvas id="corrChart"></canvas></div>
    </div>
    {% endif %}

    <div class="disclaimer">
        <strong>Disclaimer:</strong> For educational/research purposes only. Not financial advice.
        Both models use purged walk-forward cross-validation. Check weekly after COT data release (Saturdays).
        Refresh data with <code>python3 refresh.py</code> in each commodity directory.
    </div>
</div>

<script>
{% for c in commodities %}
{% if c %}
(function() {
    const ctx = document.getElementById('chart_{{ loop.index0 }}').getContext('2d');
    const dates = {{ c.price_dates | tojson }};
    const prices = {{ c.price_values | tojson }};
    const predDate = '{{ c.as_of_date }}';
    const allDates = [...dates, predDate];
    const actualPrices = [...prices, null];
    const predLine = new Array(prices.length - 1).fill(null);
    predLine.push(prices[prices.length - 1]);
    predLine.push({{ "%.2f"|format(c.predicted_price) }});
    new Chart(ctx, { type: 'line', data: { labels: allDates, datasets: [
        { label: '{{ c.name }}', data: actualPrices, borderColor: '{{ c.color }}', backgroundColor: '{{ c.color }}11', fill: true, tension: 0.1, pointRadius: 0, borderWidth: 1.5 },
        { label: 'Prediction', data: predLine, borderColor: '{{ "#22c55e" if c.direction == "UP" else "#ef4444" }}', borderDash: [4,3], pointRadius: [0,5], pointBackgroundColor: '{{ "#22c55e" if c.direction == "UP" else "#ef4444" }}', borderWidth: 1.5, fill: false },
    ]}, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#52525b', maxTicksLimit: 6, font: { size: 10 } }, grid: { display: false } }, y: { ticks: { color: '#52525b', font: { size: 10 }, callback: v => '$'+v }, grid: { color: 'rgba(39,39,42,0.3)' } } } } });
})();
{% endif %}
{% endfor %}

{% if correlation %}
(function() {
    const ctx = document.getElementById('corrChart').getContext('2d');
    new Chart(ctx, { type: 'line', data: { labels: {{ correlation.dates | tojson }}, datasets: [
        { label: '63d Correlation', data: {{ correlation.corr_vals | tojson }}, borderColor: '#818cf8', backgroundColor: 'rgba(129,140,248,0.08)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5 },
        { label: 'Average', data: new Array({{ correlation.corr_vals | length }}).fill({{ "%.3f"|format(correlation.average) }}), borderColor: '#52525b', borderDash: [4,4], pointRadius: 0, borderWidth: 1, fill: false },
    ]}, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#71717a', font: { size: 10 } } } }, scales: { x: { ticks: { color: '#52525b', maxTicksLimit: 8, font: { size: 10 } }, grid: { display: false } }, y: { min: -0.5, max: 1, ticks: { color: '#52525b', font: { size: 10 } }, grid: { color: 'rgba(39,39,42,0.3)' } } } } });
})();
{% endif %}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    coffee = load_commodity(COFFEE_DIR, "Coffee", "KC=F", "coffee_close", "#a78bfa")
    cocoa = load_commodity(COCOA_DIR, "Cocoa", "CC=F", "cocoa_close", "#f59e0b")
    sugar = load_commodity(SUGAR_DIR, "Sugar", "SB=F", "sugar_close", "#10b981")
    natgas = load_commodity(NATGAS_DIR, "Natural Gas", "NG=F", "natgas_close", "#ef4444")
    soybeans = load_commodity(SOYBEANS_DIR, "Soybeans", "ZS=F", "soybeans_close", "#84cc16")
    wheat = load_commodity(WHEAT_DIR, "Wheat", "ZW=F", "wheat_close", "#eab308")
    copper = load_commodity(COPPER_DIR, "Copper", "HG=F", "copper_close", "#f97316")
    commodities = [c for c in [coffee, cocoa, sugar, natgas, soybeans, wheat, copper] if c is not None]
    correlation = compute_correlation(COFFEE_DIR, COCOA_DIR)
    return render_template_string(TEMPLATE, commodities=commodities, correlation=correlation)


@app.route("/api/all")
def api_all():
    coffee = load_commodity(COFFEE_DIR, "Coffee", "KC=F", "coffee_close", "#a78bfa")
    cocoa = load_commodity(COCOA_DIR, "Cocoa", "CC=F", "cocoa_close", "#f59e0b")
    sugar = load_commodity(SUGAR_DIR, "Sugar", "SB=F", "sugar_close", "#10b981")
    natgas = load_commodity(NATGAS_DIR, "Natural Gas", "NG=F", "natgas_close", "#ef4444")
    soybeans = load_commodity(SOYBEANS_DIR, "Soybeans", "ZS=F", "soybeans_close", "#84cc16")
    wheat = load_commodity(WHEAT_DIR, "Wheat", "ZW=F", "wheat_close", "#eab308")
    copper = load_commodity(COPPER_DIR, "Copper", "HG=F", "copper_close", "#f97316")
    for c in [coffee, cocoa, sugar, natgas, soybeans, wheat, copper]:
        if c:
            del c["price_dates"]
            del c["price_values"]
    return {"coffee": coffee, "cocoa": cocoa, "sugar": sugar, "natgas": natgas,
            "soybeans": soybeans, "wheat": wheat, "copper": copper}


if __name__ == "__main__":
    print("Starting Combined Commodities Dashboard...")
    print("Open http://localhost:8060 in your browser")
    app.run(host="127.0.0.1", debug=False, port=8060)
