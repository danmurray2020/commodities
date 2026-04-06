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

    # Write the prediction script to a temp file to avoid f-string escaping issues
    script_path = project_dir / "_dashboard_predict.py"
    script_content = '''
import json, sys, os, joblib, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from features import prepare_dataset

PRICE_COL = sys.argv[1]

# Use prepare_dataset with a dummy horizon, then drop the target cols
df, _ = prepare_dataset(horizon=5)
# Drop target columns since we just want features
for col in ['target_return', 'target_direction']:
    if col in df.columns:
        df = df.drop(columns=[col])

models_dir = 'models'
is_ensemble = False
meta = None
for meta_file in ['ensemble_metadata.json', 'v2_production_metadata.json', 'production_metadata.json']:
    path = os.path.join(models_dir, meta_file)
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
        is_ensemble = 'ensemble' in meta_file
        break

if meta is None:
    print(json.dumps({"error": "no metadata"}))
    sys.exit(0)

current_price = float(df.iloc[-1][PRICE_COL])
as_of = df.index[-1].strftime('%Y-%m-%d')

if is_ensemble and 'models' in meta:
    models_list = meta['models']
    best = max(models_list, key=lambda x: x.get('avg_dir_acc', 0))
    best_h = best['horizon']
    best_type = best['model_type']
    features = best['features']
    available = [f for f in features if f in df.columns]

    reg_file = os.path.join(models_dir, f'ensemble_reg_{best_h}d_{best_type}.joblib')
    clf_file = os.path.join(models_dir, f'ensemble_clf_{best_h}d_{best_type}.joblib')
    if not os.path.exists(reg_file):
        reg_file = os.path.join(models_dir, f'ensemble_reg_{best_h}d.joblib')
        clf_file = os.path.join(models_dir, f'ensemble_clf_{best_h}d.joblib')

    reg = joblib.load(reg_file)
    clf = joblib.load(clf_file)
    X = df.iloc[[-1]][available].values
    pred_return = float(reg.predict(X)[0])
    pred_dir = int(clf.predict(X)[0])
    pred_proba = clf.predict_proba(X)[0]
    confidence = float(pred_proba[pred_dir])

    # Agreement: check all models
    n_agree = 0
    n_total = 0
    for m_meta in models_list:
        m_features = [f for f in m_meta['features'] if f in df.columns]
        if len(m_features) < len(m_meta['features']):
            continue
        try:
            mf = os.path.join(models_dir, f"ensemble_clf_{m_meta['horizon']}d_{m_meta['model_type']}.joblib")
            m_clf = joblib.load(mf)
            m_X = df.iloc[[-1]][m_features].values
            m_dir = int(m_clf.predict(m_X)[0])
            n_total += 1
            if m_dir == pred_dir:
                n_agree += 1
        except Exception:
            pass

    agreement = n_agree / n_total if n_total > 0 else 0.5
    best_acc = best.get('avg_dir_acc', 0)
    n_models = meta.get('n_models', len(models_list))

    result = {
        'version': f'ensemble ({best_h}d {best_type})', 'as_of': as_of,
        'current_price': current_price,
        'pred_return': pred_return, 'pred_dir': pred_dir,
        'confidence': confidence,
        'clf_accuracy': best_acc, 'reg_accuracy': best_acc,
        'fold_accuracies': [],
        'n_features': len(available),
        'strategy_cfg': meta.get('strategy', {'confidence_threshold': 0.70, 'stop_loss_pct': 0.10}),
        'ensemble': True, 'n_models': n_models,
        'best_model': f'{best_h}d {best_type}',
        'agreement': agreement, 'horizon': best_h,
    }
else:
    version = 'v2' if 'v2' in (meta_file if 'meta_file' in dir() else '') else 'v1'
    reg_name = 'v2_production_regressor.joblib' if version == 'v2' else 'production_regressor.joblib'
    clf_name = 'v2_production_classifier.joblib' if version == 'v2' else 'production_classifier.joblib'
    reg = joblib.load(os.path.join(models_dir, reg_name))
    clf = joblib.load(os.path.join(models_dir, clf_name))
    available = [f for f in meta.get('features', []) if f in df.columns]
    X = df.iloc[[-1]][available].values
    pred_return = float(reg.predict(X)[0])
    pred_dir = int(clf.predict(X)[0])
    pred_proba = clf.predict_proba(X)[0]
    confidence = float(pred_proba[pred_dir])

    result = {
        'version': version, 'as_of': as_of,
        'current_price': current_price,
        'pred_return': pred_return, 'pred_dir': pred_dir,
        'confidence': confidence,
        'clf_accuracy': meta.get('classification', {}).get('avg_accuracy', 0),
        'reg_accuracy': meta.get('regression', {}).get('avg_accuracy', 0),
        'fold_accuracies': meta.get('classification', {}).get('fold_accuracies', []),
        'n_features': len(available),
        'strategy_cfg': meta.get('strategy', {'confidence_threshold': 0.70, 'stop_loss_pct': 0.10}),
        'ensemble': False, 'n_models': 1,
        'horizon': meta.get('horizon', 63),
    }

print(json.dumps(result))
'''
    script_path.write_text(script_content)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), price_col],
            capture_output=True, text=True, cwd=str(project_dir), timeout=120,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return None
    finally:
        script_path.unlink(missing_ok=True)


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
    horizon_days = raw.get("horizon", 63)
    target_date = (as_of_dt + timedelta(days=int(horizon_days * 1.4))).strftime("%Y-%m-%d")

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

    horizon = raw.get("horizon", 63)
    is_ensemble = raw.get("ensemble", False)

    return {
        "name": commodity_name, "ticker": ticker, "color": color, "version": version,
        "as_of_date": as_of_date, "is_stale": is_stale, "data_age_days": data_age_days,
        "current_price": current_price, "predicted_price": predicted_price,
        "predicted_return": pred_return,
        "direction": "UP" if pred_dir == 1 else "DOWN",
        "confidence": confidence,
        "clf_accuracy": raw["clf_accuracy"],
        "reg_accuracy": raw["reg_accuracy"],
        "fold_accuracies": raw.get("fold_accuracies", []),
        "n_features": raw["n_features"],
        "strategy": strategy,
        "backtest": backtest,
        "equity_plays": equity_plays,
        "price_dates": price_dates, "price_values": price_values,
        "horizon": horizon,
        "is_ensemble": is_ensemble,
        "n_models": raw.get("n_models", 1),
        "best_model": raw.get("best_model", ""),
        "agreement": raw.get("agreement", 0),
    }


def load_portfolio_data(commodities_list):
    """Build portfolio position data from active signals and DB trades."""
    positions = []
    for c in commodities_list:
        if not c or c["strategy"]["action"] == "NO TRADE":
            continue
        positions.append({
            "name": c["name"],
            "color": c["color"],
            "direction": c["strategy"]["action"],
            "entry": c["strategy"]["entry"],
            "tp": c["strategy"]["take_profit"],
            "sl": c["strategy"]["stop_loss"],
            "tp_pct": c["strategy"]["take_profit_pct"],
            "sl_pct": c["strategy"]["stop_loss_pct"],
            "rr": c["strategy"]["risk_reward"],
            "confidence": c["confidence"],
            "pred_return": c["predicted_return"],
            "current_price": c["current_price"],
        })

    # Try loading trade stats and history from DB
    trade_stats = None
    recent_trades = []
    prediction_accuracy = None
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from db import get_db
        db = get_db()
        trade_stats = db.get_trade_stats()
        recent_trades = db.execute(
            "SELECT * FROM trades WHERE exit_date IS NOT NULL ORDER BY exit_date DESC LIMIT 20"
        )
        prediction_accuracy = db.get_prediction_accuracy()
    except Exception:
        pass

    return {
        "positions": positions,
        "trade_stats": trade_stats,
        "recent_trades": recent_trades,
        "prediction_accuracy": prediction_accuracy,
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
        <span class="tag">Multi-Horizon Ensemble</span>
        <span class="tag">XGBoost + LightGBM + Ridge</span>
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
                    <span class="tag">{{ c.horizon }}d</span>
                    {% if c.is_ensemble %}<span class="tag">{{ c.n_models }} models</span>{% endif %}
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
                    <div style="font-size:12px; color:#71717a">Predicted ({{ c.horizon }}d)</div>
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
                    <div class="stat-val {{ 'up' if c.clf_accuracy >= 0.70 else ('neutral' if c.clf_accuracy >= 0.60 else 'down') }}">{{ "%.0f"|format(c.clf_accuracy * 100) }}%</div>
                    <div class="stat-label">Best Model Acc</div>
                </div>
                {% if c.is_ensemble %}
                <div class="stat">
                    <div class="stat-val {{ 'up' if c.agreement >= 0.7 else 'neutral' }}">{{ "%.0f"|format(c.agreement * 100) }}%</div>
                    <div class="stat-label">Agreement</div>
                </div>
                {% else %}
                <div class="stat">
                    <div class="stat-val">{{ c.n_features }}</div>
                    <div class="stat-label">Features</div>
                </div>
                {% endif %}
            </div>
            {% if c.is_ensemble and c.best_model %}
            <div style="font-size:11px; color:#71717a; text-align:center; margin-bottom:8px">Best: {{ c.best_model }}</div>
            {% endif %}

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

    {% if portfolio and portfolio.positions %}
    <div class="section-card">
        <div class="section-title">Active Positions & Portfolio Allocation</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
            <!-- Position bars (entry/TP/SL visual) -->
            <div>
                <div style="font-size:12px; color:#71717a; text-transform:uppercase; margin-bottom:12px">Position Map (Entry / TP / SL)</div>
                {% for pos in portfolio.positions %}
                <div style="margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:600;">{{ pos.direction }} {{ pos.name }}</span>
                        <span style="color:{{ '#22c55e' if pos.direction == 'LONG' else '#ef4444' }}; font-size:13px">{{ "%.0f"|format(pos.confidence * 100) }}% conf</span>
                    </div>
                    <div style="position:relative; height:32px; background:#0f1117; border-radius:6px; overflow:hidden;">
                        {% if pos.direction == "LONG" %}
                        <!-- SL (left) → Entry (center) → TP (right) -->
                        {% set range = pos.tp - pos.sl %}
                        {% set entry_pct = ((pos.entry - pos.sl) / range * 100) if range > 0 else 50 %}
                        {% set current_pct = ((pos.current_price - pos.sl) / range * 100) if range > 0 else 50 %}
                        <div style="position:absolute; left:0; top:0; height:100%; width:{{ entry_pct }}%; background:rgba(239,68,68,0.15);"></div>
                        <div style="position:absolute; left:{{ entry_pct }}%; top:0; height:100%; right:0; background:rgba(34,197,94,0.15);"></div>
                        <div style="position:absolute; left:{{ entry_pct }}%; top:0; height:100%; width:2px; background:#a1a1aa;" title="Entry ${{ '%.2f'|format(pos.entry) }}"></div>
                        <div style="position:absolute; left:{{ current_pct }}%; top:0; height:100%; width:3px; background:#fff; border-radius:2px;" title="Current ${{ '%.2f'|format(pos.current_price) }}"></div>
                        <div style="position:absolute; left:4px; top:50%; transform:translateY(-50%); font-size:10px; color:#ef4444;">SL ${{ "%.0f"|format(pos.sl) }}</div>
                        <div style="position:absolute; right:4px; top:50%; transform:translateY(-50%); font-size:10px; color:#22c55e;">TP ${{ "%.0f"|format(pos.tp) }}</div>
                        {% else %}
                        {% set range = pos.sl - pos.tp %}
                        {% set entry_pct = ((pos.entry - pos.tp) / range * 100) if range > 0 else 50 %}
                        {% set current_pct = ((pos.current_price - pos.tp) / range * 100) if range > 0 else 50 %}
                        <div style="position:absolute; left:0; top:0; height:100%; width:{{ entry_pct }}%; background:rgba(34,197,94,0.15);"></div>
                        <div style="position:absolute; left:{{ entry_pct }}%; top:0; height:100%; right:0; background:rgba(239,68,68,0.15);"></div>
                        <div style="position:absolute; left:{{ entry_pct }}%; top:0; height:100%; width:2px; background:#a1a1aa;"></div>
                        <div style="position:absolute; left:{{ current_pct }}%; top:0; height:100%; width:3px; background:#fff; border-radius:2px;"></div>
                        <div style="position:absolute; left:4px; top:50%; transform:translateY(-50%); font-size:10px; color:#22c55e;">TP ${{ "%.0f"|format(pos.tp) }}</div>
                        <div style="position:absolute; right:4px; top:50%; transform:translateY(-50%); font-size:10px; color:#ef4444;">SL ${{ "%.0f"|format(pos.sl) }}</div>
                        {% endif %}
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:#71717a; margin-top:2px;">
                        <span>R:R {{ "%.1f"|format(pos.rr) }}x</span>
                        <span>Target {{ "%+.1f"|format(pos.pred_return * 100) }}%</span>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Allocation donut -->
            <div>
                <div style="font-size:12px; color:#71717a; text-transform:uppercase; margin-bottom:12px">Portfolio Allocation</div>
                <div style="height:220px"><canvas id="allocationChart"></canvas></div>
            </div>
        </div>
    </div>
    {% endif %}

    {% if portfolio and portfolio.trade_stats and portfolio.trade_stats.total_trades > 0 %}
    <div class="section-card">
        <div class="section-title">Trade Performance</div>
        <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:8px; margin-bottom:16px;">
            <div class="stat"><div class="stat-val">{{ portfolio.trade_stats.total_trades }}</div><div class="stat-label">Trades</div></div>
            <div class="stat"><div class="stat-val {{ 'up' if portfolio.trade_stats.win_rate > 0.55 else 'neutral' }}">{{ "%.0f"|format(portfolio.trade_stats.win_rate * 100) }}%</div><div class="stat-label">Win Rate</div></div>
            <div class="stat"><div class="stat-val up">{{ "%+.1f"|format(portfolio.trade_stats.avg_win * 100) }}%</div><div class="stat-label">Avg Win</div></div>
            <div class="stat"><div class="stat-val down">{{ "%+.1f"|format(portfolio.trade_stats.avg_loss * 100) }}%</div><div class="stat-label">Avg Loss</div></div>
            <div class="stat"><div class="stat-val {{ 'up' if portfolio.trade_stats.total_pnl > 0 else 'down' }}">{{ "%+.1f"|format(portfolio.trade_stats.total_pnl * 100) }}%</div><div class="stat-label">Total P&L</div></div>
            <div class="stat"><div class="stat-val">{{ "%.0f"|format(portfolio.trade_stats.avg_hold_days) }}d</div><div class="stat-label">Avg Hold</div></div>
        </div>

        {% if portfolio.recent_trades %}
        <div style="font-size:12px; color:#71717a; text-transform:uppercase; margin-bottom:8px;">Recent Trades</div>
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <thead><tr style="color:#71717a; text-align:left; border-bottom:1px solid #27272a;">
                <th style="padding:6px">Date</th><th>Commodity</th><th>Direction</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Days</th><th>Reason</th>
            </tr></thead>
            <tbody>
            {% for t in portfolio.recent_trades %}
            <tr style="border-bottom:1px solid #1e1e22;">
                <td style="padding:4px 6px">{{ t.entry_date }}</td>
                <td>{{ t.commodity }}</td>
                <td style="color:{{ '#22c55e' if t.direction == 'LONG' else '#ef4444' }}">{{ t.direction }}</td>
                <td>${{ "%.2f"|format(t.entry_price) }}</td>
                <td>${{ "%.2f"|format(t.exit_price) }}</td>
                <td class="{{ 'up' if t.pnl_pct > 0 else 'down' }}">{{ "%+.1f"|format(t.pnl_pct * 100) }}%</td>
                <td>{{ t.hold_days }}</td>
                <td style="color:#71717a">{{ t.exit_reason }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if portfolio and portfolio.prediction_accuracy and portfolio.prediction_accuracy.total > 0 %}
    <div class="section-card">
        <div class="section-title">Prediction Accuracy (Realized)</div>
        <div class="stats-grid">
            <div class="stat">
                <div class="stat-val">{{ portfolio.prediction_accuracy.total }}</div>
                <div class="stat-label">Verified</div>
            </div>
            <div class="stat">
                <div class="stat-val {{ 'up' if portfolio.prediction_accuracy.accuracy > 0.55 else 'neutral' }}">{{ "%.0f"|format(portfolio.prediction_accuracy.accuracy * 100) }}%</div>
                <div class="stat-label">Direction Accuracy</div>
            </div>
            {% if portfolio.prediction_accuracy.high_confidence_accuracy %}
            <div class="stat">
                <div class="stat-val {{ 'up' if portfolio.prediction_accuracy.high_confidence_accuracy > 0.6 else 'neutral' }}">{{ "%.0f"|format(portfolio.prediction_accuracy.high_confidence_accuracy * 100) }}%</div>
                <div class="stat-label">High-Conf Accuracy</div>
            </div>
            {% endif %}
        </div>
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

{% if portfolio and portfolio.positions %}
(function() {
    const ctx = document.getElementById('allocationChart');
    if (!ctx) return;
    const positions = {{ portfolio.positions | tojson }};
    const labels = positions.map(p => p.name);
    const sizes = positions.map(p => Math.abs(p.pred_return) * 100);
    const colors = positions.map(p => p.color);
    const cash = 100 - sizes.reduce((a, b) => a + b, 0);
    labels.push('Cash');
    sizes.push(Math.max(cash, 0));
    colors.push('#27272a');
    new Chart(ctx, { type: 'doughnut', data: { labels: labels, datasets: [{
        data: sizes, backgroundColor: colors, borderColor: '#18181b', borderWidth: 2,
    }]}, options: { responsive: true, maintainAspectRatio: false, cutout: '60%',
        plugins: { legend: { position: 'bottom', labels: { color: '#a1a1aa', font: { size: 11 }, padding: 12 } },
            tooltip: { callbacks: { label: function(ctx) { return ctx.label + ': ' + ctx.raw.toFixed(1) + '%'; } } } }
    }});
})();
{% endif %}

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
    portfolio = load_portfolio_data(commodities)
    return render_template_string(TEMPLATE, commodities=commodities, correlation=correlation, portfolio=portfolio)


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
