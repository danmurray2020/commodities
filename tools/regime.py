"""Regime detection for commodities using Hidden Markov Models.

Identifies market regimes (trending, mean-reverting, crisis/bubble) and
provides the current regime as context for trading decisions.
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"
SUGAR_DIR = Path(__file__).parent.parent / "sugar"
NATGAS_DIR = Path(__file__).parent.parent / "natgas"
SOYBEANS_DIR = Path(__file__).parent.parent / "soybeans"
WHEAT_DIR = Path(__file__).parent.parent / "wheat"
COPPER_DIR = Path(__file__).parent.parent / "copper"


def detect_regimes(project_dir: Path, name: str, price_col: str, n_regimes: int = 3):
    """Detect market regimes using a simple volatility + trend model.

    Uses a rule-based approach (more robust than HMM for small datasets):
    - Regime 0: Low vol, trending (normal market)
    - Regime 1: High vol, trending (strong trend / momentum)
    - Regime 2: Extreme vol (crisis / bubble)
    """
    script = f"""
import json, sys, pandas as pd, numpy as np
sys.path.insert(0, '.')

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
price = df['{price_col}']

# Compute regime features
returns = price.pct_change()
vol_21d = returns.rolling(21).std() * np.sqrt(252)
vol_63d = returns.rolling(63).std() * np.sqrt(252)
trend_strength = abs(price / price.rolling(63).mean() - 1)  # absolute deviation from 63d mean
zscore_252d = (price - price.rolling(252).mean()) / price.rolling(252).std()

# Define regime thresholds based on historical distribution
vol_median = vol_21d.median()
vol_75 = vol_21d.quantile(0.75)
vol_95 = vol_21d.quantile(0.95)
trend_median = trend_strength.median()
trend_75 = trend_strength.quantile(0.75)

# Classify each day
regimes = pd.Series(0, index=df.index)  # default: normal

# High vol trending
regimes[(vol_21d > vol_75) & (trend_strength > trend_median)] = 1

# Extreme / bubble / crisis
regimes[(vol_21d > vol_95) | (zscore_252d.abs() > 2.5)] = 2

regimes = regimes.dropna()

# Current state
latest_vol = float(vol_21d.iloc[-1]) if not vol_21d.empty else 0
latest_trend = float(trend_strength.iloc[-1]) if not trend_strength.empty else 0
latest_zscore = float(zscore_252d.iloc[-1]) if not zscore_252d.empty else 0
current_regime = int(regimes.iloc[-1])

# Regime statistics
regime_stats = []
for r in range(3):
    mask = regimes == r
    if mask.sum() > 0:
        r_returns = returns[mask].dropna()
        r_vol = vol_21d[mask].dropna()
        # Forward 63-day returns for each regime
        fwd = (price.shift(-63) / price - 1)[mask].dropna()
        regime_stats.append({{
            'regime': r,
            'name': ['Normal', 'High Vol Trend', 'Extreme/Bubble'][r],
            'days': int(mask.sum()),
            'pct_time': float(mask.mean()),
            'avg_daily_return': float(r_returns.mean()),
            'avg_volatility': float(r_vol.mean()),
            'fwd_63d_mean': float(fwd.mean()) if len(fwd) > 0 else 0,
            'fwd_63d_positive': float((fwd > 0).mean()) if len(fwd) > 0 else 0.5,
        }})

# Regime transitions: what typically follows each regime?
transitions = {{}}
for r in range(3):
    mask = regimes == r
    # Next regime after being in this one
    next_regimes = regimes.shift(-21)[mask].dropna()
    if len(next_regimes) > 0:
        transitions[r] = {{
            'to_normal': float((next_regimes == 0).mean()),
            'to_high_vol': float((next_regimes == 1).mean()),
            'to_extreme': float((next_regimes == 2).mean()),
        }}

# Historical regime timeline (last 2 years for display)
cutoff_idx = max(0, len(regimes) - 504)
recent = regimes.iloc[cutoff_idx:]
timeline_dates = [d.strftime('%Y-%m-%d') for d in recent.index]
timeline_regimes = [int(r) for r in recent.values]

result = {{
    'current_regime': current_regime,
    'current_regime_name': ['Normal', 'High Vol Trend', 'Extreme/Bubble'][current_regime],
    'current_volatility': latest_vol,
    'current_trend_strength': latest_trend,
    'current_zscore': latest_zscore,
    'vol_thresholds': {{'median': float(vol_median), '75th': float(vol_75), '95th': float(vol_95)}},
    'regime_stats': regime_stats,
    'transitions': transitions,
    'timeline_dates': timeline_dates,
    'timeline_regimes': timeline_regimes,
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=60,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def print_regime_analysis(name: str, r: dict):
    """Pretty-print regime analysis."""
    regime_colors = {0: "NORMAL", 1: "HIGH VOL TREND", 2: "EXTREME/BUBBLE"}

    print(f"\n{'='*60}")
    print(f"  {name} REGIME ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  Current Regime: {r['current_regime_name']}")
    print(f"  Volatility:     {r['current_volatility']:.1%} "
          f"(median: {r['vol_thresholds']['median']:.1%}, "
          f"75th: {r['vol_thresholds']['75th']:.1%}, "
          f"95th: {r['vol_thresholds']['95th']:.1%})")
    print(f"  Trend strength: {r['current_trend_strength']:.1%}")
    print(f"  Z-score (1yr):  {r['current_zscore']:+.2f}")

    print(f"\n  --- Regime Statistics ---")
    print(f"  {'Regime':<20} {'Time%':>6} {'Avg Vol':>8} {'63d Fwd':>9} {'P(up 63d)':>10}")
    print(f"  {'-'*56}")
    for s in r["regime_stats"]:
        marker = " <-- NOW" if s["regime"] == r["current_regime"] else ""
        print(f"  {s['name']:<20} {s['pct_time']:>5.0%} {s['avg_volatility']:>7.1%} "
              f"{s['fwd_63d_mean']:>+8.1%} {s['fwd_63d_positive']:>9.0%}{marker}")

    if r["transitions"]:
        print(f"\n  --- What Happens Next (21-day transition probabilities) ---")
        current = r["current_regime"]
        if current in r["transitions"]:
            t = r["transitions"][current]
            print(f"  From {r['current_regime_name']}:")
            print(f"    -> Normal:       {t['to_normal']:.0%}")
            print(f"    -> High Vol:     {t['to_high_vol']:.0%}")
            print(f"    -> Extreme:      {t['to_extreme']:.0%}")

    # Trading implication
    print(f"\n  --- Trading Implication ---")
    if r["current_regime"] == 0:
        print(f"  Normal regime — standard model predictions are most reliable.")
        print(f"  Use default confidence threshold and position sizing.")
    elif r["current_regime"] == 1:
        print(f"  High volatility trend — momentum signals work well but risk is elevated.")
        print(f"  Consider tighter stops and smaller position sizes.")
    elif r["current_regime"] == 2:
        print(f"  EXTREME regime — model accuracy degrades significantly.")
        print(f"  Raise confidence threshold to 80%+ or skip trades entirely.")
        print(f"  Mean reversion is unreliable in this regime.")


def main():
    print("=" * 60)
    print("REGIME DETECTION")
    print("=" * 60)

    for project_dir, name, price_col in [
        (COFFEE_DIR, "Coffee", "coffee_close"),
        (COCOA_DIR, "Cocoa", "cocoa_close"),
    ]:
        r = detect_regimes(project_dir, name, price_col)
        if r:
            print_regime_analysis(name, r)
        else:
            print(f"\n  {name}: Failed to analyze")


if __name__ == "__main__":
    main()
