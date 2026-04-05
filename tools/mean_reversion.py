"""Mean reversion probability analysis for commodities.

Answers: given the current z-score and market conditions, how likely is
a reversion to the mean over the next 63 trading days?
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


def analyze_mean_reversion(project_dir: Path, name: str, price_col: str):
    """Analyze mean-reversion probability from historical data."""

    script = f"""
import json, sys, pandas as pd, numpy as np
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = df.dropna()

price = df['{price_col}']

# Current state
latest = df.iloc[-1]
current_zscore_126 = latest.get('zscore_126d', 0)
current_zscore_252 = latest.get('zscore_252d', 0)
current_rsi = latest.get('rsi_14', 50)
current_vol = latest.get('volatility_21d', 0)
current_price_vs_sma50 = latest.get('price_vs_sma_50', 0)
current_price_vs_sma200 = latest.get('price_vs_sma_200', 0)

# COT data
cot_net_spec = latest.get('cot_net_spec', 0)
cot_oi_change = latest.get('cot_oi_change', 0)

# Compute 63-day forward returns for historical analysis
forward_63d = price.shift(-63) / price - 1

# --- Historical base rates by z-score bucket ---
zscore = df.get('zscore_252d', pd.Series(dtype=float))
buckets = []
for low, high, label in [
    (-999, -2.0, 'z < -2.0 (extreme low)'),
    (-2.0, -1.5, '-2.0 < z < -1.5'),
    (-1.5, -1.0, '-1.5 < z < -1.0'),
    (-1.0, -0.5, '-1.0 < z < -0.5'),
    (-0.5, 0.5, '-0.5 < z < 0.5 (neutral)'),
    (0.5, 1.0, '0.5 < z < 1.0'),
    (1.0, 1.5, '1.0 < z < 1.5'),
    (1.5, 2.0, '1.5 < z < 2.0'),
    (2.0, 999, 'z > 2.0 (extreme high)'),
]:
    mask = (zscore >= low) & (zscore < high) & forward_63d.notna()
    if mask.sum() > 5:
        rets = forward_63d[mask]
        # "Reverts" means: if z < 0, price goes up; if z > 0, price goes down
        if low < 0:
            revert_pct = (rets > 0).mean()
        else:
            revert_pct = (rets < 0).mean()
        buckets.append({{
            'label': label, 'n': int(mask.sum()),
            'revert_pct': float(revert_pct),
            'mean_return': float(rets.mean()),
            'median_return': float(rets.median()),
        }})

# --- RSI divergence detection ---
# Check if RSI is making higher lows while price makes lower lows (bullish divergence)
# or RSI making lower highs while price makes higher highs (bearish divergence)
rsi = df.get('rsi_14', pd.Series(dtype=float))
rsi_21d_ago = rsi.shift(21)
price_21d_ago = price.shift(21)
bullish_divergence = False
bearish_divergence = False
if len(rsi) > 21:
    curr_rsi = rsi.iloc[-1]
    prev_rsi = rsi_21d_ago.iloc[-1]
    curr_price = price.iloc[-1]
    prev_price = price_21d_ago.iloc[-1]
    if not (pd.isna(prev_rsi) or pd.isna(prev_price)):
        # Bullish: price lower but RSI higher
        if curr_price < prev_price and curr_rsi > prev_rsi:
            bullish_divergence = True
        # Bearish: price higher but RSI lower
        if curr_price > prev_price and curr_rsi < prev_rsi:
            bearish_divergence = True

# --- Z-score duration (how many consecutive days in current direction) ---
if len(zscore) > 0:
    current_sign = 1 if zscore.iloc[-1] > 0 else -1
    duration = 0
    for i in range(len(zscore)-1, -1, -1):
        if (zscore.iloc[i] > 0 and current_sign > 0) or (zscore.iloc[i] <= 0 and current_sign <= 0):
            duration += 1
        else:
            break
else:
    duration = 0

# --- Volatility regime (is vol declining from peak = panic subsiding?) ---
vol = df.get('volatility_21d', pd.Series(dtype=float))
vol_declining = False
if len(vol) > 21:
    vol_peak_21d = vol.iloc[-21:].max()
    vol_now = vol.iloc[-1]
    vol_declining = vol_now < vol_peak_21d * 0.85  # 15%+ decline from recent peak

# --- COT positioning shift ---
cot_spec = df.get('cot_net_spec', pd.Series(dtype=float))
cot_reversing = False
if len(cot_spec) > 4:
    # Check if speculative positioning is reversing direction
    recent_change = cot_spec.iloc[-1] - cot_spec.iloc[-4]  # ~1 month of weekly data
    if current_zscore_252 < 0 and recent_change > 0:
        cot_reversing = True  # specs covering shorts / adding longs while price oversold
    elif current_zscore_252 > 0 and recent_change < 0:
        cot_reversing = True  # specs reducing longs while price overbought

# --- Composite probability ---
# Start with base rate from z-score bucket
base_revert = 0.5
for b in buckets:
    low_str = b['label']
    if current_zscore_252 < -2 and 'extreme low' in low_str:
        base_revert = b['revert_pct']
    elif -2 <= current_zscore_252 < -1.5 and '-2.0 < z < -1.5' in low_str:
        base_revert = b['revert_pct']
    elif -1.5 <= current_zscore_252 < -1.0 and '-1.5 < z < -1.0' in low_str:
        base_revert = b['revert_pct']
    elif -1.0 <= current_zscore_252 < -0.5 and '-1.0 < z < -0.5' in low_str:
        base_revert = b['revert_pct']
    elif -0.5 <= current_zscore_252 < 0.5 and 'neutral' in low_str:
        base_revert = b['revert_pct']
    elif 0.5 <= current_zscore_252 < 1.0 and '0.5 < z < 1.0' in low_str:
        base_revert = b['revert_pct']
    elif 1.0 <= current_zscore_252 < 1.5 and '1.0 < z < 1.5' in low_str:
        base_revert = b['revert_pct']
    elif 1.5 <= current_zscore_252 < 2.0 and '1.5 < z < 2.0' in low_str:
        base_revert = b['revert_pct']
    elif current_zscore_252 >= 2 and 'extreme high' in low_str:
        base_revert = b['revert_pct']

# Adjustments
adjustment = 0
signals_for = []
signals_against = []

is_oversold = current_zscore_252 < 0

if is_oversold:
    if bullish_divergence:
        adjustment += 0.08
        signals_for.append('Bullish RSI divergence (+8%)')
    if vol_declining:
        adjustment += 0.05
        signals_for.append('Volatility declining from peak (+5%)')
    if cot_reversing:
        adjustment += 0.07
        signals_for.append('Speculators covering/adding longs (+7%)')
    if duration > 120:
        adjustment -= 0.10
        signals_against.append(f'Extended duration ({{duration}}d below zero) - possible regime shift (-10%)')
    elif duration > 60:
        adjustment -= 0.05
        signals_against.append(f'Prolonged deviation ({{duration}}d) (-5%)')
    if current_rsi < 25:
        adjustment += 0.05
        signals_for.append(f'RSI deeply oversold ({{current_rsi:.0f}}) (+5%)')
else:
    if bearish_divergence:
        adjustment += 0.08
        signals_for.append('Bearish RSI divergence (+8%)')
    if vol_declining:
        adjustment += 0.05
        signals_for.append('Volatility declining from peak (+5%)')
    if cot_reversing:
        adjustment += 0.07
        signals_for.append('Speculators reducing longs (+7%)')
    if duration > 120:
        adjustment -= 0.10
        signals_against.append(f'Extended duration ({{duration}}d above zero) - possible regime shift (-10%)')
    elif duration > 60:
        adjustment -= 0.05
        signals_against.append(f'Prolonged deviation ({{duration}}d) (-5%)')
    if current_rsi > 75:
        adjustment += 0.05
        signals_for.append(f'RSI deeply overbought ({{current_rsi:.0f}}) (+5%)')

composite = min(0.95, max(0.05, base_revert + adjustment))

result = {{
    'date': df.index[-1].strftime('%Y-%m-%d'),
    'price': float(price.iloc[-1]),
    'zscore_126d': float(current_zscore_126),
    'zscore_252d': float(current_zscore_252),
    'rsi': float(current_rsi),
    'volatility_21d': float(current_vol),
    'price_vs_sma50': float(current_price_vs_sma50),
    'price_vs_sma200': float(current_price_vs_sma200),
    'cot_net_spec': float(cot_net_spec),
    'zscore_duration_days': duration,
    'bullish_divergence': bool(bullish_divergence),
    'bearish_divergence': bool(bearish_divergence),
    'vol_declining': bool(vol_declining),
    'cot_reversing': bool(cot_reversing),
    'base_revert_probability': float(base_revert),
    'composite_revert_probability': float(composite),
    'signals_for': signals_for,
    'signals_against': signals_against,
    'historical_buckets': buckets,
}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  ERROR parsing output")
        return None


def print_analysis(name: str, r: dict):
    """Pretty-print the mean reversion analysis."""
    direction = "OVERSOLD" if r["zscore_252d"] < 0 else "OVERBOUGHT" if r["zscore_252d"] > 0 else "NEUTRAL"

    print(f"\n{'='*60}")
    print(f"  {name} MEAN REVERSION ANALYSIS")
    print(f"  {r['date']} | ${r['price']:.2f}")
    print(f"{'='*60}")

    print(f"\n  Current State: {direction}")
    print(f"  Z-score (6mo):  {r['zscore_126d']:+.2f}")
    print(f"  Z-score (1yr):  {r['zscore_252d']:+.2f}")
    print(f"  RSI:            {r['rsi']:.1f}")
    print(f"  Volatility:     {r['volatility_21d']:.1%}")
    print(f"  vs SMA-50:      {r['price_vs_sma50']:+.1%}")
    print(f"  vs SMA-200:     {r['price_vs_sma200']:+.1%}")
    print(f"  COT net spec:   {r['cot_net_spec']:,.0f}")
    print(f"  Days in zone:   {r['zscore_duration_days']}")

    print(f"\n  --- Reversion Probability ---")
    print(f"  Base rate (from z-score): {r['base_revert_probability']:.0%}")

    if r["signals_for"]:
        print(f"\n  Signals SUPPORTING reversion:")
        for s in r["signals_for"]:
            print(f"    + {s}")

    if r["signals_against"]:
        print(f"\n  Signals AGAINST reversion:")
        for s in r["signals_against"]:
            print(f"    - {s}")

    print(f"\n  COMPOSITE PROBABILITY: {r['composite_revert_probability']:.0%}")

    verdict = ""
    prob = r["composite_revert_probability"]
    if prob >= 0.70:
        verdict = "HIGH probability of reversion — favorable setup"
    elif prob >= 0.55:
        verdict = "MODERATE probability — wait for confirmation signals"
    else:
        verdict = "LOW probability — may be a regime shift, not a temporary dip"
    print(f"  VERDICT: {verdict}")

    # Historical context
    print(f"\n  --- Historical Base Rates (63-day forward) ---")
    print(f"  {'Z-score Bucket':<30} {'N':>5} {'Revert%':>8} {'Mean Ret':>10} {'Med Ret':>10}")
    print(f"  {'-'*68}")
    for b in r["historical_buckets"]:
        marker = "  <--" if (
            (r["zscore_252d"] < -2 and "extreme low" in b["label"]) or
            (r["zscore_252d"] >= 2 and "extreme high" in b["label"]) or
            (abs(r["zscore_252d"]) < 2 and b["label"].split("<")[0].strip("-. ") != "" and
             any(str(round(r["zscore_252d"], 1)) in b["label"] for _ in [1]))
        ) else ""
        print(f"  {b['label']:<30} {b['n']:>5} {b['revert_pct']:>7.0%} {b['mean_return']:>+9.1%} {b['median_return']:>+9.1%}{marker}")


def main():
    print("=" * 60)
    print("MEAN REVERSION PROBABILITY ANALYSIS")
    print("=" * 60)

    for project_dir, name, price_col in [
        (COFFEE_DIR, "COFFEE", "coffee_close"),
        (COCOA_DIR, "COCOA", "cocoa_close"),
        (SUGAR_DIR, "SUGAR", "sugar_close"),
        (NATGAS_DIR, "NATGAS", "natgas_close"),
        (SOYBEANS_DIR, "SOYBEANS", "soybeans_close"),
        (WHEAT_DIR, "WHEAT", "wheat_close"),
        (COPPER_DIR, "COPPER", "copper_close"),
    ]:
        r = analyze_mean_reversion(project_dir, name, price_col)
        if r:
            print_analysis(name, r)
        else:
            print(f"\n  {name}: Failed to analyze")


if __name__ == "__main__":
    main()
