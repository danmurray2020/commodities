"""Baselines Agent — benchmarks each commodity's model against simple strategies.

Compares XGBoost model performance to four naive baselines:
1. Buy-and-hold: always predict UP
2. Persistence: predict same direction as last horizon-day return
3. Momentum: predict UP if price > 50-day SMA, else DOWN
4. Mean reversion: predict opposite of last horizon-day return (contrarian)

Evaluation uses non-overlapping observations (subsampled every `horizon` rows)
on the last 126 rows of data, matching the holdout convention.

Usage:
    python -m agents baselines              # benchmark all
    python -m agents baselines coffee sugar  # benchmark specific
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import COMMODITIES, CommodityConfig
from .signals import emit_signal
from .design_log import log_observation, log_challenge
from .log import setup_logging, log_event
from .train_utils import evaluate_predictions


logger = setup_logging("baselines")

AGENT_NAME = "baselines"


def benchmark_commodity(cfg: CommodityConfig) -> dict:
    """Benchmark a commodity's model against simple baselines.

    Returns a dict with baseline accuracies, model accuracy, and comparison.
    """
    # --- Load data and model predictions via subprocess isolation ---
    script = f"""
import json, sys, joblib, pandas as pd, numpy as np
sys.path.insert(0, '.')

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)

with open('models/production_metadata.json') as f:
    meta = json.load(f)

horizon = meta.get('horizon', 63)
feature_cols = meta['features']
price_col = '{cfg.price_col}'

# Compute target return and direction the same way as training
df['target_return'] = df[price_col].shift(-horizon) / df[price_col] - 1
df['target_direction'] = (df['target_return'] > 0).astype(int)

# Drop rows where target is NaN (last horizon rows)
df = df.dropna(subset=['target_return'])

# Holdout: last 126 rows (before target NaN trimming)
holdout_size = 126
if len(df) < holdout_size:
    print(json.dumps({{"error": "not enough data", "n_rows": len(df)}}))
    sys.exit(1)

holdout = df.iloc[-holdout_size:]

# Non-overlapping indices within holdout
indices = list(range(0, holdout_size, horizon))
n_ind = len(indices)

# True directions at non-overlapping points
true_dirs = holdout['target_direction'].values[indices]
true_returns = holdout['target_return'].values[indices]

# --- Baseline 1: Buy-and-hold (always predict UP = 1) ---
buyhold_preds = np.ones(n_ind, dtype=int)
buyhold_acc = float(np.mean(buyhold_preds == true_dirs))

# --- Baseline 2: Persistence (predict same direction as last horizon-day return) ---
# For each holdout point, look at the return over the previous horizon days
persistence_preds = []
for idx in indices:
    row_pos = holdout_size - holdout_size + idx  # position within holdout
    abs_pos = len(df) - holdout_size + idx        # absolute position in df
    if abs_pos >= horizon:
        past_return = df[price_col].iloc[abs_pos] / df[price_col].iloc[abs_pos - horizon] - 1
        persistence_preds.append(1 if past_return > 0 else 0)
    else:
        persistence_preds.append(1)  # fallback
persistence_preds = np.array(persistence_preds)
persistence_acc = float(np.mean(persistence_preds == true_dirs))

# --- Baseline 3: Momentum (price > 50-day SMA => UP) ---
sma_50 = df[price_col].rolling(50).mean()
momentum_preds = []
for idx in indices:
    abs_pos = len(df) - holdout_size + idx
    price = df[price_col].iloc[abs_pos]
    sma = sma_50.iloc[abs_pos]
    if pd.notna(sma):
        momentum_preds.append(1 if price > sma else 0)
    else:
        momentum_preds.append(1)
momentum_preds = np.array(momentum_preds)
momentum_acc = float(np.mean(momentum_preds == true_dirs))

# --- Baseline 4: Mean reversion (opposite of last horizon-day return) ---
mean_rev_preds = []
for idx in indices:
    abs_pos = len(df) - holdout_size + idx
    if abs_pos >= horizon:
        past_return = df[price_col].iloc[abs_pos] / df[price_col].iloc[abs_pos - horizon] - 1
        mean_rev_preds.append(0 if past_return > 0 else 1)  # contrarian
    else:
        mean_rev_preds.append(1)
mean_rev_preds = np.array(mean_rev_preds)
mean_rev_acc = float(np.mean(mean_rev_preds == true_dirs))

# --- Model predictions on holdout ---
# Check if we have the needed features
missing = [f for f in feature_cols if f not in df.columns]
model_acc = None
if not missing:
    try:
        reg = joblib.load('models/production_regressor.joblib')
        X_holdout = holdout[feature_cols].values
        # Fill any NaN with forward-fill then 0
        X_df = holdout[feature_cols].ffill().fillna(0)
        X_holdout = X_df.values
        model_preds = reg.predict(X_holdout)
        # Direction from regression predictions
        model_dir_preds = (model_preds > 0).astype(int)
        model_dir_at_indices = model_dir_preds[indices]
        model_acc = float(np.mean(model_dir_at_indices == true_dirs))
    except Exception as e:
        model_acc = None

result = {{
    "horizon": horizon,
    "holdout_size": holdout_size,
    "n_independent": n_ind,
    "up_rate": float(np.mean(true_dirs)),
    "baselines": {{
        "buy_and_hold": round(buyhold_acc, 4),
        "persistence": round(persistence_acc, 4),
        "momentum_sma50": round(momentum_acc, 4),
        "mean_reversion": round(mean_rev_acc, 4),
    }},
    "model_acc": round(model_acc, 4) if model_acc is not None else None,
    "missing_features": missing[:5] if missing else [],
}}
print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir),
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"{cfg.name} baseline benchmark failed: {result.stderr[-500:]}")
            return {"commodity": cfg.name, "error": result.stderr[-300:]}

        data = json.loads(result.stdout.strip().split("\n")[-1])

        if "error" in data:
            logger.error(f"{cfg.name}: {data['error']}")
            return {"commodity": cfg.name, "error": data["error"]}

        data["commodity"] = cfg.name
        data["commodity_key"] = cfg.dir_name

        # --- Compare model vs baselines ---
        baselines = data["baselines"]
        model_acc = data.get("model_acc")
        best_baseline_name = max(baselines, key=baselines.get)
        best_baseline_acc = baselines[best_baseline_name]

        data["best_baseline"] = best_baseline_name
        data["best_baseline_acc"] = best_baseline_acc

        if model_acc is not None:
            data["beats_best_baseline"] = model_acc > best_baseline_acc
            data["margin_vs_best"] = round(model_acc - best_baseline_acc, 4)

            beaten_by = {k: v for k, v in baselines.items() if v >= model_acc}
            data["beaten_by"] = list(beaten_by.keys())
            data["beats_any"] = len(beaten_by) < len(baselines)

            if not data["beats_any"]:
                # Model can't beat ANY baseline
                emit_signal(
                    AGENT_NAME, "baseline_beaten", cfg.dir_name,
                    severity="high",
                    detail=(
                        f"{cfg.name} model ({model_acc:.1%}) underperforms ALL baselines. "
                        f"Best baseline: {best_baseline_name} ({best_baseline_acc:.1%})"
                    ),
                    metadata={
                        "model_acc": model_acc,
                        "baselines": baselines,
                        "n_independent": data["n_independent"],
                    },
                )
                log_challenge(
                    AGENT_NAME,
                    "XGBoost is better than baselines",
                    (
                        f"{cfg.name}: model dir accuracy {model_acc:.1%} loses to all baselines "
                        f"(best: {best_baseline_name} at {best_baseline_acc:.1%}, "
                        f"n={data['n_independent']} independent obs). "
                        f"Model may be overfitting or horizon may be wrong."
                    ),
                    commodity=cfg.dir_name,
                )
                logger.warning(
                    f"{cfg.name}: model LOSES to ALL baselines "
                    f"(model={model_acc:.1%}, best baseline={best_baseline_acc:.1%})"
                )
            elif not data["beats_best_baseline"]:
                log_observation(
                    AGENT_NAME,
                    (
                        f"{cfg.name}: model ({model_acc:.1%}) beaten by "
                        f"{best_baseline_name} ({best_baseline_acc:.1%}) but beats "
                        f"{len(baselines) - len(beaten_by)}/{len(baselines)} baselines"
                    ),
                    commodity=cfg.dir_name,
                )
                logger.info(
                    f"{cfg.name}: model ({model_acc:.1%}) beaten by {best_baseline_name} "
                    f"({best_baseline_acc:.1%})"
                )
            else:
                log_observation(
                    AGENT_NAME,
                    (
                        f"{cfg.name}: model ({model_acc:.1%}) beats all baselines "
                        f"(best: {best_baseline_name} at {best_baseline_acc:.1%}, "
                        f"margin={data['margin_vs_best']:+.1%})"
                    ),
                    commodity=cfg.dir_name,
                )
                logger.info(
                    f"{cfg.name}: model ({model_acc:.1%}) beats best baseline "
                    f"{best_baseline_name} ({best_baseline_acc:.1%})"
                )
        else:
            data["beats_best_baseline"] = None
            data["margin_vs_best"] = None
            data["beaten_by"] = []
            data["beats_any"] = None
            logger.warning(f"{cfg.name}: could not load model for comparison")

        return data

    except subprocess.TimeoutExpired:
        logger.error(f"{cfg.name} baseline benchmark timed out")
        return {"commodity": cfg.name, "error": "timeout"}
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"{cfg.name} baseline parse error: {e}")
        return {"commodity": cfg.name, "error": str(e)}


def run_baselines(commodity_keys: list[str] = None) -> dict:
    """Run baseline benchmarks for all (or specified) commodities.

    Returns dict mapping commodity key to benchmark results.
    """
    targets = commodity_keys or list(COMMODITIES.keys())
    results = {}

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            logger.warning(f"Unknown commodity: {key}")
            continue

        logger.info(f"Benchmarking {cfg.name}...")
        results[key] = benchmark_commodity(cfg)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark commodity models against simple baselines"
    )
    parser.add_argument(
        "commodities", nargs="*",
        help="Specific commodities to benchmark (default: all)"
    )
    args = parser.parse_args()

    results = run_baselines(args.commodities or None)

    # --- Summary table ---
    print(f"\n{'='*72}")
    print("BASELINE BENCHMARK SUMMARY")
    print(f"{'='*72}")
    print(
        f"  {'Commodity':<12} {'Model':>7} {'BuyHold':>8} {'Persist':>8} "
        f"{'Momentum':>9} {'MeanRev':>8}  {'Status'}"
    )
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*8}  {'-'*15}")

    n_beating_all = 0
    n_losing_all = 0
    n_total = 0

    for key, res in results.items():
        if "error" in res:
            print(f"  {key:<12} {'ERROR':>7}  {res.get('error', '')[:40]}")
            continue

        n_total += 1
        bl = res["baselines"]
        model = res.get("model_acc")
        model_str = f"{model:.1%}" if model is not None else "  N/A"

        if res.get("beats_any") is False:
            status = "LOSES TO ALL"
            n_losing_all += 1
        elif res.get("beats_best_baseline"):
            status = "BEATS ALL"
            n_beating_all += 1
        elif model is not None:
            status = f"beaten by {len(res.get('beaten_by', []))}"
        else:
            status = "no model"

        print(
            f"  {res['commodity']:<12} {model_str:>7} "
            f"{bl['buy_and_hold']:>7.1%} {bl['persistence']:>7.1%} "
            f"{bl['momentum_sma50']:>8.1%} {bl['mean_reversion']:>7.1%}"
            f"  {status}"
        )

    print(f"\n  {n_beating_all}/{n_total} models beat all baselines, "
          f"{n_losing_all}/{n_total} lose to all baselines")

    if n_losing_all > 0:
        print(f"\n  WARNING: {n_losing_all} model(s) cannot beat ANY simple baseline.")
        print("  Consider retraining or reviewing the prediction horizon.")


if __name__ == "__main__":
    main()
