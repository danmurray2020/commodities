"""Train multi-horizon ensemble models for all commodities.

Usage:
    python tools/train_ensemble.py                  # all commodities
    python tools/train_ensemble.py coffee sugar      # specific commodities
"""

import json
import sys
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import COMMODITIES
from agents.ensemble import train_multi_model_horizon, HORIZONS, MODEL_TYPES
from agents.design_log import log_observation
import joblib


def train_commodity_ensemble(key: str) -> dict:
    """Train multi-horizon ensemble for a single commodity."""
    cfg = COMMODITIES[key]
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TRAINING: {cfg.name}")
    print(f"{'='*60}")

    # Load data via subprocess to handle per-commodity imports
    script = f"""
import json, sys, pandas as pd
sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data

sys.path.insert(0, '..')
from agents.regime_features import add_regime_features

df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = add_regime_features(df, price_col='{cfg.price_col}')
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = df.ffill()
df = df.dropna()

exclude = {{'{cfg.price_col}', 'Open', 'High', 'Low', 'Volume', 'target_return', 'target_direction'}}
feature_cols = [c for c in df.columns if c not in exclude]

df.to_csv('/tmp/_ensemble_data_{key}.csv')
with open('/tmp/_ensemble_features_{key}.json', 'w') as f:
    json.dump(feature_cols, f)
print(json.dumps({{"rows": len(df), "features": len(feature_cols)}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        cwd=str(cfg.project_dir),
        timeout=120,
    )

    if result.returncode != 0:
        print(f"  ERROR loading data: {result.stderr[-300:]}")
        return {"status": "data_error", "commodity": cfg.name}

    import pandas as pd
    df = pd.read_csv(f"/tmp/_ensemble_data_{key}.csv", index_col=0, parse_dates=True)
    with open(f"/tmp/_ensemble_features_{key}.json") as f:
        all_features = json.load(f)

    info = json.loads(result.stdout.strip().split("\n")[-1])
    print(f"  Data: {info['rows']} rows, {info['features']} features")

    # Train multi-model at each horizon
    all_model_results = []
    for horizon in HORIZONS:
        print(f"\n  --- Horizon: {horizon}d ---")
        model_results = train_multi_model_horizon(
            df, all_features, horizon=horizon, price_col=cfg.price_col,
            optuna_trials=60,
        )
        all_model_results.extend(model_results)

    if not all_model_results:
        return {"status": "no_models", "commodity": cfg.name}

    # Save all models
    models_dir = cfg.models_dir
    models_dir.mkdir(exist_ok=True)

    # Weight by direction accuracy
    dir_accs = [r["avg_dir_acc"] for r in all_model_results]
    max_acc = max(max(dir_accs), 0.50)

    model_meta = []
    for r in all_model_results:
        horizon = r["horizon"]
        model_type = r["model_type"]
        weight = max(r["avg_dir_acc"] / max_acc, 0.1)

        # Save with model type in filename
        joblib.dump(r["regressor"], models_dir / f"ensemble_reg_{horizon}d_{model_type}.joblib")
        joblib.dump(r["classifier"], models_dir / f"ensemble_clf_{horizon}d_{model_type}.joblib")
        # Also save primary (for backward compat with ensemble_predict)
        joblib.dump(r["regressor"], models_dir / f"ensemble_reg_{horizon}d.joblib")
        joblib.dump(r["classifier"], models_dir / f"ensemble_clf_{horizon}d.joblib")

        model_meta.append({
            "horizon": horizon,
            "model_type": model_type,
            "features": r["features"],
            "n_features": len(r["features"]),
            "weight": round(weight, 3),
            "avg_dir_acc": round(r["avg_dir_acc"], 4),
            "avg_clf_acc": round(r["avg_clf_acc"], 4),
        })

    # Summary
    print(f"\n  --- Ensemble Summary ({len(model_meta)} models) ---")
    for mm in sorted(model_meta, key=lambda x: x["avg_dir_acc"], reverse=True):
        print(f"  {mm['horizon']:>3}d {mm['model_type']:<10} (w={mm['weight']:.2f}): "
              f"dir={mm['avg_dir_acc']:.1%}, clf={mm['avg_clf_acc']:.1%}")

    best_model = max(model_meta, key=lambda x: x["avg_dir_acc"])
    avg_acc = float(np.mean([m["avg_dir_acc"] for m in model_meta]))
    best_acc = best_model["avg_dir_acc"]

    ensemble_meta = {
        "commodity": cfg.name,
        "ticker": cfg.ticker,
        "trained_at": datetime.now().isoformat(),
        "models": model_meta,
        "n_models": len(model_meta),
        "best_model": f"{best_model['horizon']}d_{best_model['model_type']}",
        "best_accuracy": best_acc,
        "avg_accuracy": avg_acc,
        # Backward compat
        "horizons": [{"horizon": m["horizon"], "features": m["features"],
                       "weight": m["weight"], "metrics": {"avg_dir_acc": m["avg_dir_acc"],
                       "avg_clf_acc": m["avg_clf_acc"], "avg_spearman": 0}}
                      for m in model_meta],
        "features": best_model["features"],
        "horizon": best_model["horizon"],
        "regression": {"avg_accuracy": avg_acc},
        "classification": {"avg_accuracy": float(np.mean([m["avg_clf_acc"] for m in model_meta]))},
    }

    with open(models_dir / "ensemble_metadata.json", "w") as f:
        json.dump(ensemble_meta, f, indent=2, default=str)

    log_observation("ensemble_training",
        f"Trained {len(model_meta)} models, best={best_model['horizon']}d "
        f"{best_model['model_type']} at {best_acc:.1%}",
        cfg.name)

    return {
        "status": "ok",
        "commodity": cfg.name,
        "n_models": len(model_meta),
        "best": f"{best_model['horizon']}d {best_model['model_type']} ({best_acc:.1%})",
        "avg_acc": avg_acc,
        "models": model_meta,
    }


def _resolve_commodity_key(name: str) -> str | None:
    """Resolve a commodity name or directory name to its COMMODITIES key.

    Accepts the canonical key (e.g. 'cocoa'), the directory name (e.g. 'chocolate'),
    the display name (e.g. 'Cocoa'), or the ticker (e.g. 'CC=F').
    """
    # Direct key match
    if name in COMMODITIES:
        return name
    # Match by dir_name, display name (case-insensitive), or ticker
    name_lower = name.lower()
    for key, cfg in COMMODITIES.items():
        if cfg.dir_name == name or cfg.name.lower() == name_lower or cfg.ticker == name:
            return key
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train multi-horizon ensemble models")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    args = parser.parse_args()

    targets = args.commodities or list(COMMODITIES.keys())

    results = {}
    for name in targets:
        key = _resolve_commodity_key(name)
        if key is None:
            print(f"Unknown commodity: {name}")
            continue
        results[key] = train_commodity_ensemble(key)

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")

    for key, r in results.items():
        cfg = COMMODITIES[key]
        if r["status"] != "ok":
            print(f"  {cfg.name:<15} [{r['status']}]")
            continue

        print(f"  {cfg.name:<15} {r['n_models']} models, best: {r['best']}, avg: {r['avg_acc']:.1%}")


if __name__ == "__main__":
    main()
