"""Promote trained ensembles to production for any commodities that lack
a `production_metadata.json` (and the matching `production_*.joblib`
files that the prediction agent expects).

Background
----------
`tools/train_ensemble.py` writes per-horizon, per-model joblib files plus
an `ensemble_metadata.json`, but it does NOT write the
`production_metadata.json` / `production_regressor.joblib` /
`production_classifier.joblib` triple that the rest of the system reads.
The 7 original commodities had this written by hand at some point in the
past; the 13 newer commodities never did, which is why every downstream
agent reports them as `no_model` / `missing metadata`.

This script picks the **best single model** the ensemble training already
identified (`ensemble_metadata.json["best_model"]`, e.g. `"10d_lightgbm"`)
and promotes it: it copies the joblib pair to the production filenames
and emits a `production_metadata.json` containing the feature list,
horizon, and accuracy metrics that `agents/prediction.py` and the model
quality agent rely on.

Idempotent — by default it skips any commodity that already has a
`production_metadata.json`. Pass `--force` to re-promote.

Usage
-----
    python3 tools/promote_ensembles.py                  # all commodities
    python3 tools/promote_ensembles.py crude_oil gold   # specific keys
    python3 tools/promote_ensembles.py --force          # re-promote everything
    python3 tools/promote_ensembles.py --dry-run        # show what would happen
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on sys.path so we can import agents.config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.config import COMMODITIES, CommodityConfig, TRADE_DEFAULTS


def _parse_best_model(best: str) -> tuple[int, str]:
    """Parse a `best_model` string like '10d_lightgbm' into (horizon, type)."""
    if not best or "_" not in best:
        raise ValueError(f"Cannot parse best_model: {best!r}")
    horizon_part, model_type = best.split("_", 1)
    if not horizon_part.endswith("d"):
        raise ValueError(f"Unexpected horizon segment in best_model: {best!r}")
    return int(horizon_part[:-1]), model_type


def _find_model_entry(meta: dict, horizon: int, model_type: str) -> dict | None:
    """Return the `models[]` entry matching both horizon and model_type.

    `models[]` has 3 entries per horizon (xgboost / lightgbm / ridge); we want
    the specific one named in `best_model`. `horizons[]` does not carry the
    model_type field, so we always search `models[]` first.
    """
    for entry in meta.get("models") or []:
        if entry.get("horizon") == horizon and entry.get("model_type") == model_type:
            return entry
    # Fallback: any entry matching the horizon (shouldn't normally trigger)
    for entry in (meta.get("models") or []) + (meta.get("horizons") or []):
        if entry.get("horizon") == horizon:
            return entry
    return None


def _build_production_metadata(
    cfg: CommodityConfig,
    ensemble_meta: dict,
    horizon: int,
    model_type: str,
    horizon_entry: dict,
) -> dict:
    """Build the `production_metadata.json` payload from ensemble metadata."""
    features = horizon_entry.get("features") or ensemble_meta.get("features") or []

    # Pull regression accuracy: prefer the per-model entry, fall back to the
    # ensemble-level number.
    reg_acc = (
        horizon_entry.get("avg_dir_acc")
        or horizon_entry.get("metrics", {}).get("avg_dir_acc")
        or ensemble_meta.get("regression", {}).get("avg_accuracy")
    )
    clf_acc = (
        horizon_entry.get("avg_clf_acc")
        or horizon_entry.get("metrics", {}).get("avg_clf_acc")
        or ensemble_meta.get("classification", {}).get("avg_accuracy")
    )

    return {
        "commodity": cfg.dir_name,
        "ticker": cfg.ticker,
        "horizon": horizon,
        "features": features,
        "n_features": len(features),
        "purge_gap": horizon,
        "strategy": {
            "confidence_threshold": cfg.confidence_threshold,
            "stop_loss_pct": TRADE_DEFAULTS.stop_loss_pct,
            "take_profit_multiplier": TRADE_DEFAULTS.take_profit_multiplier,
            "max_hold_days": TRADE_DEFAULTS.max_hold_days,
            "allow_short": TRADE_DEFAULTS.allow_short,
        },
        "regression": {
            "avg_accuracy": reg_acc,
        },
        "classification": {
            "avg_accuracy": clf_acc,
        },
        "promoted_from_ensemble": {
            "source_metadata": "ensemble_metadata.json",
            "best_model": ensemble_meta.get("best_model"),
            "model_type": model_type,
            "best_accuracy": ensemble_meta.get("best_accuracy"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def promote(
    key: str,
    cfg: CommodityConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Promote one commodity. Returns a status dict."""
    models_dir = cfg.models_dir
    metadata_path = models_dir / "production_metadata.json"
    ensemble_path = models_dir / "ensemble_metadata.json"

    if not ensemble_path.exists():
        return {"key": key, "status": "skipped", "reason": "no ensemble_metadata.json"}

    if metadata_path.exists() and not force:
        return {"key": key, "status": "skipped", "reason": "already promoted"}

    ensemble_meta = json.loads(ensemble_path.read_text())
    best = ensemble_meta.get("best_model")
    if not best:
        return {"key": key, "status": "error", "reason": "no best_model in ensemble metadata"}

    try:
        horizon, model_type = _parse_best_model(best)
    except ValueError as e:
        return {"key": key, "status": "error", "reason": str(e)}

    reg_src = models_dir / f"ensemble_reg_{horizon}d_{model_type}.joblib"
    clf_src = models_dir / f"ensemble_clf_{horizon}d_{model_type}.joblib"
    if not reg_src.exists() or not clf_src.exists():
        return {
            "key": key,
            "status": "error",
            "reason": f"missing source joblibs ({reg_src.name}, {clf_src.name})",
        }

    horizon_entry = _find_model_entry(ensemble_meta, horizon, model_type)
    if not horizon_entry:
        return {
            "key": key,
            "status": "error",
            "reason": f"no models[] entry for {horizon}d {model_type}",
        }

    payload = _build_production_metadata(
        cfg, ensemble_meta, horizon, model_type, horizon_entry
    )

    if dry_run:
        return {
            "key": key,
            "status": "would_promote",
            "best_model": best,
            "horizon": horizon,
            "n_features": payload["n_features"],
            "reg_acc": payload["regression"]["avg_accuracy"],
        }

    reg_dst = models_dir / "production_regressor.joblib"
    clf_dst = models_dir / "production_classifier.joblib"
    shutil.copy2(reg_src, reg_dst)
    shutil.copy2(clf_src, clf_dst)
    metadata_path.write_text(json.dumps(payload, indent=2, default=str))

    return {
        "key": key,
        "status": "promoted",
        "best_model": best,
        "horizon": horizon,
        "n_features": payload["n_features"],
        "reg_acc": payload["regression"]["avg_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Promote trained ensembles to production metadata + joblibs"
    )
    parser.add_argument(
        "commodities",
        nargs="*",
        help="Specific commodity keys (default: all 20)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-promote even if production_metadata.json already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen, do not write files",
    )
    args = parser.parse_args()

    targets = args.commodities or list(COMMODITIES.keys())
    results = []
    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            results.append({"key": key, "status": "unknown_key"})
            continue
        results.append(promote(key, cfg, force=args.force, dry_run=args.dry_run))

    width_key = max(len(r["key"]) for r in results)
    print(f"\n{'commodity':<{width_key}}  status         best_model         horizon  n_feat  reg_acc")
    print("-" * 80)
    for r in results:
        bm = r.get("best_model", "—") or "—"
        horizon = r.get("horizon", "—")
        nf = r.get("n_features", "—")
        ra = r.get("reg_acc")
        ra_str = f"{ra:.3f}" if isinstance(ra, (int, float)) else "—"
        reason = r.get("reason", "")
        print(
            f"{r['key']:<{width_key}}  "
            f"{r['status']:<14} {str(bm):<18} {str(horizon):>7}  {str(nf):>6}  {ra_str:>7}  "
            f"{reason}"
        )

    summary = {}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
