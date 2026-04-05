"""Generate a comprehensive accuracy report across all commodity models.

Usage:
    python tools/accuracy_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.config import COMMODITIES


def load_metadata(key):
    cfg = COMMODITIES[key]
    # Prefer ensemble metadata if it exists
    ensemble_path = cfg.models_dir / "ensemble_metadata.json"
    if ensemble_path.exists():
        with open(ensemble_path) as f:
            meta = json.load(f)
        meta["_source"] = "ensemble"
        return meta
    if not cfg.metadata_path.exists():
        return None
    with open(cfg.metadata_path) as f:
        meta = json.load(f)
    meta["_source"] = "single"
    return meta


def format_pct(val, decimals=1):
    if val is None:
        return "   —   "
    return f"{val:>6.{decimals}%}"


def format_folds(accs):
    if not accs:
        return "—"
    return " ".join(f"{a:.0%}" for a in accs)


def main():
    print("=" * 90)
    print("MODEL ACCURACY REPORT — ALL COMMODITIES")
    print("=" * 90)

    # Header
    print(f"\n{'Commodity':<12} {'Horizon':>7} {'Feats':>5}  "
          f"{'Reg Dir':>7} {'Reg Std':>7} {'Clf Acc':>7} {'Clf Std':>7}  "
          f"{'HO Reg':>6} {'HO Clf':>6}  {'Status'}")
    print("-" * 90)

    results = []
    for key in COMMODITIES:
        meta = load_metadata(key)
        cfg = COMMODITIES[key]

        if not meta:
            print(f"  {cfg.name:<12} — NO METADATA")
            continue

        horizon = meta.get("horizon", "?")
        n_feats = meta.get("n_features", len(meta.get("features", [])))
        source = meta.get("_source", "single")
        n_models = meta.get("n_models", 1)
        best_model = meta.get("best_model", "")
        best_accuracy = meta.get("best_accuracy")

        reg = meta.get("regression", {})
        clf = meta.get("classification", {})
        holdout = meta.get("holdout", {})

        reg_acc = reg.get("avg_accuracy")
        reg_std = reg.get("std_accuracy")
        clf_acc = clf.get("avg_accuracy")
        clf_std = clf.get("std_accuracy")

        ho_reg = holdout.get("reg_direction_accuracy")
        ho_clf = holdout.get("clf_accuracy")

        # Status assessment
        issues = []
        if reg_acc is not None and reg_acc < 0.55:
            issues.append("LOW_REG")
        if reg_std is not None and reg_std > 0.15:
            issues.append("HIGH_VAR")
        if ho_reg is not None and ho_reg < 0.50:
            issues.append("POOR_HO")
        if reg.get("fold_accuracies"):
            if any(a >= 0.99 for a in reg["fold_accuracies"]):
                issues.append("OVERFIT?")
            if any(a <= 0.05 for a in reg["fold_accuracies"]):
                issues.append("COLLAPSE")

        status = ", ".join(issues) if issues else "OK"

        print(f"  {cfg.name:<12} {horizon:>5}d {n_feats:>5}  "
              f"{format_pct(reg_acc)} {format_pct(reg_std)} "
              f"{format_pct(clf_acc)} {format_pct(clf_std)}  "
              f"{format_pct(ho_reg)} {format_pct(ho_clf)}  "
              f"{status}")

        results.append({
            "commodity": cfg.name,
            "key": key,
            "horizon": horizon,
            "n_features": n_feats,
            "reg_accuracy": reg_acc,
            "reg_std": reg_std,
            "clf_accuracy": clf_acc,
            "clf_std": clf_std,
            "holdout_reg": ho_reg,
            "holdout_clf": ho_clf,
            "status": status,
            "reg_folds": reg.get("fold_accuracies", []),
            "clf_folds": clf.get("fold_accuracies", []),
        })

    # Detailed fold view
    print(f"\n{'='*90}")
    print("FOLD-BY-FOLD BREAKDOWN (Direction Accuracy)")
    print(f"{'='*90}")

    for r in results:
        print(f"\n  {r['commodity']}:")
        if r["reg_folds"]:
            print(f"    Regression:     {format_folds(r['reg_folds'])}")
        if r["clf_folds"]:
            print(f"    Classification: {format_folds(r['clf_folds'])}")

    # Warnings
    print(f"\n{'='*90}")
    print("WARNINGS & RECOMMENDATIONS")
    print(f"{'='*90}")

    for r in results:
        if r["status"] != "OK":
            print(f"\n  {r['commodity']} [{r['status']}]:")
            if "LOW_REG" in r["status"]:
                print(f"    - Regression direction acc ({r['reg_accuracy']:.1%}) near random (50%)")
                print(f"      Consider: shorter horizon, different features, or ensemble")
            if "HIGH_VAR" in r["status"]:
                print(f"    - High fold variance (std={r['reg_std']:.1%})")
                print(f"      Consider: more regularization, fewer features, or regime-aware training")
            if "POOR_HO" in r["status"]:
                print(f"    - Holdout accuracy ({r['holdout_reg']:.1%}) below coin flip")
                print(f"      Model may not generalize to new data")
            if "OVERFIT?" in r["status"]:
                print(f"    - Perfect fold detected (100%) — possible overfitting")
            if "COLLAPSE" in r["status"]:
                print(f"    - Near-zero fold detected — model completely failed in one regime")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n  Summary: {ok_count}/{len(results)} models rated OK")

    # Rankings
    print(f"\n{'='*90}")
    print("RANKINGS (by regression direction accuracy)")
    print(f"{'='*90}")
    ranked = sorted([r for r in results if r["reg_accuracy"] is not None],
                    key=lambda x: x["reg_accuracy"], reverse=True)
    for i, r in enumerate(ranked, 1):
        ho_str = f"holdout={r['holdout_reg']:.0%}" if r["holdout_reg"] else "no holdout"
        print(f"  {i}. {r['commodity']:<12} {r['reg_accuracy']:.1%}  ({ho_str})")


if __name__ == "__main__":
    main()
