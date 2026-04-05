"""Research Agent — discovers ways to improve models through systematic analysis.

Responsibilities:
- Analyze feature importance stability across folds and time
- Test new data sources and feature ideas
- Run horizon sensitivity analysis
- Benchmark against baselines (buy-hold, persistence, random)
- Identify regime-dependent performance
- Suggest concrete improvements ranked by expected impact

Usage:
    python -m agents research                        # full research suite
    python -m agents research --feature-audit        # feature stability only
    python -m agents research --baselines            # baseline comparison
    python -m agents research --regime               # regime analysis
    python -m agents research coffee sugar            # specific commodities
"""

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .log import setup_logging


logger = setup_logging("research")

RESEARCH_DIR = COMMODITIES_DIR / "research"


# ── Feature stability analysis ─────────────────────────────────────────

def analyze_feature_stability(cfg: CommodityConfig) -> dict:
    """Check if selected features are stable across walk-forward folds.

    Unstable features (important in some folds, useless in others) are
    a sign of overfitting. Returns stability scores per feature.
    """
    script = f"""
import json, sys, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
sys.path.insert(0, '.')
from features import prepare_dataset

df, all_cols = prepare_dataset(horizon=63)

# Load metadata to get selected features
with open('models/production_metadata.json') as f:
    meta = json.load(f)
feature_cols = [f for f in meta['features'] if f in all_cols]
X = df[feature_cols].values
y = df['target_direction'].values
n = len(df)

# Walk-forward folds
fold_importances = []
for i in range(5):
    test_end = n - i * 63
    test_start = test_end - 63
    train_end = test_start - 63
    if train_end < 504:
        break
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, test_end)

    val_size = min(63, len(train_idx) // 5)
    fit_idx = train_idx[:-val_size]
    val_idx = train_idx[-val_size:]

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.6,
                          min_child_weight=10, gamma=1.0,
                          eval_metric='logloss', early_stopping_rounds=30, random_state=42)
    model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)

    result = permutation_importance(model, X[test_idx], y[test_idx],
                                     n_repeats=10, random_state=42, scoring='accuracy')
    fold_importances.append(result.importances_mean.tolist())

# Stability: features that are consistently important across folds
fold_importances = np.array(fold_importances)
mean_imp = fold_importances.mean(axis=0)
std_imp = fold_importances.std(axis=0)
cv_imp = np.where(mean_imp > 0.001, std_imp / mean_imp, 99.0)  # coefficient of variation

stability = []
for j, feat in enumerate(feature_cols):
    stability.append({{
        'feature': feat,
        'mean_importance': round(float(mean_imp[j]), 6),
        'std_importance': round(float(std_imp[j]), 6),
        'cv': round(float(cv_imp[j]), 3),
        'stable': float(cv_imp[j]) < 1.0 and float(mean_imp[j]) > 0.001,
        'positive_folds': int(np.sum(fold_importances[:, j] > 0)),
    }})

stability.sort(key=lambda x: x['mean_importance'], reverse=True)
print(json.dumps(stability))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir), timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"{cfg.name} feature stability failed: {result.stderr[-300:]}")
            return {"status": "error", "stderr": result.stderr[-300:]}

        stability = json.loads(result.stdout.strip().split("\n")[-1])
        stable_count = sum(1 for s in stability if s["stable"])
        unstable = [s["feature"] for s in stability if not s["stable"]]

        return {
            "status": "ok",
            "total_features": len(stability),
            "stable_features": stable_count,
            "unstable_features": unstable,
            "details": stability,
        }
    except Exception as e:
        return {"status": "exception", "error": str(e)}


# ── Baseline comparison ────────────────────────────────────────────────

def compare_baselines(cfg: CommodityConfig) -> dict:
    """Compare model against simple baselines to quantify alpha.

    Baselines:
    - Always UP (majority class)
    - Persistence (yesterday's direction continues)
    - Random (50/50)
    - Buy and hold
    """
    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
sys.path.insert(0, '.')
from features import prepare_dataset

df, all_cols = prepare_dataset(horizon=63)

with open('models/production_metadata.json') as f:
    meta = json.load(f)
feature_cols = [f for f in meta['features'] if f in all_cols]

reg = joblib.load('models/production_regressor.joblib')
clf = joblib.load('models/production_classifier.joblib')

X = df[feature_cols].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values
price = df['{cfg.price_col}'].values

# Use last 252 trading days as evaluation window
eval_size = min(252, len(df) // 3)
eval_idx = slice(-eval_size, None)

y_eval = y_dir[eval_idx]
y_ret_eval = y_ret[eval_idx]

# Model predictions
model_dir_preds = clf.predict(X[eval_idx])
model_ret_preds = reg.predict(X[eval_idx])

# Baselines
always_up = np.ones_like(y_eval)
persistence = np.roll(y_eval, 1); persistence[0] = 1
random_preds = np.random.RandomState(42).randint(0, 2, size=len(y_eval))

# Direction accuracy
model_acc = float(np.mean(model_dir_preds == y_eval))
always_up_acc = float(np.mean(always_up == y_eval))
persistence_acc = float(np.mean(persistence == y_eval))
random_acc = float(np.mean(random_preds == y_eval))

# Model return prediction quality
model_dir_from_reg = (model_ret_preds > 0).astype(int)
reg_dir_acc = float(np.mean(model_dir_from_reg == y_eval))

# Buy and hold return
bh_return = float((price[-1] / price[-eval_size]) - 1)
# Model-signal return (long when UP, flat otherwise)
model_returns = []
for i in range(len(y_ret_eval)):
    if model_dir_preds[i] == 1:
        model_returns.append(y_ret_eval[i] / 63)  # daily-ish return
    else:
        model_returns.append(0)

result = {{
    'eval_days': eval_size,
    'class_balance': float(np.mean(y_eval)),
    'model_clf_accuracy': round(model_acc, 4),
    'model_reg_dir_accuracy': round(reg_dir_acc, 4),
    'baseline_always_up': round(always_up_acc, 4),
    'baseline_persistence': round(persistence_acc, 4),
    'baseline_random': round(random_acc, 4),
    'alpha_vs_majority': round(model_acc - always_up_acc, 4),
    'alpha_vs_persistence': round(model_acc - persistence_acc, 4),
    'buy_hold_return': round(bh_return, 4),
}}
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(cfg.project_dir), timeout=120,
        )
        if result.returncode != 0:
            return {"status": "error", "stderr": result.stderr[-300:]}
        return json.loads(result.stdout.strip().split("\n")[-1])
    except Exception as e:
        return {"status": "exception", "error": str(e)}


# ── Regime analysis ────────────────────────────────────────────────────

def analyze_regime_performance(cfg: CommodityConfig) -> dict:
    """Analyze model accuracy across different market regimes.

    Identifies whether the model performs differently in:
    - Bull vs bear vs sideways markets
    - High vs low volatility environments
    - Different seasonal periods
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data"}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    price = df[cfg.price_col]

    if len(price) < 504:
        return {"status": "insufficient_data"}

    # Define regimes
    ret_252d = price.pct_change(252)
    vol_63d = price.pct_change().rolling(63).std() * np.sqrt(252)
    vol_median = vol_63d.median()

    regimes = pd.Series("sideways", index=df.index)
    regimes[ret_252d > 0.15] = "bull"
    regimes[ret_252d < -0.15] = "bear"

    vol_regime = pd.Series("normal", index=df.index)
    vol_regime[vol_63d > vol_median * 1.5] = "high_vol"
    vol_regime[vol_63d < vol_median * 0.67] = "low_vol"

    # Seasonal
    month = df.index.month
    seasons = pd.Series("other", index=df.index)
    seasons[month.isin([12, 1, 2])] = "winter"
    seasons[month.isin([6, 7, 8])] = "summer"

    result = {
        "trend_regime_distribution": regimes.value_counts().to_dict(),
        "vol_regime_distribution": vol_regime.value_counts().to_dict(),
        "season_distribution": seasons.value_counts().to_dict(),
        "current_trend_regime": regimes.iloc[-1] if not regimes.empty else None,
        "current_vol_regime": vol_regime.iloc[-1] if not vol_regime.empty else None,
    }

    return result


# ── Improvement suggestions ────────────────────────────────────────────

def generate_improvement_suggestions(
    stability: dict, baselines: dict, regime: dict, cfg: CommodityConfig
) -> list[dict]:
    """Generate ranked improvement suggestions based on analysis results."""
    suggestions = []

    # Check if model beats baselines
    if baselines.get("status") != "error":
        alpha = baselines.get("alpha_vs_majority", 0)
        if alpha < 0.02:
            suggestions.append({
                "priority": "critical",
                "area": "model_performance",
                "suggestion": f"Model barely beats majority-class baseline (alpha={alpha:+.1%}). "
                              f"Consider: more/different features, different horizon, or ensemble.",
                "expected_impact": "high",
            })

    # Check feature stability
    if stability.get("status") == "ok":
        unstable = stability.get("unstable_features", [])
        if len(unstable) > len(stability.get("details", [])) * 0.3:
            suggestions.append({
                "priority": "high",
                "area": "feature_engineering",
                "suggestion": f"{len(unstable)} of {stability['total_features']} features are unstable "
                              f"across folds. Consider dropping: {', '.join(unstable[:5])}...",
                "expected_impact": "medium",
            })

    # Check regime exposure
    if regime.get("current_trend_regime") == "bear":
        suggestions.append({
            "priority": "medium",
            "area": "regime_adaptation",
            "suggestion": "Market is in bear regime. If model was trained primarily on bull data, "
                          "accuracy may degrade. Consider regime-aware features or separate models.",
            "expected_impact": "medium",
        })

    # Standard improvement ideas (always relevant)
    suggestions.extend([
        {
            "priority": "medium",
            "area": "data_sources",
            "suggestion": "Add fundamental data: USDA WASDE reports, inventory levels, "
                          "futures term structure (contango/backwardation).",
            "expected_impact": "medium-high",
        },
        {
            "priority": "medium",
            "area": "data_sources",
            "suggestion": "Add origin-country FX rates (BRL for coffee/sugar, "
                          "VND for coffee, CLP for copper) as features.",
            "expected_impact": "medium",
        },
        {
            "priority": "low",
            "area": "feature_engineering",
            "suggestion": "Add cross-commodity interaction features (e.g., copper-oil ratio, "
                          "sugar-ethanol spread, soybean-wheat ratio).",
            "expected_impact": "low-medium",
        },
        {
            "priority": "low",
            "area": "model_architecture",
            "suggestion": "Test ensemble of XGBoost + LightGBM + CatBoost. "
                          "Diversity in tree methods often improves robustness.",
            "expected_impact": "low-medium",
        },
    ])

    return suggestions


# ── Main entry point ───────────────────────────────────────────────────

def run_research(
    commodity_keys: list[str] = None,
    feature_audit: bool = True,
    baselines: bool = True,
    regime: bool = True,
) -> dict:
    """Run the full research suite."""
    targets = commodity_keys or list(COMMODITIES.keys())
    report = {"timestamp": datetime.now().isoformat(), "commodities": {}}

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Researching {cfg.name}")
        logger.info(f"{'='*50}")

        commodity_report = {}

        if feature_audit:
            logger.info("  Running feature stability analysis...")
            commodity_report["feature_stability"] = analyze_feature_stability(cfg)

        if baselines:
            logger.info("  Running baseline comparison...")
            commodity_report["baselines"] = compare_baselines(cfg)

        if regime:
            logger.info("  Running regime analysis...")
            commodity_report["regime"] = analyze_regime_performance(cfg)

        # Generate suggestions
        commodity_report["suggestions"] = generate_improvement_suggestions(
            commodity_report.get("feature_stability", {}),
            commodity_report.get("baselines", {}),
            commodity_report.get("regime", {}),
            cfg,
        )

        report["commodities"][key] = commodity_report

    # Save report
    RESEARCH_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESEARCH_DIR / f"research_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Research report saved to {report_path}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Research agent — find model improvements")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--feature-audit", action="store_true", help="Feature stability only")
    parser.add_argument("--baselines", action="store_true", help="Baseline comparison only")
    parser.add_argument("--regime", action="store_true", help="Regime analysis only")
    args = parser.parse_args()

    # If no specific analysis requested, run all
    run_all = not (args.feature_audit or args.baselines or args.regime)

    report = run_research(
        commodity_keys=args.commodities or None,
        feature_audit=args.feature_audit or run_all,
        baselines=args.baselines or run_all,
        regime=args.regime or run_all,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("RESEARCH FINDINGS")
    print(f"{'='*60}")

    for key, data in report["commodities"].items():
        cfg = COMMODITIES[key]
        print(f"\n--- {cfg.name} ---")

        # Baselines
        bl = data.get("baselines", {})
        if bl and bl.get("status") != "error":
            print(f"  Model accuracy:     {bl.get('model_clf_accuracy', '?'):.1%}")
            print(f"  Always-UP baseline: {bl.get('baseline_always_up', '?'):.1%}")
            print(f"  Alpha vs majority:  {bl.get('alpha_vs_majority', 0):+.1%}")

        # Feature stability
        fs = data.get("feature_stability", {})
        if fs.get("status") == "ok":
            print(f"  Stable features:    {fs['stable_features']}/{fs['total_features']}")
            if fs["unstable_features"]:
                print(f"  Unstable:           {', '.join(fs['unstable_features'][:5])}")

        # Regime
        rg = data.get("regime", {})
        if rg.get("current_trend_regime"):
            print(f"  Current regime:     {rg['current_trend_regime']} / {rg.get('current_vol_regime', '?')}")

    # Suggestions
    all_suggestions = []
    for key, data in report["commodities"].items():
        for s in data.get("suggestions", []):
            s["commodity"] = COMMODITIES[key].name
            all_suggestions.append(s)

    # Deduplicate generic suggestions
    seen = set()
    unique = []
    for s in all_suggestions:
        sig = s["suggestion"][:60]
        if sig not in seen:
            seen.add(sig)
            unique.append(s)

    if unique:
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT SUGGESTIONS ({len(unique)})")
        print(f"{'='*60}")
        for s in sorted(unique, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["priority"]]):
            print(f"\n  [{s['priority'].upper()}] {s['area']}")
            print(f"  {s['suggestion']}")
            print(f"  Expected impact: {s['expected_impact']}")


if __name__ == "__main__":
    main()
