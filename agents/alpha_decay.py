"""Alpha Decay Agent — tracks signal degradation and edge sustainability.

Monitors whether our predictions maintain their edge over time,
detects crowding, and flags commodities where alpha has fully decayed.

Usage:
    python -m agents alpha                     # full alpha analysis
    python -m agents alpha coffee              # specific commodity
    python -m agents alpha --crowding          # crowding detection only
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, LOGS_DIR
from .design_log import log_observation, log_challenge
from .log import setup_logging, log_event

# DB access (optional — gracefully degrade if unavailable)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db import get_db
    db = get_db()
except Exception:
    db = None


logger = setup_logging("alpha_decay")


# ── Alpha measurement ─────────────────────────────────────────────────


def measure_alpha_decay(cfg: CommodityConfig, windows: list[int] = None) -> dict:
    """Measure prediction accuracy at different time windows after prediction.

    For predictions with realized outcomes, computes directional accuracy
    at 5, 10, 21, 42, 63 day windows. Immediate accuracy (short windows)
    indicates good alpha; accuracy only at long windows suggests the model
    is lagging the market.

    Returns dict with accuracy_by_window and sample counts.
    """
    if windows is None:
        windows = [5, 10, 21, 42, 63]

    # Load predictions from DB or JSONL fallback
    predictions = _load_predictions(cfg)
    if not predictions:
        return {"status": "no_predictions", "commodity": cfg.name}

    # Load price data
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return {"status": "missing_price_col", "commodity": cfg.name}

    price = df[cfg.price_col].dropna()

    accuracy_by_window = {}
    for window in windows:
        correct = 0
        total = 0
        for pred in predictions:
            pred_date = pd.Timestamp(pred["date"])
            target_date = pred_date + pd.Timedelta(days=window)

            if target_date > price.index[-1]:
                continue
            # Find nearest price on or after target date
            mask = price.index >= target_date
            if not mask.any():
                continue

            realized_price = float(price.loc[mask].iloc[0])
            entry_price = pred["price"]
            realized_dir = "UP" if realized_price >= entry_price else "DOWN"
            predicted_dir = pred["direction"]

            if realized_dir == predicted_dir:
                correct += 1
            total += 1

        if total > 0:
            accuracy_by_window[window] = {
                "accuracy": round(correct / total, 4),
                "n_predictions": total,
            }

    # Determine if alpha is front-loaded (good) or back-loaded (lagging)
    if accuracy_by_window:
        accs = [(w, v["accuracy"]) for w, v in sorted(accuracy_by_window.items())]
        short_term = np.mean([a for w, a in accs if w <= 21])
        long_term = np.mean([a for w, a in accs if w > 21]) if any(w > 21 for w, _ in accs) else None

        if long_term is not None:
            if short_term > long_term + 0.05:
                alpha_profile = "front_loaded"  # good — edge is immediate
            elif long_term > short_term + 0.05:
                alpha_profile = "back_loaded"   # model may be lagging
            else:
                alpha_profile = "flat"
        else:
            alpha_profile = "unknown"
    else:
        alpha_profile = "unknown"

    return {
        "commodity": cfg.name,
        "accuracy_by_window": accuracy_by_window,
        "alpha_profile": alpha_profile,
    }


def detect_feature_drift(cfg: CommodityConfig) -> dict:
    """Compare current feature distributions to training-time distributions.

    For each feature, computes mean/std from the last 63 days and compares
    to the overall training period. Features that have drifted > 2 std
    from their training mean are flagged.

    Returns list of drifted features with drift magnitude.
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Identify numeric feature columns (exclude price column itself)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg.price_col in numeric_cols:
        numeric_cols.remove(cfg.price_col)

    if len(df) < 126:  # need at least 63 days recent + 63 days history
        return {"status": "insufficient_data", "commodity": cfg.name}

    # Training period = everything except the last 63 days
    train_df = df.iloc[:-63][numeric_cols]
    recent_df = df.iloc[-63:][numeric_cols]

    drifted_features = []
    for col in numeric_cols:
        train_vals = train_df[col].dropna()
        recent_vals = recent_df[col].dropna()

        if len(train_vals) < 30 or len(recent_vals) < 10:
            continue

        train_mean = train_vals.mean()
        train_std = train_vals.std()

        if train_std == 0:
            continue

        recent_mean = recent_vals.mean()
        drift_magnitude = abs(recent_mean - train_mean) / train_std

        if drift_magnitude > 2.0:
            drifted_features.append({
                "feature": col,
                "drift_magnitude": round(float(drift_magnitude), 2),
                "train_mean": round(float(train_mean), 4),
                "train_std": round(float(train_std), 4),
                "recent_mean": round(float(recent_mean), 4),
                "direction": "higher" if recent_mean > train_mean else "lower",
            })

    # Sort by drift magnitude descending
    drifted_features.sort(key=lambda x: x["drift_magnitude"], reverse=True)

    return {
        "commodity": cfg.name,
        "n_features_checked": len(numeric_cols),
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_ratio": round(len(drifted_features) / max(len(numeric_cols), 1), 3),
    }


def detect_crowding(cfg: CommodityConfig) -> dict:
    """Proxy for detecting if other traders use similar signals.

    Checks three crowding indicators:
    1. Volume spikes when our model generates signals (others trading same thing)
    2. Price moves happen BEFORE our signal (front-running)
    3. Signal accuracy declining over time (edge competed away)

    Returns crowding_score (0-1) and evidence list.
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    price = df[cfg.price_col].dropna() if cfg.price_col in df.columns else pd.Series()

    predictions = _load_predictions(cfg)
    evidence = []
    scores = []

    # 1. Volume spikes around signal dates
    vol_col = None
    for candidate in [f"{cfg.dir_name}_volume", "volume", "Volume"]:
        if candidate in df.columns:
            vol_col = candidate
            break

    if vol_col and predictions:
        volume = df[vol_col].dropna()
        vol_median = volume.rolling(63).median()

        signal_vol_ratios = []
        for pred in predictions:
            pred_date = pd.Timestamp(pred["date"])
            if pred_date in volume.index:
                idx = volume.index.get_loc(pred_date)
                if idx > 63:
                    ratio = volume.iloc[idx] / vol_median.iloc[idx] if vol_median.iloc[idx] > 0 else 1.0
                    signal_vol_ratios.append(float(ratio))

        if signal_vol_ratios:
            avg_ratio = np.mean(signal_vol_ratios)
            if avg_ratio > 1.5:
                evidence.append(f"Volume {avg_ratio:.1f}x median on signal days (possible crowding)")
                scores.append(min(1.0, (avg_ratio - 1.0) / 2.0))
            else:
                scores.append(0.0)

    # 2. Pre-signal price moves (front-running proxy)
    if not price.empty and predictions:
        pre_moves = []
        for pred in predictions:
            pred_date = pd.Timestamp(pred["date"])
            if pred_date in price.index:
                idx = price.index.get_loc(pred_date)
                if idx >= 5:
                    # 5-day return before signal
                    pre_return = float(price.iloc[idx] / price.iloc[idx - 5] - 1)
                    pred_dir = pred["direction"]
                    # If price already moved in predicted direction before signal
                    if (pred_dir == "UP" and pre_return > 0.02) or \
                       (pred_dir == "DOWN" and pre_return < -0.02):
                        pre_moves.append(True)
                    else:
                        pre_moves.append(False)

        if pre_moves:
            front_run_pct = np.mean(pre_moves)
            if front_run_pct > 0.5:
                evidence.append(f"{front_run_pct:.0%} of signals had price pre-move (possible front-running)")
                scores.append(min(1.0, front_run_pct))
            else:
                scores.append(0.0)

    # 3. Declining accuracy over time
    rolling = compute_rolling_accuracy(cfg)
    if rolling.get("trend_direction") == "degrading":
        evidence.append("Rolling accuracy is degrading over time")
        scores.append(0.7)
    elif rolling.get("trend_direction") == "stable":
        scores.append(0.0)

    crowding_score = round(float(np.mean(scores)), 3) if scores else 0.0

    return {
        "commodity": cfg.name,
        "crowding_score": crowding_score,
        "evidence": evidence,
        "n_indicators_checked": len(scores),
    }


def compute_rolling_accuracy(cfg: CommodityConfig, window: int = 20) -> dict:
    """Compute rolling prediction accuracy over time.

    Uses a rolling window of N predictions to show accuracy trend.
    Returns time series of accuracy and trend direction.
    """
    predictions = _load_predictions(cfg)
    if not predictions:
        return {"status": "no_predictions", "commodity": cfg.name}

    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return {"status": "missing_price_col", "commodity": cfg.name}

    price = df[cfg.price_col].dropna()

    # Evaluate each prediction
    evaluated = []
    for pred in predictions:
        pred_date = pd.Timestamp(pred["date"])
        horizon = pred.get("horizon", 63)
        target_date = pred_date + pd.Timedelta(days=horizon)

        if target_date > price.index[-1]:
            continue

        mask = price.index >= target_date
        if not mask.any():
            continue

        realized_price = float(price.loc[mask].iloc[0])
        entry_price = pred["price"]
        realized_dir = "UP" if realized_price > entry_price else "DOWN"
        correct = 1 if realized_dir == pred["direction"] else 0

        evaluated.append({"date": str(pred_date.date()), "correct": correct})

    if len(evaluated) < window:
        return {
            "status": "insufficient_predictions",
            "commodity": cfg.name,
            "n_evaluated": len(evaluated),
        }

    # Compute rolling accuracy
    correct_series = pd.Series([e["correct"] for e in evaluated])
    dates = [e["date"] for e in evaluated]
    rolling_acc = correct_series.rolling(window, min_periods=window).mean().dropna()

    if len(rolling_acc) < 2:
        trend_direction = "unknown"
    else:
        # Linear trend: compare first half to second half
        mid = len(rolling_acc) // 2
        first_half = float(rolling_acc.iloc[:mid].mean())
        second_half = float(rolling_acc.iloc[mid:].mean())
        diff = second_half - first_half

        if diff > 0.05:
            trend_direction = "improving"
        elif diff < -0.05:
            trend_direction = "degrading"
        else:
            trend_direction = "stable"

    return {
        "commodity": cfg.name,
        "dates": dates[window - 1:],
        "accuracies": [round(float(a), 4) for a in rolling_acc.values],
        "trend_direction": trend_direction,
        "latest_accuracy": round(float(rolling_acc.iloc[-1]), 4) if len(rolling_acc) > 0 else None,
        "n_predictions": len(evaluated),
        "window": window,
    }


def assess_commodity_viability(cfg: CommodityConfig) -> dict:
    """Overall viability assessment combining alpha decay, drift, and crowding.

    Returns viability score (0-100) and recommendation:
    - 'active': keep trading
    - 'watch': reduce position sizing
    - 'suspend': stop trading until retrained
    - 'drop': remove from system
    """
    alpha = measure_alpha_decay(cfg)
    drift = detect_feature_drift(cfg)
    crowding = detect_crowding(cfg)
    rolling = compute_rolling_accuracy(cfg)

    # Score components (each 0-100)
    scores = {}

    # Alpha quality (based on best window accuracy)
    if alpha.get("accuracy_by_window"):
        best_acc = max(v["accuracy"] for v in alpha["accuracy_by_window"].values())
        # 50% accuracy = 0 points, 80%+ = 100 points
        scores["alpha"] = min(100, max(0, (best_acc - 0.5) / 0.3 * 100))
    else:
        scores["alpha"] = 50  # neutral if we can't measure

    # Feature drift penalty
    if drift.get("drift_ratio") is not None:
        # 0 drift = 100, 50%+ features drifted = 0
        scores["drift"] = max(0, 100 - drift["drift_ratio"] * 200)
    else:
        scores["drift"] = 50

    # Crowding penalty
    crowding_score = crowding.get("crowding_score", 0)
    scores["crowding"] = max(0, 100 - crowding_score * 100)

    # Trend penalty
    if rolling.get("trend_direction") == "degrading":
        scores["trend"] = 25
    elif rolling.get("trend_direction") == "improving":
        scores["trend"] = 90
    elif rolling.get("trend_direction") == "stable":
        scores["trend"] = 70
    else:
        scores["trend"] = 50

    # Weighted viability score
    weights = {"alpha": 0.35, "drift": 0.20, "crowding": 0.20, "trend": 0.25}
    viability = round(sum(scores[k] * weights[k] for k in weights), 1)

    # Recommendation
    if viability >= 70:
        recommendation = "active"
    elif viability >= 50:
        recommendation = "watch"
    elif viability >= 30:
        recommendation = "suspend"
    else:
        recommendation = "drop"

    return {
        "commodity": cfg.name,
        "viability_score": viability,
        "recommendation": recommendation,
        "component_scores": scores,
        "alpha_profile": alpha.get("alpha_profile"),
        "n_drifted_features": drift.get("n_drifted"),
        "crowding_score": crowding.get("crowding_score"),
        "accuracy_trend": rolling.get("trend_direction"),
    }


# ── Reporting ─────────────────────────────────────────────────────────


def generate_alpha_report(commodity_keys: list[str] = None) -> dict:
    """Full alpha decay report across specified (or all) commodities.

    Ranks commodities by viability score and flags any requiring action.
    """
    keys = commodity_keys or list(COMMODITIES.keys())
    report = {
        "timestamp": datetime.now().isoformat(),
        "commodities": {},
    }

    # Log agent run to DB
    run_id = None
    if db:
        try:
            run_id = db.start_agent_run("alpha_decay", keys)
        except Exception as e:
            logger.warning(f"Failed to log agent run start: {e}")

    for key in keys:
        if key not in COMMODITIES:
            logger.warning(f"Unknown commodity: {key}")
            continue
        cfg = COMMODITIES[key]
        logger.info(f"Analyzing alpha decay for {cfg.name}")

        report["commodities"][key] = {
            "viability": assess_commodity_viability(cfg),
            "alpha_decay": measure_alpha_decay(cfg),
            "feature_drift": detect_feature_drift(cfg),
            "crowding": detect_crowding(cfg),
            "rolling_accuracy": compute_rolling_accuracy(cfg),
        }

    # Rank by viability
    ranked = sorted(
        [(k, v["viability"]["viability_score"])
         for k, v in report["commodities"].items()
         if isinstance(v.get("viability", {}).get("viability_score"), (int, float))],
        key=lambda x: x[1],
        reverse=True,
    )
    report["ranking"] = [{"commodity": k, "viability": s} for k, s in ranked]

    # Alerts
    report["alerts"] = []
    for k, data in report["commodities"].items():
        via = data.get("viability", {})
        if via.get("recommendation") in ("suspend", "drop"):
            report["alerts"].append({
                "commodity": k,
                "recommendation": via["recommendation"],
                "viability_score": via.get("viability_score"),
            })

    # Save report
    report_dir = LOGS_DIR / "alpha"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"alpha_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    # Finish agent run in DB
    if db and run_id:
        try:
            status = "ok" if not report["alerts"] else "warning"
            summary = f"Analyzed {len(report['commodities'])} commodities, {len(report['alerts'])} alerts"
            db.finish_agent_run(run_id, status=status, summary=summary, report=report)
        except Exception as e:
            logger.warning(f"Failed to log agent run finish: {e}")

    return report


# ── Helpers ───────────────────────────────────────────────────────────


def _load_predictions(cfg: CommodityConfig) -> list[dict]:
    """Load predictions for a commodity from DB or JSONL fallback."""
    # Try DB first
    if db:
        try:
            rows = db.execute(
                "SELECT * FROM predictions WHERE commodity = ? ORDER BY as_of_date",
                (cfg.name,),
            )
            return [
                {
                    "date": r["as_of_date"],
                    "price": r["price"],
                    "direction": r["direction"],
                    "confidence": r["confidence"],
                    "horizon": r.get("horizon_days", 63),
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug(f"DB prediction load failed for {cfg.name}: {e}")

    # Fallback to JSONL
    predictions_log = LOGS_DIR / "predictions.jsonl"
    if not predictions_log.exists():
        return []

    entries = []
    with open(predictions_log) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if (entry.get("commodity", "").lower().replace(" ", "") == cfg.name.lower().replace(" ", "")
                        or entry.get("commodity") == cfg.name):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries


# ── CLI entry point ───────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Alpha Decay Agent")
    parser.add_argument("commodities", nargs="*", help="Specific commodities to analyze")
    parser.add_argument("--crowding", action="store_true", help="Crowding detection only")
    args = parser.parse_args()

    keys = args.commodities if args.commodities else None

    if args.crowding:
        check_keys = keys or list(COMMODITIES.keys())
        print(f"\n{'='*60}")
        print("CROWDING DETECTION")
        print(f"{'='*60}")
        for key in check_keys:
            if key not in COMMODITIES:
                continue
            cfg = COMMODITIES[key]
            result = detect_crowding(cfg)
            score = result.get("crowding_score", "N/A")
            print(f"\n  {cfg.name}: crowding_score={score}")
            for ev in result.get("evidence", []):
                print(f"    - {ev}")
            if not result.get("evidence"):
                print(f"    No crowding evidence detected")
        return

    # Full report
    report = generate_alpha_report(keys)

    print(f"\n{'='*60}")
    print("ALPHA DECAY REPORT")
    print(f"{'='*60}")

    # Ranking
    print("\n  VIABILITY RANKING:")
    for entry in report.get("ranking", []):
        key = entry["commodity"]
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue
        via = report["commodities"][key]["viability"]
        rec = via.get("recommendation", "?")
        score = via.get("viability_score", 0)
        label = {"active": "ACTIVE", "watch": "WATCH", "suspend": "SUSPEND", "drop": "DROP"}.get(rec, rec)
        print(f"    {cfg.name:<15} {score:5.1f}/100  [{label}]")

    # Detail per commodity
    for key, data in report["commodities"].items():
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue

        via = data["viability"]
        drift = data["feature_drift"]
        crowd = data["crowding"]
        rolling = data["rolling_accuracy"]

        print(f"\n  {cfg.name}:")
        print(f"    Viability: {via.get('viability_score', 'N/A')}/100 "
              f"[{via.get('recommendation', '?').upper()}]")
        print(f"    Alpha profile: {via.get('alpha_profile', 'N/A')}")

        # Component scores
        components = via.get("component_scores", {})
        if components:
            parts = "  ".join(f"{k}={v:.0f}" for k, v in components.items())
            print(f"    Components: {parts}")

        n_drift = drift.get("n_drifted", 0)
        n_checked = drift.get("n_features_checked", 0)
        print(f"    Feature drift: {n_drift}/{n_checked} features drifted")
        if drift.get("drifted_features"):
            for feat in drift["drifted_features"][:3]:
                print(f"      - {feat['feature']}: {feat['drift_magnitude']:.1f} std {feat['direction']}")

        acc_trend = rolling.get("trend_direction", "N/A")
        latest = rolling.get("latest_accuracy")
        print(f"    Accuracy trend: {acc_trend}" +
              (f" (latest: {latest:.1%})" if latest is not None else ""))

    # Alerts
    if report.get("alerts"):
        print(f"\n  ALERTS:")
        for alert in report["alerts"]:
            cfg = COMMODITIES.get(alert["commodity"])
            name = cfg.name if cfg else alert["commodity"]
            print(f"    {name}: {alert['recommendation'].upper()} "
                  f"(viability={alert['viability_score']})")

    print()


if __name__ == "__main__":
    main()
