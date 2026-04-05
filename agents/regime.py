"""Regime Detection Agent — identifies market regimes and adapts model confidence.

Uses statistical methods to classify current market state and adjust
predictions when the market enters unfamiliar territory.

Usage:
    python -m agents regime                    # full regime analysis
    python -m agents regime coffee sugar       # specific commodities
    python -m agents regime --alert            # only alert on regime changes
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, LOGS_DIR
from .log import setup_logging, log_event


logger = setup_logging("regime")


# ── Regime classification ─────────────────────────────────────────────


def classify_regime(cfg: CommodityConfig) -> dict:
    """Classify the current market regime for a commodity.

    Computes three independent regime dimensions:
    - Trend: based on 252-day return (bull / bear / sideways)
    - Volatility: 63-day annualized vol vs historical median (high / low / normal)
    - Momentum: 21-day return rank over 252-day window (strong / weak / neutral)

    Returns dict with classifications and raw values.
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return {"status": "missing_price_col", "commodity": cfg.name}

    price = df[cfg.price_col].dropna()

    if len(price) < 252:
        return {"status": "insufficient_data", "commodity": cfg.name, "rows": len(price)}

    # ── Trend regime: 252-day return ──
    ret_252 = float(price.iloc[-1] / price.iloc[-252] - 1)
    if ret_252 > 0.15:
        trend = "bull"
    elif ret_252 < -0.15:
        trend = "bear"
    else:
        trend = "sideways"

    # ── Volatility regime: 63-day annualized vol vs historical median ──
    daily_returns = price.pct_change().dropna()
    vol_63 = float(daily_returns.tail(63).std() * np.sqrt(252))

    # Rolling 63-day vol for the full history to get median
    rolling_vol = daily_returns.rolling(63).std() * np.sqrt(252)
    historical_median_vol = float(rolling_vol.dropna().median())

    if historical_median_vol > 0:
        vol_ratio = vol_63 / historical_median_vol
    else:
        vol_ratio = 1.0

    if vol_ratio > 1.5:
        volatility = "high_vol"
    elif vol_ratio < 0.67:
        volatility = "low_vol"
    else:
        volatility = "normal_vol"

    # ── Momentum regime: 21-day return rank over 252-day window ──
    ret_21 = float(price.iloc[-1] / price.iloc[-21] - 1)
    rolling_21d_returns = price.pct_change(21).dropna().tail(252)
    rank_pct = float((rolling_21d_returns <= ret_21).mean())

    if rank_pct >= 0.75:
        momentum = "strong_momentum"
    elif rank_pct <= 0.25:
        momentum = "weak_momentum"
    else:
        momentum = "neutral_momentum"

    return {
        "commodity": cfg.name,
        "date": str(price.index[-1].date()),
        "trend": trend,
        "volatility": volatility,
        "momentum": momentum,
        "raw": {
            "return_252d": round(ret_252, 4),
            "vol_63d_annualized": round(vol_63, 4),
            "vol_historical_median": round(historical_median_vol, 4),
            "vol_ratio": round(vol_ratio, 2),
            "return_21d": round(ret_21, 4),
            "momentum_rank_pct": round(rank_pct, 4),
        },
    }


def detect_regime_change(cfg: CommodityConfig, lookback_days: int = 252) -> dict:
    """Compare current regime to the regime N trading days ago.

    Flags if trend or volatility regime has changed, indicating a
    structural shift that may affect model reliability.
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return {"status": "no_data", "commodity": cfg.name}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return {"status": "missing_price_col", "commodity": cfg.name}

    price = df[cfg.price_col].dropna()

    if len(price) < lookback_days + 252:
        return {"status": "insufficient_data", "commodity": cfg.name}

    # Current regime
    current = classify_regime(cfg)
    if current.get("status"):
        return current

    # Regime at lookback point — build a temporary truncated config view
    truncated_price = price.iloc[:-lookback_days]
    daily_returns = truncated_price.pct_change().dropna()

    if len(truncated_price) < 252:
        return {"status": "insufficient_history", "commodity": cfg.name}

    # Recalculate regime at the earlier point
    ret_252_prev = float(truncated_price.iloc[-1] / truncated_price.iloc[-252] - 1)
    if ret_252_prev > 0.15:
        prev_trend = "bull"
    elif ret_252_prev < -0.15:
        prev_trend = "bear"
    else:
        prev_trend = "sideways"

    vol_63_prev = float(daily_returns.tail(63).std() * np.sqrt(252))
    rolling_vol = daily_returns.rolling(63).std() * np.sqrt(252)
    hist_median_prev = float(rolling_vol.dropna().median())
    vol_ratio_prev = vol_63_prev / hist_median_prev if hist_median_prev > 0 else 1.0

    if vol_ratio_prev > 1.5:
        prev_vol = "high_vol"
    elif vol_ratio_prev < 0.67:
        prev_vol = "low_vol"
    else:
        prev_vol = "normal_vol"

    trend_changed = current["trend"] != prev_trend
    vol_changed = current["volatility"] != prev_vol
    changed = trend_changed or vol_changed

    return {
        "commodity": cfg.name,
        "changed": changed,
        "trend_changed": trend_changed,
        "vol_changed": vol_changed,
        "previous_regime": {"trend": prev_trend, "volatility": prev_vol},
        "current_regime": {"trend": current["trend"], "volatility": current["volatility"]},
        "change_date": str(truncated_price.index[-1].date()) if changed else None,
        "lookback_days": lookback_days,
    }


def compute_regime_history(cfg: CommodityConfig) -> pd.DataFrame:
    """Compute regime classification for every trading day.

    Returns DataFrame with date index and columns: trend, volatility, momentum.
    Shows how long the commodity has been in its current regime.
    """
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if cfg.price_col not in df.columns:
        return pd.DataFrame()

    price = df[cfg.price_col].dropna()

    if len(price) < 252:
        return pd.DataFrame()

    daily_returns = price.pct_change()
    rolling_vol_63 = daily_returns.rolling(63).std() * np.sqrt(252)
    rolling_vol_median = rolling_vol_63.expanding(min_periods=252).median()
    ret_252 = price.pct_change(252)
    ret_21 = price.pct_change(21)
    rolling_21d_rank = ret_21.rolling(252).rank(pct=True)

    records = []
    start_idx = max(252, 63)  # need at least 252 days of history

    for i in range(start_idx, len(price)):
        date = price.index[i]

        # Trend
        r252 = ret_252.iloc[i]
        if pd.isna(r252):
            continue
        if r252 > 0.15:
            trend = "bull"
        elif r252 < -0.15:
            trend = "bear"
        else:
            trend = "sideways"

        # Volatility
        v63 = rolling_vol_63.iloc[i]
        v_med = rolling_vol_median.iloc[i]
        if pd.notna(v63) and pd.notna(v_med) and v_med > 0:
            vr = v63 / v_med
            if vr > 1.5:
                vol = "high_vol"
            elif vr < 0.67:
                vol = "low_vol"
            else:
                vol = "normal_vol"
        else:
            vol = "unknown"

        # Momentum
        rank = rolling_21d_rank.iloc[i]
        if pd.notna(rank):
            if rank >= 0.75:
                mom = "strong_momentum"
            elif rank <= 0.25:
                mom = "weak_momentum"
            else:
                mom = "neutral_momentum"
        else:
            mom = "unknown"

        records.append({"date": date, "trend": trend, "volatility": vol, "momentum": mom})

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def assess_model_regime_fit(cfg: CommodityConfig) -> dict:
    """Check if the current regime matches the model's training regime.

    Loads model metadata for training date range, computes what regimes
    the training data covered, and compares to the current regime.
    If the model was trained in bull + normal_vol but we're in bear + high_vol,
    it flags as out-of-distribution.

    Returns compatibility score (0-1) and warning if < 0.5.
    """
    current = classify_regime(cfg)
    if current.get("status"):
        return {"status": current["status"], "commodity": cfg.name, "score": None}

    # Load model metadata for training date range
    meta_path = cfg.metadata_path
    if not meta_path.exists():
        return {
            "commodity": cfg.name,
            "status": "no_metadata",
            "score": None,
            "warning": "Cannot assess — no model metadata found",
        }

    with open(meta_path) as f:
        meta = json.load(f)

    # Get regime history to check training period coverage
    history = compute_regime_history(cfg)
    if history.empty:
        return {
            "commodity": cfg.name,
            "status": "no_regime_history",
            "score": None,
        }

    # Determine training date range from metadata
    train_end = meta.get("trained_at") or meta.get("timestamp")
    if train_end:
        train_end_date = pd.Timestamp(train_end)
    else:
        train_end_date = history.index[-1]

    # Assume training used ~2 years of data
    train_start_date = train_end_date - pd.Timedelta(days=504 * 1.5)
    train_mask = (history.index >= train_start_date) & (history.index <= train_end_date)
    training_regimes = history.loc[train_mask]

    if training_regimes.empty:
        return {
            "commodity": cfg.name,
            "status": "no_training_overlap",
            "score": 0.3,
            "warning": "Cannot determine training regime distribution",
        }

    # What fraction of training time was in the current regime?
    current_trend = current["trend"]
    current_vol = current["volatility"]

    trend_match = float((training_regimes["trend"] == current_trend).mean())
    vol_match = float((training_regimes["volatility"] == current_vol).mean())

    # Compatibility score: weighted combo (trend matters more)
    score = round(0.6 * trend_match + 0.4 * vol_match, 3)

    result = {
        "commodity": cfg.name,
        "score": score,
        "current_regime": {"trend": current_trend, "volatility": current_vol},
        "training_trend_distribution": training_regimes["trend"].value_counts(normalize=True).to_dict(),
        "training_vol_distribution": training_regimes["volatility"].value_counts(normalize=True).to_dict(),
        "trend_overlap": round(trend_match, 3),
        "vol_overlap": round(vol_match, 3),
    }

    if score < 0.5:
        result["warning"] = (
            f"Out-of-distribution: model trained mostly in "
            f"{training_regimes['trend'].mode().iloc[0]} + "
            f"{training_regimes['volatility'].mode().iloc[0]} "
            f"but current regime is {current_trend} + {current_vol}"
        )

    return result


def compute_cross_commodity_regime() -> dict:
    """Classify regime for all 7 commodities and detect global risk signals.

    Identifies 'global risk-off' (multiple commodities in bear regime)
    or 'global risk-on' (multiple in bull regime).
    """
    regimes = {}
    for key, cfg in COMMODITIES.items():
        regimes[key] = classify_regime(cfg)

    # Count regime states across commodities
    valid = {k: v for k, v in regimes.items() if not v.get("status")}
    n_valid = len(valid)

    if n_valid == 0:
        return {"status": "no_data", "regimes": regimes}

    trend_counts = {}
    vol_counts = {}
    for k, v in valid.items():
        trend_counts[v["trend"]] = trend_counts.get(v["trend"], 0) + 1
        vol_counts[v["volatility"]] = vol_counts.get(v["volatility"], 0) + 1

    n_bear = trend_counts.get("bear", 0)
    n_bull = trend_counts.get("bull", 0)
    n_high_vol = vol_counts.get("high_vol", 0)

    # Determine global state
    if n_bear >= 3:
        global_state = "global_risk_off"
    elif n_bull >= 4:
        global_state = "global_risk_on"
    elif n_high_vol >= 3:
        global_state = "global_stress"
    else:
        global_state = "mixed"

    return {
        "timestamp": datetime.now().isoformat(),
        "global_state": global_state,
        "n_commodities": n_valid,
        "trend_counts": trend_counts,
        "vol_counts": vol_counts,
        "regimes": regimes,
        "bear_commodities": [k for k, v in valid.items() if v["trend"] == "bear"],
        "bull_commodities": [k for k, v in valid.items() if v["trend"] == "bull"],
        "high_vol_commodities": [k for k, v in valid.items() if v["volatility"] == "high_vol"],
    }


# ── Reporting ─────────────────────────────────────────────────────────


def generate_regime_report(commodity_keys: list[str] = None) -> dict:
    """Full regime report for specified (or all) commodities.

    Includes classification, change detection, model fit assessment,
    and cross-commodity summary.
    """
    keys = commodity_keys or list(COMMODITIES.keys())
    report = {
        "timestamp": datetime.now().isoformat(),
        "commodities": {},
    }

    for key in keys:
        if key not in COMMODITIES:
            logger.warning(f"Unknown commodity: {key}")
            continue
        cfg = COMMODITIES[key]
        logger.info(f"Analyzing regime for {cfg.name}")

        regime = classify_regime(cfg)
        change = detect_regime_change(cfg)
        model_fit = assess_model_regime_fit(cfg)

        # Days in current regime
        history = compute_regime_history(cfg)
        days_in_regime = None
        if not history.empty:
            current_trend = regime.get("trend")
            if current_trend and not regime.get("status"):
                # Count consecutive days in current trend from the end
                trends = history["trend"]
                consecutive = 0
                for val in reversed(trends.values):
                    if val == current_trend:
                        consecutive += 1
                    else:
                        break
                days_in_regime = consecutive

        report["commodities"][key] = {
            "regime": regime,
            "change_detection": change,
            "model_fit": model_fit,
            "days_in_current_trend": days_in_regime,
        }

    # Cross-commodity view
    report["cross_commodity"] = compute_cross_commodity_regime()

    # Save report
    report_dir = LOGS_DIR / "regime"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"regime_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    return report


# ── CLI entry point ───────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Regime Detection Agent")
    parser.add_argument("commodities", nargs="*", help="Specific commodities to analyze")
    parser.add_argument("--alert", action="store_true", help="Only alert on regime changes")
    args = parser.parse_args()

    keys = args.commodities if args.commodities else None

    if args.alert:
        # Alert mode: only print regime changes
        check_keys = keys or list(COMMODITIES.keys())
        any_change = False
        for key in check_keys:
            if key not in COMMODITIES:
                continue
            cfg = COMMODITIES[key]
            change = detect_regime_change(cfg)
            if change.get("changed"):
                any_change = True
                prev = change["previous_regime"]
                curr = change["current_regime"]
                print(f"  REGIME CHANGE: {cfg.name}")
                print(f"    Trend:      {prev['trend']} -> {curr['trend']}"
                      + (" ***" if change.get("trend_changed") else ""))
                print(f"    Volatility: {prev['volatility']} -> {curr['volatility']}"
                      + (" ***" if change.get("vol_changed") else ""))
                print(f"    Approx change date: {change.get('change_date')}")
                print()
        if not any_change:
            print("  No regime changes detected.")
        return

    # Full report
    report = generate_regime_report(keys)

    print(f"\n{'='*60}")
    print("REGIME DETECTION REPORT")
    print(f"{'='*60}")

    for key, data in report["commodities"].items():
        cfg = COMMODITIES[key]
        regime = data["regime"]
        if regime.get("status"):
            print(f"\n  {cfg.name}: {regime['status']}")
            continue

        raw = regime["raw"]
        fit = data["model_fit"]
        days = data.get("days_in_current_trend")

        print(f"\n  {cfg.name}:")
        print(f"    Trend:      {regime['trend']:<12} (252d return: {raw['return_252d']:+.1%})")
        print(f"    Volatility: {regime['volatility']:<12} (vol ratio: {raw['vol_ratio']:.2f}x)")
        print(f"    Momentum:   {regime['momentum']:<18} (rank: {raw['momentum_rank_pct']:.0%})")
        if days is not None:
            print(f"    Days in current trend: {days}")

        score = fit.get("score")
        if score is not None:
            label = "OK" if score >= 0.5 else "WARNING"
            print(f"    Model fit:  {score:.2f} [{label}]")
        if fit.get("warning"):
            print(f"    >> {fit['warning']}")

    # Cross-commodity summary
    cross = report.get("cross_commodity", {})
    if not cross.get("status"):
        print(f"\n  GLOBAL STATE: {cross.get('global_state', 'unknown').upper()}")
        if cross.get("bear_commodities"):
            print(f"    Bear: {', '.join(cross['bear_commodities'])}")
        if cross.get("high_vol_commodities"):
            print(f"    High vol: {', '.join(cross['high_vol_commodities'])}")

    print()


if __name__ == "__main__":
    main()
