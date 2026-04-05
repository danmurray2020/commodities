"""Data Quality Agent — validates the data pipeline for each commodity.

Checks for price gaps, stale forward-fills, volume anomalies, supplementary
data correlation breakdown, and missing data in recent rows.

Usage:
    python -m agents data-quality                   # Check all commodities
    python -m agents data-quality coffee natgas      # Check specific commodities
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .design_log import log_observation, log_challenge
from .log import setup_logging
from .signals import emit_signal

logger = setup_logging("data_quality")

# Known supplementary columns that should correlate with each commodity's price.
# Maps commodity key -> list of (column_name, expected_relationship_description).
SUPPLEMENTARY_CORRELATIONS = {
    "coffee": ["brl_usd", "robusta", "sugar"],
    "cocoa": ["usd_index", "crude_oil"],
    "sugar": ["brl_usd", "crude_oil", "usd_index"],
    "natgas": ["crude_oil", "heating_oil", "coal"],
    "soybeans": ["crude_oil", "usd_index"],
    "wheat": ["crude_oil", "usd_index"],
    "copper": ["cny_usd", "iron_ore", "crude_oil"],
}

# Columns that are considered supplementary (COT, weather, FX, etc.) for
# stale forward-fill detection.  The price column itself is excluded.
PRICE_LIKE_COLS = {"Volume", "Date"}


# ── Per-commodity check ─────────────────────────────────────────────────

def check_commodity(cfg: CommodityConfig) -> dict:
    """Run all data quality checks for a single commodity.

    Returns a dict with keys:
        commodity, timestamp, issues (list of dicts), summary (counts by severity).
    """
    findings = {
        "commodity": cfg.name,
        "key": cfg.dir_name,
        "timestamp": datetime.now().isoformat(),
        "issues": [],
        "status": "ok",
    }

    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        findings["status"] = "error"
        findings["issues"].append({
            "check": "file_exists",
            "severity": "critical",
            "detail": f"combined_features.csv not found at {csv_path}",
        })
        return findings

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
    except Exception as e:
        findings["status"] = "error"
        findings["issues"].append({
            "check": "file_read",
            "severity": "critical",
            "detail": f"Failed to read CSV: {e}",
        })
        return findings

    if df.empty:
        findings["status"] = "error"
        findings["issues"].append({
            "check": "empty_data",
            "severity": "critical",
            "detail": "CSV is empty",
        })
        return findings

    df = df.sort_values("Date").reset_index(drop=True)
    price_col = cfg.price_col

    if price_col not in df.columns:
        findings["status"] = "error"
        findings["issues"].append({
            "check": "price_column",
            "severity": "critical",
            "detail": f"Price column '{price_col}' not found in data",
        })
        return findings

    # ── 1. Price gap detection ──────────────────────────────────────────
    _check_price_gaps(df, price_col, findings)

    # ── 2. Stale forward-fill detection ─────────────────────────────────
    _check_stale_fills(df, price_col, findings)

    # ── 3. Volume anomaly ───────────────────────────────────────────────
    _check_volume_anomaly(df, findings)

    # ── 4. Supplementary data correlation ───────────────────────────────
    _check_supplementary_correlation(df, price_col, cfg, findings)

    # ── 5. Missing data in last 63 rows ─────────────────────────────────
    _check_missing_data(df, findings)

    # Determine overall status
    severities = [i["severity"] for i in findings["issues"]]
    if "critical" in severities:
        findings["status"] = "critical"
    elif "high" in severities:
        findings["status"] = "warning"
    elif "medium" in severities:
        findings["status"] = "info"
    elif severities:
        findings["status"] = "info"

    # Summary counts
    findings["summary"] = {
        sev: sum(1 for s in severities if s == sev)
        for sev in ("critical", "high", "medium", "low")
        if any(s == sev for s in severities)
    }

    return findings


# ── Individual checks ───────────────────────────────────────────────────

def _check_price_gaps(df: pd.DataFrame, price_col: str, findings: dict):
    """Check for daily price changes >5% (log gaps); flag >10% as critical."""
    prices = df[price_col].dropna()
    if len(prices) < 2:
        return

    log_returns = np.log(prices / prices.shift(1)).dropna()
    abs_returns = log_returns.abs()

    critical_mask = abs_returns > np.log(1.10)  # >10% move
    warning_mask = (abs_returns > np.log(1.05)) & (~critical_mask)  # 5-10% move

    critical_gaps = df.loc[abs_returns[critical_mask].index]
    warning_gaps = df.loc[abs_returns[warning_mask].index]

    for idx in critical_gaps.index:
        date = df.loc[idx, "Date"]
        pct = abs_returns.loc[idx]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "?"
        findings["issues"].append({
            "check": "price_gap",
            "severity": "critical",
            "detail": f"Price gap {pct:.1%} (log) on {date_str}",
        })

    for idx in warning_gaps.index:
        date = df.loc[idx, "Date"]
        pct = abs_returns.loc[idx]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "?"
        findings["issues"].append({
            "check": "price_gap",
            "severity": "high",
            "detail": f"Price gap {pct:.1%} (log) on {date_str}",
        })


def _check_stale_fills(df: pd.DataFrame, price_col: str, findings: dict):
    """Check for >10 consecutive identical values in supplementary columns."""
    supplementary_cols = [
        c for c in df.columns
        if c != price_col and c not in PRICE_LIKE_COLS and c != "Date"
    ]

    for col in supplementary_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        # Find max run of identical consecutive values
        shifted = series != series.shift(1)
        groups = shifted.cumsum()
        max_run = groups.value_counts().max()

        if max_run > 10:
            # Find where the longest run is
            longest_group = groups.value_counts().idxmax()
            run_indices = groups[groups == longest_group].index
            start_idx = run_indices[0]
            start_date = df.loc[start_idx, "Date"] if start_idx in df.index else "?"
            date_str = pd.Timestamp(start_date).strftime("%Y-%m-%d") if pd.notna(start_date) else "?"
            severity = "high" if max_run > 30 else "medium"
            findings["issues"].append({
                "check": "stale_fill",
                "severity": severity,
                "detail": (
                    f"Column '{col}' has {max_run} consecutive identical values "
                    f"starting near {date_str}"
                ),
            })


def _check_volume_anomaly(df: pd.DataFrame, findings: dict):
    """If Volume column exists, flag days with volume <10% of 252-day rolling mean."""
    if "Volume" not in df.columns:
        return

    vol = df["Volume"].copy()
    rolling_mean = vol.rolling(window=252, min_periods=63).mean()
    threshold = rolling_mean * 0.10

    low_vol_mask = (vol < threshold) & (rolling_mean.notna()) & (vol > 0)
    low_vol_days = df.loc[low_vol_mask]

    if len(low_vol_days) > 0:
        # Report the most recent occurrences (last 63 days)
        recent = low_vol_days.tail(10)
        for idx in recent.index:
            date = df.loc[idx, "Date"]
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "?"
            actual = vol.loc[idx]
            mean_val = rolling_mean.loc[idx]
            pct = (actual / mean_val * 100) if mean_val > 0 else 0
            findings["issues"].append({
                "check": "volume_anomaly",
                "severity": "medium",
                "detail": (
                    f"Volume {actual:.0f} on {date_str} is {pct:.1f}% "
                    f"of 252-day mean ({mean_val:.0f})"
                ),
            })


def _check_supplementary_correlation(
    df: pd.DataFrame, price_col: str, cfg: CommodityConfig, findings: dict
):
    """Check that supplementary data still correlates with commodity price.

    If 63-day rolling correlation drops below 0.05 in absolute value,
    flag as potential data issue.
    """
    key = None
    for k, c in COMMODITIES.items():
        if c.dir_name == cfg.dir_name:
            key = k
            break

    corr_cols = SUPPLEMENTARY_CORRELATIONS.get(key, [])

    for col in corr_cols:
        if col not in df.columns:
            continue

        rolling_corr = df[price_col].rolling(window=63, min_periods=30).corr(df[col])

        # Check the most recent available correlation value
        recent_corr = rolling_corr.dropna()
        if recent_corr.empty:
            continue

        last_corr = recent_corr.iloc[-1]
        if abs(last_corr) < 0.05:
            last_idx = recent_corr.index[-1]
            date = df.loc[last_idx, "Date"] if last_idx in df.index else "?"
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "?"
            findings["issues"].append({
                "check": "supplementary_correlation",
                "severity": "medium",
                "detail": (
                    f"63-day rolling correlation between {price_col} and '{col}' "
                    f"is {last_corr:.3f} as of {date_str} (below 0.05 threshold)"
                ),
            })


def _check_missing_data(df: pd.DataFrame, findings: dict):
    """Count NaN values per column in the last 63 rows."""
    tail = df.tail(63)
    nan_counts = tail.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]

    for col, count in cols_with_nans.items():
        if col == "Date":
            continue
        pct = count / len(tail) * 100
        severity = "high" if pct > 50 else ("medium" if pct > 20 else "low")
        findings["issues"].append({
            "check": "missing_data",
            "severity": severity,
            "detail": f"Column '{col}' has {count}/{len(tail)} NaN values ({pct:.0f}%) in last 63 rows",
        })


# ── Pipeline entry point ────────────────────────────────────────────────

def run_data_quality(commodity_keys: list[str] | None = None) -> dict:
    """Run data quality checks across all (or specified) commodities.

    Returns a dict keyed by commodity key with check results.
    """
    targets = commodity_keys or list(COMMODITIES.keys())
    results = {}

    logger.info("=" * 60)
    logger.info("DATA QUALITY AGENT")
    logger.info("=" * 60)

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            logger.warning(f"Unknown commodity key: {key}")
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"Checking {cfg.name}...")

        result = check_commodity(cfg)
        results[key] = result

        issues = result["issues"]
        if not issues:
            logger.info(f"  {cfg.name}: OK — no data quality issues")
            log_observation("data_quality", "No data quality issues found", cfg.name)
            continue

        # Log and emit signals for each issue
        logger.warning(f"  {cfg.name}: {len(issues)} issue(s) found (status: {result['status']})")
        for issue in issues:
            logger.warning(f"    [{issue['severity'].upper()}] {issue['detail']}")

            emit_signal(
                "data_quality",
                "data_anomaly",
                key,
                severity=issue["severity"],
                detail=issue["detail"],
            )

        # Summarize in design log
        critical_count = sum(1 for i in issues if i["severity"] == "critical")
        high_count = sum(1 for i in issues if i["severity"] == "high")

        summary_msg = (
            f"Data quality: {len(issues)} issues "
            f"({critical_count} critical, {high_count} high)"
        )
        log_observation("data_quality", summary_msg, cfg.name)

        if critical_count > 0:
            critical_details = [
                i["detail"] for i in issues if i["severity"] == "critical"
            ]
            log_challenge(
                "data_quality",
                "data pipeline reliability",
                f"Critical data issues detected: {'; '.join(critical_details)}",
                cfg.name,
            )

    return results


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data Quality Agent — validate commodity data pipelines"
    )
    parser.add_argument(
        "commodities", nargs="*",
        help="Specific commodity keys to check (default: all)",
    )
    args = parser.parse_args()

    results = run_data_quality(args.commodities or None)

    # Print summary table
    print(f"\n{'='*60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*60}")

    for key, result in results.items():
        cfg = COMMODITIES.get(key)
        if not cfg:
            continue
        n_issues = len(result["issues"])
        status = result["status"]
        summary = result.get("summary", {})
        counts = ", ".join(f"{v} {k}" for k, v in summary.items()) if summary else "clean"
        print(f"  {cfg.name:<15} [{status:<8}] {n_issues} issues ({counts})")

        # Show critical and high issues inline
        for issue in result["issues"]:
            if issue["severity"] in ("critical", "high"):
                print(f"    [{issue['severity'].upper()}] {issue['detail']}")


if __name__ == "__main__":
    main()
