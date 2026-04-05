"""Compliance Agent — audit, governance, and regulatory checks.

Ensures the trading system operates within defined limits and
maintains proper documentation for all decisions.

Usage:
    python -m agents compliance                # full compliance check
    python -m agents compliance --limits       # position limits only
    python -m agents compliance --audit-trail  # trade rationale audit
    python -m agents compliance --pnl-attr     # P&L attribution
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .log import setup_logging, log_event


logger = setup_logging("compliance")

# ── DB access ─────────────────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db import get_db
    db = get_db()
except Exception:
    db = None


# ── CFTC-style position limits (approximate, in contracts) ───────────
POSITION_LIMITS = {
    "coffee":   5_000,
    "cocoa":    5_000,
    "sugar":    10_000,
    "natgas":   12_000,
    "soybeans": 5_500,
    "wheat":    5_500,
    "copper":   5_000,
}


def check_position_limits() -> dict:
    """Check current positions against CFTC-style limits per commodity.

    Loads open trades from the database and compares aggregate position
    sizes to the regulatory-style limits defined above.
    """
    results = {}

    for key, cfg in COMMODITIES.items():
        limit = POSITION_LIMITS.get(key, 5_000)
        current_position = 0.0

        if db is not None:
            try:
                conn = db._get_conn()
                rows = conn.execute(
                    """SELECT direction, position_size FROM trades
                       WHERE commodity = ? AND exit_date IS NULL""",
                    (key,),
                ).fetchall()
                for row in rows:
                    size = float(row["position_size"])
                    current_position += size
            except Exception as e:
                logger.warning(f"Could not query trades for {key}: {e}")

        utilization = current_position / limit if limit > 0 else 0.0
        status = "ok"
        if utilization > 0.90:
            status = "critical"
        elif utilization > 0.75:
            status = "warning"

        results[key] = {
            "commodity": cfg.name,
            "current_position": current_position,
            "limit": limit,
            "utilization_pct": round(utilization * 100, 2),
            "status": status,
        }

    logger.info(f"Position limit check complete: {len(results)} commodities checked")
    return results


def audit_trade_rationale() -> dict:
    """Audit recent trades for proper documentation and rationale.

    Each trade should have:
    - A prediction backing it (confidence recorded)
    - Confidence above the commodity's threshold
    - Proper entry price and direction logged
    - Exit details logged if closed

    Returns per-trade audit results with compliant/non_compliant status.
    """
    if db is None:
        return {"status": "no_db", "message": "Database not available"}

    try:
        conn = db._get_conn()
        rows = conn.execute(
            """SELECT id, commodity, entry_date, entry_price, direction,
                      position_size, confidence, pred_return,
                      exit_date, exit_price, exit_reason, pnl_pct
               FROM trades
               ORDER BY entry_date DESC
               LIMIT 100""",
        ).fetchall()
    except Exception as e:
        logger.error(f"Failed to query trades: {e}")
        return {"status": "error", "message": str(e)}

    audit_results = []
    compliant_count = 0
    non_compliant_count = 0

    for row in rows:
        issues = []
        commodity = row["commodity"]
        cfg = COMMODITIES.get(commodity)

        # Check prediction backing
        if row["confidence"] is None:
            issues.append("missing_confidence")
        elif cfg and row["confidence"] < cfg.confidence_threshold:
            issues.append(f"confidence_below_threshold ({row['confidence']:.2f} < {cfg.confidence_threshold})")

        if row["pred_return"] is None:
            issues.append("missing_predicted_return")

        # Check entry documentation
        if row["entry_price"] is None or row["entry_price"] <= 0:
            issues.append("invalid_entry_price")
        if row["direction"] not in ("LONG", "SHORT"):
            issues.append(f"invalid_direction: {row['direction']}")

        # Check exit documentation for closed trades
        if row["exit_date"] is not None:
            if row["exit_price"] is None:
                issues.append("closed_trade_missing_exit_price")
            if row["exit_reason"] is None:
                issues.append("closed_trade_missing_exit_reason")
            if row["pnl_pct"] is None:
                issues.append("closed_trade_missing_pnl")

        status = "non_compliant" if issues else "compliant"
        if status == "compliant":
            compliant_count += 1
        else:
            non_compliant_count += 1

        audit_results.append({
            "trade_id": row["id"],
            "commodity": commodity,
            "entry_date": row["entry_date"],
            "direction": row["direction"],
            "status": status,
            "issues": issues,
        })

    summary = {
        "total_audited": len(audit_results),
        "compliant": compliant_count,
        "non_compliant": non_compliant_count,
        "compliance_rate": round(compliant_count / len(audit_results), 4) if audit_results else 1.0,
        "trades": audit_results,
    }

    logger.info(
        f"Trade audit complete: {compliant_count}/{len(audit_results)} compliant"
    )
    return summary


def check_model_governance() -> dict:
    """Verify each production model has proper governance documentation.

    Checks for:
    - metadata.json with metrics
    - Model trained within last 90 days
    - Holdout accuracy recorded
    - Features list documented
    """
    results = {}
    now = datetime.now()
    max_age_days = 90

    for key, cfg in COMMODITIES.items():
        issues = []
        governance = {
            "commodity": cfg.name,
            "has_metadata": False,
            "model_age_days": None,
            "has_holdout_accuracy": False,
            "has_features_list": False,
        }

        # Check metadata file
        metadata_path = cfg.metadata_path
        if metadata_path.exists():
            governance["has_metadata"] = True
            try:
                meta = json.loads(metadata_path.read_text())

                # Check training date / staleness
                trained_at = meta.get("trained_at") or meta.get("timestamp")
                if trained_at:
                    try:
                        trained_date = datetime.fromisoformat(trained_at)
                        age_days = (now - trained_date).days
                        governance["model_age_days"] = age_days
                        if age_days > max_age_days:
                            issues.append(f"stale_model ({age_days} days old, limit {max_age_days})")
                    except ValueError:
                        issues.append("unparseable_training_date")
                else:
                    issues.append("missing_training_date")

                # Check holdout accuracy
                holdout = meta.get("holdout_accuracy") or meta.get("holdout_metrics", {}).get("accuracy")
                if holdout is not None:
                    governance["has_holdout_accuracy"] = True
                    governance["holdout_accuracy"] = holdout
                else:
                    issues.append("missing_holdout_accuracy")

                # Check features list
                features = meta.get("features") or meta.get("feature_names")
                if features and len(features) > 0:
                    governance["has_features_list"] = True
                    governance["n_features"] = len(features)
                else:
                    issues.append("missing_features_list")

            except (json.JSONDecodeError, KeyError) as e:
                issues.append(f"metadata_parse_error: {e}")
        else:
            issues.append("no_metadata_file")

        # Also check DB for model records
        if db is not None:
            try:
                conn = db._get_conn()
                row = conn.execute(
                    """SELECT trained_at, holdout_accuracy, n_features
                       FROM models
                       WHERE commodity = ? AND is_production = 1
                       ORDER BY trained_at DESC LIMIT 1""",
                    (key,),
                ).fetchone()
                if row and row["holdout_accuracy"] is not None:
                    governance["has_holdout_accuracy"] = True
                    governance["db_holdout_accuracy"] = row["holdout_accuracy"]
                if row and row["n_features"] is not None and row["n_features"] > 0:
                    governance["has_features_list"] = True
            except Exception:
                pass

        governance["issues"] = issues
        governance["status"] = "compliant" if not issues else "non_compliant"
        results[key] = governance

    compliant = sum(1 for v in results.values() if v["status"] == "compliant")
    logger.info(f"Model governance check: {compliant}/{len(results)} compliant")
    return results


def compute_pnl_attribution() -> dict:
    """Decompose closed-trade P&L into alpha, beta, and timing components.

    - Alpha: P&L from directional prediction accuracy (actual trade P&L
      minus what buy-and-hold would have produced over the same period).
    - Beta: P&L from overall market direction (buy-and-hold return over
      the trade holding period).
    - Timing: P&L from entry/exit timing vs holding for the full horizon
      (difference between actual hold P&L and full-horizon hold P&L).
    """
    if db is None:
        return {"status": "no_db", "message": "Database not available"}

    try:
        conn = db._get_conn()
        rows = conn.execute(
            """SELECT id, commodity, entry_date, entry_price, direction,
                      exit_date, exit_price, pnl_pct, hold_days
               FROM trades
               WHERE exit_date IS NOT NULL AND exit_price IS NOT NULL""",
        ).fetchall()
    except Exception as e:
        logger.error(f"Failed to query closed trades: {e}")
        return {"status": "error", "message": str(e)}

    if not rows:
        return {"status": "no_trades", "message": "No closed trades found"}

    # Cache price data per commodity
    price_cache = {}
    for key, cfg in COMMODITIES.items():
        csv_path = cfg.data_dir / "combined_features.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                price_cache[key] = df[cfg.price_col]
            except Exception:
                pass

    attributions = []
    totals = {"alpha": 0.0, "beta": 0.0, "timing": 0.0, "total_pnl": 0.0}

    for row in rows:
        commodity = row["commodity"]
        cfg = COMMODITIES.get(commodity)
        if cfg is None:
            continue

        entry_price = row["entry_price"]
        exit_price = row["exit_price"]
        direction = row["direction"]

        # Actual trade P&L
        if direction == "LONG":
            trade_return = (exit_price / entry_price) - 1
        else:
            trade_return = (entry_price / exit_price) - 1

        # Beta: buy-and-hold return over the same period
        prices = price_cache.get(commodity)
        buy_hold_return = 0.0
        full_horizon_return = 0.0

        if prices is not None:
            try:
                entry_dt = pd.Timestamp(row["entry_date"])
                exit_dt = pd.Timestamp(row["exit_date"])

                # Find closest available prices
                entry_mask = prices.index >= entry_dt
                exit_mask = prices.index >= exit_dt
                if entry_mask.any() and exit_mask.any():
                    bh_entry = float(prices.loc[entry_mask].iloc[0])
                    bh_exit = float(prices.loc[exit_mask].iloc[0])
                    buy_hold_return = (bh_exit / bh_entry) - 1

                # Full horizon return (entry + 63 trading days)
                horizon_dt = entry_dt + pd.Timedelta(days=cfg.horizon)
                horizon_mask = prices.index >= horizon_dt
                if entry_mask.any() and horizon_mask.any():
                    bh_horizon = float(prices.loc[horizon_mask].iloc[0])
                    full_horizon_return = (bh_horizon / bh_entry) - 1
            except Exception:
                pass

        # Decomposition
        beta = buy_hold_return
        alpha = trade_return - beta
        timing = trade_return - full_horizon_return if full_horizon_return != 0.0 else 0.0

        attributions.append({
            "trade_id": row["id"],
            "commodity": commodity,
            "direction": direction,
            "total_return": round(trade_return, 6),
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "timing": round(timing, 6),
        })

        totals["alpha"] += alpha
        totals["beta"] += beta
        totals["timing"] += timing
        totals["total_pnl"] += trade_return

    n = len(attributions)
    summary = {
        "n_trades": n,
        "avg_alpha": round(totals["alpha"] / n, 6) if n else 0,
        "avg_beta": round(totals["beta"] / n, 6) if n else 0,
        "avg_timing": round(totals["timing"] / n, 6) if n else 0,
        "avg_total_pnl": round(totals["total_pnl"] / n, 6) if n else 0,
        "trades": attributions,
    }

    logger.info(f"P&L attribution: {n} trades, avg alpha={summary['avg_alpha']:+.4f}")
    return summary


def generate_compliance_report() -> dict:
    """Generate a full compliance report combining all checks."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "position_limits": check_position_limits(),
        "trade_audit": audit_trade_rationale(),
        "model_governance": check_model_governance(),
        "pnl_attribution": compute_pnl_attribution(),
    }

    # Overall status
    limit_issues = sum(
        1 for v in report["position_limits"].values()
        if isinstance(v, dict) and v.get("status") != "ok"
    )
    governance_issues = sum(
        1 for v in report["model_governance"].values()
        if isinstance(v, dict) and v.get("status") != "compliant"
    )
    audit_rate = report["trade_audit"].get("compliance_rate", 1.0) if isinstance(report["trade_audit"], dict) else 1.0

    report["overall"] = {
        "position_limit_issues": limit_issues,
        "governance_issues": governance_issues,
        "trade_compliance_rate": audit_rate,
        "status": "ok" if (limit_issues == 0 and governance_issues == 0 and audit_rate >= 0.95) else "needs_attention",
    }

    # Log to DB
    if db is not None:
        try:
            conn = db._get_conn()
            conn.execute(
                """INSERT INTO agent_runs (agent_name, started_at, finished_at, status, summary, report_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "compliance",
                    report["timestamp"],
                    datetime.now().isoformat(),
                    report["overall"]["status"],
                    f"Limits:{limit_issues} issues, Governance:{governance_issues} issues, Audit:{audit_rate:.0%}",
                    json.dumps(report, default=str),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log compliance run to DB: {e}")

    log_event(logger, "Compliance report generated", data=report["overall"])
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compliance and governance checks")
    parser.add_argument("--limits", action="store_true", help="Position limits check only")
    parser.add_argument("--audit-trail", action="store_true", help="Trade rationale audit only")
    parser.add_argument("--pnl-attr", action="store_true", help="P&L attribution only")
    args = parser.parse_args()

    if args.limits:
        results = check_position_limits()
        print(f"\n{'='*60}")
        print("POSITION LIMITS CHECK")
        print(f"{'='*60}")
        for key, info in results.items():
            status_icon = {"ok": "OK", "warning": "WARN", "critical": "CRIT"}.get(info["status"], "??")
            print(f"  {info['commodity']:<15} [{status_icon}]  "
                  f"{info['current_position']:.0f} / {info['limit']}  "
                  f"({info['utilization_pct']:.1f}%)")
        return

    if args.audit_trail:
        results = audit_trade_rationale()
        print(f"\n{'='*60}")
        print("TRADE RATIONALE AUDIT")
        print(f"{'='*60}")
        if results.get("status") in ("no_db", "error"):
            print(f"  {results['message']}")
            return
        print(f"  Total audited: {results['total_audited']}")
        print(f"  Compliant:     {results['compliant']}")
        print(f"  Non-compliant: {results['non_compliant']}")
        print(f"  Rate:          {results['compliance_rate']:.1%}")
        non_compliant = [t for t in results["trades"] if t["status"] == "non_compliant"]
        if non_compliant:
            print(f"\n  NON-COMPLIANT TRADES:")
            for t in non_compliant[:20]:
                print(f"    Trade #{t['trade_id']} ({t['commodity']}, {t['entry_date']}): {', '.join(t['issues'])}")
        return

    if args.pnl_attr:
        results = compute_pnl_attribution()
        print(f"\n{'='*60}")
        print("P&L ATTRIBUTION")
        print(f"{'='*60}")
        if results.get("status") in ("no_db", "error", "no_trades"):
            print(f"  {results['message']}")
            return
        print(f"  Trades analyzed: {results['n_trades']}")
        print(f"  Avg total P&L:   {results['avg_total_pnl']:+.4%}")
        print(f"  Avg alpha:       {results['avg_alpha']:+.4%}")
        print(f"  Avg beta:        {results['avg_beta']:+.4%}")
        print(f"  Avg timing:      {results['avg_timing']:+.4%}")
        return

    # Full compliance report
    report = generate_compliance_report()

    print(f"\n{'='*60}")
    print("COMPLIANCE REPORT")
    print(f"{'='*60}")

    # Position limits
    print("\nPOSITION LIMITS:")
    for key, info in report["position_limits"].items():
        if not isinstance(info, dict):
            continue
        status_icon = {"ok": "OK", "warning": "WARN", "critical": "CRIT"}.get(info["status"], "??")
        print(f"  {info['commodity']:<15} [{status_icon}]  "
              f"{info['current_position']:.0f} / {info['limit']}  "
              f"({info['utilization_pct']:.1f}%)")

    # Model governance
    print("\nMODEL GOVERNANCE:")
    for key, gov in report["model_governance"].items():
        if not isinstance(gov, dict):
            continue
        status_icon = "OK" if gov["status"] == "compliant" else "FAIL"
        age_str = f"{gov['model_age_days']}d" if gov["model_age_days"] is not None else "?"
        print(f"  {gov['commodity']:<15} [{status_icon}]  age={age_str}  "
              f"metadata={'Y' if gov['has_metadata'] else 'N'}  "
              f"holdout={'Y' if gov['has_holdout_accuracy'] else 'N'}  "
              f"features={'Y' if gov['has_features_list'] else 'N'}")
        if gov["issues"]:
            for issue in gov["issues"]:
                print(f"    ^ {issue}")

    # Trade audit
    print("\nTRADE AUDIT:")
    audit = report["trade_audit"]
    if isinstance(audit, dict) and "total_audited" in audit:
        print(f"  Audited: {audit['total_audited']}  "
              f"Compliant: {audit['compliant']}  "
              f"Rate: {audit['compliance_rate']:.1%}")
    else:
        print(f"  {audit.get('message', 'N/A')}")

    # P&L attribution
    print("\nP&L ATTRIBUTION:")
    pnl = report["pnl_attribution"]
    if isinstance(pnl, dict) and "n_trades" in pnl:
        print(f"  Trades: {pnl['n_trades']}  "
              f"Alpha: {pnl['avg_alpha']:+.4%}  "
              f"Beta: {pnl['avg_beta']:+.4%}  "
              f"Timing: {pnl['avg_timing']:+.4%}")
    else:
        print(f"  {pnl.get('message', 'N/A')}")

    # Overall
    overall = report["overall"]
    print(f"\nOVERALL: [{overall['status'].upper()}]")


if __name__ == "__main__":
    main()
