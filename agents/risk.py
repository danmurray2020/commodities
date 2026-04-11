"""Risk Management Agent — portfolio risk analysis and limit enforcement.

Responsibilities:
- Value-at-Risk (VaR) calculation
- Stress testing against historical scenarios
- Drawdown monitoring and limits
- Position limit enforcement
- Correlation risk detection

Usage:
    python -m agents risk                      # full risk report
    python -m agents risk --var                # VaR only
    python -m agents risk --stress             # stress test scenarios
    python -m agents risk --limits             # position limit check
"""

import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR, LOGS_DIR
from .design_log import log_observation, log_challenge
from .log import setup_logging, log_event


logger = setup_logging("risk")

RISK_LOGS_DIR = LOGS_DIR / "risk"


# ── Stress scenarios ──────────────────────────────────────────────────
STRESS_SCENARIOS = {
    "2008_financial_crisis": {
        "coffee": -0.30, "cocoa": -0.25, "sugar": -0.35,
        "natgas": -0.60, "soybeans": -0.40, "wheat": -0.45, "copper": -0.55,
    },
    "2020_covid_crash": {
        "coffee": -0.15, "cocoa": -0.10, "sugar": -0.20,
        "natgas": -0.40, "soybeans": -0.10, "wheat": -0.05, "copper": -0.25,
    },
    "2022_commodity_spike": {
        "coffee": +0.50, "cocoa": +0.20, "sugar": +0.30,
        "natgas": +1.20, "soybeans": +0.25, "wheat": +0.60, "copper": +0.15,
    },
    "supply_shock": {
        "coffee": +0.40, "cocoa": +0.35, "sugar": +0.25,
        "natgas": +0.80, "soybeans": +0.30, "wheat": +0.50, "copper": +0.10,
    },
}

# ── CFTC approximate position limits (contracts) ─────────────────────
POSITION_LIMITS = {
    "coffee":   5_000,
    "cocoa":    5_000,
    "sugar":   10_000,
    "natgas":  12_000,
    "soybeans": 5_500,
    "wheat":    5_500,
    "copper":   5_000,
}


def _load_all_prices() -> pd.DataFrame:
    """Load price series for all commodities into a single DataFrame."""
    prices = {}
    for key, cfg in COMMODITIES.items():
        csv_path = cfg.data_dir / "combined_features.csv"
        if not csv_path.exists():
            logger.warning(f"{cfg.name}: data file not found at {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if cfg.price_col in df.columns:
                prices[key] = df[cfg.price_col]
        except Exception as e:
            logger.error(f"{cfg.name}: failed to load data — {e}")

    if not prices:
        logger.error("No price data loaded for any commodity")
        return pd.DataFrame()

    combined = pd.DataFrame(prices)
    combined = combined.sort_index().ffill().dropna()
    return combined


def _load_current_positions() -> dict:
    """Load current positions from DB or return default equal-weight positions.

    Returns dict mapping commodity key -> position weight (fraction).
    """
    try:
        from db import get_db
        db = get_db()
        # Open trades = trades with no exit_date set yet (per db/schema.sql)
        trades = db.conn.execute(
            "SELECT commodity, direction, position_size FROM trades WHERE exit_date IS NULL"
        ).fetchall()
        if trades:
            positions = {}
            for t in trades:
                commodity = t[0]
                direction = 1.0 if t[1] == "LONG" else -1.0
                size = t[2] if t[2] else 0.10
                positions[commodity] = direction * size
            return positions
    except Exception as e:
        logger.info(f"Could not load positions from DB: {e}")

    # Default: equal weight across all commodities (small position)
    return {key: 0.05 for key in COMMODITIES}


# ── 1. Value-at-Risk ─────────────────────────────────────────────────

def compute_var(confidence: float = 0.95, horizon_days: int = 1) -> dict:
    """Compute Value-at-Risk using historical simulation.

    Args:
        confidence: VaR confidence level (e.g., 0.95 for 95%).
        horizon_days: Holding period in days.

    Returns:
        Dict with per_commodity_var, portfolio_var, diversification_benefit.
    """
    prices = _load_all_prices()
    if prices.empty:
        return {"error": "No price data available"}

    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return {"error": "Insufficient return data for VaR calculation"}

    # Scale returns to horizon
    if horizon_days > 1:
        scaled_returns = returns * np.sqrt(horizon_days)
    else:
        scaled_returns = returns

    # Per-commodity VaR (historical simulation)
    alpha = 1 - confidence
    per_commodity_var = {}
    for col in scaled_returns.columns:
        var_value = float(np.percentile(scaled_returns[col], alpha * 100))
        per_commodity_var[col] = round(abs(var_value), 6)

    # Portfolio VaR (using correlation structure)
    positions = _load_current_positions()
    weights = np.array([positions.get(col, 0.0) for col in scaled_returns.columns])

    if np.sum(np.abs(weights)) == 0:
        weights = np.ones(len(scaled_returns.columns)) / len(scaled_returns.columns)

    portfolio_returns = scaled_returns.values @ weights
    portfolio_var = float(abs(np.percentile(portfolio_returns, alpha * 100)))

    # Diversification benefit: sum of individual VaRs vs portfolio VaR
    individual_var_sum = sum(
        per_commodity_var[col] * abs(positions.get(col, 1.0 / len(per_commodity_var)))
        for col in per_commodity_var
    )
    diversification_benefit = round(
        1 - (portfolio_var / individual_var_sum) if individual_var_sum > 0 else 0, 4
    )

    return {
        "confidence": confidence,
        "horizon_days": horizon_days,
        "per_commodity_var": per_commodity_var,
        "portfolio_var": round(portfolio_var, 6),
        "diversification_benefit": diversification_benefit,
        "n_observations": len(returns),
    }


# ── 2. Stress tests ──────────────────────────────────────────────────

def run_stress_tests() -> dict:
    """Run portfolio through historical stress scenarios.

    Returns:
        Dict mapping scenario name -> portfolio P&L and breakdown.
    """
    positions = _load_current_positions()
    results = {}

    for scenario_name, scenario_returns in STRESS_SCENARIOS.items():
        pnl_breakdown = {}
        total_pnl = 0.0

        for commodity, stress_return in scenario_returns.items():
            position = positions.get(commodity, 0.0)
            commodity_pnl = position * stress_return
            pnl_breakdown[commodity] = round(commodity_pnl, 6)
            total_pnl += commodity_pnl

        results[scenario_name] = {
            "portfolio_pnl": round(total_pnl, 6),
            "breakdown": pnl_breakdown,
            "worst_commodity": min(pnl_breakdown, key=pnl_breakdown.get),
            "best_commodity": max(pnl_breakdown, key=pnl_breakdown.get),
        }

    return results


# ── 3. Drawdown monitoring ───────────────────────────────────────────

def check_drawdown(max_allowed: float = 0.15) -> dict:
    """Check current and maximum drawdown against limit.

    Args:
        max_allowed: Maximum allowed drawdown (e.g., 0.15 = 15%).

    Returns:
        Dict with current_drawdown, max_drawdown, breach (bool).
    """
    # Try loading trade history from DB
    equity_curve = None
    try:
        from db import get_db
        db = get_db()
        # Closed trades = trades with exit_date set; pnl_pct is fractional return.
        trades = db.conn.execute(
            "SELECT exit_date, pnl_pct FROM trades WHERE exit_date IS NOT NULL ORDER BY exit_date"
        ).fetchall()
        if trades:
            cumulative = 1.0
            curve = [1.0]
            for t in trades:
                pnl = t[1] if t[1] else 0.0
                cumulative *= (1 + pnl)
                curve.append(cumulative)
            equity_curve = np.array(curve)
    except Exception as e:
        logger.info(f"Could not load trades from DB: {e}")

    # Fallback: estimate from price data
    if equity_curve is None or len(equity_curve) < 2:
        prices = _load_all_prices()
        if prices.empty:
            return {"error": "No data available for drawdown calculation"}

        # Simulate equal-weight portfolio
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
        equity_curve = (1 + portfolio_returns).cumprod().values

    # Compute drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    current_drawdown = float(drawdown[-1])
    max_drawdown = float(drawdown.min())

    return {
        "current_drawdown": round(abs(current_drawdown), 6),
        "max_drawdown": round(abs(max_drawdown), 6),
        "max_allowed": max_allowed,
        "breach": abs(current_drawdown) > max_allowed,
        "max_breach": abs(max_drawdown) > max_allowed,
        "equity_points": len(equity_curve),
    }


# ── 4. Position limits ───────────────────────────────────────────────

def check_position_limits() -> dict:
    """Check current positions against CFTC approximate limits.

    Returns:
        Dict mapping commodity -> limit check result.
    """
    positions = _load_current_positions()
    results = {}

    for key, cfg in COMMODITIES.items():
        limit = POSITION_LIMITS.get(key, 5_000)
        position_weight = abs(positions.get(key, 0.0))

        # Convert weight to approximate contract equivalent
        # Assume $1M notional portfolio
        notional = 1_000_000
        approx_contracts = int(position_weight * notional / 10_000)  # rough approximation

        utilization = approx_contracts / limit if limit > 0 else 0

        results[key] = {
            "commodity": cfg.name,
            "position_weight": round(position_weight, 4),
            "approx_contracts": approx_contracts,
            "cftc_limit": limit,
            "utilization_pct": round(utilization * 100, 2),
            "within_limits": utilization < 1.0,
        }

    return results


# ── 5. Correlation matrix ────────────────────────────────────────────

def compute_correlation_matrix(window: int = 63) -> dict:
    """Compute rolling correlation matrix and flag high correlations.

    Args:
        window: Rolling window in trading days (default: 63 = ~3 months).

    Returns:
        Dict with correlation matrix, high-correlation flags.
    """
    prices = _load_all_prices()
    if prices.empty:
        return {"error": "No price data available"}

    returns = prices.pct_change().dropna()

    if len(returns) < window:
        logger.warning(f"Insufficient data for {window}-day rolling correlation")
        corr_matrix = returns.corr()
    else:
        # Use most recent window for the correlation matrix
        recent = returns.tail(window)
        corr_matrix = recent.corr()

    # Convert to serializable dict
    corr_dict = {}
    for col in corr_matrix.columns:
        corr_dict[col] = {
            row: round(float(corr_matrix.loc[row, col]), 4)
            for row in corr_matrix.index
        }

    # Flag high correlations (> 0.7, excluding self-correlation)
    high_corr_flags = []
    seen = set()
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            corr_val = float(corr_matrix.loc[col1, col2])
            pair = tuple(sorted([col1, col2]))
            if abs(corr_val) > 0.7 and pair not in seen:
                seen.add(pair)
                high_corr_flags.append({
                    "pair": [col1, col2],
                    "correlation": round(corr_val, 4),
                    "risk": "high" if abs(corr_val) > 0.85 else "elevated",
                })

    return {
        "window_days": window,
        "matrix": corr_dict,
        "high_correlation_flags": high_corr_flags,
        "n_high_corr_pairs": len(high_corr_flags),
    }


# ── 6. Full risk report ──────────────────────────────────────────────

def generate_risk_report() -> dict:
    """Generate comprehensive risk report combining all analyses."""
    logger.info("Generating full risk report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "var": compute_var(),
        "stress_tests": run_stress_tests(),
        "drawdown": check_drawdown(),
        "position_limits": check_position_limits(),
        "correlation": compute_correlation_matrix(),
    }

    # Save report to logs/risk/
    RISK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = RISK_LOGS_DIR / f"risk_report_{date.today().isoformat()}.json"
    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Risk report saved to {report_file}")
    except Exception as e:
        logger.warning(f"Failed to save risk report: {e}")

    return report


# ── 7. Main ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Risk management analysis")
    parser.add_argument("--var", action="store_true", help="VaR calculation only")
    parser.add_argument("--stress", action="store_true", help="Stress test scenarios only")
    parser.add_argument("--limits", action="store_true", help="Position limit check only")
    parser.add_argument("--correlation", action="store_true", help="Correlation matrix only")
    args = parser.parse_args()

    # DB logging
    run_id = None
    try:
        from db import get_db
        db = get_db()
        run_id = db.start_agent_run("risk", list(COMMODITIES.keys()))
    except Exception as e:
        logger.warning(f"DB logging unavailable: {e}")

    try:
        if args.var:
            var_result = compute_var()
            print(f"\n{'='*60}")
            print("VALUE-AT-RISK (95%, 1-day)")
            print(f"{'='*60}")
            if "error" in var_result:
                print(f"  Error: {var_result['error']}")
            else:
                print(f"\n  Portfolio VaR: {var_result['portfolio_var']:.4%}")
                print(f"  Diversification benefit: {var_result['diversification_benefit']:.1%}")
                print(f"\n  Per-commodity VaR:")
                for key, var_val in var_result["per_commodity_var"].items():
                    cfg = COMMODITIES.get(key)
                    name = cfg.name if cfg else key
                    print(f"    {name:<15} {var_val:.4%}")

        elif args.stress:
            stress_results = run_stress_tests()
            print(f"\n{'='*60}")
            print("STRESS TEST RESULTS")
            print(f"{'='*60}")
            for scenario, result in stress_results.items():
                pnl = result["portfolio_pnl"]
                print(f"\n  {scenario}:")
                print(f"    Portfolio P&L: {pnl:+.4%}")
                print(f"    Worst: {result['worst_commodity']}  Best: {result['best_commodity']}")
                for commodity, cpnl in result["breakdown"].items():
                    if abs(cpnl) > 0.001:
                        print(f"      {commodity:<12} {cpnl:+.4%}")

        elif args.limits:
            limits_result = check_position_limits()
            print(f"\n{'='*60}")
            print("POSITION LIMIT CHECK")
            print(f"{'='*60}")
            for key, check in limits_result.items():
                status = "OK" if check["within_limits"] else "BREACH"
                print(f"  {check['commodity']:<15} [{status}]  "
                      f"utilization={check['utilization_pct']:.1f}%  "
                      f"limit={check['cftc_limit']:,}")

        elif args.correlation:
            corr_result = compute_correlation_matrix()
            print(f"\n{'='*60}")
            print(f"CORRELATION MATRIX ({corr_result.get('window_days', 63)}-day)")
            print(f"{'='*60}")
            if "error" in corr_result:
                print(f"  Error: {corr_result['error']}")
            else:
                # Print matrix header
                keys = list(corr_result["matrix"].keys())
                header = "            " + "  ".join(f"{k[:6]:>6}" for k in keys)
                print(header)
                for row_key in keys:
                    vals = "  ".join(
                        f"{corr_result['matrix'][row_key][col_key]:>6.2f}"
                        for col_key in keys
                    )
                    print(f"  {row_key:<10} {vals}")

                if corr_result["high_correlation_flags"]:
                    print(f"\n  HIGH CORRELATION PAIRS:")
                    for flag in corr_result["high_correlation_flags"]:
                        print(f"    {flag['pair'][0]} / {flag['pair'][1]}: "
                              f"{flag['correlation']:.2f} [{flag['risk']}]")

        else:
            # Full risk report
            report = generate_risk_report()

            print(f"\n{'='*60}")
            print("FULL RISK REPORT")
            print(f"{'='*60}")

            # VaR
            var_r = report["var"]
            if "error" not in var_r:
                print(f"\n  VaR (95%, 1-day): {var_r['portfolio_var']:.4%}")
                print(f"  Diversification:  {var_r['diversification_benefit']:.1%}")

            # Drawdown
            dd = report["drawdown"]
            if "error" not in dd:
                breach_str = " ** BREACH **" if dd["breach"] else ""
                print(f"\n  Current drawdown: {dd['current_drawdown']:.2%}{breach_str}")
                print(f"  Max drawdown:     {dd['max_drawdown']:.2%}")

            # Stress test summary
            print(f"\n  STRESS SCENARIOS:")
            for scenario, result in report["stress_tests"].items():
                pnl = result["portfolio_pnl"]
                print(f"    {scenario:<30} {pnl:+.4%}")

            # Position limits
            breaches = [
                k for k, v in report["position_limits"].items()
                if not v["within_limits"]
            ]
            if breaches:
                print(f"\n  POSITION LIMIT BREACHES: {', '.join(breaches)}")
            else:
                print(f"\n  Position limits: all OK")

            # Correlation flags
            corr = report["correlation"]
            if "error" not in corr and corr["high_correlation_flags"]:
                print(f"\n  HIGH CORRELATION PAIRS:")
                for flag in corr["high_correlation_flags"]:
                    print(f"    {flag['pair'][0]} / {flag['pair'][1]}: {flag['correlation']:.2f}")

        # Finish DB logging
        if run_id:
            try:
                db.finish_agent_run(run_id, "ok", summary="Risk analysis complete")
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Risk agent failed: {e}", exc_info=True)
        if run_id:
            try:
                db.finish_agent_run(run_id, "error", summary=str(e))
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
