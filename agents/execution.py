"""Execution Agent — models realistic trade execution costs and friction.

Sits between Strategy Agent's trade plan and actual/paper execution.
Adjusts returns for slippage, market impact, and roll costs.

Usage:
    python -m agents execute                    # cost analysis for current signals
    python -m agents execute --roll-calendar    # upcoming contract rolls
    python -m agents execute --liquidity        # liquidity analysis
"""

import json
import math
from datetime import datetime, date

import numpy as np
import pandas as pd

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR, LOGS_DIR
from .log import setup_logging, log_event


logger = setup_logging("execution")


# ── Roll calendars ────────────────────────────────────────────────────
ROLL_MONTHS = {
    "coffee":   [3, 5, 7, 9, 12],
    "cocoa":    [3, 5, 7, 9, 12],
    "sugar":    [3, 5, 7, 10],
    "natgas":   list(range(1, 13)),
    "soybeans": [1, 3, 5, 7, 8, 9, 11],
    "wheat":    [3, 5, 7, 9, 12],
    "copper":   list(range(1, 13)),
}

# Approximate per-roll cost by commodity (percent)
ROLL_COST_PCT = {
    "coffee":   0.15,
    "cocoa":    0.15,
    "sugar":    0.20,
    "natgas":   0.30,
    "soybeans": 0.10,
    "wheat":    0.15,
    "copper":   0.10,
}


def _load_price_data(cfg: CommodityConfig) -> pd.DataFrame | None:
    """Load combined_features.csv for a commodity, return None on failure."""
    csv_path = cfg.data_dir / "combined_features.csv"
    if not csv_path.exists():
        logger.warning(f"{cfg.name}: data file not found at {csv_path}")
        return None
    try:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as e:
        logger.error(f"{cfg.name}: failed to load data — {e}")
        return None


# ── 1. Slippage ───────────────────────────────────────────────────────

def estimate_slippage(cfg: CommodityConfig, trade_size_pct: float = 0.10) -> dict:
    """Estimate slippage based on average daily volume.

    Slippage model: base_slippage = 10bps, scaled by sqrt(trade_size / adv).

    Args:
        cfg: Commodity configuration.
        trade_size_pct: Trade size as fraction of average daily volume.

    Returns:
        Dict with estimated_slippage_pct, avg_daily_volume, liquidity_score.
    """
    base_slippage = 0.001  # 10 bps

    df = _load_price_data(cfg)
    if df is None or "Volume" not in df.columns:
        # No volume data — return base estimate
        logger.info(f"{cfg.name}: no volume data, using base slippage estimate")
        return {
            "commodity": cfg.name,
            "estimated_slippage_pct": round(base_slippage, 6),
            "avg_daily_volume": None,
            "liquidity_score": "unknown",
        }

    volume = df["Volume"].dropna()
    if volume.empty:
        return {
            "commodity": cfg.name,
            "estimated_slippage_pct": round(base_slippage, 6),
            "avg_daily_volume": None,
            "liquidity_score": "unknown",
        }

    avg_daily_volume = float(volume.mean())
    trade_size = trade_size_pct * avg_daily_volume

    scale_factor = math.sqrt(trade_size / avg_daily_volume) if avg_daily_volume > 0 else 1.0
    estimated_slippage = base_slippage * scale_factor

    # Liquidity score based on ADV
    if avg_daily_volume > 50_000:
        liquidity_score = "high"
    elif avg_daily_volume > 10_000:
        liquidity_score = "medium"
    else:
        liquidity_score = "low"

    return {
        "commodity": cfg.name,
        "estimated_slippage_pct": round(estimated_slippage, 6),
        "avg_daily_volume": round(avg_daily_volume, 0),
        "liquidity_score": liquidity_score,
    }


# ── 2. Roll cost ─────────────────────────────────────────────────────

def estimate_roll_cost(cfg: CommodityConfig) -> dict:
    """Estimate annualized roll cost for a futures commodity.

    Returns:
        Dict with next_roll_date, days_to_roll, estimated_roll_cost_pct,
        rolls_per_year, annual_roll_drag.
    """
    key = cfg.name.lower().replace(" ", "").replace("natural gas", "natgas")
    # Normalize key lookup
    for k in ROLL_MONTHS:
        if k in key or key in k:
            key = k
            break

    roll_months = ROLL_MONTHS.get(key, [3, 6, 9, 12])
    per_roll_cost = ROLL_COST_PCT.get(key, 0.15) / 100.0
    rolls_per_year = len(roll_months)

    today = date.today()
    current_year = today.year

    # Find next roll date (approximate: 15th of roll month)
    next_roll = None
    for year in [current_year, current_year + 1]:
        for month in roll_months:
            roll_date = date(year, month, 15)
            if roll_date > today:
                next_roll = roll_date
                break
        if next_roll:
            break

    days_to_roll = (next_roll - today).days if next_roll else 0
    annual_roll_drag = per_roll_cost * rolls_per_year

    return {
        "commodity": cfg.name,
        "next_roll_date": next_roll.isoformat() if next_roll else None,
        "days_to_roll": days_to_roll,
        "estimated_roll_cost_pct": round(per_roll_cost * 100, 2),
        "rolls_per_year": rolls_per_year,
        "annual_roll_drag": round(annual_roll_drag * 100, 4),
    }


# ── 3. Market impact ─────────────────────────────────────────────────

def estimate_market_impact(cfg: CommodityConfig, position_size_usd: float) -> dict:
    """Square-root market impact model.

    impact = 0.1 * sqrt(position_size / avg_daily_dollar_volume)

    Args:
        cfg: Commodity configuration.
        position_size_usd: Position size in USD.

    Returns:
        Dict with estimated_impact_pct.
    """
    df = _load_price_data(cfg)

    avg_daily_dollar_volume = None
    if df is not None and "Volume" in df.columns and cfg.price_col in df.columns:
        vol = df["Volume"].dropna()
        price = df[cfg.price_col].dropna()
        if not vol.empty and not price.empty:
            # Align and compute dollar volume
            aligned = pd.concat([vol, price], axis=1).dropna()
            if not aligned.empty:
                dollar_vol = aligned["Volume"] * aligned[cfg.price_col]
                avg_daily_dollar_volume = float(dollar_vol.mean())

    if avg_daily_dollar_volume is None or avg_daily_dollar_volume <= 0:
        # Fallback: assume $100M daily dollar volume
        avg_daily_dollar_volume = 100_000_000
        logger.info(f"{cfg.name}: no volume data, using fallback $100M daily dollar volume")

    impact = 0.1 * math.sqrt(position_size_usd / avg_daily_dollar_volume)

    return {
        "commodity": cfg.name,
        "position_size_usd": position_size_usd,
        "avg_daily_dollar_volume": round(avg_daily_dollar_volume, 0),
        "estimated_impact_pct": round(impact * 100, 4),
    }


# ── 4. Execution cost summary ────────────────────────────────────────

def compute_execution_cost_summary(cfg: CommodityConfig, trade_plan_signal: dict) -> dict:
    """Compute total execution friction for a trade signal.

    Total cost = slippage + half-spread + expected market impact.

    Args:
        cfg: Commodity configuration.
        trade_plan_signal: Dict with at least 'pred_return' and optionally
            'position_size_usd'.

    Returns:
        Dict with cost breakdown and adjusted_expected_return.
    """
    pred_return = trade_plan_signal.get("pred_return", 0.0)
    position_size_usd = trade_plan_signal.get("position_size_usd", 100_000)

    # Slippage
    slip = estimate_slippage(cfg)
    slippage_cost = slip["estimated_slippage_pct"]

    # Half-spread (approximate)
    half_spread = 0.0005  # 5 bps

    # Market impact
    impact = estimate_market_impact(cfg, position_size_usd)
    impact_cost = impact["estimated_impact_pct"] / 100.0

    # Roll cost (amortized per trade, approximate)
    roll = estimate_roll_cost(cfg)
    amortized_roll = (roll["annual_roll_drag"] / 100.0) / 252.0 * cfg.horizon

    total_cost = slippage_cost + half_spread + impact_cost + amortized_roll
    adjusted_return = pred_return - total_cost

    return {
        "commodity": cfg.name,
        "pred_return": round(pred_return, 6),
        "slippage_cost": round(slippage_cost, 6),
        "half_spread_cost": round(half_spread, 6),
        "market_impact_cost": round(impact_cost, 6),
        "amortized_roll_cost": round(amortized_roll, 6),
        "total_execution_cost": round(total_cost, 6),
        "adjusted_expected_return": round(adjusted_return, 6),
        "cost_as_pct_of_return": round(
            (total_cost / abs(pred_return) * 100) if pred_return != 0 else 0, 2
        ),
    }


# ── 5. Liquidity analysis ────────────────────────────────────────────

def analyze_liquidity(cfg: CommodityConfig) -> dict:
    """Analyze volume patterns and flag low-liquidity periods.

    Returns:
        Dict with avg_volume, volume_by_dow, low_liquidity_days.
    """
    df = _load_price_data(cfg)
    if df is None or "Volume" not in df.columns:
        logger.info(f"{cfg.name}: no volume data available for liquidity analysis")
        return {
            "commodity": cfg.name,
            "avg_volume": None,
            "volume_by_dow": None,
            "low_liquidity_days": None,
            "status": "no_volume_data",
        }

    volume = df["Volume"].dropna()
    if volume.empty:
        return {
            "commodity": cfg.name,
            "avg_volume": None,
            "volume_by_dow": None,
            "low_liquidity_days": None,
            "status": "empty_volume",
        }

    avg_volume = float(volume.mean())

    # Volume by day of week (0=Mon, 4=Fri)
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    volume_by_dow = {}
    for dow, name in dow_names.items():
        mask = volume.index.dayofweek == dow
        if mask.any():
            volume_by_dow[name] = round(float(volume[mask].mean()), 0)

    # Flag days with volume < 50% of average as low liquidity
    low_vol_threshold = avg_volume * 0.5
    low_days = volume[volume < low_vol_threshold]
    low_liquidity_days = len(low_days)
    low_liquidity_pct = round(low_liquidity_days / len(volume) * 100, 2) if len(volume) > 0 else 0

    return {
        "commodity": cfg.name,
        "avg_volume": round(avg_volume, 0),
        "volume_by_dow": volume_by_dow,
        "low_liquidity_days": low_liquidity_days,
        "low_liquidity_pct": low_liquidity_pct,
        "status": "ok",
    }


# ── 6. Main ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Execution cost analysis")
    parser.add_argument("--roll-calendar", action="store_true", help="Show upcoming contract rolls")
    parser.add_argument("--liquidity", action="store_true", help="Analyze volume / liquidity")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    args = parser.parse_args()

    targets = args.commodities or list(COMMODITIES.keys())

    # DB logging
    run_id = None
    try:
        from db import get_db
        db = get_db()
        run_id = db.start_agent_run("execution", targets)
    except Exception as e:
        logger.warning(f"DB logging unavailable: {e}")

    try:
        if args.roll_calendar:
            print(f"\n{'='*60}")
            print("ROLL CALENDAR")
            print(f"{'='*60}")
            for key in targets:
                cfg = COMMODITIES.get(key)
                if not cfg:
                    logger.warning(f"Unknown commodity: {key}")
                    continue
                roll = estimate_roll_cost(cfg)
                print(f"\n  {cfg.name}:")
                print(f"    Next roll:         {roll['next_roll_date']}")
                print(f"    Days to roll:      {roll['days_to_roll']}")
                print(f"    Per-roll cost:     {roll['estimated_roll_cost_pct']:.2f}%")
                print(f"    Rolls/year:        {roll['rolls_per_year']}")
                print(f"    Annual roll drag:  {roll['annual_roll_drag']:.2f}%")

        elif args.liquidity:
            print(f"\n{'='*60}")
            print("LIQUIDITY ANALYSIS")
            print(f"{'='*60}")
            for key in targets:
                cfg = COMMODITIES.get(key)
                if not cfg:
                    logger.warning(f"Unknown commodity: {key}")
                    continue
                liq = analyze_liquidity(cfg)
                print(f"\n  {cfg.name}:")
                if liq.get("status") != "ok":
                    print(f"    Status: {liq['status']}")
                    continue
                print(f"    Avg volume:        {liq['avg_volume']:,.0f}")
                if liq["volume_by_dow"]:
                    print(f"    Volume by DOW:     {liq['volume_by_dow']}")
                print(f"    Low-liq days:      {liq['low_liquidity_days']} ({liq['low_liquidity_pct']:.1f}%)")

        else:
            # Default: execution cost summary for current signals
            print(f"\n{'='*60}")
            print("EXECUTION COST ANALYSIS")
            print(f"{'='*60}")

            # Load latest predictions if available
            predictions_log = COMMODITIES_DIR / "logs" / "predictions.jsonl"
            latest_preds = {}
            if predictions_log.exists():
                with open(predictions_log) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            ckey = entry.get("commodity", "").lower().replace(" ", "")
                            for k in COMMODITIES:
                                if k in ckey or ckey in k:
                                    latest_preds[k] = entry
                                    break
                        except json.JSONDecodeError:
                            continue

            for key in targets:
                cfg = COMMODITIES.get(key)
                if not cfg:
                    continue

                pred = latest_preds.get(key, {"pred_return": 0.05})
                summary = compute_execution_cost_summary(cfg, pred)

                print(f"\n  {cfg.name}:")
                print(f"    Predicted return:      {summary['pred_return']:+.4%}")
                print(f"    Slippage:              {summary['slippage_cost']:.4%}")
                print(f"    Half-spread:           {summary['half_spread_cost']:.4%}")
                print(f"    Market impact:         {summary['market_impact_cost']:.4%}")
                print(f"    Amortized roll cost:   {summary['amortized_roll_cost']:.4%}")
                print(f"    Total cost:            {summary['total_execution_cost']:.4%}")
                print(f"    Adjusted return:       {summary['adjusted_expected_return']:+.4%}")
                if summary["pred_return"] != 0:
                    print(f"    Cost as % of return:   {summary['cost_as_pct_of_return']:.1f}%")

        # Finish DB logging
        if run_id:
            try:
                db.finish_agent_run(run_id, "ok", summary=f"Execution analysis for {len(targets)} commodities")
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Execution agent failed: {e}", exc_info=True)
        if run_id:
            try:
                db.finish_agent_run(run_id, "error", summary=str(e))
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
