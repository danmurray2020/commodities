"""Data Pipeline Agent — fetches, validates, and refreshes all external data.

Responsibilities:
- Run fetch scripts for each commodity (price, COT, weather, ENSO)
- Validate data freshness and completeness
- Report failures without crashing the pipeline
- Log all fetch results for audit

Usage:
    python -m agents.data_pipeline              # refresh all
    python -m agents.data_pipeline coffee sugar  # refresh specific
"""

import subprocess
import sys
import time
from datetime import datetime

from .config import COMMODITIES, FETCH_SCRIPTS, CommodityConfig
from .validation import check_data_freshness, check_supplementary_data
from .log import setup_logging, log_event


logger = setup_logging("data_pipeline")


def fetch_commodity(cfg: CommodityConfig) -> dict:
    """Run all fetch scripts for a single commodity.

    Returns dict with per-script success/failure status.
    """
    results = {}
    for script in FETCH_SCRIPTS:
        script_path = cfg.project_dir / script
        if not script_path.exists():
            results[script] = {"status": "skipped", "reason": "file not found"}
            continue

        log_event(logger, f"Fetching {cfg.name}/{script}")
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True,
                cwd=str(cfg.project_dir),
                timeout=300,
            )
            elapsed = round(time.time() - start, 1)

            if result.returncode == 0:
                results[script] = {"status": "ok", "elapsed_s": elapsed}
            else:
                stderr_tail = result.stderr[-500:] if result.stderr else ""
                results[script] = {
                    "status": "error",
                    "returncode": result.returncode,
                    "stderr": stderr_tail,
                    "elapsed_s": elapsed,
                }
                logger.warning(f"{cfg.name}/{script} failed (rc={result.returncode}): {stderr_tail[:200]}")
        except subprocess.TimeoutExpired:
            results[script] = {"status": "timeout", "timeout_s": 300}
            logger.error(f"{cfg.name}/{script} timed out after 300s")
        except Exception as e:
            results[script] = {"status": "exception", "error": str(e)}
            logger.error(f"{cfg.name}/{script} exception: {e}")

    return results


def refresh_all(commodity_keys: list[str] = None) -> dict:
    """Refresh data for all (or specified) commodities.

    Returns a full report with fetch results and data validation.
    """
    targets = commodity_keys or list(COMMODITIES.keys())
    report = {
        "timestamp": datetime.now().isoformat(),
        "commodities": {},
    }

    for key in targets:
        cfg = COMMODITIES.get(key)
        if not cfg:
            logger.warning(f"Unknown commodity: {key}")
            continue

        logger.info(f"{'='*50}")
        logger.info(f"Refreshing {cfg.name} ({cfg.ticker})")
        logger.info(f"{'='*50}")

        fetch_results = fetch_commodity(cfg)
        freshness = check_data_freshness(cfg)

        commodity_report = {
            "fetch": fetch_results,
            "data_freshness": freshness,
            "all_fetches_ok": all(r["status"] == "ok" for r in fetch_results.values()),
        }

        # Summarize
        ok_count = sum(1 for r in fetch_results.values() if r["status"] == "ok")
        total = len(fetch_results)
        logger.info(f"{cfg.name}: {ok_count}/{total} fetches OK, data age: {freshness.get('age_days', '?')} days")

        if freshness.get("status") == "stale":
            logger.warning(f"{cfg.name} data is {freshness['age_days']} days old!")

        # Log to database
        try:
            from db import get_db
            get_db().log_data_health(
                commodity=key,
                latest_data_date=freshness.get("latest_date", ""),
                age_days=freshness.get("age_days", -1),
                status=freshness.get("status", "unknown"),
                fetch_results=fetch_results,
            )
        except Exception as e:
            logger.warning(f"DB health log failed: {e}")

        report["commodities"][key] = commodity_report

    # Summary
    all_ok = all(c["all_fetches_ok"] for c in report["commodities"].values())
    report["all_ok"] = all_ok

    if all_ok:
        logger.info("All data refreshed successfully")
    else:
        failed = [k for k, v in report["commodities"].items() if not v["all_fetches_ok"]]
        logger.warning(f"Fetch failures in: {', '.join(failed)}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Refresh commodity data")
    parser.add_argument("commodities", nargs="*", help="Specific commodities to refresh (default: all)")
    args = parser.parse_args()

    report = refresh_all(args.commodities or None)

    # Print summary
    print(f"\n{'='*60}")
    print("DATA PIPELINE SUMMARY")
    print(f"{'='*60}")
    for key, result in report["commodities"].items():
        cfg = COMMODITIES[key]
        status = "OK" if result["all_fetches_ok"] else "FAILED"
        age = result["data_freshness"].get("age_days", "?")
        print(f"  {cfg.name:<15} [{status}]  data age: {age} days")
    print(f"\nOverall: {'ALL OK' if report['all_ok'] else 'HAS FAILURES'}")


if __name__ == "__main__":
    main()
