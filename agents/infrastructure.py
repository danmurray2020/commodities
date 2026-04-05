"""Infrastructure Agent — system reliability and operational health.

Monitors cron execution, data pipeline health, model file integrity,
and alerts on system-level issues.

Usage:
    python -m agents infra                     # full infra check
    python -m agents infra --cron              # cron job status
    python -m agents infra --integrity         # model file checksums
    python -m agents infra --disk              # disk usage check
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from .config import COMMODITIES, COMMODITIES_DIR, LOGS_DIR
from .log import setup_logging, log_event


logger = setup_logging("infrastructure")

# ── DB access ─────────────────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db import get_db
    db = get_db()
except Exception:
    db = None


# ── Expected agent run windows (hours) ────────────────────────────────
EXPECTED_AGENT_WINDOWS = {
    "data_pipeline": 24,       # should run daily
    "prediction": 24,          # should run daily
    "monitoring": 48,          # should run at least every 2 days
    "strategy": 48,
    "training": 168,           # weekly
    "compliance": 168,
    "orchestrator": 168,
}

CHECKSUMS_PATH = COMMODITIES_DIR / "models" / "checksums.json"


def check_cron_execution() -> dict:
    """Parse cron.log and check that each pipeline ran within its expected window.

    Expected log format:
        [2026-04-05] Starting daily pipeline...
        [2026-04-05] Daily pipeline complete
    """
    cron_log = LOGS_DIR / "cron.log"
    if not cron_log.exists():
        return {"status": "no_log", "message": f"Cron log not found at {cron_log}"}

    try:
        content = cron_log.read_text()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # Parse log entries: extract date and message
    date_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]\s+(.*)")
    entries = []
    for line in content.splitlines():
        m = date_pattern.match(line.strip())
        if m:
            entries.append({"date": m.group(1), "message": m.group(2).strip()})

    # Identify script types by keywords
    script_patterns = {
        "daily": {
            "keywords": ["daily", "Daily"],
            "expected_window_days": 2,  # last business day
        },
        "weekly": {
            "keywords": ["weekly", "Weekly"],
            "expected_window_days": 8,  # last Saturday
        },
        "monthly": {
            "keywords": ["monthly", "Monthly"],
            "expected_window_days": 35,  # this month's 1st
        },
    }

    now = datetime.now()
    results = {}

    for script_name, spec in script_patterns.items():
        last_run = None
        for entry in reversed(entries):
            if any(kw in entry["message"] for kw in spec["keywords"]):
                try:
                    last_run = datetime.strptime(entry["date"], "%Y-%m-%d")
                except ValueError:
                    continue
                break

        if last_run is None:
            results[script_name] = {
                "status": "never_run",
                "last_run": None,
                "message": f"No {script_name} run found in cron log",
            }
        else:
            age_days = (now - last_run).days
            overdue = age_days > spec["expected_window_days"]
            results[script_name] = {
                "status": "overdue" if overdue else "ok",
                "last_run": last_run.strftime("%Y-%m-%d"),
                "age_days": age_days,
                "expected_window_days": spec["expected_window_days"],
            }

    logger.info(f"Cron execution check: {len(results)} scripts checked")
    return results


def verify_model_integrity() -> dict:
    """Compute SHA256 hashes of production model files and compare to stored checksums.

    If no stored checksums exist, create them. If a hash mismatches,
    flag the model as potentially corrupted.
    """
    # Load existing checksums
    stored_checksums = {}
    if CHECKSUMS_PATH.exists():
        try:
            stored_checksums = json.loads(CHECKSUMS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    current_checksums = {}
    results = {}

    for key, cfg in COMMODITIES.items():
        models_dir = cfg.models_dir
        if not models_dir.exists():
            results[key] = {
                "commodity": cfg.name,
                "status": "no_models_dir",
                "files_checked": 0,
            }
            continue

        model_files = list(models_dir.glob("*.joblib"))
        if not model_files:
            results[key] = {
                "commodity": cfg.name,
                "status": "no_model_files",
                "files_checked": 0,
            }
            continue

        file_hashes = {}
        mismatches = []
        for f in model_files:
            sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
            rel_path = str(f.relative_to(COMMODITIES_DIR))
            file_hashes[rel_path] = sha256
            current_checksums[rel_path] = sha256

            # Compare to stored
            if rel_path in stored_checksums and stored_checksums[rel_path] != sha256:
                mismatches.append(rel_path)

        status = "ok"
        if mismatches:
            status = "integrity_warning"
            logger.warning(f"{cfg.name}: hash mismatch for {mismatches}")

        results[key] = {
            "commodity": cfg.name,
            "status": status,
            "files_checked": len(model_files),
            "mismatches": mismatches,
            "hashes": file_hashes,
        }

    # Save current checksums
    try:
        CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS_PATH.write_text(json.dumps(current_checksums, indent=2))
    except OSError as e:
        logger.warning(f"Could not write checksums file: {e}")

    logger.info(f"Model integrity check: {len(results)} commodities, "
                f"{sum(r['files_checked'] for r in results.values())} files")
    return results


def _dir_size_mb(path: Path) -> float:
    """Compute total size of a directory in MB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def check_disk_usage() -> dict:
    """Check disk usage of key directories and warn on thresholds."""
    dirs_to_check = {
        "logs": LOGS_DIR,
    }

    # Add per-commodity data and models dirs
    all_data_size = 0.0
    all_models_size = 0.0
    for key, cfg in COMMODITIES.items():
        all_data_size += _dir_size_mb(cfg.data_dir)
        all_models_size += _dir_size_mb(cfg.models_dir)

    sizes = {
        "logs_mb": _dir_size_mb(LOGS_DIR),
        "data_mb": round(all_data_size, 2),
        "models_mb": round(all_models_size, 2),
    }
    sizes["total_mb"] = round(sizes["logs_mb"] + sizes["data_mb"] + sizes["models_mb"], 2)

    warnings = []
    if sizes["logs_mb"] > 100:
        warnings.append(f"Logs directory exceeds 100MB ({sizes['logs_mb']:.1f}MB)")
    if sizes["total_mb"] > 500:
        warnings.append(f"Total repo data exceeds 500MB ({sizes['total_mb']:.1f}MB)")

    status = "warning" if warnings else "ok"

    result = {
        "status": status,
        "sizes_mb": sizes,
        "warnings": warnings,
    }

    logger.info(f"Disk usage: total={sizes['total_mb']:.1f}MB, status={status}")
    return result


def check_agent_run_health() -> dict:
    """Query agent_runs table to check each agent ran within expected window."""
    if db is None:
        return {"status": "no_db", "message": "Database not available"}

    now = datetime.now()
    results = {}

    for agent_name, window_hours in EXPECTED_AGENT_WINDOWS.items():
        try:
            conn = db._get_conn()
            row = conn.execute(
                """SELECT started_at, status
                   FROM agent_runs
                   WHERE agent_name = ?
                   ORDER BY started_at DESC
                   LIMIT 1""",
                (agent_name,),
            ).fetchone()

            if row is None:
                results[agent_name] = {
                    "status": "never_run",
                    "last_run": None,
                    "expected_window_hours": window_hours,
                }
            else:
                try:
                    last_run = datetime.fromisoformat(row["started_at"])
                    age_hours = (now - last_run).total_seconds() / 3600
                    overdue = age_hours > window_hours
                    results[agent_name] = {
                        "status": "overdue" if overdue else "ok",
                        "last_run": row["started_at"],
                        "last_status": row["status"],
                        "age_hours": round(age_hours, 1),
                        "expected_window_hours": window_hours,
                    }
                except ValueError:
                    results[agent_name] = {
                        "status": "parse_error",
                        "last_run": row["started_at"],
                    }
        except Exception as e:
            results[agent_name] = {"status": "error", "message": str(e)}

    logger.info(f"Agent run health: {len(results)} agents checked")
    return results


def check_data_pipeline_reliability(days: int = 30) -> dict:
    """Check data pipeline success rate per commodity over the last N days.

    Queries the data_health table and flags any commodity with < 90%
    success rate.
    """
    if db is None:
        return {"status": "no_db", "message": "Database not available"}

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    results = {}

    for key, cfg in COMMODITIES.items():
        try:
            conn = db._get_conn()
            rows = conn.execute(
                """SELECT status FROM data_health
                   WHERE commodity = ? AND checked_at >= ?
                   ORDER BY checked_at DESC""",
                (key, cutoff),
            ).fetchall()

            if not rows:
                results[key] = {
                    "commodity": cfg.name,
                    "status": "no_data",
                    "checks": 0,
                }
                continue

            total = len(rows)
            successes = sum(1 for r in rows if r["status"] in ("ok", "warning"))
            rate = successes / total if total > 0 else 0.0

            results[key] = {
                "commodity": cfg.name,
                "checks": total,
                "successes": successes,
                "success_rate": round(rate, 4),
                "status": "ok" if rate >= 0.90 else "degraded",
            }
        except Exception as e:
            results[key] = {
                "commodity": cfg.name,
                "status": "error",
                "message": str(e),
            }

    logger.info(f"Pipeline reliability: {len(results)} commodities over {days} days")
    return results


def generate_infra_report() -> dict:
    """Generate a full infrastructure health report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "cron_execution": check_cron_execution(),
        "model_integrity": verify_model_integrity(),
        "disk_usage": check_disk_usage(),
        "agent_run_health": check_agent_run_health(),
        "pipeline_reliability": check_data_pipeline_reliability(),
    }

    # Overall status
    issues = []

    # Cron issues
    cron = report["cron_execution"]
    if isinstance(cron, dict):
        for script, info in cron.items():
            if isinstance(info, dict) and info.get("status") in ("overdue", "never_run"):
                issues.append(f"cron:{script} {info.get('status')}")

    # Integrity issues
    for key, info in report["model_integrity"].items():
        if isinstance(info, dict) and info.get("mismatches"):
            issues.append(f"integrity:{key} hash mismatch")

    # Disk issues
    disk = report["disk_usage"]
    if isinstance(disk, dict) and disk.get("warnings"):
        issues.extend(disk["warnings"])

    # Agent health
    agent_health = report["agent_run_health"]
    if isinstance(agent_health, dict):
        for agent, info in agent_health.items():
            if isinstance(info, dict) and info.get("status") in ("overdue", "never_run"):
                issues.append(f"agent:{agent} {info.get('status')}")

    # Pipeline reliability
    pipeline = report["pipeline_reliability"]
    if isinstance(pipeline, dict):
        for key, info in pipeline.items():
            if isinstance(info, dict) and info.get("status") == "degraded":
                issues.append(f"pipeline:{key} degraded ({info.get('success_rate', 0):.0%})")

    report["overall"] = {
        "status": "ok" if not issues else "needs_attention",
        "issue_count": len(issues),
        "issues": issues,
    }

    # Log to DB
    if db is not None:
        try:
            conn = db._get_conn()
            conn.execute(
                """INSERT INTO agent_runs (agent_name, started_at, finished_at, status, summary, report_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "infrastructure",
                    report["timestamp"],
                    datetime.now().isoformat(),
                    report["overall"]["status"],
                    f"{len(issues)} issues found" if issues else "All systems healthy",
                    json.dumps(report, default=str),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log infra run to DB: {e}")

    log_event(logger, "Infrastructure report generated", data=report["overall"])
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Infrastructure health checks")
    parser.add_argument("--cron", action="store_true", help="Cron job status only")
    parser.add_argument("--integrity", action="store_true", help="Model file checksums only")
    parser.add_argument("--disk", action="store_true", help="Disk usage check only")
    args = parser.parse_args()

    if args.cron:
        results = check_cron_execution()
        print(f"\n{'='*60}")
        print("CRON EXECUTION STATUS")
        print(f"{'='*60}")
        if results.get("status") in ("no_log", "error"):
            print(f"  {results['message']}")
            return
        for script, info in results.items():
            if not isinstance(info, dict):
                continue
            status_icon = {"ok": "OK", "overdue": "LATE", "never_run": "NONE"}.get(info["status"], "??")
            last = info.get("last_run", "never")
            age = info.get("age_days", "?")
            print(f"  {script:<12} [{status_icon}]  last={last}  age={age}d")
        return

    if args.integrity:
        results = verify_model_integrity()
        print(f"\n{'='*60}")
        print("MODEL INTEGRITY CHECK")
        print(f"{'='*60}")
        for key, info in results.items():
            if not isinstance(info, dict):
                continue
            status_icon = "OK" if info["status"] == "ok" else "WARN"
            print(f"  {info['commodity']:<15} [{status_icon}]  "
                  f"files={info['files_checked']}")
            if info.get("mismatches"):
                for m in info["mismatches"]:
                    print(f"    ^ MISMATCH: {m}")
        return

    if args.disk:
        result = check_disk_usage()
        print(f"\n{'='*60}")
        print("DISK USAGE")
        print(f"{'='*60}")
        sizes = result["sizes_mb"]
        print(f"  Logs:    {sizes['logs_mb']:>8.1f} MB")
        print(f"  Data:    {sizes['data_mb']:>8.1f} MB")
        print(f"  Models:  {sizes['models_mb']:>8.1f} MB")
        print(f"  Total:   {sizes['total_mb']:>8.1f} MB")
        if result["warnings"]:
            print(f"\n  WARNINGS:")
            for w in result["warnings"]:
                print(f"    ^ {w}")
        return

    # Full infrastructure report
    report = generate_infra_report()

    print(f"\n{'='*60}")
    print("INFRASTRUCTURE REPORT")
    print(f"{'='*60}")

    # Cron
    print("\nCRON EXECUTION:")
    cron = report["cron_execution"]
    if isinstance(cron, dict) and cron.get("status") not in ("no_log", "error"):
        for script, info in cron.items():
            if not isinstance(info, dict):
                continue
            status_icon = {"ok": "OK", "overdue": "LATE", "never_run": "NONE"}.get(info["status"], "??")
            print(f"  {script:<12} [{status_icon}]  last={info.get('last_run', 'never')}")
    else:
        print(f"  {cron.get('message', 'N/A')}")

    # Integrity
    print("\nMODEL INTEGRITY:")
    for key, info in report["model_integrity"].items():
        if not isinstance(info, dict):
            continue
        status_icon = "OK" if info["status"] == "ok" else "WARN"
        print(f"  {info['commodity']:<15} [{status_icon}]  files={info['files_checked']}")

    # Disk
    print("\nDISK USAGE:")
    disk = report["disk_usage"]
    if isinstance(disk, dict) and "sizes_mb" in disk:
        s = disk["sizes_mb"]
        print(f"  Total: {s['total_mb']:.1f}MB  (logs={s['logs_mb']:.1f} data={s['data_mb']:.1f} models={s['models_mb']:.1f})")

    # Agent health
    print("\nAGENT RUN HEALTH:")
    agent_health = report["agent_run_health"]
    if isinstance(agent_health, dict) and agent_health.get("status") != "no_db":
        for agent, info in agent_health.items():
            if not isinstance(info, dict):
                continue
            status_icon = {"ok": "OK", "overdue": "LATE", "never_run": "NONE"}.get(info.get("status"), "??")
            last = info.get("last_run", "never")
            print(f"  {agent:<18} [{status_icon}]  last={last}")
    else:
        print(f"  {agent_health.get('message', 'N/A')}")

    # Pipeline reliability
    print("\nPIPELINE RELIABILITY (30d):")
    pipeline = report["pipeline_reliability"]
    if isinstance(pipeline, dict) and pipeline.get("status") != "no_db":
        for key, info in pipeline.items():
            if not isinstance(info, dict):
                continue
            rate = info.get("success_rate")
            rate_str = f"{rate:.0%}" if rate is not None else "N/A"
            status_icon = {"ok": "OK", "degraded": "LOW", "no_data": "NONE"}.get(info.get("status"), "??")
            print(f"  {info['commodity']:<15} [{status_icon}]  rate={rate_str}  checks={info.get('checks', 0)}")
    else:
        print(f"  {pipeline.get('message', 'N/A')}")

    # Overall
    overall = report["overall"]
    print(f"\nOVERALL: [{overall['status'].upper()}]  ({overall['issue_count']} issues)")
    if overall["issues"]:
        for issue in overall["issues"]:
            print(f"  ^ {issue}")


if __name__ == "__main__":
    main()
