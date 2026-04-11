"""Inter-agent signal bus — typed, actionable messages between agents.

Any agent can emit a signal when it detects something that other agents
should know about. Consuming agents read signals and act on them.

Signal types:
    data_anomaly     — price gap, stale data, volume spike
    model_degraded   — accuracy dropped, alpha decayed
    feature_drift    — feature distribution shifted
    calibration_off  — predicted probabilities don't match reality
    baseline_beaten  — model underperforming simple baseline
    regime_change    — market regime shifted
    retraining_needed — model should be retrained

Usage:
    from agents.signals import emit_signal, get_signals, get_active_signals

    # Emit
    emit_signal("data_quality", "data_anomaly", "coffee",
        severity="high",
        detail="5.2% price gap detected on 2026-04-03")

    # Consume
    signals = get_active_signals("coffee", signal_type="data_anomaly")
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIGNALS_LOG = Path(__file__).parent.parent / "logs" / "agent_signals.jsonl"

VALID_TYPES = {
    "data_anomaly",
    "model_degraded",
    "feature_drift",
    "calibration_off",
    "baseline_beaten",
    "regime_change",
    "retraining_needed",
}

VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def emit_signal(
    source_agent: str,
    signal_type: str,
    commodity: str,
    severity: str = "medium",
    detail: str = "",
    metadata: dict = None,
):
    """Emit a signal for other agents to consume.

    Args:
        source_agent: Name of the agent emitting the signal.
        signal_type: One of the VALID_TYPES.
        commodity: Which commodity this relates to.
        severity: low, medium, high, critical.
        detail: Human-readable description.
        metadata: Additional structured data.
    """
    if signal_type not in VALID_TYPES:
        raise ValueError(f"Unknown signal type: {signal_type}. Valid: {VALID_TYPES}")
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}. Valid: {VALID_SEVERITIES}")

    signal = {
        "timestamp": datetime.now().isoformat(),
        "source": source_agent,
        "type": signal_type,
        "commodity": commodity,
        "severity": severity,
        "detail": detail,
        "metadata": metadata or {},
        "resolved": False,
    }

    SIGNALS_LOG.parent.mkdir(exist_ok=True)
    with open(SIGNALS_LOG, "a") as f:
        f.write(json.dumps(signal) + "\n")

    return signal


def resolve_signal(source_agent: str, signal_type: str, commodity: str, resolution: str = ""):
    """Mark a signal type as resolved for a commodity.

    Appends a resolution entry so consumers know the issue was addressed.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source_agent,
        "type": signal_type,
        "commodity": commodity,
        "resolved": True,
        "resolution": resolution,
    }
    SIGNALS_LOG.parent.mkdir(exist_ok=True)
    with open(SIGNALS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_signals(
    commodity: str = None,
    signal_type: str = None,
    since_hours: int = 168,  # 1 week
    min_severity: str = None,
) -> list[dict]:
    """Read all signals, optionally filtered.

    Args:
        commodity: Filter by commodity name.
        signal_type: Filter by signal type.
        since_hours: Only return signals from the last N hours.
        min_severity: Minimum severity (low < medium < high < critical).
    """
    if not SIGNALS_LOG.exists():
        return []

    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    min_sev = severity_order.get(min_severity, 0)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    signals = []
    with open(SIGNALS_LOG) as f:
        for line in f:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = datetime.fromisoformat(s.get("timestamp", "2000-01-01"))
            # Older log entries may be tz-naive — assume UTC for comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
            if commodity and s.get("commodity") != commodity:
                continue
            if signal_type and s.get("type") != signal_type:
                continue
            sev = severity_order.get(s.get("severity", "low"), 0)
            if sev < min_sev:
                continue

            signals.append(s)

    return signals


def get_active_signals(commodity: str = None, signal_type: str = None) -> list[dict]:
    """Get unresolved signals for a commodity.

    Returns signals emitted *after* the most recent resolution for that
    (type, commodity) pair.  Earlier signals — whether resolved explicitly
    or simply preceding a resolution entry — are excluded.
    """
    all_signals = get_signals(commodity=commodity, signal_type=signal_type)

    # Find the latest resolution timestamp per (type, commodity)
    latest_resolution: dict[tuple, str] = {}
    for s in all_signals:
        if s.get("resolved"):
            key = (s.get("type"), s.get("commodity"))
            ts = s.get("timestamp", "")
            if ts > latest_resolution.get(key, ""):
                latest_resolution[key] = ts

    # Return unresolved signals emitted after the latest resolution
    active = []
    for s in all_signals:
        if not s.get("resolved"):
            key = (s.get("type"), s.get("commodity"))
            resolution_ts = latest_resolution.get(key, "")
            if s.get("timestamp", "") > resolution_ts:
                active.append(s)

    return active


def get_signal_summary() -> dict:
    """Get a summary of active signals by commodity and type."""
    active = get_active_signals()
    summary = {}
    for s in active:
        c = s.get("commodity", "unknown")
        if c not in summary:
            summary[c] = []
        summary[c].append({
            "type": s["type"],
            "severity": s.get("severity"),
            "source": s.get("source"),
            "detail": s.get("detail", "")[:80],
        })
    return summary
