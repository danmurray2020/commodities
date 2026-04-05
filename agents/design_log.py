"""Design decisions log — shared utility for agents to record and challenge assumptions.

Any agent can call log_observation() to record findings, or log_challenge() to
flag when data contradicts an existing design assumption. This creates a living
record that persists across sessions.

Usage:
    from agents.design_log import log_observation, log_challenge, read_decisions

    # Record a finding
    log_observation("research", "Sugar trend features dominate — 70% of importance")

    # Challenge an assumption
    log_challenge("alpha_decay", "63-day horizon",
        "NatGas holdout accuracy dropped to 48% — shorter horizon may work better")

    # Read existing decisions for context
    decisions = read_decisions()
"""

import json
from datetime import datetime
from pathlib import Path

DESIGN_LOG = Path(__file__).parent.parent / "logs" / "design_decisions.md"
OBSERVATIONS_LOG = Path(__file__).parent.parent / "logs" / "agent_observations.jsonl"


def log_observation(agent: str, observation: str, commodity: str = None):
    """Log an observation from an agent run.

    Args:
        agent: Name of the agent (e.g., "research", "alpha_decay").
        observation: What was observed.
        commodity: Optional commodity this relates to.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "type": "observation",
        "commodity": commodity,
        "message": observation,
    }
    OBSERVATIONS_LOG.parent.mkdir(exist_ok=True)
    with open(OBSERVATIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_challenge(agent: str, decision: str, evidence: str, commodity: str = None):
    """Log a challenge to an existing design decision.

    Args:
        agent: Name of the agent raising the challenge.
        decision: Which decision is being challenged (e.g., "63-day horizon").
        evidence: Data or reasoning that contradicts the assumption.
        commodity: Optional commodity this relates to.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "type": "challenge",
        "decision": decision,
        "commodity": commodity,
        "evidence": evidence,
    }
    OBSERVATIONS_LOG.parent.mkdir(exist_ok=True)
    with open(OBSERVATIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_decisions() -> str:
    """Read the current design decisions document."""
    if DESIGN_LOG.exists():
        return DESIGN_LOG.read_text()
    return ""


def get_recent_observations(n: int = 20) -> list[dict]:
    """Read the most recent agent observations."""
    if not OBSERVATIONS_LOG.exists():
        return []
    entries = []
    with open(OBSERVATIONS_LOG) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-n:]


def get_challenges(decision: str = None) -> list[dict]:
    """Get all challenges, optionally filtered by decision name."""
    observations = get_recent_observations(n=1000)
    challenges = [o for o in observations if o.get("type") == "challenge"]
    if decision:
        challenges = [c for c in challenges if decision.lower() in c.get("decision", "").lower()]
    return challenges
