"""Innovation Agent — researches external sources for model improvements.

Responsibilities:
- Search for new data sources (alternative data, fundamentals, sentiment)
- Find relevant academic papers and techniques
- Monitor commodity market news for structural changes
- Suggest concrete feature engineering ideas based on findings
- Track what's been tried vs what's new

Uses Claude CLI with web search to gather information, then logs
structured suggestions to the database.

Usage:
    python -m agents innovate                          # full research
    python -m agents innovate coffee --data-sources    # new data sources for coffee
    python -m agents innovate --techniques             # ML technique improvements
    python -m agents innovate --market-structure       # structural market changes
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import COMMODITIES, CommodityConfig, COMMODITIES_DIR
from .log import setup_logging


logger = setup_logging("innovation")

INNOVATION_DIR = COMMODITIES_DIR / "logs" / "innovation"
CLAUDE_PATH = Path.home() / ".local" / "bin" / "claude"


def _ask_claude(prompt: str, max_tokens: int = 2000) -> str:
    """Ask Claude CLI a question with web search enabled."""
    try:
        result = subprocess.run(
            [str(CLAUDE_PATH), "--print", "-p", prompt],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error: {result.stderr[-300:]}"
    except FileNotFoundError:
        return "Error: claude CLI not found"
    except subprocess.TimeoutExpired:
        return "Error: timeout"


def research_data_sources(cfg: CommodityConfig) -> dict:
    """Research new data sources that could improve predictions for a commodity."""
    # First, check what we already use
    meta_path = cfg.metadata_path
    current_features = []
    if meta_path.exists():
        with open(meta_path) as f:
            current_features = json.load(f).get("features", [])

    prompt = f"""Research new data sources that could improve ML predictions for {cfg.name} ({cfg.ticker}) commodity futures.

Current model uses these features: {', '.join(current_features[:10])}...
Current data sources: Yahoo Finance prices, CFTC COT positioning, Open-Meteo weather, NOAA ENSO indices.

Search the web for:
1. Alternative data sources for {cfg.name} commodity trading (sentiment, satellite, shipping, etc.)
2. Fundamental data APIs (production, inventory, consumption data)
3. Any free or low-cost APIs that provide relevant data

For each suggestion, provide:
- Data source name and URL
- What data it provides
- How it could be used as a feature
- Whether it's free or paid
- Expected impact (high/medium/low)

Return as a JSON array of objects with keys: name, url, description, feature_idea, cost, expected_impact.
Only return the JSON array, no other text."""

    response = _ask_claude(prompt)

    try:
        # Try to extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            suggestions = json.loads(response[start:end])
            return {"status": "ok", "suggestions": suggestions}
    except json.JSONDecodeError:
        pass

    return {"status": "ok", "raw_response": response}


def research_techniques(commodity_keys: list[str] = None) -> dict:
    """Research ML techniques and approaches that could improve the models."""
    prompt = """Research the latest ML techniques for commodity futures price prediction as of 2026.

Our current approach: XGBoost regressors + classifiers with 63-day horizon, walk-forward CV,
permutation importance feature selection, Kelly criterion position sizing.

Search the web for:
1. Recent papers on commodity price prediction using ML (2024-2026)
2. New feature engineering techniques for time series (beyond standard technicals)
3. Alternative model architectures that outperform gradient boosting for this task
4. Ensemble methods or stacking approaches
5. Online learning / adaptive techniques for non-stationary markets

For each finding, provide:
- Technique name
- Source/paper
- How it could be applied to our system
- Implementation complexity (easy/medium/hard)
- Expected improvement

Return as a JSON array of objects with keys: technique, source, application, complexity, expected_improvement.
Only return the JSON array, no other text."""

    response = _ask_claude(prompt)

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            techniques = json.loads(response[start:end])
            return {"status": "ok", "techniques": techniques}
    except json.JSONDecodeError:
        pass

    return {"status": "ok", "raw_response": response}


def research_market_structure() -> dict:
    """Research structural changes in commodity markets that could affect models."""
    prompt = """Research recent structural changes in commodity markets (2025-2026) that could affect
ML prediction models for coffee, cocoa, sugar, wheat, soybeans, copper, and natural gas.

Search the web for:
1. Regulatory changes affecting commodity futures (CFTC rules, margin requirements, position limits)
2. Major supply chain disruptions or shifts (new trade routes, sanctions, climate events)
3. Changes in market microstructure (new exchanges, electronic trading shifts, liquidity changes)
4. Emerging correlations or decorrelations between commodities
5. Geopolitical factors creating new price drivers

For each finding:
- What changed
- Which commodities are affected
- How it might affect our models
- Whether we need to adapt our features or strategy

Return as a JSON array of objects with keys: change, commodities_affected, model_impact, action_needed.
Only return the JSON array, no other text."""

    response = _ask_claude(prompt)

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            changes = json.loads(response[start:end])
            return {"status": "ok", "changes": changes}
    except json.JSONDecodeError:
        pass

    return {"status": "ok", "raw_response": response}


def run_innovation(
    commodity_keys: list[str] = None,
    data_sources: bool = True,
    techniques: bool = True,
    market_structure: bool = True,
) -> dict:
    """Run the full innovation research suite."""
    report = {"timestamp": datetime.now().isoformat(), "findings": {}}

    if data_sources:
        targets = commodity_keys or list(COMMODITIES.keys())
        report["findings"]["data_sources"] = {}
        for key in targets:
            cfg = COMMODITIES.get(key)
            if not cfg:
                continue
            logger.info(f"Researching data sources for {cfg.name}...")
            report["findings"]["data_sources"][key] = research_data_sources(cfg)

    if techniques:
        logger.info("Researching ML techniques...")
        report["findings"]["techniques"] = research_techniques(commodity_keys)

    if market_structure:
        logger.info("Researching market structure changes...")
        report["findings"]["market_structure"] = research_market_structure()

    # Save report
    INNOVATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = INNOVATION_DIR / f"innovation_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Innovation report saved to {report_path}")

    # Log to DB
    try:
        sys.path.insert(0, str(COMMODITIES_DIR.parent))
        from db import get_db
        db = get_db()
        run_id = db.start_agent_run("innovation", commodity_keys)
        n_findings = sum(
            len(v.get("suggestions", v.get("techniques", v.get("changes", []))))
            for v in report["findings"].values()
            if isinstance(v, dict) and "status" in v
        )
        db.finish_agent_run(run_id, "ok", summary=f"{n_findings} findings")
    except Exception:
        pass

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Innovation agent — research external improvements")
    parser.add_argument("commodities", nargs="*", help="Specific commodities (default: all)")
    parser.add_argument("--data-sources", action="store_true", help="New data sources only")
    parser.add_argument("--techniques", action="store_true", help="ML techniques only")
    parser.add_argument("--market-structure", action="store_true", help="Market changes only")
    args = parser.parse_args()

    run_all = not (args.data_sources or args.techniques or args.market_structure)

    report = run_innovation(
        commodity_keys=args.commodities or None,
        data_sources=args.data_sources or run_all,
        techniques=args.techniques or run_all,
        market_structure=args.market_structure or run_all,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("INNOVATION FINDINGS")
    print(f"{'='*60}")

    ds = report["findings"].get("data_sources", {})
    for key, result in ds.items():
        if isinstance(result, dict) and "suggestions" in result:
            cfg = COMMODITIES.get(key)
            print(f"\n--- {cfg.name if cfg else key}: New Data Sources ---")
            for s in result["suggestions"][:5]:
                print(f"  [{s.get('expected_impact', '?').upper()}] {s.get('name', '?')}")
                print(f"    {s.get('description', '')[:80]}")
                if s.get("url"):
                    print(f"    URL: {s['url']}")

    tech = report["findings"].get("techniques", {})
    if isinstance(tech, dict) and "techniques" in tech:
        print(f"\n--- ML Techniques ---")
        for t in tech["techniques"][:5]:
            print(f"  [{t.get('complexity', '?').upper()}] {t.get('technique', '?')}")
            print(f"    {t.get('application', '')[:80]}")

    ms = report["findings"].get("market_structure", {})
    if isinstance(ms, dict) and "changes" in ms:
        print(f"\n--- Market Structure Changes ---")
        for c in ms["changes"][:5]:
            print(f"  {c.get('change', '?')[:60]}")
            print(f"    Affects: {c.get('commodities_affected', '?')}")
            print(f"    Action: {c.get('action_needed', '?')[:60]}")


if __name__ == "__main__":
    main()
