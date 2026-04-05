"""Entry point for `python -m agents <command>`.

Usage:
    python -m agents refresh [commodities...]     # Data Pipeline Agent
    python -m agents predict [commodities...]     # Prediction Agent
    python -m agents train [commodities...]       # Training Agent
    python -m agents strategy                     # Strategy/Risk Agent
    python -m agents monitor [--accuracy]         # Monitoring Agent
    python -m agents research [commodities...]    # Research Agent
    python -m agents innovate [commodities...]    # Innovation Agent (web research)
    python -m agents backtest [commodities...]    # Backtesting Agent
    python -m agents weekly [commodities...]      # Full weekly pipeline
    python -m agents health                       # Quick health check
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    # Remove the command from argv so sub-parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "refresh":
        from .data_pipeline import main as run
        run()
    elif command == "predict":
        from .prediction import main as run
        run()
    elif command == "train":
        from .training import main as run
        run()
    elif command == "strategy":
        from .strategy import main as run
        run()
    elif command == "monitor":
        from .monitoring import main as run
        run()
    elif command == "research":
        from .research import main as run
        run()
    elif command == "innovate":
        from .innovation import main as run
        run()
    elif command == "backtest":
        from .backtesting import main as run
        run()
    elif command in ("weekly", "run"):
        from .orchestrator import main as run
        run()
    elif command == "health":
        from .validation import run_system_health_check
        from .config import COMMODITIES
        import json
        results = run_system_health_check()
        for key, health in results.items():
            cfg = COMMODITIES[key]
            fresh = health["data_freshness"]
            model = health["model_files"]
            print(f"{cfg.name:<15} data={fresh.get('status','?'):<6} "
                  f"age={fresh.get('age_days','?')}d  "
                  f"model={model.get('status','?')}")
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
