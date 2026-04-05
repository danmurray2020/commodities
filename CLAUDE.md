# Commodities Trading System

ML-based prediction system for 7 commodities with automated agents.

## Structure

- `coffee/`, `chocolate/`, `sugar/`, `natgas/`, `soybeans/`, `wheat/`, `copper/` — Individual commodity projects (each has: features.py, train.py, strategy.py, fetch_*.py, data/, models/)
- `agents/` — Agent system: `python3 -m agents <command>` (refresh, predict, train, strategy, monitor, research, weekly, health)
- `tests/` — Test suite: `python3 -m pytest tests/ -v`
- `tools/` — Orchestration scripts (alert.py, dashboard.py, paper_trade.py, etc.)

## Key Commands

```bash
# From repo root:
python3 -m agents weekly                  # Full weekly pipeline
python3 -m agents predict                 # Generate predictions
python3 -m agents train --dry-run         # Evaluate without promoting
python3 -m agents research                # Feature stability + baselines
python3 -m agents health                  # Quick health check

# Per commodity:
cd coffee && python3 train.py             # Retrain coffee model
cd coffee && python3 refresh.py           # Refresh coffee data
```

## Model Architecture

- XGBoost regressors + classifiers per commodity
- 63-day prediction horizon (most commodities)
- Walk-forward CV with purge gap = horizon
- Validation split from training data for early stopping (not test fold)
- Feature selection on separate CV folds from training

## Data Sources

- Price: Yahoo Finance (yfinance)
- Positioning: CFTC Commitment of Traders
- Weather: Open-Meteo API
- Climate: NOAA ENSO indices (ONI, MEI)
