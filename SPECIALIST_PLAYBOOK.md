# Commodity Specialist Playbook

When a commodity is below 70% accuracy, try these improvements IN ORDER.
Stop when accuracy hits 70%. Log every attempt in logs/design_decisions.md.

## General (apply to all commodities)

1. **Retrain with current features** — Optuna variance means retraining alone can shift accuracy ±5pp
2. **Try different horizon** — set QUALITY_HORIZON env var: `QUALITY_HORIZON=10 python3 tools/train_ensemble.py {key}`
3. **Try more Optuna trials** — `QUALITY_OPTUNA_TRIALS=300 python3 tools/train_ensemble.py {key}`
4. **Increase regularization** — `QUALITY_MIN_GAMMA=2.0 QUALITY_MIN_REG_ALPHA=0.1 python3 tools/train_ensemble.py {key}`
5. **Check feature stability** — if features are drifting, the model is extrapolating. Remove drifting features.

## Per-Commodity Specific Improvements

### Natural Gas (72% — at target)
- Maintain. Only retrain if accuracy drops below 70%.
- Possible improvement: add heating degree days (HDD) / cooling degree days (CDD) from NOAA

### Crude Oil (71% — at target)
- Maintain. Monitor OPEC meeting dates as potential feature.
- Possible: add crack spread (crude vs gasoline/heating oil price ratio)

### Wheat (65-71% — unstable)
- Priority: stabilize. Try fixing random_state across runs.
- Add USDA crop progress data (needs USDA_API_KEY env var)
- Try removing volatile features that cause fold instability

### Soybeans (70% — at target)
- Maintain. Add USDA crop progress data for potential improvement.
- Monitor soy-corn ratio stability

### Gold (68% — close)
- Add real yield features: fetch ^TNX (10Y yield) and ^TIP (TIPS ETF), compute real yield = TNX - breakeven
- Add DXY momentum (USD strength drives gold inversely)
- Gold responds to geopolitical risk — consider VIX as a fear proxy (already in supplementary data)

### Copper (67% — close)
- FRED macro data is integrated (housing starts, industrial production)
- Try fetching LME inventory data
- Add China property index if available
- copper_crude_ratio_zscore exists — verify it's being selected by feature importance

### Coffee (63% — needs work)
- Hardest commodity. Brazil frost events are binary and unpredictable.
- Try: pure momentum strategy (shorter horizons, follow the trend)
- Try: separate models for frost season (May-Aug) vs non-frost season
- Demand features (SBUX, NSRGY) exist — check if they're being selected
- Consider dropping and replacing with OJ (similar dynamics, might be easier)

### Cocoa (64% — needs work)
- Extreme regime changes (2024: +173%, 2025: -45%)
- Political risk features exist (election cycles, export ban season)
- Try: reduce max_depth to 3-4 (force simpler model that doesn't overfit to regimes)
- Try: train only on post-2020 data (regime shifted, old data may hurt)
- ACLED conflict data would help (needs ACLED_API_KEY)

### Sugar (65% — needs work)
- Best model is Ridge (linear) — suggests relationship is more linear than trees capture
- Try: increase Ridge weight in ensemble, or use Ridge as primary
- India export ban feature exists — verify it's being used
- ethanol_parity feature exists — this is the key driver
- Try: add Brazil sugarcane crush data from UNICA (free reports)

### Silver (62%)
- Highly correlated with gold. If gold model improves, silver should too.
- Add gold/silver ratio as a feature (mean-reverting)
- Add industrial demand proxy (silver has industrial uses unlike gold)

### Platinum (64%)
- Auto catalyst demand (60% of platinum goes to catalytic converters)
- Add auto sales data as a feature
- Platinum/gold ratio is mean-reverting — add as feature

### Corn (65%)
- USDA crop progress data would be the biggest improvement (needs API key)
- Ethanol mandate creates a floor price — add ethanol price as feature
- Highly seasonal — verify seasonal features are working

### Oats (65%)
- Thin market, low liquidity. May not be worth the effort.
- If can't reach 65% after 3 weeks, consider dropping.

### Live Cattle (65%)
- USDA Cattle on Feed report is a major driver (monthly)
- Try: add beef export data
- Seasonal: grilling season (May-Sep) increases demand

### Lean Hogs (65%)
- USDA Hogs and Pigs report (quarterly)
- African Swine Fever risk in Asia drives global supply
- Seasonal: winter holidays increase pork demand

### Cotton (63%)
- Driven by textile demand from China/India
- Try: add textile PMI or cotton-to-polyester price ratio
- USDA cotton stocks data would help

### Lumber (60%)
- US housing market is the primary driver
- FRED housing starts data is in copper — copy to lumber
- Very seasonal (spring building season)
- Low accuracy may be due to low liquidity / noise

### Orange Juice (60%)
- Florida weather dominates (hurricanes, frost)
- Similar dynamics to coffee — weather-driven soft commodity
- Try: add Florida citrus production data from USDA
- Small market, consider dropping if no improvement

### Heating Oil (60%)
- Highly correlated with crude oil
- Crack spread (heating oil vs crude) is the key feature to add
- Seasonal: winter demand spike
- If crude oil model is good, heating oil should follow

### Gasoline (60%)
- Also highly correlated with crude oil
- Crack spread (gasoline vs crude) feature needed
- Seasonal: summer driving season
- Same approach as heating oil
