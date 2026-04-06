# System Goals & Targets

## Performance Targets

### Model Accuracy
- **Target:** ≥70% direction accuracy (non-overlapping evaluation) per commodity
- **Current:** 3/20 at target (NatGas 74%, Wheat 71%, Soybeans 70%)
- **Near target:** Crude Oil 69%, Gold 68%, Copper 67%
- **Metric:** Direction accuracy on independent (non-overlapping) test observations

### Portfolio Returns
- **Target:** ≥20% annualized return on concentrated portfolio
- **Current:** 11.3% annualized (top-4 concentrated)
- **Benchmark:** S&P 500 ~10% annualized
- **Metric:** Annualized return net of transaction costs

### Confidence Calibration
- **Target:** Predicted confidence within ±10% of realized accuracy per bucket
  - When model says 80% confident → should be right 70-90% of the time
- **Current:** Not yet measured systematically
- **Metric:** Brier score and calibration curve per commodity

### Risk Management
- **Target:** Max drawdown ≤15% on portfolio
- **Current:** -9.2% max drawdown
- **Target:** Sharpe ratio ≥1.5
- **Current:** ~1.0

---

## Actionable Work Chunks

### Chunk 1: Get remaining models to 70%+ accuracy
- [ ] Crude Oil (69% → 70%): add term structure features (backwardation/contango)
- [ ] Gold (68% → 70%): add real yields (TIPS spread), USD strength features
- [ ] Copper (67% → 70%): get FRED housing starts integrated, add LME inventory
- [ ] Corn (65% → 70%): add USDA crop progress data
- [ ] Sugar (65% → 70%): add Brazil ethanol mix ratio from ANP
- [ ] Coffee (63% → 70%): try regime-switching model instead of single model
- [ ] Cocoa (64% → 70%): add ACLED conflict data, West African weather forecasts

### Chunk 2: Improve confidence calibration
- [ ] Train confidence meta-models for all commodities (not just top 3)
- [ ] Track live prediction accuracy vs predicted confidence
- [ ] Implement calibrated sizing: trade size = f(calibrated_confidence)
- [ ] Build calibration dashboard showing predicted vs realized accuracy

### Chunk 3: Add equity trading layer
- [ ] Build equity beta mapping (commodity → correlated stocks)
- [ ] Backtest equity trades alongside commodity trades
- [ ] Calculate optimal commodity/equity position split
- [ ] Add equity signals to daily pipeline and Slack alerts

### Chunk 4: Scale to 20+ trained commodities
- [ ] Train ensembles for remaining new commodities that don't have models yet
- [ ] Run portfolio simulation with all 20
- [ ] Identify which new commodities add diversification vs which are noise
- [ ] Drop commodities that can't reach 60% accuracy after 3 months

### Chunk 5: Per-commodity specialization
- [ ] Create per-commodity Claude agents that manage their own models
- [ ] Each agent: own data sources, own feature experiments, own training schedule
- [ ] Weekly review agent aggregates and makes portfolio decisions
- [ ] Commodity agents can challenge each other's assumptions

### Chunk 6: Live paper trading & validation
- [ ] Deploy paper trading for top-4 portfolio
- [ ] Track every prediction vs realized outcome
- [ ] Only trust models that prove themselves over 3+ months live
- [ ] Build live accuracy dashboard separate from backtest metrics
