#!/bin/bash
# Monthly Retrain + Full Analysis — 1st of month 10am
# Crontab: 0 10 1 * * /Users/danielmurray/dev2/commodities-repo/scripts/monthly.sh
set -euo pipefail

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "[$DATE] Starting monthly retrain..." >> "$LOG_DIR/cron.log"

# === DATA REFRESH ===
python3 -m agents refresh >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === RETRAIN ALL MODELS ===
python3 -m agents train >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === FULL RESEARCH SUITE ===
python3 -m agents research >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === INNOVATION (web research for new ideas) ===
python3 -m agents innovate >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === FRESH PREDICTIONS WITH NEW MODELS ===
python3 -m agents predict >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === FULL BACKTESTS WITH NEW MODELS ===
python3 -m agents backtest --sweep >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === FULL RISK + COMPLIANCE REVIEW ===
python3 -m agents risk >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1
python3 -m agents compliance >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1
python3 -m agents alpha >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1
python3 -m agents regime >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1
python3 -m agents pnl --scenarios >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# === GIT COMMIT UPDATED MODELS ===
cd "$REPO_DIR"
git add */models/production_*.joblib */models/production_metadata.json
git commit -m "Monthly retrain $(date +%Y-%m-%d)" 2>/dev/null || true
git push 2>/dev/null || true

# === SEND SUMMARY VIA CLAUDE + SLACK ===
cat "$LOG_DIR/monthly_${DATE}.txt" | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' -p \
  "Here is the monthly retrain output for all 7 commodities. Send a Slack DM to user ID U07BRUFVDDE with:

RETRAIN RESULTS: Per commodity old vs new accuracy, improved Y/N. Flag any drops.
RESEARCH & INNOVATION: Top 3 new data source ideas, top 3 technique suggestions.
BACKTESTS: Strategy performance per commodity (win rate, Sharpe, return).
RISK: VaR, stress test results, any limit breaches.
ALPHA: Viability score per commodity, any recommended suspensions.
COMPLIANCE: Model governance status, any audit issues.
NEW PREDICTIONS: Active signals from freshly trained models.

Keep under 2000 characters." \
  >> "$LOG_DIR/cron.log" 2>&1

echo "[$DATE] Monthly retrain complete" >> "$LOG_DIR/cron.log"
