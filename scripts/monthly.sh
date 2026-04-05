#!/bin/bash
# Monthly Retrain + Research — 1st of month 10am
# Crontab: 0 10 1 * * /Users/danielmurray/dev2/commodities-repo/scripts/monthly.sh
set -euo pipefail

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "[$DATE] Starting monthly retrain..." >> "$LOG_DIR/cron.log"

# Refresh data first
python3 -m agents refresh >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# Retrain all models (with backup)
python3 -m agents train >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# Full research
python3 -m agents research >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# Fresh predictions with new models
python3 -m agents predict >> "$LOG_DIR/monthly_${DATE}.txt" 2>&1

# Git commit updated models
cd "$REPO_DIR"
git add -A
git commit -m "Monthly retrain $(date +%Y-%m-%d)" 2>/dev/null || true
git push 2>/dev/null || true

# Send summary via Claude + Slack
cat "$LOG_DIR/monthly_${DATE}.txt" | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__Slack__*' -p \
  "Here is the monthly retrain output for all 7 commodities. Send a Slack DM to user ID U07BRUFVDDE with:

RETRAIN RESULTS: Per commodity old vs new accuracy, improved Y/N
RESEARCH: Unstable features, alpha vs baselines, market regimes, top 3 suggestions
NEW PREDICTIONS: Any active signals from freshly trained models

Flag any commodity where accuracy dropped." \
  >> "$LOG_DIR/cron.log" 2>&1

echo "[$DATE] Monthly retrain complete" >> "$LOG_DIR/cron.log"
