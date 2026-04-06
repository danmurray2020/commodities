#!/bin/bash
# Weekly Full Pipeline — Saturday 9am
# Crontab: 0 9 * * 6 /Users/danielmurray/dev2/commodities-repo/scripts/weekly.sh
set -euo pipefail

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "[$DATE] Starting weekly pipeline..." >> "$LOG_DIR/cron.log"

# === FULL WEEKLY PIPELINE ===
# Orchestrator: refresh → predict → strategy → monitor
WEEKLY_OUTPUT=$(python3 -m agents weekly 2>&1)
echo "$WEEKLY_OUTPUT" > "$LOG_DIR/weekly_${DATE}.txt"

# === MODEL QUALITY ===
# Diagnose model issues and retrain with per-commodity configs
QUALITY_OUTPUT=$(python3 -m agents quality 2>&1)
echo "$QUALITY_OUTPUT" >> "$LOG_DIR/weekly_${DATE}.txt"

# === WEEKLY ANALYSIS ===
# Research baselines + regime + alpha decay + backtests + risk + compliance
ANALYSIS=$(
  python3 -m agents research --baselines 2>&1
  python3 -m agents regime 2>&1
  python3 -m agents alpha 2>&1
  python3 -m agents backtest 2>&1
  python3 -m agents risk 2>&1
  python3 -m agents execute --roll-calendar 2>&1
  python3 -m agents pnl --scenarios 2>&1
  python3 -m agents compliance 2>&1
  python3 -m agents infra 2>&1
  python3 -m agents baselines 2>&1
  python3 -m agents calibration 2>&1
  python3 -m agents drift 2>&1
  python3 -m agents data-quality 2>&1
  python3 -m agents confidence 2>&1
  python3 -m agents equities 2>&1
)
echo "$ANALYSIS" >> "$LOG_DIR/weekly_${DATE}.txt"

# === DB STATUS ===
DB_OUTPUT=$(python3 -m db predictions 2>&1 && python3 -m db health 2>&1)

# Send to Claude for Slack
cat <<EOF | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' -p \
  "Here is this week's full commodities pipeline output. Send a Slack DM to user ID U07BRUFVDDE with a weekly summary.

Format with sections: SIGNALS & P&L FORECAST, RISK, REGIME & ALPHA, BACKTESTS, COMPLIANCE, RECOMMENDATIONS.
Include position sizes and total exposure for signals.
Flag any model with alpha < 2% vs majority baseline.
Flag any commodity the alpha agent recommends suspending.
Include upcoming roll dates from execution agent.
Keep under 2000 characters." \
  >> "$LOG_DIR/cron.log" 2>&1
=== WEEKLY PIPELINE ===
$WEEKLY_OUTPUT

=== MODEL QUALITY ===
$QUALITY_OUTPUT

=== ANALYSIS ===
$ANALYSIS

=== DATABASE ===
$DB_OUTPUT
EOF

# === CLAUDE WEEKLY REVIEW ===
# Meta-agent reviews everything and sends Slack report
"$REPO_DIR/scripts/claude_weekly_review.sh" || true

echo "[$DATE] Weekly pipeline complete" >> "$LOG_DIR/cron.log"
