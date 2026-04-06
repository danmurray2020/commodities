#!/bin/bash
# Daily Commodities Pipeline — Weekdays 6pm
# Crontab: 0 18 * * 1-5 /Users/danielmurray/dev2/commodities-repo/scripts/daily.sh
set -euo pipefail

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "[$DATE] Starting daily pipeline..." >> "$LOG_DIR/cron.log"

# === CORE PIPELINE ===
# Data refresh → Predictions → Strategy → Equity signals → Execution costs → P&L forecast
OUTPUT=$(
  python3 -m agents refresh 2>&1
  python3 -m agents predict 2>&1
  python3 -m agents strategy 2>&1
  python3 -m agents equities 2>&1
  python3 -m agents execute 2>&1
  python3 -m agents pnl 2>&1
)

# === DAILY OVERSIGHT ===
# Data quality + risk + regime + drift + infrastructure health
OVERSIGHT=$(
  python3 -m agents data-quality 2>&1
  python3 -m agents drift 2>&1
  python3 -m agents risk --var 2>&1
  python3 -m agents regime --alert 2>&1
  python3 -m agents infra 2>&1
  python3 -m agents health 2>&1
)

# Save raw output
echo "$OUTPUT" > "$LOG_DIR/daily_${DATE}.txt"
echo "$OVERSIGHT" >> "$LOG_DIR/daily_${DATE}.txt"

# Send to Claude to interpret and Slack
cat <<EOF | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' -p \
  "Here is today's commodities pipeline output. Send a Slack DM to user ID U07BRUFVDDE with a summary.
Include: active signals with sizes, P&L forecast (expected/best/worst), risk warnings, regime alerts, any infra issues.
Keep it concise and well-formatted." \
  >> "$LOG_DIR/cron.log" 2>&1
=== PIPELINE ===
$OUTPUT

=== OVERSIGHT ===
$OVERSIGHT
EOF

echo "[$DATE] Daily pipeline complete" >> "$LOG_DIR/cron.log"
