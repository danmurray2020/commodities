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

# Run the pipeline, capture output
OUTPUT=$(python3 -m agents refresh 2>&1 && python3 -m agents predict 2>&1 && python3 -m agents health 2>&1)

# Save raw output
echo "$OUTPUT" > "$LOG_DIR/daily_${DATE}.txt"

# Send to Claude to interpret and Slack
echo "$OUTPUT" | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__Slack__*' -p \
  "Here is today's commodities pipeline output. Summarize it and send a Slack DM to user ID U07BRUFVDDE with:
- Active signals (commodity, direction, confidence, predicted return)
- Any warnings or failures
- Data freshness status
Keep the message concise. Use the Slack MCP to send the DM." \
  >> "$LOG_DIR/cron.log" 2>&1

echo "[$DATE] Daily pipeline complete" >> "$LOG_DIR/cron.log"
