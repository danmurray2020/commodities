#!/bin/bash
# Heartbeat check — runs 1 hour after daily pipeline should complete
# Crontab: 0 19 * * 1-5 /Users/danielmurray/dev2/commodities-repo/scripts/heartbeat.sh

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)
DOW=$(date +%u)  # 1=Monday, 7=Sunday

# Only check on weekdays
if [ "$DOW" -gt 5 ]; then
    exit 0
fi

DAILY_LOG="$LOG_DIR/daily_${DATE}.txt"

if [ ! -f "$DAILY_LOG" ] || [ ! -s "$DAILY_LOG" ]; then
    echo "ALERT: Daily pipeline did not produce output for $DATE" | \
    $CLAUDE --print --permission-mode bypassPermissions \
      --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' \
      -p "Send an URGENT Slack DM to user ID U07BRUFVDDE: The daily commodities pipeline FAILED to run today ($DATE). The log file is missing or empty. Please check: 1) Is the Mac awake? 2) Run manually: cd $REPO_DIR && ./scripts/daily.sh"
fi
