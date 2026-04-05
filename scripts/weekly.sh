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

# Full weekly pipeline
WEEKLY_OUTPUT=$(python3 -m agents weekly 2>&1)
echo "$WEEKLY_OUTPUT" > "$LOG_DIR/weekly_${DATE}.txt"

# Research baselines
RESEARCH_OUTPUT=$(python3 -m agents research --baselines 2>&1)
echo "$RESEARCH_OUTPUT" > "$LOG_DIR/research_${DATE}.txt"

# DB status
DB_OUTPUT=$(python3 -m db predictions 2>&1 && python3 -m db health 2>&1)

# Combine and send to Claude for Slack
cat <<EOF | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' -p \
  "Here is this week's full commodities pipeline output. Send a Slack DM to user ID U07BRUFVDDE with a weekly summary.

Format with sections: SIGNALS, HEALTH, PERFORMANCE, RECOMMENDATIONS.
Include position sizes and total exposure for signals.
Flag any model with alpha < 2% vs majority baseline.
Keep under 2000 characters." \
  >> "$LOG_DIR/cron.log" 2>&1
=== WEEKLY PIPELINE ===
$WEEKLY_OUTPUT

=== RESEARCH ===
$RESEARCH_OUTPUT

=== DATABASE ===
$DB_OUTPUT
EOF

echo "[$DATE] Weekly pipeline complete" >> "$LOG_DIR/cron.log"
