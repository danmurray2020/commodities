#!/bin/bash
# Claude Weekly Review Agent — runs after the weekly pipeline
# Reviews the entire system, quality logs, and signals, then sends a Slack report
# Crontab: 0 12 * * 6 /Users/danielmurray/dev2/commodities-repo/scripts/claude_weekly_review.sh
set -euo pipefail

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
LOG_DIR="$REPO_DIR/logs"
CLAUDE="/Users/danielmurray/.local/bin/claude"
DATE=$(date +%Y-%m-%d)

cd "$REPO_DIR"

echo "[$DATE] Starting Claude weekly review..." >> "$LOG_DIR/cron.log"

# Gather context for Claude to review
ACCURACY_REPORT=$(python3 tools/accuracy_report.py 2>&1)
QUALITY_LOG=$(tail -100 "$LOG_DIR/model_quality.jsonl" 2>/dev/null || echo "No quality log")
SIGNALS=$(tail -50 "$LOG_DIR/agent_signals.jsonl" 2>/dev/null || echo "No signals")
OBSERVATIONS=$(tail -50 "$LOG_DIR/agent_observations.jsonl" 2>/dev/null || echo "No observations")
DESIGN_DECISIONS=$(cat "$LOG_DIR/design_decisions.md" 2>/dev/null || echo "No design log")
CONFIGS=$(for c in coffee cocoa sugar natgas soybeans wheat copper; do echo "=== $c ===" && cat "$REPO_DIR/configs/$c.json" 2>/dev/null || echo "no config" && echo; done)

# Run Claude as the meta-agent to review everything and send Slack report
cat <<EOF | $CLAUDE --print --permission-mode bypassPermissions --allowedTools 'mcp__claude_ai_Slack__slack_send_message,mcp__claude_ai_Slack__slack_search_users' -p \
  "You are the weekly review agent for a commodities ML trading system. Review the data below and send a Slack DM to user ID U07BRUFVDDE.

Your job is to:
1. Assess overall system health — are models improving or degrading?
2. Identify which commodities have genuine predictive edge vs near-random
3. Review the model quality agent's decisions — are they making progress or going in circles?
4. Check if any agent signals need human attention
5. Recommend strategic changes (not just parameter tweaks)

Format the Slack message with sections:
## System Health
## Model Performance (table: commodity, accuracy, trend, status)
## Quality Agent Review (what it tried, what worked, what didn't)
## Signals & Alerts (anything needing human attention)
## Recommendations (strategic, not tactical)

Keep under 2500 characters. Be direct and honest — if a model has no edge, say so." \
  >> "$LOG_DIR/cron.log" 2>&1

=== ACCURACY REPORT ===
$ACCURACY_REPORT

=== MODEL QUALITY LOG (recent) ===
$QUALITY_LOG

=== AGENT SIGNALS (recent) ===
$SIGNALS

=== AGENT OBSERVATIONS (recent) ===
$OBSERVATIONS

=== PER-COMMODITY CONFIGS ===
$CONFIGS

=== DESIGN DECISIONS ===
$DESIGN_DECISIONS
EOF

echo "[$DATE] Claude weekly review complete" >> "$LOG_DIR/cron.log"
