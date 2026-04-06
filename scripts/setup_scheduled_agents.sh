#!/bin/bash
# Set up scheduled Claude agents for the commodities trading system.
#
# This creates:
# - 7 commodity specialist agents (top commodities, weekly Saturday morning)
# - 1 portfolio manager agent (weekly Saturday afternoon)
# - 1 data engineer agent (daily weekday evenings)
# - 1 risk manager agent (daily weekday evenings)
#
# Run once to set up. Use `claude schedule list` to verify.

REPO_DIR="/Users/danielmurray/dev2/commodities-repo"
CLAUDE="/Users/danielmurray/.local/bin/claude"

echo "Setting up scheduled Claude agents..."
echo "======================================"

# ── Commodity Specialist Agents (Saturday 8am-11am, staggered) ──

COMMODITIES=("natgas:Natural Gas:natgas:0 8" "wheat:Wheat:wheat:15 8" "crude_oil:Crude Oil:crude_oil:30 8" "gold:Gold:gold:45 8" "copper:Copper:copper:0 9" "coffee:Coffee:coffee:15 9" "cocoa:Cocoa:chocolate:30 9")

for entry in "${COMMODITIES[@]}"; do
    IFS=':' read -r KEY NAME DIR CRON_TIME <<< "$entry"
    MINUTE=$(echo $CRON_TIME | cut -d' ' -f1)
    HOUR=$(echo $CRON_TIME | cut -d' ' -f2)

    PROMPT=$(cat "$REPO_DIR/agents/prompts/commodity_specialist.md" | \
        sed "s/{COMMODITY}/$NAME/g" | \
        sed "s/{KEY}/$KEY/g" | \
        sed "s/{DIR}/$DIR/g")

    echo "  Creating $NAME specialist (Saturday ${HOUR}:${MINUTE})..."
    # Note: uncomment the claude command below when ready to activate
    # $CLAUDE schedule create \
    #   --name "${KEY}-specialist" \
    #   --cron "$MINUTE $HOUR * * 6" \
    #   --prompt "$PROMPT" \
    #   --allowedTools "Bash,Read,Write,Edit,Grep,Glob,mcp__claude_ai_Slack__slack_send_message" \
    #   --cwd "$REPO_DIR"
done

# ── Portfolio Manager Agent (Saturday 2pm — after all specialists) ──
echo "  Creating Portfolio Manager (Saturday 14:00)..."
# $CLAUDE schedule create \
#   --name "portfolio-manager" \
#   --cron "0 14 * * 6" \
#   --prompt "$(cat $REPO_DIR/agents/prompts/portfolio_manager.md)" \
#   --allowedTools "Bash,Read,Write,Edit,Grep,Glob,mcp__claude_ai_Slack__slack_send_message" \
#   --cwd "$REPO_DIR"

# ── Data Engineer Agent (Weekdays 5pm) ──
echo "  Creating Data Engineer (Weekdays 17:00)..."
# $CLAUDE schedule create \
#   --name "data-engineer" \
#   --cron "0 17 * * 1-5" \
#   --prompt "You are the data engineer for a commodities trading system at /Users/danielmurray/dev2/commodities-repo. Run: python3 -m agents data-quality && python3 -m agents drift. If any commodity has stale data (>3 days), refresh it: python3 -m agents refresh. Send a brief Slack DM to user ID U07BRUFVDDE only if there are issues." \
#   --allowedTools "Bash,Read,Grep,mcp__claude_ai_Slack__slack_send_message" \
#   --cwd "$REPO_DIR"

# ── Risk Manager Agent (Weekdays 6:30pm — after daily pipeline) ──
echo "  Creating Risk Manager (Weekdays 18:30)..."
# $CLAUDE schedule create \
#   --name "risk-manager" \
#   --cron "30 18 * * 1-5" \
#   --prompt "You are the risk manager for a commodities trading system at /Users/danielmurray/dev2/commodities-repo. Run: python3 -m agents risk && python3 -m agents health. Check that no single commodity exceeds 30% of portfolio, total drawdown is within 15% limit, and all models are functioning. Send a Slack DM to user ID U07BRUFVDDE only if there are risk warnings." \
#   --allowedTools "Bash,Read,Grep,mcp__claude_ai_Slack__slack_send_message" \
#   --cwd "$REPO_DIR"

echo ""
echo "Done. Agents are configured but COMMENTED OUT."
echo "Review the prompts in agents/prompts/ then uncomment the claude commands above to activate."
echo ""
echo "To list active schedules: claude schedule list"
echo "To test a specialist: claude -p \"\$(cat agents/prompts/commodity_specialist.md | sed 's/{COMMODITY}/Natural Gas/g; s/{KEY}/natgas/g; s/{DIR}/natgas/g')\""
