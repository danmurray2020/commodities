#!/bin/bash
# Weekly commodities check — run Saturday mornings
# Crontab: 0 9 * * 6 /Users/danielmurray/dev2/commodities/weekly_check.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "Weekly Commodities Check - $(date)"
echo "============================================"

for dir in coffee chocolate sugar natgas soybeans wheat copper; do
    if [ -d "$SCRIPT_DIR/../$dir" ]; then
        echo ""
        echo "--- Refreshing ${dir} data ---"
        cd "$SCRIPT_DIR/../$dir" && python3 refresh.py 2>&1 | tail -3
    fi
done

echo ""
cd "$SCRIPT_DIR" && python3 alert.py

echo ""
cd "$SCRIPT_DIR" && python3 paper_trade.py

echo ""
cd "$SCRIPT_DIR" && python3 predict_latest.py

echo ""
echo "============================================"
echo "Dashboard: cd $SCRIPT_DIR && python3 dashboard.py"
echo "  -> http://localhost:8060"
echo "============================================"
