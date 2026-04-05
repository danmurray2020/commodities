"""Database CLI — query and manage the commodities database.

Usage:
    python -m db status                    # Overview of all tables
    python -m db predictions [commodity]   # Recent predictions
    python -m db signals                   # Active signals
    python -m db accuracy [commodity]      # Prediction accuracy
    python -m db trades [commodity]        # Trade history & stats
    python -m db health                    # Data health status
    python -m db runs [agent]              # Agent run history
    python -m db models [commodity]        # Model registry
    python -m db sql "SELECT ..."          # Run arbitrary SQL
"""

import json
import sys
from .store import get_db


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    db = get_db()
    cmd = sys.argv[1]

    if cmd == "status":
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        # Whitelist valid table names to prevent SQL injection
        valid_tables = {"models", "predictions", "trades", "data_health", "agent_runs", "sqlite_sequence"}
        for t in tables:
            name = t["name"]
            if name not in valid_tables:
                continue
            count = db.execute(f"SELECT COUNT(*) as n FROM [{name}]")[0]["n"]
            print(f"  {name:<20} {count:>6} rows")

    elif cmd == "predictions":
        commodity = sys.argv[2] if len(sys.argv) > 2 else None
        preds = db.get_latest_predictions(commodity)
        if not preds:
            print("No predictions found.")
            return
        for p in preds:
            signal = "SIGNAL" if p["is_signal"] else "      "
            print(f"  {p['commodity']:<12} {p['direction']:>4} {p['confidence']:>6.1%} "
                  f"ret={p['pred_return']:+.2%}  ${p['price']:.2f}  {signal}  {p['as_of_date']}")

    elif cmd == "signals":
        since = sys.argv[2] if len(sys.argv) > 2 else None
        signals = db.get_signals(since)
        if not signals:
            print("No signals found.")
            return
        for s in signals:
            print(f"  {s['predicted_at'][:10]}  {s['commodity']:<12} "
                  f"{s['direction']:>4} {s['confidence']:>6.1%}  ret={s['pred_return']:+.2%}")

    elif cmd == "accuracy":
        commodity = sys.argv[2] if len(sys.argv) > 2 else None
        stats = db.get_prediction_accuracy(commodity)
        print(f"  Total verified: {stats['total']}")
        if stats["accuracy"] is not None:
            print(f"  Direction accuracy: {stats['accuracy']:.1%}")
        if stats.get("high_confidence_accuracy") is not None:
            print(f"  High-conf accuracy: {stats['high_confidence_accuracy']:.1%} "
                  f"({stats['high_confidence_total']} trades)")

    elif cmd == "trades":
        commodity = sys.argv[2] if len(sys.argv) > 2 else None
        stats = db.get_trade_stats(commodity)
        if stats["total_trades"] == 0:
            print("No closed trades.")
            return
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Win rate:     {stats['win_rate']:.1%}")
        print(f"  Avg win:      {stats['avg_win']:+.2%}")
        print(f"  Avg loss:     {stats['avg_loss']:+.2%}")
        print(f"  Total P&L:    {stats['total_pnl']:+.2%}")
        print(f"  Avg hold:     {stats['avg_hold_days']:.0f} days")

        open_trades = db.get_open_trades(commodity)
        if open_trades:
            print(f"\n  Open trades ({len(open_trades)}):")
            for t in open_trades:
                print(f"    {t['direction']} {t['commodity']} @ ${t['entry_price']:.2f} ({t['entry_date']})")

    elif cmd == "health":
        health = db.get_latest_health()
        if not health:
            print("No health data. Run `python -m agents refresh` first.")
            return
        for h in health:
            print(f"  {h['commodity']:<12} [{h['status']:<7}] "
                  f"age={h['age_days']}d  last={h['latest_data_date']}")

    elif cmd == "runs":
        agent = sys.argv[2] if len(sys.argv) > 2 else None
        runs = db.get_recent_runs(agent)
        if not runs:
            print("No agent runs recorded.")
            return
        for r in runs:
            duration = ""
            if r["finished_at"] and r["started_at"]:
                duration = f" ({r['finished_at'][:16]})"
            print(f"  {r['started_at'][:16]}  {r['agent_name']:<16} [{r['status']}]{duration}")
            if r["summary"]:
                print(f"    {r['summary']}")

    elif cmd == "models":
        commodity = sys.argv[2] if len(sys.argv) > 2 else None
        if commodity:
            models = db.get_model_history(commodity)
        else:
            models = db.execute("SELECT * FROM models ORDER BY trained_at DESC LIMIT 20")
        if not models:
            print("No models registered.")
            return
        for m in models:
            prod = " [PRODUCTION]" if m["is_production"] else ""
            print(f"  {m['commodity']:<12} {m['model_type']:<12} {m['version']:<6} "
                  f"acc={m['cv_avg_accuracy'] or '?':>6}  {m['trained_at'][:10]}{prod}")

    elif cmd == "sql":
        if len(sys.argv) < 3:
            print("Usage: python -m db sql \"SELECT ...\"")
            return
        query = sys.argv[2]
        # Only allow read-only queries
        if not query.strip().upper().startswith("SELECT"):
            print("Error: Only SELECT queries are allowed via CLI.")
            return
        rows = db.execute(query)
        for r in rows:
            print(json.dumps(r, default=str))

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
