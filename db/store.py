"""SQLite-backed storage for the commodities trading system.

Thread-safe, auto-migrating, with a clean Python API.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


DB_DIR = Path(__file__).parent
SCHEMA_PATH = DB_DIR / "schema.sql"
DEFAULT_DB_PATH = DB_DIR / "commodities.db"

_local = threading.local()


class CommoditiesDB:
    """Main database interface for the commodities system."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            _local.conn.row_factory = sqlite3.Row
            _local.conn.execute("PRAGMA journal_mode=WAL")
            _local.conn.execute("PRAGMA foreign_keys=ON")
        return _local.conn

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        schema = SCHEMA_PATH.read_text()
        conn = self._get_conn()
        conn.executescript(schema)
        conn.commit()

    @contextmanager
    def _transaction(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Model Registry ─────────────────────────────────────────────

    def register_model(
        self,
        commodity: str,
        model_type: str,
        version: str,
        file_path: str,
        horizon: int = 63,
        features: list[str] = None,
        cv_metrics: dict = None,
        holdout_metrics: dict = None,
        params: dict = None,
        notes: str = None,
    ) -> int:
        """Register a trained model. Returns the model ID."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO models
                   (commodity, model_type, version, file_path, horizon,
                    n_features, features_json, cv_avg_accuracy, cv_std_accuracy,
                    cv_fold_accs, cv_avg_mae, cv_avg_rmse,
                    holdout_accuracy, holdout_mae, params_json, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    commodity, model_type, version, file_path, horizon,
                    len(features) if features else None,
                    json.dumps(features) if features else None,
                    cv_metrics.get("avg_accuracy") if cv_metrics else None,
                    cv_metrics.get("std_accuracy") if cv_metrics else None,
                    json.dumps(cv_metrics.get("fold_accuracies")) if cv_metrics else None,
                    cv_metrics.get("avg_mae") if cv_metrics else None,
                    cv_metrics.get("avg_rmse") if cv_metrics else None,
                    holdout_metrics.get("accuracy") if holdout_metrics else None,
                    holdout_metrics.get("mae") if holdout_metrics else None,
                    json.dumps(params) if params else None,
                    notes,
                ),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def promote_model(self, commodity: str, model_type: str, version: str):
        """Set a model as the current production model (demotes previous)."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE models SET is_production = 0 WHERE commodity = ? AND model_type = ?",
                (commodity, model_type),
            )
            conn.execute(
                """UPDATE models SET is_production = 1, promoted_at = datetime('now')
                   WHERE commodity = ? AND model_type = ? AND version = ?""",
                (commodity, model_type, version),
            )

    def get_production_model(self, commodity: str, model_type: str) -> dict | None:
        """Get the current production model for a commodity."""
        row = self._get_conn().execute(
            "SELECT * FROM models WHERE commodity = ? AND model_type = ? AND is_production = 1",
            (commodity, model_type),
        ).fetchone()
        return dict(row) if row else None

    def get_model_history(self, commodity: str, limit: int = 10) -> list[dict]:
        """Get model version history for a commodity."""
        rows = self._get_conn().execute(
            """SELECT * FROM models WHERE commodity = ?
               ORDER BY trained_at DESC LIMIT ?""",
            (commodity, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Predictions ────────────────────────────────────────────────

    def log_prediction(
        self,
        commodity: str,
        as_of_date: str,
        price: float,
        pred_return: float,
        direction: str,
        confidence: float,
        horizon_days: int = 63,
        threshold: float = None,
        is_signal: bool = False,
        model_version: str = None,
    ) -> int:
        """Log a prediction. Returns the prediction ID."""
        pred_price = price * (1 + pred_return)
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (commodity, as_of_date, model_version, price, pred_return,
                    pred_price, direction, confidence, horizon_days, threshold, is_signal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (commodity, as_of_date, model_version, price, pred_return,
                 pred_price, direction, confidence, horizon_days, threshold, is_signal),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_latest_predictions(self, commodity: str = None) -> list[dict]:
        """Get the most recent prediction per commodity."""
        if commodity:
            rows = self._get_conn().execute(
                """SELECT * FROM predictions WHERE commodity = ?
                   ORDER BY predicted_at DESC LIMIT 1""",
                (commodity,),
            ).fetchall()
        else:
            rows = self._get_conn().execute(
                """SELECT p.* FROM predictions p
                   INNER JOIN (
                       SELECT commodity, MAX(predicted_at) as max_dt
                       FROM predictions GROUP BY commodity
                   ) latest ON p.commodity = latest.commodity
                              AND p.predicted_at = latest.max_dt""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_signals(self, since: str = None) -> list[dict]:
        """Get predictions that crossed the confidence threshold."""
        query = "SELECT * FROM predictions WHERE is_signal = 1"
        params = []
        if since:
            query += " AND predicted_at >= ?"
            params.append(since)
        query += " ORDER BY predicted_at DESC"
        rows = self._get_conn().execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def backfill_realized(self, prediction_id: int, realized_price: float, realized_at: str):
        """Fill in the realized outcome for a past prediction."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT price, direction FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
            if not row:
                return

            realized_return = (realized_price / row["price"]) - 1
            actual_dir = "UP" if realized_return > 0 else "DOWN"
            correct = actual_dir == row["direction"]

            conn.execute(
                """UPDATE predictions
                   SET realized_price = ?, realized_return = ?,
                       direction_correct = ?, realized_at = ?
                   WHERE id = ?""",
                (realized_price, realized_return, correct, realized_at, prediction_id),
            )

    def get_prediction_accuracy(self, commodity: str = None, limit: int = 100) -> dict:
        """Calculate prediction accuracy from realized outcomes."""
        query = """SELECT commodity, direction, direction_correct, confidence
                   FROM predictions WHERE realized_price IS NOT NULL"""
        params = []
        if commodity:
            query += " AND commodity = ?"
            params.append(commodity)
        query += " ORDER BY predicted_at DESC LIMIT ?"
        params.append(limit)

        rows = self._get_conn().execute(query, params).fetchall()
        if not rows:
            return {"total": 0, "accuracy": None}

        total = len(rows)
        correct = sum(1 for r in rows if r["direction_correct"])
        high_conf = [r for r in rows if r["confidence"] >= 0.75]
        hc_correct = sum(1 for r in high_conf if r["direction_correct"])

        return {
            "total": total,
            "accuracy": round(correct / total, 4) if total else None,
            "high_confidence_total": len(high_conf),
            "high_confidence_accuracy": round(hc_correct / len(high_conf), 4) if high_conf else None,
        }

    # ── Trades ─────────────────────────────────────────────────────

    def open_trade(
        self,
        commodity: str,
        entry_date: str,
        entry_price: float,
        direction: str,
        position_size: float = 1.0,
        confidence: float = None,
        pred_return: float = None,
        trade_type: str = "paper",
    ) -> int:
        """Record a new trade entry. Returns trade ID."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO trades
                   (commodity, trade_type, entry_date, entry_price, direction,
                    position_size, confidence, pred_return)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (commodity, trade_type, entry_date, entry_price, direction,
                 position_size, confidence, pred_return),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def close_trade(
        self,
        trade_id: int,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
    ):
        """Record a trade exit with P&L calculation."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT entry_price, direction FROM trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
            if not row:
                return

            if row["direction"] == "LONG":
                pnl_pct = (exit_price / row["entry_price"]) - 1
            else:
                pnl_pct = (row["entry_price"] / exit_price) - 1

            entry = datetime.strptime(conn.execute(
                "SELECT entry_date FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()["entry_date"], "%Y-%m-%d")
            exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
            hold_days = (exit_dt - entry).days

            conn.execute(
                """UPDATE trades
                   SET exit_date = ?, exit_price = ?, exit_reason = ?,
                       hold_days = ?, pnl_pct = ?
                   WHERE id = ?""",
                (exit_date, exit_price, exit_reason, hold_days, round(pnl_pct, 6), trade_id),
            )

    def get_open_trades(self, commodity: str = None) -> list[dict]:
        """Get all currently open trades."""
        query = "SELECT * FROM trades WHERE exit_date IS NULL"
        params = []
        if commodity:
            query += " AND commodity = ?"
            params.append(commodity)
        return [dict(r) for r in self._get_conn().execute(query, params).fetchall()]

    def get_trade_stats(self, commodity: str = None, trade_type: str = "paper") -> dict:
        """Calculate trade statistics."""
        query = "SELECT * FROM trades WHERE exit_date IS NOT NULL AND trade_type = ?"
        params = [trade_type]
        if commodity:
            query += " AND commodity = ?"
            params.append(commodity)

        rows = self._get_conn().execute(query, params).fetchall()
        if not rows:
            return {"total_trades": 0}

        pnls = [r["pnl_pct"] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(rows),
            "win_rate": round(len(wins) / len(rows), 4) if rows else 0,
            "avg_win": round(sum(wins) / len(wins), 4) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0,
            "total_pnl": round(sum(pnls), 4),
            "best_trade": round(max(pnls), 4),
            "worst_trade": round(min(pnls), 4),
            "avg_hold_days": round(sum(r["hold_days"] for r in rows) / len(rows), 1),
        }

    # ── Data Health ────────────────────────────────────────────────

    def log_data_health(
        self,
        commodity: str,
        latest_data_date: str,
        age_days: int,
        status: str,
        fetch_results: dict = None,
        notes: str = None,
    ):
        """Log a data health check result."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO data_health
                   (commodity, latest_data_date, age_days, status, fetch_results, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (commodity, latest_data_date, age_days, status,
                 json.dumps(fetch_results) if fetch_results else None, notes),
            )

    def get_latest_health(self) -> list[dict]:
        """Get latest health check per commodity."""
        rows = self._get_conn().execute(
            """SELECT h.* FROM data_health h
               INNER JOIN (
                   SELECT commodity, MAX(checked_at) as max_dt
                   FROM data_health GROUP BY commodity
               ) latest ON h.commodity = latest.commodity
                          AND h.checked_at = latest.max_dt""",
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Agent Runs ─────────────────────────────────────────────────

    def start_agent_run(self, agent_name: str, commodities: list[str] = None) -> int:
        """Record the start of an agent run. Returns run ID."""
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO agent_runs (agent_name, commodities) VALUES (?, ?)",
                (agent_name, json.dumps(commodities) if commodities else None),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def finish_agent_run(self, run_id: int, status: str, summary: str = None, report: dict = None):
        """Record the completion of an agent run."""
        with self._transaction() as conn:
            conn.execute(
                """UPDATE agent_runs
                   SET finished_at = datetime('now'), status = ?,
                       summary = ?, report_json = ?
                   WHERE id = ?""",
                (status, summary, json.dumps(report, default=str) if report else None, run_id),
            )

    def get_recent_runs(self, agent_name: str = None, limit: int = 20) -> list[dict]:
        """Get recent agent runs."""
        query = "SELECT * FROM agent_runs"
        params = []
        if agent_name:
            query += " WHERE agent_name = ?"
            params.append(agent_name)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._get_conn().execute(query, params).fetchall()]

    # ── Utilities ──────────────────────────────────────────────────

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """Run an arbitrary SELECT query."""
        rows = self._get_conn().execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        """Close the thread-local connection."""
        if hasattr(_local, "conn") and _local.conn:
            _local.conn.close()
            _local.conn = None


# ── Module-level convenience ───────────────────────────────────────────

_default_db: CommoditiesDB | None = None


def get_db(db_path: Path = None) -> CommoditiesDB:
    """Get or create the default database instance."""
    global _default_db
    if _default_db is None or db_path:
        _default_db = CommoditiesDB(db_path)
    return _default_db
