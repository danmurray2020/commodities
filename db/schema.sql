-- Commodities Trading System — Database Schema
-- SQLite, auto-created on first use via db/store.py

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Model Registry ────────────────────────────────────────────────────
-- Tracks every model version with metrics and file locations.
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity       TEXT NOT NULL,                -- e.g. "coffee"
    model_type      TEXT NOT NULL,                -- "regressor" or "classifier"
    version         TEXT NOT NULL,                -- e.g. "v3", "production"
    file_path       TEXT NOT NULL,                -- relative path to .joblib
    horizon         INTEGER NOT NULL DEFAULT 63,
    n_features      INTEGER,
    features_json   TEXT,                         -- JSON array of feature names

    -- CV metrics
    cv_avg_accuracy REAL,
    cv_std_accuracy REAL,
    cv_fold_accs    TEXT,                         -- JSON array of fold accuracies
    cv_avg_mae      REAL,
    cv_avg_rmse     REAL,

    -- Holdout metrics
    holdout_accuracy REAL,
    holdout_mae      REAL,

    -- Hyperparameters
    params_json     TEXT,                         -- JSON dict of XGBoost params

    -- Metadata
    trained_at      TEXT NOT NULL DEFAULT (datetime('now')),
    promoted_at     TEXT,                         -- NULL if not yet promoted
    is_production   BOOLEAN NOT NULL DEFAULT 0,   -- 1 if currently in production
    notes           TEXT,

    UNIQUE(commodity, model_type, version)
);

CREATE INDEX IF NOT EXISTS idx_models_commodity ON models(commodity);
CREATE INDEX IF NOT EXISTS idx_models_production ON models(commodity, is_production);


-- ── Predictions ───────────────────────────────────────────────────────
-- Every prediction ever made, for audit and accuracy tracking.
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity       TEXT NOT NULL,
    predicted_at    TEXT NOT NULL DEFAULT (datetime('now')),
    as_of_date      TEXT NOT NULL,                -- data date used for prediction
    model_version   TEXT,

    -- Prediction
    price           REAL NOT NULL,
    pred_return     REAL NOT NULL,
    pred_price      REAL NOT NULL,
    direction       TEXT NOT NULL,                -- "UP" or "DOWN"
    confidence      REAL NOT NULL,
    horizon_days    INTEGER NOT NULL DEFAULT 63,

    -- Signal
    threshold       REAL,
    is_signal       BOOLEAN NOT NULL DEFAULT 0,

    -- Realized outcome (filled in later by monitoring)
    realized_price  REAL,
    realized_return REAL,
    direction_correct BOOLEAN,
    realized_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_commodity ON predictions(commodity, as_of_date);
CREATE INDEX IF NOT EXISTS idx_pred_signal ON predictions(is_signal, commodity);
CREATE INDEX IF NOT EXISTS idx_pred_unrealized ON predictions(realized_price) WHERE realized_price IS NULL;


-- ── Trades ────────────────────────────────────────────────────────────
-- Paper and live trade records.
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity       TEXT NOT NULL,
    trade_type      TEXT NOT NULL DEFAULT 'paper',  -- "paper" or "live"

    -- Entry
    entry_date      TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    direction       TEXT NOT NULL,                -- "LONG" or "SHORT"
    position_size   REAL NOT NULL DEFAULT 1.0,
    confidence      REAL,
    pred_return     REAL,

    -- Exit
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,                         -- "stop_loss", "take_profit", "time", "manual"
    hold_days       INTEGER,

    -- P&L
    pnl_pct         REAL,
    pnl_dollar      REAL,

    -- Metadata
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_commodity ON trades(commodity, entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(exit_date) WHERE exit_date IS NULL;


-- ── Data Health ───────────────────────────────────────────────────────
-- Tracks data freshness and fetch results over time.
CREATE TABLE IF NOT EXISTS data_health (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity       TEXT NOT NULL,
    checked_at      TEXT NOT NULL DEFAULT (datetime('now')),
    latest_data_date TEXT,
    age_days        INTEGER,
    status          TEXT NOT NULL,                -- "ok", "warning", "stale", "error"
    fetch_results   TEXT,                         -- JSON dict of per-script results
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_health_commodity ON data_health(commodity, checked_at);


-- ── Agent Runs ────────────────────────────────────────────────────────
-- Log of every agent execution for audit trail.
CREATE TABLE IF NOT EXISTS agent_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name      TEXT NOT NULL,                -- "data_pipeline", "prediction", etc.
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',  -- "running", "ok", "error"
    commodities     TEXT,                         -- JSON array of commodities processed
    summary         TEXT,                         -- Human-readable summary
    report_json     TEXT                          -- Full report as JSON
);

CREATE INDEX IF NOT EXISTS idx_runs_agent ON agent_runs(agent_name, started_at);
