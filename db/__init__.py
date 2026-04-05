"""Database layer for the commodities trading system.

Provides SQLite-backed storage for model metadata, predictions,
trades, and system health — replacing scattered JSON files.

Usage:
    from db import get_db
    db = get_db()
    db.log_prediction("coffee", direction="UP", confidence=0.82, ...)
    db.get_latest_predictions()
"""

from .store import CommoditiesDB, get_db

__all__ = ["CommoditiesDB", "get_db"]
