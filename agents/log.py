"""Structured logging for the commodities agent system."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import LOGS_DIR


def setup_logging(agent_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for an agent.

    Logs to both console (human-readable) and file (JSON lines).
    """
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(f"commodities.{agent_name}")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # Console handler — human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        f"%(asctime)s [{agent_name}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler — JSON lines for machine parsing
    log_file = LOGS_DIR / f"{agent_name}.jsonl"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter(agent_name))
    logger.addHandler(file_handler)

    return logger


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def log_event(logger: logging.Logger, message: str, data: dict = None, level: int = logging.INFO):
    """Log a structured event with optional data payload."""
    record = logger.makeRecord(
        logger.name, level, "(agent)", 0, message, (), None,
    )
    if data:
        record.data = data
    logger.handle(record)
