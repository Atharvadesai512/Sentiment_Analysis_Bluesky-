"""
db.py

All SQLite storage logic lives here, isolated from the real-time
streaming code (stream_monitor.py) and the aggregation code
(aggregate.py). Neither of those files talks to sqlite3 directly.

ONE STORED RECORD = one successfully classified post:

    {
        "timestamp":        ISO 8601 UTC string, e.g. "2026-08-15T10:03:41+00:00"
                             - when the post was received/processed,
        "text":              the original post text,
        "sentiment":         "Positive" | "Negative" | "Neutral",
        "confidence_score":  float between 0 and 1, the model's confidence,
    }

BACKUPS
-------
backup_database() makes a consistent copy of the live database using
SQLite's built-in online-backup API, so it is safe to run while the
collector is still writing. Copies go to data/backups/ and only the
newest BACKUP_KEEP_COUNT are kept.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import BACKUP_DIR, BACKUP_KEEP_COUNT, DATA_DIR, DB_PATH

logger = logging.getLogger("twitter-monitor")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_records (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    text              TEXT NOT NULL,
    sentiment         TEXT NOT NULL,
    confidence_score  REAL NOT NULL
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sentiment_records_timestamp
ON sentiment_records (timestamp)
"""

_BACKUP_PREFIX = "sentiment_"
_BACKUP_SUFFIX = ".db"


def init_db() -> None:
    """
    Create the data/ folder and the sentiment_records table if they don't
    exist yet. Safe to call every time the app starts - it won't wipe
    existing data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()
    logger.info("Database ready at %s", DB_PATH)


def insert_record(timestamp: str, text: str, sentiment: str, confidence_score: float) -> None:
    """
    Insert one classified post into SQLite.

    This function is synchronous/blocking on purpose. The caller
    (stream_monitor.py) runs it inside asyncio.to_thread() so a slow
    disk write never blocks the websocket event loop. The caller also
    wraps this in a try/except, so if a write ever fails (e.g. disk
    full, file locked), that failure is logged and the live stream
    keeps running instead of crashing.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute(
            "INSERT INTO sentiment_records (timestamp, text, sentiment, confidence_score) "
            "VALUES (?, ?, ?, ?)",
            (timestamp, text, sentiment, confidence_score),
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# BACKUP
# ============================================================

def _list_backups() -> list:
    """All completed backup files, oldest first (names sort by timestamp)."""
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob(f"{_BACKUP_PREFIX}*{_BACKUP_SUFFIX}"))


def seconds_since_last_backup() -> Optional[float]:
    """Age in seconds of the newest backup, or None if there are none."""
    backups = _list_backups()
    if not backups:
        return None
    newest_mtime = backups[-1].stat().st_mtime
    return max(0.0, datetime.now(timezone.utc).timestamp() - newest_mtime)


def backup_database() -> Optional[Path]:
    """
    Copy the live database to data/backups/sentiment_YYYYMMDD_HHMMSS.db
    and delete backups beyond the newest BACKUP_KEEP_COUNT.

    Uses SQLite's online-backup API (safe while the collector is writing)
    and writes to a temporary file first, so an interrupted backup never
    leaves a half-written file that looks like a good one.

    Returns the new backup's path, or None if there is no database yet.
    """
    if not DB_PATH.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    final_path = BACKUP_DIR / f"{_BACKUP_PREFIX}{stamp}{_BACKUP_SUFFIX}"
    temp_path = final_path.with_suffix(".tmp")

    source = sqlite3.connect(DB_PATH, timeout=30)
    target = sqlite3.connect(temp_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()

    os.replace(temp_path, final_path)
    _rotate_backups()
    return final_path


def _rotate_backups() -> None:
    """Delete the oldest backups so at most BACKUP_KEEP_COUNT remain."""
    if BACKUP_KEEP_COUNT <= 0:
        return
    backups = _list_backups()
    for old in backups[:-BACKUP_KEEP_COUNT]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not delete old backup %s: %s", old, exc)
