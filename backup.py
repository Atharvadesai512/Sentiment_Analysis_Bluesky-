"""
backup.py

Make a backup of the sentiment database right now:

    python backup.py

You normally don't need this - stream_monitor.py already backs the
database up automatically every BACKUP_INTERVAL_HOURS while it runs
(see config.py). This script is for a manual copy, e.g. before you
delete or change anything. It is safe to run while the collector is
running.

Backups are saved in data/backups/ and only the newest
BACKUP_KEEP_COUNT are kept.

To restore: stop the collector, then copy the backup you want over
data/sentiment.db.
"""

import db
from config import BACKUP_DIR, DB_PATH


def main():
    path = db.backup_database()

    if path is None:
        print(f"No database found at {DB_PATH} - nothing to back up yet.")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Backup saved: {path}  ({size_mb:.2f} MB)")
    print(f"Backups are kept in: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
