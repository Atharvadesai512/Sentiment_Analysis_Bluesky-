"""
status.py

Health check for the collector. Run any time:

    python status.py              # looks at the last 48 hours
    python status.py --hours 168  # look further back (e.g. 7 days)

It reads the database (read-only, safe while the collector is running) and
tells you:

  * whether the collector looks ALIVE or STOPPED (age of the newest post)
  * every OUTAGE: a stretch of 5+ minutes with no posts. Normal Bluesky
    traffic always produces posts every few seconds, so a silent stretch
    means the collector was off (laptop asleep, window closed, crash...).
    This finds short outages that hide inside an hourly total.
  * how long it has been collecting WITHOUT interruption, against your goal
  * database size, backups size and free disk space

An hourly file can't show a collector that died halfway through an hour
(that hour just looks a bit low). This minute-level check can.
"""

import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from config import BACKUP_DIR, DATA_DIR, DB_PATH

OUTAGE_MINUTES = 5          # this many silent minutes in a row = outage
ALIVE_WITHIN_MINUTES = 5    # newest post at most this old = collector is alive
GOAL_HOURS = 7 * 24         # target length of an uninterrupted run


def fmt_duration(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def folder_size_gb(folder) -> float:
    if not folder.exists():
        return 0.0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e9


def parse_minute(text: str) -> datetime:
    """'2026-10-30T09:12' -> aware UTC datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def find_outages(minutes_with_posts: set, start: datetime, end: datetime):
    """
    Return [(first_silent_minute, last_silent_minute), ...] for every run of
    at least OUTAGE_MINUTES consecutive minutes (from start to end,
    inclusive) that contain no posts.
    """
    outages = []
    run_start = None
    t = start
    one = timedelta(minutes=1)
    while t <= end:
        if t.strftime("%Y-%m-%dT%H:%M") not in minutes_with_posts:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None:
                if (t - run_start) >= timedelta(minutes=OUTAGE_MINUTES):
                    outages.append((run_start, t - one))
                run_start = None
        t += one
    if run_start is not None and (end - run_start + one) >= timedelta(minutes=OUTAGE_MINUTES):
        outages.append((run_start, end))
    return outages


def main():
    hours = 48
    if "--hours" in sys.argv:
        try:
            hours = int(sys.argv[sys.argv.index("--hours") + 1])
        except (IndexError, ValueError):
            print("Usage: python status.py [--hours N]")
            return

    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}. The collector has not stored anything yet.")
        return

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    # Read-only connection: can never interfere with the collector.
    conn = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        total_rows, first_ts, last_ts = conn.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM sentiment_records"
        ).fetchone()
        if not total_rows:
            print("The database exists but has no posts yet.")
            return
        rows = conn.execute(
            "SELECT substr(timestamp, 1, 16) AS minute, COUNT(*) "
            "FROM sentiment_records WHERE timestamp >= ? GROUP BY minute",
            (since.strftime("%Y-%m-%dT%H:%M"),),
        ).fetchall()
    finally:
        conn.close()

    minutes_with_posts = {m for m, _ in rows}
    last_post = datetime.fromisoformat(last_ts)
    age = now - last_post

    print("=" * 62)
    print(f"Collector status at {now:%Y-%m-%d %H:%M} UTC")
    print("=" * 62)

    if age <= timedelta(minutes=ALIVE_WITHIN_MINUTES):
        print(f"STATUS: ALIVE   (newest post {fmt_duration(age)} ago)")
    else:
        print(f"STATUS: *** STOPPED ***   (newest post was {fmt_duration(age)} ago, at {last_post:%Y-%m-%d %H:%M} UTC)")
        print("        -> restart it:  .\\scripts\\start_monitor_background.ps1")

    print(f"Total stored posts: {total_rows:,}   (first: {first_ts[:16]}, last: {last_ts[:16]})")

    # ---- outages inside the window --------------------------------------
    first_post = datetime.fromisoformat(first_ts)
    window_start = max(since, first_post).replace(second=0, microsecond=0)
    window_end = (now - timedelta(minutes=1)).replace(second=0, microsecond=0)
    outages = find_outages(minutes_with_posts, window_start, window_end)

    print()
    print(f"Outages (no posts for {OUTAGE_MINUTES}+ minutes) in the last {hours} h:")
    if not outages:
        print("  none - no interruptions found")
    total_down = timedelta()
    for start, end in outages:
        length = end - start + timedelta(minutes=1)
        total_down += length
        ongoing = "   <-- still going" if end >= window_end else ""
        print(f"  {start:%Y-%m-%d %H:%M} -> {end:%H:%M} UTC   ({fmt_duration(length)}){ongoing}")
    if outages:
        print(f"  total downtime: {fmt_duration(total_down)}")

    # ---- current uninterrupted run ---------------------------------------
    if outages and outages[-1][1] >= window_end:
        print("\nCurrent uninterrupted run: 0m (collector is down right now)")
    else:
        run_start = outages[-1][1] + timedelta(minutes=1) if outages else window_start
        run = now - run_start
        qualifier = "" if outages or window_start > since else "at least "
        pct = min(100, run.total_seconds() / 3600 / GOAL_HOURS * 100)
        print(f"\nCurrent uninterrupted run: {qualifier}{fmt_duration(run)}"
              f"   ({pct:.0f}% of the {GOAL_HOURS // 24}-day goal)")
        if outages:
            print("  (the clock restarts after every outage - aim for a 7-day run)")

    # ---- disk ---------------------------------------------------------------
    free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    print()
    print(f"Database: {DB_PATH.stat().st_size / 1e9:.2f} GB   "
          f"Backups: {folder_size_gb(BACKUP_DIR):.2f} GB   Free disk: {free_gb:.1f} GB")
    if free_gb < 10:
        print("  WARNING: less than 10 GB free disk space")


if __name__ == "__main__":
    main()