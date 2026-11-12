
"""
aggregate.py

Reads classified sentiment records from SQLite and produces
hourly and daily time-series datasets.

Updates:
1. Aggregate directly inside SQLite for better performance.
2. Export only the newest RAW_EXPORT_MAX_ROWS posts.
3. Display only the latest 24 hourly rows in the console.
4. Preserve all existing sentiment calculations.

Run:
    python aggregate.py
"""

import sqlite3
import pandas as pd

from config import (
    DB_PATH,
    HOURLY_CSV_PATH,
    DAILY_CSV_PATH,
    RAW_RECORDS_CSV_PATH,
    RAW_EXPORT_MAX_ROWS,
    DATA_DIR,
)

SENTIMENT_COLUMNS = ["Positive", "Negative", "Neutral"]

OUTPUT_COLUMNS = (
    ["Time"]
    + SENTIMENT_COLUMNS
    + [
        "Total",
        "Positive_Share",
        "Negative_Share",
        "Neutral_Share",
        "Net_Sentiment",
    ]
)


# ============================================================
# 1. COUNT RECORDS DIRECTLY IN SQLITE
# ============================================================

def load_counts(conn, frequency):
    """
    Aggregate sentiment counts inside SQLite.

    frequency:
        "h" = hourly
        "D" = daily

    Only grouped counts are loaded into pandas.
    """

    if frequency == "h":
        time_format = "%Y-%m-%d %H:00:00"
    elif frequency == "D":
        time_format = "%Y-%m-%d 00:00:00"
    else:
        raise ValueError("Frequency must be 'h' or 'D'")

    query = """
        SELECT
            strftime(?, timestamp) AS Time,

            SUM(
                CASE WHEN sentiment = 'Positive'
                THEN 1 ELSE 0 END
            ) AS Positive,

            SUM(
                CASE WHEN sentiment = 'Negative'
                THEN 1 ELSE 0 END
            ) AS Negative,

            SUM(
                CASE WHEN sentiment = 'Neutral'
                THEN 1 ELSE 0 END
            ) AS Neutral

        FROM sentiment_records

        WHERE timestamp IS NOT NULL
          AND strftime(?, timestamp) IS NOT NULL

        GROUP BY strftime(?, timestamp)
        ORDER BY Time
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(time_format, time_format, time_format),
    )


# ============================================================
# 2. CREATE HOURLY AND DAILY TABLES
# ============================================================

def aggregate_counts(counts, freq):
    """
    Create continuous hourly or daily time-series data.

    Missing time periods receive zero counts.
    Sentiment shares remain blank when Total = 0.
    """

    if counts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    counts = counts.copy()

    counts["Time"] = pd.to_datetime(
        counts["Time"],
        utc=True,
        errors="coerce",
    )

    counts = counts.dropna(subset=["Time"])

    if counts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    counts = counts.set_index("Time")

    for column in SENTIMENT_COLUMNS:
        if column not in counts.columns:
            counts[column] = 0

        counts[column] = (
            pd.to_numeric(
                counts[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int64")
        )

    counts = counts[SENTIMENT_COLUMNS]

    # Fill missing hours or days with zero counts.
    full_index = pd.date_range(
        start=counts.index.min(),
        end=counts.index.max(),
        freq=freq,
    )

    counts = counts.reindex(
        full_index,
        fill_value=0,
    )

    # Total posts in each period.
    counts["Total"] = counts[
        SENTIMENT_COLUMNS
    ].sum(axis=1)

    # Avoid division by zero.
    total = counts["Total"].where(
        counts["Total"] > 0
    )

    # Sentiment shares.
    counts["Positive_Share"] = (
        counts["Positive"] / total
    ).round(4)

    counts["Negative_Share"] = (
        counts["Negative"] / total
    ).round(4)

    counts["Neutral_Share"] = (
        counts["Neutral"] / total
    ).round(4)

    # Net sentiment ranges from -1 to +1.
    counts["Net_Sentiment"] = (
        (
            counts["Positive"]
            - counts["Negative"]
        ) / total
    ).round(4)

    counts.index.name = "Time"

    return counts.reset_index()[OUTPUT_COLUMNS]


# ============================================================
# 3. COUNT EMPTY PERIODS
# ============================================================

def count_empty_periods(table):
    """
    Count hours or days containing zero posts.
    """

    if table.empty:
        return 0

    return int(
        (table["Total"] == 0).sum()
    )


# ============================================================
# 4. EXPORT NEWEST POSTS ONLY
# ============================================================

def export_recent_posts(conn):
    """
    Export the newest RAW_EXPORT_MAX_ROWS posts.

    All records remain stored in SQLite.
    """

    query = """
        SELECT
            timestamp,
            text,
            sentiment,
            confidence_score

        FROM sentiment_records

        WHERE timestamp IS NOT NULL

        ORDER BY timestamp DESC

        LIMIT ?
    """

    raw_export = pd.read_sql_query(
        query,
        conn,
        params=(RAW_EXPORT_MAX_ROWS,),
    )

    if not raw_export.empty:
        raw_export["timestamp"] = pd.to_datetime(
            raw_export["timestamp"],
            utc=True,
            errors="coerce",
        )

    raw_export.to_csv(
        RAW_RECORDS_CSV_PATH,
        index=False,
    )

    return len(raw_export)


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DB_PATH.exists():
        print("Database not found.")
        print(f"Looked in: {DB_PATH}")
        print("Run stream_monitor.py first.")
        return

    conn = sqlite3.connect(DB_PATH)

    try:

        # Count all classified records.
        total_records = conn.execute(
            "SELECT COUNT(*) FROM sentiment_records"
        ).fetchone()[0]

        if total_records == 0:
            print("No records found yet in the database.")
            print(f"Looked in: {DB_PATH}")
            print("Run stream_monitor.py first.")
            return

        # Find earliest and latest timestamps.
        first_time, last_time = conn.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM sentiment_records
            """
        ).fetchone()

        # Aggregate directly in SQLite.
        hourly_counts = load_counts(
            conn,
            "h",
        )

        daily_counts = load_counts(
            conn,
            "D",
        )

        # Create gap-free time-series tables.
        hourly = aggregate_counts(
            hourly_counts,
            "h",
        )

        daily = aggregate_counts(
            daily_counts,
            "D",
        )

        # Save complete hourly and daily datasets.
        hourly.to_csv(
            HOURLY_CSV_PATH,
            index=False,
        )

        daily.to_csv(
            DAILY_CSV_PATH,
            index=False,
        )

        # Export only the newest posts.
        exported_rows = export_recent_posts(conn)

    finally:
        conn.close()

    # ========================================================
    # 6. DISPLAY RESULTS
    # ========================================================

    print()
    print(
        f"Total classified records in database: "
        f"{total_records:,}"
    )

    print(f"Database: {DB_PATH}")

    print(
        f"Data spans: {first_time} -> {last_time}"
    )

    print()

    print("Hourly sentiment (latest 24 rows)")
    print("-" * 100)

    print(
        hourly.tail(24).to_string(index=False)
    )

    print()

    print("Daily sentiment")
    print("-" * 100)

    print(
        daily.to_string(index=False)
    )

    print()

    # ========================================================
    # 7. EMPTY PERIODS
    # ========================================================

    empty_hours = count_empty_periods(hourly)
    empty_days = count_empty_periods(daily)

    if empty_hours or empty_days:
        print(
            f"Note: {empty_hours} hour(s) and "
            f"{empty_days} day(s) had no posts "
            "and were filled with zeros."
        )

    print(
        "Note: the last time period may still "
        "be in progress, so its counts are partial."
    )

    print()

    # ========================================================
    # 8. OUTPUT FILES
    # ========================================================

    print(f"Saved: {HOURLY_CSV_PATH}")
    print(f"Saved: {DAILY_CSV_PATH}")

    print(
        f"Saved: {RAW_RECORDS_CSV_PATH}"
    )

    print(
        f"Raw CSV contains {exported_rows:,} newest posts "
        f"(maximum {RAW_EXPORT_MAX_ROWS:,})."
    )

    print(
        f"All {total_records:,} classified records "
        "remain stored in SQLite."
    )

    print()
    print("Aggregation completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
