
"""
config.py

Central configuration shared by stream_monitor.py, filters.py,
db.py, aggregate.py and backup.py.
"""

from pathlib import Path

# ============================================================
# BLUESKY JETSTREAM
# ============================================================

STREAM_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)

# ============================================================
# SENTIMENT MODEL
# ============================================================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_MAX_TOKENS = 128

LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

# ============================================================
# FILTERING - LANGUAGE AND JUNK
# ============================================================

TARGET_LANGUAGE_PREFIX = "en"

MIN_LATIN_RATIO = 0.6
LANG_VERIFY_MIN_CHARS = 25
LANG_VERIFY_MIN_CONFIDENCE = 0.95

LANGDETECT_MIN_CONFIDENCE = 0.90

MIN_MEANINGFUL_CHARS = 3

MIN_CONFIDENCE_TO_STORE = 0.0

# ============================================================
# FILTERING - DUPLICATES AND BOTS
# ============================================================

DEDUP_WINDOW_HOURS = 24
DEDUP_MIN_CHARS = 25
DEDUP_CACHE_MAX_ENTRIES = 250_000

MAX_POSTS_PER_AUTHOR_PER_HOUR = 20
MAX_POSTS_PER_AUTHOR_PER_MINUTE = 8

# ============================================================
# KNOWN BOT TEXT PATTERNS
# ============================================================

BOT_TEXT_PATTERNS = [
    # Flights and ships
    r"^\W*air to ground message",
    r"^\W*vesselalert",
    r"^\W*icao\s*:\s*[0-9a-f]{6}|\bplanefence\b|\badsbexchange\b",

    # Music and radio
    r"#nowplaying|^\W*now playing|\bnow playing on\b",
    r"\breproduciendo ahora\b|\btune in(?: now)?\s*:",

    # Weather alerts
    r"\b(continues|issues|cancels|expires|extends)\b.*\b(warning|advisory|watch)\b.*\[[a-z]{2}\]",

    r"^\W*(?-i:#?[A-Z]{3})\s+(?:issues|continues|cancels|expires|extends|updates)\b.*\b(?:warning|advisory|watch|discussion|statement|outlook)\b",

    # Earthquake feeds
    r"\bm\d(\.\d)?\b.*\b(quake|earthquake|erdbeben)\b|\b(quake|earthquake|erdbeben)\b.*\bm\d(\.\d)?\b",

    r"^\W*magnitude\s*:|\bepicenter\s*:",

    # Sports score bots
    r"\b(?:kickoff|full[- ]time|half[- ]time)\s*[—–-]|\bupset watch\s*:|\bwin probabilities\b",

    # Shopping, deals and advertisements
    r"\bnew low tracked\b|\breg\.\s*\$[\d,.]+|amazon\.com/dp/|\bvia @amazon\b|#ad\b",

    # Job feeds
    r"/jobs?/(?:job/)?\d+",
    r"\bobjkt\.com\b",

    # Historical event bots
    r"^\W*on (?:january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2},? \d{4},?\s+.{0,80}?\b(?:released|aired|premiered|debuted)\b",

    # Miscellaneous bots
    r"\bframe \d+ of \d+\b",
]

# ============================================================
# SAMPLING - NEW
# ============================================================

# 1.0 = process 100% of eligible posts
# 0.5 = process approximately 50%
# 0.25 = process approximately 25%
#
# Keep 1.0 for complete data collection.
# Sampling requires support in stream_monitor.py.

SAMPLE_RATE = 1.0

# ============================================================
# WEBSOCKET SETTINGS
# ============================================================

PING_INTERVAL = 20
PING_TIMEOUT = 60
CLOSE_TIMEOUT = 10

# ============================================================
# RECONNECT SETTINGS
# ============================================================

INITIAL_RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 60

# ============================================================
# MAINTENANCE
# ============================================================

STATS_LOG_INTERVAL_SECONDS = 300

# Automatic database backup every 24 hours
BACKUP_INTERVAL_HOURS = 24

# Keep only the latest 3 backups
BACKUP_KEEP_COUNT = 3

# ============================================================
# STORAGE
# ============================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = DATA_DIR / "sentiment.db"
BACKUP_DIR = DATA_DIR / "backups"

HOURLY_CSV_PATH = DATA_DIR / "hourly_sentiment.csv"
DAILY_CSV_PATH = DATA_DIR / "daily_sentiment.csv"
RAW_RECORDS_CSV_PATH = DATA_DIR / "all_classified_posts.csv"

# ============================================================
# CSV EXPORT LIMIT - NEW
# ============================================================

# Export only the newest 100,000 classified posts.
# All classified posts remain stored in SQLite.
#
# Requires aggregate.py to use this setting.

RAW_EXPORT_MAX_ROWS = 100_000

# ============================================================
# CREATE STORAGE DIRECTORIES
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
