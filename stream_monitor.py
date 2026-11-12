"""
stream_monitor.py

Real-time pipeline:

    Bluesky posts (Jetstream) -> text extraction -> filtering ->
    RoBERTa sentiment classification -> confidence gate ->
    timestamped SQLite storage

Filtering, in order (cheapest first):
    1. junk        - bare links / no real text
    2. bot text    - known bot templates (flight, quake, "now playing"...)
    3. language    - English only (declared "langs", else langdetect)
    4. author rate - one account posting implausibly often
    5. duplicates  - same long text seen recently

While running it also (a) logs a one-line summary of kept/skipped posts
every few minutes and (b) copies the database to data/backups/ every
BACKUP_INTERVAL_HOURS. Neither can crash the stream.

No forecasting, trend detection, or other time-series modeling happens
in this file or anywhere in this project stage.
"""

import asyncio
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import websockets
from langdetect import DetectorFactory, LangDetectException, detect_langs
from transformers import pipeline

# langdetect's underlying algorithm is probabilistic and non-deterministic
# by default (different runs can give slightly different results for the
# same short/ambiguous text). Seeding it makes results repeatable.
DetectorFactory.seed = 0

import db

from filters import (
    AuthorRateLimiter,
    DuplicateFilter,
    is_bare_domain,
    latin_letter_ratio,
    matches_bot_pattern,
    strip_urls,
)
from config import (
    STREAM_URL,
    MODEL_NAME,
    MODEL_MAX_TOKENS,
    LABEL_MAP,
    PING_INTERVAL,
    PING_TIMEOUT,
    CLOSE_TIMEOUT,
    INITIAL_RECONNECT_DELAY,
    MAX_RECONNECT_DELAY,
    TARGET_LANGUAGE_PREFIX,
    LANGDETECT_MIN_CONFIDENCE,
    MIN_LATIN_RATIO,
    LANG_VERIFY_MIN_CHARS,
    LANG_VERIFY_MIN_CONFIDENCE,
    MIN_MEANINGFUL_CHARS,
    MIN_CONFIDENCE_TO_STORE,
    DEDUP_WINDOW_HOURS,
    DEDUP_MIN_CHARS,
    DEDUP_CACHE_MAX_ENTRIES,
    MAX_POSTS_PER_AUTHOR_PER_HOUR,
    MAX_POSTS_PER_AUTHOR_PER_MINUTE,
    STATS_LOG_INTERVAL_SECONDS,
    BACKUP_INTERVAL_HOURS,
)


# ============================================================
# LOGGING
# ============================================================
# LOGGING
# ============================================================

from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_DIR / "stream_monitor.log",
            maxBytes=5_000_000,
            backupCount=2,
            encoding="utf-8"
        ),
    ],
)

logger = logging.getLogger("bluesky-sentiment-monitor")
# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

logger.info("Loading sentiment model...")

classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
)

logger.info("Sentiment model loaded successfully.")


# ============================================================
# FILTER STATE + STATISTICS
# ============================================================

author_limiter = AuthorRateLimiter(
    limits=[
        (MAX_POSTS_PER_AUTHOR_PER_HOUR, 3600),
        (MAX_POSTS_PER_AUTHOR_PER_MINUTE, 60),
    ]
)

duplicate_filter = DuplicateFilter(
    window_seconds=DEDUP_WINDOW_HOURS * 3600,
    min_chars=DEDUP_MIN_CHARS,
    max_entries=DEDUP_CACHE_MAX_ENTRIES,
)

# Counts of what happened to each post since the last stats log line.
stats = Counter()


# ============================================================
# SENTIMENT CLASSIFICATION
# ============================================================

def classify_text(text: str):
    """
    Run sentiment analysis.

    This function is intentionally synchronous. It is executed in a
    background thread (via asyncio.to_thread) so that Transformer
    inference does not block the asyncio event loop.

    Returns None on any failure - the caller treats that as "skip this
    post" rather than letting the error propagate and kill the stream.
    """

    if not text or not text.strip():
        return None

    try:
        result = classifier(text, truncation=True, max_length=MODEL_MAX_TOKENS)

        if not result:
            return None

        prediction = result[0]
        raw_label = prediction["label"].lower()

        # Map the model's raw label to our stored vocabulary
        # (Positive / Negative / Neutral). If the model ever returns an
        # unexpected label, we skip rather than store something unclear.
        sentiment = LABEL_MAP.get(raw_label)
        if sentiment is None:
            logger.warning("Unrecognized sentiment label from model: %s", raw_label)
            return None

        return {
            "sentiment": sentiment,
            "confidence_score": round(float(prediction["score"]), 4),
        }

    except Exception as exc:
        logger.exception("Sentiment classification failed: %s", exc)
        return None


# ============================================================
# EXTRACT POST DATA FROM STREAM MESSAGE
# ============================================================

def extract_post_data(message):
    """
    Try to extract a post's text, declared language(s) and author from
    the incoming stream message.

    Checks the Bluesky Jetstream shape first:
        {
          "did": "did:plc:...",            <- the author's account id
          "kind": "commit",
          "commit": {
            "operation": "create",
            "collection": "app.bsky.feed.post",
            "record": {
                "text": "...",
                "langs": ["en"]            <- optional, BCP-47 codes
            }
          }
        }

    Falls back to a few generic shapes so this still works if you ever
    point STREAM_URL at a different provider - those shapes have no
    language or author info, so "langs" and "author" come back as None.

    Returns {"text": str, "langs": list | None, "author": str | None},
    or None if no usable text was found.
    """

    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        # If the server sends plain text instead of JSON
        if message:
            return {"text": message, "langs": None, "author": None}
        return None

    if isinstance(data, str):
        if data.strip():
            return {"text": data, "langs": None, "author": None}
        return None

    if not isinstance(data, dict):
        return None

    # --- Bluesky Jetstream shape ---
    if data.get("kind") == "commit":
        commit = data.get("commit") or {}

        if (
            commit.get("operation") == "create"
            and commit.get("collection") == "app.bsky.feed.post"
        ):
            record = commit.get("record") or {}
            text = record.get("text")
            langs = record.get("langs")
            author = data.get("did")

            if isinstance(text, str) and text.strip():
                return {
                    "text": text,
                    "langs": langs if isinstance(langs, list) else None,
                    "author": author if isinstance(author, str) else None,
                }

        # Any other commit (likes, follows, deletes, etc.) has no
        # post text to classify.
        return None

    # --- Generic fallback shapes (other providers) ---
    possible_paths = [
        ("text",),
        ("content",),
        ("tweet", "text"),
        ("data", "text"),
        ("data", "content"),
        ("post", "text"),
        ("post", "content"),
    ]

    for path in possible_paths:
        current = data

        try:
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break

                current = current.get(key)

            if isinstance(current, str) and current.strip():
                return {"text": current, "langs": None, "author": None}

        except Exception:
            continue

    return None


# ============================================================
# FILTERING (language + junk; bot/duplicate logic lives in filters.py)
# ============================================================

WORD_CHAR_PATTERN = re.compile(r"\w", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """
    Convert "styled" Unicode letters to plain ones (NFKC), e.g.
    '𝘑𝘰𝘯𝘢𝘵𝘩𝘢𝘯' -> 'Jonathan'. Filtering and classification use this
    version; the ORIGINAL text is what gets stored.
    """
    return unicodedata.normalize("NFKC", text)


def _detect_top_language(text: str):
    """Return (language, probability) from langdetect, or None if it can't tell."""
    try:
        results = detect_langs(text)
    except LangDetectException:
        return None
    if not results:
        return None
    return results[0].lang, results[0].prob


def is_english_post(langs, text: str) -> bool:
    """
    1. Script check: too few Latin letters (Thai, Arabic, CJK, Cyrillic...)
       -> not English, even if the account says "en".
    2. Declared languages without English -> not English.
    3. Declared English: trusted, EXCEPT text >= LANG_VERIFY_MIN_CHARS long
       where langdetect is >= LANG_VERIFY_MIN_CONFIDENCE sure it is another
       language. Short text is trusted (detectors fail on "lol", "yes").
    4. Nothing declared: langdetect must be >= LANGDETECT_MIN_CONFIDENCE
       sure it is English.
    """
    if latin_letter_ratio(text) < MIN_LATIN_RATIO:
        return False

    if langs:
        declared_english = any(
            isinstance(lang, str) and lang.lower().startswith(TARGET_LANGUAGE_PREFIX)
            for lang in langs
        )
        if not declared_english:
            return False

        if len(text) >= LANG_VERIFY_MIN_CHARS:
            top = _detect_top_language(text)
            if top and top[0] != "en" and top[1] >= LANG_VERIFY_MIN_CONFIDENCE:
                return False
        return True

    top = _detect_top_language(text)
    return bool(top) and top[0] == "en" and top[1] >= LANGDETECT_MIN_CONFIDENCE


def is_junk_text(text: str) -> bool:
    """
    True if the post is essentially just a link (with or without
    "https://") or a bare domain, with little or no other content.
    """
    if is_bare_domain(text):
        return True
    without_urls = strip_urls(text)
    meaningful_chars = WORD_CHAR_PATTERN.findall(without_urls)
    return len(meaningful_chars) < MIN_MEANINGFUL_CHARS


def passes_filters(post_data, text: str) -> bool:
    """
    Run every pre-classification filter, cheapest first. `text` is the
    normalized text. Records the reason in `stats` when a post is rejected.
    """
    if is_junk_text(text):
        stats["skipped_junk"] += 1
        logger.debug("Skipped (junk - no real text): %s", text[:60])
        return False

    if matches_bot_pattern(text):
        stats["skipped_bot_pattern"] += 1
        logger.debug("Skipped (known bot template): %s", text[:60])
        return False

    if not is_english_post(post_data["langs"], text):
        stats["skipped_non_english"] += 1
        logger.debug("Skipped (non-English, langs=%s): %s", post_data["langs"], text[:60])
        return False

    if not author_limiter.allow(post_data["author"]):
        stats["skipped_bot_rate"] += 1
        logger.debug("Skipped (author over posting-rate limit): %s", text[:60])
        return False

    if duplicate_filter.is_duplicate(text):
        stats["skipped_duplicate"] += 1
        logger.debug("Skipped (duplicate text): %s", text[:60])
        return False

    return True
# ============================================================
# PERSIST ONE CLASSIFIED RECORD
# ============================================================

async def save_record(timestamp: str, text: str, sentiment: str, confidence_score: float):
    """
    Save one classified post to SQLite via db.py.

    Runs the (blocking) database write in a background thread so it
    never blocks the websocket event loop. Any failure is logged and
    swallowed here - a database hiccup must never crash the live
    stream or the sentiment pipeline.
    """
    try:
        await asyncio.to_thread(
            db.insert_record,
            timestamp,
            text,
            sentiment,
            confidence_score,
        )
        stats["stored"] += 1
    except Exception as exc:
        stats["store_failed"] += 1
        logger.exception("Failed to save record to database: %s", exc)


# ============================================================
# PROCESS ONE MESSAGE
# ============================================================

async def process_message(message):
    """
    Process one incoming stream message: extract text, filter, classify
    sentiment, and store the result (if it passes the confidence gate).
    A failure at any step (bad message, failed classification, failed DB
    write) is logged and skipped - it never stops the stream.
    """

    post_data = extract_post_data(message)

    if not post_data:
        logger.debug("Received message but could not extract text.")
        return

    stats["received"] += 1

    text = post_data["text"]                 # original - this is what gets stored
    clean_text = normalize_text(text)        # used for filtering and classification

    if not passes_filters(post_data, clean_text):
        return

    logger.info("")
    logger.info("POST: %s", text)

    start_time = time.perf_counter()

    sentiment_result = await asyncio.to_thread(
        classify_text,
        clean_text,          # <- cleaned text goes to the model
    )

    elapsed = time.perf_counter() - start_time

    if sentiment_result is None:
        stats["classify_failed"] += 1
        logger.warning("Could not classify this post.")
        return

    logger.info(
        "SENTIMENT: %s | SCORE: %.4f | %.2fs",
        sentiment_result["sentiment"],
        sentiment_result["confidence_score"],
        elapsed,
    )

    if sentiment_result["confidence_score"] < MIN_CONFIDENCE_TO_STORE:
        stats["skipped_low_confidence"] += 1
        logger.debug(
            "Skipped storing (confidence %.4f below MIN_CONFIDENCE_TO_STORE=%.2f)",
            sentiment_result["confidence_score"],
            MIN_CONFIDENCE_TO_STORE,
        )
        return

    # Record the moment this post was processed - this is what the
    # hourly/daily aggregation step later groups by.
    timestamp = datetime.now(timezone.utc).isoformat()

    await save_record(
        timestamp=timestamp,
        text=text,
        sentiment=sentiment_result["sentiment"],
        confidence_score=sentiment_result["confidence_score"],
    )


# ============================================================
# MAINTENANCE: PERIODIC STATS + AUTOMATIC DATABASE BACKUP
# ============================================================

MAINTENANCE_TICK_SECONDS = 60


def log_and_reset_stats(interval_seconds: float):
    """Write one summary line of what happened to posts, then reset."""
    if not stats:
        return

    summary = ", ".join(f"{name}={count}" for name, count in sorted(stats.items()))
    logger.info("STATS (last %.0fs): %s", interval_seconds, summary)
    stats.clear()


async def maintenance_loop():
    """
    Runs alongside the stream. Once a minute it checks whether it is time
    to (a) log a stats summary and (b) back up the database. Any error
    here is logged and ignored - maintenance must never stop collection.
    """
    last_stats_log = time.monotonic()

    while True:
        await asyncio.sleep(MAINTENANCE_TICK_SECONDS)

        try:
            now = time.monotonic()

            if (
                STATS_LOG_INTERVAL_SECONDS > 0
                and now - last_stats_log >= STATS_LOG_INTERVAL_SECONDS
            ):
                log_and_reset_stats(now - last_stats_log)
                last_stats_log = now

            if BACKUP_INTERVAL_HOURS > 0:
                age = await asyncio.to_thread(db.seconds_since_last_backup)

                if age is None or age >= BACKUP_INTERVAL_HOURS * 3600:
                    path = await asyncio.to_thread(db.backup_database)
                    if path is not None:
                        logger.info("Database backup saved: %s", path)

        except Exception as exc:
            logger.exception("Maintenance task error (collection continues): %s", exc)


# ============================================================
# CONNECT TO STREAM
# ============================================================

async def connect_and_stream():
    """
    Connect to the WebSocket and continuously receive messages.
    """

    logger.info("Connecting to live public microblog stream...")

    async with websockets.connect(
        STREAM_URL,

        # Keep the connection alive.
        ping_interval=PING_INTERVAL,
        ping_timeout=PING_TIMEOUT,

        # Give the connection time to close cleanly.
        close_timeout=CLOSE_TIMEOUT,

        # Prevent unnecessarily small frame limits.
        max_size=None,
    ) as websocket:

        logger.info("Connected to live public microblog stream.")

        while True:
            try:
                message = await websocket.recv()

                if message is None:
                    logger.warning("Received empty message.")
                    continue

                await process_message(message)

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "WebSocket closed: code=%s reason=%s",
                    exc.code,
                    exc.reason,
                )
                raise

            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timed out.")
                raise

            except Exception as exc:
                logger.exception(
                    "Error while processing stream message: %s",
                    exc,
                )


# ============================================================
# MAIN STREAM LOOP WITH AUTOMATIC RECONNECT
# ============================================================

async def stream_and_classify():
    """
    Main application loop.

    If the WebSocket disconnects, the application waits and
    automatically reconnects instead of terminating.
    """

    reconnect_delay = INITIAL_RECONNECT_DELAY

    while True:

        try:

            await connect_and_stream()

            # If the connection exits normally, reset the delay.
            reconnect_delay = INITIAL_RECONNECT_DELAY

        except asyncio.CancelledError:
            logger.info("Stream task cancelled.")
            raise

        except KeyboardInterrupt:
            logger.info("Stopping application...")
            break

        except Exception as exc:
            logger.error(
                "Stream connection failed: %s",
                exc,
            )

            logger.info(
                "Reconnecting in %s seconds...",
                reconnect_delay,
            )

            try:
                await asyncio.sleep(reconnect_delay)

            except asyncio.CancelledError:
                raise

            # Exponential backoff
            reconnect_delay = min(
                reconnect_delay * 2,
                MAX_RECONNECT_DELAY,
            )


async def run_all():
    """Run the stream and the maintenance loop together."""
    maintenance_task = asyncio.create_task(maintenance_loop())

    try:
        await stream_and_classify()
    finally:
        maintenance_task.cancel()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    logger.info("============================================")
    logger.info("       BLUESKY SENTIMENT MONITOR")
    logger.info("============================================")

    # Make sure the database file and table exist before we start
    # receiving posts.
    db.init_db()

    try:

        asyncio.run(
            run_all()
        )

    except KeyboardInterrupt:

        logger.info("")
        logger.info("Application stopped by user.")

    except Exception as exc:

        logger.exception(
            "Fatal application error: %s",
            exc,
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
