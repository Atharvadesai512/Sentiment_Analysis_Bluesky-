"""
filters.py

Filters that remove automated / repeated posts before they reach the
sentiment model. Kept separate from stream_monitor.py so they are easy
to read, tune and test on their own.
"""

import re
import time
from collections import OrderedDict, deque

from config import BOT_TEXT_PATTERNS

# ============================================================
# 1. KNOWN BOT TEXT PATTERNS
# ============================================================

_BOT_REGEXES = [
    re.compile(p, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE) for p in BOT_TEXT_PATTERNS
]


def matches_bot_pattern(text: str) -> bool:
    """True if the text matches any pattern in config.BOT_TEXT_PATTERNS."""
    return any(rx.search(text) for rx in _BOT_REGEXES)


# ============================================================
# TEXT HELPERS (shared by stream_monitor.py and the duplicate filter)
# ============================================================

# Bluesky shows links as truncated display text that usually has NO
# "https://" - e.g. "youtube.com/shorts/BVp64..." - so schemeless
# "domain.tld/path" links must be recognised too.
_URL = re.compile(
    r"https?://\S+"                       # https://example.com/x
    r"|www\.\S+"                          # www.example.com/x
    r"|\b[\w-]+(?:\.[\w-]+)+/\S*",        # example.com/x  or  youtu.be/abc
    flags=re.IGNORECASE,
)
_BARE_DOMAIN = re.compile(r"^\s*(?:[a-z0-9-]+\.)+[a-z]{2,}\s*$", flags=re.IGNORECASE)


def strip_urls(text: str) -> str:
    """Remove links (with or without https://) from the text."""
    return _URL.sub(" ", text)


def is_bare_domain(text: str) -> bool:
    """True if the whole post is just a domain name, e.g. 'shop.etsy.com'."""
    return bool(_BARE_DOMAIN.match(text))


def latin_letter_ratio(text: str) -> float:
    """
    Fraction of the text's letters that are Latin-script (a-z plus accented
    Latin). ~1.0 for English/Spanish/Portuguese, ~0.0 for Thai, Arabic,
    Chinese, Japanese, Cyrillic. Returns 0.0 if there are no letters.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if ord(c) < 0x250 or 0x1E00 <= ord(c) <= 0x1EFF)
    return latin / len(letters)


# ============================================================
# 2. PER-AUTHOR POSTING RATE
# ============================================================

class AuthorRateLimiter:
    """
    Limits how often a single author may post, using one or more
    sliding-window rules, e.g. limits=[(20, 3600), (8, 60)] means "at most
    20 posts per hour AND at most 8 per minute".
    """

    def __init__(self, limits, clock=time.monotonic, sweep_every: int = 5000):
        # keep only enabled rules (max_posts > 0)
        self.limits = [(m, w) for (m, w) in limits if m > 0]
        self._longest_window = max((w for _, w in self.limits), default=0)
        self._clock = clock
        self._sweep_every = sweep_every
        self._posts = {}   # author -> deque of accepted post times
        self._calls = 0

    def allow(self, author) -> bool:
        if not self.limits or not author:
            return True  # limiting disabled, or author unknown -> can't judge

        now = self._clock()

        recent = self._posts.get(author)
        if recent is None:
            recent = deque()
            self._posts[author] = recent

        cutoff = now - self._longest_window
        while recent and recent[0] <= cutoff:
            recent.popleft()

        self._calls += 1
        if self._calls % self._sweep_every == 0:
            self._sweep(cutoff)

        for max_posts, window in self.limits:
            in_window = sum(1 for t in recent if t > now - window)
            if in_window >= max_posts:
                return False

        recent.append(now)
        return True

    def _sweep(self, cutoff: float) -> None:
        """Drop authors with no posts left inside the longest window."""
        for author in list(self._posts):
            recent = self._posts[author]
            while recent and recent[0] <= cutoff:
                recent.popleft()
            if not recent:
                del self._posts[author]


# ============================================================
# 3. DUPLICATE TEXT
# ============================================================

_MENTION = re.compile(r"@\S+")
_HASHTAG = re.compile(r"#\S+")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """
    Reduce a post to its 'template' so near-identical bot posts collapse
    to the same key: lowercase, then remove URLs, @mentions, #hashtags,
    digits and punctuation, and collapse whitespace.
    """
    text = text.lower()
    text = strip_urls(text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


class DuplicateFilter:
    """
    Remembers normalized post texts and reports repeats seen within a
    time window. Texts shorter than `min_chars` are never duplicates
    (short replies like "thank you" are written by many real people).
    """

    def __init__(self, window_seconds: float, min_chars: int, max_entries: int,
                 clock=time.monotonic):
        self.window = window_seconds
        self.min_chars = min_chars
        self.max_entries = max_entries
        self._clock = clock
        self._seen = OrderedDict()   # hash(normalized text) -> last seen time

    def is_duplicate(self, text: str) -> bool:
        if self.window <= 0:
            return False

        key_text = normalize_for_dedup(text)
        if len(key_text) < self.min_chars:
            return False

        key = hash(key_text)
        now = self._clock()

        last_seen = self._seen.get(key)
        is_dup = last_seen is not None and (now - last_seen) < self.window

        self._seen[key] = now
        self._seen.move_to_end(key)
        self._prune(now)

        return is_dup

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._seen:
            oldest_key = next(iter(self._seen))
            if self._seen[oldest_key] <= cutoff or len(self._seen) > self.max_entries:
                del self._seen[oldest_key]
            else:
                break