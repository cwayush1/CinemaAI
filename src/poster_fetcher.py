"""
Movie Poster Fetcher with SQLite & in-memory caching.
Fetches high-quality theatrical movie posters automatically via Wikipedia REST API
and falls back to dynamic cinema badge placeholders if offline or unavailable.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from functools import lru_cache
import sqlite3

POSTER_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "poster_cache.json"
)

# In-memory dictionary cache
_memory_cache = {}


def _load_cache():
    global _memory_cache
    if os.path.exists(POSTER_CACHE_FILE):
        try:
            with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
                _memory_cache = json.load(f)
        except Exception:
            _memory_cache = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(POSTER_CACHE_FILE), exist_ok=True)
        with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_memory_cache, f)
    except Exception:
        pass


_load_cache()


def clean_title_for_search(title):
    """Normalizes MovieLens title formatting (e.g. 'Dark Knight, The (2008)' -> 'The Dark Knight')."""
    # Remove year: (1995)
    t = re.sub(r"\s*\(\d{4}\)", "", title).strip()
    # Move trailing articles to front
    if ", The" in t:
        t = "The " + t.replace(", The", "").strip()
    elif ", A" in t:
        t = "A " + t.replace(", A", "").strip()
    elif ", An" in t:
        t = "An " + t.replace(", An", "").strip()
    # Remove aliases like (a.k.a. Se7en)
    t = re.sub(r"\(a\.k\.a\..*?\)", "", t).strip()
    return t


def get_fallback_poster_url(title, genres=""):
    """Generates a clean cinema placeholder image URL."""
    clean_t = clean_title_for_search(title)
    encoded_text = urllib.parse.quote(clean_t[:25])
    return f"https://placehold.co/300x440/1a202c/63b3ed.png?text={encoded_text}&font=roboto"


@lru_cache(maxsize=1024)
def get_movie_poster(movie_id, title, genres=""):
    """
    Returns the poster image URL for a given movie.
    Uses memory/JSON cache first, then Wikipedia REST API, then fallback placeholder.
    """
    key = str(movie_id)
    if key in _memory_cache and _memory_cache[key]:
        return _memory_cache[key]

    clean_t = clean_title_for_search(title)
    candidates = [
        urllib.parse.quote(clean_t.replace(" ", "_")),
        urllib.parse.quote((clean_t + " (film)").replace(" ", "_")),
        urllib.parse.quote((clean_t + " (movie)").replace(" ", "_")),
    ]

    for c in candidates:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{c}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MovieRecommenderPosterBot/1.0 (contact: ayush@example.com)"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                thumb = data.get("thumbnail", {}).get("source")
                if thumb and thumb.startswith("http"):
                    _memory_cache[key] = thumb
                    _save_cache()
                    return thumb
        except Exception:
            continue

    # Fallback placeholder if not found on Wikipedia
    fallback = get_fallback_poster_url(title, genres)
    _memory_cache[key] = fallback
    _save_cache()
    return fallback
