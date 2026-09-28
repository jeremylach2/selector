"""Gather per-track input for the teacher labelling pipeline: metadata,
lyrics, and (where a preview clip was matched and Step 9 has run) measured
audio features.

Lyrics come from lrclib.net rather than the more
obvious Genius API (see docs/LYRICS.md), which doesn't return lyrics text at all for exactly this
kind of use. A missing lyric, a missing release year, or a missing measured
feature are all valid, expected inputs, not failures. This project's tail
of niche tracks won't have all three for every row, and the teacher prompt
is written to work from whatever is available.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from selector.tagger.schema import TeacherInput
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

LYRICS_DIR = Path("data/lyrics")
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
REQUEST_DELAY_SECONDS = 0.3
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

# What the cache knows about a track's words. "unknown" covers both songs
# lrclib doesn't have and instrumentals it hasn't flagged: an empty lyrics
# file is not evidence that a track has no vocals.
LyricsStatus = Literal["lyrics", "instrumental", "unknown"]


class LyricsFetchError(Exception):
    """lrclib couldn't be reached. Never cached, so a rerun retries it."""


def _lyrics_cache_path(track_id: str) -> Path:
    return LYRICS_DIR / f"{track_id}.txt"


def _instrumental_marker_path(track_id: str) -> Path:
    return LYRICS_DIR / f"{track_id}.instrumental"


def lyrics_status(track_id: str) -> LyricsStatus | None:
    """Read a track's status from the cache. None means it was never fetched."""
    cache_path = _lyrics_cache_path(track_id)
    if not cache_path.exists():
        return None
    if cache_path.stat().st_size:
        return "lyrics"
    return "instrumental" if _instrumental_marker_path(track_id).exists() else "unknown"


def _search(client: httpx.Client, title: str, artist: str) -> list[dict]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.get(LRCLIB_SEARCH_URL, params={"track_name": title, "artist_name": artist})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 and e.response.status_code != 429:
                raise LyricsFetchError(str(e)) from e
            last = e
        except httpx.HTTPError as e:
            last = e
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise LyricsFetchError(str(last)) from last


def fetch_lyrics(
    client: httpx.Client, track_id: str, title: str, artist: str, refresh: bool = False
) -> str | None:
    """Look up plain lyrics for a track, caching to disk by track_id.

    Returns None when lrclib answered without lyrics, and caches that as an
    empty file, plus an `.instrumental` marker when lrclib flags the track
    as instrumental. A request that fails even after retries raises
    `LyricsFetchError` and caches nothing: an outage must not be remembered
    as "this song has no lyrics". `refresh` re-queries a cached empty result.
    """
    cache_path = _lyrics_cache_path(track_id)
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if text or not refresh:
            return text if text else None

    try:
        results = _search(client, title, artist)
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)

    lyrics = next((r["plainLyrics"] for r in results if r.get("plainLyrics")), None)
    instrumental = lyrics is None and any(r.get("instrumental") for r in results)

    LYRICS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(lyrics or "", encoding="utf-8")
    marker = _instrumental_marker_path(track_id)
    if instrumental:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    return lyrics


def _load_tracks(track_ids: list[str] | None, limit: int, db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as con:
        if track_ids is not None:
            return con.execute(
                "SELECT track_id, name, artist, album FROM tracks WHERE track_id = ANY(?)",
                [track_ids],
            ).df()
        return con.execute(
            "SELECT track_id, name, artist, album FROM tracks ORDER BY play_count DESC LIMIT ?",
            [limit],
        ).df()


def _measured_features_by_track(audio_features_path: Path, measured_columns: list[str]) -> dict[str, dict]:
    if not audio_features_path.exists():
        return {}
    features = pd.read_parquet(audio_features_path)
    available = [c for c in measured_columns if c in features.columns]
    return {
        row["track_id"]: {c: row[c] for c in available if pd.notna(row[c])}
        for _, row in features.iterrows()
    }


def enrich_tracks(
    track_ids: list[str] | None = None,
    limit: int = 3000,
    db_path: Path = DEFAULT_DB_PATH,
    audio_features_path: Path = Path("data/audio_features.parquet"),
) -> list[TeacherInput]:
    """Build one TeacherInput per track: warehouse metadata, cached lyrics,
    and measured audio features joined in where Step 9 has produced them."""
    tracks = _load_tracks(track_ids, limit, db_path)
    measured_by_track = _measured_features_by_track(
        audio_features_path, ["tempo_scaled", "rms_mean_scaled", "danceability", "harmonic_percussive_ratio_scaled"]
    )

    inputs: list[TeacherInput] = []
    with httpx.Client(timeout=10.0) as client:
        for _, row in tracks.iterrows():
            lyrics = fetch_lyrics(client, row["track_id"], row["name"], row["artist"])
            inputs.append(
                TeacherInput(
                    track_id=row["track_id"],
                    track_name=row["name"],
                    artist_name=row["artist"],
                    album_name=row["album"] if pd.notna(row["album"]) else None,
                    measured=measured_by_track.get(row["track_id"], {}),
                    lyrics=lyrics,
                )
            )
    return inputs
