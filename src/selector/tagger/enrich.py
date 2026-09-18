"""Gather per-track input for the teacher labelling pipeline: metadata,
lyrics, and (where a preview clip was matched and Step 9 has run) measured
audio features.

Lyrics come from lrclib.net — see docs/LYRICS.md for why, over the more
obvious Genius API, which doesn't return lyrics text at all for exactly this
kind of use. A missing lyric, a missing release year, or a missing measured
feature are all valid, expected inputs, not failures — this project's tail
of niche tracks won't have all three for every row, and the teacher prompt
is written to work from whatever is available.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd

from selector.tagger.schema import TeacherInput
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

LYRICS_DIR = Path("data/lyrics")
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
REQUEST_DELAY_SECONDS = 0.3


def _lyrics_cache_path(track_id: str) -> Path:
    return LYRICS_DIR / f"{track_id}.txt"


def fetch_lyrics(client: httpx.Client, track_id: str, title: str, artist: str) -> str | None:
    """Look up plain lyrics for a track, caching to disk by track_id.

    Returns None (and caches an empty marker file) when lrclib has no
    match — a track with no lyrics is a normal outcome, not a retry target.
    """
    cache_path = _lyrics_cache_path(track_id)
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        return text if text else None

    try:
        resp = client.get(LRCLIB_SEARCH_URL, params={"track_name": title, "artist_name": artist})
        resp.raise_for_status()
        results = resp.json()
    except httpx.HTTPError:
        results = []

    lyrics = None
    for result in results:
        if result.get("plainLyrics"):
            lyrics = result["plainLyrics"]
            break

    LYRICS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(lyrics or "", encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
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
