"""Reconcile the live Spotify saved-tracks library against the historical
listening warehouse.

Three questions the export alone can't answer, because it has no idea what's
currently saved, and the live API alone can't answer, because it has no
memory of past plays:

- saved but never actually played
- played heavily but never saved (an oversight worth fixing)
- saved once, then abandoned (played a lot, untouched recently)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from selector.spotify.client import SpotifyClient
from selector.warehouse.build import DEFAULT_DB_PATH

# Tuned for "clearly a habit, not a one-off" — kept in one place so they're
# easy to adjust without hunting through the query bodies.
RECONCILE_THRESHOLDS = {
    "played_never_saved_min_plays": 10,
    "abandoned_min_past_plays": 5,
    "abandoned_days_since_last_play": 90,
}


@dataclass
class ReconcileResult:
    saved_never_played: pd.DataFrame
    played_never_saved: pd.DataFrame
    saved_then_abandoned: pd.DataFrame


def _saved_tracks_frame(client: SpotifyClient) -> pd.DataFrame:
    items = client.all_saved_tracks()
    rows = []
    for item in items:
        track = item.get("track")
        if not track or not track.get("id"):
            continue
        artists = track.get("artists") or []
        rows.append(
            {
                "track_id": track["id"],
                "saved_name": track.get("name", ""),
                "saved_artist": artists[0]["name"] if artists else "",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["track_id", "saved_name", "saved_artist"])
    return pd.DataFrame(rows).drop_duplicates(subset="track_id")


def reconcile_library(client: SpotifyClient, db_path: Path = DEFAULT_DB_PATH) -> ReconcileResult:
    saved = _saved_tracks_frame(client)

    with duckdb.connect(str(db_path), read_only=True) as con:
        con.register("saved", saved)

        saved_never_played = con.execute(
            """
            SELECT s.track_id, s.saved_name AS name, s.saved_artist AS artist
            FROM saved s
            LEFT JOIN tracks t ON s.track_id = t.track_id
            WHERE t.track_id IS NULL
            ORDER BY s.saved_artist, s.saved_name
            """
        ).df()

        played_never_saved = con.execute(
            """
            SELECT t.track_id, t.name, t.artist, t.play_count, t.net_verdict
            FROM tracks t
            LEFT JOIN saved s ON t.track_id = s.track_id
            WHERE s.track_id IS NULL AND t.play_count >= ?
            ORDER BY t.play_count DESC
            """,
            [RECONCILE_THRESHOLDS["played_never_saved_min_plays"]],
        ).df()

        saved_then_abandoned = con.execute(
            """
            SELECT t.track_id, t.name, t.artist, t.play_count, t.last_played
            FROM tracks t
            JOIN saved s ON t.track_id = s.track_id
            WHERE t.play_count >= ?
                AND t.last_played < (SELECT MAX(last_played) FROM tracks) - INTERVAL (?) DAY
            ORDER BY t.play_count DESC
            """,
            [
                RECONCILE_THRESHOLDS["abandoned_min_past_plays"],
                RECONCILE_THRESHOLDS["abandoned_days_since_last_play"],
            ],
        ).df()

    return ReconcileResult(saved_never_played, played_never_saved, saved_then_abandoned)
