"""Tests for reconcile_library against the real warehouse, with a fake
Spotify client standing in for the live API (no network, no auth).
"""

from __future__ import annotations

import pytest

from selector.spotify.reconcile import reconcile_library
from selector.warehouse import queries
from selector.warehouse.build import DEFAULT_DB_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(), reason="warehouse not built; run selector.warehouse.build first"
)


class _FakeSpotifyClient:
    def __init__(self, saved_items: list[dict]):
        self._items = saved_items

    def all_saved_tracks(self, max_items: int = 2000) -> list[dict]:
        return self._items[:max_items]


def _saved_item(track_id: str, name: str = "Fake Song", artist: str = "Fake Artist") -> dict:
    return {"track": {"id": track_id, "name": name, "artists": [{"name": artist}]}}


def test_saved_never_played_surfaces_unknown_track_id():
    client = _FakeSpotifyClient(
        [_saved_item("not-a-real-track-id", name="Unplayed", artist="Nobody")]
    )
    result = reconcile_library(client)

    assert (result.saved_never_played["track_id"] == "not-a-real-track-id").any()
    row = result.saved_never_played.set_index("track_id").loc["not-a-real-track-id"]
    assert row["name"] == "Unplayed"
    assert row["artist"] == "Nobody"


def test_played_never_saved_when_nothing_is_saved():
    client = _FakeSpotifyClient([])
    result = reconcile_library(client)

    assert not result.played_never_saved.empty
    assert (result.played_never_saved["play_count"] >= 10).all()


def test_saved_then_abandoned_matches_a_real_dormant_track():
    # 4 dormant months comfortably clears the reconcile threshold (90 days).
    candidates = queries.rediscovery_candidates(dormant_months=4, min_past_plays=5)
    assert not candidates.empty
    track_id = candidates.iloc[0]["track_id"]

    client = _FakeSpotifyClient([_saved_item(track_id)])
    result = reconcile_library(client)

    assert (result.saved_then_abandoned["track_id"] == track_id).any()


def test_reconcile_with_no_saved_tracks_at_all_does_not_error():
    client = _FakeSpotifyClient([])
    result = reconcile_library(client)

    assert result.saved_never_played.empty
    assert not result.played_never_saved.empty
    assert result.saved_then_abandoned.empty
