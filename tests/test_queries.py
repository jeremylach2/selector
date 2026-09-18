"""Tests run against the real warehouse built by `selector.warehouse.build`.

These assume `uv run python -m selector.ingest.load_history` and
`uv run python -m selector.warehouse.build` have already been run, matching
the "done when" criteria for Steps 1-2 of the prompt pack.
"""

import pytest

from selector.warehouse import queries
from selector.warehouse.build import DEFAULT_DB_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(), reason="warehouse not built; run selector.warehouse.build first"
)


def test_top_artists():
    df = queries.top_artists(limit=5)
    assert not df.empty
    assert list(df.columns) == ["artist", "play_count", "total_hours"]
    assert len(df) <= 5


def test_binged_then_abandoned():
    df = queries.binged_then_abandoned()
    assert not df.empty
    assert list(df.columns) == ["artist", "spike_month", "spike_plays", "followup_plays"]


def test_skip_offenders():
    df = queries.skip_offenders()
    assert not df.empty
    assert list(df.columns) == ["track_id", "name", "artist", "play_count", "skip_rate"]
    assert (df["skip_rate"] >= queries.QUERY_THRESHOLDS["skip_offender_min_skip_rate"]).all()


def test_listening_clock():
    df = queries.listening_clock()
    assert not df.empty
    assert list(df.columns) == ["hour_utc", "dow", "play_count"]


def test_taste_drift():
    df = queries.taste_drift()
    assert not df.empty
    assert list(df.columns) == ["period", "rank", "artist", "play_count"]
    assert df["rank"].max() <= 5


def test_taste_drift_rejects_bad_granularity():
    with pytest.raises(ValueError):
        queries.taste_drift(granularity="fortnight")


def test_rediscovery_candidates():
    df = queries.rediscovery_candidates()
    assert not df.empty
    assert list(df.columns) == ["track_id", "name", "artist", "play_count", "last_played"]


def test_search_library():
    # Search for a common single letter so the match is virtually guaranteed
    # to hit something in any real library, without depending on its content.
    df = queries.search_library("a", limit=5)
    assert not df.empty
    assert len(df) <= 5
    assert list(df.columns) == [
        "track_id",
        "name",
        "artist",
        "album",
        "play_count",
        "skip_rate",
        "net_verdict",
    ]


def test_search_library_no_match():
    df = queries.search_library("zzzzzznonexistentquery")
    assert df.empty


def test_track_detail_by_id():
    top = queries.top_artists(limit=1)
    assert not top.empty
    some_track = queries.search_library(top.iloc[0]["artist"], limit=1)
    assert not some_track.empty
    track_id = some_track.iloc[0]["track_id"]

    df = queries.track_detail(track_id)
    assert len(df) == 1
    assert df.iloc[0]["track_id"] == track_id


def test_track_detail_no_match():
    df = queries.track_detail("zzzzzznonexistentquery")
    assert df.empty


def test_warehouse_summary():
    df = queries.warehouse_summary()
    assert len(df) == 1
    assert list(df.columns) == [
        "earliest_play",
        "latest_play",
        "total_plays",
        "unique_tracks",
        "unique_artists",
        "total_hours",
    ]
    assert df.iloc[0]["total_plays"] > 0
