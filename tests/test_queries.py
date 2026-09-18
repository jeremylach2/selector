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
