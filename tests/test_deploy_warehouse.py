"""The deploy warehouse and the deployed tool list.

The `warehouses` fixture (conftest.py) is built from a few invented plays
rather than the real warehouse, so these run in CI too.
"""

import asyncio

import duckdb
import pandas as pd
import pytest

from selector.mcp import http_server
from selector.mcp import server as stdio
from selector.warehouse import queries

DEPLOYED_TOOL_NAMES = {
    "warehouse_summary",
    "search_library",
    "track_detail",
    "top_artists",
    "binged_then_abandoned",
    "skip_offenders",
    "listening_clock",
    "taste_drift",
    "rediscovery_candidates",
    "spotify_search",
    "spotify_saved_tracks",
    "spotify_top_artists",
    "spotify_top_tracks",
    "spotify_recently_played",
    "spotify_create_playlist",
    "dj_set",
}


def _columns(db) -> list[tuple[str, str, str]]:
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns"
        ).fetchall()


def test_no_plays_or_sessions_table(warehouses):
    _, deploy = warehouses
    tables = {table for table, _, _ in _columns(deploy)}
    assert tables == {"plays_hourly", "tracks", "artists", "artist_months"}


def test_nothing_finer_than_an_hour(warehouses):
    _, deploy = warehouses
    columns = _columns(deploy)
    assert not [c for c in columns if c[2].startswith("TIME")]
    hourly = {column for table, column, _ in columns if table == "plays_hourly"}
    assert hourly == {"date", "hour_utc", "dow", "artist", "track_id", "play_count", "ms_played", "skips"}


def test_plays_hourly_keeps_the_counts(warehouses):
    _, deploy = warehouses
    with duckdb.connect(str(deploy), read_only=True) as con:
        rows = con.execute(
            "SELECT play_count, skips FROM plays_hourly WHERE track_id = 't1' AND hour_utc = 12"
        ).fetchall()
    assert rows == [(2, 1)]


@pytest.mark.parametrize(
    "query, kwargs",
    [
        (queries.top_artists, {}),
        (queries.top_artists, {"start": "2024-06-01", "end": "2025-01-01"}),
        (queries.listening_clock, {}),
        (queries.taste_drift, {"granularity": "year"}),
        (queries.skip_offenders, {"min_plays": 1}),
        (queries.search_library, {"query": "song"}),
    ],
)
def test_deploy_answers_match_the_full_warehouse(warehouses, query, kwargs):
    full, deploy = warehouses
    a = query(db_path=full, **kwargs)
    b = query(db_path=deploy, **kwargs)
    assert not b.empty
    if "period" in a:
        a["period"] = pd.to_datetime(a["period"]).dt.year
        b["period"] = pd.to_datetime(b["period"]).dt.year
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_summary_is_day_granular(warehouses):
    full, deploy = warehouses
    a = queries.warehouse_summary(db_path=full).iloc[0]
    b = queries.warehouse_summary(db_path=deploy).iloc[0]
    for col in ["total_plays", "unique_tracks", "unique_artists", "total_hours"]:
        assert a[col] == b[col]
    assert str(b["earliest_play"])[:10] == "2024-03-14"
    assert pd.Timestamp(b["earliest_play"]).time() == pd.Timestamp(0).time()


def test_every_deployed_tool_answers(warehouses, monkeypatch):
    _, deploy = warehouses
    monkeypatch.setenv(stdio.DB_ENV_VAR, str(deploy))
    calls = {
        "search_library": {"query": "song"},
        "track_detail": {"track_id_or_name": "t1"},
        "binged_then_abandoned": {"min_plays": 1},
        "skip_offenders": {"min_plays": 1},
        "rediscovery_candidates": {"dormant_months": 1, "min_past_plays": 1},
    }
    for fn in http_server.WAREHOUSE_TOOLS:
        out = fn(**calls.get(fn.__name__, {}))
        assert not out.startswith(("Query failed", "No warehouse")), (fn.__name__, out)
        assert "_No rows matched._" not in out, fn.__name__


def test_http_server_serves_only_the_deploy_tools():
    tools = asyncio.run(http_server.deploy_server.list_tools())
    assert {t.name for t in tools} == DEPLOYED_TOOL_NAMES
