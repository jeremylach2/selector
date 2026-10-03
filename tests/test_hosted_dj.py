"""The hosted DJ: the scipy-free crate it runs on, and `dj_tools.dj_set`.

Every test uses the invented crate from `test_dj.py`, so these run in CI.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from selector.dj import agent
from selector.dj.crate import (
    DEPLOY_COLUMNS,
    Crate,
    KCTags,
    deploy_crate_violations,
    load_deploy_crate,
    save_deploy_crate,
)
from selector.fly.lsh import hamming_distances
from selector.mcp import deploy_data, dj_tools, http_server, spotify_tools
from selector.spotify.client import SpotifyAPIError
from tests.test_dj import _at, _crate, _recent
from tests.test_remote_spotify import FakeSpotify

# -- KCTags -------------------------------------------------------------------


def _random_tags(n=40, n_kc=50, seed=0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, n_kc)) < 0.1).astype(float)
    dense[7] = 0  # an empty row
    return sparse.csr_matrix(dense)


def test_hamming_matches_the_fly_brain():
    matrix = _random_tags()
    tags = KCTags.from_sparse(matrix)
    for r in (0, 7, 39):
        np.testing.assert_array_equal(tags.hamming(r), hamming_distances(matrix[r], matrix))
    np.testing.assert_array_equal(tags.popcount, np.asarray(matrix.sum(axis=1)).ravel())


def test_take_keeps_the_requested_order():
    matrix = _random_tags()
    rows = np.array([5, 0, 7, 5, 39])
    taken = KCTags.from_sparse(matrix).take(rows)
    sub = matrix[rows]
    assert taken.n_rows == len(rows)
    for r in range(len(rows)):
        np.testing.assert_array_equal(taken.hamming(r), hamming_distances(sub[r], sub))


def test_crate_accepts_a_sparse_matrix():
    crate = _crate()
    assert isinstance(crate.tags, KCTags) and crate.tags.n_rows == len(crate.tracks)


# -- the deploy file ----------------------------------------------------------


def _plan(crate: Crate):
    run = agent.run_dj(
        crate, theme="golden hour", minutes=30, recent=_recent(crate), now=_at(2026, 9, 23, 17, 0), log_dir=None
    )
    return [s.to_dict() for s, _ in run.attempts], run.notes


def test_round_trip_plans_the_same_set(tmp_path):
    crate = _crate()
    loaded = load_deploy_crate(save_deploy_crate(crate, tmp_path / "crate.parquet"))
    assert len(loaded.tracks) == len(crate.tracks)
    assert _plan(loaded) == _plan(crate)


def test_only_the_columns_the_dj_reads_are_written(tmp_path):
    crate = _crate()
    crate.tracks["last_played"] = pd.Timestamp("2026-09-01 12:34:56")
    crate.tracks["play_count"] = 3
    path = save_deploy_crate(crate, tmp_path / "crate.parquet")
    assert set(pq.read_schema(path).names) == set(DEPLOY_COLUMNS)


def test_violations_catch_extra_and_time_columns():
    schema = pa.schema([(c, pa.string()) for c in DEPLOY_COLUMNS] + [("last_played", pa.timestamp("us"))])
    problems = deploy_crate_violations(schema)
    assert "unexpected column last_played" in problems
    assert any(p.startswith("time column last_played") for p in problems)
    assert deploy_crate_violations(pa.schema([("track_id", pa.string())]))


def test_upload_crate_refuses_a_bad_file_before_any_network(tmp_path, monkeypatch):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"track_id": ["t1"], "ts": [pd.Timestamp("2026-01-01")]}).to_parquet(path)
    monkeypatch.setattr(deploy_data, "_put", lambda *a: pytest.fail("uploaded a bad crate"))
    with pytest.raises(RuntimeError, match="not a deploy crate"):
        deploy_data.upload_crate(path)


# -- the hosted dj_set --------------------------------------------------------


class FakeDJSpotify(FakeSpotify):
    def __init__(self, recent_ids, **kwargs):
        super().__init__(**kwargs)
        self.recent_ids = recent_ids
        self.fail_recent = False

    def recently_played(self, limit=20):
        if self.fail_recent:
            raise SpotifyAPIError(503, "down")
        return {"items": [{"track": {"id": t, "artists": [{"name": "Someone"}]}} for t in self.recent_ids[:limit]]}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    crate = _crate()
    # Real-looking ids, since the guarded write refuses anything that isn't
    # `spotify:track:<22 characters>`.
    crate.tracks["track_id"] = [f"{i:022d}" for i in range(len(crate.tracks))]
    path = save_deploy_crate(crate, tmp_path / "crate.parquet")
    monkeypatch.setenv(dj_tools.CRATE_ENV_VAR, str(path))
    monkeypatch.setenv(dj_tools.TIMEZONE_ENV_VAR, "America/Chicago")
    monkeypatch.setenv("SELECTOR_DB", str(tmp_path / "no-warehouse.duckdb"))
    monkeypatch.setattr(dj_tools, "_crate_cache", None)
    fake = FakeDJSpotify(list(crate.tracks["track_id"].head(10)))
    monkeypatch.setattr(spotify_tools, "remote_client", lambda: fake)
    return fake


def _force_verdict(monkeypatch, passed: bool):
    real = agent.critique
    monkeypatch.setattr(agent, "critique", lambda selection, arc: replace(real(selection, arc), passed=passed))


def test_dry_run_plans_without_writing(hosted):
    out = dj_tools.dj_set(theme="focus drift", minutes=30)
    assert "Dry run" in out and "Critique chain" in out
    assert "Recent plays from live Spotify" in out and "(CDT)" in out
    assert hosted.created == [] and hosted.store.redis.commands == []


def test_commit_goes_through_the_guarded_write(hosted, monkeypatch):
    _force_verdict(monkeypatch, passed=True)
    out = dj_tools.dj_set(theme="focus drift", minutes=30, dry_run=False)
    assert "Created playlist:** https://open.spotify.com/playlist/pl1" in out
    name, description, public = hosted.created[0]
    assert name.startswith("Selector DJ · Focus Drift") and public is False
    assert description.endswith(spotify_tools.DESCRIPTION_TAG)
    (playlist_id, uris), = hosted.added
    assert playlist_id == "pl1" and uris and all(u.startswith("spotify:track:") for u in uris)
    audit = json.loads(hosted.store.redis.data[spotify_tools.AUDIT_KEY][0])
    assert audit["tracks"] == len(uris)


def test_failed_critique_is_never_written(hosted, monkeypatch):
    _force_verdict(monkeypatch, passed=False)
    out = dj_tools.dj_set(theme="focus drift", minutes=30, dry_run=False)
    assert "Not committed" in out and "Critique did not pass" in out
    assert hosted.created == [] and hosted.store.redis.commands == []


def test_daily_cap_applies_to_dj_sets(hosted, monkeypatch):
    _force_verdict(monkeypatch, passed=True)
    monkeypatch.setattr(spotify_tools, "MAX_REMOTE_PLAYLISTS_PER_DAY", 0)
    out = dj_tools.dj_set(theme="focus drift", minutes=30, dry_run=False)
    assert "Not committed" in out and "playlists today" in out
    assert hosted.created == []


def test_live_failure_falls_back_to_the_warehouse(hosted, warehouses, monkeypatch):
    monkeypatch.setenv("SELECTOR_DB", str(warehouses[1]))
    hosted.fail_recent = True
    out = dj_tools.dj_set(theme="focus drift", minutes=30)
    assert "the deploy warehouse (live Spotify failed" in out and "Dry run" in out


def test_warehouse_recent_reads_the_newest_hour_first(warehouses):
    recent = dj_tools._warehouse_recent(warehouses[1])
    assert list(recent.columns) == ["track_id", "artist_name"]
    assert recent["track_id"].iloc[0] == "t2"


def test_without_spotify_a_dry_run_still_works_but_a_commit_does_not(hosted, monkeypatch):
    def _unconfigured():
        raise spotify_tools.SpotifyNotConfigured("SPOTIFY_CLIENT_ID is not set.")

    monkeypatch.setattr(spotify_tools, "remote_client", _unconfigured)
    assert "Dry run" in dj_tools.dj_set(theme="focus drift", minutes=30)
    assert dj_tools.dj_set(theme="focus drift", minutes=30, dry_run=False).startswith("Can't create the playlist")


def test_clock_falls_back_to_utc_and_says_so(hosted, monkeypatch):
    monkeypatch.delenv(dj_tools.TIMEZONE_ENV_VAR)
    assert "Planned in UTC" in dj_tools.dj_set(theme="focus drift", minutes=30)
    monkeypatch.setenv(dj_tools.TIMEZONE_ENV_VAR, "Mars/Olympus_Mons")
    assert "isn't a known time zone" in dj_tools.dj_set(theme="focus drift", minutes=30)


@pytest.mark.parametrize("minutes", [5, 181])
def test_minutes_are_bounded(hosted, minutes):
    assert dj_tools.dj_set(minutes=minutes).startswith("Refused")


def test_bad_theme_is_a_message(hosted):
    assert "Couldn't read a mood" in dj_tools.dj_set(theme="zzz qqq")


def test_missing_crate_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setenv(dj_tools.CRATE_ENV_VAR, str(tmp_path / "none.parquet"))
    monkeypatch.delenv(deploy_data.TOKEN_ENV_VAR, raising=False)
    out = dj_tools.dj_set()
    assert "No DJ crate" in out and "selector.dj.pool --deploy" in out and "upload-crate" in out


def test_hosted_dj_set_is_marked_as_a_write():
    tools = {t.name: t for t in asyncio.run(http_server.deploy_server.list_tools())}
    tool = tools["dj_set"]
    assert tool.annotations.read_only_hint is False
    assert set(tool.input_schema["properties"]) == {
        "theme", "minutes", "dry_run", "familiar_ratio", "seed_tracks", "exclude_tracks", "exclude_artists",
    }


# -- hosted fly tools ---------------------------------------------------------


def test_more_like_this_returns_the_nearest_fingerprints(hosted):
    from selector.mcp import crate_tools

    crate = dj_tools._load_crate(dj_tools._crate_path())
    seed = crate.tracks["track_id"].iat[3]
    out = crate_tools.more_like_this(seed, k=5)
    assert out.startswith(f"Nearest to **{crate.tracks['name'].iat[3]}**")
    assert f"| {crate.tracks['name'].iat[3]} |" not in out
    # `KCTags.hamming` is checked against the fly brain's own distances above.
    shown = [int(line.rsplit("|", 2)[1]) for line in out.splitlines() if line.startswith("| Song")]
    assert shown == [int(d) for d in sorted(crate.tags.hamming(3))[1:6]]


def test_fly_score_reports_the_rank(hosted):
    from selector.mcp import crate_tools

    crate = dj_tools._load_crate(dj_tools._crate_path())
    best = crate.tracks.sort_values("fly_valence").iloc[-1]
    out = crate_tools.fly_score(f"spotify:track:{best['track_id']}")
    assert f"**{best['name']}** by {best['artist']}" in out
    assert f"ranked 1 of the {len(crate.tracks):,} tracks" in out


def test_tracks_outside_the_crate_get_a_message(hosted, warehouses, monkeypatch):
    from selector.mcp import crate_tools

    monkeypatch.setenv("SELECTOR_DB", str(warehouses[1]))
    out = crate_tools.more_like_this("Song t1")
    assert out.startswith('"Song t1" by Alpha isn\'t in the hosted crate')
    assert crate_tools.fly_score("zzz-no-such-track").startswith("No track matches")


def test_hosted_order_tracks(hosted):
    from selector.mcp import crate_tools

    crate = dj_tools._load_crate(dj_tools._crate_path())
    ids = list(crate.tracks["track_id"].head(8))
    out = crate_tools.order_tracks([f"spotify:track:{t}" for t in ids])
    uris = json.loads(out.rsplit("\n", 1)[-1])
    assert sorted(uris) == sorted(f"spotify:track:{t}" for t in ids)
    assert crate_tools.order_tracks(ids[:1]).startswith("Refused")


def test_tie_note_counts_ties_and_never_the_seed():
    from selector.mcp.crate_tools import tie_note

    distances = np.array([0, 0, 0, 4, 4, 9])
    note = tie_note(distances, [0, 0, 4])
    assert "2 tracks tie at distance 0" in note and "distance 4" not in note
    assert tie_note(distances, [4, 9]) == ""


def test_hosted_fly_tools_are_read_only():
    tools = {t.name: t for t in asyncio.run(http_server.deploy_server.list_tools())}
    for name in ("more_like_this", "fly_score", "order_tracks", "resolve_tracks"):
        assert tools[name].annotations.read_only_hint is True
