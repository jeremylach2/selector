"""The hosted server's Spotify path: the encrypted Redis token store, the
no-browser auth mode, and the guardrails on remote playlist creation.

No network: Redis is an in-memory fake (or respx for the REST client
itself), and Spotify is a fake client.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from selector.mcp import http_server, spotify_tools
from selector.spotify import auth, remote_store
from selector.spotify.client import SpotifyAPIError, SpotifyClient
from selector.spotify.remote_store import RedisTokenStore, RemoteStoreNotConfigured, UpstashRedis

TRACK = "spotify:track:" + "a" * 22


class FakeRedis:
    """The handful of commands the store and the guardrails send."""

    def __init__(self):
        self.data: dict[str, object] = {}
        self.commands: list[tuple] = []

    def command(self, *args):
        self.commands.append(args)
        op, key = args[0], args[1] if len(args) > 1 else None
        if op == "GET":
            return self.data.get(key)
        if op == "SET":
            if "NX" in args and key in self.data:
                return None
            self.data[key] = args[2]
            return "OK"
        if op == "EVAL":
            lock_key, nonce = args[3], args[4]
            if self.data.get(lock_key) == nonce:
                del self.data[lock_key]
                return 1
            return 0
        if op == "INCR":
            self.data[key] = int(self.data.get(key, 0)) + 1
            return self.data[key]
        if op == "EXPIRE":
            return 1
        if op == "LPUSH":
            self.data.setdefault(key, []).insert(0, args[2])
            return len(self.data[key])
        if op == "LTRIM":
            self.data[key] = self.data[key][int(args[2]) : int(args[3]) + 1]
            return "OK"
        raise AssertionError(f"unexpected command {op}")


def _token(access="a", refresh="r", ttl=3600.0) -> auth.TokenSet:
    return auth.TokenSet(access_token=access, refresh_token=refresh, expires_at=time.time() + ttl, scope="")


@pytest.fixture
def key() -> str:
    return Fernet.generate_key().decode("ascii")


@pytest.fixture
def store(key) -> RedisTokenStore:
    return RedisTokenStore(FakeRedis(), key)


# -- Upstash REST client ------------------------------------------------------


@respx.mock
def test_upstash_sends_one_json_command():
    route = respx.post("https://kv.example").mock(return_value=httpx.Response(200, json={"result": "OK"}))
    assert UpstashRedis("https://kv.example", "t").command("SET", "k", 5) == "OK"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer t"
    assert json.loads(request.content) == ["SET", "k", "5"]


@respx.mock
def test_upstash_error_never_echoes_the_command():
    respx.post("https://kv.example").mock(return_value=httpx.Response(400, json={"error": "ERR bad"}))
    with pytest.raises(remote_store.RedisError) as exc:
        UpstashRedis("https://kv.example", "t").command("SET", "k", "secret-ciphertext")
    assert "secret-ciphertext" not in str(exc.value)


def test_upstash_from_env_needs_both_vars(monkeypatch):
    for name in (*remote_store.URL_ENV_VARS, *remote_store.TOKEN_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RemoteStoreNotConfigured):
        UpstashRedis.from_env()
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://kv.example")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    assert UpstashRedis.from_env().url == "https://kv.example"


# -- token store ----------------------------------------------------------------


def test_store_round_trip_is_encrypted(store):
    token = _token(access="access-123", refresh="refresh-456")
    store.save(token)
    raw = store.redis.data[remote_store.TOKEN_KEY]
    assert "refresh-456" not in raw and "access-123" not in raw
    assert store.load() == token


def test_store_empty_loads_none(store):
    assert store.load() is None


def test_wrong_key_fails_loudly(store):
    store.save(_token())
    other = RedisTokenStore(store.redis, Fernet.generate_key().decode("ascii"))
    with pytest.raises(auth.SpotifyAuthError, match="Re-seed"):
        other.load()


def test_bad_key_is_a_config_error():
    with pytest.raises(RemoteStoreNotConfigured):
        RedisTokenStore(FakeRedis(), "not-a-fernet-key")


def test_from_env_needs_the_key(monkeypatch):
    monkeypatch.delenv(remote_store.KEY_ENV_VAR, raising=False)
    with pytest.raises(RemoteStoreNotConfigured, match=remote_store.KEY_ENV_VAR):
        RedisTokenStore.from_env(FakeRedis())


def test_lock_is_released_and_only_by_its_holder(store):
    with store.refresh_lock():
        assert remote_store.LOCK_KEY in store.redis.data
    assert remote_store.LOCK_KEY not in store.redis.data


def test_lock_held_elsewhere_waits_then_proceeds(store, monkeypatch):
    monkeypatch.setattr(remote_store, "LOCK_WAIT_SECONDS", 0.0)
    store.redis.data[remote_store.LOCK_KEY] = "someone-else"
    with store.refresh_lock():
        pass
    assert store.redis.data[remote_store.LOCK_KEY] == "someone-else"


# -- non-interactive auth -----------------------------------------------------


@pytest.fixture
def no_browser(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("the hosted server must never start a browser login")

    monkeypatch.setattr(auth, "_run_authorize_flow", _boom)


def test_no_token_raises_instead_of_opening_a_browser(store, no_browser):
    with pytest.raises(auth.SpotifyAuthError, match="seed_remote_spotify_token"):
        auth.get_valid_token("cid", store, interactive=False)


def test_dead_refresh_token_raises_instead_of_opening_a_browser(store, no_browser, monkeypatch):
    store.save(_token(ttl=-10))

    def _fail(client_id, refresh_token):
        raise auth.SpotifyAuthError("Token refresh failed: 400 invalid_grant")

    monkeypatch.setattr(auth, "_refresh_token", _fail)
    with pytest.raises(auth.SpotifyAuthError, match="Re-seed"):
        auth.get_valid_token("cid", store, interactive=False)


def test_refresh_saves_the_rotated_refresh_token(store, no_browser, monkeypatch):
    store.save(_token(access="old", refresh="old-r", ttl=-10))
    monkeypatch.setattr(auth, "_refresh_token", lambda cid, r: _token(access="new", refresh="new-r"))
    assert auth.get_valid_token("cid", store, interactive=False).access_token == "new"
    assert store.load().refresh_token == "new-r"
    assert remote_store.LOCK_KEY not in store.redis.data


def test_a_refresh_by_another_request_is_reused(store, no_browser, monkeypatch):
    """The token was expired on first read, but fresh once the lock was
    held: the other request already refreshed, so this one mustn't."""
    stale, fresh = _token(access="old", ttl=-10), _token(access="theirs")
    reads = iter([stale, fresh])
    monkeypatch.setattr(store, "load", lambda: next(reads))

    def _boom(*args):
        raise AssertionError("should reuse the other request's refresh")

    monkeypatch.setattr(auth, "_refresh_token", _boom)
    assert auth.get_valid_token("cid", store, interactive=False).access_token == "theirs"


# -- client --------------------------------------------------------------------


@respx.mock
def test_long_retry_after_raises_instead_of_sleeping(monkeypatch):
    monkeypatch.setattr(SpotifyClient, "_auth_headers", lambda self: {"Authorization": "Bearer x"})
    monkeypatch.setattr("selector.spotify.client.time.sleep", lambda s: pytest.fail("slept"))
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "30"}, json={})
    )
    client = SpotifyClient(client_id="cid", max_retry_after=10)
    with pytest.raises(SpotifyAPIError, match="retry in 30s"):
        client.current_user()


# -- remote playlist creation -------------------------------------------------


class FakeSpotify:
    def __init__(self, fail_add=False, fail_unfollow=False):
        self.store = type("S", (), {"redis": FakeRedis()})()
        self.created: list[tuple] = []
        self.added: list[tuple] = []
        self.unfollowed: list[str] = []
        self.fail_add, self.fail_unfollow = fail_add, fail_unfollow

    def create_playlist(self, name, description="", public=False, track_uris=None):
        assert track_uris is None, "tracks are added separately, so a failure can be undone"
        self.created.append((name, description, public))
        return {"id": "pl1", "name": name, "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}

    def add_tracks_to_playlist(self, playlist_id, uris):
        if self.fail_add:
            raise SpotifyAPIError(403, "Forbidden")
        self.added.append((playlist_id, uris))

    def unfollow_playlist(self, playlist_id):
        if self.fail_unfollow:
            raise SpotifyAPIError(500, "boom")
        self.unfollowed.append(playlist_id)


@pytest.fixture
def fake_spotify(monkeypatch):
    fake = FakeSpotify()
    monkeypatch.setattr(spotify_tools, "remote_client", lambda: fake)
    return fake


def test_create_playlist_success(fake_spotify):
    out = spotify_tools.spotify_create_playlist("  Night drive ", "for driving", [TRACK, TRACK])
    assert "Created playlist **Night drive** with 2 tracks" in out
    assert "https://open.spotify.com/playlist/pl1" in out
    name, description, public = fake_spotify.created[0]
    assert name == "Night drive" and public is False
    assert description == "for driving" + spotify_tools.DESCRIPTION_TAG
    assert fake_spotify.added == [("pl1", [TRACK, TRACK])]
    audit = json.loads(fake_spotify.store.redis.data[spotify_tools.AUDIT_KEY][0])
    assert audit["id"] == "pl1" and audit["tracks"] == 2 and "at" in audit


def test_description_is_trimmed_to_fit_the_tag(fake_spotify):
    spotify_tools.spotify_create_playlist("x", "d" * 400)
    description = fake_spotify.created[0][1]
    assert len(description) == spotify_tools.SPOTIFY_DESCRIPTION_LIMIT
    assert description.endswith(spotify_tools.DESCRIPTION_TAG)


@pytest.mark.parametrize(
    ("name", "uris", "why"),
    [
        ("   ", [], "needs a name"),
        ("n" * 101, [], "capped at 100"),
        ("ok", [TRACK] * 501, "at most 500"),
        ("ok", ["spotify:album:" + "a" * 22], "aren't Spotify track URIs"),
        ("ok", ["https://open.spotify.com/track/abc"], "aren't Spotify track URIs"),
    ],
)
def test_bad_requests_are_refused_before_touching_spotify(fake_spotify, name, uris, why):
    out = spotify_tools.spotify_create_playlist(name, track_uris=uris)
    assert out.startswith("Refused") and why in out
    assert fake_spotify.created == [] and fake_spotify.store.redis.commands == []


def test_daily_cap(fake_spotify, monkeypatch):
    monkeypatch.setattr(spotify_tools, "MAX_REMOTE_PLAYLISTS_PER_DAY", 2)
    for _ in range(2):
        assert spotify_tools.spotify_create_playlist("p").startswith("Created")
    out = spotify_tools.spotify_create_playlist("p")
    assert out.startswith("Refused") and "2 playlists today" in out
    assert len(fake_spotify.created) == 2


def test_failed_add_deletes_the_empty_playlist(fake_spotify):
    fake_spotify.fail_add = True
    out = spotify_tools.spotify_create_playlist("p", track_uris=[TRACK])
    assert "adding its tracks failed" in out and "deleted again" in out
    assert fake_spotify.unfollowed == ["pl1"]
    assert spotify_tools.AUDIT_KEY not in fake_spotify.store.redis.data


def test_failed_cleanup_points_at_the_orphan(fake_spotify):
    fake_spotify.fail_add = fake_spotify.fail_unfollow = True
    out = spotify_tools.spotify_create_playlist("p", track_uris=[TRACK])
    assert "remove it by hand: https://open.spotify.com/playlist/pl1" in out


# -- remote client config -----------------------------------------------------


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(spotify_tools.REMOTE_SWITCH_ENV_VAR, "off")
    assert "switched off" in spotify_tools.REMOTE_SPOTIFY_TOOLS[0]("anything")


def test_missing_config_is_a_message_not_a_crash(monkeypatch):
    monkeypatch.delenv(spotify_tools.REMOTE_SWITCH_ENV_VAR, raising=False)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setattr(spotify_tools, "_remote_client", None)
    monkeypatch.delenv(remote_store.KEY_ENV_VAR, raising=False)
    assert remote_store.KEY_ENV_VAR in spotify_tools.REMOTE_SPOTIFY_TOOLS[0]("anything")


def test_remote_client_never_allows_a_browser(monkeypatch, key):
    monkeypatch.delenv(spotify_tools.REMOTE_SWITCH_ENV_VAR, raising=False)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv(remote_store.KEY_ENV_VAR, key)
    monkeypatch.setenv("KV_REST_API_URL", "https://kv.example")
    monkeypatch.setenv("KV_REST_API_TOKEN", "t")
    monkeypatch.setattr(spotify_tools, "_remote_client", None)
    client = spotify_tools.remote_client()
    assert client.interactive is False
    assert isinstance(client.store, RedisTokenStore)
    assert client.max_retry_after == spotify_tools.REMOTE_MAX_RETRY_AFTER_SECONDS


# -- what the hosted server advertises -------------------------------------------


def _deployed_tools() -> dict:
    return {t.name: t for t in asyncio.run(http_server.deploy_server.list_tools())}


def test_remote_create_playlist_schema_has_no_public_flag():
    schema = _deployed_tools()["spotify_create_playlist"].input_schema
    assert set(schema["properties"]) == {"name", "description", "track_uris"}


def test_write_tool_is_marked_as_a_write():
    tools = _deployed_tools()
    assert tools["spotify_create_playlist"].annotations.read_only_hint is False
    for name in ("spotify_search", "spotify_saved_tracks", "spotify_recently_played"):
        assert tools[name].annotations.read_only_hint is True
