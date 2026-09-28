"""MCP OAuth on the hosted server, logging in through Spotify.

Drives the real ASGI app (the SDK's auth routes plus our provider) end to
end with an in-memory Redis. Spotify's code exchange and `/me` are
replaced with fakes, so there's no network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet
from starlette.testclient import TestClient

from selector.mcp import http_server, oauth
from selector.spotify import auth
from selector.spotify.remote_store import TOKEN_KEY

PUBLIC = "https://selector.example"
OWNER = "owner-id"
CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"


class FakeRedis:
    """The commands the OAuth provider and token store send. TTLs are
    ignored; expiry is tested through the records' `expires_at`."""

    def __init__(self):
        self.data: dict[str, object] = {}

    def command(self, *args):
        op, rest = args[0], [str(a) for a in args[1:]]
        if op == "GET":
            return self.data.get(rest[0])
        if op == "SET":
            if "NX" in rest and rest[0] in self.data:
                return None
            self.data[rest[0]] = rest[1]
            return "OK"
        if op == "GETDEL":
            return self.data.pop(rest[0], None)
        if op == "DEL":
            return sum(self.data.pop(k, None) is not None for k in rest)
        if op == "INCR":
            self.data[rest[0]] = int(self.data.get(rest[0], 0)) + 1
            return self.data[rest[0]]
        if op == "EXPIRE":
            return int(rest[0] in self.data)
        if op == "SCAN":
            prefix = rest[2].rstrip("*")
            return ["0", [k for k in self.data if k.startswith(prefix)]]
        raise AssertionError(f"unexpected command {op}")

    def keys(self, prefix: str) -> list[str]:
        return [k for k in self.data if k.startswith(prefix)]


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def fernet_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def provider(redis, fernet_key, monkeypatch):
    config = oauth.OAuthConfig(public_url=PUBLIC, spotify_client_id="spotify-cid", owner_id=OWNER)
    p = oauth.SpotifyOAuthProvider(redis, fernet_key, config, legacy_bearer="legacy-s3cret")
    spotify = {"user": OWNER, "exchanges": []}

    def fake_exchange(client_id, code, verifier, redirect_uri=auth.REDIRECT_URI, scopes=auth.SCOPES):
        spotify["exchanges"].append({"code": code, "redirect_uri": redirect_uri, "verifier": verifier})
        return auth.TokenSet("sp-access", "sp-refresh", time.time() + 3600, scopes)

    monkeypatch.setattr(auth, "_exchange_code_for_token", fake_exchange)
    monkeypatch.setattr(oauth, "_fetch_spotify_user_id", lambda access_token: spotify["user"])
    p.spotify = spotify
    return p


@pytest.fixture
def client(provider):
    _, app = http_server.create_app(provider)
    with TestClient(app, base_url=PUBLIC, follow_redirects=False) as c:
        yield c


def _register(client, redirect=CLAUDE_CALLBACK):
    r = client.post(
        "/register",
        json={"redirect_uris": [redirect], "token_endpoint_auth_method": "none", "client_name": "test"},
    )
    return r


def _login(client, provider, *, resource=f"{PUBLIC}/mcp"):
    """Register, authorize, come back from Spotify. Returns (client_id,
    verifier, the redirect to the MCP client)."""
    client_id = _register(client).json()["client_id"]
    verifier, challenge = _pkce()
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": CLAUDE_CALLBACK,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "client-state",
    }
    if resource:
        query["resource"] = resource
    r = client.get("/authorize", params=query)
    assert r.status_code == 302, r.text
    spotify_url = urlparse(r.headers["location"])
    assert spotify_url.netloc == "accounts.spotify.com"
    sq = parse_qs(spotify_url.query)
    assert sq["redirect_uri"] == [f"{PUBLIC}/oauth/spotify/callback"]
    assert sq["scope"] == [auth.REMOTE_SCOPES]
    assert sq["code_challenge_method"] == ["S256"]
    back = client.get(oauth.CALLBACK_PATH, params={"code": "spotify-code", "state": sq["state"][0]})
    return client_id, verifier, back


def _code_from(back) -> str:
    assert back.status_code == 302
    q = parse_qs(urlparse(back.headers["location"]).query)
    assert q["state"] == ["client-state"]
    return q["code"][0]


def _token(client, client_id, code, verifier):
    return client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_verifier": verifier,
        },
    )


def _list_tools(client, bearer):
    return client.post(
        "/mcp",
        headers={
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        },
    )


# -- discovery ----------------------------------------------------------------


def test_unauthenticated_mcp_points_at_resource_metadata(client):
    r = client.post("/mcp", json={})
    assert r.status_code == 401
    assert f'resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource/mcp"' in r.headers["www-authenticate"]


def test_well_known_metadata(client):
    resource = client.get("/.well-known/oauth-protected-resource/mcp").json()
    assert resource["resource"] == f"{PUBLIC}/mcp"
    assert resource["authorization_servers"] == [PUBLIC]
    server = client.get("/.well-known/oauth-authorization-server").json()
    assert server["issuer"] == PUBLIC
    assert server["authorization_endpoint"] == f"{PUBLIC}/authorize"
    assert server["token_endpoint"] == f"{PUBLIC}/token"
    assert server["registration_endpoint"] == f"{PUBLIC}/register"
    assert server["revocation_endpoint"] == f"{PUBLIC}/revoke"
    assert server["code_challenge_methods_supported"] == ["S256"]


# -- the happy path -------------------------------------------------------------


def test_owner_login_issues_working_tokens_and_stores_spotify_token(client, provider, redis):
    client_id, verifier, back = _login(client, provider)
    tokens = _token(client, client_id, _code_from(back), verifier)
    assert tokens.status_code == 200, tokens.text
    body = tokens.json()
    assert body["expires_in"] == oauth.ACCESS_TTL and body["refresh_token"]

    assert _list_tools(client, body["access_token"]).status_code == 200
    # Logging in replaced the Spotify token the tools use.
    assert provider.token_store.load().access_token == "sp-access"
    assert provider.spotify["exchanges"][0]["redirect_uri"] == f"{PUBLIC}/oauth/spotify/callback"
    # Nothing is stored in plain text: not the tokens, not the Spotify token.
    dump = json.dumps(redis.data)
    for secret in (body["access_token"], body["refresh_token"], "sp-access", "sp-refresh"):
        assert secret not in dump
    assert redis.data[TOKEN_KEY]


def test_refresh_rotates_and_old_tokens_die(client, provider):
    client_id, verifier, back = _login(client, provider)
    first = _token(client, client_id, _code_from(back), verifier).json()
    refresh = lambda rt: client.post(
        "/token", data={"grant_type": "refresh_token", "refresh_token": rt, "client_id": client_id}
    )
    second = refresh(first["refresh_token"])
    assert second.status_code == 200
    assert refresh(first["refresh_token"]).status_code == 400  # single use
    assert _list_tools(client, first["access_token"]).status_code == 401
    assert _list_tools(client, second.json()["access_token"]).status_code == 200


def test_revoke_endpoint_kills_access_and_refresh(client, provider):
    client_id, verifier, back = _login(client, provider)
    body = _token(client, client_id, _code_from(back), verifier).json()
    r = client.post("/revoke", data={"token": body["access_token"], "client_id": client_id, "client_secret": ""})
    assert r.status_code == 200
    assert _list_tools(client, body["access_token"]).status_code == 401
    r = client.post(
        "/token", data={"grant_type": "refresh_token", "refresh_token": body["refresh_token"], "client_id": client_id}
    )
    assert r.status_code == 400


def test_legacy_static_bearer_still_works_during_migration(client):
    assert _list_tools(client, "legacy-s3cret").status_code == 200
    assert _list_tools(client, "legacy-wrong").status_code == 401


# -- refusals ---------------------------------------------------------------------


def test_someone_elses_spotify_account_is_refused(client, provider, redis):
    provider.spotify["user"] = "stranger"
    _, _, back = _login(client, provider)
    q = parse_qs(urlparse(back.headers["location"]).query)
    assert q["error"] == ["access_denied"] and "code" not in q
    assert TOKEN_KEY not in redis.data  # their Spotify token was dropped
    assert not redis.keys(oauth.CODE_PREFIX)


def test_codes_cannot_be_reused(client, provider):
    client_id, verifier, back = _login(client, provider)
    code = _code_from(back)
    assert _token(client, client_id, code, verifier).status_code == 200
    again = _token(client, client_id, code, verifier)
    assert again.status_code == 400 and again.json()["error"] == "invalid_grant"


def test_wrong_pkce_verifier_is_refused(client, provider):
    client_id, _, back = _login(client, provider)
    assert _token(client, client_id, _code_from(back), "not-the-verifier" * 4).status_code == 400


def test_unknown_or_replayed_state_is_rejected(client, provider):
    r = client.get(oauth.CALLBACK_PATH, params={"code": "x", "state": "made-up"})
    assert r.status_code == 400
    assert not provider.spotify["exchanges"]

    client_id = _register(client).json()["client_id"]
    _, challenge = _pkce()
    r = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
    )
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    assert client.get(oauth.CALLBACK_PATH, params={"code": "c", "state": state}).status_code == 302
    assert client.get(oauth.CALLBACK_PATH, params={"code": "c", "state": state}).status_code == 400


def test_spotify_error_goes_back_to_the_client(client, provider):
    client_id = _register(client).json()["client_id"]
    _, challenge = _pkce()
    r = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
        },
    )
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    back = client.get(oauth.CALLBACK_PATH, params={"error": "access_denied", "state": state})
    assert parse_qs(urlparse(back.headers["location"]).query)["error"] == ["access_denied"]
    assert not provider.spotify["exchanges"]


def test_expired_access_token_gets_401(client, provider, redis):
    client_id, verifier, back = _login(client, provider)
    access = _token(client, client_id, _code_from(back), verifier).json()["access_token"]
    key = oauth.ACCESS_PREFIX + oauth._hash(access)
    record = provider._decode(redis.data[key])
    record["expires_at"] = int(time.time()) - 5
    redis.data[key] = provider._fernet.encrypt(json.dumps(record).encode()).decode()
    assert _list_tools(client, access).status_code == 401


def test_registration_refuses_unknown_redirects(client):
    r = _register(client, redirect="https://evil.example/callback")
    assert r.status_code == 400 and r.json()["error"] == "invalid_redirect_uri"
    assert _register(client, redirect="http://127.0.0.1:33418/callback").status_code == 201
    assert _register(client, redirect="http://localhost:6274/oauth/callback").status_code == 201


def test_registrations_are_capped_per_day(client):
    for _ in range(oauth.MAX_REGISTRATIONS_PER_DAY):
        assert _register(client).status_code == 201
    assert _register(client).status_code == 400


def test_authorize_refuses_another_resource(client, provider):
    client_id = _register(client).json()["client_id"]
    _, challenge = _pkce()
    r = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
            "resource": "https://other.example/mcp",
        },
    )
    assert parse_qs(urlparse(r.headers["location"]).query)["error"] == ["invalid_target"]


def test_redirect_allowlist():
    assert oauth.redirect_allowed(CLAUDE_CALLBACK)
    assert oauth.redirect_allowed("http://[::1]:5000/cb")
    assert not oauth.redirect_allowed("https://localhost/cb")  # loopback is plain http only
    assert not oauth.redirect_allowed("https://claude.ai.evil.example/api/mcp/auth_callback")
    assert oauth.redirect_allowed("https://x.example/cb", frozenset({"https://x.example/cb"}))


# -- revocation script and configuration ------------------------------------------


def test_revoke_all_sessions(client, provider, redis):
    client_id, verifier, back = _login(client, provider)
    body = _token(client, client_id, _code_from(back), verifier).json()
    assert oauth.revoke_all_sessions(redis) == 2  # one access, one refresh
    assert _list_tools(client, body["access_token"]).status_code == 401
    assert redis.keys(oauth.CLIENT_PREFIX)  # clients kept by default
    oauth.revoke_all_sessions(redis, include_clients=True)
    assert not redis.keys(oauth.CLIENT_PREFIX)
    assert TOKEN_KEY in redis.data  # the Spotify token isn't a session


@pytest.fixture
def bare_env(monkeypatch):
    for var in (
        "KV_REST_API_URL",
        "KV_REST_API_TOKEN",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "SELECTOR_TOKEN_KEY",
        oauth.OWNER_ENV_VAR,
        oauth.PUBLIC_URL_ENV_VAR,
        oauth.VERCEL_PRODUCTION_URL_ENV_VAR,
        oauth.LEGACY_BEARER_ENV_VAR,
        http_server.ALLOW_NO_AUTH_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")


def test_missing_config_refuses_everything(bare_env, monkeypatch):
    # Even with the old bearer set, a half-configured deploy serves nothing.
    monkeypatch.setenv(oauth.LEGACY_BEARER_ENV_VAR, "s3cret")
    _, app = http_server.create_app()
    with TestClient(app) as c:
        assert c.post("/mcp", headers={"Authorization": "Bearer s3cret"}).status_code == 503
        assert c.get("/.well-known/oauth-authorization-server").status_code == 503


def test_explicit_opt_out_serves_without_auth(bare_env, monkeypatch):
    monkeypatch.setenv(http_server.ALLOW_NO_AUTH_ENV_VAR, "1")
    _, app = http_server.create_app()
    with TestClient(app) as c:
        assert _list_tools(c, "anything").status_code == 200


def test_full_env_builds_oauth(bare_env, monkeypatch):
    monkeypatch.setenv("KV_REST_API_URL", "https://redis.example")
    monkeypatch.setenv("KV_REST_API_TOKEN", "t")
    monkeypatch.setenv("SELECTOR_TOKEN_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv(oauth.OWNER_ENV_VAR, OWNER)
    monkeypatch.setenv(oauth.VERCEL_PRODUCTION_URL_ENV_VAR, "selector.example")
    p = oauth.SpotifyOAuthProvider.from_env()
    assert p.config.public_url == PUBLIC and p.config.owner_id == OWNER
    assert p.legacy_bearer is None
    monkeypatch.setenv(oauth.PUBLIC_URL_ENV_VAR, "https://custom.example/")
    assert oauth.config_from_env().callback_url == "https://custom.example/oauth/spotify/callback"


def test_minted_check_token_works_until_deleted(client, redis, fernet_key):
    token, redis_key = oauth.mint_access_token(redis, fernet_key, OWNER)
    assert _list_tools(client, token).status_code == 200
    redis.command("DEL", redis_key)
    assert _list_tools(client, token).status_code == 401
