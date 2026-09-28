"""Where the hosted MCP server keeps its Spotify token: Upstash Redis.

A serverless function has no home directory that outlives it, so the
token cache in `~/.selector/token.json` has no equivalent there. Instead
the token set lives in one Redis key, encrypted with Fernet under
`SELECTOR_TOKEN_KEY`, so a leaked Redis dump or Redis token alone can't be
used against the Spotify account. It's written by the MCP OAuth login
(`selector.mcp.oauth`) or by `scripts/seed_remote_spotify_token.py`, and
refreshed in place after that.

Redis is spoken over Upstash's REST API with `httpx`, so the function needs
no Redis client package. The Upstash integration on Vercel sets
`KV_REST_API_URL` and `KV_REST_API_TOKEN` (newer installs may name them
`UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`; either works).
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from cryptography.fernet import Fernet, InvalidToken

from selector.spotify.auth import RELOGIN_HINT, SpotifyAuthError, TokenSet

URL_ENV_VARS = ("KV_REST_API_URL", "UPSTASH_REDIS_REST_URL")
TOKEN_ENV_VARS = ("KV_REST_API_TOKEN", "UPSTASH_REDIS_REST_TOKEN")
KEY_ENV_VAR = "SELECTOR_TOKEN_KEY"

TOKEN_KEY = "spotify:token"
LOCK_KEY = "spotify:refresh-lock"
OWNER_KEY = "spotify:owner"

# A refresh is one HTTPS round trip. Ten seconds is plenty, and a crashed
# holder can't block refreshes for longer than that.
LOCK_TTL_MS = 10_000
LOCK_WAIT_SECONDS = 5.0
LOCK_POLL_SECONDS = 0.25

# Delete the lock only if it's still ours, so a holder that outlived its
# TTL can't release someone else's lock.
_RELEASE_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)


class RemoteStoreNotConfigured(RuntimeError):
    """The Redis or encryption settings are missing."""


class RedisError(RuntimeError):
    pass


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return None


class UpstashRedis:
    """Just enough of Redis over Upstash's REST API: one command per call."""

    def __init__(self, url: str, token: str, http: httpx.Client | None = None) -> None:
        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._http = http or httpx.Client(timeout=10.0)

    @classmethod
    def from_env(cls) -> UpstashRedis:
        url, token = _first_env(URL_ENV_VARS), _first_env(TOKEN_ENV_VARS)
        if not url or not token:
            raise RemoteStoreNotConfigured(
                "Redis isn't configured: connect the Upstash integration to the MCP project"
            )
        return cls(url, token)

    def command(self, *args: Any) -> Any:
        response = self._http.post(self.url, headers=self._headers, json=[str(a) for a in args])
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200 or "error" in body:
            # The command itself is never echoed: it may hold a ciphertext.
            raise RedisError(f"Redis {args[0]} failed: HTTP {response.status_code} {body.get('error', '')}")
        return body.get("result")


class RedisTokenStore:
    """A `TokenStore` in one encrypted Redis key, with a Redis lock around
    refreshes so concurrent requests on reused instances don't race."""

    def __init__(self, redis: UpstashRedis, fernet_key: str) -> None:
        self.redis = redis
        try:
            self._fernet = Fernet(fernet_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise RemoteStoreNotConfigured(f"{KEY_ENV_VAR} isn't a valid Fernet key") from exc

    @classmethod
    def from_env(cls, redis: UpstashRedis | None = None) -> RedisTokenStore:
        key = os.environ.get(KEY_ENV_VAR)
        if not key:
            raise RemoteStoreNotConfigured(f"{KEY_ENV_VAR} isn't set")
        return cls(redis or UpstashRedis.from_env(), key)

    def load(self) -> TokenSet | None:
        blob = self.redis.command("GET", TOKEN_KEY)
        if blob is None:
            return None
        try:
            data = json.loads(self._fernet.decrypt(blob.encode("ascii")))
        except InvalidToken as exc:
            raise SpotifyAuthError(
                f"The stored Spotify token can't be decrypted ({KEY_ENV_VAR} changed?). {RELOGIN_HINT}"
            ) from exc
        return TokenSet.from_dict(data)

    def save(self, token: TokenSet) -> None:
        blob = self._fernet.encrypt(json.dumps(token.to_dict()).encode("utf-8")).decode("ascii")
        self.redis.command("SET", TOKEN_KEY, blob)

    @contextmanager
    def refresh_lock(self) -> Iterator[None]:
        nonce = secrets.token_hex(16)
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        held = False
        while True:
            if self.redis.command("SET", LOCK_KEY, nonce, "NX", "PX", LOCK_TTL_MS) == "OK":
                held = True
                break
            if time.monotonic() >= deadline:
                # Proceed unlocked rather than fail the call: the caller
                # reloads the token first, so the likely outcome is that the
                # holder already refreshed it.
                break
            time.sleep(LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            if held:
                self.redis.command("EVAL", _RELEASE_SCRIPT, 1, LOCK_KEY, nonce)
