"""MCP OAuth for the hosted server, with Spotify as the login.

Replaces the static `SELECTOR_MCP_TOKEN` bearer (Phase 2 of
`docs/REMOTE_SPOTIFY_PLAN.md`). The server is its own OAuth authorization
server, built on the SDK's `mcp.server.auth` handlers, and this module is
the provider behind them:

1. A client without a token gets a 401 pointing at
   `/.well-known/oauth-protected-resource/mcp`, registers itself
   (`/register`) and sends the user to `/authorize` with PKCE.
2. `authorize` parks the request in Redis and sends the browser to
   Spotify's authorize page with `REMOTE_SCOPES`. The Spotify leg is PKCE
   too, with its verifier kept in the parked request, so no Spotify client
   secret is needed.
3. Spotify redirects to `/oauth/spotify/callback`. The server exchanges the
   code, calls `GET /me`, and refuses unless the id is
   `SELECTOR_OWNER_SPOTIFY_ID`. On success the Spotify token set replaces
   the one in `RedisTokenStore` (so logging in is also how the Spotify tools
   get a fresh token), and the client gets an MCP authorization code.
4. `/token` swaps the code for a 1 hour access token and a 30 day refresh
   token. Refresh tokens rotate on use.

Everything lives in the same Upstash Redis as the Spotify token, under
`mcp:*` keys with TTLs. Keys are SHA-256 hashes of the tokens, so no token
is stored in plain text, and every value is Fernet-encrypted with
`SELECTOR_TOKEN_KEY`, since client records can hold a client secret.

Registration only accepts redirect URIs on an allowlist (claude.ai's
callback and loopback addresses). Without that, anyone could register a
client with their own redirect URI and send the owner a login link that,
since Spotify skips its consent page for an app already approved, would
hand them an MCP token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode, urlparse

import anyio.to_thread
import httpx
from cryptography.fernet import Fernet, InvalidToken
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from selector.spotify import auth
from selector.spotify.remote_store import (
    KEY_ENV_VAR,
    OWNER_KEY,
    RedisTokenStore,
    RemoteStoreNotConfigured,
    UpstashRedis,
)

OWNER_ENV_VAR = "SELECTOR_OWNER_SPOTIFY_ID"
PUBLIC_URL_ENV_VAR = "SELECTOR_MCP_PUBLIC_URL"
# Set by Vercel on every deployment: the production domain, no scheme.
VERCEL_PRODUCTION_URL_ENV_VAR = "VERCEL_PROJECT_PRODUCTION_URL"
# The static bearer from before OAuth. While it's set, it's accepted
# alongside OAuth tokens, so clients can move over one at a time.
LEGACY_BEARER_ENV_VAR = "SELECTOR_MCP_TOKEN"
EXTRA_REDIRECTS_ENV_VAR = "SELECTOR_OAUTH_REDIRECT_URIS"

CALLBACK_PATH = "/oauth/spotify/callback"
MCP_PATH = "/mcp"
ME_URL = "https://api.spotify.com/v1/me"

ACCESS_TTL = 3600
REFRESH_TTL = 30 * 86400
CODE_TTL = 300
PENDING_TTL = 600
CLIENT_TTL = 90 * 86400
# /register is open to anyone, so cap how many Redis keys it can create.
MAX_REGISTRATIONS_PER_DAY = 20

LEGACY_CLIENT_ID = "static-bearer"

CLAUDE_REDIRECTS = frozenset(
    {
        "https://claude.ai/api/mcp/auth_callback",
        "https://claude.com/api/mcp/auth_callback",
    }
)
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

PREFIX = "mcp:"
CLIENT_PREFIX = PREFIX + "client:"
PENDING_PREFIX = PREFIX + "pending:"
CODE_PREFIX = PREFIX + "code:"
ACCESS_PREFIX = PREFIX + "access:"
REFRESH_PREFIX = PREFIX + "refresh:"
REGISTRATIONS_PREFIX = PREFIX + "registrations:"
SESSION_PREFIXES = (PENDING_PREFIX, CODE_PREFIX, ACCESS_PREFIX, REFRESH_PREFIX)


class StoredAccessToken(AccessToken):
    refresh_hash: str | None = None


class StoredRefreshToken(RefreshToken):
    access_hash: str | None = None


@dataclass(frozen=True)
class OAuthConfig:
    public_url: str  # e.g. https://selector-mcp.vercel.app, no trailing slash
    spotify_client_id: str
    owner_id: str
    extra_redirects: frozenset[str] = frozenset()

    @property
    def callback_url(self) -> str:
        return self.public_url + CALLBACK_PATH

    @property
    def resource_url(self) -> str:
        return self.public_url + MCP_PATH

    def auth_settings(self) -> AuthSettings:
        return AuthSettings(
            issuer_url=self.public_url,
            resource_server_url=self.resource_url,
            # `authorize` refuses a `resource` other than this server, and
            # only this server issues tokens, so there's nothing to check.
            validate_token_resource=False,
            client_registration_options=ClientRegistrationOptions(enabled=True),
            revocation_options=RevocationOptions(enabled=True),
        )


def public_url_from_env() -> str | None:
    explicit = os.environ.get(PUBLIC_URL_ENV_VAR)
    if explicit:
        return explicit.rstrip("/")
    domain = os.environ.get(VERCEL_PRODUCTION_URL_ENV_VAR)
    return f"https://{domain}" if domain else None


def config_from_env() -> OAuthConfig:
    values = {
        "public URL": public_url_from_env(),
        "SPOTIFY_CLIENT_ID": os.environ.get("SPOTIFY_CLIENT_ID"),
        OWNER_ENV_VAR: os.environ.get(OWNER_ENV_VAR),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RemoteStoreNotConfigured(f"OAuth isn't configured, missing: {', '.join(missing)}")
    extra = frozenset(os.environ.get(EXTRA_REDIRECTS_ENV_VAR, "").split())
    return OAuthConfig(
        public_url=values["public URL"],
        spotify_client_id=values["SPOTIFY_CLIENT_ID"],
        owner_id=values[OWNER_ENV_VAR],
        extra_redirects=extra,
    )


def redirect_allowed(uri: str, extra: frozenset[str] = frozenset()) -> bool:
    """claude.ai's callback, anything on a loopback address (Claude Code,
    MCP Inspector), or an exact URI from `SELECTOR_OAUTH_REDIRECT_URIS`."""
    if uri in CLAUDE_REDIRECTS or uri in extra:
        return True
    parsed = urlparse(uri)
    return parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _same_resource(a: str, b: str) -> bool:
    return a.rstrip("/").lower() == b.rstrip("/").lower()


def _fetch_spotify_user_id(access_token: str) -> str:
    response = httpx.get(ME_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
    response.raise_for_status()
    return response.json()["id"]


class SpotifyOAuthProvider:
    """`OAuthAuthorizationServerProvider` backed by Redis, logging in
    through Spotify. Only `config.owner_id` can get a token."""

    def __init__(
        self,
        redis: UpstashRedis,
        fernet_key: str,
        config: OAuthConfig,
        legacy_bearer: str | None = None,
    ) -> None:
        self.redis = redis
        self.config = config
        self.legacy_bearer = legacy_bearer or None
        self.token_store = RedisTokenStore(redis, fernet_key)
        self._fernet = Fernet(fernet_key.encode("ascii"))

    @classmethod
    def from_env(cls) -> SpotifyOAuthProvider:
        """Raises `RemoteStoreNotConfigured` if anything is missing, so the
        caller can fail closed."""
        config = config_from_env()
        key = os.environ.get(KEY_ENV_VAR)
        if not key:
            raise RemoteStoreNotConfigured(f"{KEY_ENV_VAR} isn't set")
        return cls(UpstashRedis.from_env(), key, config, os.environ.get(LEGACY_BEARER_ENV_VAR))

    # -- Redis helpers --------------------------------------------------------

    async def _cmd(self, *args: Any) -> Any:
        # The Upstash client is synchronous; keep it off the event loop.
        return await anyio.to_thread.run_sync(lambda: self.redis.command(*args))

    async def _put(self, key: str, value: dict, ttl: int) -> None:
        blob = self._fernet.encrypt(json.dumps(value).encode("utf-8")).decode("ascii")
        await self._cmd("SET", key, blob, "EX", ttl)

    def _decode(self, blob: str | None) -> dict | None:
        if blob is None:
            return None
        try:
            return json.loads(self._fernet.decrypt(blob.encode("ascii")))
        except InvalidToken:
            # Written under an older SELECTOR_TOKEN_KEY: treat as gone.
            return None

    async def _get(self, key: str) -> dict | None:
        return self._decode(await self._cmd("GET", key))

    async def _take(self, key: str) -> dict | None:
        """Read and delete in one step, so a value can be used only once."""
        return self._decode(await self._cmd("GETDEL", key))

    # -- clients --------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        data = await self._get(CLIENT_PREFIX + client_id)
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            if not redirect_allowed(str(uri), self.config.extra_redirects):
                raise RegistrationError(
                    "invalid_redirect_uri",
                    f"{uri} isn't an allowed redirect URI for this server",
                )
        day_key = REGISTRATIONS_PREFIX + datetime.now(UTC).date().isoformat()
        count = int(await self._cmd("INCR", day_key))
        if count == 1:
            await self._cmd("EXPIRE", day_key, 2 * 86400)
        if count > MAX_REGISTRATIONS_PER_DAY:
            raise RegistrationError("invalid_client_metadata", "Too many client registrations today")
        await self._put(CLIENT_PREFIX + client_info.client_id, client_info.model_dump(mode="json"), CLIENT_TTL)

    # -- authorize, and the Spotify leg ----------------------------------------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if params.resource and not _same_resource(params.resource, self.config.resource_url):
            raise AuthorizeError("invalid_target", "This server only issues tokens for its own /mcp endpoint")
        verifier, challenge = auth._generate_pkce_pair()
        state = secrets.token_urlsafe(32)
        pending = {
            "client_id": client.client_id,
            "params": params.model_dump(mode="json"),
            "verifier": verifier,
        }
        await self._put(PENDING_PREFIX + _hash(state), pending, PENDING_TTL)
        query = {
            "client_id": self.config.spotify_client_id,
            "response_type": "code",
            "redirect_uri": self.config.callback_url,
            "scope": auth.REMOTE_SCOPES,
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        return f"{auth.AUTHORIZE_URL}?{urlencode(query)}"

    async def handle_spotify_callback(self, request: Request) -> Response:
        """Where Spotify sends the browser back. Ends with a redirect to the
        MCP client, carrying either an authorization code or an error."""
        state = request.query_params.get("state", "")
        pending = await self._take(PENDING_PREFIX + _hash(state)) if state else None
        if pending is None:
            return HTMLResponse(
                "<h1>This login link expired or was already used</h1>"
                "<p>Start again from your MCP client.</p>",
                status_code=400,
            )
        params = AuthorizationParams.model_validate(pending["params"])

        def back(**query: str | None) -> Response:
            url = construct_redirect_uri(str(params.redirect_uri), state=params.state, **query)
            return RedirectResponse(url, status_code=302)

        code = request.query_params.get("code")
        if request.query_params.get("error") or not code:
            return back(error="access_denied", error_description="Spotify login was cancelled or failed")

        try:
            token = await anyio.to_thread.run_sync(
                lambda: auth._exchange_code_for_token(
                    self.config.spotify_client_id,
                    code,
                    pending["verifier"],
                    redirect_uri=self.config.callback_url,
                    scopes=auth.REMOTE_SCOPES,
                )
            )
            user_id = await anyio.to_thread.run_sync(lambda: _fetch_spotify_user_id(token.access_token))
        except (auth.SpotifyAuthError, httpx.HTTPError, KeyError, ValueError):
            return back(error="server_error", error_description="Couldn't complete the Spotify login")

        if not hmac.compare_digest(user_id.encode("utf-8"), self.config.owner_id.encode("utf-8")):
            # Someone else's account: drop their token, never store it.
            return back(error="access_denied", error_description="This server only accepts its owner's Spotify account")

        await anyio.to_thread.run_sync(lambda: self.token_store.save(token))
        await self._cmd("SET", OWNER_KEY, user_id)

        mcp_code = secrets.token_urlsafe(32)
        record = AuthorizationCode(
            code=_hash(mcp_code),
            scopes=params.scopes or [],
            expires_at=time.time() + CODE_TTL,
            client_id=pending["client_id"],
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject=user_id,
        )
        await self._put(CODE_PREFIX + _hash(mcp_code), record.model_dump(mode="json"), CODE_TTL)
        return back(code=mcp_code)

    # -- codes and tokens ------------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        data = await self._get(CODE_PREFIX + _hash(authorization_code))
        if data is None or data["client_id"] != client.client_id:
            return None
        return AuthorizationCode.model_validate(data)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # `code` holds the hash. DEL returning 0 means another request
        # already spent it between load and exchange.
        if int(await self._cmd("DEL", CODE_PREFIX + authorization_code.code)) == 0:
            raise TokenError("invalid_grant", "Authorization code already used")
        await self._cmd("EXPIRE", CLIENT_PREFIX + client.client_id, CLIENT_TTL)
        return await self._issue(
            client.client_id, authorization_code.scopes, authorization_code.resource, authorization_code.subject
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> StoredRefreshToken | None:
        data = await self._get(REFRESH_PREFIX + _hash(refresh_token))
        if data is None or data["client_id"] != client.client_id:
            return None
        return StoredRefreshToken.model_validate(data)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: StoredRefreshToken, scopes: list[str]
    ) -> OAuthToken:
        # Rotate: the old refresh token works exactly once.
        if int(await self._cmd("DEL", REFRESH_PREFIX + refresh_token.token)) == 0:
            raise TokenError("invalid_grant", "Refresh token already used")
        if refresh_token.access_hash:
            await self._cmd("DEL", ACCESS_PREFIX + refresh_token.access_hash)
        await self._cmd("EXPIRE", CLIENT_PREFIX + client.client_id, CLIENT_TTL)
        return await self._issue(client.client_id, scopes, refresh_token.resource, refresh_token.subject)

    async def _issue(self, client_id: str, scopes: list[str], resource: str | None, subject: str | None) -> OAuthToken:
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        access_hash, refresh_hash = _hash(access), _hash(refresh)
        now = int(time.time())
        await self._put(
            ACCESS_PREFIX + access_hash,
            StoredAccessToken(
                token=access_hash,
                client_id=client_id,
                scopes=scopes,
                expires_at=now + ACCESS_TTL,
                resource=resource,
                subject=subject,
                refresh_hash=refresh_hash,
            ).model_dump(mode="json"),
            ACCESS_TTL,
        )
        await self._put(
            REFRESH_PREFIX + refresh_hash,
            StoredRefreshToken(
                token=refresh_hash,
                client_id=client_id,
                scopes=scopes,
                expires_at=now + REFRESH_TTL,
                resource=resource,
                subject=subject,
                access_hash=access_hash,
            ).model_dump(mode="json"),
            REFRESH_TTL,
        )
        return OAuthToken(
            access_token=access,
            expires_in=ACCESS_TTL,
            refresh_token=refresh,
            scope=" ".join(scopes) or None,
        )

    async def load_access_token(self, token: str) -> StoredAccessToken | None:
        if self.legacy_bearer and hmac.compare_digest(token.encode("utf-8"), self.legacy_bearer.encode("utf-8")):
            return StoredAccessToken(token=LEGACY_CLIENT_ID, client_id=LEGACY_CLIENT_ID, scopes=[])
        data = await self._get(ACCESS_PREFIX + _hash(token))
        if data is None:
            return None
        stored = StoredAccessToken.model_validate(data)
        if stored.expires_at is not None and stored.expires_at < time.time():
            return None
        return stored

    async def revoke_token(self, token: StoredAccessToken | StoredRefreshToken) -> None:
        if token.client_id == LEGACY_CLIENT_ID:
            return  # the static bearer is revoked by unsetting it
        if isinstance(token, StoredAccessToken):
            keys = [ACCESS_PREFIX + token.token]
            if token.refresh_hash:
                keys.append(REFRESH_PREFIX + token.refresh_hash)
        else:
            keys = [REFRESH_PREFIX + token.token]
            if token.access_hash:
                keys.append(ACCESS_PREFIX + token.access_hash)
        await self._cmd("DEL", *keys)


def revoke_all_sessions(redis: UpstashRedis, include_clients: bool = False) -> int:
    """Delete every MCP token, code and pending login (and, optionally,
    every registered client). Returns how many keys went."""
    prefixes = SESSION_PREFIXES + ((CLIENT_PREFIX,) if include_clients else ())
    deleted = 0
    for prefix in prefixes:
        cursor = "0"
        while True:
            cursor, keys = redis.command("SCAN", cursor, "MATCH", prefix + "*", "COUNT", 200)
            if keys:
                deleted += int(redis.command("DEL", *keys))
            if str(cursor) == "0":
                break
    return deleted


def mint_access_token(redis: UpstashRedis, fernet_key: str, subject: str, ttl: int = 300) -> tuple[str, str]:
    """An access token written straight to Redis, for
    `scripts/check_mcp_deploy.py`. Whoever holds the Redis credentials and
    `SELECTOR_TOKEN_KEY` already controls everything this could reach.
    Returns (token, its Redis key), so the caller can delete it after."""
    token = secrets.token_urlsafe(32)
    key = ACCESS_PREFIX + _hash(token)
    record = StoredAccessToken(
        token=_hash(token), client_id="check-script", scopes=[], expires_at=int(time.time()) + ttl, subject=subject
    )
    blob = Fernet(fernet_key.encode("ascii")).encrypt(record.model_dump_json().encode("utf-8")).decode("ascii")
    redis.command("SET", key, blob, "EX", ttl)
    return token, key
