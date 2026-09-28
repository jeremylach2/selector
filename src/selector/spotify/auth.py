"""Authorization Code with PKCE against the Spotify Web API.

Only the surviving endpoints (search, library, top items, playlists,
playback) are ever called through the token this produces. Audio-features,
audio-analysis, recommendations, related-artists, and preview URLs were all
removed for new apps in November 2024 and nothing here assumes they exist.

Flow: open the system browser to Spotify's authorize page, catch the
redirect on a local loopback server, exchange the code (plus the PKCE
verifier) for tokens, and cache them to ``~/.selector/token.json``. Later
calls silently refresh an expired access token using the cached refresh
token, so only the very first run needs a browser.

Where the token lives is a `TokenStore`. Locally that's `FileTokenStore`
(the JSON file above). The hosted MCP server uses
`selector.spotify.remote_store.RedisTokenStore` with `interactive=False`,
since a serverless function has no browser and no loopback port: there,
a missing or dead token is an error telling you to log in again through
the MCP client, never a browser flow.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar, Protocol
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

# Registered as the sole redirect URI on the Spotify dev app. Must match
# exactly, port included, or the authorize call is rejected.
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = 8899
REDIRECT_URI = f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}/callback"

SCOPES = (
    "user-library-read user-top-read playlist-modify-private playlist-modify-public "
    "playlist-read-private user-read-recently-played"
)

# What the hosted MCP server's own grant asks for (seeded by
# `scripts/seed_remote_spotify_token.py`): reads, plus creating playlists
# kept off the profile. No public-playlist writes, no reading private
# playlists.
REMOTE_SCOPES = (
    "user-library-read user-top-read user-read-recently-played playlist-modify-private"
)

DEFAULT_TOKEN_PATH = Path.home() / ".selector" / "token.json"

# Refresh this many seconds before actual expiry, so a token doesn't die
# mid-request due to clock skew or request latency.
EXPIRY_SAFETY_MARGIN_SECONDS = 60


# What a non-interactive caller (the hosted server) should do about a
# missing or dead token.
RELOGIN_HINT = (
    "Reconnect the MCP client to log in through Spotify again, or re-seed the token: "
    "uv run python scripts/seed_remote_spotify_token.py"
)


class SpotifyAuthError(RuntimeError):
    """Raised when the PKCE flow or a token refresh fails."""


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at: float  # unix timestamp
    scope: str

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at - EXPIRY_SAFETY_MARGIN_SECONDS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TokenSet:
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            scope=data.get("scope", ""),
        )


def _generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures exactly one redirect from Spotify, then lets the server stop."""

    result: ClassVar[dict[str, str]] = {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        _CallbackHandler.result = params

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if "error" in params:
            body = f"<h1>Spotify auth failed</h1><p>{params['error']}</p><p>You can close this tab.</p>"
        else:
            body = "<h1>Selector is connected</h1><p>You can close this tab and go back to Claude.</p>"
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        pass  # silence the default stderr access log


def _run_authorize_flow(client_id: str, scopes: str = SCOPES) -> dict[str, str]:
    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": scopes,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    authorize_url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    _CallbackHandler.result = {}
    httpd = HTTPServer((LOOPBACK_HOST, LOOPBACK_PORT), _CallbackHandler)
    webbrowser.open(authorize_url)
    httpd.handle_request()  # blocks for exactly one request, then returns
    httpd.server_close()

    result = _CallbackHandler.result
    if "error" in result:
        raise SpotifyAuthError(f"Spotify returned an error: {result['error']}")
    if result.get("state") != state:
        raise SpotifyAuthError("OAuth state mismatch — possible CSRF, aborting")
    if "code" not in result:
        raise SpotifyAuthError("No authorization code in Spotify's redirect")

    return {"code": result["code"], "verifier": verifier}


def _exchange_code_for_token(
    client_id: str,
    code: str,
    verifier: str,
    redirect_uri: str = REDIRECT_URI,
    scopes: str = SCOPES,
) -> TokenSet:
    """`redirect_uri` must be the one the authorize request used: the
    loopback one here, the hosted callback for the MCP OAuth login."""
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )
    if response.status_code != 200:
        raise SpotifyAuthError(f"Token exchange failed: {response.status_code} {response.text}")
    payload = response.json()
    return TokenSet(
        access_token=payload["access_token"],
        refresh_token=payload["refresh_token"],
        expires_at=time.time() + payload["expires_in"],
        scope=payload.get("scope", scopes),
    )


def _refresh_token(client_id: str, refresh_token: str) -> TokenSet:
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
    )
    if response.status_code != 200:
        raise SpotifyAuthError(f"Token refresh failed: {response.status_code} {response.text}")
    payload = response.json()
    return TokenSet(
        access_token=payload["access_token"],
        # Spotify doesn't always return a new refresh token; keep the old one.
        refresh_token=payload.get("refresh_token", refresh_token),
        expires_at=time.time() + payload["expires_in"],
        scope=payload.get("scope", SCOPES),
    )


def _load_cached_token(token_path: Path) -> TokenSet | None:
    if not token_path.exists():
        return None
    try:
        return TokenSet.from_dict(json.loads(token_path.read_text()))
    except (json.JSONDecodeError, KeyError):
        return None


def _save_token(token_path: Path, token: TokenSet) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(token.to_dict(), indent=2))
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass  # best-effort on platforms without POSIX permissions (Windows)


class TokenStore(Protocol):
    """Somewhere a `TokenSet` survives between calls."""

    def load(self) -> TokenSet | None: ...

    def save(self, token: TokenSet) -> None: ...

    def refresh_lock(self) -> AbstractContextManager[None]:
        """Held while refreshing, so concurrent callers don't all refresh."""
        ...


class FileTokenStore:
    """The local cache, `~/.selector/token.json` by default. One process
    at a time uses it, so its lock is a no-op."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH) -> None:
        self.path = Path(path)

    def load(self) -> TokenSet | None:
        return _load_cached_token(self.path)

    def save(self, token: TokenSet) -> None:
        _save_token(self.path, token)

    @contextmanager
    def refresh_lock(self) -> Iterator[None]:
        yield


def get_valid_token(
    client_id: str,
    store: TokenStore | Path = DEFAULT_TOKEN_PATH,
    interactive: bool = True,
) -> TokenSet:
    """Return a usable access token, refreshing or running the full PKCE
    browser flow as needed. This is the only function most callers need.

    With `interactive=False` the browser flow never runs: no token, or a
    refresh token Spotify no longer accepts, raises `SpotifyAuthError`.
    """
    if isinstance(store, (str, Path)):
        store = FileTokenStore(Path(store))

    cached = store.load()
    if cached is not None and not cached.is_expired():
        return cached

    if cached is not None:
        with store.refresh_lock():
            # Another request may have refreshed while this one waited.
            cached = store.load() or cached
            if not cached.is_expired():
                return cached
            try:
                refreshed = _refresh_token(client_id, cached.refresh_token)
                store.save(refreshed)
                return refreshed
            except SpotifyAuthError as exc:
                if not interactive:
                    raise SpotifyAuthError(
                        f"{exc}. {RELOGIN_HINT}"
                    ) from exc
                # refresh token itself expired/revoked: fall through to a full re-auth

    if not interactive:
        raise SpotifyAuthError(f"No Spotify token is stored. {RELOGIN_HINT}")

    auth_result = _run_authorize_flow(client_id)
    token = _exchange_code_for_token(client_id, auth_result["code"], auth_result["verifier"])
    store.save(token)
    return token
