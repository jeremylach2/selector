"""A thin, typed client over the Spotify Web API endpoints that survived the
November 2024 deprecation.

Deliberately does not implement audio-features, audio-analysis,
recommendations, related-artists, or 30-second preview URLs. Spotify
removed all of those for new apps, and writing code against them would just
be a 404 waiting to happen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from selector.spotify.auth import DEFAULT_TOKEN_PATH, TokenStore, get_valid_token

API_BASE = "https://api.spotify.com/v1"

MAX_RETRIES = 4
DEFAULT_RETRY_AFTER_SECONDS = 1.0


class SpotifyAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Spotify API error {status_code}: {message}")
        self.status_code = status_code


@dataclass
class RequestLogEntry:
    method: str
    path: str
    status_code: int
    attempt: int
    elapsed_ms: float


@dataclass
class SpotifyClient:
    """One client per process is enough, it re-authenticates lazily and
    caches the token in memory for the life of the object.

    `store` overrides `token_path` (the hosted server passes a Redis
    store), and `interactive=False` means a missing token is an error
    rather than a browser login. `max_retry_after` caps how long a 429 may
    make it sleep: past that it raises instead, so one throttled call
    can't outlast a serverless function's time limit.
    """

    client_id: str
    token_path: Path = DEFAULT_TOKEN_PATH
    store: TokenStore | None = None
    interactive: bool = True
    max_retry_after: float | None = None
    request_log: list[RequestLogEntry] = field(default_factory=list)
    _http: httpx.Client = field(default_factory=lambda: httpx.Client(base_url=API_BASE, timeout=15.0))

    def _auth_headers(self) -> dict[str, str]:
        token = get_valid_token(
            self.client_id, self.store or self.token_path, interactive=self.interactive
        )
        return {"Authorization": f"Bearer {token.access_token}"}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        last_response: httpx.Response | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            start = time.monotonic()
            response = self._http.request(method, path, headers=self._auth_headers(), **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000
            self.request_log.append(
                RequestLogEntry(method, path, response.status_code, attempt, elapsed_ms)
            )
            last_response = response

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_SECONDS))
                if self.max_retry_after is not None and retry_after > self.max_retry_after:
                    raise SpotifyAPIError(429, f"rate limited, retry in {retry_after:.0f}s")
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                time.sleep(2 ** (attempt - 1))
                continue
            break

        assert last_response is not None
        if last_response.status_code >= 400:
            detail = last_response.text
            try:
                detail = last_response.json().get("error", {}).get("message", detail)
            except ValueError:
                pass
            raise SpotifyAPIError(last_response.status_code, detail)

        if not last_response.content:
            return {}
        return last_response.json()

    # -- read endpoints -----------------------------------------------------

    def current_user(self) -> dict:
        return self._request("GET", "/me")

    def search(self, query: str, types: str = "track", limit: int = 10) -> dict:
        return self._request(
            "GET", "/search", params={"q": query, "type": types, "limit": limit}
        )

    def saved_tracks(self, limit: int = 50, offset: int = 0) -> dict:
        return self._request(
            "GET", "/me/tracks", params={"limit": min(limit, 50), "offset": offset}
        )

    def all_saved_tracks(self, max_items: int = 2000) -> list[dict]:
        """Page through `/me/tracks` until exhausted or `max_items` is hit."""
        items: list[dict] = []
        offset = 0
        while len(items) < max_items:
            page = self.saved_tracks(limit=50, offset=offset)
            batch = page.get("items", [])
            if not batch:
                break
            items.extend(batch)
            offset += len(batch)
            if page.get("next") is None:
                break
        return items[:max_items]

    def top_items(self, item_type: str, time_range: str = "medium_term", limit: int = 20) -> dict:
        if item_type not in ("artists", "tracks"):
            raise ValueError("item_type must be 'artists' or 'tracks'")
        return self._request(
            "GET",
            f"/me/top/{item_type}",
            params={"time_range": time_range, "limit": limit},
        )

    def recently_played(self, limit: int = 20) -> dict:
        return self._request("GET", "/me/player/recently-played", params={"limit": limit})

    # -- write endpoints ------------------------------------------------

    def create_playlist(
        self, name: str, description: str = "", public: bool = False, track_uris: list[str] | None = None
    ) -> dict:
        # Spotify's February 2026 Web API migration removed
        # `/users/{user_id}/playlists` for Development Mode apps, it now
        # returns a bare 403 regardless of scope. `/me/playlists` is the
        # replacement and needs no separate `current_user()` lookup.
        playlist = self._request(
            "POST",
            "/me/playlists",
            json={"name": name, "description": description, "public": public},
        )
        if track_uris:
            self.add_tracks_to_playlist(playlist["id"], track_uris)
        return playlist

    def add_tracks_to_playlist(self, playlist_id: str, track_uris: list[str]) -> dict:
        # The API caps a single add at 100 URIs; chunk anything larger.
        # `/playlists/{id}/tracks` is the other half of the same Feb 2026
        # migration that killed `/users/{id}/playlists`, the replacement is
        # `/playlists/{id}/items`, same method and body shape.
        result: dict = {}
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i : i + 100]
            result = self._request(
                "POST", f"/playlists/{playlist_id}/items", json={"uris": chunk}
            )
        return result

    def unfollow_playlist(self, playlist_id: str) -> dict:
        # Unfollowing your own playlist is how the Web API deletes it.
        return self._request("DELETE", f"/playlists/{playlist_id}/followers")
