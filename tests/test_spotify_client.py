"""Unit tests for the Spotify API client: retry/backoff behaviour and
request shaping, against a mocked HTTP layer (respx) with no real network
calls and no auth flow triggered.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from selector.spotify import client as client_module
from selector.spotify.client import SpotifyAPIError, SpotifyClient


@pytest.fixture(autouse=True)
def _no_real_auth(monkeypatch):
    """Every test gets a fake bearer token, none of this touches auth.py."""
    monkeypatch.setattr(
        SpotifyClient, "_auth_headers", lambda self: {"Authorization": "Bearer fake"}
    )


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Retry backoff would otherwise make the retry tests slow for real."""
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)


@pytest.fixture
def client():
    return SpotifyClient(client_id="test-client-id")


@respx.mock
def test_search_returns_json(client):
    route = respx.get("https://api.spotify.com/v1/search").mock(
        return_value=httpx.Response(200, json={"tracks": {"items": []}})
    )
    result = client.search("some track", types="track", limit=5)
    assert route.called
    assert result == {"tracks": {"items": []}}
    assert respx.calls.last.request.url.params["q"] == "some track"


@respx.mock
def test_retries_on_429_then_succeeds(client):
    route = respx.get("https://api.spotify.com/v1/me").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={}),
            httpx.Response(200, json={"id": "user1"}),
        ]
    )
    result = client.current_user()
    assert result == {"id": "user1"}
    assert route.call_count == 2


@respx.mock
def test_retries_on_5xx_then_succeeds(client):
    route = respx.get("https://api.spotify.com/v1/me").mock(
        side_effect=[
            httpx.Response(500, json={}),
            httpx.Response(502, json={}),
            httpx.Response(200, json={"id": "user1"}),
        ]
    )
    result = client.current_user()
    assert result == {"id": "user1"}
    assert route.call_count == 3


@respx.mock
def test_raises_after_exhausting_retries(client):
    respx.get("https://api.spotify.com/v1/me").mock(return_value=httpx.Response(500, text="down"))
    with pytest.raises(SpotifyAPIError) as excinfo:
        client.current_user()
    assert excinfo.value.status_code == 500


@respx.mock
def test_raises_spotify_api_error_on_4xx_with_message(client):
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(403, json={"error": {"message": "insufficient scope"}})
    )
    with pytest.raises(SpotifyAPIError, match="insufficient scope"):
        client.current_user()


@respx.mock
def test_all_saved_tracks_pages_until_next_is_null(client):
    respx.get("https://api.spotify.com/v1/me/tracks").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "items": [{"track": {"id": "t1"}}, {"track": {"id": "t2"}}],
                    "next": "https://api.spotify.com/v1/me/tracks?offset=2",
                },
            ),
            httpx.Response(200, json={"items": [{"track": {"id": "t3"}}], "next": None}),
        ]
    )
    items = client.all_saved_tracks()
    assert [i["track"]["id"] for i in items] == ["t1", "t2", "t3"]


@respx.mock
def test_create_playlist_chunks_track_uris_over_100(client):
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(201, json={"id": "playlist1"})
    )
    tracks_route = respx.post("https://api.spotify.com/v1/playlists/playlist1/items").mock(
        return_value=httpx.Response(201, json={"snapshot_id": "abc"})
    )

    uris = [f"spotify:track:{i}" for i in range(150)]
    playlist = client.create_playlist("My Set", track_uris=uris)

    assert playlist == {"id": "playlist1"}
    assert tracks_route.call_count == 2
    first_body = tracks_route.calls[0].request.content
    assert b'"uris"' in first_body
