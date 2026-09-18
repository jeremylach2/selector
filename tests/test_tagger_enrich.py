from __future__ import annotations

import httpx
import respx

from selector.tagger import enrich as enrich_module
from selector.tagger.enrich import fetch_lyrics


@respx.mock
def test_fetch_lyrics_returns_plain_lyrics(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(
        return_value=httpx.Response(200, json=[{"plainLyrics": "la la la"}])
    )

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track1", "Some Song", "Some Artist")

    assert lyrics == "la la la"
    assert (tmp_path / "track1.txt").read_text(encoding="utf-8") == "la la la"


@respx.mock
def test_fetch_lyrics_returns_none_when_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(return_value=httpx.Response(200, json=[]))

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track2", "Unknown Song", "Unknown Artist")

    assert lyrics is None
    assert (tmp_path / "track2.txt").read_text(encoding="utf-8") == ""


@respx.mock
def test_fetch_lyrics_uses_cache_without_a_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    (tmp_path / "track3.txt").write_text("cached lyrics", encoding="utf-8")
    # No route registered - respx raises if fetch_lyrics tries a real call,
    # so a cache hit is the only way this test can pass.

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track3", "irrelevant", "irrelevant")

    assert lyrics == "cached lyrics"
