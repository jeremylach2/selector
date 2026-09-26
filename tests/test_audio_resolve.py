from selector.audio.resolve import Candidate, normalize, score_candidate


def test_normalize_strips_parens_and_feat():
    cleaned, mods = normalize("Blinding Lights (feat. Someone) [Remastered 2011]")
    assert "feat" not in cleaned
    assert "remaster" not in cleaned
    assert "remaster" in mods


def test_normalize_extracts_modifier_tokens():
    _, mods = normalize("Wonderwall - Live at Wembley")
    assert mods == {"live"}


def test_score_candidate_exact_match_scores_high():
    candidate = Candidate(
        source="itunes",
        source_id="1",
        title="Blinding Lights",
        artist="The Weeknd",
        album="After Hours",
        preview_url="https://example.com/x.m4a",
    )
    assert score_candidate("Blinding Lights", "The Weeknd", candidate) > 0.9


def test_score_candidate_penalises_live_mismatch():
    studio = Candidate(
        source="itunes",
        source_id="1",
        title="Wonderwall",
        artist="Oasis",
        album="(What's the Story) Morning Glory?",
        preview_url="https://example.com/x.m4a",
    )
    live = Candidate(
        source="itunes",
        source_id="2",
        title="Wonderwall - Live",
        artist="Oasis",
        album="Live Forever",
        preview_url="https://example.com/y.m4a",
    )
    query_title, query_artist = "Wonderwall", "Oasis"
    assert score_candidate(query_title, query_artist, studio) > score_candidate(
        query_title, query_artist, live
    )


def test_score_candidate_different_song_scores_low():
    candidate = Candidate(
        source="itunes",
        source_id="1",
        title="Somewhere Only We Know",
        artist="Keane",
        album="Hopes and Fears",
        preview_url="https://example.com/x.m4a",
    )
    assert score_candidate("Blinding Lights", "The Weeknd", candidate) < 0.4


def test_resolve_tracks_checkpoints_only_real_verdicts(tmp_path, monkeypatch):
    # An API outage or failed download must be retried on the next run,
    # never remembered as "no match". A genuine no-match is remembered.
    import pandas as pd

    from selector.audio import resolve

    monkeypatch.setattr(resolve, "CHECKPOINT_PATH", tmp_path / "ckpt.jsonl")
    monkeypatch.setattr(
        resolve,
        "_top_tracks",
        lambda limit, db_path, track_ids=None: pd.DataFrame(
            {"track_id": ["ok", "nomatch", "outage", "nodl"], "name": ["a", "b", "c", "d"],
             "artist": ["x"] * 4, "play_count": [4, 3, 2, 1]}
        ),
    )
    good = Candidate(source="deezer", source_id="1", title="a", artist="x", album="", preview_url="u")
    verdicts = {"a": (good, 0.95, True), "b": (None, -1.0, True), "c": (None, -1.0, False), "d": (good, 0.95, True)}
    monkeypatch.setattr(resolve, "_best_candidate", lambda client, title, artist: verdicts[title])
    monkeypatch.setattr(
        resolve, "download_preview", lambda client, url, track_id: None if track_id == "nodl" else f"{track_id}.mp3"
    )

    matches = resolve.resolve_tracks(limit=4)

    assert sorted(matches["track_id"]) == ["nomatch", "ok"]
    assert matches.set_index("track_id").at["ok", "local_path"] == "ok.mp3"
