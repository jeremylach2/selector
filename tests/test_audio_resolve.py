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
