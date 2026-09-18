from selector.ingest.load_history import _records_to_frame


def _record(**overrides) -> dict:
    base = {
        "ts": "2025-06-01T12:00:00Z",
        "platform": "windows",
        "ms_played": 100_000,
        "conn_country": "US",
        "ip_addr": "203.0.113.5",
        "master_metadata_track_name": "Track A",
        "master_metadata_album_artist_name": "Artist A",
        "master_metadata_album_album_name": "Album A",
        "spotify_track_uri": "spotify:track:abc123",
        "episode_name": None,
        "episode_show_name": None,
        "spotify_episode_uri": None,
        "audiobook_title": None,
        "audiobook_uri": None,
        "audiobook_chapter_uri": None,
        "audiobook_chapter_title": None,
        "reason_start": "playbtn",
        "reason_end": "trackdone",
        "shuffle": False,
        "skipped": False,
        "offline": False,
        "offline_timestamp": 1765651773,
        "incognito_mode": False,
    }
    base.update(overrides)
    return base


def test_drops_personal_fields():
    frame = _records_to_frame([_record()])
    assert "ip_addr" not in frame.columns
    assert "offline_timestamp" not in frame.columns
    assert "incognito_mode" not in frame.columns


def test_drops_rows_without_track_uri():
    frame = _records_to_frame([_record(), _record(spotify_track_uri=None)])
    assert len(frame) == 1


def test_renames_metadata_columns():
    frame = _records_to_frame([_record()])
    assert frame.loc[0, "track_name"] == "Track A"
    assert frame.loc[0, "artist_name"] == "Artist A"
    assert frame.loc[0, "album_name"] == "Album A"


def test_extracts_track_id():
    frame = _records_to_frame([_record()])
    assert frame.loc[0, "track_id"] == "abc123"


def test_verdict_trackdone_is_reward():
    frame = _records_to_frame([_record(reason_end="trackdone")])
    assert frame.loc[0, "verdict"] == 1


def test_verdict_backbtn_is_weak_reward():
    frame = _records_to_frame([_record(reason_end="backbtn")])
    assert frame.loc[0, "verdict"] == 1


def test_verdict_fwdbtn_below_threshold_is_punishment():
    records = [
        _record(spotify_track_uri="spotify:track:full", ms_played=200_000, reason_end="trackdone"),
        _record(spotify_track_uri="spotify:track:full", ms_played=50_000, reason_end="fwdbtn"),
    ]
    frame = _records_to_frame(records)
    skipped = frame[frame["reason_end"] == "fwdbtn"].iloc[0]
    assert skipped["completion"] < 0.8
    assert skipped["verdict"] == -1


def test_verdict_fwdbtn_above_threshold_is_neutral():
    records = [
        _record(spotify_track_uri="spotify:track:full", ms_played=200_000, reason_end="trackdone"),
        _record(spotify_track_uri="spotify:track:full", ms_played=190_000, reason_end="fwdbtn"),
    ]
    frame = _records_to_frame(records)
    near_complete_skip = frame[frame["reason_end"] == "fwdbtn"].iloc[0]
    assert near_complete_skip["completion"] >= 0.8
    assert near_complete_skip["verdict"] == 0


def test_verdict_endplay_low_completion_is_punishment():
    records = [
        _record(spotify_track_uri="spotify:track:full", ms_played=200_000, reason_end="trackdone"),
        _record(spotify_track_uri="spotify:track:full", ms_played=10_000, reason_end="endplay"),
    ]
    frame = _records_to_frame(records)
    row = frame[frame["reason_end"] == "endplay"].iloc[0]
    assert row["completion"] < 0.3
    assert row["verdict"] == -1


def test_verdict_endplay_high_completion_is_neutral():
    records = [
        _record(spotify_track_uri="spotify:track:full", ms_played=200_000, reason_end="trackdone"),
        _record(spotify_track_uri="spotify:track:full", ms_played=150_000, reason_end="endplay"),
    ]
    frame = _records_to_frame(records)
    row = frame[frame["reason_end"] == "endplay"].iloc[0]
    assert row["completion"] >= 0.3
    assert row["verdict"] == 0


def test_verdict_unhandled_reason_is_neutral():
    frame = _records_to_frame([_record(reason_end="logout")])
    assert frame.loc[0, "verdict"] == 0


def test_completion_capped_at_one():
    # A play can exceed the observed max if it's the only play of that track.
    # Completion should never exceed 1.0 regardless.
    frame = _records_to_frame([_record()])
    assert frame.loc[0, "completion"] <= 1.0
