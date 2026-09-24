import pandas as pd

from selector.audio import metadata


def test_representative_albums_dedupes_to_one_row_per_album(tmp_path, monkeypatch):
    matches = pd.DataFrame(
        [
            {
                "track_id": "t1",
                "local_path": "data/audio/t1.mp3",
                "source_id": "111",
                "match_source": "deezer",
            },
            {
                "track_id": "t2",
                "local_path": "data/audio/t2.mp3",
                "source_id": "222",
                "match_source": "deezer",
            },
            {
                "track_id": "t3",
                "local_path": None,  # unmatched, should be excluded
                "source_id": None,
                "match_source": None,
            },
        ]
    )
    matches_path = tmp_path / "audio_matches.parquet"
    matches.to_parquet(matches_path)

    tracks = pd.DataFrame(
        [
            {"track_id": "t1", "artist": "The Beatles", "album": "Abbey Road"},
            {"track_id": "t2", "artist": "the beatles", "album": "Abbey Road (Remastered)"},
            {"track_id": "t3", "artist": "The Beatles", "album": "Let It Be"},
        ]
    )

    class FakeConnection:
        def __init__(self, df):
            self.df_ = df

        def execute(self, *_args, **_kwargs):
            return self

        def df(self):
            return self.df_

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(metadata, "_connect", lambda db_path: FakeConnection(tracks))

    reps = metadata.representative_albums(matches_path, db_path=tmp_path / "unused.duckdb")

    # t1 and t2 are edition variants of the same album and collapse to one row
    assert len(reps) == 1
    assert reps.iloc[0]["album_key"] == "the beatles|abbey road"


def test_checkpoint_round_trips(tmp_path):
    cache_path = tmp_path / "checkpoint.jsonl"
    rows = [
        metadata.AlbumRelease(
            album_key="the beatles|abbey road",
            artist="The Beatles",
            album="Abbey Road",
            release_date="1969-09-26",
            release_year=1969,
            source="deezer",
        ),
        metadata.AlbumRelease(
            album_key="unknown artist|unknown album",
            artist="Unknown Artist",
            album="Unknown Album",
            release_date=None,
            release_year=None,
            source=None,
        ),
    ]
    metadata._append_checkpoint(cache_path, rows)

    done = metadata._load_checkpoint(cache_path)
    assert set(done) == {"the beatles|abbey road", "unknown artist|unknown album"}
    assert done["the beatles|abbey road"]["release_year"] == 1969


def test_coverage_report_handles_empty():
    assert "No albums" in metadata.coverage_report(pd.DataFrame())


def test_coverage_report_counts_by_source():
    df = pd.DataFrame(
        [
            {"release_year": 1969, "source": "deezer"},
            {"release_year": 1972, "source": "itunes"},
            {"release_year": None, "source": None},
        ]
    )
    report = metadata.coverage_report(df)
    assert "66.7%" in report or "2/3" in report
    assert "deezer: 1" in report
    assert "itunes: 1" in report
