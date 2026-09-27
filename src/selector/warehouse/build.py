"""Build the DuckDB taste warehouse from the ingested plays Parquet table.

Safe to re-run: every table is created with CREATE OR REPLACE, so a rebuild
from an updated `plays.parquet` just recomputes everything in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

DEFAULT_PLAYS_PATH = Path("data/plays.parquet")
DEFAULT_DB_PATH = Path("data/selector.duckdb")

# A play more than this many minutes after the previous one starts a new
# listening session.
SESSION_GAP_MINUTES = 30


def build_warehouse(
    plays_path: Path = DEFAULT_PLAYS_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE plays AS SELECT * FROM read_parquet(?)",
            [str(plays_path)],
        )
        _create_derived_tables(con)
        counts = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM plays) AS plays,
                (SELECT COUNT(*) FROM tracks) AS tracks,
                (SELECT COUNT(*) FROM artists) AS artists,
                (SELECT COUNT(*) FROM sessions) AS sessions,
                (SELECT COUNT(*) FROM artist_months) AS artist_months
            """
        ).fetchone()

    labels = ["plays", "tracks", "artists", "sessions", "artist_months"]
    for label, count in zip(labels, counts):
        print(f"{label}: {count:,} rows")


def build_window_warehouse(
    source_db: Path,
    db_path: Path,
    start: str | None,
    end: str | None,
    timezone: str = "UTC",
) -> Path:
    """A warehouse over only the plays in `[start, end)`, with the same
    derived tables as the full one, so every query in
    `selector.warehouse.queries` works on a single window unchanged. Used by
    the Wrapped reports. Rebuilt from scratch on each call.

    `start`/`end` are local dates in `timezone` (an IANA name), and the
    derived tables' month buckets (`artist_months`) are cut in that zone
    too, rather than in whatever zone the machine happens to be in.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    if not all(part.replace("_", "").isalnum() for part in timezone.split("/")):
        raise ValueError(f"not an IANA time zone name: {timezone!r}")
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"SET TimeZone = '{timezone}'")
        con.execute(f"ATTACH '{Path(source_db).as_posix()}' AS src (READ_ONLY)")
        con.execute(
            """
            CREATE TABLE plays AS SELECT * FROM src.plays
            WHERE (? IS NULL OR ts >= ?::TIMESTAMPTZ) AND (? IS NULL OR ts < ?::TIMESTAMPTZ)
            """,
            [start, start, end, end],
        )
        con.execute("DETACH src")
        _create_derived_tables(con)
    return db_path


def _create_derived_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Every table derived from `plays`, which must already exist on `con`."""
    # skip_rate here means "share of plays on this track/artist that ended
    # in a forward skip" (reason_end = 'fwdbtn'). That's narrower and more
    # legible than the plays-level verdict, which also folds in abandoned
    # endplay rows.
    con.execute(
        """
        CREATE OR REPLACE TABLE tracks AS
        SELECT
            track_id,
            any_value(track_name) AS name,
            any_value(artist_name) AS artist,
            any_value(album_name) AS album,
            MIN(ts) AS first_played,
            MAX(ts) AS last_played,
            COUNT(*) AS play_count,
            SUM(ms_played) AS total_ms,
            AVG(CASE WHEN reason_end = 'fwdbtn' THEN 1.0 ELSE 0.0 END) AS skip_rate,
            AVG(completion) AS mean_completion,
            SUM(verdict) AS net_verdict
        FROM plays
        GROUP BY track_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE artists AS
        SELECT
            artist_name AS artist,
            COUNT(DISTINCT track_id) AS track_count,
            COUNT(*) AS play_count,
            SUM(ms_played) / 3600000.0 AS total_hours,
            MIN(ts) AS first_played,
            MAX(ts) AS last_played,
            AVG(CASE WHEN reason_end = 'fwdbtn' THEN 1.0 ELSE 0.0 END) AS skip_rate
        FROM plays
        GROUP BY artist_name
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE sessions AS
        WITH ordered AS (
            SELECT *, ts - LAG(ts) OVER (ORDER BY ts) AS gap_from_prev
            FROM plays
        ),
        flagged AS (
            SELECT *,
                CASE
                    WHEN gap_from_prev IS NULL
                        OR gap_from_prev > INTERVAL '{SESSION_GAP_MINUTES} minutes'
                    THEN 1 ELSE 0
                END AS starts_session
            FROM ordered
        ),
        numbered AS (
            SELECT *,
                SUM(starts_session) OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS session_id
            FROM flagged
        )
        SELECT
            session_id,
            MIN(ts) AS started_at,
            MAX(ts) AS ended_at,
            COUNT(*) AS n_tracks,
            SUM(CASE WHEN reason_end = 'fwdbtn' THEN 1 ELSE 0 END) AS n_skips,
            SUM(ms_played) AS total_ms,
            mode(artist_name) AS dominant_artist
        FROM numbered
        GROUP BY session_id
        ORDER BY session_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE artist_months AS
        SELECT
            artist_name AS artist,
            date_trunc('month', ts) AS month,
            COUNT(*) AS play_count,
            SUM(ms_played) AS total_ms
        FROM plays
        GROUP BY artist_name, date_trunc('month', ts)
        ORDER BY artist, month
        """
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plays-path", type=Path, default=DEFAULT_PLAYS_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)
    build_warehouse(args.plays_path, args.db_path)


if __name__ == "__main__":
    main()
