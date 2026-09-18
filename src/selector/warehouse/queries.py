"""Named, parameterised queries over the Selector taste warehouse.

Each function opens its own read-only connection and returns a pandas
DataFrame, so these compose cleanly as MCP tool implementations later.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from selector.warehouse.build import DEFAULT_DB_PATH

# Thresholds not exposed as function parameters, kept in one place so they're
# easy to tune without hunting through the query bodies.
QUERY_THRESHOLDS = {
    # A spike month's follow-up window counts as "abandoned" if it has at
    # most this many plays.
    "abandon_max_plays": 1,
    "skip_offender_min_skip_rate": 0.3,
}


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def top_artists(
    start: str | None = None,
    end: str | None = None,
    limit: int = 20,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Which artists got the most plays in a given date range?

    `start`/`end` are ISO date strings (inclusive/exclusive); either may be
    None for an open-ended range.
    """
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT
                artist_name AS artist,
                COUNT(*) AS play_count,
                SUM(ms_played) / 3600000.0 AS total_hours
            FROM plays
            WHERE (? IS NULL OR ts >= ?) AND (? IS NULL OR ts < ?)
            GROUP BY artist_name
            ORDER BY play_count DESC
            LIMIT ?
            """,
            [start, start, end, end, limit],
        ).df()


def binged_then_abandoned(
    min_plays: int = 15,
    window_months: int = 3,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Which artists had a sharp one-month play spike followed by near-silence?

    A "spike" is a calendar month with at least `min_plays` plays for that
    artist. "Abandoned" means the following `window_months` calendar months
    together have at most `QUERY_THRESHOLDS["abandon_max_plays"]` plays.
    """
    with _connect(db_path) as con:
        artist_months = con.execute(
            "SELECT artist, month, play_count FROM artist_months ORDER BY artist, month"
        ).df()

    artist_months["month"] = pd.to_datetime(artist_months["month"])

    rows = []
    for artist, group in artist_months.groupby("artist", sort=False):
        group = group.sort_values("month")
        for _, spike in group[group["play_count"] >= min_plays].iterrows():
            window_start = spike["month"] + pd.DateOffset(months=1)
            window_end = spike["month"] + pd.DateOffset(months=window_months)
            in_window = group[(group["month"] >= window_start) & (group["month"] <= window_end)]
            followup_plays = int(in_window["play_count"].sum())
            if followup_plays <= QUERY_THRESHOLDS["abandon_max_plays"]:
                rows.append(
                    {
                        "artist": artist,
                        "spike_month": spike["month"].date(),
                        "spike_plays": int(spike["play_count"]),
                        "followup_plays": followup_plays,
                    }
                )

    columns = ["artist", "spike_month", "spike_plays", "followup_plays"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("spike_plays", ascending=False)
        .reset_index(drop=True)
    )


def skip_offenders(
    min_plays: int = 5,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Which tracks get played a lot but skipped a lot, i.e. songs you keep queuing up and bailing on?"""
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT track_id, name, artist, play_count, skip_rate
            FROM tracks
            WHERE play_count >= ? AND skip_rate >= ?
            ORDER BY skip_rate DESC, play_count DESC
            """,
            [min_plays, QUERY_THRESHOLDS["skip_offender_min_skip_rate"]],
        ).df()


def listening_clock(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """When during the week does listening actually happen, by hour of day and day of week?"""
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT hour_utc, dow, COUNT(*) AS play_count
            FROM plays
            GROUP BY hour_utc, dow
            ORDER BY dow, hour_utc
            """
        ).df()


VALID_DRIFT_GRANULARITIES = {"month", "quarter", "year"}


def taste_drift(
    granularity: str = "quarter",
    top_n: int = 5,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """How has top-artist taste evolved over time, one ranked list per period?

    `granularity` is 'month', 'quarter', or 'year'.
    """
    if granularity not in VALID_DRIFT_GRANULARITIES:
        raise ValueError(f"granularity must be one of {VALID_DRIFT_GRANULARITIES}")

    # `granularity` is interpolated directly (not bound as a `?` parameter)
    # because DuckDB won't match two independently-parameterised
    # `date_trunc(?, ts)` expressions between SELECT and GROUP BY. Safe here
    # since it's checked against the allow-list above first.
    with _connect(db_path) as con:
        return con.execute(
            f"""
            WITH per_period AS (
                SELECT
                    date_trunc('{granularity}', ts) AS period,
                    artist_name AS artist,
                    COUNT(*) AS play_count
                FROM plays
                GROUP BY date_trunc('{granularity}', ts), artist_name
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY period ORDER BY play_count DESC
                ) AS rank
                FROM per_period
            )
            SELECT period, rank, artist, play_count
            FROM ranked
            WHERE rank <= ?
            ORDER BY period, rank
            """,
            [top_n],
        ).df()


def rediscovery_candidates(
    dormant_months: int = 6,
    min_past_plays: int = 10,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Which tracks were loved once, played often, but haven't been touched in a while?"""
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT track_id, name, artist, play_count, last_played
            FROM tracks
            WHERE play_count >= ?
                AND last_played < (SELECT MAX(last_played) FROM tracks) - INTERVAL (?) MONTH
            ORDER BY play_count DESC
            """,
            [min_past_plays, dormant_months],
        ).df()


def search_library(
    query: str,
    limit: int = 20,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Fuzzy substring search over track and artist names, ranked by play count."""
    pattern = f"%{query.lower()}%"
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT track_id, name, artist, album, play_count, skip_rate, net_verdict
            FROM tracks
            WHERE lower(name) LIKE ? OR lower(artist) LIKE ?
            ORDER BY play_count DESC
            LIMIT ?
            """,
            [pattern, pattern, limit],
        ).df()


def track_detail(
    track_id_or_name: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Full stats for one track, matched by exact `track_id` first, then by name substring.

    Returns at most one row: the exact `track_id` match if there is one, else the
    highest-play-count track whose name contains `track_id_or_name`.
    """
    with _connect(db_path) as con:
        exact = con.execute(
            "SELECT * FROM tracks WHERE track_id = ?", [track_id_or_name]
        ).df()
        if not exact.empty:
            return exact
        return con.execute(
            """
            SELECT * FROM tracks
            WHERE lower(name) LIKE ?
            ORDER BY play_count DESC
            LIMIT 1
            """,
            [f"%{track_id_or_name.lower()}%"],
        ).df()


def warehouse_summary(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Date range, total plays, unique tracks/artists, and total hours listened."""
    with _connect(db_path) as con:
        return con.execute(
            """
            SELECT
                MIN(ts) AS earliest_play,
                MAX(ts) AS latest_play,
                COUNT(*) AS total_plays,
                COUNT(DISTINCT track_id) AS unique_tracks,
                COUNT(DISTINCT artist_name) AS unique_artists,
                SUM(ms_played) / 3600000.0 AS total_hours
            FROM plays
            """
        ).df()


