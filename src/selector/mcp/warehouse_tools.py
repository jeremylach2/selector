"""The read-only warehouse tools, kept free of heavy imports.

Both MCP entrypoints register these same functions: `server.py` (stdio,
alongside the Spotify, fly-brain, DJ and Rewind tools) and
`http_server.py` (the Vercel deployment, which serves only these). This
module imports nothing beyond `requirements.txt`, the Vercel function's
whole dependency list. `server.py` pulls in scipy (via the fly brain) and
the Spotify client, and importing it from the deployment once crashed
every request with `No module named 'scipy'`.
`tests/test_deploy_imports.py` keeps it that way.

The warehouse path comes from the `SELECTOR_DB` env var, defaulting to
`./data/selector.duckdb`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from selector.warehouse import queries

DB_ENV_VAR = "SELECTOR_DB"
DEFAULT_DB_PATH = Path("data/selector.duckdb")
MAX_ROWS = 40

INSTRUCTIONS = (
    "Selector exposes a personal Spotify listening-history warehouse. "
    "Call `warehouse_summary` first to orient yourself on the date range "
    "and scale of the data before running more specific queries."
)


def _db_path() -> Path:
    return Path(os.environ.get(DB_ENV_VAR, str(DEFAULT_DB_PATH)))


def _missing_db_message(db_path: Path) -> str:
    return (
        f"No warehouse found at `{db_path}`. It hasn't been built yet. Run:\n\n"
        "```\n"
        "uv run python -m selector.ingest.load_history\n"
        "uv run python -m selector.warehouse.build\n"
        "```\n\n"
        "then retry this call."
    )


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        rounded = round(value, 3)
        if rounded == int(rounded):
            return f"{int(rounded):,}"
        return f"{rounded:,.3f}".rstrip("0").rstrip(".")
    if hasattr(value, "isoformat"):
        return str(value)[:19]
    return str(value)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    if df.empty:
        return "_No rows matched._"
    truncated = len(df) > max_rows
    shown = df.head(max_rows)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(v) for v in row) + " |")
    table = "\n".join(lines)
    if truncated:
        table += f"\n\n… {len(df) - max_rows} more rows"
    return table


def _run(fn: Callable[..., pd.DataFrame], **kwargs: Any) -> str:
    """Call a warehouse query function against the configured db, as markdown.

    Missing warehouse and query errors both come back as a plain message
    instead of a stack trace, since the caller is a model that should be
    able to recover (or tell the user what to run) rather than fail loudly.
    """
    db_path = _db_path()
    if not db_path.exists():
        return _missing_db_message(db_path)
    try:
        df = fn(db_path=db_path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: surface any query
        # failure to the calling model as text instead of letting it become a
        # stack trace on the MCP transport.
        return f"Query failed: {exc}"
    return _df_to_markdown(df)


def warehouse_summary() -> str:
    """Get the shape of the listening-history warehouse: date range, total
    plays, unique tracks and artists, and total hours listened.

    Call this first when starting a conversation about this person's
    listening history, before running any more specific query, so you know
    the scale and time span of the data you're working with.
    """
    return _run(queries.warehouse_summary)


def search_library(query: str, limit: int = 20) -> str:
    """Search the listening history for tracks or artists whose name
    contains `query` (case-insensitive substring match), ranked by play
    count. Use this to find a `track_id` before calling `track_detail`, or
    to check whether an artist appears in this person's history at all.
    """
    return _run(queries.search_library, query=query, limit=limit)


def track_detail(track_id_or_name: str) -> str:
    """Get full stats for one track: play count, skip rate, mean completion,
    net verdict (reward minus punishment across all plays), first and last
    played. Accepts either an exact `track_id` (from `search_library`) or a
    track name substring, in which case the highest-play-count match is
    returned.
    """
    return _run(queries.track_detail, track_id_or_name=track_id_or_name)


def top_artists(start: str | None = None, end: str | None = None, limit: int = 20) -> str:
    """List the most-played artists by play count and total hours, over an
    optional date range. `start` and `end` are ISO date strings like
    "2024-01-01"; omit either for an open-ended range. Use this to answer
    "who did I listen to most" for a period, or with no range at all for
    all-time favourites.
    """
    return _run(queries.top_artists, start=start, end=end, limit=limit)


def binged_then_abandoned(min_plays: int = 15, window_months: int = 3) -> str:
    """Find artists with a sharp one-month listening spike (at least
    `min_plays` plays in a single calendar month) followed by near-silence
    for the next `window_months` months. This answers questions shaped like
    "what did I binge and then abandon" or "what phases did my taste go
    through that didn't last".
    """
    return _run(queries.binged_then_abandoned, min_plays=min_plays, window_months=window_months)


def skip_offenders(min_plays: int = 5) -> str:
    """Find tracks that get played often but also skipped often, songs this
    person keeps queuing up and then bailing on. `min_plays` filters out
    tracks with too few plays to have a meaningful skip rate.
    """
    return _run(queries.skip_offenders, min_plays=min_plays)


def listening_clock() -> str:
    """Get play counts broken down by hour of day and day of week (UTC), to
    answer questions about when listening actually happens: late-night
    habits, weekday-vs-weekend patterns, commute-time spikes.
    """
    return _run(queries.listening_clock)


def taste_drift(granularity: str = "quarter", top_n: int = 5) -> str:
    """Get the top artists per time period, to show how taste evolved.
    `granularity` is "month", "quarter", or "year". Use a coarser
    granularity (quarter or year) for a broad multi-year overview, and
    "month" only for a short, recent window. Otherwise the table gets long.
    """
    return _run(queries.taste_drift, granularity=granularity, top_n=top_n)


def rediscovery_candidates(dormant_months: int = 6, min_past_plays: int = 10) -> str:
    """Find tracks that were played often in the past (at least
    `min_past_plays` times) but haven't been played in at least
    `dormant_months` months, songs this person used to love and forgot
    about. Good for "what should I revisit" style questions.
    """
    return _run(
        queries.rediscovery_candidates,
        dormant_months=dormant_months,
        min_past_plays=min_past_plays,
    )


# In registration order: what `tools/list` shows first.
WAREHOUSE_TOOLS = (
    warehouse_summary,
    search_library,
    track_detail,
    top_artists,
    binged_then_abandoned,
    skip_offenders,
    listening_clock,
    taste_drift,
    rediscovery_candidates,
)
