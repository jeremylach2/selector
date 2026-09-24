"""Wrapped-style report cards over the Selector warehouse.

Implements the "v1 — warehouse only" slice of the Wrapped extension plan
(see `Selector - Project Extension Plan.md`, stages 0-1 and 6-8): pure local
SQL, no network calls, no dependency on the vibe tagger or fly brain. Later
slices (listening age, taste clusters, hidden gems, archetype) need album
release years and/or trained Components B/C and are not implemented here.

`build_report()` assembles the versioned report object; each `_card_*`
function is a pure function that returns a card dict or `None` if it can't
be computed, matching stage 6 of the plan ("a card that can't be computed
returns None rather than raising — the assembler drops it").
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from selector.warehouse import queries
from selector.warehouse.build import DEFAULT_DB_PATH

SCHEMA_VERSION = "1.0"

# Fixed per the plan's "Card order is fixed in config, not emergent" rule:
# open with volume, then rankings, then the artist sprint and listening
# habits. Later slices (listening age, taste clusters, archetype) slot in
# after `skip_offenders` and before the archetype card once they exist.
CARD_ORDER = [
    "total_hours",
    "top_artists",
    "top_tracks",
    "top_albums",
    "artist_sprint",
    "time_of_day",
    "skip_offenders",
]


def freeze_config(top_n: int = 5, **overrides: Any) -> dict[str, Any]:
    """Stage 0: one hashed config dict everything downstream is keyed by."""
    config = {"schema_version": SCHEMA_VERSION, "top_n": top_n, **overrides}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[
        :8
    ]
    return {**config, "config_hash": config_hash}


def _warehouse_totals(db_path: Path) -> dict[str, Any]:
    df = queries.warehouse_summary(db_path=db_path)
    row = df.iloc[0]
    earliest, latest = row["earliest_play"], row["latest_play"]
    return {
        "window": {
            "from": str(earliest)[:10] if pd.notna(earliest) else None,
            "to": str(latest)[:10] if pd.notna(latest) else None,
        },
        "totals": {
            "plays": int(row["total_plays"]),
            "hours": round(float(row["total_hours"]), 1),
            "tracks": int(row["unique_tracks"]),
            "artists": int(row["unique_artists"]),
        },
    }


def _card_total_hours(totals: dict[str, Any]) -> dict[str, Any] | None:
    t = totals["totals"]
    if not t["hours"]:
        return None
    return {
        "id": "total_hours",
        "headline": f"You listened for {t['hours']:,.0f} hours",
        "value": t["hours"],
        "sublabel": f"across {t['plays']:,} plays",
        "evidence": [f"{t['tracks']:,} unique tracks", f"{t['artists']:,} unique artists"],
    }


def _card_top_artists(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    top = df.iloc[0]
    return {
        "id": "top_artists",
        "headline": f"Your #1 artist was {top['artist']}",
        "value": [
            {"artist": r.artist, "play_count": int(r.play_count)} for r in df.itertuples()
        ],
        "sublabel": f"{int(top['play_count']):,} plays",
        "evidence": [
            f"#{i + 1} {r.artist} ({int(r.play_count):,} plays)"
            for i, r in enumerate(df.head(3).itertuples())
        ],
    }


def _card_top_tracks(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    top = df.iloc[0]
    return {
        "id": "top_tracks",
        "headline": f"Your #1 track was \"{top['name']}\" by {top['artist']}",
        "value": [
            {"name": r.name, "artist": r.artist, "play_count": int(r.play_count)}
            for r in df.itertuples()
        ],
        "sublabel": f"{int(top['play_count']):,} plays",
        "evidence": [
            f"#{i + 1} \"{r.name}\" — {r.artist} ({int(r.play_count):,} plays)"
            for i, r in enumerate(df.head(3).itertuples())
        ],
    }


def _card_top_albums(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    top = df.iloc[0]
    return {
        "id": "top_albums",
        "headline": f"Your #1 album was \"{top['album']}\" by {top['artist']}",
        "value": [
            {"album": r.album, "artist": r.artist, "play_count": int(r.play_count)}
            for r in df.itertuples()
        ],
        "sublabel": f"{int(top['play_count']):,} plays",
        "evidence": [
            f"#{i + 1} \"{r.album}\" — {r.artist} ({int(r.play_count):,} plays)"
            for i, r in enumerate(df.head(3).itertuples())
        ],
    }


def _card_artist_sprint(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    finals = df.sort_values("month").groupby("artist", as_index=False).tail(1)
    finals = finals.sort_values("cumulative_plays", ascending=False)
    leader = finals.iloc[0]
    return {
        "id": "artist_sprint",
        "headline": f"{leader['artist']} led your longest artist sprint",
        "value": [
            {
                "artist": r.artist,
                "month": str(r.month.date()),
                "play_count": int(r.play_count),
                "cumulative_plays": int(r.cumulative_plays),
            }
            for r in df.itertuples()
        ],
        "sublabel": f"{len(finals)} artists ever cracked a monthly top spot",
        "evidence": [
            f"{r.artist}: {int(r.cumulative_plays):,} plays total"
            for r in finals.head(3).itertuples()
        ],
    }


def _card_time_of_day(db_path: Path) -> dict[str, Any] | None:
    clock = queries.listening_clock(db_path=db_path)
    if clock.empty:
        return None
    by_hour = clock.groupby("hour_utc", as_index=False)["play_count"].sum()
    by_hour = by_hour.sort_values("play_count", ascending=False)
    peak = by_hour.iloc[0]
    return {
        "id": "time_of_day",
        "headline": f"You listen most around {int(peak['hour_utc']):02d}:00 UTC",
        "value": int(peak["hour_utc"]),
        "sublabel": f"{int(peak['play_count']):,} plays in that hour, all-time",
        "evidence": [
            f"{int(r.hour_utc):02d}:00 UTC — {int(r.play_count):,} plays"
            for r in by_hour.head(3).itertuples()
        ],
    }


def _card_skip_offenders(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    worst = df.iloc[0]
    return {
        "id": "skip_offenders",
        "headline": f"You keep queuing up and bailing on \"{worst['name']}\"",
        "value": [
            {
                "name": r.name,
                "artist": r.artist,
                "play_count": int(r.play_count),
                "skip_rate": round(float(r.skip_rate), 3),
            }
            for r in df.itertuples()
        ],
        "sublabel": f"{worst['skip_rate']:.0%} skip rate over {int(worst['play_count'])} plays",
        "evidence": [
            f"\"{r.name}\" — {r.artist} ({r.skip_rate:.0%} skipped)"
            for r in df.head(3).itertuples()
        ],
    }


def build_report(top_n: int = 5, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Stages 1 + 6 + 7: aggregate the warehouse, build every v1 card, and
    assemble the versioned report object. Raises FileNotFoundError if the
    warehouse hasn't been built yet; never raises for an individual card
    failing — that card is just dropped.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No warehouse found at {db_path}")

    config = freeze_config(top_n=top_n)
    totals = _warehouse_totals(db_path)

    cards_by_id = {
        "total_hours": _card_total_hours(totals),
        "top_artists": _card_top_artists(queries.top_artists(limit=top_n, db_path=db_path)),
        "top_tracks": _card_top_tracks(queries.top_tracks(limit=top_n, db_path=db_path)),
        "top_albums": _card_top_albums(queries.top_albums(limit=top_n, db_path=db_path)),
        "artist_sprint": _card_artist_sprint(
            queries.artist_sprint(top_n=top_n, db_path=db_path)
        ),
        "time_of_day": _card_time_of_day(db_path),
        "skip_offenders": _card_skip_offenders(queries.skip_offenders(db_path=db_path)),
    }
    cards = [cards_by_id[cid] for cid in CARD_ORDER if cards_by_id.get(cid) is not None]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config["config_hash"],
        "window": totals["window"],
        "totals": totals["totals"],
        "cards": cards,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Stage 8, chat renderer: the report object as a short markdown story."""
    if not report["cards"]:
        return "_No cards could be computed — is the warehouse built and populated?_"

    lines = [
        f"# Your Selector Wrapped ({report['window']['from']} → {report['window']['to']})",
        "",
        (
            f"*{report['totals']['plays']:,} plays · {report['totals']['hours']:,.0f} hours · "
            f"{report['totals']['tracks']:,} tracks · {report['totals']['artists']:,} artists*"
        ),
        "",
    ]
    for card in report["cards"]:
        lines.append(f"## {card['headline']}")
        if card.get("sublabel"):
            lines.append(f"*{card['sublabel']}*")
        for point in card.get("evidence", []):
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_html(report: dict[str, Any]) -> str:
    """Stage 8, standalone HTML story renderer — one static page, one card
    per section, no JS framework. Exercises the report schema the way a
    real (non-chat) consumer eventually will, per the plan's "de-risk the
    demo" rationale for building the HTML render alongside v1.
    """
    window = f"{report['window']['from']} → {report['window']['to']}"
    totals = report["totals"]
    cards_html = []
    for card in report["cards"]:
        evidence_html = "".join(f"<li>{e}</li>" for e in card.get("evidence", []))
        cards_html.append(
            f"""<section class="card">
  <h2>{card['headline']}</h2>
  <p class="sublabel">{card.get('sublabel', '')}</p>
  <ul>{evidence_html}</ul>
</section>"""
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Selector Wrapped</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto;
         padding: 0 1rem; background: #0d0d12; color: #f2f2f5; }}
  h1 {{ font-size: 1.4rem; }}
  .totals {{ color: #a0a0ad; margin-bottom: 2rem; }}
  .card {{ background: #17171f; border-radius: 12px; padding: 1.25rem 1.5rem;
           margin-bottom: 1rem; }}
  .card h2 {{ margin: 0 0 0.25rem; font-size: 1.15rem; }}
  .sublabel {{ color: #a0a0ad; margin: 0 0 0.75rem; }}
  ul {{ margin: 0; padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>Your Selector Wrapped</h1>
<p class="totals">{window} · {totals['plays']:,} plays · {totals['hours']:,.0f} hours · \
{totals['tracks']:,} tracks · {totals['artists']:,} artists</p>
{''.join(cards_html)}
</body>
</html>
"""
