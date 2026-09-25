"""The DJ agent: Brief -> Arc -> Select -> Critique (-> Select -> Critique)
-> Commit, with every stage's output logged.

Each stage is its own module with a plain-data output, so the chain is
inspectable end to end: `data/dj_runs/<timestamp>-<theme>.json` holds the
brief, the arc curve, every Select step's shortlist, each critique verdict,
the revision (if there was one) and the commit result, with the liner notes
beside it as `.md`. That log is the demo: it shows a plan being made,
checked, and -- when the check fails -- revised.

Critique may send the set back to Select **once**. If the revision also
fails, the run still returns its best set and notes (useful as a dry run),
but Commit refuses to write it to Spotify.

Entry points: the `dj_set` MCP tool, and `scripts/dj_cron.py` for the
shell and scheduled runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from selector.dj.arc import EnergyArc
from selector.dj.brief import Brief, build_brief
from selector.dj.commit import CommitResult, commit, liner_notes, playlist_title
from selector.dj.critique import Verdict, critique, revised_config
from selector.dj.pool import Crate, recent_plays
from selector.dj.select import SelectConfig, Selection, select
from selector.spotify.client import SpotifyClient
from selector.warehouse.build import DEFAULT_DB_PATH

DJ_RUNS_DIR = Path("data/dj_runs")
MAX_REVISIONS = 1


@dataclass
class DJRun:
    brief: Brief
    arc: EnergyArc
    attempts: list[tuple[Selection, Verdict]]
    notes: str
    result: CommitResult
    recent_source: str
    log_path: Path | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def final(self) -> tuple[Selection, Verdict]:
        return self.attempts[-1]

    def to_dict(self) -> dict:
        return {
            "brief": self.brief.to_dict(),
            "recent_source": self.recent_source,
            "arc": self.arc.to_dict(),
            "attempts": [
                {"attempt": i + 1, "selection": s.to_dict(), "critique": v.to_dict()}
                for i, (s, v) in enumerate(self.attempts)
            ],
            "revised": len(self.attempts) > 1,
            "final_passed": self.final[1].passed,
            "commit": vars(self.result),
            "errors": self.errors,
            "liner_notes": self.notes,
        }


def gather_recent(client: SpotifyClient | None, db_path: Path, limit: int = 50) -> tuple[pd.DataFrame, str]:
    """Live recently-played from Spotify when a cached token exists (never
    triggers a browser login from here), else the newest warehouse plays."""
    if client is not None and client.token_path.exists():
        try:
            items = client.recently_played(limit=min(limit, 50)).get("items", [])
            rows = [
                {
                    "track_id": (it.get("track") or {}).get("id"),
                    "artist_name": ((it.get("track") or {}).get("artists") or [{}])[0].get("name"),
                    "ts": it.get("played_at"),
                }
                for it in items
            ]
            live = pd.DataFrame(rows).dropna(subset=["track_id"])
            if not live.empty:
                return live, "spotify_live"
        except Exception as exc:  # noqa: BLE001 - live context is a nicety; the warehouse always works
            return recent_plays(db_path, limit=limit), f"warehouse (live Spotify failed: {exc})"
    return recent_plays(db_path, limit=limit), "warehouse"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def write_log(run: DJRun, now: datetime, log_dir: Path = DJ_RUNS_DIR) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%Y%m%d-%H%M%S}-{_slug(run.brief.theme.name)}"
    path = log_dir / f"{stem}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
    (log_dir / f"{stem}.md").write_text(run.notes, encoding="utf-8")
    return path


def run_dj(
    crate: Crate,
    theme: str | None = None,
    minutes: float = 45,
    dry_run: bool = True,
    familiar_ratio: float = 0.6,
    now: datetime | None = None,
    client: SpotifyClient | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    recent: pd.DataFrame | None = None,
    log_dir: Path | None = DJ_RUNS_DIR,
) -> DJRun:
    now = now or datetime.now().astimezone()
    recent_source = "given"
    if recent is None:
        recent, recent_source = gather_recent(client, db_path)

    brief = build_brief(recent, crate, now, requested_theme=theme)
    arc = EnergyArc.for_theme(brief.theme, minutes)

    config = SelectConfig(familiar_ratio=familiar_ratio)
    selection = select(crate, brief, arc, config)
    verdict = critique(selection, arc)
    attempts = [(selection, verdict)]

    exclude: set[str] = set()
    for _ in range(MAX_REVISIONS):
        if verdict.passed:
            break
        exclude |= set(verdict.revision_exclude)
        config = revised_config(config)
        selection = select(crate, brief, arc, config, exclude=exclude)
        verdict = critique(selection, arc)
        attempts.append((selection, verdict))

    # Commit decides the title; build it here too so notes can carry it on a
    # dry run without a second code path.
    title = playlist_title(brief, now)
    notes = liner_notes(brief, selection, verdict, title)

    errors: list[str] = []
    try:
        result = commit(brief, selection, verdict, now, client, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - logged with the run, then re-raised below
        errors.append(str(exc))
        result = CommitResult(dry_run=dry_run, title=title, description="")

    run = DJRun(brief, arc, attempts, notes, result, recent_source, errors=errors)
    if log_dir is not None:
        run.log_path = write_log(run, now, log_dir)
    return run


def render(run: DJRun) -> str:
    """Human-readable summary of a run: what the critique did, then the notes."""
    lines = []
    for i, (_, v) in enumerate(run.attempts):
        label = "First pass" if i == 0 else "Revision"
        lines.append(f"- **{label}:** {v.summary}")
        for issue in v.issues[:6]:
            where = f"track {issue.slot + 1}" if issue.slot >= 0 else "whole set"
            lines.append(f"  - {where}: {issue.kind.replace('_', ' ')} - {issue.detail}")
    header = "## Critique chain\n\n" + "\n".join(lines) + "\n\n"

    if run.errors:
        status = f"**Not committed:** {run.errors[0]}"
    elif run.result.dry_run:
        status = "**Dry run** - nothing was written to Spotify. Re-run with `dry_run=False` to create it."
    else:
        status = f"**Created playlist:** {run.result.playlist_url}"
    footer = f"\n{status}"
    if run.log_path:
        footer += f"\n\n_Full chain logged to `{run.log_path}`._"
    return header + run.notes + footer
