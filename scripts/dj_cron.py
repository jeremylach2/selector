"""Scheduled DJ runs: plan a set for right now and (with --commit) create it.

Dry run by default, same as the `dj_set` MCP tool, so a mis-scheduled task
can't fill the account with playlists. Every run, dry or not, is logged to
`data/dj_runs/`. Exits non-zero if a requested commit was refused (critique
failed twice, or no Spotify credentials), so the scheduler records it as a
failure.

Spotify auth must already be cached (`~/.selector/token.json`) before a
scheduled --commit run: this script never opens a browser login.

Windows Task Scheduler, every Friday at 18:00, from the repo root:

    schtasks /Create /TN "Selector DJ" /SC WEEKLY /D FRI /ST 18:00 ^
        /TR "cmd /c cd /d D:\\codeprojects\\spotifyProject && uv run python scripts\\dj_cron.py --commit"

cron equivalent:

    0 18 * * 5  cd /path/to/spotifyProject && uv run python scripts/dj_cron.py --commit
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from selector.dj.agent import render, run_dj
from selector.dj.pool import build_crate
from selector.spotify.auth import DEFAULT_TOKEN_PATH
from selector.spotify.client import SpotifyClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", default=None, help="named theme or mood words; default: chosen by the brief")
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--familiar-ratio", type=float, default=0.6)
    parser.add_argument("--commit", action="store_true", help="create the playlist (default is a dry run)")
    parser.add_argument("--at", default=None, help="plan as if it were this local time, e.g. 2026-09-25T22:00")
    args = parser.parse_args(argv)

    load_dotenv()
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if args.commit and not Path(DEFAULT_TOKEN_PATH).exists():
        print(f"No cached Spotify token at {DEFAULT_TOKEN_PATH}; authorise interactively once first.")
        return 2
    client = SpotifyClient(client_id=client_id) if client_id else None

    run = run_dj(
        build_crate(),
        theme=args.theme,
        minutes=args.minutes,
        dry_run=not args.commit,
        familiar_ratio=args.familiar_ratio,
        now=datetime.fromisoformat(args.at).astimezone() if args.at else None,
        client=client,
    )
    print(render(run))
    return 1 if run.errors else 0


if __name__ == "__main__":
    sys.exit(main())
