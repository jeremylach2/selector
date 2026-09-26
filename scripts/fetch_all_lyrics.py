"""Fetch lyrics for every track in the warehouse (not just the ~3,500
labelled tracks), so infer.py's full run (Step 11 item 6) has lyrics
available for arm A/C predictions on the long tail.

`enrich.fetch_lyrics` already caches to `data/lyrics/{track_id}.txt` and
skips tracks with a cache hit without a network call, so this is safe to
kill and rerun - it picks up wherever it left off. Long-running (~3.5h for
~15,900 uncached tracks at the 0.3s per-request rate limit); run it as an
unattended background job.

`--recheck-empty` re-queries only the tracks cached with no lyrics. The
first full run cached failed requests as "no lyrics"; this recovers them
and records lrclib's instrumental flag. It writes the track ids whose
lyrics were recovered to `data/lyrics_recovered.txt`, the input to
`infer.py --retag-ids`.

Requests that still fail after retries are left uncached and counted, so a
rerun picks them up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from selector.tagger.enrich import LyricsFetchError, fetch_lyrics, lyrics_status
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

RECOVERED_PATH = Path("data/lyrics_recovered.txt")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recheck-empty", action="store_true", help="re-query tracks cached with no lyrics")
    args = parser.parse_args(argv)

    with _connect(DEFAULT_DB_PATH) as con:
        tracks = con.execute("SELECT track_id, name, artist FROM tracks").df()
    if args.recheck_empty:
        tracks = tracks[[lyrics_status(t) == "unknown" for t in tracks.track_id]]

    total = len(tracks)
    counts = {"lyrics": 0, "instrumental": 0, "unknown": 0, "failed": 0}
    recovered: list[str] = []
    start = time.monotonic()
    with httpx.Client(timeout=10.0) as client:
        for i, row in enumerate(tracks.itertuples(index=False), start=1):
            try:
                lyrics = fetch_lyrics(client, row.track_id, row.name, row.artist, refresh=args.recheck_empty)
            except LyricsFetchError:
                counts["failed"] += 1
            else:
                counts[lyrics_status(row.track_id)] += 1
                if lyrics and args.recheck_empty:
                    recovered.append(row.track_id)
            if i % 200 == 0 or i == total:
                elapsed = time.monotonic() - start
                summary = ", ".join(f"{k} {v:,}" for k, v in counts.items())
                print(f"{i:,}/{total:,} processed ({summary}), {elapsed / 60:.1f} min elapsed", flush=True)

    if args.recheck_empty:
        RECOVERED_PATH.write_text("".join(f"{t}\n" for t in recovered), encoding="utf-8")
        print(f"Recovered lyrics for {len(recovered):,} tracks, ids in {RECOVERED_PATH}.")
    print(f"Done. {counts['failed']:,} requests failed and were left uncached; rerun to retry them.")


if __name__ == "__main__":
    main()
