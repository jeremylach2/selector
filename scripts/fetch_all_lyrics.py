"""Fetch lyrics for every track in the warehouse (not just the ~3,500
labelled tracks), so infer.py's full run (Step 11 item 6) has lyrics
available for arm A/C predictions on the long tail.

`enrich.fetch_lyrics` already caches to `data/lyrics/{track_id}.txt` and
skips tracks with a cache hit without a network call, so this is safe to
kill and rerun - it picks up wherever it left off. Long-running (~3.5h for
~15,900 uncached tracks at the 0.3s per-request rate limit); run it as an
unattended background job.
"""

from __future__ import annotations

import time

import httpx

from selector.tagger.enrich import fetch_lyrics
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect


def main() -> None:
    with _connect(DEFAULT_DB_PATH) as con:
        tracks = con.execute("SELECT track_id, name, artist FROM tracks").df()

    total = len(tracks)
    hits = 0
    start = time.monotonic()
    with httpx.Client(timeout=10.0) as client:
        for i, row in enumerate(tracks.itertuples(index=False), start=1):
            lyrics = fetch_lyrics(client, row.track_id, row.name, row.artist)
            if lyrics:
                hits += 1
            if i % 200 == 0 or i == total:
                elapsed = time.monotonic() - start
                print(f"{i:,}/{total:,} processed, {hits:,} with lyrics ({hits / i:.1%}), {elapsed / 60:.1f} min elapsed", flush=True)

    print(f"Done. {hits:,}/{total:,} tracks have lyrics ({hits / total:.1%}).")


if __name__ == "__main__":
    main()
