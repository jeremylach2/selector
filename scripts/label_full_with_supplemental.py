"""Regenerate data/labels.jsonl for the full data-fix scope: the top-3,000
tracks by play count plus the 500-track supplemental tail sample from
Step 4 (250 skipped-once + 250 completed-once, both play_count <= 2).

label.py's CLI only supports a plain --limit N (top-N-by-play-count) query,
not an explicit id list, so this script builds the combined id list
directly and calls enrich_tracks/run_labelling the way label.py's main()
does internally. See docs/DATA_FIX_PLAN.md Step 5.

Checkpointed by run_labelling itself (flushes after every record, skips
track_ids already in the output file) - safe to rerun after an interruption.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from selector.tagger.enrich import enrich_tracks
from selector.tagger.label import run_labelling
from selector.warehouse.build import DEFAULT_DB_PATH

TOP_N = 3000
SUPPLEMENTAL_IDS_PATH = Path("data/supplemental_track_ids.txt")
OUTPUT_PATH = Path("data/labels.jsonl")


def build_combined_ids() -> list[str]:
    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        top_ids = con.execute(
            "SELECT track_id FROM tracks ORDER BY play_count DESC LIMIT ?",
            [TOP_N],
        ).df()["track_id"].tolist()

    supplemental_ids = [
        line.strip() for line in SUPPLEMENTAL_IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    combined = top_ids + supplemental_ids
    print(f"{len(top_ids)} top-N + {len(supplemental_ids)} supplemental = {len(combined)} combined ids")
    return combined


def main() -> None:
    combined_ids = build_combined_ids()
    inputs = enrich_tracks(track_ids=combined_ids)
    print(f"Built {len(inputs)} TeacherInput records; starting labelling -> {OUTPUT_PATH}")
    run_labelling(inputs, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
