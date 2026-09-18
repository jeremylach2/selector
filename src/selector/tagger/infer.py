"""Tag every track in the warehouse with the winning fine-tuned
configuration from Step 11's eval table, producing
`data/track_features.parquet` — the file Step 12 consumes.

Falls back to the text-only arm (A: metadata + lyrics) for tracks with no
matched audio preview, and flags which arm produced each row via a
`label_source` column, since a fine-tuned student and a text-only fallback
are not the same claim about a track.

Requires a trained adapter (see `selector.tagger.train`) for the arm(s) it's
pointed at. `--dry-run` predicts using the trivial train-mean baseline
instead of a real model, so the output schema and the Step 12 handoff can
be exercised end to end before a real fine-tune exists — every row in that
mode is flagged `label_source="dry_run_baseline"`, never something a
downstream consumer could mistake for a real prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selector.tagger.dataset import load_labels, split_by_artist
from selector.tagger.eval import train_mean_baseline_predictions
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

DEFAULT_OUTPUT_PATH = Path("data/track_features.parquet")


def _all_tracks(db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as con:
        return con.execute("SELECT track_id, name, artist, album FROM tracks").df()


def _has_audio_match(audio_features_path: Path) -> set[str]:
    if not audio_features_path.exists():
        return set()
    return set(pd.read_parquet(audio_features_path)["track_id"])


def infer_dry_run(
    db_path: Path = DEFAULT_DB_PATH,
    labels_path: Path = Path("data/labels.jsonl"),
    audio_features_path: Path = Path("data/audio_features.parquet"),
) -> pd.DataFrame:
    """Tag every warehouse track with the trivial baseline, so the output
    schema and provenance flagging can be exercised without a real
    fine-tuned model. Never mistakeable for real predictions - see the
    module docstring."""
    tracks = _all_tracks(db_path)
    matched_ids = _has_audio_match(audio_features_path)

    train_records = load_labels(labels_path) if labels_path.exists() else []
    if not train_records:
        raise RuntimeError(f"No labels found at {labels_path} - run selector.tagger.label first.")
    train, _val, _test = split_by_artist(train_records)
    baseline = train_mean_baseline_predictions(train, n_test=1)[0]

    rows = []
    for _, row in tracks.iterrows():
        arm = "C" if row["track_id"] in matched_ids else "A"
        rows.append({**baseline.model_dump(), "track_id": row["track_id"], "label_source": "dry_run_baseline", "arm": arm})

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--audio-features", type=Path, default=Path("data/audio_features.parquet"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tag with the trivial baseline instead of a trained adapter (no adapter loading implemented yet)",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        raise SystemExit(
            "Real-model inference needs a trained adapter from selector.tagger.train, "
            "which this project hasn't run yet (see docs/EVAL.md). Pass --dry-run to "
            "exercise the output schema with the trivial baseline instead."
        )

    result = infer_dry_run(args.db_path, args.labels, args.audio_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result):,} rows to {args.output} ({(result['arm'] == 'C').mean():.1%} arm C)")


if __name__ == "__main__":
    main()
