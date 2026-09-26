"""Tag every track in the warehouse with the winning fine-tuned
configuration from Step 11's eval table, producing
`data/track_features.parquet` — the file Step 12 consumes.

Arm C (lyrics + measured audio + metadata) for the ~3,198 tracks with a
matched audio preview, arm A (lyrics + metadata) for the rest, since arm C
needs a measured feature vector arm A doesn't. Every row carries a
`label_source` column ("finetuned_gpu" or "parse_fallback") and an `arm`
column, since a fine-tuned student's prediction and a train-mean fallback
used when a generation didn't parse are not the same claim about a track.

Generation runs through llama.cpp's Vulkan `llama-server` (see
docs/GPU_INFERENCE.md) rather than the CPU eval path used for the 487-track
test split in `selector.tagger.eval` — at 19,386 tracks, CPU generation
would take on the order of a day; the GPU path was verified to match the
CPU path's metrics on the same held-out test split before being trusted
here (`scripts/gpu_parity_check.py`, results in docs/GPU_INFERENCE.md).
Requires both arm servers running (ports 8711/A, 8712/C — see
docs/GPU_INFERENCE.md for the exact launch commands) and checkpoints to
`data/track_features_checkpoint.jsonl` after every track, so a kill resumes
instead of restarting.

Every row also carries `lyrics_status` ("lyrics", "instrumental" or
"unknown", from `selector.tagger.enrich.lyrics_status`). The model sees
"Lyrics: not available" for both of the last two, so it tends to describe
every lyric-less track as an instrumental. The column says which ones
lrclib actually confirmed.

`--retag-ids FILE` drops those tracks from the checkpoint and re-tags just
them, e.g. after `scripts/fetch_all_lyrics.py --recheck-empty` recovers
lyrics. `--status-only` rewrites the parquet with a fresh `lyrics_status`
column and no generation.

`--dry-run` predicts using the trivial train-mean baseline instead of
calling a model, so the output schema and the Step 12 handoff can be
exercised end to end without a GPU server running — every row in that mode
is flagged `label_source="dry_run_baseline"`, never something a downstream
consumer could mistake for a real prediction.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from selector.tagger.dataset import build_prompt_from_input, load_labels, split_by_artist
from selector.tagger.enrich import LYRICS_DIR, _measured_features_by_track, lyrics_status
from selector.tagger.eval import _parse_prediction, train_mean_baseline_predictions
from selector.tagger.gpu_infer import SERVER_URLS, generate_completion_gpu, server_healthy
from selector.tagger.schema import TeacherInput
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

DEFAULT_OUTPUT_PATH = Path("data/track_features.parquet")
CHECKPOINT_PATH = Path("data/track_features_checkpoint.jsonl")
MEASURED_COLUMNS = ["tempo_scaled", "rms_mean_scaled", "danceability", "harmonic_percussive_ratio_scaled"]
N_WORKERS = 8  # 4 parallel slots per arm's llama-server (arm A + arm C)


def _all_tracks(db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as con:
        return con.execute("SELECT track_id, name, artist, album FROM tracks").df()


def _has_audio_match(audio_features_path: Path) -> set[str]:
    if not audio_features_path.exists():
        return set()
    return set(pd.read_parquet(audio_features_path)["track_id"])


def _load_lyrics(track_id: str) -> str | None:
    """Reads the cache `selector.tagger.enrich.fetch_lyrics` already wrote
    (see `scripts/fetch_all_lyrics.py`) - no network call here, since
    inference should not be re-fetching lyrics one track at a time."""
    path = LYRICS_DIR / f"{track_id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text else None


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cached: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["track_id"]] = row
    return cached


def _with_lyrics_status(result: pd.DataFrame) -> pd.DataFrame:
    return result.assign(lyrics_status=[lyrics_status(t) or "unknown" for t in result["track_id"]])


def drop_from_checkpoint(path: Path, track_ids: set[str]) -> int:
    """Remove `track_ids` from the checkpoint so the next run re-tags them."""
    rows = _load_checkpoint(path)
    kept = [row for tid, row in rows.items() if tid not in track_ids]
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    return len(rows) - len(kept)


def write_output(checkpoint_path: Path, output_path: Path) -> pd.DataFrame:
    result = _with_lyrics_status(pd.DataFrame(list(_load_checkpoint(checkpoint_path).values())))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


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


def _infer_one(track_id: str, arm: str, item: TeacherInput, fallback, client: httpx.Client) -> dict:
    prompt = build_prompt_from_input(item, arm) + "\n"
    try:
        raw = generate_completion_gpu(prompt, client, server_url=SERVER_URLS[arm])
        parsed = _parse_prediction(raw)
    except httpx.HTTPError:
        parsed = None
    prediction = parsed or fallback
    return {
        "track_id": track_id,
        **prediction.model_dump(),
        "arm": arm,
        "label_source": "finetuned_gpu" if parsed is not None else "parse_fallback",
    }


def infer_all(
    db_path: Path = DEFAULT_DB_PATH,
    labels_path: Path = Path("data/labels.jsonl"),
    audio_features_path: Path = Path("data/audio_features.parquet"),
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    n_workers: int = N_WORKERS,
) -> pd.DataFrame:
    """Tag every warehouse track through the real fine-tuned models, running
    generation on both arm servers concurrently. See the module docstring
    for the arm-selection rule, checkpointing, and why this goes through
    llama-server rather than the CPU eval path."""
    tracks = _all_tracks(db_path)
    matched_ids = _has_audio_match(audio_features_path)
    measured_by_track = _measured_features_by_track(audio_features_path, MEASURED_COLUMNS)

    train_records = load_labels(labels_path)
    train, _val, _test = split_by_artist(train_records)
    fallback = train_mean_baseline_predictions(train, 1)[0]

    cached = _load_checkpoint(checkpoint_path)
    todo = [row for _, row in tracks.iterrows() if row["track_id"] not in cached]
    print(f"{len(cached):,} tracks already checkpointed, {len(todo):,} remaining")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=60.0, limits=httpx.Limits(max_connections=n_workers * 2))

    def submit(row) -> dict:
        track_id = row["track_id"]
        arm = "C" if track_id in matched_ids else "A"
        item = TeacherInput(
            track_id=track_id,
            track_name=row["name"],
            artist_name=row["artist"],
            album_name=row["album"] if pd.notna(row["album"]) else None,
            measured=measured_by_track.get(track_id, {}) if arm == "C" else {},
            lyrics=_load_lyrics(track_id),
        )
        return _infer_one(track_id, arm, item, fallback, client)

    start = time.monotonic()
    done = 0
    with checkpoint_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(submit, todo):
            f.write(json.dumps(result) + "\n")
            f.flush()
            done += 1
            if done % 200 == 0 or done == len(todo):
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed else 0.0
                remaining_min = (len(todo) - done) / rate / 60 if rate else float("inf")
                print(f"  {done:,}/{len(todo):,} done, {rate:.2f}/s, ~{remaining_min:.0f} min remaining")

    client.close()
    return write_output(checkpoint_path, output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--audio-features", type=Path, default=Path("data/audio_features.parquet"))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tag with the trivial baseline instead of calling llama-server",
    )
    parser.add_argument("--retag-ids", type=Path, help="file of track ids to drop from the checkpoint and re-tag")
    parser.add_argument(
        "--status-only", action="store_true", help="refresh the lyrics_status column without generating"
    )
    args = parser.parse_args(argv)

    if args.status_only:
        result = write_output(args.checkpoint, args.output)
        print(f"Wrote {len(result):,} rows to {args.output}: {result['lyrics_status'].value_counts().to_dict()}")
        return

    if args.dry_run:
        result = infer_dry_run(args.db_path, args.labels, args.audio_features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        print(f"Wrote {len(result):,} rows to {args.output} ({(result['arm'] == 'C').mean():.1%} arm C)")
        return

    for arm, url in SERVER_URLS.items():
        if not server_healthy(url):
            raise SystemExit(
                f"llama-server for arm {arm} is not reachable at {url}. "
                "Start it first - see docs/GPU_INFERENCE.md for the exact commands."
            )

    if args.retag_ids:
        ids = set(args.retag_ids.read_text(encoding="utf-8").split())
        print(f"Dropped {drop_from_checkpoint(args.checkpoint, ids):,} tracks from the checkpoint to re-tag")

    result = infer_all(args.db_path, args.labels, args.audio_features, args.checkpoint, args.output, args.workers)
    fallback_rate = (result["label_source"] == "parse_fallback").mean()
    print(
        f"Wrote {len(result):,} rows to {args.output} "
        f"({(result['arm'] == 'C').mean():.1%} arm C, {fallback_rate:.1%} parse_fallback)"
    )


if __name__ == "__main__":
    main()
