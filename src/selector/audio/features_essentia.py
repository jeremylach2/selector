"""Extract audio features from preview clips using Essentia's pretrained
TensorFlow classifiers, inside WSL2.

This half of Step 9 maps almost one-to-one onto Spotify's removed
audio-features endpoint, which is the whole point: danceability and the five
mood dimensions below are the same kind of "how does this sound" summary
Spotify used to compute server-side.

Must run inside WSL2 (or Linux/macOS) — ``essentia-tensorflow`` ships no
Windows wheels. See ``docs/WSL_SETUP.md`` for the exact setup, and
``scripts/run_essentia.sh`` to run this from Windows in one command. Reads
``data/audio/`` and the models in ``data/essentia_models/`` through
``/mnt/d/...``; both paths work unmodified because the Windows drive is
already mounted there.

Honesty constraint: every prediction here is made on a 30-second excerpt,
not the full track — see the module docstring in ``features_librosa`` for
the full statement of that limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_MATCHES_PATH = Path("data/audio_matches.parquet")
DEFAULT_MODEL_DIR = Path("data/essentia_models")
DEFAULT_OUTPUT_PATH = Path("data/features_essentia.parquet")

SAMPLE_RATE = 16000  # required by the musicnn-msd model family, per each model's .json

# Each model's positive-valence class label. Class *order* in the .pb's
# softmax output differs per model (see the downloaded .json metadata) so
# the label, not a hardcoded index, is what's looked up at runtime.
MODEL_POSITIVE_CLASS = {
    "danceability": "danceable",
    "mood_happy": "happy",
    "mood_sad": "sad",
    "mood_aggressive": "aggressive",
    "mood_party": "party",
    "mood_relaxed": "relaxed",
}


def _load_model_metadata(model_dir: Path) -> dict[str, dict]:
    """Load each model's .pb path and positive-class index from its .json."""
    metadata: dict[str, dict] = {}
    for name, positive_label in MODEL_POSITIVE_CLASS.items():
        pb_path = model_dir / f"{name}-musicnn-msd-2.pb"
        json_path = model_dir / f"{name}-musicnn-msd-2.json"
        if not pb_path.exists() or not json_path.exists():
            continue
        classes = json.loads(json_path.read_text())["classes"]
        metadata[name] = {
            "pb_path": pb_path,
            "positive_index": classes.index(positive_label),
        }
    return metadata


def _build_graphs(models: dict[str, dict]) -> dict[str, tuple[object, int]]:
    """Instantiate each TensorFlow graph once. Loading a graph parses a
    multi-MB protobuf and initialises TF session state, so doing this per
    clip rather than once up front turns a few-minute job into a very slow
    one at ~3,000-track scale."""
    import essentia.standard as es

    graphs = {}
    for name, meta in models.items():
        graphs[name] = (
            es.TensorflowPredictMusiCNN(graphFilename=str(meta["pb_path"])),
            meta["positive_index"],
        )
    return graphs


def _predict_all(audio, graphs: dict[str, tuple[object, int]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, (graph, positive_index) in graphs.items():
        preds = graph(audio)  # (n_frames, n_classes)
        scores[name] = float(preds[:, positive_index].mean())
    return scores


def extract_features(path: str, graphs: dict[str, tuple[object, int]]) -> dict[str, float] | None:
    """Compute one row of Essentia model predictions for a single clip.

    Returns None on any decode or inference failure so one bad clip doesn't
    kill the whole batch — this run has no checkpointing, unlike
    ``resolve.py``, so an unhandled exception here loses everything, not
    just the one clip (see ``_predict_all``, which can raise on a
    degenerate/very-short clip where a model returns a plain list instead
    of the expected ndarray).
    """
    import essentia.standard as es

    try:
        loader = es.MonoLoader(filename=path, sampleRate=SAMPLE_RATE)
        audio = loader()
    except RuntimeError:
        return None

    if audio.size == 0:
        return None

    try:
        return _predict_all(audio, graphs)
    except Exception:  # noqa: BLE001 - any inference failure means "skip this clip", not "crash the run"
        return None


def extract_all(matches: pd.DataFrame, model_dir: Path = DEFAULT_MODEL_DIR) -> pd.DataFrame:
    models = _load_model_metadata(model_dir)
    if not models:
        print(f"No Essentia model files found under {model_dir} - nothing to extract.")
        return pd.DataFrame(columns=["track_id"])

    print(f"Loaded {len(models)}/{len(MODEL_POSITIVE_CLASS)} model(s): {sorted(models)}")
    graphs = _build_graphs(models)

    matched = matches[matches["local_path"].notna()]
    rows: list[dict] = []
    failures: list[str] = []

    for i, (_, r) in enumerate(matched.iterrows()):
        # local_path is written by fetch.py on Windows and comes out with
        # backslashes; Linux (WSL) treats those as literal filename
        # characters, not separators, so normalise before opening.
        path = r["local_path"].replace("\\", "/")
        feats = extract_features(path, graphs)
        if feats is None:
            failures.append(r["track_id"])
            continue
        feats["track_id"] = r["track_id"]
        rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(matched)} clips processed")

    if failures:
        print(f"{len(failures)} clips failed to decode: {failures[:10]}{'...' if len(failures) > 10 else ''}")

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    matches = pd.read_parquet(args.matches)
    features = extract_all(matches, model_dir=args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features):,} rows of Essentia features to {args.output}")


if __name__ == "__main__":
    main()
