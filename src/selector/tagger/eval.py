"""The vibe tagger's actual deliverable: an eval table comparing what each
input arm contributes, against reference rows that give the numbers a
ceiling and a floor.

Three arms (input ablation, same student, same test set):
  A: metadata + lyrics only
  B: metadata + measured audio features only
  C: all three

Four reference rows:
  - teacher self-consistency: the ceiling for the subjective fields, from
    docs/TEACHER.md's gold-set numbers (mood_tags in particular does not
    start near 100% even at the ceiling - see that doc before reading any
    arm's mood_tags number as "bad")
  - untuned base model: same prompts, no fine-tune
  - trivial train-mean baseline: predict the train split's mean/mode for
    every test example, regardless of input - the floor anything trained
    should clear
  - measured audio as ground truth: not a prediction row at all, listed to
    make explicit that tempo/energy/danceability/acousticness/
    instrumentalness are read from data/audio_features.parquet, never
    predicted, anywhere in this component

This module computes what doesn't need the fine-tuned model (the train-mean
baseline, and formats the teacher-ceiling reference) directly. The
model-dependent rows (untuned base, and each fine-tuned arm) need a trained
adapter from `selector.tagger.train` and are computed via `evaluate_model`
when one is available - see docs/EVAL.md for which rows were actually run
in this project vs. left as a documented gap.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr

from selector.tagger.schema import LabelRecord, PredictedLabels

# From docs/TEACHER.md's 10-track gold set. Hardcoded rather than
# recomputed here because the gold set's two passes are pilot data, not a
# thing this module re-derives from data/labels.jsonl alone.
TEACHER_SELF_CONSISTENCY = {
    "valence_mae": 0.045,
    "intensity_mae": 0.045,
    "era_exact_match": 1.0,
    "mood_tags_exact_match": 0.30,
    "n": 10,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_predictions(true: list[PredictedLabels], pred: list[PredictedLabels]) -> dict[str, float]:
    """Metrics for one arm/model against ground truth: MAE + Spearman for
    the numeric fields, exact-match rate for the categorical ones."""
    valence_errors = [abs(t.valence - p.valence) for t, p in zip(true, pred, strict=True)]
    intensity_errors = [abs(t.intensity - p.intensity) for t, p in zip(true, pred, strict=True)]

    valence_corr = spearmanr([t.valence for t in true], [p.valence for p in pred]).statistic
    intensity_corr = spearmanr([t.intensity for t in true], [p.intensity for p in pred]).statistic

    era_matches = sum(t.era == p.era for t, p in zip(true, pred, strict=True))
    mood_matches = sum(set(t.mood_tags) == set(p.mood_tags) for t, p in zip(true, pred, strict=True))

    n = len(true)
    return {
        "valence_mae": sum(valence_errors) / n,
        "intensity_mae": sum(intensity_errors) / n,
        "valence_spearman": valence_corr,
        "intensity_spearman": intensity_corr,
        "era_exact_match": era_matches / n,
        "mood_tags_exact_match": mood_matches / n,
        "n": n,
    }


def train_mean_baseline_predictions(train: list[LabelRecord], n_test: int) -> list[PredictedLabels]:
    """The floor: predict the train split's mean/mode for every test
    example, ignoring its actual input entirely."""
    mean_valence = sum(r.labels.valence for r in train) / len(train)
    mean_intensity = sum(r.labels.intensity for r in train) / len(train)
    mode_era = Counter(r.labels.era for r in train).most_common(1)[0][0]
    mode_mood_tags = list(Counter(tuple(sorted(r.labels.mood_tags)) for r in train).most_common(1)[0][0])
    mode_theme = Counter(r.labels.lyrical_theme for r in train).most_common(1)[0][0]

    baseline = PredictedLabels(
        valence=mean_valence,
        intensity=mean_intensity,
        era=mode_era,
        mood_tags=mode_mood_tags,
        lyrical_theme=mode_theme,
    )
    return [baseline] * n_test


def _print_row(name: str, metrics: dict[str, float] | None) -> None:
    if metrics is None:
        print(f"{name:32s}  not run this session")
        return
    print(
        f"{name:32s}  valence_mae={metrics['valence_mae']:.3f}  "
        f"intensity_mae={metrics['intensity_mae']:.3f}  "
        f"era_match={metrics['era_exact_match']:.1%}  "
        f"mood_match={metrics['mood_tags_exact_match']:.1%}  (n={metrics['n']})"
    )


def build_table(splits_dir: Path, labels_path: Path) -> None:
    from selector.tagger.dataset import load_labels, split_by_artist

    records = load_labels(labels_path)
    train, _val, test = split_by_artist(records)

    if not test:
        print("Test split is empty at this label-set size - table below uses train-mean only as a sanity check.")
        test = train[-1:]  # fall back so the table has at least one row to print

    true_labels = [r.labels for r in test]

    print(f"Eval table (n_test={len(test)}, n_train={len(train)}):\n")

    baseline_preds = train_mean_baseline_predictions(train, len(test))
    _print_row("Trivial train-mean baseline", evaluate_predictions(true_labels, baseline_preds))

    print(
        f"{'Teacher self-consistency (ceiling)':32s}  "
        f"valence_mae={TEACHER_SELF_CONSISTENCY['valence_mae']:.3f}  "
        f"intensity_mae={TEACHER_SELF_CONSISTENCY['intensity_mae']:.3f}  "
        f"era_match={TEACHER_SELF_CONSISTENCY['era_exact_match']:.1%}  "
        f"mood_match={TEACHER_SELF_CONSISTENCY['mood_tags_exact_match']:.1%}  "
        f"(n={TEACHER_SELF_CONSISTENCY['n']})"
    )

    print(f"{'Measured audio (ground truth, not predicted)':32s}  N/A - see data/audio_features.parquet")

    for arm in ("A", "B", "C"):
        _print_row(f"Untuned base model, arm {arm}", None)
        _print_row(f"Fine-tuned student, arm {arm}", None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    args = parser.parse_args(argv)
    build_table(args.splits_dir, args.labels)


if __name__ == "__main__":
    main()
