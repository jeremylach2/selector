"""Teacher self-consistency on a 200-track gold set.

Samples 200 records from data/labels.jsonl (seeded), relabels each with
label.py's `alt_phrasing` prompt variant, and compares the two passes. The
existing `default`-variant labels are the first pass, so this costs one
teacher call per track. The result is the ceiling row in docs/EVAL.md: a
student can't meaningfully beat its teacher's agreement with itself.

    uv run python scripts/gold_set.py            # label (resumable), then report
    uv run python scripts/gold_set.py --report   # report only
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from selector.tagger.dataset import load_labels
from selector.tagger.label import DEFAULT_MODEL, run_labelling

LABELS_PATH = Path("data/labels.jsonl")
GOLD_PATH = Path("data/labels_gold_200.jsonl")
N_GOLD = 200
SEED = 0


def sample_gold(records):
    return random.Random(SEED).sample(records, N_GOLD)


def report(first_pass, second_pass) -> None:
    by_id = {r.track_id: r for r in first_pass}
    pairs = [(by_id[r.track_id].labels, r.labels) for r in second_pass if r.track_id in by_id]
    n = len(pairs)
    valence_mae = sum(abs(a.valence - b.valence) for a, b in pairs) / n
    intensity_mae = sum(abs(a.intensity - b.intensity) for a, b in pairs) / n
    era = sum(a.era == b.era for a, b in pairs) / n
    mood_exact = sum(set(a.mood_tags) == set(b.mood_tags) for a, b in pairs) / n
    jaccard = sum(len(set(a.mood_tags) & set(b.mood_tags)) / len(set(a.mood_tags) | set(b.mood_tags)) for a, b in pairs) / n
    print(f"Gold set: n={n}")
    print(f"  valence MAE        {valence_mae:.3f}")
    print(f"  intensity MAE      {intensity_mae:.3f}")
    print(f"  era exact match    {era:.1%}")
    print(f"  mood_tags exact    {mood_exact:.1%}")
    print(f"  mood_tags Jaccard  {jaccard:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="skip labelling, just compare")
    args = parser.parse_args()

    gold = sample_gold(load_labels(LABELS_PATH))
    if not args.report:
        run_labelling([r.input for r in gold], output_path=GOLD_PATH, model=DEFAULT_MODEL, variant="alt_phrasing")
    report(gold, load_labels(GOLD_PATH))


if __name__ == "__main__":
    main()
