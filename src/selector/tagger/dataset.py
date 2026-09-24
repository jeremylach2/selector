"""Build train/val/test splits and prompt/completion pairs from
`data/labels.jsonl` for Step 11's LoRA fine-tune.

Splits **by artist, not by track**: every track by a given artist lands in
exactly one split. Splitting by track would let the model see, say, three
Pink Floyd tracks in training and a fourth in test — it could then get the
fourth "right" by memorising Pink Floyd's general vibe rather than by
reading that track's actual lyrics and features, which would make the eval
number a measure of artist memorisation, not of the task this component
claims to do.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from selector.tagger.schema import LabelRecord, TeacherInput

Arm = Literal["A", "B", "C"]

# A: metadata + lyrics only. B: metadata + measured audio features only.
# C: all three. Step 11's eval table runs the same student against all
# three to show what each input actually contributes.
ARMS: tuple[Arm, ...] = ("A", "B", "C")


def load_labels(path: Path) -> list[LabelRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(LabelRecord.model_validate_json(line))
    return records


def split_by_artist(
    records: list[LabelRecord],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[LabelRecord], list[LabelRecord], list[LabelRecord]]:
    """Partition records into train/val/test with no artist crossing a
    split boundary. Fractions are applied to the artist list, not the
    track list, so split sizes in tracks will vary with how many tracks
    each artist happens to contribute."""
    artists = sorted({r.input.artist_name for r in records})
    rng = random.Random(seed)
    rng.shuffle(artists)

    n = len(artists)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac)) if n - n_train > 1 else 0

    train_artists = set(artists[:n_train])
    val_artists = set(artists[n_train : n_train + n_val])
    test_artists = set(artists[n_train + n_val :])

    train = [r for r in records if r.input.artist_name in train_artists]
    val = [r for r in records if r.input.artist_name in val_artists]
    test = [r for r in records if r.input.artist_name in test_artists]
    return train, val, test


def build_prompt_from_input(item: TeacherInput, arm: Arm) -> str:
    """The student's input text for one track under one arm. Takes a bare
    `TeacherInput` rather than a `LabelRecord` so callers with no ground
    truth - `infer.py` tagging the full warehouse, which has no label for
    most tracks - can build the same prompt the fine-tuned adapters were
    trained on without fabricating a `LabelRecord`."""
    lines = [f"Track: {item.track_name!r} by {item.artist_name!r}"]
    if item.album_name:
        lines.append(f"Album: {item.album_name!r}")

    if arm in ("B", "C") and item.measured:
        lines.append(f"Measured audio features: {item.measured}")
    elif arm == "B":
        lines.append("Measured audio features: none available")

    if arm in ("A", "C"):
        lines.append(f"Lyrics: {item.lyrics}" if item.lyrics else "Lyrics: not available")

    return "\n".join(lines)


def build_prompt(record: LabelRecord, arm: Arm) -> str:
    """The student's input text for one track under one arm."""
    return build_prompt_from_input(record.input, arm)


def build_completion(record: LabelRecord) -> str:
    return record.labels.model_dump_json()


def to_examples(records: list[LabelRecord], arm: Arm) -> list[dict[str, str]]:
    return [{"prompt": build_prompt(r, arm), "completion": build_completion(r)} for r in records]


def write_split_files(records: list[LabelRecord], arm: Arm, out_dir: Path) -> None:
    train, val, test = split_by_artist(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        examples = to_examples(split, arm)
        path = out_dir / f"{name}_{arm}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(examples)} examples ({len({r.input.artist_name for r in split})} artists) -> {path}")
