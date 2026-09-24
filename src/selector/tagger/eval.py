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
import math
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import get_args

from pydantic import ValidationError
from scipy.stats import spearmanr

from selector.tagger.dataset import Arm
from selector.tagger.schema import Era, LabelRecord, MoodTag, PredictedLabels

# From docs/TEACHER.md's 200-track gold set (scripts/gold_set.py; 199
# scored, one relabel failed validation). Hardcoded rather than recomputed
# here because the second pass lives in data/labels_gold_200.jsonl, not in
# data/labels.jsonl.
TEACHER_SELF_CONSISTENCY = {
    "valence_mae": 0.017,
    "intensity_mae": 0.022,
    "era_exact_match": 0.905,
    "mood_tags_exact_match": 0.648,
    "n": 199,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_predictions(true: list[PredictedLabels], pred: list[PredictedLabels]) -> dict[str, float]:
    """Metrics for one arm/model against ground truth: MAE + Spearman for
    the numeric fields, exact-match rate for the categorical ones."""
    valence_errors = [abs(t.valence - p.valence) for t, p in zip(true, pred, strict=True)]
    intensity_errors = [abs(t.intensity - p.intensity) for t, p in zip(true, pred, strict=True)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant predictions -> NaN, reported as n/a
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


def _load_model_and_tokenizer(model_name: str, adapter_path: Path | None):
    # Imported lazily, same rationale as train.py: torch/transformers/peft
    # are heavy and only needed for this model-backed path.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


class _JSONCompleteStoppingCriteria:
    """Halts generation as soon as the tokens produced so far contain one
    complete, balanced top-level JSON object.

    `train.py` never appends an EOS token to a training completion (see
    `_format_example`), so the model has no learned "stop here" signal and
    will otherwise run to `max_new_tokens` on every single example even
    after it has already emitted a fully valid label - on this CPU that's
    the difference between ~15s and ~45s per example, and it compounds
    across ~2,900 generations (486-track test split x 6 rows) into hours.
    Only fires for arms that actually emit `{...}` (i.e. a fine-tuned
    adapter); the untuned base model row gets no benefit from this since it
    doesn't produce JSON at all, and correctly still uses the full budget -
    see the module's `parse_failure_rate` reporting for that finding."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][self.prompt_len :], skip_special_tokens=True)
        start = text.find("{")
        if start == -1:
            return False
        depth = 0
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


def _generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    """Free-generate a completion the same way train.py frames the task: the
    prompt is a raw continuation, not a chat-templated turn, and the model
    is expected to produce the label JSON directly after it - matching how
    `_format_example` concatenates prompt + completion for training.

    `max_new_tokens=80` is a margin over the true distribution of
    tokenized completions across all 3,492 labels (min 48, p99 67, max 70)
    - not a tight cap, but not padded far past it either, because this
    value is also the backstop budget the untuned base model burns on
    *every* example (it never closes a JSON object, so the stopping
    criteria never fires for it - see `_JSONCompleteStoppingCriteria`), and
    that row dominates total eval wall time on this CPU."""
    import torch
    from transformers import StoppingCriteriaList

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([_JSONCompleteStoppingCriteria(tokenizer, inputs["input_ids"].shape[1])])
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
    completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


ZERO_SHOT_SYSTEM_PROMPT = f"""You are labelling music tracks for a recommendation system.
For each track you are given its metadata and, depending on availability, its lyrics and/or
MEASURED audio features. Respond with ONLY a single JSON object with exactly these keys:
- "valence": float from 0.0 (negative) to 1.0 (positive), the song's emotional positivity
- "intensity": float from 0.0 to 1.0, its overall dramatic/emotional intensity
- "era": one of {list(get_args(Era))}
- "mood_tags": list of 1 to 3 values from {list(get_args(MoodTag))}
- "lyrical_theme": a short phrase of at most 60 characters, e.g. "breakup" or "party"
No explanation, no markdown, just the JSON object."""


def _zero_shot_prompt(tokenizer, record: LabelRecord, arm: Arm) -> str:
    """The same per-arm track input the fine-tuned student sees, but framed
    as an instructed chat turn with the output schema spelled out - the fair
    "just prompt the base model" baseline. The raw training-format prompt
    carries no instruction at all, so the untuned model can't be expected to
    produce the label format from it (0% parse rate in practice).

    Qwen3's thinking mode is disabled so the token budget goes to the JSON
    rather than a <think> block."""
    from selector.tagger.dataset import build_prompt

    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(record, arm)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _parse_prediction(text: str) -> PredictedLabels | None:
    """Best-effort extraction of the JSON object from a free generation.

    `train.py`'s completions never have an EOS token appended (see
    `_format_example`), so the model has no learned "stop here" signal and
    routinely keeps generating past a complete, valid JSON object - in
    practice, trailing off into a second lookalike object. Using
    `json.JSONDecoder.raw_decode` instead of `rfind("}")` grabs just the
    first complete JSON value and ignores whatever comes after it, rather
    than spanning both objects into one unparseable blob. Returns None
    rather than raising so a single bad generation doesn't kill the whole
    eval run - the caller is responsible for counting and reporting that as
    a parse failure, not silently dropping it."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    try:
        return PredictedLabels.model_validate(obj)
    except ValidationError:
        return None


def evaluate_model(
    model_name: str,
    test_records: list[LabelRecord],
    arm: Arm,
    fallback: PredictedLabels,
    adapter_path: Path | None = None,
    zero_shot: bool = False,
    predictions_path: Path | None = None,
) -> dict[str, float]:
    """Generate a completion per test-split prompt (optionally with a LoRA
    adapter applied on top of the base model), parse it into
    `PredictedLabels`, and score against ground truth via
    `evaluate_predictions`. A generation that fails to parse is scored using
    `fallback` (the train-mean baseline) instead of being dropped, so a
    single malformed output can't shrink the sample size or vanish quietly
    - `parse_failure_rate` in the returned metrics reports how often that
    happened.

    `zero_shot=True` swaps the raw training-format prompt for the instructed
    chat prompt from `_zero_shot_prompt` (base model only - an adapter was
    trained on the raw format and shouldn't be scored on a different one).
    Its budget is larger than the raw path's because an instructed model
    may pretty-print the JSON; the stopping criteria still ends generation
    as soon as one object is complete.

    `predictions_path`, if given, gets one JSON line per test example (track
    id, ground truth, the scored prediction, whether it parsed, and the raw
    generation) so rows can be compared per example later - e.g. a paired
    bootstrap between arms - without regenerating."""
    from selector.tagger.dataset import build_prompt

    if zero_shot and adapter_path is not None:
        raise ValueError("zero_shot prompting is for the untuned base model only")

    model, tokenizer = _load_model_and_tokenizer(model_name, adapter_path)

    true_labels = [r.labels for r in test_records]
    predictions: list[PredictedLabels] = []
    n_failures = 0
    rows: list[dict] = []
    for record in test_records:
        if zero_shot:
            raw = _generate_completion(
                model, tokenizer, _zero_shot_prompt(tokenizer, record, arm), max_new_tokens=120
            )
        else:
            raw = _generate_completion(model, tokenizer, build_prompt(record, arm) + "\n")
        parsed = _parse_prediction(raw)
        rows.append(
            {
                "track_id": record.track_id,
                "true": record.labels.model_dump(),
                "pred": (parsed or fallback).model_dump(),
                "parsed": parsed is not None,
                "raw": raw,
            }
        )
        if parsed is None:
            n_failures += 1
            parsed = fallback
        predictions.append(parsed)

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    metrics = evaluate_predictions(true_labels, predictions)
    metrics["parse_failure_rate"] = n_failures / len(test_records)
    return metrics


def _fmt_corr(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def _print_row(name: str, metrics: dict[str, float] | None) -> None:
    if metrics is None:
        print(f"{name:32s}  not run this session")
        return
    parse_fail = f"  parse_fail={metrics['parse_failure_rate']:.1%}" if "parse_failure_rate" in metrics else ""
    print(
        f"{name:32s}  valence_mae={metrics['valence_mae']:.3f}  "
        f"intensity_mae={metrics['intensity_mae']:.3f}  "
        f"valence_rho={_fmt_corr(metrics['valence_spearman'])}  "
        f"intensity_rho={_fmt_corr(metrics['intensity_spearman'])}  "
        f"era_match={metrics['era_exact_match']:.1%}  "
        f"mood_match={metrics['mood_tags_exact_match']:.1%}  (n={metrics['n']}){parse_fail}"
    )


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"  # must match train.py's DEFAULT_MODELS["qwen"]


def build_table(
    splits_dir: Path,
    labels_path: Path,
    runs_dir: Path = Path("data/runs"),
    model_name: str = DEFAULT_MODEL_NAME,
    untuned_limit: int | None = 50,
    predictions_dir: Path | None = Path("data/eval_predictions"),
    tuned_only: bool = False,
) -> None:
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
    fallback = baseline_preds[0]

    print(
        f"{'Teacher self-consistency (ceiling)':32s}  "
        f"valence_mae={TEACHER_SELF_CONSISTENCY['valence_mae']:.3f}  "
        f"intensity_mae={TEACHER_SELF_CONSISTENCY['intensity_mae']:.3f}  "
        f"era_match={TEACHER_SELF_CONSISTENCY['era_exact_match']:.1%}  "
        f"mood_match={TEACHER_SELF_CONSISTENCY['mood_tags_exact_match']:.1%}  "
        f"(n={TEACHER_SELF_CONSISTENCY['n']})"
    )

    print(f"{'Measured audio (ground truth, not predicted)':32s}  N/A - see data/audio_features.parquet")

    # The untuned base model gets no instruction or schema in its prompt, so
    # it never emits a parseable label (it continues the lyrics or echoes the
    # prompt) and always burns the full token budget. A seeded subsample
    # shows that just as clearly for a fraction of the run time.
    untuned_test = test
    if untuned_limit is not None and untuned_limit < len(test):
        untuned_test = random.Random(0).sample(test, untuned_limit)

    def pred_path(name: str) -> Path | None:
        return predictions_dir / f"{name}.jsonl" if predictions_dir is not None else None

    for arm in ("A", "B", "C"):
        if not tuned_only:
            print(f"  [arm {arm}] generating untuned base model predictions ({len(untuned_test)} examples)...")
            untuned_metrics = evaluate_model(
                model_name, untuned_test, arm, fallback, predictions_path=pred_path(f"untuned_{arm}")
            )
            _print_row(f"Untuned base model, arm {arm}", untuned_metrics)

        adapter_path = runs_dir / arm / "adapter"
        if not adapter_path.exists():
            _print_row(f"Fine-tuned student, arm {arm}", None)
            continue

        print(f"  [arm {arm}] generating fine-tuned predictions ({len(test)} examples)...")
        tuned_metrics = evaluate_model(
            model_name, test, arm, fallback, adapter_path=adapter_path, predictions_path=pred_path(f"tuned_{arm}")
        )
        _print_row(f"Fine-tuned student, arm {arm}", tuned_metrics)


def build_zero_shot_table(
    labels_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: int | None = None,
    predictions_dir: Path | None = Path("data/eval_predictions"),
) -> None:
    """The instructed zero-shot rows on their own, so they can be run after
    (not alongside) `build_table` without redoing the fine-tuned rows. Uses
    the same artist split and train-mean fallback as `build_table`, so the
    rows line up with its output."""
    from selector.tagger.dataset import load_labels, split_by_artist

    records = load_labels(labels_path)
    train, _val, test = split_by_artist(records)
    if limit is not None and limit < len(test):
        test = random.Random(0).sample(test, limit)

    baseline_preds = train_mean_baseline_predictions(train, len(test))
    print(f"Zero-shot eval (n_test={len(test)}, n_train={len(train)}):\n")
    _print_row("Trivial train-mean baseline", evaluate_predictions([r.labels for r in test], baseline_preds))

    for arm in ("A", "B", "C"):
        print(f"  [arm {arm}] generating instructed zero-shot predictions ({len(test)} examples)...")
        path = predictions_dir / f"zero_shot_{arm}.jsonl" if predictions_dir is not None else None
        metrics = evaluate_model(model_name, test, arm, baseline_preds[0], zero_shot=True, predictions_path=path)
        _print_row(f"Untuned zero-shot, arm {arm}", metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--untuned-limit",
        type=int,
        default=50,
        help="Seeded subsample size for the untuned base model rows (0 = full test split).",
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Run only the instructed zero-shot rows (base model, schema in the prompt).",
    )
    parser.add_argument(
        "--zero-shot-limit",
        type=int,
        default=0,
        help="Seeded subsample size for --zero-shot-only (0 = full test split).",
    )
    parser.add_argument(
        "--tuned-only",
        action="store_true",
        help="Skip the untuned rows, e.g. to regenerate only the fine-tuned predictions files.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/eval_predictions"),
        help="Where each row's per-example predictions are written as <row>.jsonl.",
    )
    args = parser.parse_args(argv)
    if args.zero_shot_only:
        build_zero_shot_table(args.labels, args.model_name, args.zero_shot_limit or None, args.predictions_dir)
        return
    build_table(
        args.splits_dir,
        args.labels,
        args.runs_dir,
        args.model_name,
        args.untuned_limit or None,
        args.predictions_dir,
        args.tuned_only,
    )


if __name__ == "__main__":
    main()
