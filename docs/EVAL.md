# Vibe tagger eval — status

The plan is explicit that the eval table, not the fine-tuned model itself,
is this component's actual deliverable. This doc says plainly which parts
of that table are real numbers from this project and which parts are
pipeline that's built and tested but hasn't been run against a real
fine-tune yet — and why.

## What's built

All four modules from the plan exist and are unit-tested without needing
the ML stack installed for most of the logic:

- `selector/tagger/dataset.py` — splits `data/labels.jsonl` **by artist**
  (never by track — see its docstring for why a track-level split would
  quietly let the model memorise artists instead of reading the input),
  and builds the three arms' prompts (A: lyrics+metadata, B: measured
  features+metadata, C: all three).
- `selector/tagger/train.py` — LoRA fine-tune via `peft`+`transformers`,
  pinned to `device_map="cpu"` (see "Why CPU-only" below), with prompt
  tokens masked out of the loss so the model is trained to produce the
  label JSON, not to reproduce the prompt.
- `selector/tagger/eval.py` — the table itself: per-arm MAE and Spearman
  correlation on the numeric fields, exact-match rate on the categorical
  ones, plus the trivial train-mean baseline and the teacher
  self-consistency ceiling (from `docs/TEACHER.md`).
- `selector/tagger/infer.py` — tags every warehouse track with the winning
  configuration, flagging provenance (`arm`, `label_source`) per row so a
  text-only fallback is never confused with a real audio-informed
  prediction downstream.

## What's actually run

- **The trivial train-mean baseline**, for real, against the pilot's
  2-track test split (30 labelled tracks total, split by artist — see
  below for why the test split is this small).
- **`infer.py --dry-run`**, for real, against the full 19,386-track
  warehouse — tags every track with the baseline, flags 198 of them
  (1.0%) as `arm="C"` (has a matched audio preview) and the rest as
  `arm="A"`, and writes `data/track_features.parquet`. This exists
  specifically to prove the Step 12 handoff (schema, provenance flagging,
  full-warehouse coverage) works end to end before a real fine-tune exists
  to plug into it.
- **Teacher self-consistency**, for real, on a 10-track gold set — see
  `docs/TEACHER.md`.

## What's not run, and why

**No actual LoRA fine-tune has been executed.** Two independent reasons,
not one:

1. **Not enough labelled data yet.** Step 10's pilot produced 30 labelled
   tracks (documented honestly in `docs/TEACHER.md` as interactively
   labelled, not a `label.py` API run). Splitting those by artist for
   train/val/test gives 24/4/2 tracks across 13/3/2 artists. A LoRA
   fine-tune on 24 examples would produce a number, but not a meaningful
   one — it would mostly measure noise in which 2 tracks happened to land
   in the test split. Running `train.py` for real is honestly not worth
   doing until `label.py` has been run at real scale (the plan's target is
   ~3,000 tracks) with a real `ANTHROPIC_API_KEY`.
2. **No usable GPU on this machine.** The dev machine has an AMD Radeon RX
   5600 XT — RDNA1, which ROCm doesn't support on Windows (and support is
   spotty even on Linux). `train.py` is written CPU-only
   (`device_map="cpu"`) rather than pretending otherwise. Estimated
   runtime on this machine's Ryzen 5 3600: roughly 20-45 minutes for a
   small pilot-scale run (150-200 examples, 2-3 epochs), and several hours
   for the full ~2,000+ track training split once it exists. That's a
   real but bounded cost, not a blocker — it just means the run should
   happen once, deliberately, against real data, not as a smoke test
   against 24 examples.

## The honest headline finding, deferred

The plan asks for a plain answer to "does adding measured audio beat
lyrics and metadata alone, and by how much?" That question needs arms A
and C run against the same fine-tuned student on a test set large enough
to move the needle — not answerable from 30 tracks. This doc will be
updated with that table once Step 10 has run at full scale and `train.py`
has executed for real. Until then, this section stays a placeholder
rather than a number dressed up to look like an answer.

## Reference rows, as they stand today

| Row | Status |
|---|---|
| Measured audio (ground truth) | N/A by design — never predicted, read from `data/audio_features.parquet` |
| Teacher self-consistency (ceiling) | Real, n=10 (see `docs/TEACHER.md`) — valence/intensity MAE 0.045, era 100%, mood_tags 30% |
| Trivial train-mean baseline (floor) | Real, n=2 test tracks — too small to be meaningful yet, structurally correct |
| Untuned base model, arms A/B/C | Not run — needs the ML stack installed and a base model download |
| Fine-tuned student, arms A/B/C | Not run — needs real-scale labels (see above) |

Re-run `uv run python -m selector.tagger.eval` after scaling Step 10 and
running `selector.tagger.train` for each arm; `build_table` already knows
how to print all nine rows once model-backed predictions exist to feed it.
