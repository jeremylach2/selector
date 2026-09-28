# Vibe tagger eval

The plan is explicit that the eval table, not the fine-tuned model itself,
is this component's actual deliverable. This doc reports that table, says
how each row was produced, and is plain about what it does and doesn't
show.

## Headline

**Does adding measured audio beat lyrics and metadata alone?** Yes, modestly,
and mostly on intensity. On the 487-track held-out test set, the arm that
sees lyrics + audio + metadata (C) beats lyrics + metadata (A) by 22% on
intensity MAE (0.094 → 0.073) and 9% on valence MAE (0.106 → 0.096). Era
and mood-tag accuracy are effectively tied (within 0.4 points). Audio on its
own (arm B) helps intensity but barely moves valence off the trivial
baseline, which fits what the features measure: tempo, loudness and
danceability describe energy, not emotional positivity.

**Is fine-tuning necessary, or would prompting the base model do?** It's
necessary. Given the same inputs plus explicit instructions and the full
output schema, the untuned base model scores worse than the trivial
train-mean baseline on every metric in every arm. It returns valid JSON
almost every time, but the content is close to a fixed guess: "euphoric",
intensity near 0.9, and an era from the 2000s or 2020s regardless of the
track. Fine-tuning is what turns this 0.6B model into a usable tagger.

These are point estimates from one test split with no significance test yet
(see "Still open"). The intensity gain from audio is large enough to be
believable. The valence gain should be read as suggestive until a paired
bootstrap confirms it. The gap between fine-tuned and zero-shot is large
enough that no test is needed to believe it.

## The table

Test split: 487 tracks from 170 artists that never appear in train (split
by artist, see `dataset.py`). Train split: 2,359 tracks.

| Row | n | valence MAE ↓ | intensity MAE ↓ | era match ↑ | mood_tags exact ↑ | parse fail |
|---|---|---|---|---|---|---|
| Teacher self-consistency (ceiling) | 199 | 0.017 | 0.022 | 90.5% | 64.8% | n/a |
| Trivial train-mean baseline (floor) | 487 | 0.163 | 0.120 | 42.3% | 6.4% | n/a |
| Untuned base model, raw prompt, arms A/B/C | 50 | 0.153 | 0.127 | 42.0% | 6.0% | 100% |
| Untuned base model, instructed zero-shot, arm A | 487 | 0.277 | 0.408 | 17.2% | 2.3% | 3.9% |
| Untuned base model, instructed zero-shot, arm B | 487 | 0.246 | 0.277 | 23.6% | 0.4% | 0.0% |
| Untuned base model, instructed zero-shot, arm C | 487 | 0.234 | 0.290 | 22.4% | 1.2% | 7.0% |
| Fine-tuned, arm A (lyrics + metadata) | 487 | 0.106 | 0.094 | 61.4% | 18.5% | 0.6% |
| Fine-tuned, arm B (audio + metadata) | 487 | 0.157 | 0.104 | 61.2% | 9.2% | 0.8% |
| **Fine-tuned, arm C (lyrics + audio + metadata)** | 487 | **0.096** | **0.073** | **61.8%** | **18.9%** | 0.8% |
| Measured audio (tempo, energy, ...) | n/a | N/A by design: read from `data/audio_features.parquet`, never predicted | | | | |

**Spearman correlation** (rank agreement with the teacher, higher is
better) was only printed for the zero-shot run:

| Row | valence ρ | intensity ρ |
|---|---|---|
| Zero-shot, arm A | 0.163 | 0.117 |
| Zero-shot, arm B | 0.004 | 0.053 |
| Zero-shot, arm C | −0.037 | 0.240 |

Even ranking tracks relative to each other, which ignores the model's
upward bias, is close to chance. The fine-tuned rows' Spearman values
weren't printed in the main run. Rerunning them with `--tuned-only` would
fill that in (see "Still open"). The baseline's Spearman is undefined,
because it predicts the same value for every track.

**Share of the floor-to-ceiling gap closed** (MAE, lower is better):

| Arm | valence | intensity | era | mood_tags exact |
|---|---|---|---|---|
| A | 39% | 27% | 40% | 21% |
| B | 4% | 16% | 39% | 5% |
| C | **46%** | **48%** | **40%** | **21%** |

Computed as (floor − arm) / (floor − ceiling) for MAE, and
(arm − floor) / (ceiling − floor) for the match rates. Even the best arm
covers less than half the distance to the teacher on every field, so
there's real headroom left for a bigger student, more data, or better
inputs.

## How to read the rows

- **Teacher self-consistency (ceiling).** The teacher (`claude-sonnet-5`)
  relabelled a seeded random 200 tracks with a differently worded prompt
  (`alt_phrasing`), compared against its original labels
  (`scripts/gold_set.py`, 199 scored, `docs/TEACHER.md`). A student can't
  meaningfully beat its teacher's agreement with itself. The teacher is
  very stable on valence and intensity (MAE ≈ 0.02) and agrees on the exact
  mood-tag set 64.8% of the time (mean Jaccard 0.84), so mood_tags is
  subjective but far from arbitrary. An earlier 10-track estimate put the
  mood ceiling at 30%. The 200-track number replaces it.
- **Trivial train-mean baseline (floor).** Predicts the train split's mean
  valence/intensity and modal era/mood_tags/theme for every test track,
  ignoring the input. Anything trained should clear it.
- **Untuned base model, raw prompt.** Base `Qwen/Qwen3-0.6B` given exactly
  the prompt format the fine-tuned arms were trained on. That prompt is just
  `Track: ... / Album: ... / Lyrics: ...` (or measured features) with no
  instruction and no schema, so the untuned model has no way to know a JSON
  label is expected. A 5-example probe showed it continuing the lyrics,
  echoing the prompt back, or looping ("Key signature: C" repeated), always
  to the full token budget. It parsed 0% of the time, so every prediction
  fell back to the train-mean baseline. That's why this row's numbers equal
  the baseline scored on its 50-track subsample (the small differences from
  the 487-track baseline row are sampling, not the model doing anything).
  It was run on a seeded 50-track subsample (`--untuned-limit`, seed 0)
  because the full split would have spent ~9 hours confirming the same 0%.
  What it shows is that **the output format itself has to be taught**.
- **Untuned base model, instructed zero-shot.** The fair "just prompt the
  base model well" baseline: the same per-arm inputs, wrapped in Qwen3's chat
  template (thinking disabled) with a system prompt that spells out the JSON
  keys, value ranges, and the allowed `Era`/`MoodTag` values (read from
  `schema.py`). A 5-example probe parsed 5/5, but the predictions were poor:
  "2000s" for four 1970s tracks, valence and intensity skewed high, a
  measured `danceability` value copied straight into `valence`, and the song
  title returned as the `lyrical_theme`. The full run confirmed that
  pattern at scale (see "Why zero-shot fails" below).
- **Fine-tuned arms.** LoRA adapters trained on the teacher's labels (see
  "Setup"), generating free-form (not teacher-forced) on the test prompts.
  Parse failures are under 1% for all three.

## What the numbers say

1. **Fine-tuning works, and it's the lyrics doing most of the work.** Arm A
   alone nearly triples mood_tags exact match over the baseline (6.4% →
   18.5%) and cuts valence MAE by a third.
2. **Audio adds on top of lyrics, mainly for intensity.** C vs A: intensity
   MAE −22%, valence MAE −9%, era and mood_tags flat.
3. **Audio alone is a weak signal for valence.** Arm B's valence MAE (0.157)
   is within 0.006 of the baseline. Its intensity MAE (0.104) is clearly
   better than baseline, though still worse than lyrics alone.
4. **Era comes from metadata.** All three fine-tuned arms land at ~61% era
   match regardless of whether they see lyrics, audio, or both. The shared
   input is track/album/artist names, so that's where the era signal is.
5. **mood_tags is the hardest field for B.** Without lyrics, exact match is
   9.2%, barely above the 6.4% floor.
6. **The student is well short of the teacher on the categorical fields.**
   mood_tags exact match is 18.9% against a 64.8% ceiling, and era is
   61.8% against 90.5%. Valence and intensity are closer in relative terms
   but still well above the teacher's ~0.02 MAE. See the gap table above.
7. **Instructions alone don't get a 0.6B model there.** Zero-shot is below
   the floor on every metric, in every arm. Its best numbers are arm C's
   intensity ρ of 0.24 (weak but nonzero, probably from the measured
   energy features) and arm A's valence ρ of 0.16.

## Why zero-shot fails

From the per-example predictions (`data/eval_predictions/zero_shot_*.jsonl`):

- **It predicts nearly the same label for everything.** "euphoric" appears
  in the mood_tags of 87–92% of parsed predictions in every arm (431 of 468
  in arm A), usually alongside "triumphant" or "romantic".
- **It's biased high.** Mean predicted intensity is 0.94 in arm A (0.69 in
  B, 0.79 in C) against a true mean of 0.52. Mean predicted valence is
  0.63–0.76 against a true 0.52. That bias alone explains most of the MAE.
- **Its eras skew recent.** It picks the 2000s or 2020s for 88–94% of
  tracks. The test set's most common eras are the 2010s (206), 2020s (118)
  and 1970s (117), and it picks the 1970s at most a handful of times.
- **Most parse failures are valid JSON that breaks the schema.** Across
  arms A and C (B had none), 40 of 53 failures are a `lyrical_theme` over the
  60-character limit the prompt states, 6 are mood_tags outside the allowed
  vocabulary or over the 3-tag limit, 2 are invalid eras, and 5 are cut-off
  or malformed JSON. Arm C's longer inputs produce the most failures (7.0%).

A larger instruction-tuned model, or few-shot examples in the prompt, would
likely do better. Neither changes the comparison this table exists for:
at this model size, the labels have to be taught.

## Setup

Labels: `data/labels.jsonl`: 3,492 tracks labelled by the teacher,
`claude-sonnet-5` (top 3,000 by play count plus a 500-track stratified
sample from the 1–2-play tail, 8 schema-validation failures dropped). 91.4% of them have measured audio
features. The set was rebuilt after the first round of fine-tunes ran on only 198 tracks with audio (see "Before the data fix" below).

The supplemental tail sample. The top 3,000 tracks by play count are
effectively "played 3+ times", which leaves out the 80% of the library
played only once or twice. To get some of that tail into the labels, 500
tracks with `play_count <= 2` were sampled at random: 250 skipped every time
(`skip_rate >= 0.99`) and 250 never skipped (`skip_rate = 0`). The list is
in `data/supplemental_track_ids.txt`, and they went through the same
matching, feature extraction and labelling as the top 3,000 (249 + 250
labelled).

Mood-tag share (% of tracks whose tags include it) and averages by group:

| Group | n | euphoric | aggressive | triumphant | playful | chill | melancholic | nostalgic | intensity | has lyrics |
|---|---|---|---|---|---|---|---|---|---|---|
| Top 3,000 | 2,993 | 19.0% | 9.9% | 11.7% | 23.6% | 39.8% | 42.0% | 42.7% | 0.488 | 77.1% |
| Skipped once | 249 | **29.3%** | **15.3%** | **19.7%** | 29.3% | 30.5% | 35.3% | 33.7% | **0.541** | 77.9% |
| Completed once | 250 | 13.6% | 9.2% | 11.6% | 22.0% | **46.4%** | 44.4% | 39.6% | 0.472 | **61.2%** |

- **Tracks skipped after one play lean high-energy.** They carry more
  euphoric, aggressive and triumphant tags and have higher intensity, with
  fewer nostalgic and melancholic ones. They're also less often from the
  1970s (14% vs 27% for the top 3,000) and more often from the 2010s.
- **Tracks finished once but never replayed lean calm.** chill is the most
  common tag (46%), euphoric the least common of the three groups, and
  fewer have lyrics available (61% vs 77%), which suggests more
  instrumental or obscure tracks. 58% are from the 2010s.
- **The net effect on the overall label distribution is small.** No
  mood tag's share moves by more than 0.9 points between the top 3,000
  alone and the combined 3,492, and mean valence is unchanged (0.534 vs
  0.533). That's partly because 500 is a small share of 3,492, and partly
  because the two halves pull in opposite directions (euphoric is up in
  one, down in the other).

So the sample adds real variety at the level of individual tracks without
shifting the overall balance the models train on. Of the 487 test tracks,
71 come from this sample (30 skipped, 41 completed). Whether the models do
worse on those tail tracks than on the top 3,000 can't be answered yet: it
needs the fine-tuned rows' per-example predictions, which come from the
optional `--tuned-only` run (see "Still open").

The teacher labels describe the song, not how this listener reacted to it.
"Skipped once" says something about the listener. It isn't a label.

Split: By artist, never by track, so the model can't score well by
memorising artists: 2,359 train / 646 val / 487 test, 790 / 169 / 170
artists. Identical across arms. Only the prompt differs.

Training: `Qwen/Qwen3-0.6B`, LoRA r=8, alpha=16, targets `q_proj` and
`v_proj`, 3 epochs, CPU-only (Ryzen 5 3600). Prompt tokens are masked out of
the loss. Examples whose completion would be fully truncated at
`max_length=768` are dropped (see "Bugs"), which is why A and C train on
slightly fewer examples than B.

| Arm | train / val used | eval_loss (epoch 1 → 2 → 3) |
|---|---|---|
| A (lyrics + metadata) | 2,310 / 636 | 0.5784 → 0.5608 → **0.5565** |
| B (measured audio + metadata) | 2,359 / 646 | 0.6678 → 0.6408 → **0.6319** |
| C (lyrics + audio + metadata) | 2,277 / 630 | 0.5851 → 0.5485 → **0.5431** |

The eval_loss ordering (C < A < B) matches the task-level ordering in the
main table, but eval_loss is next-token perplexity on the label JSON, not
correctness. The main table is the real answer.

Eval: `selector.tagger.eval`. Greedy decoding, `max_new_tokens=80` for the
raw-prompt rows (true completion lengths: min 48, p99 67, max 70 tokens),
120 for zero-shot (the instructed model pretty-prints and wraps the JSON in
a code fence). Generation stops as soon as one complete `{...}` object has
been emitted. The first JSON object in the output is extracted and validated
against `PredictedLabels`. Failures fall back to the train-mean prediction
and are counted in `parse_fail`, so they can't shrink n or disappear
silently. Metrics: MAE and Spearman for valence/intensity, exact match for
era and for the mood_tags set.

Wall time on CPU: 4h04m for the main run (3 × 50 untuned + 3 × 487
fine-tuned generations), and roughly 7h for the zero-shot run (3 × 487),
which has longer prompts and outputs. The stopping criterion only checks
brace balance, not schema validity, so it still fires early on most
zero-shot failures (over-length `lyrical_theme`, bad `mood_tags`/`era` -
48 of the 53, see below). Only the 5 cut-off/malformed ones ran the full
120-token budget.

## Bugs found and fixed along the way

- **Fully truncated examples NaN'd eval loss.** When a lyrics-heavy prompt
  alone reached `max_length`, the completion was truncated to nothing, so
  every label token was masked. One such example makes the averaged eval
  loss NaN. Fixed by `_drop_fully_masked` in `train.py`.
- **No EOS token in training completions.** `train.py` never appends EOS, so
  the fine-tuned model has no learned stop signal and keeps generating after
  a valid label, typically into a second lookalike object. Parsing from the
  first `{` to the last `}` made every fine-tuned prediction unparseable in a
  smoke test. Fixed by extracting only the first complete JSON value
  (`json.JSONDecoder.raw_decode`), plus a stopping criterion that ends
  generation once one object is complete. The stopping criterion roughly
  tripled throughput. Appending EOS at training time is the cleaner fix for
  a future retrain.

## Before the data fix

The first round of fine-tunes (adapters preserved in
`data/runs_pilot_audio_v1/`) trained on labels where only 198 of 3,000
tracks (6.6%) had any measured audio, because audio matching had only been
run for a 200-track pilot. 93% of arm B's training prompts said
`Measured audio features: none available`, so that round couldn't answer the
audio question at all. It never got a task-level eval. Its eval_loss values
aren't comparable to the table above (different label set and split). The fix
was to scale audio matching and feature extraction to the full top 3,000
tracks, add the supplemental tail sample, relabel from scratch and retrain all
three arms.

## GPU inference, latency, and cost

CPU generation is far too slow to tag all 19,386 warehouse tracks (a
day-plus of wall time at the measured CPU rate, see below), so `infer.py`'s
real inference path runs through llama.cpp's Vulkan backend instead of the
CPU path this doc's main table used. The full setup, the parity check that
justified trusting it, and the reproduction commands are in
`docs/GPU_INFERENCE.md`. Summarised here:

**Single-request median generation latency** (`n_predict=80`, greedy):

| Path | Median |
|---|---|
| CPU (Ryzen 5 3600), arm C | 13.34s |
| GPU (RX 5600 XT, Vulkan), arm A | 2.891s |
| GPU (RX 5600 XT, Vulkan), arm C | 2.907s |

About 4.6x faster per request, and a further ~2.5x on top of that from
running 4 requests concurrently against each arm's server (1.14s/example
wall for arm A, 1.29s/example for arm C, measured across the full 487-track
test split during the parity check below).

**JSON-validity rate** is the complement of `parse_fail` in the main table
above: 99.4% for arm A, 99.2% for arm B, 99.2% for arm C on CPU.

Cost per thousand tracks vs the teacher. The teacher
(`claude-sonnet-5`) costs $5.03 per thousand tracks (`docs/TEACHER.md`).
The fine-tuned student has no equivalent per-call cost once trained, it
runs locally with no metered API in the loop, so the honest framing is that
the teacher's cost was a one-time price for ~3,500 labelled training
examples, and every track inferred afterwards (the other ~15,900, or any
future one) is free at the margin. See `docs/GPU_INFERENCE.md` for the
fuller version of this argument.

Parity check: Before trusting the GPU path, the same 487-track test
split was re-scored through both arm servers and compared against the CPU
rows above. Every metric landed within noise (≤0.001 MAE, ≤0.8 percentage
points on match rates). The full table is in `docs/GPU_INFERENCE.md`.

## Llama-3.2-1B: dropped

The comparison model was never trained. It needed the same CPU-only LoRA
pipeline as Qwen3-0.6B (training has no GPU path here, llama.cpp is
inference-only, see `docs/GPU_INFERENCE.md`), which would mean a second
multi-hour-per-arm CPU fine-tune run for a model roughly 1.7x the parameter
count, on top of the three arms already trained. Decided with the project
owner (2026-09-23) to drop it rather than spend that time: Qwen3-0.6B
already answers this component's headline question (does fine-tuning work,
and does audio help), a second base model's main plausible contribution -
showing whether a bigger sub-1B model closes more of the floor-to-ceiling
gap, is a real question but a secondary one, and the eval table's honesty
doesn't depend on having it. Left here rather than silently dropped, per
this doc's own standard for reporting what wasn't run and why.

## The full-warehouse run

`infer.py`'s real inference path (item 6) has run: `data/track_features.parquet`
holds a real fine-tuned prediction for every one of the 19,386 tracks in
the warehouse, replacing the earlier dry-run baseline.

| | n | % |
|---|---|---|
| Arm A (no audio match) | 2,647 | 13.7% |
| Arm C (matched audio) | 16,739 | 86.3% |
| `label_source: finetuned_gpu` | 19,278 | 99.4% |
| `label_source: parse_fallback` | 108 | 0.6% |

The parse-fallback rate (1.9% for arm A, 0.4% for arm C) is in the same
range as the CPU/GPU eval rows above, as expected since it's the same
models generating. Every row carries `arm` and `label_source`, so a
downstream consumer (the fly brain, the DJ agent) can tell a
fine-tuned prediction from a fallback rather than treating them the same.

**Update: audio expansion.** The rows above originally reflected the
top-3,000-by-play-count audio scope (16.5% arm C). `docs/AUDIO_MATCHING.md`'s
resolve pipeline was re-run over the full warehouse, taking audio coverage
to 86.4%. The 13,541 tracks that gained a match were re-tagged with the
audio arm via `infer.py --retag-ids` (GPU inference only, no new teacher
labels, so the cost figures in `docs/TEACHER.md` are unaffected). The
numbers above are post-expansion. Effect on the fly brain and skip
prediction: `docs/MBON_EVAL.md`.

## Still open

- **Optional overnight run: fine-tuned per-example predictions.** Not run.
  The main run printed aggregates only, before `eval.py` started writing
  per-example predictions (`data/eval_predictions/<row>.jsonl`) and
  printing Spearman. Running
  `uv run python -m selector.tagger.eval --tuned-only` (~3–4h on CPU)
  regenerates just the three fine-tuned rows and would allow:
  - a paired bootstrap on C vs A, to show whether the valence gain is real;
  - Spearman values for the fine-tuned rows;
  - a comparison of model error on the supplemental tail tracks vs the top
    3,000 (71 of the test tracks come from the tail sample).

## Reproducing

```powershell
# main table (baseline, ceiling, untuned raw-prompt x3, fine-tuned x3)
uv run python -m selector.tagger.eval
# fine-tuned rows only, e.g. to regenerate their per-example predictions
uv run python -m selector.tagger.eval --tuned-only
# instructed zero-shot rows (~7h)
uv run python -m selector.tagger.eval --zero-shot-only
```

Adapters are read from `data/runs/{A,B,C}/adapter` (gitignored).
