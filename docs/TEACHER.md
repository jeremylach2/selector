# Teacher labelling

Step 10 produces the subjective half of the vibe tagger's schema (valence,
mood tags, era, lyrical theme, intensity) for the label set used to
fine-tune the student model in Step 11. `selector/tagger/label.py` calls
the teacher through the Anthropic API with strict tool use against the
`PredictedLabels` schema, and requires `ANTHROPIC_API_KEY` (plus
`ANTHROPIC_WORKSPACE_ID` if the key isn't workspace-scoped) in `.env`.

## Current label set (2026-09-19)

**Teacher: `claude-sonnet-5`** (`label.py`'s `DEFAULT_MODEL`). Every record
in `data/labels.jsonl` has `teacher_model: "claude-sonnet-5"` and
`prompt_variant: "default"`.

**Scope:** 3,500 tracks: the top 3,000 by play count plus a 500-track
stratified sample from the tail of tracks played only once or twice (250
always skipped, 250 never skipped). See `docs/DATA_FIX_PLAN.md` Step 4 and
`docs/EVAL.md` ("The supplemental tail sample") for why and what it
changed. Built by `scripts/label_full_with_supplemental.py`, because
`label.py`'s CLI only takes `--limit N`.

**Result:** 3,492 of 3,500 labelled (99.8%). The 8 failures are responses
that failed schema validation on both attempts, logged by track id in
`data/step5_label.log` and not investigated further.

**Inputs the teacher saw:** track, album, artist, lyrics from lrclib where
available (see `docs/LYRICS.md`), and measured audio features where a
preview clip was matched. 91.4% of records (3,190) have measured features.
Before the data fix, only 6.6% did.

## Cost

Exact totals from `data/step5_label.log`:

| | Tokens | Rate (Sonnet 5) | Cost |
|---|---|---|---|
| Input | 6,006,012 | $2 / MTok | $12.01 |
| Output | 557,909 | $10 / MTok | $5.58 |
| **Total** | | | **$17.59** |

That's **$5.03 per thousand tracks attempted**, or about 1,716 input and
159 output tokens per track. Retries on validation failures are included.
Calls were sequential, not batched. The Message Batches API would halve
this.

## Self-consistency ceiling (gold set, n=199)

Run 2026-09-23 with `scripts/gold_set.py`. A seeded random 200 tracks
(seed 0) from `data/labels.jsonl` were relabelled by `claude-sonnet-5` with
`label.py`'s `alt_phrasing` prompt variant, which asks the teacher to
describe how the song feels and what it's about before labelling. The
original `default` labels are the first pass, so the only thing that
changes between passes is the prompt wording. One relabel failed schema
validation, leaving 199 pairs. Second-pass labels are in
`data/labels_gold_200.jsonl`, the log in `data/step10_gold.log`.

Cost: 341,868 input + 32,167 output tokens, **$1.01**.

| Metric | Value |
|---|---|
| Valence MAE | 0.017 |
| Intensity MAE | 0.022 |
| Era exact-match rate | 90.5% |
| Mood-tags exact-set-match rate | 64.8% |
| Mood-tags mean Jaccard similarity | 0.837 |

The teacher barely moves on valence and intensity when the prompt is
reworded. Era changes on about 1 track in 10. mood_tags is the most
subjective field, but the exact tag set still matches two times in three,
and the average overlap is high. This is the ceiling row in
`docs/EVAL.md`'s table and the `TEACHER_SELF_CONSISTENCY` constant in
`eval.py`.

It measures sensitivity to prompt wording with the same inputs, not
agreement between independent annotators, so it's a ceiling on how
consistently this teacher labels, not on how "correct" the labels are.

**The earlier 10-track estimate** (pilot, done interactively, records in
`data/labels_gold.jsonl`) put valence and intensity MAE at 0.045, era at
100% and mood_tags exact match at 30%. The 200-track run replaces it. The
biggest change is mood_tags: 3 of 10 was a small-sample low.

## History: the first full run (2026-09-18, superseded)

The first full-scale run used `claude-haiku-4-5` on the top 3,000 tracks
only: 2,999 labelled, 4,211,537 input + 375,873 output tokens, about $6.32.
It was superseded because only 198 of those tracks had measured audio
features at the time (the audio pipeline had only run on a 200-track
pilot). The file is backed up as `data/labels_pilot_audio_v1.jsonl`.

Two things learned in that run still apply:

- Strict tool use enforces types and enums but rejects numeric range and
  length keywords, so `label.py` strips them from the tool schema and states
  the limits in the system prompt instead (1–3 mood tags, `lyrical_theme`
  ≤ 60 chars, valence and intensity as 0.0–1.0 floats). Responses are
  re-validated against the pydantic model afterwards.
- One track persistently returned `intensity: 6`, a 1–10 scale instead of
  0–1. The system prompt now says explicitly never to use a 1–10 scale.

## Re-running or extending the set

```bash
# top-N by play count
uv run python -m selector.tagger.label --limit 3000
# the current 3,500-track set (top 3,000 + supplemental sample)
uv run python scripts/label_full_with_supplemental.py
```

Resumable: `_load_cached` skips any `track_id` already in the output file,
so a kill mid-run only loses the call in flight. That also means a rerun
after the inputs change (new audio features, new lyrics) must start from a
fresh output file, or stale labels are silently kept. Progress and running
token totals print every 20 tracks.
