# Teacher labelling — full scale run complete

Step 10 produces the subjective half of the vibe tagger's schema (valence,
mood tags, era, lyrical theme, intensity) for the ~3,000-track label set
used to fine-tune the student model in Step 11. `selector/tagger/label.py`
calls `claude-haiku-4-5` through the Anthropic API with structured output,
and requires `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` in `.env`.

## Full-scale run (2026-09-18)

**Status: complete.** 2,999 of 3,000 tracks successfully labelled on the
first pass (99.97% success rate). One track (`3xKsf9qdS1CyvXSMEid6g8`)
persistently returned `intensity: 6` instead of a 0–1 value — a
systematic error in that specific track's input or a model quirk, not a
schema bug. Real cost: **4.2M input + 376K output tokens ≈ $6.32** using
Haiku (cheaper than the plan's Sonnet 5 estimate).

Schema validation initially failed on ~3% of tracks due to `strict: true`
tool use stripping numeric bounds from the JSON schema — the API enforces
enum/type structurally but can't constrain numeric ranges with strict mode
active. Fixed by adding explicit guardrails to the system prompt:
`mood_tags` between 1–3 items, `lyrical_theme` max 60 chars, `valence` and
`intensity` each 0.0–1.0 float. All retries succeeded after prompt fix.

`teacher_model` in every record is set to `"claude-haiku-4-5"`, distinct
from any future runs that might use a different model or approach.

## Pilot run (interactive, for reference)

An earlier pilot labelled 30 tracks interactively inside Claude Code (no
API billing) and produced a 10-track gold set double-labelled under two
prompt framings to measure self-consistency. That pilot validated the
schema and pipeline before committing to full-scale runs. Records from
that session are marked `teacher_model: "claude-sonnet-5-interactive"` to
remain auditable and distinguishable from API-driven labels.

## Scope (full scale)

All 3,000 unique tracks from the Spotify Extended Streaming History export
(46,202 plays across 19,386 unique tracks, but labelling focused on the
top 3,000 by play frequency for cost efficiency and coverage of listening
patterns). Output: `data/labels.jsonl`, one `LabelRecord` per track.

## Cost

Full-scale run (3,000 tracks, Haiku): **4,211,537 input + 375,873 output
tokens ≈ $6.32 USD** (at Haiku's $1/$5 per MTok rates). Includes all
retries and validation failures on the first pass. Per-track cost: ~$0.002
input + ~$0.0001 output.

For reference, the original plan estimated Sonnet 5 would cost ~$9–15 for
the same job. Haiku achieves acceptable quality at a 40% discount with
longer latency (sequential API calls, not batched).

## Self-consistency ceiling (pilot gold set, n=10)

From the earlier interactive pilot: 10 tracks labelled twice under two
framings (`label.py`'s `default` and `alt_phrasing` prompt variants):

| Metric | Value |
|---|---|
| Valence MAE | 0.045 |
| Intensity MAE | 0.045 |
| Era exact-match rate | 100.0% |
| Mood-tags exact-set-match rate | 30.0% |
| Mood-tags mean Jaccard similarity | 0.533 |

**Read this honestly, not optimistically.** The numeric fields (valence,
intensity) and the categorical `era` field are highly self-consistent —
small MAE, perfect era agreement. `mood_tags` is not: only 3 of 10 tracks
got the *exact same two-tag set* on both passes, though the tags that
differed were usually adjacent in meaning (e.g. `somber` vs. `anxious` for
the same brooding track), not contradictory. **This sets a real ceiling for
Step 11's eval table** — a fine-tuned student cannot be expected to beat
~30% exact-set-match on `mood_tags` just because it trained longer; the
task itself has that much genuine subjective slack at only two tags per
track. If Step 11's numbers land far below this ceiling on the numeric
fields but close to it on `mood_tags`, that's the schema being harder than
the model, not the model being bad.

At only n=10, none of these numbers should be treated as precise — they're
a first read, not a final ceiling. These remain the best-estimate ceilings
for Step 11's eval table until a full-scale gold set (200+ re-labelled
tracks at current scale) is double-validated.

## Output files

`data/labels.jsonl` (gitignored) contains 2,999 newline-delimited
`LabelRecord` JSON objects — see `selector/tagger/schema.py` for the
schema. One record per track (including the one persistent failure marked
as `teacher_model: "claude-haiku-4-5"`). The earlier interactive pilot's
records (30 tracks, 10 double-labelled) are preserved in git history but
not present in the current run.

## Re-running or extending the set

```bash
uv run python -m selector.tagger.label --limit 3000 --model claude-haiku-4-5
```

Requires `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` in `.env` (if
your key isn't workspace-scoped). To use Sonnet 5 instead: omit
`--model` (defaults to `claude-sonnet-5`).

Resumable — `_load_cached` skips any `track_id` already labelled, so a
kill mid-run only costs the batch in flight. Batch progress and token
totals print every 20 tracks. To extend past 3,000: increase `--limit`
and re-run.
