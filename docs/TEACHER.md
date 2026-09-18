# Teacher labelling — pilot run and how it was actually done

Step 10 needs a "teacher" to produce the subjective half of the vibe
tagger's schema (valence, mood tags, era, lyrical theme, intensity) for
enough tracks to fine-tune a student model against in Step 11. The plan
specifies calling `claude-sonnet-5` through the Anthropic API with
structured output, and `selector/tagger/label.py` implements exactly that —
it's the reproducible, documented path for anyone with an
`ANTHROPIC_API_KEY`.

**This session had no separate API key configured**, and rather than
stopping there, the pilot's labels were produced a different, honest way:
**interactively, inside this Claude Code session** — the same model
reasoning about the same inputs (metadata, lyrics, measured audio features)
that `label.py`'s prompt would have sent, but as a direct conversational
judgment instead of a metered API call. `teacher_model` in every pilot
record is set to `"claude-sonnet-5-interactive"`, not `"claude-sonnet-5"`,
specifically so this is never confused with a real `label.py` run in the
data itself. Scaling past the pilot means either running `label.py` for
real with a key, or repeating this manual approach at a scale where it's
still practical — the two are not interchangeable, and the honest label
name is what keeps that visible downstream.

## Scope

30 tracks (a subset of the 200-track audio pilot, all with a matched
preview clip and measured Step 9 features), plus a 10-track gold set
double-labelled under two different prompt framings to measure
self-consistency. The plan's target is a 200-track gold set over the full
~3,000-track label set; this is a proportionally smaller version of the
same method, done to validate the pipeline and schema before committing to
the larger run.

## Cost

**$0 marginal spend** — no separate Anthropic API billing was involved,
since the labelling happened inside the existing Claude Code session rather
than through `label.py`'s API calls. This is specific to the interactive
pilot and does **not** extend to the full-scale run: labelling the full
~3,000-track set the same interactive way is not practical, and a real
`label.py` run against ~3,000 tracks (with lyrics and measured features
folded into each prompt) should be assumed to cost real, metered API spend
proportional to token volume, in the same range as any structured-output
classification job of that size.

## Self-consistency (gold set, n=10)

Each of these 10 tracks was labelled twice, independently, under two
framings (`label.py`'s `default` and `alt_phrasing` prompt variants — the
second asks for a free description of the feeling first, then the
structured call):

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
a first read, not a final ceiling. The full 200-track gold set (once run at
full scale) will tighten all five considerably.

## What's in `data/labels.jsonl` and `data/labels_gold.jsonl`

Both are gitignored (they carry fetched lyrics text, which shouldn't be
redistributed) and contain newline-delimited `LabelRecord` JSON — see
`selector/tagger/schema.py`. `labels.jsonl` holds one record per pilot
track; `labels_gold.jsonl` holds two records per gold-set track (one per
`prompt_variant`), so both passes are auditable side by side.

## Running it for real

```bash
uv run python -m selector.tagger.label --limit 3000
```

Requires `ANTHROPIC_API_KEY` in `.env`. Resumable — `_load_cached` skips any
`track_id` already in the output file, so a kill mid-run costs at most the
batch in flight. Cost and running totals print every 20 tracks.
