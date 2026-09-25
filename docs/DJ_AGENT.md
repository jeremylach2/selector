# The DJ agent

The DJ reads recent listening and plans a themed set of about 45 minutes. It
checks its own work, and only then builds the Spotify playlist, with liner
notes that explain each transition. It is five modules in `src/selector/dj/`,
each with a plain-data output, so every decision in a run can be inspected.

```
Brief ──► Arc ──► Select ──► Critique ──pass──► Commit
                    ▲            │
                    └──reject────┘   (once)
```

| Stage | Module | Reads | Decides |
|---|---|---|---|
| Brief | `brief.py` | recent plays (live Spotify if a token is cached, else the warehouse), local time, weekday | theme, seed tracks, rationale |
| Arc | `arc.py` | theme | energy curve over the set: opener, build, peak, comedown; transition rules |
| Select | `select.py` | the crate, brief, arc | one track per step, judged at the point in the set where it would play |
| Critique | `critique.py` | the finished set, arc | pass, or reject with the tracks to swap and a stricter config |
| Commit | `commit.py` | set, verdict | liner notes; the playlist, only if `dry_run` is off **and** critique passed |

`agent.py` wires the stages together and writes the full chain to
`data/dj_runs/<timestamp>-<theme>.json`, with the liner notes beside it as
`.md`. The chain covers the brief, arc curve, each Select step's top-5
shortlist with scores, each critique verdict, the revision, and the commit
result.

## The crate: measured, predicted, learned

Only tracks with **measured** audio features are playable. That is 3,198 of
19,386, the top tracks by play count that matched a preview clip in Step 8.
The arc is a hard constraint, and checking it against a model's guess at
energy would make it a constraint on nothing. Each crate row combines four
sources:

| Source | Columns | Used by |
|---|---|---|
| Measured (DSP on a 30 s preview) | `tempo`, `energy` | Arc, Select transitions, Critique, liner notes |
| Predicted (fine-tuned tagger) | `mood_tags` | Brief theme choice, Select theme fit |
| Learned (fly brain) | fly tag row, `fly_valence` → `taste` | Select coherence and taste |
| Observed (warehouse) | durations, recency | set length, familiar/fresh split |

**Energy** is a weighted sum of scaled RMS loudness (0.5), onset density
(0.25) and spectral centroid (0.25). These are loudness, activity and
brightness, the ingredients Spotify documented for its removed `energy`
feature. The sum is rank-normalised over the crate, so a target of 0.8
means louder than 80% of the crate.

**Tempo** comparisons are octave-folded, because librosa's beat tracker
routinely reports half- or double-time. With the fold, 90 → 180 BPM counts
as the same pulse, not a 100% jump.

**Taste** is the production mushroom body's valence as a percentile. Raw
valence is positive for every crate track, because play-outs outnumber skips
about 4:1 in the history, so only its rank carries information.

**Familiar vs fresh.** Every crate track has been played, since the crate is
drawn from listening history. So "fresh" honestly means *not played in the
last 90 days*, i.e. a rediscovery, not an unheard track.

## Select

Select walks forward through the set. For each candidate it computes where
that track's midpoint would land, given its own duration, and looks up the
arc target there. Hard filters:

1. Energy is within ±0.15 of that target.
2. The artist has fewer than 2 tracks in the set so far.
3. The track is the kind the familiar ratio currently calls for.
4. In revision mode only, the move from the previous track isn't jarring.

When nothing qualifies, the filters relax in a fixed order, and every
relaxation is logged: ratio first, then the transition rule, then the band
widens to 1.5× and 2×.

Survivors are scored on these terms:

| Term | Weight | What it measures |
|---|---|---|
| arc fit | 0.25 | closeness to the target energy |
| coherence | 0.25 | fly-brain Dice similarity (normalised Hamming) to the previous pick and the brief's seeds |
| taste | 0.20 | mushroom-body valence percentile |
| theme | 0.30 | mood-tag overlap with the theme |
| transition | −0.15 | tempo and energy jump from the previous pick |

Taste is weighted below theme on purpose. At equal weight, the fly's
favourite few dozen tracks won every slot of every theme, and a 08:00
"slow sunrise" set shared half its tracks with a 22:00 "peak time" set.
After reweighting, seven different contexts overlap by 0–4 tracks.

## Critique

Select is greedy and only ever looks one track back, so it can paint itself
into a corner that only shows when the whole set is read at once. Critique
rejects a set for any of these:

- **off-arc:** a track's energy is outside the band at the point it plays
- **jarring:** an energy step over 0.25, an octave-folded tempo shift over
  24%, or tempo over 12% *and* energy over 0.125 at the same time
- **flat shape:** the peak phase doesn't average at least 0.10 above both
  the opening and the closing track
- **length:** more than 15% off the requested running time
- **artist cap:** more than 2 tracks by one artist

A rejection excludes the offending tracks: the off-arc track, or the
incoming side of a jarring transition. It also switches Select into
revision mode, where the transition rules become a hard filter and the
transition weight doubles. There is exactly one revision. If it also fails,
the run still returns its set and notes, but Commit refuses to write it.

## Results (dry runs, 2026-09-24, live Spotify recent plays)

| Context | Theme chosen | First pass | Final |
|---|---|---|---|
| Fri 22:00 | peak time | **Reject**: 2 jarring (112→83 BPM, 172→112 BPM) | Pass after revision, 12 tracks, 45.3 min, peak lifts 0.31 |
| Tue 08:00 | slow sunrise | Pass, 11 tracks, 43.8 min | — |
| Wed 15:00, `theme="sad rainy day"` | custom (melancholic) | Pass, 12 tracks, 44.1 min | — |

Across a wider sweep of seven contexts during tuning, 4 of 7 first passes
were rejected. All 7 passed after one revision.

Excerpt from the Friday set's notes:

> 7. **The Moment** - Tame Impala `22:18` · build · 118 BPM · energy 0.87
>    Tempo drops 123 -> 118 BPM while energy lifts 0.69 -> 0.87. Fly-brain
>    neighbour of the last track (Hamming 62); fly taste score in the top 4%
>    of the crate; in current rotation.
> 8. **What Is Love - 7" Mix** - Haddaway `26:34` · peak · 123 BPM · energy 0.93
>    Tempo pushes 118 -> 123 BPM while energy lifts 0.87 -> 0.93, into the
>    peak. ...

## Running it

```
# Dry run (default): prints the critique chain and notes, logs to data/dj_runs/
uv run python scripts/dj_cron.py
uv run python scripts/dj_cron.py --theme "night drive" --minutes 60
uv run python scripts/dj_cron.py --at 2026-09-25T22:00     # plan as if it were Friday night

# Create the playlist (private) in your account
uv run python scripts/dj_cron.py --commit
```

From an MCP client, use `dj_set(theme=None, minutes=45, dry_run=True,
familiar_ratio=0.6)`. `scripts/dj_cron.py` has Task Scheduler and cron
lines in its docstring. A scheduled `--commit` needs a cached Spotify token
(`~/.selector/token.json`) and never opens a browser login.

## Honest limits

- Tempo and energy come from a 30-second preview, not the whole track. A
  quiet intro or a late drop is invisible.
- librosa's tempo estimates fall on a discrete grid (for example 112.3,
  117.5, 123.0 BPM), so "tempo locks in" often means two tracks landed on
  the same grid step, not an exact beatmatch.
- The same context and the same recent plays always produce the same set.
  Variety comes from recent listening changing, not from randomness.
- Themes and their energy ranges are hand-written, and mood fit uses the
  tagger's *predicted* mood tags.
