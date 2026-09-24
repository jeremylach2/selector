# Audio matching — full-scope results (top 3,000 tracks)

Spotify removed 30-second preview URLs for new apps in November 2024, so
Step 8's pipeline (`selector/audio/resolve.py` + `selector/audio/fetch.py`)
resolves tracks against two public, keyless catalog APIs instead: the
**iTunes Search API** (tried first) and the **Deezer public catalog API**
(fallback). Neither is scraped, and no auth token is needed for search.

A 200-track pilot (see "Pilot results" below) validated matching quality
and the download pipeline first. This section documents the full
top-3,000-by-play-count run (`--limit 3000`, the default), done per
`docs/DATA_FIX_PLAN.md` Step 2.

## Headline number

**91.5% match rate (2,745/3,000)**, in line with the plan's 75–90%
estimate (slightly above it, in fact) once the long tail is included,
against the pilot's inflated 99.0% ceiling.

By play-count decile (0 = most-played, 9 = least-played of the 3,000):

| Decile | Match rate |
|---|---|
| 0 | 98.7% |
| 1 | 94.7% |
| 2 | 92.0% |
| 3 | 95.0% |
| 4 | 92.7% |
| 5 | 90.3% |
| 6 | 90.0% |
| 7 | 87.3% |
| 8 | 87.7% |
| 9 | 86.7% |

A gentle, expected downward slope from the most- to least-played tracks —
no cliff — meaning the pipeline holds up reasonably well even on the tail,
not just the mainstream top slice the pilot sampled.

## Supplemental tail sample (500 tracks)

`docs/DATA_FIX_PLAN.md` Step 4 added 500 tracks from outside the top 3,000,
all with `play_count <= 2`: 250 skipped every time and 250 never skipped
(see `docs/EVAL.md`, "The supplemental tail sample"). They were resolved
with `resolve.py --track-ids-file data/supplemental_track_ids.txt`.

| Group | Matched |
|---|---|
| Skipped once | 224 / 250 (89.6%) |
| Completed once | 229 / 250 (91.6%) |
| **Total** | **453 / 500 (90.6%)** |

That's in line with the least-played deciles of the top 3,000 (87–88%), so
matching doesn't fall off further on tracks played only once or twice. The
combined pool is 3,500 tracks with 3,198 matched (91.4%).
`data/audio_matches.parquet` covers all 3,500.

## Matched source split — and an operational finding

Of the 2,745 matches: **545 from iTunes, 2,200 from Deezer.** This is the
*opposite* ratio from the 200-track pilot (121 iTunes / 77 Deezer), which
looked suspicious enough to investigate directly. Breaking the split down
by decile explains it:

| Decile | iTunes | Deezer |
|---|---|---|
| 0 | 181 | 115 |
| 1 | 97 | 187 |
| 2 | 120 | 156 |
| 3 | 118 | 167 |
| 4 | 3 | 275 |
| 5 | 9 | 262 |
| 6 | 3 | 267 |
| 7 | 3 | 259 |
| 8 | 6 | 257 |
| 9 | 5 | 255 |

iTunes matches fall off a cliff after decile 3 and effectively stop.
Querying the iTunes Search API directly, post-run, confirms why: it now
returns **HTTP 403** for this run's request pattern. `resolve.py`'s
`_best_candidate` catches `httpx.HTTPError` and treats a failed iTunes call
as "no candidates," silently falling through to Deezer — which is why the
overall match rate stayed healthy (91.5%) despite iTunes going dark partway
through the ~2,800-track run. Apple's Search API has no documented rate
limit, but empirically this volume of sequential calls (even at the
existing 0.2s inter-call delay) triggered a block that a slower pilot run
of 200 calls did not.

**Practical implication for future reruns:** Deezer alone is carrying most
of the match rate here and appears robust at this volume — treat iTunes as
a highest-confidence-when-available bonus, not a load-bearing source, for
any full-scale rerun. If iTunes coverage matters (e.g. its metadata is
occasionally cleaner), consider a longer inter-call delay or a resumable
per-source retry rather than assuming both APIs stay available for the
full run.

## Borderline matches, sampled for manual review

The scoring function (`score_candidate` in `resolve.py`) weights normalised
title similarity (55%) and artist similarity (45%), and subtracts a 0.35
penalty when one side has a modifier the other doesn't (live/remix/cover/
acoustic/remaster/demo/instrumental) — a "Live" recording is a different
recording of the song, not a fuzzy-matching nuisance to shrug off.
`DEFAULT_MATCH_THRESHOLD = 0.72`; anything scoring below that is recorded as
unmatched rather than guessed at, on the theory that a silent bad match
poisons every downstream audio feature.

A sample of scores near that threshold, from the full 3,000-track run:

```
[0.76] 'The Beatles'/'Taxman - 2022 Mix' -> 'The Beatles'/'Taxman (2022 Mix)' (deezer)
[0.84] 'Paul McCartney'/'Let Me Roll It - 2010 Remaster' -> 'Paul McCartney & Wings'/'Let Me Roll It (2010 Remaster)' (itunes)
[0.81] 'Grateful Dead'/"Not Fade Away / Goin' down the Road Feeling Bad - Live at Manhattan Center, New York, NY, April 5, 1971" -> 'Grateful Dead'/"Not Fade Away / Goin' down the Road Feeling Bad (Live at Manhattan Center, New York, NY, April 5, 1971)" (deezer)
[0.63] 'The Beatles'/'Rain - 2022 Stereo Mix' -> no candidate cleared the threshold (rejected)
[0.72] 'Arctic Monkeys'/'505' -> 'BassTon'/'505 (TECHNO)' (deezer)  <- accepted; wrong recording entirely
[0.67] 'Steely Dan'/"Rikki Don't Lose That Number" -> no candidate cleared the threshold (rejected)
[0.82] 'Pink Floyd'/'Money - 2023 Remaster' -> 'Pink Floyd'/'Money (2023 Remaster)' (deezer)
[0.66] 'Yellowcard'/'Only One' -> no candidate cleared the threshold (rejected)
[0.65] 'Elton John'/'Daniel' -> no candidate cleared the threshold (rejected)
[0.65] 'Keith Urban'/'Long Hot Summer' -> no candidate cleared the threshold (rejected)
[0.65] 'Nirvana'/'Breed' -> no candidate cleared the threshold (rejected)
[0.84] 'Eagles'/'My Man - 2013 Remaster' -> 'Eagles'/'My Man (2013 Remaster)' (deezer)
```

Two things worth calling out honestly:

- **'505' → 'BassTon - 505 (TECHNO)' at 0.72** (the exact threshold) is a
  clear false positive: an Arctic Monkeys track matched to an unrelated
  techno remix/bootleg by a different artist that happens to share a title
  token. This is the same failure mode the pilot's "Mine Again"/"Mine" case
  flagged as a risk of the 0.72 cutoff being too permissive for short,
  generic titles — now confirmed at full scale. Worth a follow-up: either
  raise the threshold slightly or add an artist-similarity floor
  independent of the blended score, since here the title similarity alone
  was carrying a match with weak artist agreement.
- Several straightforward catalog tracks (Elton John "Daniel", Nirvana
  "Breed", Yellowcard "Only One") were rejected rather than guessed at —
  consistent with the pilot's Eagles/Steely Dan misses. These are almost
  certainly available from both APIs; the rejection is a
  scoring/normalisation gap (title variant the query didn't anticipate),
  not evidence the song is absent from either catalog. The "reject over
  guess" policy means this shows up as a slightly lower match rate rather
  than a silent bad label, which is the intended trade-off.

## Format split

Of the 2,745 clips downloaded: **iTunes serves AAC-in-M4A, Deezer serves
plain MP3.** `libsndfile` (soundfile's backend, and librosa's default
loader) has no AAC decoder at all, so `features_librosa.py` falls back to
PyAV (which bundles static ffmpeg libraries) for `.m4a` files — see the
`_load_via_pyav` docstring there. No system-level ffmpeg install was needed.
With iTunes now supplying only ~20% of matches (see the operational finding
above), the large majority of clips are MP3 decoded natively by libsndfile,
with PyAV as the AAC fallback for the iTunes minority.

## Known limits of this run

- No manual listen-through was done to confirm every accepted match is
  actually the right *recording*, not just the right *song* — the '505'
  false positive above confirms this is a real, not just theoretical, gap
  at the 0.72 threshold.
- iTunes coverage is now effectively capped by the mid-run 403 block
  described above; Deezer is doing most of the work. A rerun with a longer
  inter-call delay might recover more iTunes-sourced matches, but wasn't
  attempted here since the overall match rate already met the plan's
  target.
- The 255 unmatched tracks (8.5%) were not manually re-queried with relaxed
  parameters — some are likely recoverable with query reformulation (e.g.
  stripping more remaster/edition suffixes before searching).

## Pilot results (200-track pilot, superseded above)

Kept for the before/after record. The pilot was the top 200 tracks by play
count only, run before scaling to the full 3,000-track scope.

**99.0% match rate (198/200)** — expectedly higher than the full-scope
number, since the top 200 by play count are the most mainstream,
best-catalogued tracks in the library; this was flagged at the time as a
ceiling, not the final figure, which the full run above confirms (91.5%).

By play-count decile (0 = most-played, 9 = least-played of the 200):

| Decile | Match rate |
|---|---|
| 0-3 | 100.0% |
| 4 | 90.0% |
| 5-9 | 100.0% |

Source split: **121 from iTunes, 77 from Deezer** — roughly reversed from
the full run's 545/2,200, because the pilot's much lower request volume
never triggered iTunes's rate limiting (see the operational finding above).

Borderline sample from the pilot run:

```
[0.84] 'Kingston Wall'/'Shine On Me - 2023 Mix' -> 'Kingston Wall'/'Shine On Me (2023 Mix)' (deezer)
[0.82] 'The Beatles'/'Something - 2019 Mix' -> 'The Beatles'/'Something (2019 Mix)' (deezer)
[0.84] 'Fleetwood Mac'/'Dreams - 2004 Remaster' -> 'Fleetwood Mac'/'Dreams (2004 Remaster)' (itunes)
[0.75] 'Prince'/'Purple Rain - 2015 Paisley Park Remaster' -> 'Prince'/'Purple Rain (2015 Paisley Park Remaster)' (deezer)
[0.83] 'Jimi Hendrix'/'Purple Haze' -> 'The Jimi Hendrix Experience'/'Purple Haze' (itunes)
[0.80] 'Σtella'/'Charmed' -> 'Σtella & Redinho'/'Charmed' (itunes)
[0.76] 'Zach Bryan'/'Mine Again' -> 'Zach Bryan'/'Mine' (itunes)  <- accepted; flagged at the time as borderline
[0.59] 'Eagles'/'Peaceful Easy Feeling - 2013 Remaster' -> no candidate cleared the threshold (rejected)
[0.65] 'Steely Dan'/"Reelin' In The Years" -> no candidate cleared the threshold (rejected)
```

The pilot flagged "Mine Again"/"Mine" as a risk that the 0.72 threshold
might be slightly too permissive for short, generic titles. The full run's
'505' → 'BassTon - 505 (TECHNO)' match confirms that risk materialized at
scale.

## Feature sanity check (Step 9)

Run 2026-09-23 against `data/audio_features.parquet` (3,198 tracks), to
confirm the extracted features mean what they claim.

**Energy (`rms_mean`): passes.** The quietest clips are Pink Floyd's "Wish
You Were Here" and "Echoes", Hans Zimmer's "Day One (Interstellar Theme)",
Frank Zappa's "Watermelon In Easter Hay" and The Alan Parsons Project's
"Total Eclipse". The loudest are Greta Van Fleet ("Runway Blues", "Waited
All Your Life"), Wolfmother ("Dimension"), DJ Khaled and Black Pumas.
Both ends are what you'd expect.

**Tempo: plausible overall, with octave errors at the top end.** The median
is 117.5 BPM, and values sit on librosa's discrete tempo grid (123.0, 129.2,
117.5, ...). But the highest-tempo list is dominated by songs that are
clearly not that fast: The Beatles' "In My Life" (~103 BPM) reads as 199,
and several slow Hermanos Gutiérrez pieces read as 199. That's the
classic beat-tracker octave error: the tracker locks onto eighth notes and
reports double the tempo. 62 tracks (1.9%) are above 180 BPM, and most of
those are probably doubled. 91 are below 70 BPM, and some of those may be
halved.

Consequences: `tempo_scaled` is a noisy input for a small share of tracks,
and arms B and C (and the teacher, which saw the same values) inherited
that noise. Not fixed. A fix would mean re-extracting features (for example
folding tempos outside roughly 70–180 BPM by a factor of 2, or using
Essentia's tempo estimate), then relabelling and retraining, which isn't
worth it for a ~2–4% slice. Energy, danceability and the Essentia mood
scores are unaffected.
