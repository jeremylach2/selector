# Audio matching — pilot results (top 200 tracks)

Spotify removed 30-second preview URLs for new apps in November 2024, so
Step 8's pipeline (`selector/audio/resolve.py` + `selector/audio/fetch.py`)
resolves tracks against two public, keyless catalog APIs instead: the
**iTunes Search API** (tried first) and the **Deezer public catalog API**
(fallback). Neither is scraped, and no auth token is needed for search.

This is a **pilot run** — the top 200 tracks by play count, not the full
top-3,000 scope in the plan — done first to validate matching quality and
the download pipeline before committing to the larger, slower run. Scaling
to 3,000 is a config flag (`--limit 3000`) away; nothing about the pipeline
changes.

## Headline number

**99.0% match rate (198/200).**

That's expected to look unrealistically good relative to the plan's
75-90%-mainstream estimate — the top 200 tracks by play count are, almost
by definition, the most mainstream, best-catalogued 200 tracks in the whole
library. The match rate on the full top-3,000 run, once done, will include
much more of the tail and should land closer to the plan's estimate; this
pilot number is a ceiling, not the final figure.

By play-count decile (0 = most-played, 9 = least-played of the 200):

| Decile | Match rate |
|---|---|
| 0-3 | 100.0% |
| 4 | 90.0% |
| 5-9 | 100.0% |

The one dip at decile 4 is noise at this sample size (one miss out of 20),
not a real trend — see the two rejected candidates below, both of which are
older catalog tracks (Eagles, Steely Dan) where the closest API result
apparently didn't clear the confidence bar, not tracks the APIs don't carry
at all.

## Matched source split

Of the 198 matches: **121 from iTunes, 77 from Deezer** (iTunes is tried
first and Deezer is only queried as a fallback, so this split roughly
reflects how often iTunes alone was already confident enough to stop).

## Borderline matches, sampled for manual review

The scoring function (`score_candidate` in `resolve.py`) weights normalised
title similarity (55%) and artist similarity (45%), and subtracts a 0.35
penalty when one side has a modifier the other doesn't (live/remix/cover/
acoustic/remaster/demo/instrumental) — a "Live" recording is a different
recording of the song, not a fuzzy-matching nuisance to shrug off.
`DEFAULT_MATCH_THRESHOLD = 0.72`; anything scoring below that is recorded as
unmatched rather than guessed at, on the theory that a silent bad match
poisons every downstream audio feature.

A sample of scores near that threshold, from the actual pilot run:

```
[0.84] 'Kingston Wall'/'Shine On Me - 2023 Mix' -> 'Kingston Wall'/'Shine On Me (2023 Mix)' (deezer)
[0.82] 'The Beatles'/'Something - 2019 Mix' -> 'The Beatles'/'Something (2019 Mix)' (deezer)
[0.84] 'Fleetwood Mac'/'Dreams - 2004 Remaster' -> 'Fleetwood Mac'/'Dreams (2004 Remaster)' (itunes)
[0.75] 'Prince'/'Purple Rain - 2015 Paisley Park Remaster' -> 'Prince'/'Purple Rain (2015 Paisley Park Remaster)' (deezer)
[0.83] 'Jimi Hendrix'/'Purple Haze' -> 'The Jimi Hendrix Experience'/'Purple Haze' (itunes)
[0.80] 'Σtella'/'Charmed' -> 'Σtella & Redinho'/'Charmed' (itunes)
[0.76] 'Zach Bryan'/'Mine Again' -> 'Zach Bryan'/'Mine' (itunes)  <- accepted; borderline, worth a second look
[0.59] 'Eagles'/'Peaceful Easy Feeling - 2013 Remaster' -> no candidate cleared the threshold (rejected)
[0.65] 'Steely Dan'/"Reelin' In The Years" -> no candidate cleared the threshold (rejected)
```

Two things worth calling out honestly:

- **"Mine Again" → "Mine" at 0.76** is the scoring function's weakest
  accepted match in this sample: these read as two different Zach Bryan
  songs with similar titles, not the same recording with formatting noise.
  This is exactly the failure mode the threshold exists to catch, and here
  it let one through — a real risk of the 0.72 cutoff being slightly too
  permissive for short, generic-word titles. Worth revisiting before
  scaling to 3,000 tracks, where more collisions like this are likely.
- The two **rejected** tracks (Eagles, Steely Dan) are classic-rock catalog
  staples that are almost certainly available from both APIs — the rejection
  is a scoring/normalisation miss (probably remaster-suffix handling, or a
  title variant the query didn't anticipate), not evidence the song is
  actually absent from either catalog. Recorded as unmatched rather than
  guessed at, per the "reject over guess" policy above.

## Format split

Of the 198 clips downloaded: **iTunes serves AAC-in-M4A, Deezer serves
plain MP3.** `libsndfile` (soundfile's backend, and librosa's default
loader) has no AAC decoder at all, so `features_librosa.py` falls back to
PyAV (which bundles static ffmpeg libraries) for `.m4a` files — see the
`_load_via_pyav` docstring there. No system-level ffmpeg install was needed.

## Known limits of this pilot

- 200 tracks, not 3,000. The full run will surface more of the mismatch and
  rejection cases seen above.
- No manual listen-through was done to confirm every accepted match is
  actually the right *recording*, not just the right *song* — the "Mine
  Again"/"Mine" case above suggests that check is worth doing before this
  scales further.
