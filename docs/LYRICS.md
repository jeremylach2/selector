# Lyrics source for the teacher labelling pipeline

Teacher labelling needs lyrics as one input to the teacher (alongside metadata and
measured audio features). This needed a source whose terms actually permit
the use, feeding the text into an LLM call, not republishing it, rather
than the first API that returns text.

## Source: lrclib.net

[lrclib.net](https://lrclib.net) is a free, open, community-contributed
lyrics database built primarily to serve synced lyrics to music player
apps. It was chosen over the obvious alternative (Genius) for a concrete
reason:

- **Genius's official API** deliberately does **not** return full lyrics
  text in its search/song responses, publisher licensing means the API
  gives metadata and a link to the Genius webpage, not the lyrics
  themselves. Getting the actual text off the webpage means scraping, which
  Genius's terms prohibit. That rules it out for this project's "no
  ToS-violating pipeline" rule (the same rule audio matching applied to ruling out
  YouTube).
- **lrclib.net's API is public, keyless, and returns full lyrics text
  directly** (`plainLyrics` and, where available, line-synced
  `syncedLyrics`) specifically so client apps can display them. It's
  community-sourced rather than licensed from publishers, which is a real
  limitation, coverage skews toward well-known tracks, and there's no
  guarantee of accuracy, but the terms of using the API for this purpose
  are unambiguous, unlike scraping a page that says not to.

`GET https://lrclib.net/api/search?track_name=...&artist_name=...` returns
a list of candidate matches. This project takes the first result's
`plainLyrics` (or `null` if the field is absent or no result matches well
enough), never the synced/timed variant, since only the words matter here.

## How a null lyric is handled

A track with no lyrics match is a **valid input, not an error**. The
teacher prompt is written to work from metadata and measured audio features
alone when `lyrics` is `null`.

What a null means matters, though, and lrclib distinguishes more than the
first version of `enrich.py` did. `enrich.lyrics_status` now reports one of
three states:

| Status | Meaning | Cache |
|---|---|---|
| `lyrics` | lrclib returned lyrics | `{track_id}.txt` with the text |
| `instrumental` | lrclib flags the track as instrumental | empty `.txt` plus a `{track_id}.instrumental` marker |
| `unknown` | lrclib has no entry | empty `.txt` |

"Unknown" is not "instrumental". Plenty of vocal tracks are simply missing
from a community database, so an empty lookup is never treated as evidence
that a track has no words.

## A bug that cost 830 tracks their lyrics

The first full fetch (`scripts/fetch_all_lyrics.py`, 2026-09-23) caught
every `httpx.HTTPError`, including timeouts and server errors, and cached
the failure exactly like a genuine "no match". A request that failed once
was remembered as "this song has no lyrics" and never retried. Its hit rate
fell from 82% early in the run to 68% by 13,000 tracks. Some of that may be
track order, but the failures were silent, so the log couldn't say.

Found 2026-09-25, when Steely Dan's "Your Gold Teeth" turned up with an
empty cache entry and full lyrics on lrclib. The fix:

- failed requests retry with backoff (3 attempts) and, if they still fail,
  raise `LyricsFetchError` and cache nothing, so a rerun picks them up;
- lrclib's `instrumental` flag is recorded;
- `fetch_all_lyrics.py --recheck-empty` re-queries every empty entry.

The recheck over all 7,101 empty entries, with 0 failed requests:

| Outcome | Tracks | Share |
|---|---|---|
| Lyrics recovered | 830 | 11.7% |
| Confirmed instrumental | 1,168 | 16.4% |
| Still unknown | 5,103 | 71.9% |

The 830 recovered tracks were re-tagged (`infer.py --retag-ids`). The
tagger itself was not retrained: it saw `Lyrics: not available` for both
instrumental and unknown tracks during training, and it tends to describe
all of them as instrumentals in `lyrical_theme`. `track_features.parquet`
carries a `lyrics_status` column so consumers can tell which ones lrclib
actually confirmed. Feeding the status to the fly as two input bits was
tried and reverted: it cost the full-feature fly 0.052 AUC on unseen
tracks. The ablation is in `docs/MBON_EVAL.md`.

## Caching

Lyrics are cached to `data/lyrics/{track_id}.txt` (gitignored, this is
fetched content, not something to redistribute) so a re-run of the
enrichment or labelling step never re-fetches a track it already has.
