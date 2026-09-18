# Lyrics source for the teacher labelling pipeline

Step 10 needs lyrics as one input to the teacher (alongside metadata and
measured audio features). This needed a source whose terms actually permit
the use — feeding the text into an LLM call, not republishing it — rather
than the first API that returns text.

## Source: lrclib.net

[lrclib.net](https://lrclib.net) is a free, open, community-contributed
lyrics database built primarily to serve synced lyrics to music player
apps. It was chosen over the obvious alternative (Genius) for a concrete
reason:

- **Genius's official API** deliberately does **not** return full lyrics
  text in its search/song responses — publisher licensing means the API
  gives metadata and a link to the Genius webpage, not the lyrics
  themselves. Getting the actual text off the webpage means scraping, which
  Genius's terms prohibit. That rules it out for this project's "no
  ToS-violating pipeline" rule (the same rule Step 8 applied to ruling out
  YouTube).
- **lrclib.net's API is public, keyless, and returns full lyrics text
  directly** (`plainLyrics` and, where available, line-synced
  `syncedLyrics`) specifically so client apps can display them. It's
  community-sourced rather than licensed from publishers, which is a real
  limitation — coverage skews toward well-known tracks, and there's no
  guarantee of accuracy — but the terms of using the API for this purpose
  are unambiguous, unlike scraping a page that says not to.

`GET https://lrclib.net/api/search?track_name=...&artist_name=...` returns
a list of candidate matches; this project takes the first result's
`plainLyrics` (or `null` if the field is absent or no result matches well
enough), never the synced/timed variant, since only the words matter here.

## How a null lyric is handled

A track with no lyrics match is a **valid input, not an error** — the
teacher prompt is written to work from metadata and measured audio features
alone when `lyrics` is `null` (true for instrumentals as well as tracks
lrclib simply doesn't have). `enrich.py` records this explicitly rather than
retrying indefinitely or substituting placeholder text.

## Caching

Lyrics are cached to `data/lyrics/{track_id}.txt` (gitignored — this is
fetched content, not something to redistribute) so a re-run of the
enrichment or labelling step never re-fetches a track it already has.
