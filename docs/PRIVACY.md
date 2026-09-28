# Privacy

The project is public; the listening history behind it is not. This covers
what is published, what is gated, and the checks that keep it that way.

## What the public site shows

| Surface | Contents |
| --- | --- |
| `/` and `/watch` | Your own export, parsed in the browser tab. Nothing is uploaded. |
| `/rewind` | The year-in-listening story for an **invented listener**, never a real one |
| `web/public/fly/catalog.json.gz` | The library's track names and artists, no play counts or timestamps |
| `web/public/sample/sample_spotify_data.zip` | The invented listener: real catalog tracks, generated plays |

## Rewind: two profiles

`python -m selector.warehouse.wrapped --profile <name>` runs the same
pipeline for either audience. The profile sets the data root, the output
folder, whether model-written cluster names may appear, and the `audience`
marker stamped on every report.

| | `synthetic` | `private` |
| --- | --- | --- |
| Plays | `web/public/sample/sample_spotify_data.zip`, ingested into `data/sample/` | `data/plays.parquet` |
| Mushroom body (hidden gems) | trained on the sample's plays | trained on the real plays |
| Cluster names | built from features only | may use the cached model-written names |
| Output | `web/public/rewind/` (committed, public) | `data/wrapped_private/` (gitignored) |
| `audience` | `synthetic` | `private` |

The taste clusters themselves are fitted from fingerprints alone, with no
plays, so both profiles share one fit. The model-written names are not
shared: they were written from the real listener's most-played tracks.

## The private share link

The real reports are served only by `web/app/rewind/private/[id]/route.ts`,
and only to a request carrying the share token. The story page for them is
`/rewind/p/<token>`.

- **Token:** `REWIND_SHARE_TOKEN`, a Vercel environment variable of at least
  32 characters, compared in constant time. If it's unset or too short,
  nothing matches: the gate fails closed.
- **Wrong or missing token:** 404, not 401, so the route's existence isn't
  advertised. An unknown report id or a failed storage read is also a 404.
- **Storage:** a private Vercel Blob store, under `rewind-private/`. The
  reports never enter git, `web/public/` or the build output, and
  re-exporting doesn't need a redeploy.
- **Headers:** `Cache-Control: private, no-store` (the CDN never caches a
  report), `X-Robots-Tag: noindex, nofollow`, and `Referrer-Policy:
  no-referrer`, since the token is in the page URL. The page also sets
  `robots: noindex`.

### Setting it up

```
vercel blob create-store selector-rewind --access private   # once; connect it to the web project
vercel env add REWIND_SHARE_TOKEN production                 # a long random secret
uv run python -m selector.warehouse.wrapped --profile private
cd web && vercel env pull .env.local
node --env-file=.env.local scripts/upload-private-rewind.mjs
```

The upload script refuses any file not marked `audience: private`.

### Rotation

Changing `REWIND_SHARE_TOKEN` (and redeploying, so functions pick it up)
kills every old link at once. Rotate whenever a link has gone further than
intended.

## The hosted MCP server

`src/selector/mcp/http_server.py` requires `SELECTOR_MCP_TOKEN`. With it
unset, every request gets 503. Local testing without a token needs the
explicit `SELECTOR_MCP_ALLOW_NO_AUTH=1`, which is never set on Vercel.

- **Data:** it ships `data/selector_deploy.duckdb`
  (`python -m selector.warehouse.build --deploy`), not the full warehouse.
  Plays are reduced to counts per UTC day, hour and track. There are no
  exact timestamps, sessions, platforms, countries or reason codes, and the
  build refuses to write a file with any time column left.
- **Tools:** only the nine read-only warehouse tools. The live Spotify,
  fly-brain, DJ and Rewind tools are local-only.
- **Where it lives:** in a private Blob store connected only to the MCP
  project, downloaded into `/tmp` on cold start. It's never in git or a
  deployment. `.vercelignore` is an allowlist (`api/`, `src/`,
  `requirements.txt`, `vercel.json`), so env files, private reports and
  `data/` never reach Vercel. The upload refuses any file with per-play
  tables or time columns.
- **Token rotation:** `scripts/rotate_mcp_token.py` sets the new value
  without ever printing it.

The hour buckets still show roughly what was played when. The token is the
only thing keeping that private. See [DEPLOY_MCP.md](DEPLOY_MCP.md).

## Guardrails

`scripts/check_personal_data.py` runs in pre-commit on staged files and in CI
on every tracked file. On top of the export, warehouse, audio and secret
rules, it blocks:

- any JSON under `web/public/rewind/` not marked `audience: synthetic`, and
  an `index.json` without its `privacy` note;
- any file marked `audience: private`, wherever it sits;
- a public asset with hourly counts on a real clock (`America/Chicago`)
  that isn't marked synthetic;
- lyrics text anywhere under `web/`, including inside `.gz` assets.

CI also fails if public copy (`web/app`, `web/components`, the README) uses
"Wrapped", Spotify's product name, or if a built page lacks the
not-affiliated disclaimer.

## Spotify terms

Selector is non-commercial and not affiliated with or endorsed by Spotify.
The public site uses no Spotify logo, wordmark or brand colour, plays no
audio, and ships only derived numbers.

## Training boundary

Spotify's [Developer Policy](https://developer.spotify.com/policy) says not
to use the Spotify Platform or Spotify Content to train a machine-learning
model. Nothing here trains on Web API responses. This was checked by
tracing every input, not assumed (audit of 2026-09-27):

| Model or learned artifact | Trained or built from |
| --- | --- |
| Measured audio features (`selector.audio`) | 30-second preview clips matched on the iTunes Search and Deezer public APIs, by track and artist name |
| Teacher labels (`selector.tagger.label`) | Track, artist and album names from the warehouse, lrclib lyrics, and the measured features. The schema's `artist_genres` and `release_year` fields are never filled (0 of 3,512 label records). |
| Vibe tagger fine-tune (`selector.tagger.train`) | Those teacher labels |
| Fly fingerprints (`selector.fly.pipeline`) | Tagger outputs, measured features and the FlyWire connectome (Zenodo, GitHub) |
| Mushroom body (`selector.fly.mbon`) | Play and skip verdicts from `data/plays.parquet` |
| Taste clusters and names (`selector.fly.clusters`, `cluster_names`) | The fingerprints, plus most-played track names from the warehouse |

The warehouse and `data/plays.parquet` come only from the GDPR data export
(`selector.ingest.load_history`), which Spotify gives the account holder as
their own personal data. It is not fetched from the Web API.

`tests/test_ml_boundary.py` enforces the code side. It walks the import
graph of `selector.tagger`, `selector.fly`, `selector.audio` and
`selector.ingest`, including imports made inside functions, and fails if
any of them can reach `selector.spotify`.

**Inference is the gray area.** The local-only `spotify_*` MCP tools hand
live Web API results to Claude in a conversation. The DJ agent may read the
live recently-played list to pick a theme, which is deterministic scoring,
not a model, and nothing it reads is stored as training data. None of this
runs on the hosted MCP server.

This is a risk reading, not legal advice.

## Preview clips

The 30-second clips in `data/audio/` exist only to measure features. They
are gitignored, blocked by the guard's audio-suffix rule, excluded from the
Vercel upload, and never played on the site. Apple's and Deezer's preview
terms are written for in-app playback, not bulk download, so the plan is to
delete them. That waits until the deferred authenticity features
(acoustic-vs-electronic, timing looseness) have been extracted, because
that needs the clips. After that, the measured features in parquet are all
the project needs.
