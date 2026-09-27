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
explicit `SELECTOR_MCP_ALLOW_NO_AUTH=1`, which is never set on Vercel. See
[DEPLOY_MCP.md](DEPLOY_MCP.md).

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
