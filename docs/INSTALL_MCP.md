# Installing the Selector MCP server

Selector exposes the local taste warehouse (built in Phase 0) as an MCP
server over stdio. **Verified working in Claude Code**. Claude Desktop is
currently a known issue, see the section below before spending time on it.

## Prerequisites

Build the warehouse first, if you haven't already:

```
uv run python -m selector.ingest.load_history
uv run python -m selector.warehouse.build
```

This produces `data/selector.duckdb`. The MCP server refuses to run queries
against a missing warehouse, every tool call returns a message telling you
to run the two commands above, instead of a stack trace.

## Configure Claude Code (verified working)

From the repo root:

```
claude mcp add selector --scope project -- uv --directory "$PWD" run selector-mcp
```

This writes `.mcp.json` at the repo root. It holds an absolute path to your
clone, so it's gitignored. Project-scoped servers from
`.mcp.json` need a one-time approval, run `claude` in the repo and approve
`selector` when prompted (`claude mcp list` shows it as "⏸ Pending approval"
until then). After approval, ask something like "what did I binge in March
and then abandon?" in a session started in this directory.

## Configure Claude Desktop (known issue, not currently working)

The classic approach, a `claude_desktop_config.json` with an `mcpServers`
block, **did not work** when tried against the installed app (version
2.110.0, a Microsoft Store/MSIX package). Root cause, worked out by
inspecting the app's actual data directory:

- Windows silently redirects an MSIX-packaged app's `%APPDATA%` reads and
  writes to an isolated per-package folder, in this case
  `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\`,
  not the real `%APPDATA%\Claude\`. A config file written to the classic
  path is invisible to the app. It never even opens it.
- The `claude_desktop_config.json` that actually exists at the redirected
  path is not an MCP-servers file at all in this app version, it holds
  window layout, account/OAuth cache, and feature flags, with no
  `mcpServers` key anywhere. This app generation (it bundles Claude Code
  itself, the same data directory has `claude-code-sessions`,
  `git-worktrees.json`, etc.) appears to manage MCP server registration
  through a different mechanism than the flat JSON file documented for
  older Claude Desktop releases, most likely a Settings UI (look for
  "Connectors" or "Extensions") or a DXT extension package, and possibly the
  same project-scoped `.mcp.json` mechanism as Claude Code, given the two
  are clearly the same underlying product now.
- This wasn't chased further because the actual data format for this app
  version isn't something to guess at from outside, misconfiguring
  account-level state (OAuth token cache, etc.) in an undocumented file is a
  worse outcome than leaving this unresolved and documented.

**Status: unresolved.** If revisiting this, start by opening the Selector
project as a workspace inside the Claude app itself (not a plain chat),
since it bundles Claude Code, it may just pick up the same project-scoped
`.mcp.json` used above, with the same approval step. If that doesn't surface
it, check the app's Settings for a connectors/extensions panel before
touching any file in
`%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\`
directly.

## Shared details

By default the server reads `./data/selector.duckdb`, resolved relative to
the `--directory` passed to `uv`. To point it at a different warehouse file,
set `SELECTOR_DB` to an absolute path (an `-e SELECTOR_DB=...` flag on
`claude mcp add`, or an `"env"` block in a hand-edited config).

To also enable the live Spotify tools (search, saved tracks, top
artists/tracks, recently played, playlist creation, library reconciliation),
put `SPOTIFY_CLIENT_ID` in `.env` per `docs/OAUTH_NOTES.md`. Without it,
those tools just return a message explaining what to set up instead of
failing, the warehouse tools below work either way.

## What's exposed

**Warehouse tools** (read `data/selector.duckdb`, built in Phase 0), one per
query plus `warehouse_summary` as an orientation call: `warehouse_summary`,
`search_library`, `track_detail`, `top_artists`, `binged_then_abandoned`,
`skip_offenders`, `listening_clock`, `taste_drift`, `rediscovery_candidates`.

**Live Spotify tools** (need `SPOTIFY_CLIENT_ID`, see `docs/OAUTH_NOTES.md`):
`spotify_search`, `spotify_saved_tracks`, `spotify_top_artists`,
`spotify_top_tracks`, `spotify_recently_played`, `spotify_create_playlist`,
`reconcile_library`. All but `reconcile_library` are also on the hosted
server, with its own Spotify login (see `docs/DEPLOY_MCP.md`). After a
deploy that changes the hosted tool list, reconnect the claude.ai connector
so it picks up the new list. The hosted server logs in with OAuth through
Spotify: add it as a custom connector with just the URL
(`https://selector-mcp.vercel.app/mcp`), no token, and claude.ai walks
through the login. In Claude Code, `claude mcp add --transport http
selector-remote https://selector-mcp.vercel.app/mcp`, then `/mcp` to log
in.

**Fly brain tools** (read `data/fly_tags.npz`, built by Step 12's
`uv run python -m selector.fly.pipeline`): `more_like_this(track, k)`,
nearest tracks by Hamming distance over fly-brain fingerprints, i.e. content
similarity, and `fly_score(track)`, the mushroom body's predicted
approach/avoid valence for a track, trained on this person's actual
skip/play-out history. Both accept a `track_id` or a name substring.

**DJ agent** (needs the fly tags plus `data/audio_features.parquet`):
`dj_set(theme=None, minutes=45, dry_run=True, familiar_ratio=0.6)` plans a
themed set through Brief → Arc → Select → Critique → Commit and returns the
critique chain plus liner notes. It is a dry run unless `dry_run=False` is
passed explicitly, and it never writes a set that failed critique. See
`docs/DJ_AGENT.md`.

Every tool returns a markdown table (or a short status message for the
write-actions), capped at 40 rows with a "… N more rows" note when
truncated.

## Verified transcript

Asked in Claude Code, with the `selector` server approved and running,
against a real 46,202-play warehouse:

> **Me:** What did I binge in March and then abandon?
>
> **Claude:** I checked your listening history for artists with a sharp
> spike in some March, followed by almost no plays for the next three
> months. Here's what stands out:
>
> | Artist | March binged | Plays since |
> | --- | --- | --- |
> | flipturn | 80 plays (Mar 2024) | 1 |
> | Rebelution | 50 plays (Mar 2026) | 1 |
> | kidstrange | 31 plays (Mar 2026) | 0 |
> | Old Mervs | 24 plays (Mar 2025) | 0 |
> | aku.mu | 20 plays (Mar 2026) | 0 |
> | The Velvet Underground | 20 plays (Mar 2024) | 0 |
> | Michael Jackson | 17 plays (Mar 2023) | 0 |
> | ROSALÍA | 17 plays (Mar 2026) | 0 |
> | Del Water Gap | 15 plays (Mar 2024) | 0 |
>
> flipturn is the sharpest case, 80 plays in March 2024, then almost
> nothing afterward. A few of these (kidstrange, Old Mervs, aku.mu,
> ROSALÍA) are from March 2026, so "abandoned" there just means the
> follow-up window hasn't fully played out yet.

This came straight from the `binged_then_abandoned` tool with its default
thresholds (`min_plays=15`, `window_months=3`), filtered by the model to
March spikes in its own reply, the tool itself doesn't take a month
parameter, so the filtering step is a good demonstration of the model
reasoning over structured data rather than just relaying it.
