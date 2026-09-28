// A visitor's own Rewind, built in the tab from their parsed export. A port of
// `selector.warehouse.wrapped.build_report` (the card functions, windows and
// archetype rules) over the `History` the parse worker sends, so the story UI
// renders it exactly like the precomputed reports. Nothing here is uploaded.
//
// Only the fly-brain cards depend on our data, and only on catalog metadata
// (public/fly/rewind.json.gz): tracks the catalog has no fingerprint for can't
// be clustered or ranked, and albums it has no release year for can't be
// dated, so those cards say how much of the export they could see.
// No model calls: cluster names are the feature-built ones.

import { type Catalog, fetchBytes } from "./catalog";
import { matchHistory, type MushroomBody, percentileRank, trainMushroomBody } from "./fly";
import type { AnyCard, Coverage, LocalReports, Report, ReportIndex } from "./rewind";
import type { History } from "./types";
import { strFromU8 } from "fflate";

const SCHEMA_VERSION = "2.1";

// Same constants as `selector.warehouse.wrapped` and `queries`.
const TOP_N = 5;
const OLD_RELEASE_YEARS = 10;
const GEM_MAX_PLAYS = 2;
const GEM_COUNT = 5;
const SKIP_OFFENDER_MIN_PLAYS = 5;
const SKIP_OFFENDER_MIN_RATE = 0.3;
const ARCHETYPE_OLD_YEARS = 20;
const ARCHETYPE_MIN_MONTH_PLAYS = 100;

type Rule = { id: string; label: string; metric: string; dir: ">=" | "<="; threshold: number; line: (v: string) => string };
const ARCHETYPES: Rule[] = [
  { id: "explorer", label: "The Explorer", metric: "discovery_ratio", dir: ">=", threshold: 0.4, line: (v) => `${v} of your plays were first listens (Explorer: 40%+)` },
  { id: "loyalist", label: "The Loyalist", metric: "top1_share", dir: ">=", threshold: 0.25, line: (v) => `your top 1% of tracks take ${v} of your plays (Loyalist: 25%+)` },
  { id: "time_traveller", label: "The Time Traveller", metric: "old_share", dir: ">=", threshold: 0.5, line: (v) => `${v} of your dated plays are from albums 20+ years old (Time Traveller: 50%+)` },
  { id: "specialist", label: "The Specialist", metric: "cluster_evenness", dir: "<=", threshold: 0.7, line: (v) => `your plays spread across taste clusters at ${v} evenness (Specialist: 70% or less)` },
  { id: "restless", label: "The Restless", metric: "skip_rate", dir: ">=", threshold: 0.25, line: (v) => `${v} of your plays end in a skip (Restless: 25%+)` },
];

const CARD_ORDER = [
  "total_hours",
  "top_artists",
  "top_tracks",
  "top_albums",
  "artist_sprint",
  "time_of_day",
  "skip_offenders",
  "listening_age",
  "decade_histogram",
  "taste_clusters",
  "hidden_gems",
  "archetype",
];

const pct = (x: number) => `${Math.round(100 * x)}%`;
const n = (x: number) => Math.round(x).toLocaleString("en-US");
const clockLabel = (h: number) => `${h % 12 || 12} ${h < 12 ? "AM" : "PM"}`;

// -- assets -----------------------------------------------------------------

export type RewindAssets = {
  clusterKey: string;
  medoidRows: number[]; // catalog rows
  builtNames: string[];
  clusterSize: number[]; // library-wide member count
  clusterMeasured: number[];
  unmeasured: Set<number>; // catalog rows without measured audio
  releaseYears: Map<string, number>; // normalised album key -> year
};

type RewindJson = {
  cluster_key: string;
  clusters: { medoid: string; built_name: string; track_count: number; measured_share: number }[];
  unmeasured_rows: number[];
  release_years: Record<string, number>;
};

export async function loadRewindAssets(cat: Catalog, base = "/fly"): Promise<RewindAssets> {
  const json = JSON.parse(strFromU8(await fetchBytes(`${base}/rewind.json.gz`))) as RewindJson;
  const medoidRows = json.clusters.map((c) => {
    const row = cat.row.get(c.medoid);
    if (row === undefined) throw new Error("rewind assets don't match the fly catalog");
    return row;
  });
  return {
    clusterKey: json.cluster_key,
    medoidRows,
    builtNames: json.clusters.map((c) => c.built_name),
    clusterSize: json.clusters.map((c) => c.track_count),
    clusterMeasured: json.clusters.map((c) => c.measured_share),
    unmeasured: new Set(json.unmeasured_rows),
    releaseYears: new Map(Object.entries(json.release_years)),
  };
}

// `selector.warehouse.queries.normalize_album_key`.
const EDITION_SUFFIX = /\s*[([](deluxe|remaster(ed)?|expanded|bonus track|anniversary|special)[^)\]]*[)\]]\s*/gi;
export function normalizeAlbumKey(artist: string, album: string): string {
  return `${artist}|${album}`.toLowerCase().trim().replace(EDITION_SUFFIX, " ").replace(/\s+/g, " ").trim();
}

// -- local clock ------------------------------------------------------------

type LocalTime = { year: number; month: string; date: string; hour: number };

// Intl formatting is slow per call; cache by quarter hour, which is fine
// for every zone offset in use.
function localClock(timeZone: string): (t: number) => LocalTime {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    hourCycle: "h23",
  });
  const cache = new Map<number, LocalTime>();
  return (t) => {
    const bucket = Math.floor(t / 900_000);
    let out = cache.get(bucket);
    if (!out) {
      const p: Record<string, string> = {};
      for (const part of fmt.formatToParts(new Date(bucket * 900_000))) p[part.type] = part.value;
      out = {
        year: Number(p.year),
        month: `${p.year}-${p.month}`,
        date: `${p.year}-${p.month}-${p.day}`,
        hour: Number(p.hour) % 24,
      };
      cache.set(bucket, out);
    }
    return out;
  };
}

// "America/Chicago" -> "Central", as the author's reports say.
function zoneLabel(timeZone: string): string {
  try {
    const name = new Intl.DateTimeFormat("en-US", { timeZone, timeZoneName: "longGeneric" })
      .formatToParts(new Date())
      .find((p) => p.type === "timeZoneName")?.value;
    if (name && / Time$/.test(name)) return name.replace(/ Time$/, "");
  } catch {
    // Older engines lack longGeneric; fall through.
  }
  return (timeZone.split("/").pop() ?? timeZone).replace(/_/g, " ");
}

// -- per-window aggregates ----------------------------------------------------

type TrackAgg = { t: number; plays: number; fwd: number };

type Ctx = {
  h: History;
  from: number; // seq index range [from, to)
  to: number;
  local: LocalTime[]; // per seq index
  tracks: Map<number, TrackAgg>; // window plays per history track
  totalMs: number;
};

function windowCtx(h: History, local: LocalTime[], from: number, to: number): Ctx {
  const tracks = new Map<number, TrackAgg>();
  let totalMs = 0;
  for (let i = from; i < to; i++) {
    const t = h.seqTrack[i];
    let agg = tracks.get(t);
    if (!agg) tracks.set(t, (agg = { t, plays: 0, fwd: 0 }));
    agg.plays++;
    agg.fwd += h.seqFwd[i];
    totalMs += h.seqMs[i];
  }
  return { h, from, to, local, tracks, totalMs };
}

// Descending by count; ties keep first-seen order, as a stable sort does.
function ranked<K>(counts: Map<K, number>): [K, number][] {
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function cardTotalHours(ctx: Ctx, totals: Report["totals"]): AnyCard | null {
  if (!totals.hours) return null;
  return {
    id: "total_hours",
    headline: `You listened for ${n(totals.hours)} hours`,
    value: totals.hours,
    sublabel: `across ${n(totals.plays)} plays`,
    evidence: [`${n(totals.tracks)} unique tracks`, `${n(totals.artists)} unique artists`],
  };
}

function cardTopArtists(ctx: Ctx): AnyCard | null {
  const counts = new Map<string, number>();
  for (const a of ctx.tracks.values()) {
    const artist = ctx.h.artists[a.t];
    counts.set(artist, (counts.get(artist) ?? 0) + a.plays);
  }
  const top = ranked(counts).slice(0, TOP_N);
  if (!top.length) return null;
  return {
    id: "top_artists",
    headline: `Your #1 artist was ${top[0][0]}`,
    value: top.map(([artist, play_count]) => ({ artist, play_count })),
    sublabel: `${n(top[0][1])} plays`,
    evidence: top.slice(0, 3).map(([a, c], i) => `#${i + 1} ${a} (${n(c)} plays)`),
  };
}

function cardTopTracks(ctx: Ctx): AnyCard | null {
  const top = [...ctx.tracks.values()].sort((a, b) => b.plays - a.plays).slice(0, TOP_N);
  if (!top.length) return null;
  const { names, artists } = ctx.h;
  return {
    id: "top_tracks",
    headline: `Your #1 track was "${names[top[0].t]}" by ${artists[top[0].t]}`,
    value: top.map((r) => ({ name: names[r.t], artist: artists[r.t], play_count: r.plays })),
    sublabel: `${n(top[0].plays)} plays`,
    evidence: top.slice(0, 3).map((r, i) => `#${i + 1} "${names[r.t]}" — ${artists[r.t]} (${n(r.plays)} plays)`),
  };
}

function cardTopAlbums(ctx: Ctx): AnyCard | null {
  const groups = new Map<string, { album: string; artist: string; plays: number }>();
  for (const a of ctx.tracks.values()) {
    const album = ctx.h.albums[a.t];
    if (!album) continue;
    const key = normalizeAlbumKey(ctx.h.artists[a.t], album);
    const g = groups.get(key);
    if (g) g.plays += a.plays;
    else groups.set(key, { album, artist: ctx.h.artists[a.t], plays: a.plays });
  }
  const top = [...groups.values()].sort((a, b) => b.plays - a.plays).slice(0, TOP_N);
  if (!top.length) return null;
  return {
    id: "top_albums",
    headline: `Your #1 album was "${top[0].album}" by ${top[0].artist}`,
    value: top.map((r) => ({ album: r.album, artist: r.artist, play_count: r.plays })),
    sublabel: `${n(top[0].plays)} plays`,
    evidence: top.slice(0, 3).map((r, i) => `#${i + 1} "${r.album}" — ${r.artist} (${n(r.plays)} plays)`),
  };
}

// `queries.artist_sprint` + `_card_artist_sprint`: every artist who was ever
// in a month's top five, with their full cumulative monthly counts.
function cardArtistSprint(ctx: Ctx): AnyCard | null {
  const byMonth = new Map<string, Map<string, number>>();
  for (let i = ctx.from; i < ctx.to; i++) {
    const m = ctx.local[i].month;
    let counts = byMonth.get(m);
    if (!counts) byMonth.set(m, (counts = new Map()));
    const artist = ctx.h.artists[ctx.h.seqTrack[i]];
    counts.set(artist, (counts.get(artist) ?? 0) + 1);
  }
  if (!byMonth.size) return null;
  const racers = new Set<string>();
  for (const counts of byMonth.values()) {
    // Ties rank alphabetically, as the SQL's ORDER BY artist leaves them.
    const order = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    for (const [a] of order.slice(0, TOP_N)) racers.add(a);
  }
  const played = [...byMonth.keys()].filter((m) => [...racers].some((a) => byMonth.get(m)!.has(a))).sort();
  const months: string[] = [];
  let [y, mo] = played[0].split("-").map(Number);
  const [ly, lm] = played[played.length - 1].split("-").map(Number);
  while (y < ly || (y === ly && mo <= lm)) {
    months.push(`${y}-${String(mo).padStart(2, "0")}`);
    if (++mo > 12) [y, mo] = [y + 1, 1];
  }
  const totals = new Map<string, number>();
  const running = new Map<string, number>();
  const rows = months.map((m) => {
    const counts = byMonth.get(m);
    for (const a of racers) running.set(a, (running.get(a) ?? 0) + (counts?.get(a) ?? 0));
    return new Map(running);
  });
  for (const a of racers) totals.set(a, running.get(a) ?? 0);
  const artists = ranked(totals).map(([a]) => a);
  return {
    id: "artist_sprint",
    headline: `${artists[0]} won your artist race`,
    value: { months, artists, cumulative: rows.map((r) => artists.map((a) => r.get(a) ?? 0)) },
    sublabel: `${artists.length} artists ever cracked a monthly top spot`,
    evidence: artists.slice(0, 3).map((a) => `${a}: ${n(totals.get(a)!)} plays total`),
  };
}

function cardTimeOfDay(ctx: Ctx, timeZone: string): AnyCard | null {
  if (ctx.to <= ctx.from) return null;
  const byHour = new Array(24).fill(0);
  for (let i = ctx.from; i < ctx.to; i++) byHour[ctx.local[i].hour]++;
  const order = byHour.map((c, h) => [h, c]).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const label = zoneLabel(timeZone);
  const peak = order[0][0];
  return {
    id: "time_of_day",
    headline: `You listen most around ${clockLabel(peak)} ${label}`,
    value: { timezone: timeZone, timezone_label: label, peak_hour: peak, by_hour: byHour },
    sublabel: `${n(order[0][1])} plays in that hour`,
    evidence: order.slice(0, 3).map(([h, c]) => `${clockLabel(h)} ${label} — ${n(c)} plays`),
  };
}

function cardSkipOffenders(ctx: Ctx): AnyCard | null {
  const rows = [...ctx.tracks.values()]
    .map((a) => ({ a, rate: a.fwd / a.plays }))
    .filter((r) => r.a.plays >= SKIP_OFFENDER_MIN_PLAYS && r.rate >= SKIP_OFFENDER_MIN_RATE)
    .sort((x, y) => y.rate - x.rate || y.a.plays - x.a.plays);
  if (!rows.length) return null;
  const { names, artists } = ctx.h;
  const worst = rows[0];
  return {
    id: "skip_offenders",
    headline: `You keep queuing up and bailing on "${names[worst.a.t]}"`,
    value: rows.map((r) => ({
      name: names[r.a.t],
      artist: artists[r.a.t],
      play_count: r.a.plays,
      skip_rate: Math.round(r.rate * 1000) / 1000,
    })),
    sublabel: `${pct(worst.rate)} skip rate over ${worst.a.plays} plays`,
    evidence: rows.slice(0, 3).map((r) => `"${names[r.a.t]}" — ${artists[r.a.t]} (${pct(r.rate)} skipped)`),
  };
}

// -- release years (tier B) ----------------------------------------------------

type Dated = { t: number; year: number; plays: number };

function datedTracks(ctx: Ctx, assets: RewindAssets): Dated[] {
  const out: Dated[] = [];
  for (const a of ctx.tracks.values()) {
    const album = ctx.h.albums[a.t];
    if (!album) continue;
    const year = assets.releaseYears.get(normalizeAlbumKey(ctx.h.artists[a.t], album));
    if (year !== undefined) out.push({ t: a.t, year, plays: a.plays });
  }
  return out;
}

function tierB(ctx: Ctx, dated: Dated[]): Coverage {
  return {
    tier: "B",
    tracks_used: dated.length,
    of: ctx.tracks.size,
    plays_used: dated.reduce((s, d) => s + d.plays, 0),
    plays_of: ctx.to - ctx.from,
  };
}

function cardListeningAge(ctx: Ctx, dated: Dated[], windowTo: string | null): AnyCard | null {
  if (!dated.length) return null;
  // Play-weighted median, as `np.median` over each year repeated per play.
  const byYear = [...dated].sort((a, b) => a.year - b.year);
  const total = byYear.reduce((s, d) => s + d.plays, 0);
  const at = (k: number) => {
    let seen = 0;
    for (const d of byYear) if ((seen += d.plays) > k) return d.year;
    return byYear[byYear.length - 1].year;
  };
  const median = total % 2 ? at((total - 1) / 2) : Math.floor((at(total / 2 - 1) + at(total / 2)) / 2);

  const endYear = windowTo ? Number(windowTo.slice(0, 4)) : byYear[byYear.length - 1].year;
  const cutoff = endYear - OLD_RELEASE_YEARS;
  const old = dated.filter((d) => d.year < cutoff).reduce((s, d) => s + d.plays, 0);
  const oldest = [...dated].sort((a, b) => a.year - b.year || b.plays - a.plays)[0];
  const coverage = tierB(ctx, dated);
  return {
    id: "listening_age",
    headline: `You listen like it's ${median}`,
    value: median,
    sublabel: "median album release year, play-weighted",
    coverage,
    evidence: [
      `${pct(old / total)} of your dated plays are from before ${cutoff}`,
      `oldest: ${ctx.h.artists[oldest.t]}, "${ctx.h.albums[oldest.t]}" (${oldest.year})`,
      `dated from ${n(coverage.plays_used!)} of ${n(coverage.plays_of!)} plays (${n(coverage.tracks_used)} tracks)`,
    ],
  };
}

function cardDecadeHistogram(ctx: Ctx, dated: Dated[]): AnyCard | null {
  if (!dated.length) return null;
  const counts = new Map<number, number>();
  for (const d of dated) {
    const decade = Math.floor(d.year / 10) * 10;
    counts.set(decade, (counts.get(decade) ?? 0) + d.plays);
  }
  const total = [...counts.values()].reduce((s, c) => s + c, 0);
  const rows = [...counts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([decade, play_count]) => ({ decade, play_count, share: Math.round((1000 * play_count) / total) / 1000 }));
  const top = [...rows].sort((a, b) => b.play_count - a.play_count);
  return {
    id: "decade_histogram",
    headline: `The ${top[0].decade}s own ${pct(top[0].play_count / total)} of your plays`,
    value: rows,
    sublabel: `${rows.length} decades represented`,
    coverage: tierB(ctx, dated),
    evidence: top.slice(0, 3).map((r) => `${r.decade}s — ${pct(r.play_count / total)} (${n(r.play_count)} plays)`),
  };
}

// -- fly brain (tier A) -------------------------------------------------------

type Fly = {
  catalogRow: Int32Array; // history track -> catalog row, -1 if unknown
  cluster: Int32Array; // history track -> cluster, -1 if unknown
  score: Float64Array; // per catalog row: approach minus avoid drive
  rank: Int32Array; // per catalog row: 1 = the fly's favourite, ties share the lowest rank
  percentile: Float64Array;
};

function flyBase(cat: Catalog, h: History, assets: RewindAssets, mb: MushroomBody): Fly {
  const { catalogRow } = matchHistory(cat, h);
  const words = cat.words;
  const cluster = new Int32Array(h.trackIds.length).fill(-1);
  catalogRow.forEach((row, t) => {
    if (row < 0) return;
    // Nearest medoid by Hamming distance, first on a tie, which the asset
    // builder checked reproduces the fitted labels.
    let best = -1;
    let bestShared = -1;
    assets.medoidRows.forEach((m, c) => {
      let shared = 0;
      for (let w = 0; w < words; w++) {
        let x = cat.bits[row * words + w] & cat.bits[m * words + w];
        x -= (x >>> 1) & 0x55555555;
        x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
        shared += (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
      }
      if (shared > bestShared) [best, bestShared] = [c, shared];
    });
    cluster[t] = best;
  });

  const score = new Float64Array(cat.size);
  for (let r = 0; r < cat.size; r++) score[r] = mb.valence(cat, r);
  const sorted = Float64Array.from(score).sort().reverse();
  const rank = new Int32Array(cat.size);
  for (let r = 0; r < cat.size; r++) {
    // 1 + how many rows score strictly higher (pandas rank method="min").
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] > score[r]) lo = mid + 1;
      else hi = mid;
    }
    rank[r] = lo + 1;
  }
  return { catalogRow, cluster, score, rank, percentile: percentileRank(score) };
}

function tierA(ctx: Ctx, fly: Fly, assets: RewindAssets): Coverage {
  let used = 0;
  let measured = 0;
  for (const a of ctx.tracks.values()) {
    const row = fly.catalogRow[a.t];
    if (row < 0) continue;
    used++;
    if (!assets.unmeasured.has(row)) measured++;
  }
  return { tier: "A", tracks_used: used, of: ctx.tracks.size, measured };
}

type ClusterRow = {
  c: number;
  name: string;
  built_name: string;
  plays: number;
  play_share: number;
  track_count: number;
  measured_share: number;
  top_artists: string[];
};

// `clusters.cluster_summary` over the window's catalog plays, biggest first.
function clusterSummary(ctx: Ctx, fly: Fly, assets: RewindAssets): ClusterRow[] {
  const k = assets.medoidRows.length;
  const plays = new Array(k).fill(0);
  const artistPlays = Array.from({ length: k }, () => new Map<string, number>());
  for (const a of ctx.tracks.values()) {
    const c = fly.cluster[a.t];
    if (c < 0) continue;
    plays[c] += a.plays;
    const artist = ctx.h.artists[a.t];
    artistPlays[c].set(artist, (artistPlays[c].get(artist) ?? 0) + a.plays);
  }
  const total = plays.reduce((s, p) => s + p, 0);
  const rows: ClusterRow[] = plays.map((p, c) => ({
    c,
    name: assets.builtNames[c],
    built_name: assets.builtNames[c],
    plays: p,
    play_share: total ? p / total : 0,
    track_count: assets.clusterSize[c],
    measured_share: assets.clusterMeasured[c],
    top_artists: ranked(artistPlays[c]).slice(0, 3).map(([a]) => a),
  }));
  // Two clusters can share a built name; tell them apart by top artist.
  const seen = new Map<string, number>();
  for (const r of rows) seen.set(r.built_name, (seen.get(r.built_name) ?? 0) + 1);
  for (const r of rows) {
    if (seen.get(r.built_name)! > 1 && r.top_artists.length) r.name = r.built_name = `${r.built_name} (${r.top_artists[0]})`;
  }
  return rows.sort((a, b) => b.play_share - a.play_share);
}

function cardTasteClusters(ctx: Ctx, fly: Fly, assets: RewindAssets, summary: ClusterRow[]): AnyCard | null {
  const top = summary[0];
  if (!top?.plays) return null;
  return {
    id: "taste_clusters",
    headline: `Your biggest vibe: ${top.name}`,
    value: summary.map((r) => ({
      name: r.name,
      built_name: r.built_name,
      play_share: Math.round(1000 * r.play_share) / 1000,
      track_count: r.track_count,
      measured_share: r.measured_share,
      top_artists: r.top_artists,
    })),
    sublabel: `${summary.length} taste clusters in your fly-brain fingerprints; this one holds ${pct(top.play_share)} of the plays the fly could see`,
    coverage: tierA(ctx, fly, assets),
    evidence: summary
      .filter((r) => r.plays > 0)
      .slice(0, 3)
      .map((r) => `${r.name}: ${pct(r.play_share)} of plays (${r.top_artists.slice(0, 2).join(", ")})`),
  };
}

function cardHiddenGems(ctx: Ctx, fly: Fly, assets: RewindAssets, summary: ClusterRow[], catSize: number): AnyCard | null {
  const names = new Map(summary.map((r) => [r.c, r.name]));
  const { h } = ctx;
  const candidates = [...ctx.tracks.values()]
    .filter((a) => fly.catalogRow[a.t] >= 0 && a.plays >= 1 && a.plays <= GEM_MAX_PLAYS && a.fwd === 0)
    .sort((x, y) => {
      const d = fly.score[fly.catalogRow[y.t]] - fly.score[fly.catalogRow[x.t]];
      return d || (h.trackIds[x.t] < h.trackIds[y.t] ? -1 : 1);
    });
  const gems: TrackAgg[] = [];
  const artists = new Set<string>();
  for (const a of candidates) {
    if (artists.has(h.artists[a.t])) continue;
    artists.add(h.artists[a.t]);
    gems.push(a);
    if (gems.length === GEM_COUNT) break;
  }
  if (!gems.length) return null;
  const top = gems[0];
  const rank = (a: TrackAgg) => fly.rank[fly.catalogRow[a.t]];
  const clusterName = (a: TrackAgg) => names.get(fly.cluster[a.t]) ?? null;
  return {
    id: "hidden_gems",
    headline: `Hidden gem: "${h.names[top.t]}" by ${h.artists[top.t]}`,
    value: gems.map((a) => ({
      track_id: h.trackIds[a.t],
      name: h.names[a.t],
      artist: h.artists[a.t],
      play_count: a.plays,
      fly_score_rank: rank(a),
      fly_score_percentile: Math.round(1000 * fly.percentile[fly.catalogRow[a.t]]) / 1000,
      cluster: clusterName(a),
    })),
    sublabel: `the fly brain ranks it #${n(rank(top))} of ${n(catSize)} for your taste, but you've played it ${
      top.plays === 1 ? "once" : `${top.plays} times`
    }`,
    coverage: tierA(ctx, fly, assets),
    evidence: gems.slice(0, 3).map((a) => `"${h.names[a.t]}" — ${h.artists[a.t]} (#${n(rank(a))}, ${clusterName(a) ?? "unclustered"})`),
  };
}

// -- archetype ----------------------------------------------------------------

// The values the rules test, plus each busy month's first-listen share for
// the "most adventurous month" line. A metric whose inputs are missing is
// absent, and its rule can't fire.
type Metrics = { values: Record<string, number>; monthly: Map<string, number> };

function archetypeMetrics(
  ctx: Ctx,
  first: Uint8Array,
  dated: Dated[],
  windowTo: string | null,
  summary: ClusterRow[] | null,
): Metrics {
  const count = ctx.to - ctx.from;
  if (!count) return { values: {}, monthly: new Map() };
  let firsts = 0;
  let skips = 0;
  const monthly = new Map<string, [number, number]>();
  for (let i = ctx.from; i < ctx.to; i++) {
    firsts += first[i];
    if (ctx.h.seqVerdict[i] < 0) skips++;
    const m = ctx.local[i].month;
    const s = monthly.get(m) ?? [0, 0];
    s[0]++;
    s[1] += first[i];
    monthly.set(m, s);
  }
  const counts = [...ctx.tracks.values()].map((a) => a.plays).sort((a, b) => b - a);
  const top1 = Math.max(1, Math.floor(counts.length / 100));
  const metrics: Record<string, number> = {
    discovery_ratio: firsts / count,
    top1_share: counts.slice(0, top1).reduce((s, c) => s + c, 0) / count,
    skip_rate: skips / count,
  };
  const byMonth = new Map(
    [...monthly.entries()].filter(([, [size]]) => size >= ARCHETYPE_MIN_MONTH_PLAYS).map(([m, [size, f]]) => [m, f / size]),
  );

  if (dated.length && windowTo) {
    const cutoff = Number(windowTo.slice(0, 4)) - ARCHETYPE_OLD_YEARS;
    const all = dated.reduce((s, d) => s + d.plays, 0);
    metrics.old_share = dated.filter((d) => d.year < cutoff).reduce((s, d) => s + d.plays, 0) / all;
  }
  if (summary) {
    const shares = summary.map((r) => r.play_share).filter((s) => s > 0);
    if (shares.length > 1) metrics.cluster_evenness = -shares.reduce((s, p) => s + p * Math.log(p), 0) / Math.log(shares.length);
  }
  return { values: metrics, monthly: byMonth };
}

function cardArchetype({ values, monthly }: Metrics, summary: ClusterRow[] | null): AnyCard | null {
  if (!Object.keys(values).length) return null;
  let best: { margin: number; rule: Rule; line: string } | null = null;
  for (const rule of ARCHETYPES) {
    const v = values[rule.metric];
    if (v === undefined) continue;
    let margin = 0;
    if (rule.dir === ">=" && v >= rule.threshold) margin = v / rule.threshold;
    else if (rule.dir === "<=" && v > 0 && v <= rule.threshold) margin = rule.threshold / v;
    if (margin && (!best || margin > best.margin)) best = { margin, rule, line: rule.line(pct(v)) };
  }
  const evidence = [best ? best.line : "no single habit clears an archetype threshold"];
  if (monthly.size) {
    const [m, share] = [...monthly.entries()].reduce((a, b) => (b[1] > a[1] ? b : a));
    const [y, mo] = m.split("-").map(Number);
    const label = new Date(Date.UTC(y, mo - 1, 1)).toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
    evidence.push(`your most adventurous month was ${label} (${pct(share)} first listens)`);
  }
  if (summary?.[0]?.plays) evidence.push(`anchored in your biggest vibe, ${summary[0].name}`);
  return {
    id: "archetype",
    headline: `You're ${best ? best.rule.label : "The All-Rounder"}`,
    value: best ? best.rule.id : "all_rounder",
    sublabel: "fixed, documented thresholds on your own habits, not a ranking against other listeners",
    metrics: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Math.round(1000 * v) / 1000])),
    evidence: evidence.slice(0, 3),
  } as AnyCard;
}

// -- assembly -------------------------------------------------------------------

function buildReport(
  label: { id: string; label: string },
  ctx: Ctx,
  first: Uint8Array,
  fly: Fly | null,
  assets: RewindAssets | null,
  catSize: number,
): Report {
  const { h, from, to } = ctx;
  const artists = new Set<string>();
  for (const a of ctx.tracks.values()) artists.add(h.artists[a.t]);
  const totals = {
    plays: to - from,
    hours: Math.round(ctx.totalMs / 360_000) / 10,
    tracks: ctx.tracks.size,
    artists: artists.size,
  };
  const windowTo = to > from ? ctx.local[to - 1].date : null;
  const dated = assets ? datedTracks(ctx, assets) : [];
  const summary = fly && assets ? clusterSummary(ctx, fly, assets) : null;

  const cards: Record<string, AnyCard | null> = {
    total_hours: cardTotalHours(ctx, totals),
    top_artists: cardTopArtists(ctx),
    top_tracks: cardTopTracks(ctx),
    top_albums: cardTopAlbums(ctx),
    artist_sprint: cardArtistSprint(ctx),
    time_of_day: cardTimeOfDay(ctx, h.timeZone),
    skip_offenders: cardSkipOffenders(ctx),
    listening_age: cardListeningAge(ctx, dated, windowTo),
    decade_histogram: cardDecadeHistogram(ctx, dated),
  };
  if (fly && assets && summary) {
    cards.taste_clusters = cardTasteClusters(ctx, fly, assets, summary);
    cards.hidden_gems = cardHiddenGems(ctx, fly, assets, summary, catSize);
  }
  cards.archetype = cardArchetype(archetypeMetrics(ctx, first, dated, windowTo, summary), summary);

  return {
    schema_version: SCHEMA_VERSION,
    audience: "visitor",
    generated_at: new Date().toISOString(),
    config_hash: assets ? `browser-${assets.clusterKey}` : "browser",
    window: { ...label, from: to > from ? ctx.local[from].date : null, to: windowTo },
    totals,
    cards: CARD_ORDER.map((id) => cards[id]).filter((c): c is AnyCard => c != null),
  };
}

// All-time first, then one report per calendar year in the visitor's zone,
// as `export_reports` writes them. Without the catalog or the assets the
// fly and release-year cards drop and the rest still render. `mb` reuses a
// mushroom body already trained on this history.
export function buildVisitorReports(
  h: History,
  cat: Catalog | null,
  assets: RewindAssets | null,
  mb?: MushroomBody,
): LocalReports {
  const clock = localClock(h.timeZone);
  const count = h.seqTrack.length;
  const local = Array.from(h.seqTs, clock);

  // A first listen is the first play in the whole history, so a favourite
  // from an earlier year replayed later isn't a discovery.
  const first = new Uint8Array(count);
  const heard = new Uint8Array(h.trackIds.length);
  for (let i = 0; i < count; i++) {
    if (!heard[h.seqTrack[i]]) first[i] = heard[h.seqTrack[i]] = 1;
  }

  let fly: Fly | null = null;
  if (cat && assets) {
    const trained = mb ?? trainMushroomBody(cat, h, matchHistory(cat, h));
    fly = flyBase(cat, h, assets, trained);
  }

  const windows: { id: string; label: string; from: number; to: number }[] = [{ id: "all", label: "All time", from: 0, to: count }];
  // `seq*` is chronological, so each local year is one contiguous run.
  for (let i = 0; i < count; ) {
    const year = local[i].year;
    let j = i;
    while (j < count && local[j].year === year) j++;
    windows.push({ id: String(year), label: String(year), from: i, to: j });
    i = j;
  }

  const reports: Record<string, Report> = {};
  for (const w of windows) {
    reports[w.id] = buildReport(w, windowCtx(h, local, w.from, w.to), first, fly, assets, cat?.size ?? 0);
  }
  const index: ReportIndex = {
    schema_version: SCHEMA_VERSION,
    audience: "visitor",
    privacy: "Built in this browser tab from your own export. Nothing was uploaded or stored.",
    windows: windows.map((w) => ({ ...reports[w.id].window, plays: w.to - w.from })),
  };
  return { index, reports };
}
