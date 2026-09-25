/// <reference lib="webworker" />
// Parses a Spotify Extended Streaming History zip entirely inside this
// worker. Nothing here makes a network request except fetching the bundled
// sample export, and the visitor's own file never leaves the tab.
//
// Files are unzipped and parsed one at a time, and a dashboard snapshot is
// posted after each, so the first charts appear while later years are
// still being read.

import { strFromU8, unzipSync } from "fflate";
import type { ArtistRow, Dashboard, Drift, History, SkipRow, WorkerIn, WorkerOut } from "./types";

type RawPlay = {
  ts?: string;
  ms_played?: number;
  master_metadata_track_name?: string | null;
  master_metadata_album_artist_name?: string | null;
  spotify_track_uri?: string | null;
  reason_end?: string | null;
};

// Same thresholds as `selector.ingest.load_history.VERDICT_RULES`.
const FWDBTN_SKIP_COMPLETION_MAX = 0.8;
const ENDPLAY_PUNISH_COMPLETION_MAX = 0.3;
// Same as `selector.warehouse.queries.skip_offenders`.
const SKIP_OFFENDER_MIN_PLAYS = 5;
const SKIP_OFFENDER_MIN_RATE = 0.3;

const R_TRACKDONE = 0;
const R_FWDBTN = 1;
const R_ENDPLAY = 2;
const R_BACKBTN = 3;
const R_OTHER = 4;
const REASON_CODE: Record<string, number> = {
  trackdone: R_TRACKDONE,
  fwdbtn: R_FWDBTN,
  endplay: R_ENDPLAY,
  backbtn: R_BACKBTN,
};

const post = (msg: WorkerOut, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

class State {
  trackIndex = new Map<string, number>();
  trackIds: string[] = [];
  trackNames: string[] = [];
  trackArtist: number[] = [];
  artistIndex = new Map<string, number>();
  artistNames: string[] = [];

  ts: number[] = [];
  ms: number[] = [];
  track: number[] = [];
  reason: number[] = [];
  local: number[] = []; // dow * 24 + hour, in the display time zone

  private hourCache = new Map<number, number>();
  private fmt: Intl.DateTimeFormat;

  constructor(timeZone: string) {
    this.fmt = new Intl.DateTimeFormat("en-US", {
      timeZone,
      weekday: "short",
      hour: "numeric",
      hourCycle: "h23",
    });
  }

  // Intl formatting is slow per call; every play in the same UTC hour
  // shares a local weekday and hour, so cache by the hour bucket.
  private localSlot(t: number): number {
    const bucket = Math.floor(t / 3_600_000);
    let slot = this.hourCache.get(bucket);
    if (slot === undefined) {
      const parts = this.fmt.formatToParts(new Date(bucket * 3_600_000));
      const wd = parts.find((p) => p.type === "weekday")?.value ?? "Mon";
      const hour = Number(parts.find((p) => p.type === "hour")?.value ?? 0) % 24;
      const dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].indexOf(wd);
      slot = Math.max(dow, 0) * 24 + hour;
      this.hourCache.set(bucket, slot);
    }
    return slot;
  }

  add(records: RawPlay[]) {
    for (const r of records) {
      // Podcast and audiobook rows carry no track URI; they aren't music plays.
      const uri = r.spotify_track_uri;
      if (!uri || !r.ts) continue;
      const t = Date.parse(r.ts);
      if (Number.isNaN(t)) continue;
      const id = uri.startsWith("spotify:track:") ? uri.slice(14) : uri;

      let ti = this.trackIndex.get(id);
      if (ti === undefined) {
        const artist = r.master_metadata_album_artist_name ?? "Unknown artist";
        let ai = this.artistIndex.get(artist);
        if (ai === undefined) {
          ai = this.artistNames.length;
          this.artistIndex.set(artist, ai);
          this.artistNames.push(artist);
        }
        ti = this.trackIds.length;
        this.trackIndex.set(id, ti);
        this.trackIds.push(id);
        this.trackNames.push(r.master_metadata_track_name ?? "Unknown track");
        this.trackArtist.push(ai);
      }
      this.ts.push(t);
      this.ms.push(r.ms_played ?? 0);
      this.track.push(ti);
      this.reason.push(REASON_CODE[r.reason_end ?? ""] ?? R_OTHER);
      this.local.push(this.localSlot(t));
    }
  }

  // The export has no track durations, so the longest play of a track
  // stands in for its length, as in `selector.ingest.load_history`.
  verdicts(): Int8Array {
    const n = this.ts.length;
    const maxMs = new Float64Array(this.trackIds.length);
    for (let i = 0; i < n; i++) if (this.ms[i] > maxMs[this.track[i]]) maxMs[this.track[i]] = this.ms[i];
    const v = new Int8Array(n);
    for (let i = 0; i < n; i++) {
      const m = maxMs[this.track[i]];
      const completion = m > 0 ? Math.min(this.ms[i] / m, 1) : 0;
      const r = this.reason[i];
      if (r === R_TRACKDONE || r === R_BACKBTN) v[i] = 1;
      else if (r === R_FWDBTN && completion < FWDBTN_SKIP_COMPLETION_MAX) v[i] = -1;
      else if (r === R_ENDPLAY && completion < ENDPLAY_PUNISH_COMPLETION_MAX) v[i] = -1;
    }
    return v;
  }

  dashboard(filesDone: number, filesTotal: number, timeZone: string): Dashboard {
    const n = this.ts.length;
    const nArtists = this.artistNames.length;
    const nTracks = this.trackIds.length;

    let first = Infinity;
    let last = -Infinity;
    let totalMs = 0;
    const clock = new Array(7 * 24).fill(0);
    const artistPlays = new Float64Array(nArtists);
    const artistMs = new Float64Array(nArtists);
    const trackPlays = new Uint32Array(nTracks);
    const trackSkips = new Uint32Array(nTracks);
    for (let i = 0; i < n; i++) {
      const t = this.ts[i];
      if (t < first) first = t;
      if (t > last) last = t;
      totalMs += this.ms[i];
      clock[this.local[i]]++;
      const a = this.trackArtist[this.track[i]];
      artistPlays[a]++;
      artistMs[a] += this.ms[i];
      trackPlays[this.track[i]]++;
      if (this.reason[i] === R_FWDBTN) trackSkips[this.track[i]]++;
    }

    const byPlays = Array.from({ length: nArtists }, (_, i) => i).sort((a, b) => artistPlays[b] - artistPlays[a]);
    const topArtists: ArtistRow[] = byPlays.slice(0, 10).map((a) => ({
      artist: this.artistNames[a],
      plays: artistPlays[a],
      hours: artistMs[a] / 3_600_000,
    }));

    const skipOffenders: SkipRow[] = [];
    for (let t = 0; t < nTracks; t++) {
      const rate = trackPlays[t] ? trackSkips[t] / trackPlays[t] : 0;
      if (trackPlays[t] >= SKIP_OFFENDER_MIN_PLAYS && rate >= SKIP_OFFENDER_MIN_RATE) {
        skipOffenders.push({
          track: this.trackNames[t],
          artist: this.artistNames[this.trackArtist[t]],
          plays: trackPlays[t],
          skipRate: rate,
        });
      }
    }
    skipOffenders.sort((a, b) => b.skipRate - a.skipRate || b.plays - a.plays);

    const v = this.verdicts();
    let reward = 0;
    let punish = 0;
    for (let i = 0; i < n; i++) {
      if (v[i] > 0) reward++;
      else if (v[i] < 0) punish++;
    }

    return {
      summary: {
        plays: n,
        tracks: nTracks,
        artists: nArtists,
        hours: totalMs / 3_600_000,
        first: n ? first : 0,
        last: n ? last : 0,
        filesDone,
        filesTotal,
      },
      clock,
      timeZone,
      drift: this.drift(first, last, artistPlays),
      topArtists,
      skipOffenders: skipOffenders.slice(0, 8),
      verdicts: { reward, punish, neutral: n - reward - punish },
    };
  }

  // Share of each period's plays per artist. The series are the artists
  // that led at least one period (so each era of listening gets a colour),
  // most-played first, topped up with the overall top artists, max five.
  private drift(first: number, last: number, artistPlays: Float64Array): Drift {
    const spanYears = (last - first) / (365.25 * 86_400_000);
    const granularity: Drift["granularity"] = spanYears > 6 ? "year" : "quarter";
    const key = (t: number) => {
      const d = new Date(t);
      const y = d.getUTCFullYear();
      return granularity === "year" ? `${y}` : `${y} Q${Math.floor(d.getUTCMonth() / 3) + 1}`;
    };

    const counts = new Map<string, Map<number, number>>();
    for (let i = 0; i < this.ts.length; i++) {
      const k = key(this.ts[i]);
      let m = counts.get(k);
      if (!m) counts.set(k, (m = new Map()));
      const a = this.trackArtist[this.track[i]];
      m.set(a, (m.get(a) ?? 0) + 1);
    }
    const periods = [...counts.keys()].sort();
    const leaders = periods.map((p) => {
      let best = -1;
      let bestN = -1;
      for (const [a, c] of counts.get(p)!) if (c > bestN) [best, bestN] = [a, c];
      return best;
    });

    const series: number[] = [];
    const byTotal = (a: number, b: number) => artistPlays[b] - artistPlays[a];
    for (const a of [...new Set(leaders)].sort(byTotal)) if (series.length < 5) series.push(a);
    const overall = Array.from(artistPlays.keys()).sort(byTotal);
    for (const a of overall) {
      if (series.length >= 5) break;
      if (!series.includes(a)) series.push(a);
    }

    const totals = periods.map((p) => [...counts.get(p)!.values()].reduce((s, c) => s + c, 0));
    const shares = periods.map((p, i) => series.map((a) => (counts.get(p)!.get(a) ?? 0) / totals[i]));
    return {
      granularity,
      periods,
      series: series.map((a) => this.artistNames[a]),
      shares,
      other: shares.map((row) => Math.max(0, 1 - row.reduce((s, x) => s + x, 0))),
      totals,
      leaders: leaders.map((a) => this.artistNames[a]),
    };
  }

  history(): History {
    const n = this.ts.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => this.ts[a] - this.ts[b]);
    const v = this.verdicts();
    const seqTrack = new Uint32Array(n);
    const seqVerdict = new Int8Array(n);
    const plays = new Uint32Array(this.trackIds.length);
    const lastPlayed = new Float64Array(this.trackIds.length);
    order.forEach((i, k) => {
      const t = this.track[i];
      seqTrack[k] = t;
      seqVerdict[k] = v[i];
      plays[t]++;
      if (this.ts[i] > lastPlayed[t]) lastPlayed[t] = this.ts[i];
    });
    return {
      trackIds: this.trackIds,
      names: this.trackNames,
      artists: this.trackArtist.map((a) => this.artistNames[a]),
      plays,
      lastPlayed,
      seqTrack,
      seqVerdict,
    };
  }
}

async function readSource(msg: WorkerIn): Promise<Uint8Array> {
  if (msg.source.kind === "file") return new Uint8Array(await msg.source.file.arrayBuffer());
  const res = await fetch(msg.source.url);
  if (!res.ok || !res.body) throw new Error(`Couldn't fetch the sample export (${res.status}).`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: "progress", stage: "download", loaded, total });
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

const isAudioHistory = (name: string) => {
  const base = name.split("/").pop() ?? name;
  return base.startsWith("Streaming_History_Audio_") && base.endsWith(".json");
};

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const started = performance.now();
  try {
    const zip = await readSource(e.data);

    // List the audio history files without decompressing anything.
    const names: string[] = [];
    unzipSync(zip, {
      filter: (f) => {
        if (isAudioHistory(f.name)) names.push(f.name);
        return false;
      },
    });
    names.sort();
    if (!names.length) {
      throw new Error(
        "No Streaming_History_Audio_*.json files in this zip. Selector needs the " +
          "\"Extended streaming history\" export, not the default account data.",
      );
    }

    const state = new State(e.data.timeZone);
    for (let i = 0; i < names.length; i++) {
      const files = unzipSync(zip, { filter: (f) => f.name === names[i] });
      const records = JSON.parse(strFromU8(files[names[i]])) as RawPlay[];
      state.add(records);
      post({ type: "progress", stage: "unzip", loaded: i + 1, total: names.length });
      if (i < names.length - 1 && state.ts.length) {
        post({ type: "snapshot", dashboard: state.dashboard(i + 1, names.length, e.data.timeZone) });
      }
    }
    if (!state.ts.length) throw new Error("The export parsed, but it contains no music plays.");

    const history = state.history();
    post(
      {
        type: "done",
        dashboard: state.dashboard(names.length, names.length, e.data.timeZone),
        history,
        ms: performance.now() - started,
      },
      [history.plays.buffer, history.lastPlayed.buffer, history.seqTrack.buffer, history.seqVerdict.buffer],
    );
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
