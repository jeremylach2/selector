// Messages between the page and the parse worker, and the shapes they carry.

export type ParseSource = { kind: "file"; file: File } | { kind: "url"; url: string };

export type WorkerIn = { source: ParseSource; timeZone: string };

export type Summary = {
  plays: number;
  tracks: number;
  artists: number;
  hours: number;
  first: number; // epoch ms
  last: number;
  filesDone: number;
  filesTotal: number;
};

export type Drift = {
  granularity: "quarter" | "year";
  periods: string[];
  // Series are artists, in fixed legend order. `shares[p][s]` is the share
  // of period p's plays that went to series s; `other[p]` is the rest.
  series: string[];
  shares: number[][];
  other: number[];
  totals: number[];
  leaders: string[];
};

export type ArtistRow = { artist: string; plays: number; hours: number };
export type SkipRow = { track: string; artist: string; plays: number; skipRate: number };

export type Dashboard = {
  summary: Summary;
  // 7 x 24, Monday first, in `timeZone`.
  clock: number[];
  timeZone: string;
  drift: Drift;
  topArtists: ArtistRow[];
  skipOffenders: SkipRow[];
  verdicts: { reward: number; punish: number; neutral: number };
};

// Everything the fly brain and the DJ need from the history, sent once the
// whole export is parsed. Per-track arrays share one index; `seq*` arrays
// are every play in chronological order.
export type History = {
  trackIds: string[];
  names: string[];
  artists: string[];
  plays: Uint32Array;
  lastPlayed: Float64Array;
  seqTrack: Uint32Array;
  seqVerdict: Int8Array;
};

export type WorkerOut =
  | { type: "progress"; stage: "download" | "unzip"; loaded: number; total: number }
  | { type: "snapshot"; dashboard: Dashboard }
  | { type: "done"; dashboard: Dashboard; history: History; ms: number }
  | { type: "error"; message: string };
