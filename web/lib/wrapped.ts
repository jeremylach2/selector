// The Wrapped report object, schema 2.0, as written by
// `python -m selector.warehouse.wrapped` into public/wrapped/. Mirrors
// `selector.warehouse.wrapped` -- change both together.

export const SCHEMA_MAJOR = "2";

export type Coverage = {
  tier: "A" | "B";
  tracks_used: number;
  of: number;
  plays_used?: number; // tier B: plays with a release year
  plays_of?: number;
  measured?: number; // tier A: tracks with measured audio
};

type CardOf<I extends string, V> = {
  id: I;
  headline: string;
  sublabel?: string;
  evidence: string[];
  coverage?: Coverage;
  value: V;
};

export type ArtistRow = { artist: string; play_count: number };
export type TrackRow = { name: string; artist: string; play_count: number };
export type AlbumRow = { album: string; artist: string; play_count: number };

export type Card =
  | CardOf<"total_hours", number>
  | CardOf<"top_artists", ArtistRow[]>
  | CardOf<"top_tracks", TrackRow[]>
  | CardOf<"top_albums", AlbumRow[]>
  // cumulative[m][a]: artist a's running play total at the end of months[m].
  | CardOf<"artist_sprint", { months: string[]; artists: string[]; cumulative: number[][] }>
  // Local hours in `timezone` (IANA, e.g. America/Chicago).
  | CardOf<"time_of_day", { timezone: string; timezone_label: string; peak_hour: number; by_hour: number[] }>
  | CardOf<"skip_offenders", (TrackRow & { skip_rate: number })[]>
  | CardOf<"listening_age", number>
  | CardOf<"decade_histogram", { decade: number; play_count: number; share: number }[]>
  | CardOf<
      "taste_clusters",
      {
        name: string;
        built_name: string;
        play_share: number;
        track_count: number;
        measured_share: number;
        top_artists: string[];
      }[]
    >
  | CardOf<
      "hidden_gems",
      (TrackRow & {
        track_id: string;
        fly_score_rank: number;
        fly_score_percentile: number;
        cluster: string | null;
      })[]
    >
  | (CardOf<"archetype", string> & { metrics: Record<string, number> });

// Any card the renderer doesn't know draws as headline + evidence.
export type AnyCard = Card | CardOf<string, unknown>;

export type Report = {
  schema_version: string;
  generated_at: string;
  config_hash: string;
  window: { id: string; label: string; from: string | null; to: string | null };
  totals: { plays: number; hours: number; tracks: number; artists: number };
  cards: AnyCard[];
};

export type ReportIndex = {
  schema_version: string;
  windows: { id: string; label: string; from: string | null; to: string | null; plays: number }[];
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

function checkSchema(version: string, url: string): void {
  if (version.split(".")[0] !== SCHEMA_MAJOR) {
    throw new Error(`${url} is schema ${version}; this page reads ${SCHEMA_MAJOR}.x`);
  }
}

export async function loadIndex(): Promise<ReportIndex> {
  const url = "/wrapped/index.json";
  const index = await getJson<ReportIndex>(url);
  checkSchema(index.schema_version, url);
  return index;
}

export async function loadReport(id: string): Promise<Report> {
  const url = `/wrapped/${encodeURIComponent(id)}.json`;
  const report = await getJson<Report>(url);
  checkSchema(report.schema_version, url);
  return report;
}

export function monthLabel(ym: string): string {
  const [y, m] = ym.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en", { month: "short", year: "numeric", timeZone: "UTC" });
}

export function coverageNote(c: Coverage | undefined): string | null {
  if (!c) return null;
  if (c.plays_of) return `Tier ${c.tier} · covers ${Math.round((100 * (c.plays_used ?? 0)) / c.plays_of)}% of plays`;
  if (c.measured !== undefined && c.tracks_used) {
    return `Tier ${c.tier} · ${Math.round((100 * c.measured) / c.tracks_used)}% of these tracks have measured audio, the rest predicted features`;
  }
  return `Tier ${c.tier} · ${c.tracks_used.toLocaleString()} of ${c.of.toLocaleString()} tracks`;
}
