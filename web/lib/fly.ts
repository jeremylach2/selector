// The fly brain, client side. A port of `selector/fly/lsh.py`'s Hamming
// search and `selector/fly/mbon.py`'s plasticity rule over the precomputed
// catalog tags. The tags themselves (FlyWire PN->KC projection, then
// winner-take-all to 130 of 2,597 Kenyon cells) were computed offline.

import type { Catalog } from "./catalog";
import type { History } from "./types";

function popcount(x: number): number {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

// Shared active Kenyon cells between two catalog rows. Every tag has the
// same number of active cells, so Hamming distance = 2 * (k - shared).
export function sharedCells(cat: Catalog, a: number, b: number): number {
  const w = cat.words;
  let s = 0;
  for (let i = 0; i < w; i++) s += popcount(cat.bits[a * w + i] & cat.bits[b * w + i]);
  return s;
}

export function hamming(cat: Catalog, a: number, b: number): number {
  return 2 * (cat.kActive - sharedCells(cat, a, b));
}

// Hamming distance from `query` to each row in `rows` (all rows if omitted).
export function hammingTo(cat: Catalog, query: number, rows?: ArrayLike<number>): Uint16Array {
  const n = rows ? rows.length : cat.size;
  const out = new Uint16Array(n);
  const w = cat.words;
  const q = cat.bits.subarray(query * w, query * w + w);
  for (let j = 0; j < n; j++) {
    const r = rows ? rows[j] : j;
    const base = r * w;
    let s = 0;
    for (let i = 0; i < w; i++) s += popcount(q[i] & cat.bits[base + i]);
    out[j] = 2 * (cat.kActive - s);
  }
  return out;
}

export type Neighbour = { row: number; distance: number; shared: number };

// "Song - 2011 Remaster" and "Song" are the same recording to a listener.
const baseTitle = (name: string) => name.toLowerCase().split(" - ")[0].replace(/\s*\(.*?\)\s*/g, "").trim();

export function moreLikeThis(cat: Catalog, query: number, k = 10): Neighbour[] {
  const d = hammingTo(cat, query);
  const title = baseTitle(cat.name[query]);
  const idx: number[] = [];
  // Alternate versions of the seed itself (remasters, live cuts) land at or
  // near distance 0, which is correct but not a recommendation.
  for (let i = 0; i < cat.size; i++) {
    if (i !== query && !(cat.artist[i] === cat.artist[query] && baseTitle(cat.name[i]) === title)) idx.push(i);
  }
  idx.sort((a, b) => d[a] - d[b]);
  return idx.slice(0, k).map((row) => ({ row, distance: d[row], shared: cat.kActive - d[row] / 2 }));
}

// Same constants as `selector.fly.pipeline` (MBON_LR, MBON_DECAY).
export const MBON_LR = 0.05;
export const MBON_DECAY = 0.01;

// Two mushroom-body output neurons, approach and avoid. A reward depresses
// the avoid synapses of the active Kenyon cells, a punishment depresses the
// approach synapses; nothing is ever strengthened. Decay pulls every
// synapse a little back towards the naive weight after each lesson.
export class MushroomBody {
  wApproach: Float64Array;
  wAvoid: Float64Array;
  lessons = 0;

  constructor(
    nKc: number,
    readonly lr = MBON_LR,
    readonly decay = MBON_DECAY,
  ) {
    this.wApproach = new Float64Array(nKc).fill(1);
    this.wAvoid = new Float64Array(nKc).fill(1);
  }

  learn(cat: Catalog, row: number, verdict: number) {
    const k = cat.kActive;
    const act = cat.active.subarray(row * k, row * k + k);
    const f = 1 - this.lr;
    if (verdict > 0) for (const i of act) this.wAvoid[i] *= f;
    else if (verdict < 0) for (const i of act) this.wApproach[i] *= f;
    if (this.decay) {
      const d = this.decay;
      for (let i = 0; i < this.wApproach.length; i++) {
        this.wApproach[i] += d * (1 - this.wApproach[i]);
        this.wAvoid[i] += d * (1 - this.wAvoid[i]);
      }
    }
    if (verdict !== 0) this.lessons++;
  }

  valence(cat: Catalog, row: number): number {
    const k = cat.kActive;
    let s = 0;
    for (let j = row * k; j < row * k + k; j++) s += this.wApproach[cat.active[j]] - this.wAvoid[cat.active[j]];
    return s;
  }
}

export type Overlap = {
  // History track index -> catalog row, for tracks the catalog knows.
  catalogRow: Int32Array;
  tracksMatched: number;
  playsMatched: number;
  playsTotal: number;
};

export function matchHistory(cat: Catalog, h: History): Overlap {
  const catalogRow = new Int32Array(h.trackIds.length).fill(-1);
  let tracksMatched = 0;
  h.trackIds.forEach((id, t) => {
    const r = cat.row.get(id);
    if (r !== undefined) {
      catalogRow[t] = r;
      tracksMatched++;
    }
  });
  let playsMatched = 0;
  for (const t of h.seqTrack) if (catalogRow[t] >= 0) playsMatched++;
  return { catalogRow, tracksMatched, playsMatched, playsTotal: h.seqTrack.length };
}

// Replay the visitor's plays in order through the plasticity rule, as
// `train_production_mbon` does over the author's warehouse. Only plays of
// catalog tracks can teach it: an untagged track has no Kenyon cells.
export function trainMushroomBody(cat: Catalog, h: History, overlap: Overlap): MushroomBody {
  const mb = new MushroomBody(cat.nKc);
  for (let i = 0; i < h.seqTrack.length; i++) {
    const r = overlap.catalogRow[h.seqTrack[i]];
    if (r >= 0) mb.learn(cat, r, h.seqVerdict[i]);
  }
  return mb;
}

// pandas `rank(pct=True, method="average")`.
export function percentileRank(values: Float64Array | number[]): Float64Array {
  const n = values.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
  const out = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[order[k]] = avg / n;
    i = j + 1;
  }
  return out;
}
