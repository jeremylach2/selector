// The fly circuit itself, for the `/watch` visualiser. The catalog ships
// only finished fingerprints; this reruns the hash that made them, live:
// input vector -> projection neurons -> FlyWire PN->KC wiring -> Kenyon
// cell drive -> winner-take-all. Built by `scripts/build_demo_assets.py`
// (`write_circuit`) from the same `FlyHash` fit as `data/fly_tags.npz`.

import { type Catalog, fetchBytes } from "./catalog";
import { strFromU8 } from "fflate";

export type Column = { name: string; kind: "predicted" | "measured" };

export type Circuit = {
  flywire: string;
  hemisphere: string;
  nPnReal: number; // real ALPNs before pooling
  rawSynapses: number;
  columns: Column[];
  dIn: number;
  nKc: number;
  mean: Float64Array;
  std: Float64Array;
  // Pooled PN->KC projection, CSR, one row per Kenyon cell, in the exact
  // order scipy stored it so the float sums match the offline run.
  indptr: Int32Array;
  indices: Uint8Array;
  data: Float64Array;
  valence: Float64Array; // per catalog row
  intensity: Float64Array;
  measured: Map<number, number[]>;
};

type CircuitJson = {
  flywire: string;
  hemisphere: string;
  n_pn_real: number;
  raw_synapses: number;
  columns: Column[];
  mean: number[];
  std: number[];
  indptr: number[];
  indices: number[];
  data: number[];
  valence: number[];
  intensity: number[];
  measured_rows: number[];
  measured: number[][];
};

export async function loadCircuit(base = "/fly"): Promise<Circuit> {
  const j = JSON.parse(strFromU8(await fetchBytes(`${base}/circuit.json.gz`))) as CircuitJson;
  return {
    flywire: j.flywire,
    hemisphere: j.hemisphere,
    nPnReal: j.n_pn_real,
    rawSynapses: j.raw_synapses,
    columns: j.columns,
    dIn: j.columns.length,
    nKc: j.indptr.length - 1,
    mean: Float64Array.from(j.mean),
    std: Float64Array.from(j.std),
    indptr: Int32Array.from(j.indptr),
    indices: Uint8Array.from(j.indices),
    data: Float64Array.from(j.data),
    valence: Float64Array.from(j.valence),
    intensity: Float64Array.from(j.intensity),
    measured: new Map(j.measured_rows.map((r, i) => [r, j.measured[i]])),
  };
}

// `selector.fly.pipeline.build_feature_matrix("full")` for one track:
// valence, intensity, one-hot era, multi-hot moods, four measured audio
// columns and a has-audio flag.
export function inputVector(cat: Catalog, c: Circuit, row: number): Float64Array {
  const x = new Float64Array(c.dIn);
  let j = 0;
  x[j++] = c.valence[row];
  x[j++] = c.intensity[row];
  for (let e = 0; e < cat.eraVocab.length; e++) x[j++] = cat.era[row] === e ? 1 : 0;
  for (let m = 0; m < cat.moodVocab.length; m++) x[j++] = cat.mood[row] & (1 << m) ? 1 : 0;
  const measured = c.measured.get(row);
  for (let m = 0; m < 4; m++) x[j++] = measured ? measured[m] : 0;
  x[j++] = measured ? 1 : 0;
  return x;
}

export type Firing = {
  x: Float64Array; // raw input vector
  pn: Float64Array; // centred, scaled, rectified: projection neuron rates
  drive: Float64Array; // per Kenyon cell, before inhibition
  driven: number; // cells with any drive at all
  maxDrive: number;
  cutoff: number; // drive of the k-th strongest cell
  winners: Uint16Array; // the k survivors, ascending index
  isWinner: Uint8Array;
  // Cells exactly at the cutoff competing for fewer slots than there are
  // of them (0 when there's no contest), and how many slots that was.
  tied: number;
  tiedSlots: number;
  matchesShipped: boolean;
};

// One pass through the circuit, as `FlyHash.transform`. Winner-take-all
// keeps the k strongest cells; where several cells tie exactly at the
// cutoff, numpy's argpartition picked among them offline and that choice
// isn't reproducible here, so ties are settled by the shipped fingerprint.
// Everything strictly above the cutoff is decided by this computation.
export function fire(cat: Catalog, c: Circuit, row: number): Firing {
  const x = inputVector(cat, c, row);
  const pn = new Float64Array(c.dIn);
  for (let j = 0; j < c.dIn; j++) pn[j] = Math.max(0, (x[j] - c.mean[j]) / c.std[j]);

  const drive = new Float64Array(c.nKc);
  let driven = 0;
  let maxDrive = 0;
  for (let i = 0; i < c.nKc; i++) {
    let s = 0;
    for (let jj = c.indptr[i]; jj < c.indptr[i + 1]; jj++) s += c.data[jj] * pn[c.indices[jj]];
    drive[i] = s;
    if (s > 0) driven++;
    if (s > maxDrive) maxDrive = s;
  }

  const k = cat.kActive;
  const cutoff = Float64Array.from(drive).sort()[c.nKc - k];
  const shipped = new Uint8Array(c.nKc);
  for (let j = row * k; j < row * k + k; j++) shipped[cat.active[j]] = 1;

  const isWinner = new Uint8Array(c.nKc);
  let n = 0;
  let atCutoff = 0;
  for (let i = 0; i < c.nKc; i++) {
    if (drive[i] > cutoff) (isWinner[i] = 1), n++;
    else if (drive[i] === cutoff) atCutoff++;
  }
  const slots = k - n;
  for (let i = 0; i < c.nKc && n < k; i++) if (drive[i] === cutoff && shipped[i]) (isWinner[i] = 1), n++;
  const winners = new Uint16Array(k);
  let w = 0;
  let same = n === k;
  for (let i = 0; i < c.nKc; i++) {
    if (isWinner[i] && w < k) winners[w++] = i;
    if (isWinner[i] !== shipped[i]) same = false;
  }
  return { x, pn, drive, driven, maxDrive, cutoff, winners, isWinner, tied: atCutoff > slots ? atCutoff : 0, tiedSlots: atCutoff > slots ? slots : 0, matchesShipped: same };
}
