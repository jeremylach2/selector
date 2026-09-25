// The fly brain's catalog: every tagged track and its Kenyon-cell
// fingerprint, precomputed by `scripts/build_demo_assets.py` and served as
// two static files. Only derived numbers ship; there is no audio.

import { gunzipSync, strFromU8 } from "fflate";

export type Catalog = {
  size: number;
  nKc: number;
  kActive: number;
  words: number; // Uint32 words per bitset row
  moodVocab: string[];
  eraVocab: string[];
  id: string[];
  name: string[];
  artist: string[];
  mood: number[]; // bitmask over moodVocab
  era: number[];
  tempo: (number | null)[]; // measured BPM
  energy: (number | null)[]; // measured, rank-normalised over the crate
  duration: (number | null)[]; // seconds, crate tracks only
  bits: Uint32Array; // size * words
  active: Uint16Array; // size * kActive, sorted per row
  row: Map<string, number>;
};

type CatalogJson = {
  n_kc: number;
  k_active: number;
  mood_vocab: string[];
  era_vocab: string[];
  id: string[];
  name: string[];
  artist: string[];
  mood: number[];
  era: number[];
  tempo: (number | null)[];
  energy: (number | null)[];
  duration: (number | null)[];
};

// A CDN may already have decoded the gzip for us (Content-Encoding), so
// only gunzip when the gzip magic bytes are actually there.
export async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Couldn't load ${url} (${res.status})`);
  const buf = new Uint8Array(await res.arrayBuffer());
  return buf[0] === 0x1f && buf[1] === 0x8b ? gunzipSync(buf) : buf;
}

export async function loadCatalog(base = "/fly"): Promise<Catalog> {
  const [meta, tagBytes] = await Promise.all([
    fetchBytes(`${base}/catalog.json.gz`).then((b) => JSON.parse(strFromU8(b)) as CatalogJson),
    fetchBytes(`${base}/tags.bin.gz`),
  ]);

  const size = meta.id.length;
  const k = meta.k_active;
  const words = Math.ceil(meta.n_kc / 32);
  const bits = new Uint32Array(size * words);
  const active = new Uint16Array(size * k);

  // Inverse of `_encode_tags`: uint8 gaps from the previous index, with 0
  // escaping a uint16 gap.
  let p = 0;
  for (let r = 0; r < size; r++) {
    let idx = -1;
    for (let j = 0; j < k; j++) {
      let gap = tagBytes[p++];
      if (gap === 0) {
        gap = tagBytes[p] | (tagBytes[p + 1] << 8);
        p += 2;
      }
      idx += gap;
      active[r * k + j] = idx;
      bits[r * words + (idx >>> 5)] |= 1 << (idx & 31);
    }
  }
  if (p !== tagBytes.length) throw new Error("fly tag file is corrupt");

  return {
    size,
    nKc: meta.n_kc,
    kActive: k,
    words,
    moodVocab: meta.mood_vocab,
    eraVocab: meta.era_vocab,
    id: meta.id,
    name: meta.name,
    artist: meta.artist,
    mood: meta.mood,
    era: meta.era,
    tempo: meta.tempo,
    energy: meta.energy,
    duration: meta.duration,
    bits,
    active,
    row: new Map(meta.id.map((id, i) => [id, i])),
  };
}

export function moodsOf(cat: Catalog, row: number): string[] {
  return cat.moodVocab.filter((_, i) => cat.mood[row] & (1 << i));
}
