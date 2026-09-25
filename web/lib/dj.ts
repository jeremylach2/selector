// The DJ agent, client side: Brief -> Arc -> Select -> Critique
// (-> Select -> Critique) -> liner notes. A port of `selector/dj/*.py`
// with the same constants, run over the catalog's DJ crate (tracks with
// measured tempo and energy) and a mushroom body trained on the visitor's
// own plays. Nothing is committed anywhere; this is always a dry run.

import type { Catalog } from "./catalog";
import { hammingTo, type MushroomBody, type Overlap, percentileRank } from "./fly";
import type { History } from "./types";

// -- themes (brief.py) ------------------------------------------------------

export type Theme = {
  name: string;
  description: string;
  moods: string[];
  floor: number;
  ceiling: number;
  hours: number[];
  weekendBonus: number;
};

const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);

export const THEMES: Theme[] = [
  { name: "slow sunrise", description: "Easing into the morning: warm, unhurried, a gentle lift.", moods: ["chill", "nostalgic", "romantic"], floor: 0.15, ceiling: 0.6, hours: range(5, 10), weekendBonus: 0 },
  { name: "focus drift", description: "Daytime concentration: steady, low-drama, nothing that grabs the wheel.", moods: ["chill", "nostalgic", "somber"], floor: 0.2, ceiling: 0.55, hours: range(9, 17), weekendBonus: 0 },
  { name: "golden hour", description: "Late afternoon into evening: bright, playful, building towards the night.", moods: ["euphoric", "playful", "romantic"], floor: 0.3, ceiling: 0.8, hours: range(15, 20), weekendBonus: 0 },
  { name: "night drive", description: "After dark on an empty road: moody, restless, a slow-burning peak.", moods: ["melancholic", "anxious", "somber", "nostalgic"], floor: 0.3, ceiling: 0.75, hours: [20, 21, 22, 23, 0], weekendBonus: 0 },
  { name: "peak time", description: "Weekend night energy: big, triumphant, built to peak hard.", moods: ["euphoric", "triumphant", "aggressive", "playful"], floor: 0.4, ceiling: 0.95, hours: [20, 21, 22, 23, 0, 1], weekendBonus: 0.6 },
  { name: "after hours", description: "The small hours: hushed, reflective, winding all the way down.", moods: ["chill", "melancholic", "somber"], floor: 0.1, ceiling: 0.5, hours: [0, 1, 2, 3, 4], weekendBonus: 0 },
  { name: "adrenaline", description: "Workout fuel: aggressive, driving, relentless.", moods: ["aggressive", "triumphant", "euphoric"], floor: 0.5, ceiling: 1.0, hours: [], weekendBonus: 0 },
];

const N_SEEDS = 5;
const FAMILIAR_WINDOW_DAYS = 90;

// -- arc (arc.py) -------------------------------------------------------------

const PHASES: [string, number, number, number, number][] = [
  ["opener", 0.0, 0.15, 0.2, 0.2],
  ["build", 0.15, 0.55, 0.2, 0.9],
  ["peak", 0.55, 0.8, 0.95, 0.95],
  ["comedown", 0.8, 1.0, 0.95, 0.25],
];
const TOLERANCE = 0.15;
const MAX_TEMPO_SHIFT = 0.12;
const MAX_ENERGY_STEP = 0.25;
const HARD_TEMPO_SHIFT = 2 * MAX_TEMPO_SHIFT;
const HARD_ENERGY_STEP = MAX_ENERGY_STEP;
const COMBINED_TEMPO_SHIFT = MAX_TEMPO_SHIFT;
const COMBINED_ENERGY_STEP = MAX_ENERGY_STEP / 2;

export class Arc {
  constructor(
    readonly minutes: number,
    readonly floor: number,
    readonly ceiling: number,
    readonly tolerance = TOLERANCE,
  ) {}
  get seconds() {
    return this.minutes * 60;
  }
  private segment(t: number) {
    t = Math.min(Math.max(t, 0), 1);
    return PHASES.find((p) => t < p[2]) ?? PHASES[PHASES.length - 1];
  }
  phase(t: number) {
    return this.segment(t)[0];
  }
  target(t: number) {
    const [, start, end, s0, s1] = this.segment(t);
    const u = (Math.min(Math.max(t, 0), 1) - start) / (end - start);
    return this.floor + (s0 + (s1 - s0) * u) * (this.ceiling - this.floor);
  }
}

// Fractional tempo change, folded across octaves (half and double time
// count as the same pulse). null if either side is unmeasured.
export function tempoShift(a: number | null, b: number | null): number | null {
  if (!a || !b) return null;
  return Math.min(...[0.5, 1, 2].map((k) => Math.abs(b * k - a) / a));
}

function jarringReason(ta: number | null, ea: number, tb: number | null, eb: number): string | null {
  const shift = tempoShift(ta, tb);
  const step = Math.abs(eb - ea);
  if (step > HARD_ENERGY_STEP) return `energy jump of ${step.toFixed(2)}`;
  if (shift !== null && shift > HARD_TEMPO_SHIFT) return `tempo lurch of ${pct(shift)}`;
  if (shift !== null && shift > COMBINED_TEMPO_SHIFT && step > COMBINED_ENERGY_STEP)
    return `tempo ${pct(shift)} and energy ${step.toFixed(2)} move together`;
  return null;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

// -- crate (pool.py) ----------------------------------------------------------

export type Crate = {
  rows: Int32Array; // catalog rows
  tempo: (number | null)[];
  energy: Float64Array;
  duration: Float64Array;
  moods: string[][];
  familiar: Uint8Array;
  played: Uint8Array;
  taste: Float64Array; // percentile of MBON valence over the crate
  valence: Float64Array;
};

export function buildCrate(cat: Catalog, h: History, overlap: Overlap, mb: MushroomBody): Crate {
  const rows: number[] = [];
  for (let r = 0; r < cat.size; r++) if (cat.energy[r] !== null && cat.duration[r] !== null) rows.push(r);

  // "Familiar" = played within 90 days of the visitor's newest play.
  let newest = 0;
  for (const t of h.lastPlayed) if (t > newest) newest = t;
  const cutoff = newest - FAMILIAR_WINDOW_DAYS * 86_400_000;
  const lastByRow = new Map<number, number>();
  overlap.catalogRow.forEach((r, t) => {
    if (r >= 0) lastByRow.set(r, h.lastPlayed[t]);
  });

  const valence = Float64Array.from(rows, (r) => mb.valence(cat, r));
  return {
    rows: Int32Array.from(rows),
    tempo: rows.map((r) => cat.tempo[r]),
    energy: Float64Array.from(rows, (r) => cat.energy[r]!),
    duration: Float64Array.from(rows, (r) => cat.duration[r]!),
    moods: rows.map((r) => cat.moodVocab.filter((_, i) => cat.mood[r] & (1 << i))),
    familiar: Uint8Array.from(rows, (r) => ((lastByRow.get(r) ?? 0) >= cutoff && lastByRow.has(r) ? 1 : 0)),
    played: Uint8Array.from(rows, (r) => (lastByRow.has(r) ? 1 : 0)),
    taste: percentileRank(valence),
    valence,
  };
}

// -- brief (brief.py) ---------------------------------------------------------

export type Brief = {
  theme: Theme;
  hour: number;
  weekday: string;
  isWeekend: boolean;
  recentArtists: [string, number][];
  recentMoods: [string, number][];
  seeds: number[]; // crate indices
  rationale: string;
  scores: Record<string, number>;
};

export function buildBrief(cat: Catalog, crate: Crate, h: History, overlap: Overlap, now: Date, requested: string | null): Brief {
  const vocab = cat.moodVocab;
  const crateIndex = new Map<number, number>();
  crate.rows.forEach((r, i) => crateIndex.set(r, i));

  // The newest 50 plays, newest first.
  const recent: number[] = [];
  for (let i = h.seqTrack.length - 1; i >= 0 && recent.length < 50; i--) recent.push(h.seqTrack[i]);

  const profile = new Array(vocab.length).fill(0);
  let nVec = 0;
  for (const t of recent) {
    const ci = crateIndex.get(overlap.catalogRow[t]);
    if (ci === undefined) continue;
    crate.moods[ci].forEach((m) => profile[vocab.indexOf(m)]++);
    nVec++;
  }
  const total = profile.reduce((s, x) => s + x, 0);
  const moodProfile = profile.map((x) => (nVec && total ? x / total : 0));

  const hour = now.getHours();
  const day = now.getDay(); // 0 = Sunday
  const isWeekend = day === 0 || day === 6 || (day === 5 && hour >= 17);
  const scores: Record<string, number> = {};
  const pNorm = Math.hypot(...moodProfile);
  for (const t of THEMES) {
    const tv: number[] = vocab.map((m) => (t.moods.includes(m) ? 1 : 0));
    const denom = Math.hypot(...tv) * pNorm;
    const moodFit = denom ? tv.reduce((s, x, i) => s + x * moodProfile[i], 0) / denom : 0;
    scores[t.name] = +((t.hours.includes(hour) ? 1 : 0) + (isWeekend ? t.weekendBonus : 0) + moodFit).toFixed(3);
  }

  const weekday = now.toLocaleDateString("en-US", { weekday: "long" });
  const time = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  let theme: Theme;
  let why: string;
  if (requested) {
    theme = THEMES.find((t) => t.name === requested) ?? THEMES[0];
    why = `Theme "${theme.name}" was requested.`;
  } else {
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
    theme = THEMES.find((t) => t.name === best)!;
    why =
      `Chose "${theme.name}" (score ${scores[best]}) for ${weekday} at ${time}: ` +
      (theme.hours.includes(hour) ? "it fits this hour" : "it doesn't fit the hour, but") +
      " and its moods overlap what's been playing lately.";
  }

  const artistCounts = new Map<string, number>();
  for (const t of recent) artistCounts.set(h.artists[t], (artistCounts.get(h.artists[t]) ?? 0) + 1);
  const recentArtists = [...artistCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const recentMoods = vocab
    .map((m, i) => [m, +moodProfile[i].toFixed(3)] as [string, number])
    .filter(([, w]) => w > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  // Seeds: recent crate tracks sharing a mood with the theme, topped up
  // with the fly's highest-taste on-theme tracks.
  const onTheme = (ci: number) => crate.moods[ci].some((m) => theme.moods.includes(m));
  const seeds: number[] = [];
  for (const t of recent) {
    const ci = crateIndex.get(overlap.catalogRow[t]);
    if (ci !== undefined && onTheme(ci) && !seeds.includes(ci)) seeds.push(ci);
    if (seeds.length === N_SEEDS) break;
  }
  if (seeds.length < N_SEEDS) {
    const fallback = Array.from(crate.rows.keys())
      .filter((ci) => onTheme(ci) && !seeds.includes(ci))
      .sort((a, b) => crate.taste[b] - crate.taste[a]);
    seeds.push(...fallback.slice(0, N_SEEDS - seeds.length));
  }

  let rationale = why;
  if (recentArtists.length) {
    rationale += ` Recent rotation leans on ${recentArtists.slice(0, 3).map(([a]) => a).join(", ")}`;
    rationale += recentMoods.length ? `, mostly ${recentMoods.slice(0, 2).map(([m]) => m).join(", ")}.` : ".";
  }
  return { theme, hour, weekday, isWeekend, recentArtists, recentMoods, seeds, rationale, scores };
}

// -- select (select.py) -------------------------------------------------------

export type SelectConfig = {
  familiarRatio: number;
  maxPerArtist: number;
  wArc: number;
  wCoherence: number;
  wTaste: number;
  wTheme: number;
  wTransition: number;
  hardTransitions: boolean;
};

export const DEFAULT_CONFIG: SelectConfig = {
  familiarRatio: 0.6,
  maxPerArtist: 2,
  wArc: 0.25,
  wCoherence: 0.25,
  wTaste: 0.2,
  wTheme: 0.3,
  wTransition: 0.15,
  hardTransitions: false,
};

const BAND_WIDENING = [1.0, 1.5, 2.0];

export type Pick = {
  crateIndex: number;
  row: number;
  name: string;
  artist: string;
  trackId: string;
  startS: number;
  durationS: number;
  tMid: number;
  phase: string;
  target: number;
  energy: number;
  tempo: number | null;
  familiar: boolean;
  played: boolean;
  hammingToPrev: number | null;
  taste: number;
  notes: string[];
};

function rank01(x: number[]): number[] {
  if (x.length <= 1) return x.map(() => 1);
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
  const out = new Array(x.length);
  order.forEach((i, k) => (out[i] = k / (x.length - 1)));
  return out;
}

export function select(cat: Catalog, crate: Crate, brief: Brief, arc: Arc, config: SelectConfig, exclude: Set<number>): Pick[] {
  const n = crate.rows.length;
  const k2 = 2 * cat.kActive;
  const dice = (d: number) => 1 - d / k2;
  const themeMoods = new Set(brief.theme.moods);
  const themeFit = crate.moods.map((m) => {
    if (!m.length) return 0;
    const inter = m.filter((x) => themeMoods.has(x)).length;
    return inter / new Set([...m, ...themeMoods]).size;
  });

  const seedSim = new Float64Array(n);
  for (const s of brief.seeds) {
    const d = hammingTo(cat, crate.rows[s], crate.rows);
    for (let i = 0; i < n; i++) seedSim[i] += dice(d[i]);
  }
  if (brief.seeds.length) for (let i = 0; i < n; i++) seedSim[i] /= brief.seeds.length;

  const sortedDur = Array.from(crate.duration).sort((a, b) => a - b);
  const typical = sortedDur[Math.floor(n / 2)];
  const available = new Uint8Array(n).fill(1);
  exclude.forEach((i) => (available[i] = 0));
  const artistCount = new Map<string, number>();
  const picks: Pick[] = [];
  let elapsed = 0;
  let prev: number | null = null;
  let prevDist: Uint16Array | null = null;

  while (arc.seconds - elapsed > typical / 2) {
    const target = new Float64Array(n);
    const tMid = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      tMid[i] = Math.min(Math.max((elapsed + crate.duration[i] / 2) / arc.seconds, 0), 1);
      target[i] = arc.target(tMid[i]);
    }
    const familiarSoFar = picks.filter((p) => p.familiar).length;
    const wantFamiliar = familiarSoFar < config.familiarRatio * (picks.length + 1);

    const base = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const a = cat.artist[crate.rows[i]];
      base[i] = available[i] && (artistCount.get(a) ?? 0) < config.maxPerArtist ? 1 : 0;
    }

    let coherenceRaw: Float64Array;
    const transition = new Float64Array(n);
    const smooth = new Uint8Array(n).fill(1);
    if (prev === null) {
      coherenceRaw = seedSim;
      prevDist = null;
    } else {
      prevDist = hammingTo(cat, crate.rows[prev], crate.rows);
      coherenceRaw = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const prevSim = dice(prevDist[i]);
        coherenceRaw[i] = brief.seeds.length ? 0.5 * prevSim + 0.5 * seedSim[i] : prevSim;
        const shift = tempoShift(crate.tempo[prev], crate.tempo[i]);
        const step = Math.abs(crate.energy[i] - crate.energy[prev]);
        const tempoPen = shift === null ? 0.5 : shift / MAX_TEMPO_SHIFT;
        transition[i] = Math.min(tempoPen, 2) / 2 + Math.min(step / MAX_ENERGY_STEP, 2) / 2;
        if (config.hardTransitions) {
          const bad =
            step > HARD_ENERGY_STEP ||
            (shift !== null && shift > HARD_TEMPO_SHIFT) ||
            (shift !== null && shift > COMBINED_TEMPO_SHIFT && step > COMBINED_ENERGY_STEP);
          smooth[i] = bad ? 0 : 1;
        }
      }
    }

    let eligible: number[] | null = null;
    const notes: string[] = [];
    for (const mult of BAND_WIDENING) {
      const width = arc.tolerance * mult;
      const inBand = (i: number) => base[i] === 1 && Math.abs(crate.energy[i] - target[i]) <= width;
      const kind = (i: number) => (crate.familiar[i] === 1) === wantFamiliar;
      const tiers: [(i: number) => boolean, string | null][] = [
        [(i) => inBand(i) && kind(i) && smooth[i] === 1, null],
        [(i) => inBand(i) && smooth[i] === 1, "ratio relaxed: no in-band track of the wanted kind"],
        [(i) => inBand(i) && kind(i), "transition rule relaxed: every in-band track would jar"],
        [inBand, "ratio and transition rule relaxed"],
      ];
      for (const [test, note] of tiers) {
        const idx: number[] = [];
        for (let i = 0; i < n; i++) if (test(i)) idx.push(i);
        if (idx.length) {
          eligible = idx;
          if (note) notes.push(note);
          break;
        }
      }
      if (eligible) {
        if (mult > 1) notes.push(`band widened to ±${width.toFixed(2)}`);
        break;
      }
    }
    if (!eligible) {
      eligible = [];
      for (let i = 0; i < n; i++) if (base[i]) eligible.push(i);
      notes.push("no track within the widest band; arc constraint broken here");
    }
    if (!eligible.length) break;

    const coherence = rank01(eligible.map((i) => coherenceRaw[i]));
    let best = -1;
    let bestTotal = -Infinity;
    eligible.forEach((i, k) => {
      const arcFit = Math.min(Math.max(1 - Math.abs(crate.energy[i] - target[i]) / arc.tolerance, 0), 1);
      const total =
        config.wArc * arcFit +
        config.wCoherence * coherence[k] +
        config.wTaste * crate.taste[i] +
        config.wTheme * themeFit[i] -
        config.wTransition * transition[i];
      if (total > bestTotal) [best, bestTotal] = [i, total];
    });

    const row = crate.rows[best];
    picks.push({
      crateIndex: best,
      row,
      name: cat.name[row],
      artist: cat.artist[row],
      trackId: cat.id[row],
      startS: elapsed,
      durationS: crate.duration[best],
      tMid: tMid[best],
      phase: arc.phase(tMid[best]),
      target: target[best],
      energy: crate.energy[best],
      tempo: crate.tempo[best],
      familiar: crate.familiar[best] === 1,
      played: crate.played[best] === 1,
      hammingToPrev: prevDist ? prevDist[best] : null,
      taste: crate.taste[best],
      notes,
    });
    available[best] = 0;
    artistCount.set(cat.artist[row], (artistCount.get(cat.artist[row]) ?? 0) + 1);
    elapsed += crate.duration[best];
    prev = best;
  }
  return picks;
}

// -- critique (critique.py) ---------------------------------------------------

const MIN_PEAK_LIFT = 0.1;
const LENGTH_TOLERANCE = 0.15;

export type Issue = { slot: number; kind: string; detail: string };
export type Verdict = { passed: boolean; issues: Issue[]; summary: string; exclude: number[] };

export function critique(picks: Pick[], arc: Arc, maxPerArtist: number): Verdict {
  if (!picks.length) return { passed: false, issues: [{ slot: 0, kind: "empty", detail: "Select produced no tracks" }], summary: "Empty set.", exclude: [] };
  const issues: Issue[] = [];
  picks.forEach((p, i) => {
    if (Math.abs(p.energy - p.target) > arc.tolerance)
      issues.push({ slot: i, kind: "off arc", detail: `energy ${p.energy.toFixed(2)} vs target ${p.target.toFixed(2)} in the ${p.phase}` });
  });
  for (let i = 1; i < picks.length; i++) {
    const a = picks[i - 1];
    const b = picks[i];
    const reason = jarringReason(a.tempo, a.energy, b.tempo, b.energy);
    if (reason) {
      const tempoTxt = tempoShift(a.tempo, b.tempo) !== null ? `${a.tempo!.toFixed(0)}→${b.tempo!.toFixed(0)} BPM` : "tempo unmeasured";
      issues.push({ slot: i, kind: "jarring", detail: `${reason} (${tempoTxt}, energy ${a.energy.toFixed(2)}→${b.energy.toFixed(2)})` });
    }
  }
  const peakE = picks.filter((p) => p.phase === "peak").map((p) => p.energy);
  const peak = peakE.length ? peakE.reduce((s, x) => s + x, 0) / peakE.length : NaN;
  const edges = Math.max(picks[0].energy, picks[picks.length - 1].energy);
  if (Number.isNaN(peak) || peak - edges < MIN_PEAK_LIFT)
    issues.push({ slot: -1, kind: "flat shape", detail: `peak averages ${peak.toFixed(2)} against edges up to ${edges.toFixed(2)}; needs a lift of ${MIN_PEAK_LIFT}` });
  const seconds = picks.reduce((s, p) => s + p.durationS, 0);
  if (Math.abs(seconds / arc.seconds - 1) > LENGTH_TOLERANCE)
    issues.push({ slot: -1, kind: "length", detail: `runs ${(seconds / 60).toFixed(1)} min for a ${arc.minutes}-min brief` });
  const counts = new Map<string, number>();
  picks.forEach((p) => counts.set(p.artist, (counts.get(p.artist) ?? 0) + 1));
  counts.forEach((c, a) => {
    if (c > maxPerArtist) issues.push({ slot: -1, kind: "artist cap", detail: `${a} appears ${c} times` });
  });

  const exclude = [...new Set(issues.filter((i) => i.slot >= 0).map((i) => picks[i.slot].crateIndex))];
  const passed = issues.length === 0;
  let summary: string;
  if (passed) {
    summary = `Pass: ${picks.length} tracks, ${(seconds / 60).toFixed(1)} min, every track inside the ±${arc.tolerance.toFixed(2)} band, no jarring transitions, peak lifts ${(peak - edges).toFixed(2)} above the edges.`;
  } else {
    const kinds = new Map<string, number>();
    issues.forEach((i) => kinds.set(i.kind, (kinds.get(i.kind) ?? 0) + 1));
    summary = "Reject: " + [...kinds].map(([k, c]) => `${c} ${k}`).join(", ") + ".";
  }
  return { passed, issues, summary, exclude };
}

// -- notes (commit.py) --------------------------------------------------------

const CLOSE_HAMMING = 120;
const cap = (s: string) => s.slice(0, 1).toUpperCase() + s.slice(1);

function tempoPhrase(a: Pick, b: Pick): string {
  const shift = tempoShift(a.tempo, b.tempo);
  if (shift === null || a.tempo === null || b.tempo === null) return "tempo unmeasured on one side";
  const octave = Math.abs(b.tempo - a.tempo) / a.tempo > shift + 0.01;
  if (octave) {
    const rel = b.tempo > a.tempo ? "double" : "half";
    if (shift < 0.04) return `${b.tempo.toFixed(0)} BPM runs at ${rel} time against ${a.tempo.toFixed(0)}`;
    return `tempo moves ${a.tempo.toFixed(0)} → ${b.tempo.toFixed(0)} BPM, near ${rel} time (${pct(shift)} off)`;
  }
  if (shift < 0.04) return `tempo locks in at ~${b.tempo.toFixed(0)} BPM`;
  return `tempo ${b.tempo > a.tempo ? "pushes" : "drops"} ${a.tempo.toFixed(0)} → ${b.tempo.toFixed(0)} BPM`;
}

function energyPhrase(a: Pick, b: Pick): string {
  const step = b.energy - a.energy;
  if (step > 0.05) return `energy lifts ${a.energy.toFixed(2)} → ${b.energy.toFixed(2)}`;
  if (step < -0.05) return `energy eases ${a.energy.toFixed(2)} → ${b.energy.toFixed(2)}`;
  return `energy holds at ${b.energy.toFixed(2)}`;
}

function flyPhrase(b: Pick): string {
  const parts: string[] = [];
  if (b.hammingToPrev !== null && b.hammingToPrev < CLOSE_HAMMING) parts.push(`fly-brain neighbour of the last track (Hamming ${b.hammingToPrev})`);
  if (b.taste >= 0.5) parts.push(`fly taste score in the top ${Math.max(1, Math.round((1 - b.taste) * 100))}% of the crate`);
  else parts.push(`a taste risk: the fly ranks it in the bottom ${Math.max(1, Math.round(b.taste * 100))}%`);
  parts.push(b.familiar ? "in your current rotation" : b.played ? "back from your vault" : "new to you");
  return parts.join("; ");
}

export function linerNote(prev: Pick | null, p: Pick): string {
  if (!prev) {
    const tempo = p.tempo !== null ? `${p.tempo.toFixed(0)} BPM` : "tempo n/a";
    return `Opens at energy ${p.energy.toFixed(2)} (target ${p.target.toFixed(2)}), ${tempo}. ${cap(flyPhrase(p))}.`;
  }
  const into = p.phase !== prev.phase ? `, into the ${p.phase}` : "";
  return `${cap(tempoPhrase(prev, p))} while ${energyPhrase(prev, p)}${into}. ${cap(flyPhrase(p))}.`;
}

// -- the agent (agent.py) -----------------------------------------------------

export type Attempt = { picks: Pick[]; verdict: Verdict; config: SelectConfig };
export type DJRun = { brief: Brief; arc: Arc; attempts: Attempt[]; ms: number };

export function runDJ(
  cat: Catalog,
  crate: Crate,
  h: History,
  overlap: Overlap,
  opts: { theme: string | null; minutes: number; now?: Date },
): DJRun {
  const started = performance.now();
  const brief = buildBrief(cat, crate, h, overlap, opts.now ?? new Date(), opts.theme);
  const arc = new Arc(opts.minutes, brief.theme.floor, brief.theme.ceiling);
  const attempts: Attempt[] = [];
  let config = DEFAULT_CONFIG;
  let picks = select(cat, crate, brief, arc, config, new Set());
  let verdict = critique(picks, arc, config.maxPerArtist);
  attempts.push({ picks, verdict, config });
  // Critique may send the set back to Select once, with the transition
  // rules as a hard filter and double weight, and the offenders excluded.
  if (!verdict.passed) {
    config = { ...config, wTransition: config.wTransition * 2, hardTransitions: true };
    picks = select(cat, crate, brief, arc, config, new Set(verdict.exclude));
    verdict = critique(picks, arc, config.maxPerArtist);
    attempts.push({ picks, verdict, config });
  }
  return { brief, arc, attempts, ms: performance.now() - started };
}
