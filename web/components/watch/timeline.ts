// The eight beats of `/watch`, in seconds. About 30 s end to end.

export type Beat = { key: string; title: string; alias: string; dur: number };

export const BEATS: Beat[] = [
  { key: "stimulus", title: "The stimulus", alias: "input vector", dur: 3 },
  { key: "pn", title: "Projection neurons", alias: "input layer", dur: 2.5 },
  { key: "fanout", title: "The fan-out", alias: "sparse projection", dur: 3.5 },
  { key: "kc", title: "Kenyon cells fire", alias: "hash bits, before sparsification", dur: 2.5 },
  { key: "wta", title: "Winner-take-all", alias: "top-k sparsification", dur: 4.5 },
  { key: "tag", title: "The fingerprint", alias: "binary hash code", dur: 2 },
  { key: "match", title: "The match", alias: "Hamming distance / similarity", dur: 6.5 },
  { key: "verdict", title: "The verdict", alias: "learned linear readout", dur: 6 },
];

export const STARTS = BEATS.reduce<number[]>((acc, b, i) => [...acc, i ? acc[i - 1] + BEATS[i - 1].dur : 0], []);
export const TOTAL = STARTS[STARTS.length - 1] + BEATS[BEATS.length - 1].dur;
export const MATCH = 6;
export const VERDICT = 7;

export function beatAt(t: number): number {
  for (let i = BEATS.length - 1; i >= 0; i--) if (t >= STARTS[i]) return i;
  return 0;
}

// Beat index, progress through it (0..1) and seconds into it.
export function position(t: number): { beat: number; u: number; local: number } {
  const beat = beatAt(Math.min(t, TOTAL - 1e-9));
  const local = t - STARTS[beat];
  return { beat, u: Math.min(Math.max(local / BEATS[beat].dur, 0), 1), local };
}

export const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);
export const ease = (x: number) => {
  x = clamp01(x);
  return x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2;
};
