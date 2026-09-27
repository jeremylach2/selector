"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "@/components/watch/palette";
import { monthLabel } from "@/lib/rewind";

const TOP = 8;
const ROW = 34; // px per bar slot
// A race should feel like a race whatever the window: a year's 12 months
// and all-time's ~50 both finish in 5-15 s.
const MS_PER_MONTH_MIN = 220;
const MS_PER_MONTH_MAX = 700;
const TARGET_MS = 12000;

// Series colours, assigned by final standing so the winner is always s1.
// Only the five validated series hues: bars are labelled, so a repeat five
// places apart reads fine, while extra near-identical blues did not.
const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];

type Props = {
  months: string[];
  artists: string[];
  cumulative: number[][];
  active: boolean; // only run while this slide is on screen
};

export default function ChartRace({ months, artists, cumulative, active }: Props) {
  const reduced = useReducedMotion();
  const last = months.length - 1;
  const msPerMonth = Math.min(MS_PER_MONTH_MAX, Math.max(MS_PER_MONTH_MIN, TARGET_MS / Math.max(last, 1)));
  const [t, setT] = useState(0); // fractional month index
  const [run, setRun] = useState(0); // bump to replay
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!active) return;
    if (reduced || last <= 0) {
      setT(last);
      return;
    }
    const start = performance.now();
    const step = (now: number) => {
      const next = Math.min(last, (now - start) / msPerMonth);
      setT(next);
      if (next < last) raf.current = requestAnimationFrame(step);
    };
    setT(0);
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [active, reduced, last, msPerMonth, run]);

  const color = useMemo(() => new Map(artists.map((a, i) => [a, COLORS[i % COLORS.length]])), [artists]);

  const i = Math.floor(t);
  const f = t - i;
  const lo = cumulative[i] ?? [];
  const hi = cumulative[Math.min(i + 1, last)] ?? lo;
  const values = artists.map((_, a) => lo[a] + (hi[a] - lo[a]) * f);
  const order = values.map((v, a) => [v, a] as const).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  const max = Math.max(order[0]?.[0] ?? 1, 1);
  // One extra row so a bar dropping out of the top 8 visibly slides away.
  const shown = order.slice(0, TOP + 1).filter(([v]) => v > 0);
  const done = t >= last;

  return (
    <div className="race">
      <div className="race-bars" style={{ height: TOP * ROW }} role="img" aria-label={`Artist race, ${monthLabel(months[i])}: ${shown.slice(0, 3).map(([v, a]) => `${artists[a]} ${Math.round(v)}`).join(", ")}`}>
        {shown.map(([v, a], rank) => (
          <div
            key={artists[a]}
            className="race-row"
            style={{ transform: `translateY(${rank * ROW}px)`, opacity: rank < TOP ? 1 : 0 }}
          >
            <div className="race-track">
              <div className="race-bar" style={{ width: `${(100 * v) / max}%`, background: color.get(artists[a]) }} />
              <span className="race-name">{artists[a]}</span>
            </div>
            <span className="race-val">{Math.round(v).toLocaleString()}</span>
          </div>
        ))}
      </div>
      <div className="race-foot">
        <span className="race-month">{monthLabel(months[i])}</span>
        <button
          type="button"
          className="btn ghost race-replay"
          onClick={(e) => {
            e.stopPropagation();
            setRun((r) => r + 1);
          }}
          disabled={!done || reduced}
        >
          Replay
        </button>
      </div>
    </div>
  );
}
