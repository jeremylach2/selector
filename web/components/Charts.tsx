"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import type { ArtistRow, Drift, SkipRow } from "@/lib/types";

// -- shared plumbing ------------------------------------------------------

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

type Tip = { x: number; y: number; body: ReactNode } | null;

// Tooltip positioned inside the chart wrapper, kept on-screen at the edges.
function Tooltip({ tip, width }: { tip: Tip; width: number }) {
  if (!tip) return null;
  const left = Math.min(Math.max(tip.x + 12, 0), Math.max(width - 200, 0));
  return (
    <div className="tooltip" style={{ left, top: tip.y + 12 }} role="status">
      {tip.body}
    </div>
  );
}

const fmt = new Intl.NumberFormat("en-US");
export const compact = (n: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

// -- listening clock: weekday x hour heatmap --------------------------------

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function ListeningClock({ clock, timeZone }: { clock: number[]; timeZone: string }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const max = Math.max(1, ...clock);
  const labelW = 34;
  const cw = W ? (W - labelW) / 24 : 0;
  const ch = Math.min(Math.max(cw, 14), 24);
  const top = 0;
  const H = top + 7 * ch + 22;
  const bin = (v: number) => (v === 0 ? 0 : 1 + Math.min(6, Math.floor((v / max) * 7)));

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const h = Math.floor((x - labelW) / cw);
    const d = Math.floor((y - top) / ch);
    if (h < 0 || h > 23 || d < 0 || d > 6) return setTip(null);
    const v = clock[d * 24 + h];
    setTip({
      x,
      y,
      body: (
        <>
          <b>
            {DAYS[d]} {String(h).padStart(2, "0")}:00–{String((h + 1) % 24).padStart(2, "0")}:00
          </b>
          <div className="muted">{fmt.format(v)} plays</div>
        </>
      ),
    });
  };

  // Busiest slot, for the caption.
  const peak = clock.indexOf(max);
  return (
    <div className="chart" ref={ref} style={{ position: "relative" }}>
      <p className="note">
        Busiest: {DAYS[Math.floor(peak / 24)]}s around {String(peak % 24).padStart(2, "0")}:00. Hours in {timeZone}.
      </p>
      {W > 0 && (
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`Heatmap of plays by weekday and hour. Busiest slot ${DAYS[Math.floor(peak / 24)]} ${peak % 24}:00.`}
          onPointerMove={onMove}
          onPointerLeave={() => setTip(null)}
        >
          {DAYS.map((d, di) => (
            <text key={d} className="tick" x={0} y={top + di * ch + ch / 2 + 4}>
              {d}
            </text>
          ))}
          {clock.map((v, i) => {
            const d = Math.floor(i / 24);
            const h = i % 24;
            return (
              <rect
                key={i}
                x={labelW + h * cw + 1}
                y={top + d * ch + 1}
                width={Math.max(cw - 2, 1)}
                height={ch - 2}
                rx={2}
                fill={`var(--seq-${bin(v)})`}
              />
            );
          })}
          {[0, 6, 12, 18].map((h) => (
            <text key={h} className="tick" x={labelW + h * cw + 1} y={top + 7 * ch + 15}>
              {String(h).padStart(2, "0")}:00
            </text>
          ))}
        </svg>
      )}
      <div className="scale" aria-hidden="true">
        fewer
        <span className="ramp">
          {[1, 2, 3, 4, 5, 6, 7].map((s) => (
            <i key={s} style={{ background: `var(--seq-${s})` }} />
          ))}
        </span>
        more plays
      </div>
      <Tooltip tip={tip} width={W} />
      <details>
        <summary>Show as table</summary>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Day</th>
                <th>Plays</th>
                <th>Busiest hour</th>
              </tr>
            </thead>
            <tbody>
              {DAYS.map((d, di) => {
                const row = clock.slice(di * 24, di * 24 + 24);
                const total = row.reduce((s, x) => s + x, 0);
                const bh = row.indexOf(Math.max(...row));
                return (
                  <tr key={d}>
                    <td>{d}</td>
                    <td>{fmt.format(total)}</td>
                    <td>{String(bh).padStart(2, "0")}:00</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

// -- taste drift: 100% stacked columns, artist share per period -------------

const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];

export function TasteDrift({ drift }: { drift: Drift }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const axisW = 36;
  const plotH = 200;
  const H = plotH + 24;
  const n = drift.periods.length;
  const band = W && n ? (W - axisW) / n : 0;
  const barW = Math.max(Math.min(24, band - 4), 2);
  const labelEvery = Math.max(1, Math.ceil(52 / Math.max(band, 1)));
  // "Everyone else" is usually most of every period and would flatten the
  // named artists into slivers, so the scale fits the named artists only.
  const maxStack = Math.max(0.01, ...drift.shares.map((row) => row.reduce((a, b) => a + b, 0)));
  const yMax = Math.min(1, Math.ceil(maxStack * 10) / 10);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left;
    const p = Math.floor((x - axisW) / band);
    if (p < 0 || p >= n) return setTip(null);
    setTip({
      x,
      y: e.clientY - r.top,
      body: (
        <>
          <b>{drift.periods[p]}</b> <span className="muted">· {fmt.format(drift.totals[p])} plays</span>
          <div className="muted">Top artist: {drift.leaders[p]}</div>
          <div className="muted">Everyone else: {Math.round(drift.other[p] * 100)}%</div>
          {drift.series.map((s, i) =>
            drift.shares[p][i] > 0 ? (
              <div key={s}>
                <span className="swatch" style={{ background: SERIES[i], display: "inline-block", marginRight: 6 }} />
                {s}: {Math.round(drift.shares[p][i] * 100)}%
              </div>
            ) : null,
          )}
        </>
      ),
    });
  };

  return (
    <div className="chart" ref={ref} style={{ position: "relative" }}>
      <ul className="legend">
        {drift.series.map((s, i) => (
          <li key={s}>
            <span className="swatch" style={{ background: SERIES[i] }} />
            {s}
          </li>
        ))}
      </ul>
      {W > 0 && (
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`Share of plays per ${drift.granularity} for ${drift.series.join(", ")}.`}
          onPointerMove={onMove}
          onPointerLeave={() => setTip(null)}
        >
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={axisW} x2={W} y1={plotH * (1 - f)} y2={plotH * (1 - f)} stroke={f === 0 ? "var(--axis)" : "var(--grid)"} strokeWidth={1} />
              <text className="tick" x={0} y={plotH * (1 - f) + 4}>
                {Math.round(f * yMax * 100)}%
              </text>
            </g>
          ))}
          {drift.periods.map((p, pi) => {
            const x = axisW + pi * band + (band - barW) / 2;
            const segs = drift.shares[pi].map((s, i) => ({ s, fill: SERIES[i] }));
            let y = plotH;
            const lastIdx = segs.map((g) => g.s > 0).lastIndexOf(true);
            return (
              <g key={p}>
                {segs.map((g, gi) => {
                  if (g.s <= 0) return null;
                  const h = (g.s / yMax) * plotH;
                  y -= h;
                  // 2px surface gap between stacked segments.
                  const hh = Math.max(h - (gi === 0 ? 0 : 2), 0.5);
                  const yy = y;
                  if (gi === lastIdx) {
                    const r = Math.min(4, hh / 2, barW / 2);
                    return (
                      <path
                        key={gi}
                        fill={g.fill}
                        d={`M${x},${yy + hh}V${yy + r}Q${x},${yy} ${x + r},${yy}H${x + barW - r}Q${x + barW},${yy} ${x + barW},${yy + r}V${yy + hh}Z`}
                      />
                    );
                  }
                  return <rect key={gi} x={x} y={yy} width={barW} height={hh} fill={g.fill} />;
                })}
                {pi % labelEvery === 0 && (
                  <text className="tick" x={axisW + pi * band + band / 2} y={plotH + 16} textAnchor="middle">
                    {p}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
      <Tooltip tip={tip} width={W} />
      <details>
        <summary>Show as table</summary>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Period</th>
                <th>Plays</th>
                <th>Top artist</th>
                {drift.series.map((s) => (
                  <th key={s}>{s}</th>
                ))}
                <th>Everyone else</th>
              </tr>
            </thead>
            <tbody>
              {drift.periods.map((p, pi) => (
                <tr key={p}>
                  <td>{p}</td>
                  <td>{fmt.format(drift.totals[pi])}</td>
                  <td>{drift.leaders[pi]}</td>
                  {drift.shares[pi].map((s, i) => (
                    <td key={i}>{Math.round(s * 100)}%</td>
                  ))}
                  <td>{Math.round(drift.other[pi] * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

// -- bar lists ----------------------------------------------------------------

export function TopArtists({ rows }: { rows: ArtistRow[] }) {
  const max = Math.max(1, ...rows.map((r) => r.plays));
  return (
    <ol className="hbars" aria-label="Top artists by plays">
      {rows.map((r) => (
        <li key={r.artist}>
          <div className="row-label">
            <span className="name">{r.artist}</span>
            <span className="val">
              {fmt.format(r.plays)} plays · {r.hours.toFixed(r.hours < 10 ? 1 : 0)} h
            </span>
          </div>
          <div className="track">
            <div className="fill" style={{ width: `${(r.plays / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ol>
  );
}

export function SkipOffenders({ rows }: { rows: SkipRow[] }) {
  if (!rows.length) return <p className="note">Nothing qualifies: no track with 5+ plays gets skipped 30% of the time.</p>;
  return (
    <ol className="hbars" aria-label="Most-skipped tracks">
      {rows.map((r) => (
        <li key={r.track + r.artist}>
          <div className="row-label">
            <span className="name">
              {r.track} <span className="artist">· {r.artist}</span>
            </span>
            <span className="val">
              {Math.round(r.skipRate * 100)}% of {r.plays}
            </span>
          </div>
          <div className="track">
            <div className="fill" style={{ width: `${r.skipRate * 100}%` }} />
          </div>
        </li>
      ))}
    </ol>
  );
}

// -- DJ energy arc: target curve and band vs the realised picks ---------------

export type ArcPoint = { tMid: number; energy: number; target: number; name: string; artist: string; phase: string };

export function ArcChart({
  minutes,
  target,
  tolerance,
  picks,
}: {
  minutes: number;
  target: (t: number) => number;
  tolerance: number;
  picks: ArcPoint[];
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const axisW = 30;
  const plotH = 170;
  const H = plotH + 26;
  const pw = Math.max(W - axisW - 8, 1);
  const X = (t: number) => axisW + t * pw;
  const Y = (e: number) => plotH * (1 - Math.min(Math.max(e, 0), 1));
  const ts = Array.from({ length: 81 }, (_, i) => i / 80);
  const line = ts.map((t, i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(target(t)).toFixed(1)}`).join("");
  const band =
    ts.map((t, i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(target(t) + tolerance).toFixed(1)}`).join("") +
    ts
      .slice()
      .reverse()
      .map((t) => `L${X(t).toFixed(1)},${Y(target(t) - tolerance).toFixed(1)}`)
      .join("") +
    "Z";

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left;
    let best = -1;
    let bestD = Infinity;
    picks.forEach((p, i) => {
      const d = Math.abs(X(p.tMid) - x);
      if (d < bestD) [best, bestD] = [i, d];
    });
    if (best < 0 || bestD > 30) return setTip(null);
    const p = picks[best];
    setTip({
      x,
      y: e.clientY - r.top,
      body: (
        <>
          <b>
            {best + 1}. {p.name}
          </b>
          <div className="muted">{p.artist}</div>
          <div>
            Energy {p.energy.toFixed(2)} · target {p.target.toFixed(2)} ({p.phase})
          </div>
        </>
      ),
    });
  };

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="chart" ref={ref} style={{ position: "relative" }}>
      <ul className="legend">
        <li>
          <span className="swatch" style={{ background: "var(--s1)", height: 2 }} />
          Target arc (±{tolerance} band)
        </li>
        <li>
          <span className="swatch" style={{ background: "var(--s2)", borderRadius: "50%" }} />
          Picked tracks, measured energy
        </li>
      </ul>
      {W > 0 && (
        <svg width={W} height={H} role="img" aria-label="The set's energy arc and each track's measured energy" onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={axisW} x2={W} y1={Y(v)} y2={Y(v)} stroke={v === 0 ? "var(--axis)" : "var(--grid)"} />
              <text className="tick" x={0} y={Y(v) + 4}>
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          <path d={band} fill="var(--accent-wash)" />
          <path d={line} fill="none" stroke="var(--s1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {picks.map((p, i) => (
            <circle key={i} cx={X(p.tMid)} cy={Y(p.energy)} r={5} fill="var(--s2)" stroke="var(--surface)" strokeWidth={2} />
          ))}
          {ticks.map((t) => (
            <text key={t} className="tick" x={X(t)} y={plotH + 18} textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"}>
              {Math.round(t * minutes)} min
            </text>
          ))}
        </svg>
      )}
      <Tooltip tip={tip} width={W} />
    </div>
  );
}
