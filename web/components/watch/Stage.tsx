"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Circuit, Firing } from "@/lib/circuit";
import { fitCanvas, GRID_COLS, type Palette } from "./palette";
import { clamp01, ease } from "./timeline";

type Props = { circuit: Circuit; firing: Firing; kActive: number; beat: number; u: number; palette: Palette };

const PAD = 12;
const PN_Y = 16;
const GRID_TOP = 54;

// Beats 2-6 on one canvas: the projection-neuron row on top, the 2,597
// Kenyon cells as a grid below, and between them only the wiring leaving
// the neurons this track actually drives.
export default function Stage({ circuit, firing, kActive, beat, u, palette }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // Outgoing edges per PN channel, bucketed by synapse weight so each
  // bucket strokes as one path.
  const edges = useMemo(() => {
    const cols: { kc: number[]; w: number[] }[] = Array.from({ length: circuit.dIn }, () => ({ kc: [], w: [] }));
    let maxW = 0;
    for (let i = 0; i < circuit.nKc; i++) {
      for (let jj = circuit.indptr[i]; jj < circuit.indptr[i + 1]; jj++) {
        const c = cols[circuit.indices[jj]];
        c.kc.push(i);
        c.w.push(circuit.data[jj]);
        if (circuit.data[jj] > maxW) maxW = circuit.data[jj];
      }
    }
    return { cols, maxW };
  }, [circuit]);

  // Rank of each cell by drive, strongest first; among exact ties the
  // winners go first, so the shrinking survivor count ends on exactly them.
  const rank = useMemo(() => {
    const order = Array.from({ length: circuit.nKc }, (_, i) => i).sort(
      (a, b) => firing.drive[b] - firing.drive[a] || firing.isWinner[b] - firing.isWinner[a],
    );
    const r = new Uint16Array(circuit.nKc);
    order.forEach((i, k) => (r[i] = k));
    return r;
  }, [circuit, firing]);

  const rows = Math.ceil(circuit.nKc / GRID_COLS);
  const pitch = width ? Math.min(9, (width - 2 * PAD) / GRID_COLS) : 6;
  const height = GRID_TOP + rows * pitch + PAD;

  useEffect(() => {
    if (!width || !canvas.current) return;
    const ctx = fitCanvas(canvas.current, width, height);
    const gx0 = (width - GRID_COLS * pitch) / 2;
    const cell = (i: number): [number, number] => [gx0 + (i % GRID_COLS) * pitch + pitch / 2, GRID_TOP + Math.floor(i / GRID_COLS) * pitch + pitch / 2];
    const pnX = (j: number) => PAD + ((j + 0.5) * (width - 2 * PAD)) / circuit.dIn;
    const pnR = Math.max(3, Math.min(7, (width - 2 * PAD) / circuit.dIn / 2.6));
    let maxPn = 0;
    for (const v of firing.pn) if (v > maxPn) maxPn = v;
    const level = (j: number) => {
      const l = maxPn ? firing.pn[j] / maxPn : 0;
      if (beat < 1) return 0;
      if (beat === 1) return l * clamp01(u * 1.6 - (j / circuit.dIn) * 0.6);
      return l;
    };

    // Edges: swept in top to bottom on beat 2, fading on beat 3.
    if (beat === 2 || beat === 3) {
      const reach = beat === 2 ? ease(u) * 1.05 : 1;
      const fade = beat === 2 ? 1 : 1 - u;
      ctx.strokeStyle = palette.active;
      ctx.lineWidth = 0.6;
      for (let j = 0; j < circuit.dIn; j++) {
        const l = level(j);
        if (l <= 0) continue;
        const { kc, w } = edges.cols[j];
        for (let bucket = 0; bucket < 3; bucket++) {
          ctx.beginPath();
          for (let e = 0; e < kc.length; e++) {
            const b = Math.min(2, Math.floor((3 * w[e]) / (edges.maxW + 1e-9)));
            if (b !== bucket) continue;
            if (Math.floor(kc[e] / GRID_COLS) / rows > reach) continue;
            const [x, y] = cell(kc[e]);
            ctx.moveTo(pnX(j), PN_Y + pnR);
            ctx.lineTo(x, y);
          }
          ctx.globalAlpha = fade * (0.05 + 0.1 * bucket) * (0.4 + 0.6 * l);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }

    // Projection neurons.
    for (let j = 0; j < circuit.dIn; j++) {
      const l = level(j);
      ctx.beginPath();
      ctx.arc(pnX(j), PN_Y, pnR, 0, Math.PI * 2);
      ctx.fillStyle = palette.surface;
      ctx.fill();
      if (l > 0) {
        ctx.globalAlpha = 0.2 + 0.8 * l;
        ctx.fillStyle = palette.active;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      ctx.lineWidth = 1;
      ctx.strokeStyle = l > 0 ? palette.active : palette.inhibited;
      ctx.stroke();
    }

    // Kenyon cells.
    const k = kActive;
    const alive = beat === 4 ? Math.round(firing.driven - (firing.driven - k) * ease(u)) : beat > 4 ? k : firing.driven;
    const shrink = beat === 5 ? 1 - 0.7 * ease(u) : 1;
    const size = pitch * 0.72 * shrink;
    for (let i = 0; i < circuit.nKc; i++) {
      let [x, y] = cell(i);
      if (shrink !== 1) {
        x = gx0 + (x - gx0) * shrink;
        y = GRID_TOP + (y - GRID_TOP) * shrink;
      }
      const d = firing.drive[i];
      let color = palette.idle;
      let alpha = 1;
      let scale = 1;
      if (beat === 3 && d > 0) {
        color = palette.active;
        alpha = (0.25 + 0.75 * (d / firing.maxDrive)) * ease(u);
      } else if (beat === 4 && d > 0) {
        if (rank[i] < alive) {
          color = palette.active;
          alpha = 0.25 + 0.75 * (d / firing.maxDrive);
        } else {
          // Silenced cells shrink as well as grey out, so inhibited never
          // rests on colour alone.
          color = palette.inhibited;
          scale = 0.55;
        }
      } else if (beat === 5) {
        if (firing.isWinner[i]) color = palette.active;
        else {
          color = d > 0 ? palette.inhibited : palette.idle;
          scale = d > 0 ? 0.55 : 1;
          alpha = 1 - ease(u);
        }
      }
      if (alpha <= 0.01) continue;
      if (beat === 3 && d > 0 && alpha < 1) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = palette.idle;
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      }
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      const sz = size * scale;
      ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
    }
    ctx.globalAlpha = 1;
  }, [width, height, pitch, rows, beat, u, circuit, firing, kActive, edges, rank, palette]);

  const alive =
    beat === 4 ? Math.round(firing.driven - (firing.driven - kActive) * ease(u)) : beat > 4 ? kActive : beat === 3 ? Math.round(firing.driven * ease(u)) : 0;
  return (
    <div ref={wrap} className="stage">
      <p className="counter" aria-live="off">
        {beat >= 3 ? (
          <>
            Kenyon cells <span className="num">{circuit.nKc.toLocaleString()}</span> →{" "}
            <span className="num">{alive.toLocaleString()}</span> active{" "}
            <span className="pct">({((alive / circuit.nKc) * 100).toFixed(1)}%)</span>
          </>
        ) : (
          <>
            {circuit.dIn} projection-neuron channels above, {circuit.nKc.toLocaleString()} Kenyon cells below
          </>
        )}
      </p>
      <canvas
        ref={canvas}
        role="img"
        aria-label={`${circuit.dIn} projection neuron channels wired to ${circuit.nKc.toLocaleString()} Kenyon cells; ${alive.toLocaleString()} currently active.`}
      />
    </div>
  );
}
