"use client";

import { useEffect, useRef, useState } from "react";
import { fitCanvas, type Palette } from "./palette";

type Props = {
  cells: ArrayLike<number>; // the chosen track's active Kenyon cells
  wApproach: Float64Array;
  wAvoid: Float64Array;
  // The last lesson: which MBON it hit, which cells, and each touched
  // synapse's weight before it, so the thinning can be animated.
  lesson: { mbon: "approach" | "avoid"; before: Map<number, number>; progress: number } | null;
  palette: Palette;
};

// KC -> MBON synapses of the chosen track: its active Kenyon cells along
// the top, the approach and avoid output neurons below, line width
// proportional to synaptic weight (naive weight 1.0).
export default function Synapses({ cells, wApproach, wAvoid, lesson, palette }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const height = 150;
  useEffect(() => {
    if (!width) return;
    const ctx = fitCanvas(ref.current!, width, height);
    const n = cells.length;
    const perRow = width < 480 ? Math.ceil(n / 2) : n;
    const step = (width - 24) / perRow;
    const kc = (s: number): [number, number] => [12 + ((s % perRow) + 0.5) * step, 10 + Math.floor(s / perRow) * 8];
    const mbon = { approach: [width * 0.3, height - 22], avoid: [width * 0.7, height - 22] } as const;
    const weight = (which: "approach" | "avoid", i: number) => {
      const now = (which === "approach" ? wApproach : wAvoid)[i];
      if (lesson && lesson.mbon === which && lesson.before.has(i)) {
        const b = lesson.before.get(i)!;
        return b + (now - b) * lesson.progress;
      }
      return now;
    };
    for (const which of ["approach", "avoid"] as const) {
      const [mx, my] = mbon[which];
      for (let s = 0; s < n; s++) {
        const i = cells[s];
        const hit = lesson?.mbon === which && lesson.before.has(i);
        const [x, y] = kc(s);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(mx, my - 12);
        ctx.lineWidth = 2.4 * weight(which, i);
        ctx.strokeStyle = hit ? palette.depressed : which === "approach" ? palette.approach : palette.avoid;
        ctx.globalAlpha = hit ? 0.75 : 0.28;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = palette.active;
    for (let s = 0; s < n; s++) {
      const [x, y] = kc(s);
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
    for (const which of ["approach", "avoid"] as const) {
      const [mx, my] = mbon[which];
      ctx.beginPath();
      ctx.arc(mx, my, 11, 0, Math.PI * 2);
      ctx.fillStyle = which === "approach" ? palette.approach : palette.avoid;
      ctx.fill();
    }
  }, [width, cells, wApproach, wAvoid, lesson, palette]);

  return (
    <div ref={wrap} className="synapses">
      <canvas
        ref={ref}
        role="img"
        aria-label={`${cells.length} Kenyon cells wired to the approach and avoid output neurons; line width is synaptic weight.`}
      />
      <div className="syn-legend" aria-hidden="true">
        <span style={{ left: "30%" }}>approach MBON</span>
        <span style={{ left: "70%" }}>avoid MBON</span>
      </div>
    </div>
  );
}
