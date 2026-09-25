"use client";

import { useEffect, useRef } from "react";
import { fitCanvas, GRID_COLS, type Palette } from "./palette";

type Props = {
  active: ArrayLike<number>; // this track's firing Kenyon cells
  nKc: number;
  palette: Palette;
  size?: number;
  // Cells shared with the query, and how many of them to light so far.
  shared?: ArrayLike<number>;
  sharedShown?: number;
  label: string;
};

// A fingerprint: the Kenyon-cell grid from the stage, shrunk to a chip.
export default function TagChip({ active, nKc, palette, size = 60, shared, sharedShown = 0, label }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = fitCanvas(ref.current!, size, size);
    const rows = Math.ceil(nKc / GRID_COLS);
    const pitch = size / Math.max(GRID_COLS, rows);
    const dot = Math.max(1.6, pitch * 1.3);
    ctx.fillStyle = palette.idle;
    ctx.globalAlpha = 0.6;
    ctx.fillRect(0, 0, size, size);
    ctx.globalAlpha = 1;
    const at = (i: number) => [(i % GRID_COLS) * pitch + pitch / 2 - dot / 2, Math.floor(i / GRID_COLS) * pitch + pitch / 2 - dot / 2];
    const lit = new Set<number>();
    if (shared) for (let s = 0; s < Math.min(sharedShown, shared.length); s++) lit.add(shared[s]);
    ctx.fillStyle = palette.active;
    for (let j = 0; j < active.length; j++) {
      if (lit.has(active[j])) continue;
      const [x, y] = at(active[j]);
      ctx.fillRect(x, y, dot, dot);
    }
    ctx.fillStyle = palette.shared;
    for (const i of lit) {
      const [x, y] = at(i);
      ctx.fillRect(x - 0.5, y - 0.5, dot + 1, dot + 1);
    }
  }, [active, nKc, palette, size, shared, sharedShown]);
  return <canvas ref={ref} className="chip-canvas" role="img" aria-label={label} />;
}
