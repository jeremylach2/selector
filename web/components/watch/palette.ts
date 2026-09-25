"use client";

import { useEffect, useState } from "react";

// Every colour on the canvases carries one meaning, read from the page's
// theme tokens so light and dark each get their own validated steps:
// active cells (s1), depressed synapses (s2), the approach MBON (s3),
// cells shared with the query (s4), the avoid MBON (s5).
export type Palette = {
  active: string;
  depressed: string;
  approach: string;
  shared: string;
  avoid: string;
  idle: string; // a cell that never fired
  inhibited: string; // a cell that fired and was silenced
  ink: string;
  muted: string;
  surface: string;
};

function read(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    active: v("--s1"),
    depressed: v("--s2"),
    approach: v("--s3"),
    shared: v("--s4"),
    avoid: v("--s5"),
    idle: v("--grid"),
    inhibited: v("--axis"),
    ink: v("--ink"),
    muted: v("--muted"),
    surface: v("--surface"),
  };
}

export function usePalette(): Palette | null {
  const [p, setP] = useState<Palette | null>(null);
  useEffect(() => {
    setP(read());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setP(read());
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return p;
}

export function useReducedMotion(): boolean {
  const [rm, setRm] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setRm(mq.matches);
    const on = () => setRm(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return rm;
}

// Canvas backing store sized to its CSS box at the device pixel ratio.
export function fitCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

// The Kenyon-cell grid layout, shared by the stage and every tag chip so a
// fingerprint chip is literally the stage grid, shrunk.
export const GRID_COLS = 51;
