"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CardSlide, IntroSlide } from "@/components/rewind/Slides";
import { PUBLIC_SOURCE, type Report, type ReportIndex, type ReportSource, loadIndex, loadReport } from "@/lib/rewind";

const SWIPE_PX = 48;

export default function Story({ source = PUBLIC_SOURCE }: { source?: ReportSource }) {
  const [index, setIndex] = useState<ReportIndex | null>(null);
  const [windowId, setWindowId] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [slide, setSlide] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  // `?slide=N` deep-links one card, honoured on the first report load only.
  const initialSlide = useRef<number | null>(null);

  useEffect(() => {
    loadIndex(source)
      .then((idx) => {
        setIndex(idx);
        const params = new URLSearchParams(window.location.search);
        const fromUrl = params.get("window");
        initialSlide.current = Number(params.get("slide")) || 0;
        setWindowId(idx.windows.some((w) => w.id === fromUrl) ? fromUrl : (idx.windows[0]?.id ?? null));
      })
      .catch((e: Error) => setError(e.message));
  }, [source]);

  useEffect(() => {
    if (!windowId) return;
    let live = true;
    setReport(null);
    loadReport(windowId, source)
      .then((r) => {
        if (!live) return;
        setReport(r);
        setSlide(Math.max(0, Math.min(r.cards.length, initialSlide.current ?? 0)));
        initialSlide.current = null;
        const url = new URL(window.location.href);
        url.searchParams.set("window", windowId);
        window.history.replaceState(null, "", url);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [windowId, source]);

  const count = report ? report.cards.length + 1 : 0; // + intro
  const go = useCallback((d: number) => setSlide((s) => Math.max(0, Math.min(count - 1, s + d))), [count]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("input, textarea, select")) return;
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  // Tap the left third to go back, anywhere else to go forward; a horizontal
  // swipe goes either way. Presses that start on a control are left alone.
  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as Element).closest("button, a")) return;
    down.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const start = down.current;
    down.current = null;
    if (!start || !stage.current) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (Math.abs(dx) > SWIPE_PX && Math.abs(dx) > Math.abs(dy)) {
      go(dx < 0 ? 1 : -1);
    } else if (Math.abs(dx) < 10 && Math.abs(dy) < 10) {
      const box = stage.current.getBoundingClientRect();
      go(e.clientX - box.left < box.width / 3 ? -1 : 1);
    }
  };

  if (error) {
    return (
      <p className="story-error" role="alert">
        Couldn&apos;t load the report: {error}
      </p>
    );
  }

  return (
    <div className="story">
      {index && index.windows.length > 1 && (
        <div className="story-windows" role="group" aria-label="Report window">
          {index.windows.map((w) => (
            <button key={w.id} type="button" className="chip" aria-pressed={w.id === windowId} onClick={() => setWindowId(w.id)}>
              {w.label}
            </button>
          ))}
        </div>
      )}

      <div
        ref={stage}
        className="story-stage"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        aria-roledescription="story"
      >
        <div className="story-progress" aria-hidden="true">
          {Array.from({ length: count }, (_, i) => (
            <span key={i} className={i <= slide ? "seg on" : "seg"} />
          ))}
        </div>

        {!report ? (
          <div className="slide-body">
            <p className="slide-sub">Loading…</p>
          </div>
        ) : (
          <div key={`${report.window.id}-${slide}`} className="slide" aria-live="polite">
            {slide === 0 ? <IntroSlide report={report} /> : <CardSlide card={report.cards[slide - 1]} active />}
          </div>
        )}
      </div>

      <div className="story-nav">
        <button type="button" className="btn ghost" onClick={() => go(-1)} disabled={slide === 0}>
          ← Back
        </button>
        <span className="story-count">
          {report ? `${slide + 1} / ${count}` : ""}
        </span>
        <button type="button" className="btn" onClick={() => go(1)} disabled={!report || slide >= count - 1}>
          Next →
        </button>
      </div>
    </div>
  );
}
