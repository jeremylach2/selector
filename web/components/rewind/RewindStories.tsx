"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import Story from "@/components/rewind/Story";
import { loadCatalog } from "@/lib/catalog";
import { PUBLIC_SOURCE, type LocalReports, type ReportSource } from "@/lib/rewind";
import { buildVisitorReports, loadRewindAssets } from "@/lib/rewind-visitor";
import { getSessionHistory } from "@/lib/session";

type Mode = "own" | "sample";

// The invented listener's precomputed reports, or, when the visitor already
// dropped their own export on the landing page this session, a Rewind built
// from it right here in the tab.
export default function RewindStories() {
  const [own, setOwn] = useState<LocalReports | null>(null);
  const [status, setStatus] = useState<"none" | "building" | "ready" | "error">("none");
  const [note, setNote] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("sample");

  useEffect(() => {
    const session = getSessionHistory();
    if (session?.source !== "own") return;
    const { history } = session;
    setStatus("building");
    setMode("own");
    let live = true;
    // Without the fly catalog the report still builds, minus the fly and
    // release-year cards.
    loadCatalog()
      .then(async (cat) => ({ cat, assets: await loadRewindAssets(cat) }))
      .catch(() => {
        if (live) setNote("The fly catalog didn't load, so the taste, gem and listening-age cards are missing.");
        return { cat: null, assets: null };
      })
      .then(({ cat, assets }) => {
        if (!live) return;
        setOwn(buildVisitorReports(history, cat, assets));
        setStatus("ready");
      })
      .catch((e: Error) => {
        if (!live) return;
        setNote(`Couldn't build your Rewind: ${e.message}`);
        setStatus("error");
        setMode("sample");
      });
    return () => {
      live = false;
    };
  }, []);

  const source: ReportSource | null = useMemo(() => (own ? { local: own } : null), [own]);
  const showOwn = mode === "own" && source;

  return (
    <>
      <div className="rewind-whose">
        {status === "none" ? (
          <p className="rewind-own-cta">
            Want yours? <Link href="/">Drop your export on the home page</Link>, then come back: your Rewind is built right here
            in the tab.
          </p>
        ) : (
          <div className="story-windows" role="group" aria-label="Whose Rewind">
            <button
              type="button"
              className="chip"
              aria-pressed={mode === "own"}
              disabled={!source}
              onClick={() => setMode("own")}
            >
              {status === "building" ? "Building yours…" : "Your export"}
            </button>
            <button type="button" className="chip" aria-pressed={mode === "sample"} onClick={() => setMode("sample")}>
              Invented listener
            </button>
          </div>
        )}
        {note && (
          <p className="slide-cov" role="status">
            {note}
          </p>
        )}
      </div>
      {showOwn ? <Story key="own" source={source} /> : status !== "building" && <Story key="sample" source={PUBLIC_SOURCE} />}
    </>
  );
}
