"use client";

import { useMemo, useState } from "react";
import type { Catalog } from "@/lib/catalog";
import { buildCrate, type DJRun, linerNote, runDJ, THEMES } from "@/lib/dj";
import type { MushroomBody, Overlap } from "@/lib/fly";
import type { History } from "@/lib/types";
import { ArcChart } from "./Charts";

type Props = { catalog: Catalog; history: History; overlap: Overlap; mb: MushroomBody };

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const title = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());

export default function DJSection({ catalog, history, overlap, mb }: Props) {
  const [theme, setTheme] = useState<string>("");
  const [minutes, setMinutes] = useState(45);
  const [run, setRun] = useState<DJRun | null>(null);
  const [busy, setBusy] = useState(false);

  const crate = useMemo(() => buildCrate(catalog, history, overlap, mb), [catalog, history, overlap, mb]);

  const build = () => {
    setBusy(true);
    // Let the button repaint before the (synchronous) run.
    setTimeout(() => {
      setRun(runDJ(catalog, crate, history, overlap, { theme: theme || null, minutes }));
      setBusy(false);
    }, 20);
  };

  const final = run?.attempts[run.attempts.length - 1];
  return (
    <section className="block" aria-labelledby="dj-h">
      <h2 id="dj-h">A DJ set, planned and self-critiqued</h2>
      <p className="sub">
        The DJ agent reads the clock and your recent plays to pick a theme, lays down an energy arc (opener, build, peak,
        comedown) as a hard constraint, fills it track by track from the {crate.rows.length.toLocaleString()} tracks with
        measured audio, scoring fly-brain coherence and your trained mushroom body&apos;s taste, then critiques the whole
        set. A set that fails critique goes back for one revision.
      </p>
      <div className="controls">
        <label>
          Theme
          <select value={theme} onChange={(e) => setTheme(e.target.value)}>
            <option value="">Auto: read the room</option>
            {THEMES.map((t) => (
              <option key={t.name} value={t.name}>
                {title(t.name)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Length
          <select value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
            {[30, 45, 60].map((m) => (
              <option key={m} value={m}>
                {m} minutes
              </option>
            ))}
          </select>
        </label>
        <button className="btn" onClick={build} disabled={busy}>
          {busy ? "Planning…" : run ? "Build another set" : "Build a set"}
        </button>
      </div>

      {run && final && (
        <div className="grid2">
          <div className="card">
            <h3>Selector DJ · {title(run.brief.theme.name)}</h3>
            <p className="note">
              {run.brief.theme.description} Planned in {Math.round(run.ms)} ms, in your browser.
            </p>
            <p style={{ fontSize: 14, margin: "0 0 12px" }}>
              <b>Brief.</b> {run.brief.rationale}
            </p>
            <ol className="verdicts" aria-label="Critique passes">
              {run.attempts.map((a, i) => (
                <li key={i}>
                  <span className={`icon ${a.verdict.passed ? "pass" : "fail"}`} aria-hidden="true">
                    {a.verdict.passed ? "✓" : "✕"}
                  </span>
                  <div>
                    <b>{i === 0 ? "First draft" : "Revision"}:</b> {a.verdict.summary}
                    {!a.verdict.passed && a.verdict.issues.length > 0 && (
                      <ul>
                        {a.verdict.issues.slice(0, 4).map((is, k) => (
                          <li key={k}>
                            {is.slot >= 0 ? `Track ${is.slot + 1}: ` : ""}
                            {is.detail}
                          </li>
                        ))}
                        {i === 0 && run.attempts.length > 1 && (
                          <li>Sent back to Select with transitions as a hard rule and the offenders excluded.</li>
                        )}
                      </ul>
                    )}
                  </div>
                </li>
              ))}
            </ol>
            <ArcChart
              minutes={run.arc.minutes}
              target={(t) => run.arc.target(t)}
              tolerance={run.arc.tolerance}
              picks={final.picks.map((p) => ({ tMid: p.tMid, energy: p.energy, target: p.target, name: p.name, artist: p.artist, phase: p.phase }))}
            />
          </div>
          <div className="card">
            <h3>Tracklist and liner notes</h3>
            <p className="note">Tempo and energy are measured from 30-second previews, not predicted.</p>
            <ol className="setlist">
              {final.picks.map((p, i) => (
                <li key={p.trackId}>
                  <span className="clock">{clock(p.startS)}</span>
                  <span className="title">
                    <a href={`https://open.spotify.com/track/${p.trackId}`} target="_blank" rel="noreferrer">
                      {p.name}
                    </a>
                    <span className="meta" style={{ fontWeight: 400 }}>
                      {" "}
                      · {p.artist}
                    </span>
                    <div className="meta" style={{ fontWeight: 400 }}>
                      {p.phase} · {p.tempo !== null ? `${Math.round(p.tempo)} BPM` : "tempo n/a"} · energy {p.energy.toFixed(2)}
                    </div>
                  </span>
                  <span className="liner">{linerNote(i ? final.picks[i - 1] : null, p)}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </section>
  );
}
