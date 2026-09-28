"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Catalog, loadCatalog } from "@/lib/catalog";
import { matchHistory, trainMushroomBody } from "@/lib/fly";
import { setSessionHistory } from "@/lib/session";
import type { Dashboard, History, ParseSource, WorkerOut } from "@/lib/types";
import { compact, ListeningClock, SkipOffenders, TasteDrift, TopArtists } from "./Charts";
import DJSection from "./DJSection";
import FlySection from "./FlySection";

const SAMPLE_URL = "/sample/sample_spotify_data.zip";
// The invented sample listener lives on US Central time; a visitor's own
// export is shown in their browser's time zone.
const SAMPLE_TZ = "America/Chicago";

type Run = {
  source: "sample" | "own";
  started: number;
  firstResultMs: number | null;
  doneMs: number | null;
  progress: string;
  frac: number;
};

export default function Demo() {
  const [run, setRun] = useState<Run | null>(null);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [history, setHistory] = useState<History | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const catalogPromise = useRef<Promise<Catalog> | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const ensureCatalog = useCallback(() => {
    if (!catalogPromise.current) {
      catalogPromise.current = loadCatalog();
      catalogPromise.current.then(setCatalog, (e) => setCatalogError(String(e?.message ?? e)));
    }
  }, []);

  // Warm the fly catalog shortly after first paint, so "more like this" is
  // ready by the time the dashboard is.
  useEffect(() => {
    const id = setTimeout(ensureCatalog, 1200);
    return () => clearTimeout(id);
  }, [ensureCatalog]);

  const start = useCallback(
    (source: ParseSource, kind: Run["source"]) => {
      ensureCatalog();
      workerRef.current?.terminate();
      setError(null);
      setDashboard(null);
      setHistory(null);
      const started = performance.now();
      setRun({ source: kind, started, firstResultMs: null, doneMs: null, progress: "Starting…", frac: 0.02 });
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

      const worker = new Worker(new URL("../lib/parse.worker.ts", import.meta.url), { type: "module" });
      workerRef.current = worker;
      worker.onmessage = (e: MessageEvent<WorkerOut>) => {
        const msg = e.data;
        if (msg.type === "progress") {
          setRun((r) =>
            r && {
              ...r,
              progress:
                msg.stage === "download"
                  ? `Downloading the sample export… ${Math.round(msg.loaded / 1024)} KB`
                  : `Parsed file ${msg.loaded} of ${msg.total}`,
              frac: msg.stage === "download" ? 0.05 + 0.25 * (msg.total ? msg.loaded / msg.total : 0.5) : 0.3 + 0.7 * (msg.loaded / msg.total),
            },
          );
        } else if (msg.type === "snapshot" || msg.type === "done") {
          setDashboard(msg.dashboard);
          setRun((r) => r && { ...r, firstResultMs: r.firstResultMs ?? performance.now() - r.started });
          if (msg.type === "done") {
            setHistory(msg.history);
            setSessionHistory({ history: msg.history, source: kind });
            setRun((r) => r && { ...r, doneMs: performance.now() - r.started, progress: "Done", frac: 1 });
            worker.terminate();
          }
        } else if (msg.type === "error") {
          setError(msg.message);
          setRun(null);
          worker.terminate();
        }
      };
      const timeZone = kind === "sample" ? SAMPLE_TZ : Intl.DateTimeFormat().resolvedOptions().timeZone;
      worker.postMessage({ source, timeZone });
    },
    [ensureCatalog],
  );

  const onFile = (file: File | undefined) => {
    if (!file) return;
    if (!/\.zip$/i.test(file.name)) {
      setError("That isn't a .zip. Drop the my_spotify_data.zip file exactly as Spotify sent it.");
      return;
    }
    start({ kind: "file", file }, "own");
  };

  const fly = useMemo(() => {
    if (!catalog || !history) return null;
    const overlap = matchHistory(catalog, history);
    const mb = trainMushroomBody(catalog, history, overlap);
    return { overlap, mb };
  }, [catalog, history]);

  const s = dashboard?.summary;
  const dateFmt = (t: number) => new Date(t).toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });

  return (
    <main>
      <div className="wrap">
        <header className="hero">
          <p className="eyebrow">Selector · a portfolio project</p>
          <h1>A fruit fly&apos;s brain picks your next song.</h1>
          <p className="lede">
            A personal taste engine built from your own streaming-history export. With recommendations and audio features
            gone from the public API since 2024, it works from what you own: your listening history, audio features
            measured from public previews, and the fly&apos;s olfactory circuit, wired from the real FlyWire connectome,
            as the similarity engine.
          </p>

          <Link className="cta-watch" href="/watch">
            Watch the fly brain pick a track, live →
          </Link>

          <div className="privacy">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <rect x="4" y="11" width="16" height="10" rx="2" />
              <path d="M8 11V7a4 4 0 0 1 8 0v4" />
            </svg>
            <div>
              <strong>Your export never leaves this browser tab.</strong>
              <span>
                The zip is unpacked and analysed right here in JavaScript. There is no upload and no server that could
                receive it. Close the tab and it&apos;s gone.
              </span>
            </div>
          </div>

          <div className="start">
            <button className="btn-primary" onClick={() => start({ kind: "url", url: SAMPLE_URL }, "sample")}>
              Try it with a sample listener →<small>No Spotify account needed. About ten seconds.</small>
            </button>
            <label
              className={`drop${dragOver ? " over" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                onFile(e.dataTransfer.files[0]);
              }}
            >
              <input type="file" accept=".zip,application/zip" onChange={(e) => onFile(e.target.files?.[0])} />
              <strong>Or use your own export</strong>
              <span>Drop my_spotify_data.zip here, or tap to choose it.</span>
            </label>
          </div>
          <details className="howto">
            <summary>How do I get my export?</summary>
            <p>
              On spotify.com go to Account → Privacy settings → Download your data, tick{" "}
              <b>Extended streaming history</b>, and confirm the email Spotify sends (unconfirmed requests are silently
              dropped). It can take up to 30 days to arrive. The sample listener above is synthetic: invented timestamps,
              skips and habits over real catalog tracks.
            </p>
          </details>
        </header>
      </div>

      <div ref={resultsRef} className="wrap" style={{ scrollMarginTop: 16 }}>
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}

        {run && (
          <div className="status" aria-live="polite">
            <span>{run.source === "sample" ? "Sample listener" : "Your export"}</span>
            <span className="bar" aria-hidden="true">
              <i style={{ width: `${run.frac * 100}%` }} />
            </span>
            <span>
              {run.doneMs !== null
                ? `Parsed ${s ? compact(s.plays) : ""} plays in ${(run.doneMs / 1000).toFixed(1)} s`
                : run.progress}
              {run.firstResultMs !== null && run.doneMs === null && ` · first charts after ${(run.firstResultMs / 1000).toFixed(1)} s`}
            </span>
          </div>
        )}

        {history && run?.source === "own" && (
          <Link className="cta-watch" href="/rewind">
            See your listening as a Rewind story →
          </Link>
        )}

        {dashboard && s && (
          <section className="block" aria-labelledby="dash-h">
            <h2 id="dash-h">{run?.source === "sample" ? "The sample listener, in charts" : "Your listening, in charts"}</h2>
            <p className="sub">
              {dateFmt(s.first)} to {dateFmt(s.last)}
              {s.filesDone < s.filesTotal && ` · still reading (${s.filesDone} of ${s.filesTotal} files so far)`}
            </p>
            <div className="tiles">
              <div className="tile hero-tile">
                <div className="label">Hours listened</div>
                <div className="value">{s.hours < 10_000 ? Math.round(s.hours).toLocaleString("en-US") : compact(s.hours)}</div>
              </div>
              <div className="tile">
                <div className="label">Plays</div>
                <div className="value">{compact(s.plays)}</div>
              </div>
              <div className="tile">
                <div className="label">Tracks</div>
                <div className="value">{compact(s.tracks)}</div>
              </div>
              <div className="tile">
                <div className="label">Artists</div>
                <div className="value">{compact(s.artists)}</div>
              </div>
              <div className="tile">
                <div className="label">Skip signal</div>
                <div className="value">{Math.round((dashboard.verdicts.punish / Math.max(s.plays, 1)) * 100)}%</div>
                <div className="detail">of plays read as a skip; the fly learns from these</div>
              </div>
            </div>
            <div className="grid2">
              <div className="card">
                <h3>Listening clock</h3>
                <ListeningClock clock={dashboard.clock} timeZone={dashboard.timeZone} />
              </div>
              <div className="card">
                <h3>Taste drift</h3>
                <p className="note">
                  Share of each {dashboard.drift.granularity}&apos;s plays going to the artists who led a period. Everyone
                  else is in the tooltip and table.
                </p>
                <TasteDrift drift={dashboard.drift} />
              </div>
              <div className="card">
                <h3>Top artists</h3>
                <p className="note">By plays, with hours listened.</p>
                <TopArtists rows={dashboard.topArtists} />
              </div>
              <div className="card">
                <h3>Skip offenders</h3>
                <p className="note">Tracks you keep queuing and bailing on: 5+ plays, skipped forward 30%+ of the time.</p>
                <SkipOffenders rows={dashboard.skipOffenders} />
              </div>
            </div>
          </section>
        )}

        {history && (
          <FlySection catalog={catalog} catalogError={catalogError} history={history} fly={fly} />
        )}
        {history && catalog && fly && (
          <DJSection catalog={catalog} history={history} overlap={fly.overlap} mb={fly.mb} />
        )}
      </div>

      <footer>
        <div className="wrap">
          <p>
            <b>What the fly is, honestly.</b> Dasgupta, Stevens &amp; Navlakha (Science, 2017) showed that the fruit
            fly&apos;s olfactory circuit works as a locality-sensitive hash: ~50 projection neurons fan out to ~2,000
            Kenyon cells, and inhibition leaves about 5% firing. Selector runs that algorithm with the real projection
            neuron to Kenyon cell wiring from the FlyWire connectome (CC-BY 4.0), over each track&apos;s features, and
            the mushroom body&apos;s plasticity rule learns taste from skips. It is a wiring diagram used as a hash, not
            a fly brain doing general tasks.
          </p>
          <p>
            <b>Where the numbers come from.</b> The catalog is 19,386 tracks from the author&apos;s own listening
            history. Tempo and energy are measured from 30-second previews (iTunes and Deezer public APIs) for the
            16,741 tracks that matched. Mood and era are predicted by a fine-tuned small model. Your tracks outside that
            catalog still count in the charts but can&apos;t be fingerprinted. No audio is served by this page, only
            derived numbers.
          </p>
        </div>
      </footer>
    </main>
  );
}
