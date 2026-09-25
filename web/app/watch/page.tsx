import type { Metadata } from "next";
import Link from "next/link";
import Watch from "@/components/watch/Watch";

export const metadata: Metadata = {
  title: "Selector · watch the fly pick a track",
  description:
    "A live run of the fruit fly's olfactory circuit, wired from the FlyWire connectome, choosing the next song. Every number is computed in your browser.",
};

export default function WatchPage() {
  return (
    <main>
      <div className="wrap watch">
        <header className="hero watch-hero">
          <p className="eyebrow">
            <Link href="/">Selector</Link> · the fly DJ, live
          </p>
          <h1>Watch a fly brain pick the next track.</h1>
          <p className="lede">
            A song goes in as numbers. The fly&apos;s olfactory circuit hashes it into a sparse fingerprint, finds its neighbours, and
            its mushroom body votes on which to play. Nothing here is scripted: every dot is computed in this tab from the real
            FlyWire wiring and the precomputed fingerprints. Skip the fly&apos;s pick and watch it learn.
          </p>
        </header>
        <Watch />
        <footer className="watch-foot">
          <p>
            Each stage carries both names: the biology and the algorithm it implements. The circuit is a locality-sensitive hash
            (Dasgupta, Stevens &amp; Navlakha, <i>Science</i> 2017) run over the real projection-neuron to Kenyon-cell wiring from
            FlyWire (CC-BY 4.0). It is a wiring diagram used as a hash, not a fly brain doing general tasks.{" "}
            <Link href="/">Try it on your own Spotify export →</Link>
          </p>
        </footer>
      </div>
    </main>
  );
}
