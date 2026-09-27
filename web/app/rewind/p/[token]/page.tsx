import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Story from "@/components/rewind/Story";
import { tokenMatches } from "@/lib/rewind-private";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Selector · Rewind",
  robots: { index: false, follow: false },
};

// The private share link. The token in the URL is the only credential; the
// story sends it to the gated report route as a bearer header.
export default async function PrivateRewindPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  if (!tokenMatches(token)) notFound();
  return (
    <main>
      <div className="wrap rewind">
        <header className="hero rewind-hero">
          <p className="eyebrow">
            <Link href="/">Selector</Link> · Rewind
          </p>
          <h1>My listening, as a story.</h1>
          <p className="lede">A private link: please don&apos;t share it further.</p>
        </header>
        <Story source={{ base: "/rewind/private", token }} />
      </div>
    </main>
  );
}
