// Server-only: the gate in front of the real Rewind reports. See
// docs/PRIVACY.md. The reports live in a private Vercel Blob store, never in
// public/ or the build, and are served only to a request carrying
// REWIND_SHARE_TOKEN. Rotating that env var kills every old share link.
import "server-only";
import { createHash, timingSafeEqual } from "node:crypto";
import { get } from "@vercel/blob";

export const SHARE_TOKEN_ENV = "REWIND_SHARE_TOKEN";
export const BLOB_PREFIX = "rewind-private/";
// "index", "all" or a calendar year: nothing else is ever looked up.
const REPORT_ID = /^(index|all|\d{4})$/;

export const PRIVATE_HEADERS = {
  "Cache-Control": "private, no-store",
  "X-Robots-Tag": "noindex, nofollow",
  "Referrer-Policy": "no-referrer",
};

const digest = (s: string) => createHash("sha256").update(s).digest();

// Constant time, and the digests make the comparison length-independent too.
// An unset or short env var never matches, so a misconfigured deploy fails
// closed.
export function tokenMatches(got: string | null | undefined): boolean {
  const expected = process.env[SHARE_TOKEN_ENV];
  if (!expected || expected.length < 32 || !got) return false;
  return timingSafeEqual(digest(got), digest(expected));
}

export async function readPrivateReport(id: string): Promise<ReadableStream<Uint8Array> | null> {
  if (!REPORT_ID.test(id)) return null;
  try {
    const result = await get(`${BLOB_PREFIX}${id}.json`, { access: "private" });
    return result?.statusCode === 200 ? result.stream : null;
  } catch (e) {
    // A missing store or credentials still answers 404 to the caller.
    console.error("private rewind read failed", e);
    return null;
  }
}
