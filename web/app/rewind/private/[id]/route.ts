import { PRIVATE_HEADERS, readPrivateReport, tokenMatches } from "@/lib/rewind-private";

export const dynamic = "force-dynamic";

// 404, never 401, for a missing or wrong token: the route doesn't advertise
// that anything is here.
const notFound = () => new Response("Not found", { status: 404, headers: PRIVATE_HEADERS });

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const auth = request.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ") || !tokenMatches(auth.slice(7))) return notFound();
  const { id } = await params;
  const stream = await readPrivateReport(id);
  if (!stream) return notFound();
  return new Response(stream, {
    headers: { ...PRIVATE_HEADERS, "Content-Type": "application/json; charset=utf-8" },
  });
}
