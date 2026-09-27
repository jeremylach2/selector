// Upload the private Rewind reports (data/wrapped_private/*.json, written by
// `python -m selector.warehouse.wrapped --profile private`) to the project's
// private Vercel Blob store, where app/rewind/private/ reads them. No
// redeploy needed. See docs/PRIVACY.md.
//
// Usage (from web/, after `vercel env pull .env.local`):
//   node --env-file=.env.local scripts/upload-private-rewind.mjs
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { put } from "@vercel/blob";

const SOURCE = path.resolve(import.meta.dirname, "../../data/wrapped_private");
const PREFIX = "rewind-private/";

const files = (await readdir(SOURCE)).filter((f) => f.endsWith(".json"));
if (!files.length) throw new Error(`no reports in ${SOURCE}; run the private export first`);
for (const file of files) {
  const body = await readFile(path.join(SOURCE, file));
  if (JSON.parse(body.toString("utf-8")).audience !== "private") {
    throw new Error(`${file} is not a private report; refusing to upload`);
  }
  await put(PREFIX + file, body, {
    access: "private",
    contentType: "application/json",
    addRandomSuffix: false,
    allowOverwrite: true,
  });
  console.log(`uploaded ${PREFIX}${file}`);
}
