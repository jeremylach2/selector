import type { NextConfig } from "next";

const PRIVATE_HEADERS = [
  { key: "Cache-Control", value: "private, no-store" },
  { key: "X-Robots-Tag", value: "noindex, nofollow" },
  // The share token is in the page URL; never send it on as a referrer.
  { key: "Referrer-Policy", value: "no-referrer" },
];

const nextConfig: NextConfig = {
  // The fly catalog and the sample export are immutable per build; the
  // file names don't change, so keep the browser cache short but let the
  // CDN hold them.
  async headers() {
    return [
      {
        source: "/:dir(fly|sample)/:file*",
        headers: [{ key: "Cache-Control", value: "public, max-age=3600, s-maxage=86400" }],
      },
      { source: "/rewind/p/:token*", headers: PRIVATE_HEADERS },
      { source: "/rewind/private/:id*", headers: PRIVATE_HEADERS },
    ];
  },
  // The story was first published as /wrapped.
  async redirects() {
    return [{ source: "/wrapped", destination: "/rewind", permanent: true }];
  },
};

export default nextConfig;
