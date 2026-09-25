import type { NextConfig } from "next";

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
    ];
  },
};

export default nextConfig;
