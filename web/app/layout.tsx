import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Selector · a fly brain picks your next song",
  description:
    "Drop in your Spotify export and see your listening, then let a fruit fly's olfactory circuit find more like it. Parsed in your browser; nothing is uploaded.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9f9f7" },
    { media: "(prefers-color-scheme: dark)", color: "#0d0d0d" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <footer className="site-foot">
          <p>Not affiliated with or endorsed by Spotify. A non-commercial personal project.</p>
        </footer>
      </body>
    </html>
  );
}
