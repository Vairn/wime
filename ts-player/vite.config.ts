import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));
const publicMusic = path.resolve(root, "public/music");
const combinedMusic = path.resolve(root, "../combined");

/** Prefer public/music; fall back to ../combined for in-repo dev. */
function serveMusic(): Plugin {
  const musicRoot = fs.existsSync(publicMusic) ? publicMusic : combinedMusic;

  const handler = (
    req: { url?: string },
    res: {
      statusCode: number;
      end: (s: string) => void;
      setHeader: (k: string, v: string) => void;
    },
    next: () => void,
  ) => {
    try {
      const rel = decodeURIComponent((req.url ?? "/").split("?")[0] ?? "/");
      const filePath = path.normalize(path.join(musicRoot, rel));
      if (!filePath.startsWith(musicRoot)) {
        res.statusCode = 403;
        res.end("Forbidden");
        return;
      }
      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        next();
        return;
      }
      res.setHeader("Cache-Control", "public, max-age=3600");
      res.setHeader("Content-Type", "application/octet-stream");
      fs.createReadStream(filePath).pipe(res as unknown as NodeJS.WritableStream);
    } catch {
      next();
    }
  };

  return {
    name: "serve-music",
    configureServer(server) {
      // Only needed when falling back to ../combined (public/ is auto-served)
      if (musicRoot === combinedMusic) {
        server.middlewares.use("/music", handler);
      }
    },
    configurePreviewServer(server) {
      if (musicRoot === combinedMusic) {
        server.middlewares.use("/music", handler);
      }
    },
  };
}

export default defineConfig({
  plugins: [serveMusic()],
  publicDir: "public",
  server: {
    fs: { allow: [root, combinedMusic] },
  },
});
