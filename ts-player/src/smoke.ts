/**
 * Headless smoke test — load a few songs from disk and render ~2s of audio.
 *
 * Run: npx vite-node src/smoke.ts
 *
 * Resolves music from `public/music` (zip package) or `../combined` (repo layout).
 * Exits non-zero if any song renders silence (peak below threshold).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SmusEngine } from "./engine";
import { defaultInstrument, detectAndLoadInstrument, loadSampledInstr } from "./instruments";
import { parseSmus } from "./smus";

const here = path.dirname(fileURLToPath(import.meta.url));
const candidates = [
  path.resolve(here, "../public/music"),
  path.resolve(here, "../../combined"),
];
const root = candidates.find((p) => fs.existsSync(p));
if (!root) throw new Error("No music folder found (public/music or ../combined)");
const musicRoot: string = root;

/** Lowercase filename → real filename (Amiga disks are case-insensitive). */
const index = new Map(fs.readdirSync(musicRoot).map((f) => [f.toLowerCase(), f]));

/** Read a music asset from disk as bytes. */
const read = (name: string) => new Uint8Array(fs.readFileSync(path.join(musicRoot, name)));

/**
 * Load an instrument by SMUS name from the local music folder.
 * Detects Synthesis vs SampledSound and pulls the linked .ss when needed.
 */
async function loadInstr(name: string) {
  const file = index.get(`${name}.instr`.toLowerCase());
  if (!file) throw new Error(`missing ${name}`);
  const data = read(file);
  const loaded = detectAndLoadInstrument(data, name);
  if (loaded === "sampled") {
    // SampledSound .instr points at a .ss payload — fetch it by name.
    return loadSampledInstr(data, name, musicRoot, async (ssName) => {
      const mapped = index.get(ssName.toLowerCase());
      if (!mapped) throw new Error(ssName);
      return read(mapped);
    });
  }
  return loaded;
}

// Render a short burst of each song and assert non-silent output.
for (const song of ["Hob.Riven", "Title", "Gandalf", "Frodo"]) {
  const score = parseSmus(read(`${song}.smus`), `${song}.smus`);
  const instruments = new Map<number, Awaited<ReturnType<typeof loadInstr>>>();
  for (const [reg, name] of score.instruments) {
    try {
      instruments.set(reg, await loadInstr(name));
    } catch (e) {
      console.warn(`  ${song} ${name}:`, e);
      instruments.set(reg, defaultInstrument(name));
    }
  }
  if (!instruments.has(0)) instruments.set(0, defaultInstrument());

  const eng = new SmusEngine(score, instruments, 44100, 0.28);
  eng.kick();
  const block = new Float32Array(2048 * 2);
  let peak = 0;
  let samples = 0;
  // ~2 seconds of audio is enough to prove the engine is producing sound.
  while (samples < 44100 * 2 && !eng.finished) {
    eng.renderBlock(2048, block);
    for (let i = 0; i < block.length; i++) peak = Math.max(peak, Math.abs(block[i]!));
    samples += 2048;
  }
  const kinds = [...instruments.values()].map((i) => `${i.name}:${i.kind}`).join(", ");
  console.log(
    `${song.padEnd(12)} peak=${peak.toFixed(4)} beat=${eng.beatPos.toFixed(1)}  [${kinds}]`,
  );
  if (peak < 1e-5) throw new Error(`${song} silent`);
}
console.log("OK");
