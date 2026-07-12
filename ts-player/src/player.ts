/**
 * Song loading + realtime / offline playback glue for the browser.
 *
 * Loads SMUS + instruments over HTTP, drives SmusEngine through a
 * ScriptProcessorNode for live audio, and can offline-render to WAV.
 */
import { SmusEngine } from "./engine";
import { defaultInstrument, loadInstrumentByName, type Instrument } from "./instruments";
import { parseSmus, type SmusScore } from "./smus";
import { downloadBlob, encodeWav } from "./wav";

/** A score plus its resolved instrument bank and the catalog file index. */
export interface LoadedSong {
  score: SmusScore;
  instruments: Map<number, Instrument>;
  fileIndex: Map<string, string>;
}

/**
 * Fetch catalog.json and build a lowercase → real-filename map.
 * Amiga disks mix case; the index lets lookups ignore casing.
 */
export async function buildFileIndex(catalogUrl = "/catalog.json"): Promise<Map<string, string>> {
  const res = await fetch(catalogUrl);
  const catalog = (await res.json()) as { files: string[] };
  const map = new Map<string, string>();
  for (const f of catalog.files) {
    map.set(f.toLowerCase(), f);
  }
  return map;
}

/**
 * Load a SMUS song by stem (e.g. "Hob.Riven") and all of its instruments.
 *
 * 1. Resolve the .smus filename via the catalog
 * 2. Parse the score
 * 3. Load each INS1 instrument (falling back to a sine default on failure)
 * 4. Ensure register 0 always exists
 */
export async function loadSong(
  songStem: string,
  fileIndex: Map<string, string>,
  folderUrl = "/music",
): Promise<LoadedSong> {
  const smusName = fileIndex.get(`${songStem}.smus`.toLowerCase()) ?? `${songStem}.smus`;
  const res = await fetch(`${folderUrl}/${encodeURIComponent(smusName)}`);
  if (!res.ok) throw new Error(`Failed to load ${smusName}`);
  const data = new Uint8Array(await res.arrayBuffer());
  const score = parseSmus(data, smusName);

  // Load every instrument register referenced by the score.
  const instruments = new Map<number, Instrument>();
  for (const [reg, name] of score.instruments) {
    try {
      instruments.set(reg, await loadInstrumentByName(folderUrl, name, fileIndex));
    } catch (e) {
      console.warn(`Instrument ${name} failed:`, e);
      instruments.set(reg, defaultInstrument(name));
    }
  }
  // Register 0 is the fallback voice when a track hasn't selected an instrument yet.
  if (!instruments.has(0)) {
    instruments.set(0, defaultInstrument());
  }

  return { score, instruments, fileIndex };
}

/**
 * Realtime player: owns an AudioContext, ScriptProcessorNode, and SmusEngine.
 * Pulls audio blocks from the engine on the audio thread callback and
 * notifies the UI via `onFrame` each animation frame.
 */
export class AudioPlayer {
  ctx: AudioContext | null = null;
  engine: SmusEngine | null = null;
  private node: ScriptProcessorNode | null = null;
  private gain: GainNode | null = null;
  playing = false;
  /** Called every animation frame while playing (drives scopes / playhead). */
  onFrame: ((engine: SmusEngine) => void) | null = null;
  /** Called once when the song finishes naturally. */
  onEnded: (() => void) | null = null;
  private raf = 0;

  /**
   * Lazily create (or resume) the AudioContext.
   * Browsers require a user gesture before audio can start — Play click covers that.
   */
  async ensureCtx(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: 44100 });
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    return this.ctx;
  }

  /**
   * Start playing a score from the beginning.
   * Stops any current playback, builds a fresh engine, and wires ScriptProcessor → speakers.
   */
  async play(score: SmusScore, instruments: Map<number, Instrument>, volume = 0.28): Promise<void> {
    this.stop(false);
    const ctx = await this.ensureCtx();
    this.engine = new SmusEngine(score, instruments, ctx.sampleRate, volume);
    this.engine.kick(); // consume leading control events before first audio block

    const bufferSize = 2048;
    // ScriptProcessor pulls `bufferSize` frames whenever the audio graph needs data.
    const node = ctx.createScriptProcessor(bufferSize, 0, 2);
    const gain = ctx.createGain();
    gain.gain.value = 1;
    this.node = node;
    this.gain = gain;

    const interleaved = new Float32Array(bufferSize * 2);
    node.onaudioprocess = (ev) => {
      const eng = this.engine;
      // Song over → output silence and tear down.
      if (!eng || eng.finished) {
        ev.outputBuffer.getChannelData(0).fill(0);
        ev.outputBuffer.getChannelData(1).fill(0);
        if (eng?.finished) this.stop(true);
        return;
      }
      // Render stereo interleaved floats, then de-interleave into Web Audio channels.
      eng.renderBlock(bufferSize, interleaved);
      const L = ev.outputBuffer.getChannelData(0);
      const R = ev.outputBuffer.getChannelData(1);
      for (let i = 0; i < bufferSize; i++) {
        L[i] = interleaved[i * 2]!;
        R[i] = interleaved[i * 2 + 1]!;
      }
    };

    node.connect(gain);
    gain.connect(ctx.destination);
    this.playing = true;
    this.tick(); // start UI refresh loop
  }

  /** Animation-frame loop that pushes engine state to the UI via `onFrame`. */
  private tick = (): void => {
    if (!this.playing || !this.engine) return;
    this.onFrame?.(this.engine);
    this.raf = requestAnimationFrame(this.tick);
  };

  /**
   * Stop playback and disconnect the audio graph.
   * @param emitEnded - if true (and we were playing), fire `onEnded`
   */
  stop(emitEnded = false): void {
    const wasPlaying = this.playing;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    try {
      this.node?.disconnect();
    } catch {
      /* already disconnected */
    }
    try {
      this.gain?.disconnect();
    } catch {
      /* already disconnected */
    }
    this.node = null;
    this.gain = null;
    this.engine = null;
    if (emitEnded && wasPlaying) this.onEnded?.();
  }
}

/**
 * Offline-render an entire song to a WAV Blob.
 *
 * Runs the engine in chunks on the main thread, yielding to the event loop
 * every ~40 ms so the progress callback can update the UI without freezing.
 * Trims trailing silence and encodes 16-bit stereo PCM.
 */
export async function exportSongWav(
  score: SmusScore,
  instruments: Map<number, Instrument>,
  opts: {
    sampleRate?: number;
    volume?: number;
    onProgress?: (ratio: number) => void;
  } = {},
): Promise<{ blob: Blob; seconds: number }> {
  const sampleRate = opts.sampleRate ?? 44100;
  const volume = opts.volume ?? 0.28;
  const engine = new SmusEngine(score, instruments, sampleRate, volume);
  engine.kick();

  const block = 2048;
  const maxSamples = sampleRate * 600; // hard cap: 10 minutes
  const chunks: Float32Array[] = [];
  let total = 0;
  const buf = new Float32Array(block * 2);
  let lastYield = performance.now();

  // Render until the sequencer and all voices are idle.
  while (total < maxSamples) {
    engine.renderBlock(block, buf);
    chunks.push(buf.slice());
    total += block;
    if (engine.finished && engine.voices.every((v) => !v.active)) break;

    // Rough progress estimate (capped); real length unknown until finished.
    opts.onProgress?.(Math.min(0.95, total / (sampleRate * 120)));
    if (performance.now() - lastYield > 40) {
      await new Promise((r) => setTimeout(r, 0)); // yield to UI
      lastYield = performance.now();
    }
  }

  // Concatenate chunk buffers into one interleaved stream.
  const audio = new Float32Array(total * 2);
  let off = 0;
  for (const c of chunks) {
    audio.set(c, off);
    off += c.length;
  }

  // Find last audible sample and keep a short pad of silence after it.
  const thresh = 1e-4;
  let last = 0;
  for (let i = 0; i < total; i++) {
    if (Math.abs(audio[i * 2]!) > thresh || Math.abs(audio[i * 2 + 1]!) > thresh) last = i;
  }
  const keep = Math.min(total, last + Math.floor(sampleRate / 4) + 1);
  const trimmed = audio.subarray(0, keep * 2);

  opts.onProgress?.(1);
  const wav = encodeWav(trimmed, sampleRate);
  return {
    blob: new Blob([new Uint8Array(wav)], { type: "audio/wav" }),
    seconds: keep / sampleRate,
  };
}

export { downloadBlob };
