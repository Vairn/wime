/**
 * War in Middle-earth — IFF-SMUS Tracker UI bootstrap.
 *
 * This module is the application shell: it injects the tracker layout into `#app`,
 * wires DOM controls (song picker, transport, export), builds the per-channel
 * pattern grid and beat rail, and drives live updates from the audio engine
 * (scopes, VU meters, score canvas, playhead scroll).
 *
 * Playback and offline WAV export are delegated to `AudioPlayer`; pattern parsing
 * and score rendering use `smus` and `ScoreView` respectively.
 */

import "./style.css";
import type { SmusEngine } from "./engine";
import { midiToName, expandPattern, type PatternRow, type SmusScore } from "./smus";
import { AudioPlayer, buildFileIndex, loadSong, exportSongWav, downloadBlob } from "./player";
import { ScoreView } from "./score";

/** Pixel height of one tracker row; used for playhead scroll math. */
const ROW_H = 22;

/** Root mount point for the entire tracker UI. */
const app = document.querySelector<HTMLDivElement>("#app")!;

// --- Static layout: header transport, scopes, score panel, tracker grid, footer ---
app.innerHTML = `
  <header class="topbar">
    <div class="brand">
      <h1>War in Middle‑earth</h1>
      <div class="tag">Aegis Sonix · IFF‑SMUS Tracker</div>
    </div>
    <div class="transport">
      <select id="song" aria-label="Song"></select>
      <button type="button" class="primary" id="play">Play</button>
      <button type="button" id="stop" disabled>Stop</button>
      <button type="button" id="export">Export WAV</button>      <div class="meta">
        <span>BPM <strong id="bpm">—</strong></span>
        <span>TRK <strong id="trks">4</strong></span>
        <span>POS <strong id="pos">0.00</strong></span>
      </div>
    </div>
  </header>

  <main class="stage">
    <div class="scopes" id="scopes"></div>
    <section class="score-panel" aria-label="Musical score">
      <div class="score-head">
        <span class="score-title">Score</span>
        <span class="score-hint">treble · four voices</span>
      </div>
      <canvas id="score" class="score-canvas"></canvas>
    </section>
    <div class="tracker">
      <div class="playhead"></div>
      <div class="beat-rail"><div class="beat-rows" id="beatRows"></div></div>
      <div class="pattern-scroll">
        <div class="channels" id="channels"></div>
      </div>
    </div>
  </main>

  <footer class="status">
    <span><i class="pulse" id="pulse"></i><span id="status">Ready</span></span>
    <span>Paula L‑R‑R‑L · 44.1 kHz</span>
  </footer>
`;

// --- DOM references: transport controls, metadata readouts, tracker regions ---
const songSelect = app.querySelector<HTMLSelectElement>("#song")!;
const playBtn = app.querySelector<HTMLButtonElement>("#play")!;
const stopBtn = app.querySelector<HTMLButtonElement>("#stop")!;
const exportBtn = app.querySelector<HTMLButtonElement>("#export")!;
const bpmEl = app.querySelector("#bpm")!;
const trksEl = app.querySelector("#trks")!;
const posEl = app.querySelector("#pos")!;
const statusEl = app.querySelector("#status")!;
const pulseEl = app.querySelector("#pulse")!;
const scopesEl = app.querySelector("#scopes")!;
const channelsEl = app.querySelector("#channels")!;
const beatRowsEl = app.querySelector<HTMLElement>("#beatRows")!;
const scoreCanvas = app.querySelector<HTMLCanvasElement>("#score")!;
const scoreView = new ScoreView(scoreCanvas);

// Per-channel DOM handles populated in the loop below (scopes + pattern columns).
const scopeCanvases: HTMLCanvasElement[] = [];
const noteEls: HTMLElement[] = [];
const instEls: HTMLElement[] = [];
const vuEls: HTMLElement[] = [];
const rowContainers: HTMLElement[] = [];

// Build four channel scope cards (waveform + VU) and matching pattern columns.
for (let ch = 0; ch < 4; ch++) {
  const card = document.createElement("div");
  card.className = "scope-card";
  card.dataset.ch = String(ch);
  card.innerHTML = `
    <div class="scope-head">
      <span>CH ${ch}</span>
      <span class="note" data-note>—</span>
      <span class="inst" data-inst></span>
    </div>
    <canvas class="scope" width="240" height="48"></canvas>
    <div class="vu"><i></i></div>
  `;
  scopesEl.appendChild(card);
  scopeCanvases.push(card.querySelector("canvas")!);
  noteEls.push(card.querySelector("[data-note]")!);
  instEls.push(card.querySelector("[data-inst]")!);
  vuEls.push(card.querySelector(".vu > i")!);

  const col = document.createElement("div");
  col.className = "channel-col";
  col.style.setProperty("--ch", `var(--channel-${ch})`);
  col.innerHTML = `<div class="channel-label">Channel ${ch}</div><div class="rows" data-rows></div>`;
  channelsEl.appendChild(col);
  rowContainers.push(col.querySelector("[data-rows]")!);
}

const player = new AudioPlayer();
let fileIndex: Map<string, string> = new Map();
let songs: string[] = [];
/** Expanded pattern rows per Paula channel (0–3), rebuilt when a song loads. */
let patternByChannel: PatternRow[][] = [[], [], [], []];
/** Total pattern length in beats (max beat + duration across all rows). */
let maxBeat = 0;
let currentScore: SmusScore | null = null;
let loadedInstruments: Map<number, import("./instruments").Instrument> | null = null;
/** Guards against overlapping async song loads. */
let loading = false;

/**
 * Format a byte-sized number as two uppercase hex digits (e.g. instrument/volume cells).
 *
 * @param n - Integer in 0–255 range (caller responsibility).
 * @returns Two-character hex string, zero-padded.
 */
function hex2(n: number): string {
  return n.toString(16).toUpperCase().padStart(2, "0");
}

/**
 * Rebuild the tracker pattern grid and beat rail from a loaded SMUS score.
 *
 * Expands the score into timed rows, quantizes unique beat positions into a
 * shared vertical grid, renders note/rest/empty cells per channel, updates the
 * beat rail labels, and refreshes the score canvas data.
 *
 * @param score - Parsed SMUS score for the currently selected song.
 */
function buildPatternDom(score: SmusScore): void {
  const rows = expandPattern(score);
  patternByChannel = [[], [], [], []];
  maxBeat = 0;

  // Partition note/rest rows by channel; skip control/meta rows.
  for (const r of rows) {
    if (r.kind === "ctrl") continue;
    patternByChannel[r.channel]?.push(r);
    maxBeat = Math.max(maxBeat, r.beat + r.duration);
  }

  // Quantize display to sixteenth-ish grid from unique beat starts
  const beats = new Set<number>();
  for (const ch of patternByChannel) {
    for (const r of ch) beats.add(Math.round(r.beat * 1000) / 1000);
  }
  const beatList = [...beats].sort((a, b) => a - b);
  if (beatList.length === 0) beatList.push(0);

  // One DOM row per quantized beat, per channel column.
  for (let ch = 0; ch < 4; ch++) {
    const container = rowContainers[ch]!;
    container.innerHTML = "";
    const byBeat = new Map<number, PatternRow>();
    for (const r of patternByChannel[ch]!) {
      const key = Math.round(r.beat * 1000) / 1000;
      if (!byBeat.has(key)) byBeat.set(key, r);
    }

    for (let i = 0; i < beatList.length; i++) {
      const b = beatList[i]!;
      const r = byBeat.get(b);
      const el = document.createElement("div");
      el.className = "row";
      el.dataset.beat = String(b);
      // Whole-number beats divisible by 4 get a bar-line style.
      if (Math.round(b) === b && b % 4 === 0) el.classList.add("bar");
      if (!r) {
        // No event at this beat — dim placeholder cell.
        el.innerHTML = `<span class="cell-note">···</span>`;
        el.style.opacity = "0.35";
      } else if (r.kind === "rest") {
        el.classList.add("rest");
        el.innerHTML = `<span class="cell-note">---</span><span class="cell-inst"></span><span class="cell-vol"></span>`;
      } else {
        el.innerHTML = `
          <span class="cell-note">${r.noteName ?? "???"}</span>
          <span class="cell-inst">${hex2(r.instrument ?? 0)}</span>
          <span class="cell-vol">${hex2(r.volume ?? 127)}</span>`;
      }
      container.appendChild(el);
    }
  }

  // Beat rail: left column labels aligned with pattern rows.
  beatRowsEl.innerHTML = "";
  for (let i = 0; i < beatList.length; i++) {
    const b = beatList[i]!;
    const el = document.createElement("div");
    el.className = "row";
    if (Math.round(b) === b && b % 4 === 0) el.classList.add("bar");
    el.textContent = b.toFixed(b % 1 === 0 ? 0 : 2);
    beatRowsEl.appendChild(el);
  }

  // Store beat list on containers for scroll sync
  (beatRowsEl as HTMLElement & { _beats?: number[] })._beats = beatList;

  scoreView.setScore(rows, maxBeat);
}

/**
 * Vertically scroll the beat rail and pattern columns so `beat` sits at the playhead.
 *
 * Finds the bracketing rows in the quantized beat list, interpolates fractional
 * position between them, applies a shared `translateY` to rail + channels, and
 * toggles the `active` class on the current row in each channel.
 *
 * @param beat - Current playback position in beats (from engine).
 */
function scrollToBeat(beat: number): void {
  const beats = (beatRowsEl as HTMLElement & { _beats?: number[] })._beats ?? [];
  if (!beats.length) return;

  // Index of the last beat row at or before the current position.
  let idx = 0;
  for (let i = 0; i < beats.length; i++) {
    if (beats[i]! <= beat + 1e-6) idx = i;
    else break;
  }
  // Smooth between rows
  let frac = 0;
  if (idx < beats.length - 1) {
    const a = beats[idx]!;
    const b = beats[idx + 1]!;
    frac = b > a ? Math.min(1, Math.max(0, (beat - a) / (b - a))) : 0;
  }

  // Center the interpolated row on the fixed playhead in the tracker viewport.
  const center = (scopesEl.parentElement?.querySelector(".tracker") as HTMLElement)?.clientHeight ?? 400;
  const mid = center / 2 - ROW_H / 2 - 28;
  const y = mid - (idx + frac) * ROW_H;

  beatRowsEl.style.transform = `translateY(${y}px)`;
  for (const c of rowContainers) c.style.transform = `translateY(${y}px)`;

  // Highlight active rows
  for (let ch = 0; ch < 4; ch++) {
    const kids = rowContainers[ch]!.children;
    for (let i = 0; i < kids.length; i++) {
      kids[i]!.classList.toggle("active", i === idx);
    }
  }
}

/**
 * Draw the mini waveform scope for one Paula channel.
 *
 * When the voice is active, samples the instrument wave table from the current
 * phase position and scales amplitude by the envelope; otherwise draws a flat line.
 * Always overlays a faint horizontal center grid.
 *
 * @param ch - Channel index 0–3.
 * @param eng - Live engine state, or `null` when idle (flat scope).
 */
function drawScope(ch: number, eng: SmusEngine | null): void {
  const canvas = scopeCanvases[ch]!;
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0a120e";
  ctx.fillRect(0, 0, w, h);

  const colors = ["#c4783a", "#5a9e7a", "#6a8ab8", "#b87a6a"];
  ctx.strokeStyle = colors[ch]!;
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  const v = eng?.voices[ch];
  const wave = v?.instrument?.wave;
  const active = v?.active && wave && wave.length > 0;

  if (active) {
    const len = Math.min(wave.length, 128);
    const start = Math.floor(v!.pos) % wave.length;
    for (let i = 0; i < w; i++) {
      const wi = (start + Math.floor((i / w) * len)) % wave.length;
      const sample = wave[wi]! * (0.35 + v!.envLevel * 0.65);
      const y = h / 2 - sample * (h * 0.42);
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
  } else {
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
  }
  ctx.stroke();

  // grid
  ctx.strokeStyle = "rgba(138,158,140,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

/**
 * Per-frame UI refresh callback wired to `AudioPlayer.onFrame`.
 *
 * Updates BPM/position readouts, scrolls the tracker playhead, advances the score
 * view, and refreshes each channel's note label, instrument name, VU meter, and scope.
 *
 * @param eng - Current `SmusEngine` snapshot for this audio frame.
 */
function updateUi(eng: SmusEngine): void {
  bpmEl.textContent = eng.bpm.toFixed(1);
  posEl.textContent = eng.beatPos.toFixed(2);
  scrollToBeat(eng.beatPos);
  scoreView.setBeat(eng.beatPos);

  for (let ch = 0; ch < 4; ch++) {
    const v = eng.voices[ch]!;
    if (v.active && v.instrument) {
      noteEls[ch]!.textContent = midiToName(v.midi);
      instEls[ch]!.textContent = v.instrument.name;
      const pct = Math.min(100, v.peak * 180);
      vuEls[ch]!.style.width = `${pct}%`;
    } else {
      noteEls[ch]!.textContent = "---";
      // Decay VU bar smoothly when voice is silent.
      const p = parseFloat(vuEls[ch]!.style.width || "0") * 0.88;
      vuEls[ch]!.style.width = `${p}%`;
    }
    drawScope(ch, eng);
  }
}

/**
 * Load a song by catalog stem: fetch SMUS + instruments, rebuild pattern DOM, reset UI.
 *
 * Stops playback, disables transport while loading, and re-enables play/export on success.
 * On failure, surfaces the error in the status line.
 *
 * @param stem - Song filename stem from `catalog.json` (select option value).
 */
async function selectSong(stem: string): Promise<void> {
  if (loading) return;
  loading = true;
  player.stop();
  playBtn.disabled = true;
  stopBtn.disabled = true;
  exportBtn.disabled = true;
  pulseEl.classList.remove("on");
  statusEl.textContent = `Loading ${stem}…`;

  try {
    const loaded = await loadSong(stem, fileIndex);
    currentScore = loaded.score;
    loadedInstruments = loaded.instruments;
    bpmEl.textContent = (loaded.score.tempo / 128).toFixed(1);
    trksEl.textContent = String(loaded.score.tracks.length);
    buildPatternDom(loaded.score);
    scrollToBeat(0);
    scoreView.setBeat(0);
    for (let ch = 0; ch < 4; ch++) drawScope(ch, null);
    statusEl.textContent = `${loaded.score.name} · ${loaded.instruments.size} instruments`;
    playBtn.disabled = false;
    exportBtn.disabled = false;
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Error: ${e instanceof Error ? e.message : e}`;
  } finally {
    loading = false;
  }
}

// --- Play: start engine, enable per-frame UI updates, toggle transport state ---
playBtn.addEventListener("click", async () => {
  if (!currentScore || !loadedInstruments) return;
  statusEl.textContent = `Playing ${currentScore.name}`;
  pulseEl.classList.add("on");
  playBtn.disabled = true;
  stopBtn.disabled = false;
  player.onFrame = updateUi;
  await player.play(currentScore, loadedInstruments);
  // rebuild pattern position when engine recreates
  if (player.engine) scrollToBeat(0);
});

// --- Stop: halt playback and reset visual state to idle ---
stopBtn.addEventListener("click", () => {
  player.stop();
  pulseEl.classList.remove("on");
  playBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Stopped";
  posEl.textContent = "0.00";
  scrollToBeat(0);
  scoreView.setBeat(0);
  for (let ch = 0; ch < 4; ch++) {
    noteEls[ch]!.textContent = "---";
    vuEls[ch]!.style.width = "0%";
    drawScope(ch, null);
  }
});

// --- Song picker: load newly selected stem into the tracker ---
songSelect.addEventListener("change", () => {
  void selectSong(songSelect.value);
});

// --- Export WAV: offline render via worker, download blob, restore UI ---
exportBtn.addEventListener("click", async () => {
  if (!currentScore || !loadedInstruments || loading) return;
  const wasPlaying = player.playing;
  if (wasPlaying) {
    player.stop();
    pulseEl.classList.remove("on");
    stopBtn.disabled = true;
  }

  exportBtn.disabled = true;
  playBtn.disabled = true;
  songSelect.disabled = true;
  statusEl.textContent = "Rendering WAV… 0%";

  try {
    const stem = songSelect.value || currentScore.name;
    const { blob, seconds } = await exportSongWav(currentScore, loadedInstruments, {
      onProgress: (r) => {
        statusEl.textContent = `Rendering WAV… ${Math.round(r * 100)}%`;
      },
    });
    downloadBlob(blob, `${stem}.wav`);
    statusEl.textContent = `Exported ${stem}.wav (${seconds.toFixed(1)}s)`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Export failed: ${e instanceof Error ? e.message : e}`;
  } finally {
    exportBtn.disabled = false;
    playBtn.disabled = false;
    songSelect.disabled = false;
  }
});

// Default frame hook (overwritten again on play); handles natural song end.
player.onFrame = updateUi;
player.onEnded = () => {
  pulseEl.classList.remove("on");
  playBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Finished";
};

/**
 * Application entry: fetch song catalog, build asset file index, populate picker, load initial song.
 *
 * Prefers `Hob.Riven` when present; otherwise selects the first catalog entry.
 */
async function boot(): Promise<void> {
  statusEl.textContent = "Loading catalog…";
  const catalog = await fetch("/catalog.json").then((r) => r.json());
  songs = catalog.songs as string[];
  fileIndex = await buildFileIndex();

  for (const s of songs) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    songSelect.appendChild(opt);
  }

  const initial = songs.includes("Hob.Riven") ? "Hob.Riven" : songs[0]!;
  songSelect.value = initial;
  await selectSong(initial);
}

void boot();
