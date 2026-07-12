import "./style.css";
import type { SmusEngine } from "./engine";
import { midiToName, expandPattern, type PatternRow, type SmusScore } from "./smus";
import { AudioPlayer, buildFileIndex, loadSong, exportSongWav, downloadBlob } from "./player";
import { ScoreView } from "./score";

const ROW_H = 22;

const app = document.querySelector<HTMLDivElement>("#app")!;

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

const scopeCanvases: HTMLCanvasElement[] = [];
const noteEls: HTMLElement[] = [];
const instEls: HTMLElement[] = [];
const vuEls: HTMLElement[] = [];
const rowContainers: HTMLElement[] = [];

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
let patternByChannel: PatternRow[][] = [[], [], [], []];
let maxBeat = 0;
let currentScore: SmusScore | null = null;
let loadedInstruments: Map<number, import("./instruments").Instrument> | null = null;
let loading = false;

function hex2(n: number): string {
  return n.toString(16).toUpperCase().padStart(2, "0");
}

function buildPatternDom(score: SmusScore): void {
  const rows = expandPattern(score);
  patternByChannel = [[], [], [], []];
  maxBeat = 0;
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
      if (Math.round(b) === b && b % 4 === 0) el.classList.add("bar");
      if (!r) {
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

function scrollToBeat(beat: number): void {
  const beats = (beatRowsEl as HTMLElement & { _beats?: number[] })._beats ?? [];
  if (!beats.length) return;

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
      const p = parseFloat(vuEls[ch]!.style.width || "0") * 0.88;
      vuEls[ch]!.style.width = `${p}%`;
    }
    drawScope(ch, eng);
  }
}

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

songSelect.addEventListener("change", () => {
  void selectSong(songSelect.value);
});

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

player.onFrame = updateUi;
player.onEnded = () => {
  pulseEl.classList.remove("on");
  playBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Finished";
};

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
