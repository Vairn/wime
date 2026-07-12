import type { PatternRow } from "./smus";

const CH_COLORS = ["#c4783a", "#5a9e7a", "#6a8ab8", "#b87a6a"];

/** Diatonic staff step for MIDI (C=0 … B=6), ignoring accidentals' vertical offset. */
function midiToStep(midi: number): number {
  const pc = ((midi % 12) + 12) % 12;
  const oct = Math.floor(midi / 12) - 1;
  // C C# D D# E F F# G G# A A# B
  const steps = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6];
  return oct * 7 + steps[pc]!;
}

function isSharp(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);
}

type NoteKind = "whole" | "half" | "quarter" | "eighth" | "sixteenth" | "thirtysecond";

function durationKind(beats: number): NoteKind {
  if (beats >= 3.5) return "whole";
  if (beats >= 1.75) return "half";
  if (beats >= 0.875) return "quarter";
  if (beats >= 0.4375) return "eighth";
  if (beats >= 0.2) return "sixteenth";
  return "thirtysecond";
}

function isDotted(beats: number, kind: NoteKind): boolean {
  const base =
    kind === "whole"
      ? 4
      : kind === "half"
        ? 2
        : kind === "quarter"
          ? 1
          : kind === "eighth"
            ? 0.5
            : kind === "sixteenth"
              ? 0.25
              : 0.125;
  return beats > base * 1.2 && beats < base * 1.8;
}

export class ScoreView {
  canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private notes: PatternRow[] = [];
  private maxBeat = 4;
  private beatPos = 0;
  private pxPerBeat = 48;
  private dpr = 1;
  private staffGap = 72;
  private topPad = 28;
  private leftPad = 56;
  private scrollX = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize(): void {
    const parent = this.canvas.parentElement;
    const cssW = parent?.clientWidth || 800;
    const cssH = this.topPad + this.staffGap * 4 + 20;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(cssW * this.dpr);
    this.canvas.height = Math.floor(cssH * this.dpr);
    this.canvas.style.width = `${cssW}px`;
    this.canvas.style.height = `${cssH}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.draw();
  }

  setScore(rows: PatternRow[], maxBeat: number): void {
    this.notes = rows.filter((r) => r.kind === "note" || r.kind === "rest");
    this.maxBeat = Math.max(maxBeat, 4);
    this.beatPos = 0;
    this.scrollX = 0;
    this.draw();
  }

  setBeat(beat: number): void {
    this.beatPos = beat;
    const cssW = this.canvas.width / this.dpr;
    const playX = this.leftPad + beat * this.pxPerBeat;
    const target = playX - cssW * 0.28;
    this.scrollX += (target - this.scrollX) * 0.35;
    this.scrollX = Math.max(0, this.scrollX);
    this.draw();
  }

  private staffY(ch: number): number {
    return this.topPad + ch * this.staffGap + 8;
  }

  /** Y of middle line (B4 for treble). */
  private midiToY(ch: number, midi: number): number {
    const midLine = this.staffY(ch) + 2 * 6; // B4 = step of B4
    // Treble: B4 (midi 71) is middle line. Step of B4 = 4*7+6 = 34
    const refStep = midiToStep(71);
    const step = midiToStep(midi);
    return midLine - (step - refStep) * 3; // 3px per diatonic step (half of 6px line gap)
  }

  private draw(): void {
    const ctx = this.ctx;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;
    ctx.clearRect(0, 0, w, h);

    // parchment wash
    ctx.fillStyle = "rgba(212, 196, 160, 0.06)";
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(-Math.floor(this.scrollX), 0);

    const endBeat = this.maxBeat + 4;
    const contentW = this.leftPad + endBeat * this.pxPerBeat;

    for (let ch = 0; ch < 4; ch++) {
      this.drawStaff(ch, contentW);
    }

    // bar lines every 4 beats
    ctx.strokeStyle = "rgba(212, 196, 160, 0.28)";
    ctx.lineWidth = 1;
    for (let b = 0; b <= endBeat; b += 4) {
      const x = this.leftPad + b * this.pxPerBeat;
      ctx.beginPath();
      ctx.moveTo(x, this.staffY(0));
      ctx.lineTo(x, this.staffY(3) + 24);
      ctx.stroke();
    }

    // notes
    for (const n of this.notes) {
      if (n.kind === "rest") this.drawRest(n);
      else if (n.kind === "note" && n.midi != null) this.drawNote(n);
    }

    // playhead
    const px = this.leftPad + this.beatPos * this.pxPerBeat;
    ctx.strokeStyle = "rgba(232, 160, 92, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 4);
    ctx.lineTo(px, h - 4);
    ctx.stroke();
    ctx.fillStyle = "#e8a05c";
    ctx.beginPath();
    ctx.moveTo(px, 4);
    ctx.lineTo(px - 5, 12);
    ctx.lineTo(px + 5, 12);
    ctx.closePath();
    ctx.fill();

    ctx.restore();

    // fixed clef gutter
    ctx.fillStyle = "rgba(8, 14, 11, 0.92)";
    ctx.fillRect(0, 0, this.leftPad - 4, h);
    ctx.strokeStyle = "rgba(196, 120, 58, 0.3)";
    ctx.beginPath();
    ctx.moveTo(this.leftPad - 4, 0);
    ctx.lineTo(this.leftPad - 4, h);
    ctx.stroke();

    for (let ch = 0; ch < 4; ch++) {
      this.drawClefGutter(ch);
    }
  }

  private drawStaff(ch: number, contentW: number): void {
    const ctx = this.ctx;
    const y0 = this.staffY(ch);
    const color = CH_COLORS[ch]!;

    ctx.strokeStyle = "rgba(212, 196, 160, 0.35)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = y0 + i * 6;
      ctx.beginPath();
      ctx.moveTo(this.leftPad - 8, y);
      ctx.lineTo(contentW, y);
      ctx.stroke();
    }

    // channel tint under staff
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(this.leftPad - 8, y0 - 2, 3, 28);
    ctx.globalAlpha = 1;
  }

  private drawClefGutter(ch: number): void {
    const ctx = this.ctx;
    const y0 = this.staffY(ch);
    const color = CH_COLORS[ch]!;

    ctx.strokeStyle = "rgba(212, 196, 160, 0.4)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = y0 + i * 6;
      ctx.beginPath();
      ctx.moveTo(8, y);
      ctx.lineTo(this.leftPad - 8, y);
      ctx.stroke();
    }

    // stylized G-clef mark
    ctx.fillStyle = color;
    ctx.font = "700 22px Georgia, 'Times New Roman', serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("𝄞", 26, y0 + 14);

    ctx.font = "600 9px 'IBM Plex Mono', monospace";
    ctx.fillStyle = color;
    ctx.fillText(`CH${ch}`, 26, y0 + 34);
  }

  private drawNote(n: PatternRow): void {
    const ctx = this.ctx;
    const midi = n.midi!;
    const ch = n.channel;
    const color = CH_COLORS[ch]!;
    const x = this.leftPad + n.beat * this.pxPerBeat + 10;
    const y = this.midiToY(ch, midi);
    const kind = durationKind(n.duration);
    const dotted = isDotted(n.duration, kind);
    const active = this.beatPos >= n.beat && this.beatPos < n.beat + n.duration;

    // ledger lines
    const y0 = this.staffY(ch);
    const top = y0;
    const bot = y0 + 24;
    ctx.strokeStyle = "rgba(212, 196, 160, 0.45)";
    ctx.lineWidth = 1;
    if (y < top) {
      for (let ly = top - 6; ly >= y - 1; ly -= 6) {
        ctx.beginPath();
        ctx.moveTo(x - 8, ly);
        ctx.lineTo(x + 8, ly);
        ctx.stroke();
      }
    } else if (y > bot) {
      for (let ly = bot + 6; ly <= y + 1; ly += 6) {
        ctx.beginPath();
        ctx.moveTo(x - 8, ly);
        ctx.lineTo(x + 8, ly);
        ctx.stroke();
      }
    }

    // accidental
    if (isSharp(midi)) {
      ctx.fillStyle = active ? "#e8a05c" : color;
      ctx.font = "14px Georgia, serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("♯", x - 12, y);
    }

    const filled = kind !== "whole" && kind !== "half";
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-0.2);
    ctx.beginPath();
    ctx.ellipse(0, 0, 5.5, 4, 0, 0, Math.PI * 2);
    if (filled) {
      ctx.fillStyle = active ? "#e8a05c" : color;
      ctx.fill();
    } else {
      ctx.strokeStyle = active ? "#e8a05c" : color;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
    ctx.restore();

    // stem (except whole)
    if (kind !== "whole") {
      const stemUp = midi < 71;
      const stemX = stemUp ? x + 5 : x - 5;
      const stemY2 = stemUp ? y - 22 : y + 22;
      ctx.strokeStyle = active ? "#e8a05c" : color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(stemX, y);
      ctx.lineTo(stemX, stemY2);
      ctx.stroke();

      // flags
      const flags =
        kind === "eighth" ? 1 : kind === "sixteenth" ? 2 : kind === "thirtysecond" ? 3 : 0;
      for (let f = 0; f < flags; f++) {
        const fy = stemUp ? stemY2 + f * 5 : stemY2 - f * 5;
        ctx.beginPath();
        if (stemUp) {
          ctx.moveTo(stemX, fy);
          ctx.quadraticCurveTo(stemX + 10, fy + 4, stemX + 8, fy + 10);
        } else {
          ctx.moveTo(stemX, fy);
          ctx.quadraticCurveTo(stemX - 10, fy - 4, stemX - 8, fy - 10);
        }
        ctx.stroke();
      }
    }

    if (dotted) {
      ctx.fillStyle = active ? "#e8a05c" : color;
      ctx.beginPath();
      ctx.arc(x + 10, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // duration beam hint as thin duration line under staff for long notes
    if (n.duration >= 1.5) {
      const x2 = this.leftPad + (n.beat + n.duration) * this.pxPerBeat - 4;
      ctx.strokeStyle = active ? "rgba(232,160,92,0.35)" : `${color}44`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 8, this.staffY(ch) + 30);
      ctx.lineTo(Math.max(x + 8, x2), this.staffY(ch) + 30);
      ctx.stroke();
    }
  }

  private drawRest(n: PatternRow): void {
    const ctx = this.ctx;
    const ch = n.channel;
    const x = this.leftPad + n.beat * this.pxPerBeat + 10;
    const y = this.staffY(ch) + 12;
    const color = CH_COLORS[ch]!;
    const kind = durationKind(n.duration);
    const active = this.beatPos >= n.beat && this.beatPos < n.beat + n.duration;

    ctx.fillStyle = active ? "rgba(232,160,92,0.55)" : `${color}66`;
    ctx.font = "16px Georgia, serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const glyph =
      kind === "whole" || kind === "half"
        ? "𝄻"
        : kind === "quarter"
          ? "𝄽"
          : kind === "eighth"
            ? "𝄾"
            : " belieb";
    // Fallback if glyphs missing: draw a simple rest
    ctx.fillText(glyph, x, y);
    if (ctx.measureText(glyph).width < 2) {
      ctx.fillRect(x - 3, y - 6, 6, 3);
      ctx.fillRect(x - 1, y - 3, 2, 10);
    }
  }
}
