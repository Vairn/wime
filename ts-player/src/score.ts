/**
 * @file score.ts — Canvas-based staff notation renderer for the SMUS player.
 *
 * {@link ScoreView} draws a four-channel (CH0–CH3) piano-roll-style score on an
 * HTML canvas: five-line treble staves, bar lines every four beats, note heads
 * with stems/flags/accidentals, rests, and a scrolling playhead. The left gutter
 * stays fixed while note content scrolls horizontally so the current beat stays
 * near the left third of the viewport. Channel colors, parchment wash, and clef
 * labels give each staff a distinct visual identity without changing pitch logic.
 */

import type { PatternRow } from "./smus";

/** Per-channel accent colors used for staff tint, notes, clefs, and rests. */
const CH_COLORS = ["#c4783a", "#5a9e7a", "#6a8ab8", "#b87a6a"];

/**
 * Maps a MIDI note number to a diatonic staff step index (C=0 … B=6 per octave).
 * Accidentals share the same vertical step as their natural neighbors; sharp
 * display is handled separately by {@link isSharp}.
 *
 * @param midi - MIDI note number (0–127).
 * @returns Cumulative diatonic step from MIDI 0 (e.g. middle C ≈ step 21).
 */
function midiToStep(midi: number): number {
  const pc = ((midi % 12) + 12) % 12;
  const oct = Math.floor(midi / 12) - 1;
  // Pitch-class → diatonic step within octave: C C# D D# E F F# G G# A A# B
  const steps = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6];
  return oct * 7 + steps[pc]!;
}

/**
 * Returns whether a MIDI pitch class requires a sharp accidental on the staff.
 *
 * @param midi - MIDI note number.
 * @returns `true` for C#, D#, F#, G#, and A# pitch classes.
 */
function isSharp(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);
}

/** Visual note-head style derived from beat duration thresholds. */
type NoteKind = "whole" | "half" | "quarter" | "eighth" | "sixteenth" | "thirtysecond";

/**
 * Classifies a beat duration into the nearest standard note value for drawing.
 * Thresholds sit between nominal values so slightly swung timings still map
 * to a readable glyph (e.g. ~1.5 beats → half note).
 *
 * @param beats - Note or rest length in beats.
 * @returns The {@link NoteKind} used for head shape, stem, and flag count.
 */
function durationKind(beats: number): NoteKind {
  if (beats >= 3.5) return "whole";
  if (beats >= 1.75) return "half";
  if (beats >= 0.875) return "quarter";
  if (beats >= 0.4375) return "eighth";
  if (beats >= 0.2) return "sixteenth";
  return "thirtysecond";
}

/**
 * Detects a dotted note: duration is roughly 1.5× the base value for its kind.
 *
 * @param beats - Actual duration in beats.
 * @param kind - Base note kind from {@link durationKind}.
 * @returns `true` when duration is between 1.2× and 1.8× the undotted base.
 */
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

/**
 * Renders SMUS pattern rows as scrolling four-staff notation on a canvas.
 *
 * Public API: construct with a canvas, call {@link setScore} when the pattern
 * changes, and {@link setBeat} each animation frame to move the playhead and
 * auto-scroll. {@link resize} runs on construction and window resize.
 */
export class ScoreView {
  canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /** Filtered note/rest rows from the current SMUS pattern. */
  private notes: PatternRow[] = [];
  /** Total beats in the score; at least 4 for empty/minimal patterns. */
  private maxBeat = 4;
  /** Current playback position in beats (drives playhead and note highlighting). */
  private beatPos = 0;
  /** Horizontal scale: CSS pixels per beat along the time axis. */
  private pxPerBeat = 48;
  /** Device pixel ratio cap (max 2) for crisp canvas backing store. */
  private dpr = 1;
  /** Vertical distance between adjacent staff systems (CH0–CH3). */
  private staffGap = 72;
  /** Top margin before the first staff line. */
  private topPad = 28;
  /** Left margin: clef gutter width before the first beat column. */
  private leftPad = 56;
  /** Horizontal scroll offset (CSS px); content translates left by this amount. */
  private scrollX = 0;

  /**
   * Binds to a canvas, acquires a 2D context, and sets up resize handling.
   *
   * @param canvas - Target element; parent width determines drawable width.
   */
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  /**
   * Sizes the canvas to the parent width and fixed staff height, applies DPR
   * scaling on the context, then redraws.
   */
  resize(): void {
    const parent = this.canvas.parentElement;
    const cssW = parent?.clientWidth || 800;
    // Four staves plus bottom padding; height is independent of parent.
    const cssH = this.topPad + this.staffGap * 4 + 20;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(cssW * this.dpr);
    this.canvas.height = Math.floor(cssH * this.dpr);
    this.canvas.style.width = `${cssW}px`;
    this.canvas.style.height = `${cssH}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.draw();
  }

  /**
   * Loads pattern data and resets playback scroll state.
   *
   * @param rows - Full SMUS pattern rows; only `note` and `rest` kinds are kept.
   * @param maxBeat - Pattern length in beats (floored at 4).
   */
  setScore(rows: PatternRow[], maxBeat: number): void {
    this.notes = rows.filter((r) => r.kind === "note" || r.kind === "rest");
    this.maxBeat = Math.max(maxBeat, 4);
    this.beatPos = 0;
    this.scrollX = 0;
    this.draw();
  }

  /**
   * Updates the playhead beat and eases horizontal scroll so the playhead sits
   * near 28% from the left edge of the viewport.
   *
   * @param beat - Current playback position in beats.
   */
  setBeat(beat: number): void {
    this.beatPos = beat;
    const cssW = this.canvas.width / this.dpr;
    const playX = this.leftPad + beat * this.pxPerBeat;
    // Target scroll keeps playhead at ~28% of visible width (read-ahead to the right).
    const target = playX - cssW * 0.28;
    // Exponential smoothing toward target (35% of remaining distance per frame).
    this.scrollX += (target - this.scrollX) * 0.35;
    this.scrollX = Math.max(0, this.scrollX);
    this.draw();
  }

  /**
   * Y coordinate of the top staff line for a channel (staff system origin).
   *
   * @param ch - Channel index 0–3.
   * @returns CSS Y of the uppermost of the five staff lines.
   */
  private staffY(ch: number): number {
    return this.topPad + ch * this.staffGap + 8;
  }

  /**
   * Converts MIDI pitch to canvas Y on a given channel's treble staff.
   * Middle line is B4 (MIDI 71); each diatonic step is 3px (half the 6px line gap).
   *
   * @param ch - Channel whose staff system to use.
   * @param midi - MIDI note number.
   * @returns CSS Y for the note head center.
   */
  private midiToY(ch: number, midi: number): number {
    const midLine = this.staffY(ch) + 2 * 6; // B4 sits on the third (middle) line
    // Treble reference: B4 (midi 71) → diatonic step 4*7+6 = 34
    const refStep = midiToStep(71);
    const step = midiToStep(midi);
    return midLine - (step - refStep) * 3; // 3px per diatonic step (half of 6px line gap)
  }

  /**
   * Full frame render: background, scrolled score content, then fixed clef gutter.
   */
  private draw(): void {
    const ctx = this.ctx;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;
    ctx.clearRect(0, 0, w, h);

    // Subtle parchment tint over the full canvas (not scrolled).
    ctx.fillStyle = "rgba(212, 196, 160, 0.06)";
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    // Scroll note content left; playhead and staves move with this transform.
    ctx.translate(-Math.floor(this.scrollX), 0);

    const endBeat = this.maxBeat + 4;
    const contentW = this.leftPad + endBeat * this.pxPerBeat;

    // Draw four channel staves (five lines each, plus channel color bar).
    for (let ch = 0; ch < 4; ch++) {
      this.drawStaff(ch, contentW);
    }

    // Measure bar lines every 4 beats, spanning all four staves.
    ctx.strokeStyle = "rgba(212, 196, 160, 0.28)";
    ctx.lineWidth = 1;
    for (let b = 0; b <= endBeat; b += 4) {
      const x = this.leftPad + b * this.pxPerBeat;
      ctx.beginPath();
      ctx.moveTo(x, this.staffY(0));
      ctx.lineTo(x, this.staffY(3) + 24);
      ctx.stroke();
    }

    // Note heads, stems, rests, and duration hints in beat order.
    for (const n of this.notes) {
      if (n.kind === "rest") this.drawRest(n);
      else if (n.kind === "note" && n.midi != null) this.drawNote(n);
    }

    // Playhead: vertical line plus downward triangle at the top.
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

    // Fixed clef gutter: opaque strip over scrolled content on the left.
    ctx.fillStyle = "rgba(8, 14, 11, 0.92)";
    ctx.fillRect(0, 0, this.leftPad - 4, h);
    ctx.strokeStyle = "rgba(196, 120, 58, 0.3)";
    ctx.beginPath();
    ctx.moveTo(this.leftPad - 4, 0);
    ctx.lineTo(this.leftPad - 4, h);
    ctx.stroke();

    // Clef symbols and CH labels redrawn in screen space (not scrolled).
    for (let ch = 0; ch < 4; ch++) {
      this.drawClefGutter(ch);
    }
  }

  /**
   * Draws one channel's five horizontal staff lines and a narrow color accent bar.
   *
   * @param ch - Channel index 0–3.
   * @param contentW - Right edge X of the staff lines in beat-scrolled coordinates.
   */
  private drawStaff(ch: number, contentW: number): void {
    const ctx = this.ctx;
    const y0 = this.staffY(ch);
    const color = CH_COLORS[ch]!;

    // Five lines, 6px apart; extend slightly left of the beat origin for continuity.
    ctx.strokeStyle = "rgba(212, 196, 160, 0.35)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = y0 + i * 6;
      ctx.beginPath();
      ctx.moveTo(this.leftPad - 8, y);
      ctx.lineTo(contentW, y);
      ctx.stroke();
    }

    // Vertical channel tint at the left edge of the staff system.
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(this.leftPad - 8, y0 - 2, 3, 28);
    ctx.globalAlpha = 1;
  }

  /**
   * Draws the non-scrolling left gutter: staff line stubs, G-clef glyph, and CH label.
   *
   * @param ch - Channel index 0–3.
   */
  private drawClefGutter(ch: number): void {
    const ctx = this.ctx;
    const y0 = this.staffY(ch);
    const color = CH_COLORS[ch]!;

    // Short staff segments from the left margin into the gutter (before leftPad).
    ctx.strokeStyle = "rgba(212, 196, 160, 0.4)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = y0 + i * 6;
      ctx.beginPath();
      ctx.moveTo(8, y);
      ctx.lineTo(this.leftPad - 8, y);
      ctx.stroke();
    }

    // Stylized treble G-clef (Unicode 𝄞) centered in the gutter.
    ctx.fillStyle = color;
    ctx.font = "700 22px Georgia, 'Times New Roman', serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("𝄞", 26, y0 + 14);

    ctx.font = "600 9px 'IBM Plex Mono', monospace";
    ctx.fillStyle = color;
    ctx.fillText(`CH${ch}`, 26, y0 + 34);
  }

  /**
   * Renders a single note: ledger lines, accidental, elliptical head, stem, flags,
   * dot, and optional duration underline for longer values.
   *
   * @param n - Pattern row with `kind === "note"` and valid `midi`.
   */
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

    // Ledger lines above or below the five-line staff when pitch exceeds span.
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

    // Sharp accidental to the left of the note head when required.
    if (isSharp(midi)) {
      ctx.fillStyle = active ? "#e8a05c" : color;
      ctx.font = "14px Georgia, serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("♯", x - 12, y);
    }

    // Note head: filled ellipse for quarter and shorter; hollow for whole/half.
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

    // Stem and beamed flags (whole notes have neither).
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

      // Curved flags: 1 for eighth, 2 for sixteenth, 3 for thirty-second.
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

    // Augmentation dot for dotted durations.
    if (dotted) {
      ctx.fillStyle = active ? "#e8a05c" : color;
      ctx.beginPath();
      ctx.arc(x + 10, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Duration beam hint: horizontal segment under the staff for notes ≥ 1.5 beats.
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

  /**
   * Renders a rest symbol at the center of the channel staff, with a bitmap
   * fallback when SMuFL glyphs are unavailable in the font.
   *
   * @param n - Pattern row with `kind === "rest"`.
   */
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
    // If the font did not render the glyph, draw a simple rectangular rest shape.
    ctx.fillText(glyph, x, y);
    if (ctx.measureText(glyph).width < 2) {
      ctx.fillRect(x - 3, y - 6, 6, 3);
      ctx.fillRect(x - 1, y - 3, 2, 10);
    }
  }
}
