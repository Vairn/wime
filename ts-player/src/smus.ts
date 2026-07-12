/**
 * SMUS score parsing — Aegis Sonix / Instant Music IFF-SMUS format.
 *
 * A score is a FORM SMUS containing:
 *   SHDR  — tempo, volume, track count
 *   NAME  — song title
 *   INS1  — instrument register → name
 *   TRAK  — sequence of 2-byte events (sid, data) per track
 */
import { iterIffChunks, readCString, readU16BE } from "./iff";

/** Event opcodes: values ≥ 0x80 are controls; below that, sid is a MIDI note. */
export const SID_REST = 0x80;
export const SID_INSTRUMENT = 0x81;
export const SID_TIME_SIG = 0x82;
export const SID_KEY_SIG = 0x83;
export const SID_DYNAMIC = 0x84;
export const SID_MIDI_CHNL = 0x85;
export const SID_MIDI_PRESET = 0x86;
export const SID_CLEF = 0x87;
/** Instant Music / Sonix inline tempo change (BPM in data byte). */
export const SID_TEMPO = 0x88;

/** One SMUS track event: opcode/note in `sid`, parameter/flags in `data`. */
export interface SEvent {
  sid: number;
  data: number;
}

/** Fully parsed SMUS score ready for the sequencer / UI. */
export interface SmusScore {
  /** SHDR tempo: 128ths of a quarter-note per minute (÷128 → BPM). */
  tempo: number;
  volume: number;
  name: string;
  /** Instrument register index → instrument file stem. */
  instruments: Map<number, string>;
  tracks: SEvent[][];
  path: string;
}

/**
 * Convert an SNote / rest duration flags byte into quarter-note beats.
 *
 * Bits:
 *   0..2  division  (0=whole, 1=half, 2=quarter, 3=8th, …)
 *   3     dotted    (×1.5)
 *   4..5  ntuplet   (1=triplet, 2=quintuplet, 3=septuplet)
 */
export function noteDurationBeats(flags: number): number {
  const division = flags & 0x07;
  const dotted = Boolean(flags & 0x08);
  const ntuplet = (flags >> 4) & 0x03;
  // Whole note = 4 beats; each division step halves that.
  let beats = 4 / (1 << division);
  if (dotted) beats *= 1.5;
  if (ntuplet === 1) beats *= 2 / 3;
  else if (ntuplet === 2) beats *= 4 / 5;
  else if (ntuplet === 3) beats *= 4 / 7;
  return beats;
}

/**
 * Parse a raw FORM SMUS file into a SmusScore.
 * Walks IFF chunks and assembles header, instrument map, and track event lists.
 */
export function parseSmus(data: Uint8Array, path: string): SmusScore {
  // Validate FORM wrapper + SMUS type id at bytes 8..11.
  const form = String.fromCharCode(data[0]!, data[1]!, data[2]!, data[3]!);
  const type = String.fromCharCode(data[8]!, data[9]!, data[10]!, data[11]!);
  if (form !== "FORM" || type !== "SMUS") {
    throw new Error(`Not a FORM SMUS file: ${path}`);
  }

  // Defaults match Instant Music when SHDR is missing.
  let tempo = 128 * 120;
  let volume = 127;
  let ntracks = 0;
  let name = path.replace(/^.*[/\\]/, "").replace(/\.smus$/i, "");
  const instruments = new Map<number, string>();
  const tracks: SEvent[][] = [];

  for (const { cid, body } of iterIffChunks(data)) {
    if (cid === "SHDR" && body.length >= 4) {
      // tempo:u16, volume:u8, ntracks:u8
      tempo = readU16BE(body, 0);
      volume = body[2]!;
      ntracks = body[3]!;
    } else if (cid === "NAME") {
      const n = readCString(body);
      if (n) name = n;
    } else if (cid === "INS1" && body.length >= 4) {
      // byte0 = register, bytes4+ = instrument name
      instruments.set(body[0]!, readCString(body, 4));
    } else if (cid === "TRAK") {
      // Events are packed as (sid, data) pairs.
      const evs: SEvent[] = [];
      for (let i = 0; i + 1 < body.length; i += 2) {
        evs.push({ sid: body[i]!, data: body[i + 1]! });
      }
      tracks.push(evs);
    }
  }

  if (!tracks.length && ntracks) {
    throw new Error(`SHDR says ${ntracks} tracks but none found`);
  }

  return { tempo, volume, name, instruments, tracks, path };
}

const NOTE_NAMES = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"];

/** Format a MIDI note number as a tracker-style name, e.g. 60 → "C-4". */
export function midiToName(midi: number): string {
  const n = ((midi % 12) + 12) % 12;
  const oct = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[n]}${oct}`;
}

/**
 * One display row for the pattern editor / score view.
 * Timed in quarter-note beats from the start of the song.
 */
export interface PatternRow {
  beat: number;
  channel: number;
  kind: "note" | "rest" | "ctrl";
  midi?: number;
  noteName?: string;
  duration: number;
  instrument?: number;
  volume?: number;
  chord?: boolean;
  label?: string;
}

/**
 * Expand all tracks into a flat, time-sorted list of PatternRows.
 *
 * Walks each track's event stream, advancing a beat cursor for notes/rests.
 * Chord notes (data bit 7) share the same beat until a non-chord note lands.
 * Control events (instrument / volume / tempo) are recorded at the current beat
 * without advancing time.
 */
export function expandPattern(score: SmusScore): PatternRow[] {
  const rows: PatternRow[] = [];
  const nCh = Math.min(4, score.tracks.length);

  for (let ch = 0; ch < nCh; ch++) {
    const events = score.tracks[ch]!;
    let beat = 0;
    let inst = 0;
    let vol = 127;
    let i = 0;
    while (i < events.length) {
      const ev = events[i]!;
      i++;
      if (ev.sid < 0x80) {
        // Note event — sid is MIDI number.
        const chord = Boolean(ev.data & 0x80);
        const flags = ev.data & 0x3f;
        const dur = noteDurationBeats(flags);
        rows.push({
          beat,
          channel: ch,
          kind: "note",
          midi: ev.sid,
          noteName: midiToName(ev.sid),
          duration: dur,
          instrument: inst,
          volume: vol,
          chord,
        });
        // Chord members don't advance time; the final non-chord note does.
        if (!chord) beat += dur;
      } else if (ev.sid === SID_REST) {
        const dur = noteDurationBeats(ev.data & 0x3f);
        rows.push({ beat, channel: ch, kind: "rest", duration: dur, noteName: "---" });
        beat += dur;
      } else if (ev.sid === SID_INSTRUMENT) {
        inst = ev.data;
        rows.push({
          beat,
          channel: ch,
          kind: "ctrl",
          duration: 0,
          instrument: inst,
          label: `INS ${inst}`,
        });
      } else if (ev.sid === SID_DYNAMIC) {
        vol = ev.data;
        rows.push({
          beat,
          channel: ch,
          kind: "ctrl",
          duration: 0,
          volume: vol,
          label: `VOL ${vol}`,
        });
      } else if (ev.sid === SID_TEMPO) {
        rows.push({
          beat,
          channel: ch,
          kind: "ctrl",
          duration: 0,
          label: `TMP ${ev.data}`,
        });
      }
    }
  }

  // Sort so the UI can walk events in time order across channels.
  rows.sort((a, b) => a.beat - b.beat || a.channel - b.channel);
  return rows;
}
