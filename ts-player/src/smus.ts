import { iterIffChunks, readCString, readU16BE } from "./iff";

export const SID_REST = 0x80;
export const SID_INSTRUMENT = 0x81;
export const SID_TIME_SIG = 0x82;
export const SID_KEY_SIG = 0x83;
export const SID_DYNAMIC = 0x84;
export const SID_MIDI_CHNL = 0x85;
export const SID_MIDI_PRESET = 0x86;
export const SID_CLEF = 0x87;
export const SID_TEMPO = 0x88;

export interface SEvent {
  sid: number;
  data: number;
}

export interface SmusScore {
  tempo: number;
  volume: number;
  name: string;
  instruments: Map<number, string>;
  tracks: SEvent[][];
  path: string;
}

export function noteDurationBeats(flags: number): number {
  const division = flags & 0x07;
  const dotted = Boolean(flags & 0x08);
  const ntuplet = (flags >> 4) & 0x03;
  let beats = 4 / (1 << division);
  if (dotted) beats *= 1.5;
  if (ntuplet === 1) beats *= 2 / 3;
  else if (ntuplet === 2) beats *= 4 / 5;
  else if (ntuplet === 3) beats *= 4 / 7;
  return beats;
}

export function parseSmus(data: Uint8Array, path: string): SmusScore {
  const form = String.fromCharCode(data[0]!, data[1]!, data[2]!, data[3]!);
  const type = String.fromCharCode(data[8]!, data[9]!, data[10]!, data[11]!);
  if (form !== "FORM" || type !== "SMUS") {
    throw new Error(`Not a FORM SMUS file: ${path}`);
  }

  let tempo = 128 * 120;
  let volume = 127;
  let ntracks = 0;
  let name = path.replace(/^.*[/\\]/, "").replace(/\.smus$/i, "");
  const instruments = new Map<number, string>();
  const tracks: SEvent[][] = [];

  for (const { cid, body } of iterIffChunks(data)) {
    if (cid === "SHDR" && body.length >= 4) {
      tempo = readU16BE(body, 0);
      volume = body[2]!;
      ntracks = body[3]!;
    } else if (cid === "NAME") {
      const n = readCString(body);
      if (n) name = n;
    } else if (cid === "INS1" && body.length >= 4) {
      instruments.set(body[0]!, readCString(body, 4));
    } else if (cid === "TRAK") {
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

export function midiToName(midi: number): string {
  const n = ((midi % 12) + 12) % 12;
  const oct = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[n]}${oct}`;
}

/** Expand a track into timed display rows for the pattern editor. */
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

  rows.sort((a, b) => a.beat - b.beat || a.channel - b.channel);
  return rows;
}
