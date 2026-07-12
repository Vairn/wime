import type { Instrument } from "./instruments";
import { defaultInstrument, sampleOctaveForMidi, sonixRateUnits } from "./instruments";
import {
  SID_DYNAMIC,
  SID_INSTRUMENT,
  SID_REST,
  SID_TEMPO,
  noteDurationBeats,
  type SEvent,
  type SmusScore,
} from "./smus";

/** Amiga Paula stereo: AUD0/AUD3 → left, AUD1/AUD2 → right */
export const CHANNEL_PAN = [0, 1, 1, 0] as const;

const NOTE_PERIOD = [
  0x8000, 0x78d1, 0x7209, 0x6ba2, 0x6598, 0x5fe4, 0x5a82, 0x556e, 0x50a3, 0x4c1c, 0x47d6,
  0x43ce,
];

export interface Voice {
  active: boolean;
  channel: number;
  instrument: Instrument | null;
  pos: number;
  step: number;
  vol: number;
  samplesLeft: number;
  release: boolean;
  envLevel: number;
  envFixed: number;
  envStage: number;
  lfoPhase: number;
  lfoFrozen: boolean;
  lfoMod: number;
  vibPhase: number;
  vibDelayLeft: number;
  sampleWave: Float32Array | null;
  sampleLoopStart: number;
  sampleLoopEnd: number;
  noteFreq: number;
  inHold: boolean;
  midi: number;
  peak: number;
}

function freshVoice(channel: number): Voice {
  return {
    active: false,
    channel,
    instrument: null,
    pos: 0,
    step: 0,
    vol: 0,
    samplesLeft: 0,
    release: false,
    envLevel: 0,
    envFixed: 0,
    envStage: 0,
    lfoPhase: 0,
    lfoFrozen: false,
    lfoMod: 0,
    vibPhase: 0,
    vibDelayLeft: 0,
    sampleWave: null,
    sampleLoopStart: 0,
    sampleLoopEnd: 0,
    noteFreq: 440,
    inHold: false,
    midi: 60,
    peak: 0,
  };
}

interface TrackState {
  events: SEvent[];
  index: number;
  wait: number;
  instrumentReg: number;
  volume: number;
  chordNotes: Array<[number, number]>;
  done: boolean;
}

export class SmusEngine {
  score: SmusScore;
  instruments: Map<number, Instrument>;
  sr: number;
  master: number;
  bpm: number;
  beatSamples: number;
  tracks: TrackState[];
  voices: Voice[];
  scoreVolume: number;
  /** Elapsed beats for UI playhead */
  beatPos = 0;

  constructor(
    score: SmusScore,
    instruments: Map<number, Instrument>,
    sampleRate = 44100,
    masterVolume = 0.28,
  ) {
    this.score = score;
    this.instruments = instruments;
    this.sr = sampleRate;
    this.master = masterVolume;
    this.bpm = Math.max(score.tempo / 128, 1);
    this.beatSamples = (60 / this.bpm) * sampleRate;
    this.tracks = score.tracks.map((t) => ({
      events: t.slice(),
      index: 0,
      wait: 0,
      instrumentReg: 0,
      volume: 1,
      chordNotes: [],
      done: false,
    }));
    this.voices = [0, 1, 2, 3].map(freshVoice);
    this.scoreVolume = score.volume / 127;
    for (const tr of this.tracks) this.primeTrack(tr);
  }

  private instForReg(reg: number): Instrument {
    return this.instruments.get(reg) ?? defaultInstrument(`reg${reg}`);
  }

  private primeTrack(tr: TrackState): void {
    while (tr.index < tr.events.length) {
      const ev = tr.events[tr.index]!;
      if (ev.sid < 0x80 || ev.sid === SID_REST) break;
      this.handleControl(tr, ev);
      tr.index++;
    }
  }

  private handleControl(tr: TrackState, ev: SEvent): void {
    if (ev.sid === SID_INSTRUMENT) tr.instrumentReg = ev.data;
    else if (ev.sid === SID_DYNAMIC) tr.volume = Math.max(ev.data, 1) / 127;
    else if (ev.sid === SID_TEMPO && ev.data > 0) {
      this.bpm = ev.data;
      this.beatSamples = (60 / this.bpm) * this.sr;
    }
  }

  private startVoice(ch: number, midi: number, flags: number, tr: TrackState): void {
    const inst = this.instForReg(tr.instrumentReg);
    const durBeats = noteDurationBeats(flags);
    let nSamples = Math.max(1, Math.floor(durBeats * this.beatSamples));
    const freq = 440 * 2 ** ((midi - 69) / 12);

    let sampleWave: Float32Array | null = null;
    let sampleLoopStart = 0;
    let sampleLoopEnd = 0;
    let step: number;

    if (inst.kind === "synth") {
      step = (freq * 128) / this.sr;
    } else if (inst.kind === "sample" && inst.ssData) {
      const octv = sampleOctaveForMidi(midi, inst.ssLo, inst.ssHi);
      const oneshot = inst.ssOneshot;
      const repeat = inst.ssRepeat;
      const lo = inst.ssLo;
      const offset = oneshot * ((1 << octv) - (1 << lo));
      const length = oneshot << octv;
      sampleWave = inst.ssData.subarray(offset, offset + length);
      if (sampleWave.length === 0) sampleWave = inst.wave;
      const wlen = sampleWave.length;
      if (repeat > 0 && repeat < oneshot && wlen > 0) {
        sampleLoopStart = Math.min(wlen - 1, repeat << octv);
        sampleLoopEnd = Math.min(wlen, oneshot << octv);
        if (sampleLoopEnd - sampleLoopStart < 2) {
          sampleLoopStart = 0;
          sampleLoopEnd = 0;
        } else {
          const ls = sampleLoopStart;
          const le = sampleLoopEnd;
          const loop = sampleWave.slice(ls, le);
          const fade = Math.min(Math.max(Math.floor((le - ls) / 4), 2), 32);
          if (fade >= 2) {
            for (let i = 0; i < fade; i++) {
              const t = (i + 1) / fade;
              const a = loop[i]!;
              const b = loop[le - ls - fade + i]!;
              loop[i] = a * t + b * (1 - t);
            }
            sampleWave = sampleWave.slice();
            sampleWave.set(loop, ls);
          }
        }
      }
      const noteInOct = midi % 12;
      const rate = inst.baseRate * (NOTE_PERIOD[0]! / NOTE_PERIOD[noteInOct]!);
      step = rate / this.sr;
      if (sampleLoopEnd === 0 && wlen > 0) {
        nSamples = Math.min(nSamples, Math.floor(wlen / Math.max(step, 1e-6)) + Math.floor(this.sr / 20));
      }
    } else {
      const baseFreq = 440 * 2 ** ((inst.baseMidi - 69) / 12);
      step = (inst.baseRate / this.sr) * (freq / Math.max(baseFreq, 1e-6));
      sampleWave = inst.wave;
      sampleLoopStart = inst.loopStart;
      sampleLoopEnd = inst.loopEnd;
    }

    let vibDelay = 0;
    if (inst.vibDelay > 0) {
      vibDelay = Math.floor((sonixRateUnits(inst.vibDelay) / 8000) * this.sr);
    }

    const v = this.voices[ch]!;
    v.active = true;
    v.channel = ch;
    v.instrument = inst;
    v.pos = 0;
    v.step = step;
    v.vol = tr.volume * inst.volume * this.scoreVolume;
    v.samplesLeft = nSamples;
    v.release = false;
    v.envLevel = 0;
    v.envFixed = 0;
    v.envStage = 0;
    v.lfoPhase = 0;
    v.lfoFrozen = false;
    v.lfoMod = 0;
    v.vibPhase = 0;
    v.vibDelayLeft = vibDelay;
    v.sampleWave = sampleWave;
    v.sampleLoopStart = sampleLoopStart;
    v.sampleLoopEnd = sampleLoopEnd;
    v.noteFreq = freq;
    v.inHold = false;
    v.midi = midi;
    v.peak = 0;
  }

  private consumeEvent(tr: TrackState, ch: number): void {
    if (tr.index >= tr.events.length) {
      tr.done = true;
      return;
    }
    const ev = tr.events[tr.index]!;
    tr.index++;

    if (ev.sid < 0x80) {
      const chord = Boolean(ev.data & 0x80);
      const flags = ev.data & 0x3f;
      const midi = ev.sid;
      if (chord) {
        tr.chordNotes.push([midi, flags]);
        this.consumeEvent(tr, ch);
        return;
      }
      const notes = tr.chordNotes.concat([[midi, flags]]);
      tr.chordNotes = [];
      this.startVoice(ch, notes[0]![0], notes[0]![1], tr);
      for (let n = 1; n < notes.length; n++) {
        const free = this.voices.findIndex((v) => !v.active);
        if (free >= 0) this.startVoice(free, notes[n]![0], notes[n]![1], tr);
      }
      tr.wait = noteDurationBeats(flags);
      return;
    }

    if (ev.sid === SID_REST) {
      tr.wait = noteDurationBeats(ev.data & 0x3f);
      return;
    }

    this.handleControl(tr, ev);
    this.consumeEvent(tr, ch);
  }

  private advanceTracks(beats: number): void {
    this.beatPos += beats;
    for (let ch = 0; ch < Math.min(4, this.tracks.length); ch++) {
      const tr = this.tracks[ch]!;
      if (tr.done) continue;
      tr.wait -= beats;
      while (tr.wait <= 1e-9 && !tr.done) {
        this.consumeEvent(tr, ch);
        if (tr.wait <= 1e-9 && !tr.done && tr.index >= tr.events.length) tr.done = true;
      }
    }
  }

  private sonixEnvStep(
    v: Voice,
    n: number,
    envOut: Float32Array,
    bankOut: Float32Array | null,
  ): void {
    const inst = v.instrument!;
    const sr = this.sr;
    const levels = inst.envLevels;

    const rates = inst.envRates.map((r) => {
      const units = sonixRateUnits(r);
      const secs = Math.max(0.008, units / 2500);
      return 255 / (secs * sr);
    });

    let env = v.envFixed;
    let stage = v.envStage;
    let lfo = v.lfoPhase;
    let frozen = v.lfoFrozen;
    let modHeld = v.lfoMod;
    let gateLeft = v.samplesLeft;

    const lfoSpeed = inst.lfoRate || inst.lfoInc;
    const useLfo =
      (inst.lfoEnable || inst.fMod || inst.volMod || inst.pitchMod) &&
      lfoSpeed > 0 &&
      inst.modTable != null;
    let lfoStep = 0;
    if (useLfo) {
      const lfoHz = inst.lfoOneshot
        ? 0.35 + (lfoSpeed / 255) * 5
        : 0.15 + (lfoSpeed / 255) * 1.2;
      lfoStep = (lfoHz * 256) / sr;
    }

    const modTable = inst.modTable;
    const fBase = inst.fBase;
    const fEnv = inst.fEnv;
    const fMod = inst.fMod;

    for (let i = 0; i < n; i++) {
      if (gateLeft <= 0 && stage < 3) {
        stage = 3;
        v.release = true;
      }

      let target = levels[Math.min(stage, 3)]!;
      if (stage >= 3) target = 0;
      let spd = rates[Math.min(stage, 3)]!;
      if (stage >= 3 && spd < 1e-6) spd = 255 / (0.05 * sr);
      const dist = Math.abs(env - target);
      if (dist <= spd) {
        env = target;
        if (stage < 2) stage++;
      } else if (env < target) env += spd;
      else env -= spd;

      let mod = modHeld;
      if (modTable && lfoStep > 0 && !frozen) {
        mod = modTable[Math.floor(lfo) & 255]! * 128;
        lfo += lfoStep;
        if (inst.lfoOneshot && lfo >= 254) {
          lfo = 254;
          frozen = true;
          mod = modTable[254]! * 128;
        } else if (!inst.lfoOneshot && lfo >= 256) {
          lfo -= 256;
        }
        modHeld = mod;
      }

      const envI = Math.max(0, Math.min(255, Math.floor(env)));
      envOut[i] = envI / 255;

      if (bankOut) {
        let filt = 255 - fBase - ((envI * fEnv) >> 8) + Math.floor((mod * fMod) / 256);
        filt = Math.max(0, Math.min(255, filt));
        bankOut[i] = filt >> 2;
      }

      if (gateLeft > 0) gateLeft--;
    }

    v.envFixed = env;
    v.envStage = stage;
    v.lfoPhase = lfo;
    v.lfoFrozen = frozen;
    v.lfoMod = modHeld;
    v.envLevel = n ? envOut[n - 1]! : v.envLevel;
    v.samplesLeft = gateLeft;
  }

  private renderVoice(v: Voice, n: number, out: Float32Array): void {
    const inst = v.instrument!;
    out.fill(0);
    const env = new Float32Array(n);

    if (inst.kind === "synth" && inst.filterBanks) {
      const bank = new Float32Array(n);
      this.sonixEnvStep(v, n, env, bank);
      let pos = v.pos;
      let peak = 0;
      for (let i = 0; i < n; i++) {
        const idx = Math.floor(pos) & 127;
        const b0 = Math.max(0, Math.min(63, Math.floor(bank[i]!)));
        const b1 = Math.min(63, b0 + 1);
        const frac = bank[i]! - b0;
        const s =
          inst.filterBanks[b0]![idx]! * (1 - frac) + inst.filterBanks[b1]![idx]! * frac;
        const amp = inst.volEnv ? env[i]! : 1;
        const sample = s * amp * v.vol * 1.4;
        out[i] = sample;
        const a = Math.abs(sample);
        if (a > peak) peak = a;
        pos += v.step;
      }
      v.pos = pos;
      v.peak = peak;
      if (v.envFixed <= 1 && (v.envStage >= 3 || inst.envLevels[Math.min(v.envStage, 3)] === 0)) {
        v.active = false;
      }
      return;
    }

    const wave = v.sampleWave ?? inst.wave;
    const ls = v.sampleLoopStart;
    const le = v.sampleLoopEnd;
    const wlen = wave.length;
    if (!wlen) {
      v.active = false;
      return;
    }

    this.sonixEnvStep(v, n, env, null);

    const baseStep = v.step;
    let vibDelay = v.vibDelayLeft;
    let vibPhase = v.vibPhase;
    const vibHz = 0.8 + (inst.vibRate / 255) * 6;
    const depth = (inst.vibDepth / 128) * 0.015;
    const useVib = inst.vibDepth > 0 && inst.vibRate > 0;

    let pos = v.pos;
    let peak = 0;
    let take = n;

    if (le > ls) {
      const ll = le - ls;
      for (let i = 0; i < n; i++) {
        let step = baseStep;
        if (useVib) {
          if (vibDelay > 0) vibDelay--;
          else {
            step = baseStep * (1 + depth * Math.sin(vibPhase));
            vibPhase += (2 * Math.PI * vibHz) / this.sr;
          }
        }
        let idxF: number;
        if (pos < le) idxF = Math.min(pos, wlen - 1.001);
        else idxF = ls + ((pos - ls) % ll);

        let i0 = Math.floor(idxF);
        let i1 = i0 + 1;
        const frac = idxF - i0;
        if (pos >= le) {
          if (i1 >= le) i1 = ls;
          i0 = Math.min(Math.max(i0, ls), le - 1);
        } else {
          i0 = Math.min(Math.max(i0, 0), wlen - 1);
          i1 = Math.min(i1, wlen - 1);
        }
        const sample = (wave[i0]! * (1 - frac) + wave[i1]! * frac) * env[i]! * v.vol;
        out[i] = sample;
        const a = Math.abs(sample);
        if (a > peak) peak = a;
        pos += step;
      }
      if (pos >= le) {
        v.pos = ls + ((pos - ls) % ll);
        v.inHold = true;
      } else {
        v.pos = pos;
      }
      v.vibDelayLeft = vibDelay;
      v.vibPhase = vibPhase;
    } else {
      for (let i = 0; i < n; i++) {
        if (pos >= wlen) {
          take = i;
          break;
        }
        let step = baseStep;
        if (useVib) {
          if (vibDelay > 0) vibDelay--;
          else {
            step = baseStep * (1 + depth * Math.sin(vibPhase));
            vibPhase += (2 * Math.PI * vibHz) / this.sr;
          }
        }
        const idx = Math.min(Math.floor(pos), wlen - 1);
        const sample = wave[idx]! * env[i]! * v.vol;
        out[i] = sample;
        const a = Math.abs(sample);
        if (a > peak) peak = a;
        pos += step;
      }
      v.pos = pos;
      v.vibDelayLeft = vibDelay;
      v.vibPhase = vibPhase;
      if (take === 0) {
        v.active = false;
        return;
      }
      if (pos >= wlen) v.active = false;
    }

    v.peak = peak;
    if (
      v.envFixed <= 1 &&
      (v.envStage >= 3 || (v.envStage >= 2 && inst.envLevels[2] === 0))
    ) {
      v.active = false;
    }
  }

  /** Render n stereo interleaved float samples into dest (length >= n*2). */
  renderBlock(n: number, dest: Float32Array): void {
    dest.fill(0);
    const grain = 128;
    const mono = new Float32Array(grain);
    let pos = 0;
    while (pos < n) {
      const g = Math.min(grain, n - pos);
      this.advanceTracks(g / this.beatSamples);
      for (const v of this.voices) {
        if (v.active && v.instrument) {
          this.renderVoice(v, g, mono);
          const side = CHANNEL_PAN[v.channel & 3]!;
          for (let i = 0; i < g; i++) {
            dest[(pos + i) * 2 + side]! += mono[i]!;
          }
        } else {
          v.peak *= 0.85;
        }
      }
      pos += g;
    }
    for (let i = 0; i < n * 2; i++) {
      let s = dest[i]! * this.master;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      dest[i] = s;
    }
  }

  get finished(): boolean {
    const tracksDone = this.tracks.every((t) => t.done || t.index >= t.events.length);
    const voicesIdle = this.voices.every((v) => !v.active);
    return tracksDone && voicesIdle;
  }

  kick(): void {
    this.advanceTracks(0);
  }

  /** Offline render to interleaved stereo float32, trimmed trailing silence. */
  renderAll(maxSeconds = 600): Float32Array {
    this.kick();
    const block = 2048;
    const maxSamples = Math.floor(maxSeconds * this.sr);
    const chunks: Float32Array[] = [];
    let total = 0;
    const buf = new Float32Array(block * 2);
    while (total < maxSamples) {
      this.renderBlock(block, buf);
      chunks.push(buf.slice());
      total += block;
      if (this.finished && this.voices.every((v) => !v.active)) break;
    }
    const audio = new Float32Array(total * 2);
    let off = 0;
    for (const c of chunks) {
      audio.set(c, off);
      off += c.length;
    }
    const thresh = 1e-4;
    let last = 0;
    for (let i = 0; i < total; i++) {
      if (Math.abs(audio[i * 2]!) > thresh || Math.abs(audio[i * 2 + 1]!) > thresh) last = i;
    }
    const keep = Math.min(total, last + Math.floor(this.sr / 4) + 1);
    return audio.subarray(0, keep * 2);
  }
}
