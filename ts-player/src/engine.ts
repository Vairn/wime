/**
 * @file engine.ts
 *
 * SmusEngine — real-time and offline renderer for the Smus (Sonix Music) score format.
 *
 * Architecture:
 * - **Sequencer**: up to four independent tracks advance in lock-step with the score's
 *   tempo. Each track reads a stream of {@link SEvent} records (notes, rests, instrument
 *   changes, dynamics, tempo) and schedules work on the voice pool.
 * - **Four Paula voices**: mirrors the Amiga's four hardware channels (AUD0–AUD3).
 *   Each voice plays one note at a time — either a Sonix synth patch (filter-bank
 *   wavetable) or a sampled instrument — with ADSR envelope, LFO modulation, vibrato,
 *   and stereo panning (AUD0/AUD3 left, AUD1/AUD2 right).
 *
 * Rendering is block-based: {@link renderBlock} advances the sequencer, mixes active
 * voices into interleaved stereo float samples, and applies master limiting.
 * {@link renderAll} loops renderBlock for offline export with trailing-silence trim.
 */

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

/**
 * Per-channel stereo pan indices for the four Paula voices.
 * AUD0 and AUD3 map to left (0); AUD1 and AUD2 map to right (1).
 */
export const CHANNEL_PAN = [0, 1, 1, 0] as const;

/**
 * Amiga note-period lookup table (12 semitones within one octave).
 * Used to derive sample playback rate from MIDI pitch for SS-format samples.
 */
const NOTE_PERIOD = [
  0x8000, 0x78d1, 0x7209, 0x6ba2, 0x6598, 0x5fe4, 0x5a82, 0x556e, 0x50a3, 0x4c1c, 0x47d6,
  0x43ce,
];

/**
 * Runtime state for a single Paula voice channel.
 * Holds playback position, envelope/LFO/vibrato state, and the active instrument.
 */
export interface Voice {
  /** Whether this voice is currently sounding a note. */
  active: boolean;
  /** Hardware channel index (0–3), used for stereo pan lookup. */
  channel: number;
  /** Instrument definition currently assigned to this voice. */
  instrument: Instrument | null;
  /** Sample or wavetable read position (fractional). */
  pos: number;
  /** Samples advanced per output sample (playback rate). */
  step: number;
  /** Combined track × instrument × score volume multiplier. */
  vol: number;
  /** Remaining samples before envelope release stage is forced. */
  samplesLeft: number;
  /** True once the note gate has expired and release is underway. */
  release: boolean;
  /** Last envelope level written (0–1), for UI metering. */
  envLevel: number;
  /** Fixed-point envelope level (0–255) used by sonixEnvStep. */
  envFixed: number;
  /** Current ADSR stage index (0=attack, 1=decay, 2=sustain, 3=release). */
  envStage: number;
  /** LFO phase accumulator (0–256). */
  lfoPhase: number;
  /** True when a one-shot LFO has reached its end and holds final value. */
  lfoFrozen: boolean;
  /** Last LFO modulation value (scaled mod-table output). */
  lfoMod: number;
  /** Vibrato LFO phase in radians. */
  vibPhase: number;
  /** Samples remaining before vibrato begins (post-delay). */
  vibDelayLeft: number;
  /** Active sample waveform slice (may differ from instrument.wave). */
  sampleWave: Float32Array | null;
  /** Loop region start index within sampleWave. */
  sampleLoopStart: number;
  /** Loop region end index within sampleWave (exclusive). */
  sampleLoopEnd: number;
  /** Fundamental frequency of the current note in Hz. */
  noteFreq: number;
  /** True after playback has entered the sustain loop region. */
  inHold: boolean;
  /** MIDI note number of the sounding pitch. */
  midi: number;
  /** Peak absolute sample value in the last rendered block (for metering). */
  peak: number;
}

/**
 * Create a silent, idle voice pre-bound to the given Paula channel.
 * @param channel - Hardware channel index (0–3).
 */
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

/**
 * Per-track sequencer cursor and playback context.
 * Each of the four tracks maintains its own event pointer, wait counter,
 * instrument register, and dynamic volume.
 */
interface TrackState {
  /** Copy of the track's event list from the score. */
  events: SEvent[];
  /** Index of the next event to consume. */
  index: number;
  /** Beats remaining before the next event fires (note duration / rest). */
  wait: number;
  /** Current instrument register number (set by SID_INSTRUMENT events). */
  instrumentReg: number;
  /** Track dynamic level (0–1), set by SID_DYNAMIC events. */
  volume: number;
  /** Pending chord tones accumulated before the chord-closing note. */
  chordNotes: Array<[number, number]>;
  /** True when the track has no more events to process. */
  done: boolean;
}

/**
 * SmusEngine — sequencer plus four-voice Paula mixer.
 *
 * Drives score playback by advancing track cursors in beat time, spawning notes
 * on free voices, and rendering mixed stereo audio blocks.
 */
export class SmusEngine {
  /** Parsed score containing tempo, volume, and per-track event lists. */
  score: SmusScore;
  /** Instrument bank keyed by register number. */
  instruments: Map<number, Instrument>;
  /** Output sample rate in Hz. */
  sr: number;
  /** Master output gain applied after voice mixing. */
  master: number;
  /** Current tempo in beats per minute. */
  bpm: number;
  /** Number of output samples per beat at the current tempo. */
  beatSamples: number;
  /** Per-track sequencer state (one entry per score track). */
  tracks: TrackState[];
  /** Four Paula voice channels. */
  voices: Voice[];
  /** Global score volume scalar (0–1). */
  scoreVolume: number;
  /** Elapsed beats for UI playhead */
  beatPos = 0;

  /**
   * Construct an engine ready to render the given score.
   * @param score - Parsed Smus score with tracks and global tempo/volume.
   * @param instruments - Map of instrument register → {@link Instrument} definition.
   * @param sampleRate - Output sample rate (default 44100 Hz).
   * @param masterVolume - Post-mix master gain (default 0.28).
   */
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
    // Score tempo is stored as BPM×128; divide to get real BPM.
    this.bpm = Math.max(score.tempo / 128, 1);
    this.beatSamples = (60 / this.bpm) * sampleRate;
    // Initialise one TrackState per score track.
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
    // Fast-forward past leading control events (instrument, volume, tempo).
    for (const tr of this.tracks) this.primeTrack(tr);
  }

  /**
   * Resolve an instrument register to its definition, falling back to a
   * placeholder synth if the register is not in the bank.
   * @param reg - Instrument register number from a SID_INSTRUMENT event.
   */
  private instForReg(reg: number): Instrument {
    return this.instruments.get(reg) ?? defaultInstrument(`reg${reg}`);
  }

  /**
   * Advance a track's cursor past any leading control events so playback
   * begins at the first note or rest.
   * @param tr - Track state to prime.
   */
  private primeTrack(tr: TrackState): void {
    while (tr.index < tr.events.length) {
      const ev = tr.events[tr.index]!;
      // Stop at the first musical event (note < 0x80) or explicit rest.
      if (ev.sid < 0x80 || ev.sid === SID_REST) break;
      this.handleControl(tr, ev);
      tr.index++;
    }
  }

  /**
   * Apply a non-note control event to track or global engine state.
   * Handles instrument selection, dynamic volume, and tempo changes.
   * @param tr - Track receiving the control event.
   * @param ev - The control {@link SEvent} to process.
   */
  private handleControl(tr: TrackState, ev: SEvent): void {
    if (ev.sid === SID_INSTRUMENT) tr.instrumentReg = ev.data;
    else if (ev.sid === SID_DYNAMIC) tr.volume = Math.max(ev.data, 1) / 127;
    else if (ev.sid === SID_TEMPO && ev.data > 0) {
      this.bpm = ev.data;
      this.beatSamples = (60 / this.bpm) * this.sr;
    }
  }

  /**
   * Start (or restart) a voice channel with a new note.
   *
   * Computes playback rate, sample slice, loop points, vibrato delay, and
   * note duration from the instrument kind and event flags.
   *
   * @param ch - Paula channel index (0–3).
   * @param midi - MIDI note number.
   * @param flags - Note duration / articulation flags from the event data byte.
   * @param tr - Owning track state (supplies instrument register and volume).
   */
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
      // Synth voices advance through a 128-entry wavetable at note frequency.
      step = (freq * 128) / this.sr;
    } else if (inst.kind === "sample" && inst.ssData) {
      // SS-format sample: select octave slice and optional loop region.
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
        // Define sustain loop within the octave-shifted slice.
        sampleLoopStart = Math.min(wlen - 1, repeat << octv);
        sampleLoopEnd = Math.min(wlen, oneshot << octv);
        if (sampleLoopEnd - sampleLoopStart < 2) {
          // Loop too short to be useful — disable looping.
          sampleLoopStart = 0;
          sampleLoopEnd = 0;
        } else {
          const ls = sampleLoopStart;
          const le = sampleLoopEnd;
          const loop = sampleWave.slice(ls, le);
          // Crossfade loop wrap point to reduce click at boundary.
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
      // Rate from Amiga period table, scaled to output sample rate.
      const noteInOct = midi % 12;
      const rate = inst.baseRate * (NOTE_PERIOD[0]! / NOTE_PERIOD[noteInOct]!);
      step = rate / this.sr;
      // One-shot (no loop): cap note length to sample duration + 50 ms tail.
      if (sampleLoopEnd === 0 && wlen > 0) {
        nSamples = Math.min(nSamples, Math.floor(wlen / Math.max(step, 1e-6)) + Math.floor(this.sr / 20));
      }
    } else {
      // Generic wave instrument: rate relative to base pitch.
      const baseFreq = 440 * 2 ** ((inst.baseMidi - 69) / 12);
      step = (inst.baseRate / this.sr) * (freq / Math.max(baseFreq, 1e-6));
      sampleWave = inst.wave;
      sampleLoopStart = inst.loopStart;
      sampleLoopEnd = inst.loopEnd;
    }

    // Convert Sonix vibrato delay units to output samples.
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

  /**
   * Consume the next event on a track and act on it.
   *
   * Note events start voices; chord-flagged notes are buffered until the
   * closing tone arrives. Rests and control events update wait counters
   * or engine state, then may recurse for chained controls.
   *
   * @param tr - Track whose event cursor advances.
   * @param ch - Default Paula channel for the track's primary voice.
   */
  private consumeEvent(tr: TrackState, ch: number): void {
    if (tr.index >= tr.events.length) {
      tr.done = true;
      return;
    }
    const ev = tr.events[tr.index]!;
    tr.index++;

    if (ev.sid < 0x80) {
      // Musical note event: sid is MIDI pitch, data holds flags + chord bit.
      const chord = Boolean(ev.data & 0x80);
      const flags = ev.data & 0x3f;
      const midi = ev.sid;
      if (chord) {
        // Chord handling: accumulate this tone and read the next event immediately.
        tr.chordNotes.push([midi, flags]);
        this.consumeEvent(tr, ch);
        return;
      }
      // Chord close (or single note): play all buffered tones on free voices.
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

    // Control event (instrument, dynamic, tempo): apply and consume next.
    this.handleControl(tr, ev);
    this.consumeEvent(tr, ch);
  }

  /**
   * Advance all tracks by the given number of beats.
   * Fires events whenever a track's wait counter reaches zero.
   * @param beats - Elapsed beat fraction since the last advance.
   */
  private advanceTracks(beats: number): void {
    this.beatPos += beats;
    // Only the first four tracks map to Paula channels 0–3.
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

  /**
   * Advance the Sonix ADSR envelope (and optional LFO / filter bank) for
   * `n` consecutive output samples.
   *
   * Writes per-sample envelope gain to `envOut` and, when `bankOut` is
   * provided, per-sample filter-bank index for synth voices.
   *
   * @param v - Voice whose envelope/LFO state is updated in place.
   * @param n - Number of samples to generate.
   * @param envOut - Receives normalised envelope level (0–1) per sample.
   * @param bankOut - Optional; receives filter index (0–63) per sample for synth.
   */
  private sonixEnvStep(
    v: Voice,
    n: number,
    envOut: Float32Array,
    bankOut: Float32Array | null,
  ): void {
    const inst = v.instrument!;
    const sr = this.sr;
    const levels = inst.envLevels;

    // Convert Sonix rate bytes to per-sample envelope delta (0–255 scale).
    const rates = inst.envRates.map((r) => {
      const units = sonixRateUnits(r);
      const secs = Math.max(0.008, units / 2500);
      return 255 / (secs * sr);
    });

    // Local copies avoid repeated property access in the hot loop.
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
      // Envelope stages: force release when note gate expires.
      if (gateLeft <= 0 && stage < 3) {
        stage = 3;
        v.release = true;
      }

      let target = levels[Math.min(stage, 3)]!;
      if (stage >= 3) target = 0;
      let spd = rates[Math.min(stage, 3)]!;
      // Ensure release stage always has a finite fall rate.
      if (stage >= 3 && spd < 1e-6) spd = 255 / (0.05 * sr);
      const dist = Math.abs(env - target);
      if (dist <= spd) {
        env = target;
        // Advance to next stage once target level is reached.
        if (stage < 2) stage++;
      } else if (env < target) env += spd;
      else env -= spd;

      let mod = modHeld;
      if (modTable && lfoStep > 0 && !frozen) {
        mod = modTable[Math.floor(lfo) & 255]! * 128;
        lfo += lfoStep;
        if (inst.lfoOneshot && lfo >= 254) {
          // One-shot LFO: freeze at final table entry.
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
        // Filter bank interpolation index: base − envelope×fEnv + LFO×fMod.
        let filt = 255 - fBase - ((envI * fEnv) >> 8) + Math.floor((mod * fMod) / 256);
        filt = Math.max(0, Math.min(255, filt));
        bankOut[i] = filt >> 2;
      }

      if (gateLeft > 0) gateLeft--;
    }

    // Write back accumulated state for the next block.
    v.envFixed = env;
    v.envStage = stage;
    v.lfoPhase = lfo;
    v.lfoFrozen = frozen;
    v.lfoMod = modHeld;
    v.envLevel = n ? envOut[n - 1]! : v.envLevel;
    v.samplesLeft = gateLeft;
  }

  /**
   * Render `n` mono samples for a single active voice into `out`.
   *
   * Dispatches to synth (filter-bank wavetable) or sample playback paths.
   * Deactivates the voice when the envelope finishes or the sample ends.
   *
   * @param v - Voice to render (must be active with a valid instrument).
   * @param n - Number of output samples to produce.
   * @param out - Pre-allocated mono buffer (length ≥ n).
   */
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
        // Filter bank interpolation: linear blend between adjacent banks.
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
      // Sample loop/hold path: play attack, then wrap in sustain region.
      const ll = le - ls;
      for (let i = 0; i < n; i++) {
        let step = baseStep;
        if (useVib) {
          if (vibDelay > 0) vibDelay--;
          else {
            // Vibrato: sinusoidal playback-rate modulation after delay.
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
        // Enter sustain hold: wrap position inside loop region.
        v.pos = ls + ((pos - ls) % ll);
        v.inHold = true;
      } else {
        v.pos = pos;
      }
      v.vibDelayLeft = vibDelay;
      v.vibPhase = vibPhase;
    } else {
      // One-shot sample: linear interpolation until end of waveform.
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

  /**
   * Render n stereo interleaved float samples into dest (length >= n*2).
   *
   * Advances the sequencer in 128-sample grains, mixes all active voices
   * with Paula stereo pan, and applies master gain with hard clipping.
   *
   * @param n - Number of stereo frames to render.
   * @param dest - Interleaved stereo output buffer (L,R,L,R,…).
   */
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
          // Stereo pan L-R-R-L: route mono voice to left or right channel.
          const side = CHANNEL_PAN[v.channel & 3]!;
          for (let i = 0; i < g; i++) {
            dest[(pos + i) * 2 + side]! += mono[i]!;
          }
        } else {
          // Decay peak meter when voice is silent.
          v.peak *= 0.85;
        }
      }
      pos += g;
    }
    // Master gain with hard clip to ±1.
    for (let i = 0; i < n * 2; i++) {
      let s = dest[i]! * this.master;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      dest[i] = s;
    }
  }

  /**
   * True when every track is exhausted and all voices are idle.
   * Used by {@link renderAll} to detect end-of-song.
   */
  get finished(): boolean {
    const tracksDone = this.tracks.every((t) => t.done || t.index >= t.events.length);
    const voicesIdle = this.voices.every((v) => !v.active);
    return tracksDone && voicesIdle;
  }

  /**
   * Prime playback by advancing zero beats (fires any wait=0 events).
   * Called once before offline rendering begins.
   */
  kick(): void {
    this.advanceTracks(0);
  }

  /**
   * Offline render to interleaved stereo float32, trimmed trailing silence.
   *
   * Renders the full score in 2048-sample blocks until finished or
   * `maxSeconds` is reached, then trims trailing silence while keeping
   * a 250 ms tail after the last audible sample.
   *
   * @param maxSeconds - Safety cap on render duration (default 600 s).
   * @returns Trimmed interleaved stereo Float32Array.
   */
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
    // renderAll trim: find last sample above silence threshold.
    const thresh = 1e-4;
    let last = 0;
    for (let i = 0; i < total; i++) {
      if (Math.abs(audio[i * 2]!) > thresh || Math.abs(audio[i * 2 + 1]!) > thresh) last = i;
    }
    // Keep 250 ms of tail after the last audible frame.
    const keep = Math.min(total, last + Math.floor(this.sr / 4) + 1);
    return audio.subarray(0, keep * 2);
  }
}
