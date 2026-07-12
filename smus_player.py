#!/usr/bin/env python3
"""
Aegis Sonix 2 / IFF-SMUS player and WAV renderer.

Converts .smus scores (with accompanying .instr / .ss / 8SVX instruments
in the same directory) to WAV, and can play them in realtime.

This module is a from-scratch reimplementation of the playback engine used
by the Aegis "Sonix 2" / Instant Music family of Amiga music tools. It:

  * Parses the IFF "SMUS" score format (SHDR/NAME/INS1/TRAK chunks) into a
    `SmusScore` (see `parse_smus`).
  * Loads the instrument bank referenced by the score: Synthesis wavetable
    instruments (`load_synth_instr`), multi-octave SampledSound instruments
    (`load_ss` / `load_sampled_instr`), and plain 8SVX samples (`load_8svx`).
  * Sequences the four Amiga "Paula" style tracks and renders them sample by
    sample with a `SmusEngine`, reproducing the original driver's envelope,
    LFO/vibrato, filter-bank, and sample-loop/hold behaviour as closely as
    possible.
  * Either streams the result live via `sounddevice` (`play_realtime`) or
    renders the whole song to a WAV file (`render_all` + `write_wav`).

Usage:
  python smus_player.py song.smus                  # play
  python smus_player.py song.smus -o song.wav      # render WAV
  python smus_player.py song.smus -o song.wav -p   # render then play
  python smus_player.py disk2/B --all -o wavs/     # batch convert folder
"""

from __future__ import annotations

import argparse
import struct
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# IFF helpers
# ---------------------------------------------------------------------------


def iter_iff_chunks(data: bytes, start: int = 12, end: Optional[int] = None):
    """Yield (chunk_id, chunk_data, absolute_offset) from an IFF FORM body.

    IFF files are a flat sequence of chunks: a 4-byte ASCII chunk ID,
    followed by a big-endian 32-bit size, followed by that many bytes of
    chunk data. `start` defaults to 12 to skip the outer "FORM"+size+type
    header (e.g. "FORM"+size+"SMUS") and begin at the first real chunk.
    """
    if end is None:
        end = len(data)
    pos = start
    while pos + 8 <= end:
        cid = data[pos : pos + 4]
        size = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        yield cid, body, pos
        # IFF chunks are word-aligned: odd-sized chunks get one pad byte
        # after their data that is NOT counted in `size`, so skip it too.
        pos += 8 + size + (size & 1)


def read_cstring(data: bytes) -> str:
    """Decode a NUL-terminated (or NUL-padded) byte string to text.

    Splits at the first NUL byte, decodes as Latin-1 (so any byte value is
    representable), and strips surrounding whitespace.
    """
    return data.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()


# ---------------------------------------------------------------------------
# SMUS score
# ---------------------------------------------------------------------------

SID_REST = 0x80
SID_INSTRUMENT = 0x81
SID_TIME_SIG = 0x82
SID_KEY_SIG = 0x83
SID_DYNAMIC = 0x84
SID_MIDI_CHNL = 0x85
SID_MIDI_PRESET = 0x86
SID_CLEF = 0x87
SID_TEMPO = 0x88  # Instant Music / Sonix inline tempo (BPM)


@dataclass
class SEvent:
    """A single raw 2-byte SMUS track event.

    Each TRAK chunk is a flat stream of (sid, data) byte pairs. If `sid` is
    < 0x80 it is a MIDI-style note number and `data` carries duration/chord/
    tie flags (see `note_duration_beats` and `SmusEngine._consume_event`).
    Otherwise `sid` is one of the SID_* control-event IDs above and `data`
    is that event's single-byte payload (rest length, instrument register,
    dynamic level, tempo, etc.).
    """

    sid: int
    data: int


@dataclass
class SmusScore:
    """A fully parsed SMUS score: header info plus one event list per track."""

    tempo: int  # 128ths of a quarter-note per minute
    volume: int
    name: str
    instruments: Dict[int, str]  # register -> name
    tracks: List[List[SEvent]]
    path: Path


def parse_smus(path: Path) -> SmusScore:
    """Parse a FORM SMUS file into a `SmusScore`.

    Walks the IFF chunks looking for:
      SHDR - score header: tempo, master volume, declared track count.
      NAME - optional song title (falls back to the file's stem).
      INS1 - instrument bank slot: register number + instrument name.
      TRAK - one track's worth of (sid, data) event byte pairs.
    """
    data = path.read_bytes()
    if data[:4] != b"FORM" or data[8:12] != b"SMUS":
        raise ValueError(f"Not a FORM SMUS file: {path}")

    tempo, volume, ntracks = 128 * 120, 127, 0
    name = path.stem
    instruments: Dict[int, str] = {}
    tracks: List[List[SEvent]] = []

    for cid, body, _ in iter_iff_chunks(data):
        if cid == b"SHDR" and len(body) >= 4:
            # Score header: 16-bit tempo (128ths of a QPM), byte volume,
            # byte declared track count.
            tempo, volume, ntracks = struct.unpack(">HBB", body[:4])
        elif cid == b"NAME":
            name = read_cstring(body) or name
        elif cid == b"INS1" and len(body) >= 4:
            # Byte 0 = instrument register (channel/slot index used by
            # SID_INSTRUMENT events); name string starts at byte 4.
            reg = body[0]
            instruments[reg] = read_cstring(body[4:])
        elif cid == b"TRAK":
            # Track body is a flat list of (sid, data) byte pairs.
            evs: List[SEvent] = []
            for i in range(0, len(body) - 1, 2):
                evs.append(SEvent(body[i], body[i + 1]))
            tracks.append(evs)

    if not tracks and ntracks:
        raise ValueError(f"SHDR says {ntracks} tracks but none found")
    return SmusScore(tempo, volume, name, instruments, tracks, path)


def note_duration_beats(flags: int) -> float:
    """Duration of an SNote data byte in quarter-note beats.

    The low 6 bits of a note/rest event's data byte pack the duration:
      bits 0-2 (0x07): base division (0=whole, 1=half, 2=quarter, ...).
      bit 3    (0x08): dotted (adds half the base duration).
      bits 4-5 (0x30): n-tuplet selector (0=none, 1=triplet, 2=quintuplet,
                       3=septuplet).
    Bits 6-7 (chord/tie) are stripped by the caller before this is invoked.
    """
    division = flags & 0x07
    dotted = bool(flags & 0x08)
    ntuplet = (flags >> 4) & 0x03
    beats = 4.0 / (1 << division)  # whole=4 beats, half=2, quarter=1, ...
    if dotted:
        beats *= 1.5
    if ntuplet == 1:  # triplet
        beats *= 2.0 / 3.0
    elif ntuplet == 2:  # quintuplet
        beats *= 4.0 / 5.0
    elif ntuplet == 3:  # septuplet
        beats *= 4.0 / 7.0
    return beats


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


# Sonix OneFilter coefficient table (from Wanted Team / Aegis driver)
_FILTER_COEFFS = [
    0x8000, 0x7683, 0x6DBA, 0x6597, 0x5E10, 0x5717, 0x50A2, 0x4AA8,
    0x451F, 0x4000, 0x3B41, 0x36DD, 0x32CB, 0x2F08, 0x2B8B, 0x2851,
    0x2554, 0x228F, 0x2000, 0x1DA0, 0x1B6E, 0x1965, 0x1784, 0x15C5,
    0x1428, 0x12AA, 0x1147, 0x1000, 0x0ED0, 0x0DB7, 0x0CB2, 0x0BC2,
    0x0AE2, 0x0A14, 0x0955, 0x08A3, 0x0800, 0x0768, 0x06DB, 0x0659,
    0x05E1, 0x0571, 0x050A, 0x04AA, 0x0451, 0x0400, 0x03B4, 0x036D,
    0x032C, 0x02F0, 0x02B8, 0x0285, 0x0255, 0x0228, 0x0200, 0x01DA,
    0x01B6, 0x0196, 0x0178, 0x015C, 0x0142, 0x012A, 0x0114, 0x0100,
]


def _to_i16(x: int) -> int:
    """Reinterpret the low 16 bits of `x` as a signed 16-bit integer (like a 68k word register)."""
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def _to_i32(x: int) -> int:
    """Reinterpret the low 32 bits of `x` as a signed 32-bit integer (like a 68k long register)."""
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def sonix_one_filter(wave128: bytes) -> np.ndarray:
    """
    Port of Sonix OneFilter: 128-byte signed wavetable -> 64 banks x 128
    samples (progressively low-passed). Returns int8 array length 8192.

    This is a direct, fixed-point-faithful translation of the original
    68k driver routine that pre-computes 64 increasingly low-pass
    filtered copies of a 128-sample wavetable, so that at playback time
    a "brightness" parameter can simply select (and interpolate between)
    banks instead of running a filter in real time. All arithmetic below
    intentionally uses 16/32-bit wraparound (`_to_i16`/`_to_i32`) to match
    the original assembly's register widths bit-for-bit.
    """
    # Sign-extend the raw wavetable bytes once up front.
    wave = [b if b < 128 else b - 256 for b in wave128]
    out = bytearray(64 * 128)
    d3 = 0
    # d4 (the filter's running "state") is seeded from the wavetable's
    # last sample, matching the driver's initial condition.
    d4 = _to_i16(wave[127] << 7)
    oi = 0
    for step in range(64):
        # d1/d2 are the two filter coefficients for this bank, derived
        # from the coefficient table; d1 is halved (>>1) per-bank as in
        # the original code, d2 is a fixed scaling of the complement.
        d1 = _FILTER_COEFFS[step]
        d2 = (0x8000 - d1) & 0xFFFF
        d2 = ((d2 * 0xE666) & 0xFFFFFFFF) >> 16
        d1 >>= 1
        for s in range(128):
            # One-pole low-pass update: d6 is the error between the
            # (scaled) input sample and the current filter state d4.
            d6 = _to_i16(_to_i16(wave[s] << 7) - d4)
            prod = _to_i32(_to_i16(d1) * d6)
            prod = _to_i32(prod << 2)
            # d3 accumulates the filtered "velocity", d4 integrates it
            # into the filtered "position" (the actual output sample).
            d3 = _to_i16(d3 + (prod >> 16))
            d4 = _to_i16(d4 + d3)
            d4u = d4 & 0xFFFF
            # Rotate the 16-bit result right by 7 bits to fold the
            # fixed-point fraction back down into an 8-bit sample.
            ror = ((d4u >> 7) | ((d4u & 0x7F) << 9)) & 0xFFFF
            out[oi] = ror & 0xFF
            oi += 1
            # Second-order damping term applied to d3 each sample, using
            # the bank's d2 coefficient, to gradually decay the velocity.
            prod3 = _to_i32(_to_i16(d3) * _to_i16(d2))
            d3 = _to_i16(_to_i32(prod3 << 1) >> 16)
    return np.frombuffer(bytes(out), dtype=np.int8)


@dataclass
class Instrument:
    """A loaded, ready-to-play instrument (synth wavetable, multi-octave
    sample, or plain 8SVX one-shot/loop), plus every driver parameter the
    engine needs to reproduce its envelope, filter, LFO and vibrato
    behaviour. Which fields are meaningful depends on `kind`:

      "synth"  - uses `filter_banks`, `mod_table`, `f_base/f_env/f_mod`,
                 `lfo_*`, `vol_env/vol_mod/pitch_mod`.
      "sample" - uses `ss_*` fields for multi-octave slicing plus
                 `vib_*` for vibrato.
      "8svx"   - only `wave`/`loop_start`/`loop_end`/`base_midi`/`base_rate`.
    """

    name: str
    kind: str  # "synth" | "sample" | "8svx"
    wave: np.ndarray  # float32 mono -1..1 (default / fallback playback table)
    loop_start: int = 0
    loop_end: int = 0  # exclusive; 0 = oneshot (no loop)
    base_midi: int = 60
    base_rate: float = 8363.0
    volume: float = 1.0
    # Synth-only
    filter_banks: Optional[np.ndarray] = None  # shape (64, 128)
    mod_table: Optional[np.ndarray] = None  # int8 length 256, LFO/mod
    # ADSR levels 0..255 and rates (raw driver words)
    env_levels: Tuple[int, int, int, int] = (255, 255, 200, 0)
    env_rates: Tuple[int, int, int, int] = (128, 128, 128, 64)
    # Filter: bank ~= ((255-f_base) - env*f_env/256 + mod*f_mod/256) >> 2
    f_base: int = 128   # body+0x1B6
    f_env: int = 0      # body+0x1B8
    f_mod: int = 0      # body+0x1BA
    lfo_rate: int = 0   # body+0x1C0 (delay before LFO)
    lfo_inc: int = 0    # body+0x1BC — running LFO increment
    lfo_enable: bool = False  # body+0x1BE != 0
    # 0x1BE: 0=off, >0=run once then freeze (hold), <0=loop forever (Echo3)
    lfo_oneshot: bool = True
    vol_raw: int = 255  # body+0x1AC
    # SampledSound multi-octave (.ss)
    ss_oneshot: int = 0
    ss_repeat: int = 0
    ss_lo: int = 0
    ss_hi: int = 0
    ss_data: Optional[np.ndarray] = None  # float32 full sample payload after header
    ss_data_off: int = 0x3E
    # SampledSound vibrato (body+$5A/$5C/$5E)
    vib_depth: int = 0
    vib_rate: int = 0
    vib_delay: int = 0
    # Synth: $1AE env→vol, $1B0 LFO→vol, $1B4 LFO→pitch
    vol_env: bool = True
    vol_mod: int = 0
    pitch_mod: int = 0
    # Kept for simple path / default instrument
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.15
    filter_start: float = 0.35
    filter_end: float = 0.7


def _i8_to_f32(raw) -> np.ndarray:
    """Convert raw signed 8-bit PCM bytes to float32 samples in -1..1."""
    return np.asarray(np.frombuffer(bytes(raw), dtype=np.int8), dtype=np.float32) / 128.0


def load_8svx(data: bytes, name: str) -> Instrument:
    """Load a plain Amiga 8SVX one-shot/loop sample into an `Instrument`.

    Reads VHDR for the one-shot sample count, repeat (loop) length, and
    playback rate, then BODY for the raw signed-8-bit waveform.
    """
    if data[:4] != b"FORM" or data[8:12] != b"8SVX":
        raise ValueError("Not 8SVX")
    oneshot = repeat = 0
    rate = 8363
    body = b""
    for cid, chunk, _ in iter_iff_chunks(data):
        if cid == b"VHDR" and len(chunk) >= 14:
            # oneshotHiSamples, repeatHiSamples, samplesPerHiCycle (unused),
            # then a 16-bit playback rate.
            oneshot, repeat, _sphc = struct.unpack(">III", chunk[:12])
            rate = struct.unpack(">H", chunk[12:14])[0] or 8363
        elif cid == b"BODY":
            body = chunk
    wave = _i8_to_f32(body)
    loop_start = oneshot
    loop_end = oneshot + repeat if repeat else 0
    return Instrument(name, "8svx", wave, loop_start, loop_end, 60, float(rate))


# Amiga Paula stereo: AUD0/AUD3 → left, AUD1/AUD2 → right (L-R-R-L)
_CHANNEL_PAN = (0, 1, 1, 0)  # 0=left, 1=right

# Note-within-octave period fractions from Sonix driver (C .. B)
_NOTE_PERIOD = [
    0x8000, 0x78D1, 0x7209, 0x6BA2, 0x6598, 0x5FE4,
    0x5A82, 0x556E, 0x50A3, 0x4C1C, 0x47D6, 0x43CE,
]


def _sample_octave_for_midi(midi: int, lo: int, hi: int) -> int:
    """Sonix: sample_oct = 10 - (midi // 12), clamped to .ss lo..hi."""
    octv = 10 - (midi // 12)
    return max(lo, min(hi, octv))


def load_ss(
    path: Path,
    name: str,
    volume: float = 1.0,
    env_levels: Tuple[int, int, int, int] = (255, 255, 255, 0),
    env_rates: Tuple[int, int, int, int] = (255, 255, 255, 128),
    vib_depth: int = 0,
    vib_rate: int = 0,
    vib_delay: int = 0,
) -> Instrument:
    """Sonix SampledSound .ss file (multi-octave, data at 0x3E).

    A .ss file stores ONE waveform repeated at doubling lengths for each
    octave it supports (so higher octaves get progressively more samples
    per cycle at the same pitch-independent sample rate). The header gives
    the base one-shot/repeat lengths (at the lowest octave) and the
    lo/hi octave range; `_sample_octave_for_midi` + the offset formula
    below (mirrored again in `SmusEngine._start_voice`) locate the right
    octave's slice within the concatenated payload.

    ADSR/vibrato parameters are passed in from the owning SampledSound
    .instr (see `load_sampled_instr`) since the .ss file itself has none.
    """
    data = path.read_bytes()
    if len(data) < 64:
        raise ValueError(f"Truncated .ss: {path}")
    oneshot, repeat = struct.unpack(">HH", data[:4])
    lo_oct, hi_oct = data[4], data[5]
    if hi_oct < lo_oct:
        hi_oct = lo_oct
    data_off = 0x3E
    payload = _i8_to_f32(data[data_off:])
    # Default wave: octave for middle C
    mid = _sample_octave_for_midi(60, lo_oct, hi_oct)
    off = oneshot * ((1 << mid) - (1 << lo_oct))
    ln = oneshot << mid
    wave = payload[off : off + ln]
    if len(wave) == 0:
        wave = payload[: max(1, min(len(payload), oneshot << lo_oct))]

    sustain = (env_levels[2] / 255.0) if env_levels[2] else 0.0

    return Instrument(
        name=name,
        kind="sample",
        wave=wave,
        loop_start=0,
        loop_end=0,
        base_midi=60,
        base_rate=8363.0,
        volume=volume,
        env_levels=env_levels,
        env_rates=env_rates,
        ss_oneshot=oneshot,
        ss_repeat=repeat,
        ss_lo=lo_oct,
        ss_hi=hi_oct,
        ss_data=payload,
        ss_data_off=data_off,
        vib_depth=vib_depth,
        vib_rate=vib_rate,
        vib_delay=vib_delay,
        sustain=sustain,
    )


def _sonix_rate_units(r: int) -> float:
    """Convert packed Sonix rate word to relative time units.

    The driver packs an envelope/LFO rate as a 3-bit exponent (bits 5-7,
    XORed with 7 so a larger stored exponent means a *slower* rate) and a
    5-bit mantissa offset by 0x21 (bits 0-4), i.e. a small floating-point
    encoding: units = mantissa << exponent. A raw value of 0 is treated as
    "very slow" (an arbitrary large unit count).
    """
    r &= 0xFFFF
    if r == 0:
        return 4000.0
    exp = 7 ^ ((r >> 5) & 7)
    mant = (r & 0x1F) + 0x21
    return float(mant << exp)


def load_synth_instr(data: bytes, name: str) -> Instrument:
    """
    Sonix Synthesis .instr (502 bytes).

    Layout (from Aegis / Wanted Team driver):
      [0:32]    type string ("Synthesis" or zeros)
      [32:68]   name / meta
      [68:196]  128-byte signed oscillator wavetable -> OneFilter source
      [196:452] 256-byte LFO / modulation table (body+0xA4)
      body+0x1AC..  volume, filter, ADSR
    """
    if len(data) < 68 + 128:
        raise ValueError("Truncated Synthesis instrument")

    wave_raw = bytes(data[68:196])
    banks_i8 = sonix_one_filter(wave_raw)
    banks = banks_i8.astype(np.float32).reshape(64, 128) / 128.0

    # All the following byte offsets in the docstring/driver are relative
    # to `body` (file offset 32), not the raw file.
    body = data[32:]

    def u16(off: int) -> int:
        """Read a big-endian 16-bit word from `body` at `off` (0 if out of range)."""
        if off + 2 > len(body):
            return 0
        return struct.unpack(">H", body[off : off + 2])[0]

    # Modulation table at body+0xA4 = file+196, 256 bytes
    mod_off = 0xA4
    mod_table = np.frombuffer(body[mod_off : mod_off + 256], dtype=np.int8).astype(np.float32) / 128.0

    vol_raw = u16(0x1AC) & 0xFF
    # Map Amiga 0..255 toward a balanced float gain (quiet patches stay present)
    vol = (0.35 + 0.65 * (max(vol_raw, 1) / 255.0))
    vol_env = u16(0x1AE) != 0
    vol_mod = u16(0x1B0) & 0xFF
    pitch_mod = u16(0x1B4) & 0xFF

    # 4-stage ADSR-style envelope: level 0..255 and rate word per stage,
    # stored as consecutive 16-bit words starting at these offsets.
    levels = tuple(u16(0x1C6 + i * 2) & 0xFF for i in range(4))
    rates = tuple(u16(0x1CE + i * 2) for i in range(4))

    # Approximate seconds for UI/fallback path
    def units_to_sec(u: float) -> float:
        """Clamp a `_sonix_rate_units` value into a plausible seconds range."""
        return max(0.005, min(1.5, u / 10000.0))

    attack = units_to_sec(_sonix_rate_units(rates[0]))
    decay = units_to_sec(_sonix_rate_units(rates[1]))
    release = units_to_sec(_sonix_rate_units(rates[3]))
    sustain = (levels[2] / 255.0) if levels[2] else 0.15

    # Filter parameters: f_base sets the resting cutoff, f_env/f_mod scale
    # how much the envelope and LFO/mod-wheel push the cutoff around.
    f_base = u16(0x1B6) & 0xFF
    f_env = u16(0x1B8) & 0xFF
    f_mod = u16(0x1BA) & 0xFF
    lfo_inc = u16(0x1BC) & 0xFF
    lfo_rate = u16(0x1C0) & 0xFF
    lfo_word = u16(0x1BE)
    lfo_signed = lfo_word - 0x10000 if lfo_word >= 0x8000 else lfo_word
    lfo_enable = lfo_word != 0
    # Driver: 1BE > 0 → ramp once then freeze (sustain holds steady);
    #         1BE < 0 → loop forever (Echo3).
    lfo_oneshot = lfo_signed >= 0

    # Precompute nominal bank at full env for default wave display
    bank0 = max(0, min(63, ((255 - f_base) - ((255 * f_env) >> 8)) >> 2))
    mid = banks[bank0].copy()

    # filter_start/end for any legacy use
    b_full = ((255 - f_base) - ((255 * f_env) >> 8)) >> 2
    b_zero = (255 - f_base) >> 2
    filter_start = max(0, min(63, b_full)) / 63.0
    filter_end = max(0, min(63, b_zero)) / 63.0

    return Instrument(
        name=name,
        kind="synth",
        wave=mid,
        loop_start=0,
        loop_end=128,
        base_midi=60,
        base_rate=16574.27,
        volume=vol,
        filter_banks=banks,
        mod_table=mod_table,
        env_levels=levels,  # type: ignore[arg-type]
        env_rates=rates,  # type: ignore[arg-type]
        f_base=f_base,
        f_env=f_env,
        f_mod=f_mod,
        lfo_rate=lfo_rate,
        lfo_inc=lfo_inc,
        lfo_enable=lfo_enable,
        lfo_oneshot=lfo_oneshot,
        vol_raw=vol_raw,
        vol_env=vol_env,
        vol_mod=vol_mod,
        pitch_mod=pitch_mod,
        attack=attack,
        decay=decay,
        sustain=sustain,
        release=release,
        filter_start=filter_start,
        filter_end=filter_end,
    )


def load_sampled_instr(instr_path: Path, data: bytes) -> Instrument:
    """SampledSound .instr (128 bytes) — .ss name at 68, ADSR/vib at body+$4A..

    A SampledSound .instr is just a small wrapper that names which .ss
    waveform file to load and supplies its envelope/vibrato parameters;
    the actual multi-octave sample data lives in the referenced .ss file
    (loaded via `load_ss`).
    """
    ss_name = read_cstring(data[68 : 68 + 24])
    if not ss_name:
        raise ValueError(f"No .ss name in {instr_path}")
    folder = instr_path.parent
    body = data[32:] if len(data) >= 32 else data

    def u16(off: int) -> int:
        """Read a big-endian 16-bit word from `body` at `off` (0 if out of range)."""
        if off + 2 > len(body):
            return 0
        return struct.unpack(">H", body[off : off + 2])[0]

    # body+$48 volume, +$4A levels, +$52 rates, +$5A vib (matches SampledSound play)
    vol_word = u16(0x48) if len(body) > 0x4A else 0xC0
    volume = max(vol_word, 1) / 255.0
    levels = tuple(u16(0x4A + i * 2) & 0xFF for i in range(4))
    rates = tuple(u16(0x52 + i * 2) for i in range(4))
    vib_depth = u16(0x5A) & 0xFF
    vib_rate = u16(0x5C) & 0xFF
    vib_delay = u16(0x5E) & 0xFF

    # Look for the referenced .ss file case-insensitively (try exact case,
    # upper-case extension, then scan the folder for a case-insensitive
    # stem match before giving up).
    candidates = [folder / f"{ss_name}.ss", folder / f"{ss_name}.SS"]
    lower = ss_name.lower()
    for p in folder.glob("*.ss"):
        if p.stem.lower() == lower:
            candidates.insert(0, p)
            break
    for p in candidates:
        if p.exists():
            return load_ss(
                p,
                instr_path.stem,
                volume=volume,
                env_levels=levels,  # type: ignore[arg-type]
                env_rates=rates,  # type: ignore[arg-type]
                vib_depth=vib_depth,
                vib_rate=vib_rate,
                vib_delay=vib_delay,
            )
    raise FileNotFoundError(f"Missing sample '{ss_name}.ss' for {instr_path.name}")


def load_instrument(folder: Path, name: str) -> Instrument:
    """Locate and load instrument by SMUS INS1 name.

    Finds the file in `folder` matching `name` (preferring a `.instr`
    extension, case-insensitively) and dispatches to the right loader
    based on its FORM type / size / magic bytes: 8SVX, SampledSound
    (.instr wrapper, 128 bytes), or Synthesis (502 bytes).
    """
    # Case-insensitive match for .instr
    instr_path = None
    target = name.lower()
    for p in folder.glob("*.instr"):
        if p.stem.lower() == target:
            instr_path = p
            break
    if instr_path is None:
        # Bare name file without extension (rare)
        for p in folder.iterdir():
            if p.is_file() and p.stem.lower() == target:
                instr_path = p
                break
    if instr_path is None:
        raise FileNotFoundError(f"Instrument not found: {name}")

    data = instr_path.read_bytes()
    if data[:4] == b"FORM" and data[8:12] == b"8SVX":
        return load_8svx(data, name)
    if data[:4] == b"FORM" and data[8:12] == b"AIFF":
        # Minimal AIFF: treat BODY-like ssnd if present — skip for now
        raise ValueError(f"AIFF instrument not supported: {name}")
    if len(data) == 128 and data[:12] == b"SampledSound":
        return load_sampled_instr(instr_path, data)
    if len(data) == 502 and (
        data[:9] == b"Synthesis" or data[:4] == b"\x00\x00\x00\x00"
    ):
        return load_synth_instr(data, name)
    # Heuristic: 502-byte synth without header
    if len(data) == 502:
        return load_synth_instr(data, name)
    if len(data) == 128:
        return load_sampled_instr(instr_path, data)
    raise ValueError(f"Unknown instrument format: {instr_path} ({len(data)} bytes)")


def default_instrument(name: str = "default") -> Instrument:
    """Build a simple fallback synth instrument (sine + weak 2nd harmonic).

    Used when a score references an instrument that fails to load or is
    entirely missing, so playback can continue with an audible placeholder.
    """
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False, dtype=np.float32)
    wave = (0.4 * np.sin(t) + 0.2 * np.sin(2 * t)).astype(np.float32)
    return Instrument(name, "synth", wave, 0, len(wave), 60, 16574.27)


# ---------------------------------------------------------------------------
# Voice / sequencer
# ---------------------------------------------------------------------------


@dataclass
class Voice:
    """Runtime state for one of the 4 emulated Paula hardware channels.

    Holds everything needed to keep rendering a currently-playing note:
    playback position/step, current envelope/LFO/vibrato phase, gate time
    remaining, and (for sample-based instruments) the specific octave
    slice and loop/hold region selected for this note.
    """

    active: bool = False
    channel: int = 0  # Paula channel 0..3 → pan via _CHANNEL_PAN
    instrument: Optional[Instrument] = None
    pos: float = 0.0
    step: float = 0.0
    vol: float = 0.0
    samples_left: int = 0  # gate time remaining; 0 => release
    release: bool = False
    env_level: float = 0.0
    env_phase: str = "attack"
    note_samples: int = 0
    note_total: int = 0  # original gate length
    # Synth envelope as 16.16 fixed (matches driver 12(A2) high word = 0..255)
    env_fixed: float = 0.0  # 0..255
    env_stage: int = 0  # 0=attack,1=decay,2=sustain,3=release
    lfo_phase: float = 0.0
    lfo_frozen: bool = False
    lfo_mod: float = 0.0  # last LFO mod value (-128..127)
    vib_phase: float = 0.0
    vib_delay_left: int = 0
    # Per-note sample slice (SampledSound multi-octave)
    sample_wave: Optional[np.ndarray] = None
    sample_loop_start: int = 0
    sample_loop_end: int = 0
    # After oneshot: steady hold (no sample wrap) for sustaining patches
    note_freq: float = 440.0
    hold_phase: float = 0.0
    hold_amp: float = 0.0
    can_hold: bool = False
    in_hold: bool = False


@dataclass
class TrackState:
    """Sequencer cursor + control state for one SMUS track.

    Tracks its position in the raw event list, beats remaining before the
    next event should fire, the currently-selected instrument register and
    dynamic-level volume, and any pending chord notes collected while
    walking chord-flagged note events.
    """

    events: List[SEvent]
    index: int = 0
    wait: float = 0.0  # beats remaining
    instrument_reg: int = 0
    volume: float = 1.0
    chord_notes: List[Tuple[int, int]] = field(default_factory=list)  # (midi, flags)
    done: bool = False


class SmusEngine:
    """Sequences a `SmusScore`'s 4 tracks and renders them to stereo audio.

    Owns the beat clock, one `TrackState` per score track, and a fixed
    pool of 4 `Voice` slots (mirroring the Amiga's 4 Paula hardware
    channels). Callers either pull fixed-size blocks via `render_block`
    for realtime playback, or call `render_all` to render an entire song
    to a single numpy array for WAV export.
    """

    def __init__(
        self,
        score: SmusScore,
        instruments: Dict[int, Instrument],
        sample_rate: int = 44100,
        master_volume: float = 0.35,
    ):
        """Set up tempo/beat timing, per-track cursors and voice pool, then
        prime each track past its leading control events."""
        self.score = score
        self.instruments = instruments
        self.sr = sample_rate
        self.master = master_volume
        # tempo: SHDR is 128ths of a quarter note per minute
        self.bpm = max(score.tempo / 128.0, 1.0)
        self.beat_samples = (60.0 / self.bpm) * sample_rate
        self.tracks = [TrackState(list(t)) for t in score.tracks]
        self.voices = [Voice(channel=i) for i in range(4)]
        self.score_volume = score.volume / 127.0
        # Advance past initial meta events on each track
        for tr in self.tracks:
            self._prime_track(tr)

    def _inst_for_reg(self, reg: int) -> Instrument:
        """Look up the instrument bound to register `reg`, or a synthesized default if missing."""
        return self.instruments.get(reg) or default_instrument(f"reg{reg}")

    def _prime_track(self, tr: TrackState):
        """Consume leading non-note control events.

        Called once at engine construction so that each track's
        `instrument_reg`/`volume` reflect any SID_INSTRUMENT/SID_DYNAMIC
        etc. events that precede the first actual note or rest, before
        the main sequencing loop starts consuming musical events.
        """
        while tr.index < len(tr.events):
            ev = tr.events[tr.index]
            if ev.sid < 0x80:
                break
            if ev.sid == SID_REST:
                break
            self._handle_control(tr, ev)
            tr.index += 1

    def _handle_control(self, tr: TrackState, ev: SEvent):
        """Apply a single non-note control event's effect to `tr` (or global tempo)."""
        if ev.sid == SID_INSTRUMENT:
            tr.instrument_reg = ev.data
        elif ev.sid == SID_DYNAMIC:
            tr.volume = max(ev.data, 1) / 127.0
        elif ev.sid == SID_TEMPO and ev.data > 0:
            self.bpm = float(ev.data)
            self.beat_samples = (60.0 / self.bpm) * self.sr
        # time/key/clef/midi ignored for audio

    def _start_voice(
        self, ch: int, midi: int, flags: int, tr: TrackState, tied: bool = False
    ):
        """Trigger a new note on Paula channel `ch`.

        Computes the note's gate/duration in samples, its playback step
        (pitch) according to the bound instrument's kind (synth wavetable
        vs. multi-octave sample vs. generic wave), selects/prepares any
        per-note sample slice and loop/hold region, then resets the given
        `Voice` slot to start playing from sample 0.
        """
        inst = self._inst_for_reg(tr.instrument_reg)
        dur_beats = note_duration_beats(flags)
        note_samples = max(1, int(dur_beats * self.beat_samples))
        # Sonix articulation (driver MULU #$C000 / SWAP): gate ≈ 75% of
        # notated duration, then release for the remaining 25%. Without this
        # drum rolls smear into a continuous wash and sound "too slow".
        if tied:
            gate_samples = note_samples
        else:
            gate_samples = max(1, (note_samples * 0xC000) >> 16)
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))

        sample_wave = None
        sample_loop_start = 0
        sample_loop_end = 0
        can_hold = False
        hold_amp = 0.0

        if inst.kind == "synth":
            # Synth wavetable is fixed at 128 samples/cycle; step directly
            # scales with note frequency (no separate octave slices needed
            # since `_render_voice` re-samples via the filter banks).
            step = (freq * 128.0) / self.sr
        elif inst.kind == "sample" and inst.ss_data is not None:
            # Multi-octave SampledSound: pick slice by MIDI, fine-tune within octave
            octv = _sample_octave_for_midi(midi, inst.ss_lo, inst.ss_hi)
            oneshot, repeat, lo = inst.ss_oneshot, inst.ss_repeat, inst.ss_lo
            # Same offset/length formula as `load_ss`'s default-wave calc,
            # but for the octave slice matching this specific note.
            offset = oneshot * ((1 << octv) - (1 << lo))
            length = oneshot << octv
            sample_wave = inst.ss_data[offset : offset + length]
            if len(sample_wave) == 0:
                sample_wave = inst.wave
            wlen = len(sample_wave)
            # Sonix: play the whole oneshot once, then hold on the SHORT tail
            # [repeat << oct .. oneshot << oct). For Trumpet that tail is ~1
            # wavelength — a steady hold, not a re-trigger of the body.
            if 0 < repeat < oneshot and wlen > 0:
                sample_loop_start = min(wlen - 1, repeat << octv)
                sample_loop_end = min(wlen, oneshot << octv)
                if sample_loop_end - sample_loop_start < 2:
                    sample_loop_start, sample_loop_end = 0, 0
                else:
                    # Make the hold cycle seamless (removes join click → dop)
                    ls, le = sample_loop_start, sample_loop_end
                    loop = sample_wave[ls:le].copy()
                    fade = min(max(le - ls, 2) // 4, 32)
                    if fade >= 2:
                        for i in range(fade):
                            t = (i + 1) / fade
                            a = loop[i]
                            b = loop[le - ls - fade + i]
                            loop[i] = a * t + b * (1.0 - t)
                        sample_wave = sample_wave.copy()
                        sample_wave[ls:le] = loop
            else:
                sample_loop_start, sample_loop_end = 0, 0
            # Fine pitch adjustment within the octave slice, using the
            # driver's 12-entry period-fraction table (C..B relative to C).
            note_in_oct = midi % 12
            rate = inst.base_rate * (_NOTE_PERIOD[0] / _NOTE_PERIOD[note_in_oct])
            step = rate / self.sr
            if sample_loop_end == 0 and wlen > 0:
                # No hold region: cap the gate so a pure one-shot sample
                # doesn't keep "playing" (silently, past its end) for the
                # full notated duration.
                max_play = int(wlen / max(step, 1e-6)) + self.sr // 20
                gate_samples = min(gate_samples, max_play)
        else:
            # Generic path (8SVX or synth-with-no-filter-banks fallback):
            # pitch-shift the instrument's single base wave/rate by the
            # ratio between the requested note and the instrument's base note.
            base_freq = 440.0 * (2.0 ** ((inst.base_midi - 69) / 12.0))
            step = (inst.base_rate / self.sr) * (freq / max(base_freq, 1e-6))
            sample_wave = inst.wave
            sample_loop_start, sample_loop_end = inst.loop_start, inst.loop_end

        vol = tr.volume * inst.volume * self.score_volume
        vib_delay = 0
        if inst.vib_delay > 0:
            vib_delay = int((_sonix_rate_units(inst.vib_delay) / 8000.0) * self.sr)

        v = self.voices[ch]
        v.active = True
        v.channel = ch
        v.instrument = inst
        v.pos = 0.0
        v.step = step
        v.vol = vol
        v.samples_left = gate_samples
        v.release = False
        v.env_level = 0.0
        v.env_phase = "attack"
        v.note_samples = 0
        v.note_total = gate_samples
        v.env_fixed = 0.0
        v.env_stage = 0
        v.lfo_phase = 0.0
        v.lfo_frozen = False
        v.lfo_mod = 0.0
        v.vib_phase = 0.0
        v.vib_delay_left = vib_delay
        v.sample_wave = sample_wave
        v.sample_loop_start = sample_loop_start
        v.sample_loop_end = sample_loop_end
        v.note_freq = freq
        v.hold_phase = 0.0
        v.hold_amp = 0.0
        v.can_hold = False
        v.in_hold = False

    def _consume_event(self, tr: TrackState, ch: int):
        """Process the next event on track `tr`, using Paula channel `ch`.

        Notes carry two high bits in their data byte: bit 7 (0x80) marks a
        chord note (more notes follow that all start together), and bit 6
        (0x40) marks a tie (this note is a continuation of the previous
        one, so it should not be re-gated/re-articulated). Chord notes are
        buffered in `tr.chord_notes` until a non-chord note completes the
        chord; the whole chord's notes are then started together, with the
        primary note on `ch` and extra chord tones stealing any free voice
        slots. Control events (instrument/dynamic/tempo/etc.) are applied
        immediately and recursion continues to the next event without
        advancing the beat clock, so a control event never consumes time.
        """
        if tr.index >= len(tr.events):
            tr.done = True
            return
        ev = tr.events[tr.index]
        tr.index += 1

        if ev.sid < 0x80:
            # Note. Chord bit: accumulate until a non-chord note.
            chord = bool(ev.data & 0x80)
            tie = bool(ev.data & 0x40)
            flags = ev.data & 0x3F
            midi = ev.sid
            if chord:
                tr.chord_notes.append((midi, flags))
                # Keep reading; duration taken from last note of chord
                self._consume_event(tr, ch)
                return
            notes = tr.chord_notes + [(midi, flags)]
            tr.chord_notes.clear()
            # Play first note on this channel; extra chord tones steal free voices
            self._start_voice(ch, notes[0][0], notes[0][1], tr, tied=tie)
            for midi_n, flags_n in notes[1:]:
                free = next((i for i, v in enumerate(self.voices) if not v.active), None)
                if free is not None:
                    self._start_voice(free, midi_n, flags_n, tr, tied=tie)
            tr.wait = note_duration_beats(flags)
            return

        if ev.sid == SID_REST:
            tr.wait = note_duration_beats(ev.data & 0x3F)
            return

        self._handle_control(tr, ev)
        # Immediately continue to next event
        self._consume_event(tr, ch)

    def _advance_tracks(self, beats: float):
        """Advance every track's beat-wait counter by `beats` and fire any
        events whose wait time has elapsed (looping in case multiple
        zero-duration control events chain together)."""
        for ch, tr in enumerate(self.tracks[:4]):
            if tr.done:
                continue
            tr.wait -= beats
            while tr.wait <= 1e-9 and not tr.done:
                self._consume_event(tr, ch)
                if tr.wait <= 1e-9 and not tr.done and tr.index >= len(tr.events):
                    tr.done = True

    def _sonix_env_step(
        self, v: Voice, n: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Driver-style 4-stage envelope → amp 0..1, and optional filter banks.

        Stages ramp toward level[stage]; only advance while stage < 2 (sustain).
        Sustain may itself ramp (e.g. Piano level 0 = decay-to-silence while held).
        Gate end (samples_left==0) moves to stage 3 (release).
        """
        inst = v.instrument
        assert inst is not None
        sr = float(self.sr)
        want_bank = inst.kind == "synth" and inst.filter_banks is not None

        levels = [float(x) for x in inst.env_levels]

        def step_per_sample(rate_word: int) -> float:
            """Convert a packed envelope rate word into an amount-per-sample ramp speed (0..255 scale)."""
            units = _sonix_rate_units(rate_word)
            # Calibrated so Piano sustain→0 (~rate 48) rings ~1.2–1.5s, not organ-held
            secs = max(0.008, units / 2500.0)
            return 255.0 / (secs * sr)

        rates = [step_per_sample(r) for r in inst.env_rates]

        env_out = np.empty(n, dtype=np.float32)
        bank_out = np.empty(n, dtype=np.float32) if want_bank else None

        env = v.env_fixed
        stage = v.env_stage
        lfo = v.lfo_phase

        # Decide whether an LFO/mod-wheel sweep should run at all: it must
        # be enabled (or feeding filter/volume/pitch modulation) and have a
        # non-zero speed and modulation table to sample from.
        lfo_speed = inst.lfo_rate or inst.lfo_inc
        use_lfo = (
            (inst.lfo_enable or inst.f_mod or inst.vol_mod or inst.pitch_mod)
            and lfo_speed > 0
            and inst.mod_table is not None
        )
        if use_lfo:
            if inst.lfo_oneshot:
                lfo_hz = 0.35 + (lfo_speed / 255.0) * 5.0
            else:
                # Echo3-style looping LFO — keep gentle so sustain isn't dopdopdop
                lfo_hz = 0.15 + (lfo_speed / 255.0) * 1.2
            lfo_step = (lfo_hz * 256.0) / sr
        else:
            lfo_step = 0.0

        mod_table = inst.mod_table
        f_base = inst.f_base
        f_env = inst.f_env
        f_mod = inst.f_mod
        gate_left = v.samples_left
        lfo = v.lfo_phase
        frozen = v.lfo_frozen
        mod_held = v.lfo_mod

        # Sample-by-sample envelope/LFO/filter-bank simulation. Kept as an
        # explicit Python loop (rather than vectorized) because each
        # sample's envelope/LFO state depends on the previous one.
        for i in range(n):
            # Gate has run out: force a transition into the release stage
            # regardless of where the envelope currently is.
            if gate_left <= 0 and stage < 3:
                stage = 3
                v.release = True

            target = levels[min(stage, 3)]
            if stage >= 3:
                target = 0.0
            spd = rates[min(stage, 3)]
            if stage >= 3 and spd < 1e-6:
                spd = 255.0 / (0.05 * sr)
            dist = abs(env - target)
            if dist <= spd:
                # Reached this stage's target level: snap to it and, if
                # still ramping up (attack/decay), advance to the next
                # stage. Stage 2 (sustain) never auto-advances — it only
                # moves to release (stage 3) when the gate ends above.
                env = target
                if stage < 2:
                    stage += 1
            elif env < target:
                env = env + spd
            else:
                env = env - spd

            mod = mod_held
            if mod_table is not None and lfo_step > 0 and not frozen:
                mod = float(mod_table[int(lfo) & 255]) * 128.0
                lfo += lfo_step
                # Oneshhot LFO: freeze near end of table (driver FE00>>8 ≈ 254)
                if inst.lfo_oneshot and lfo >= 254.0:
                    lfo = 254.0
                    frozen = True
                    mod = float(mod_table[254]) * 128.0
                elif not inst.lfo_oneshot:
                    # looping LFO (Echo3): wrap
                    if lfo >= 256.0:
                        lfo -= 256.0
                mod_held = mod

            env_i = int(max(0, min(255, env)))
            env_out[i] = env_i / 255.0

            if bank_out is not None:
                # Filter cutoff bank index: base cutoff, pulled down by the
                # envelope (f_env) and pushed by the LFO/mod (f_mod), then
                # divided by 4 to map the 0..255 range onto the 0..63 banks
                # produced by `sonix_one_filter`.
                filt = (255 - f_base) - ((env_i * f_env) >> 8) + int((mod * f_mod) / 256.0)
                filt = max(0, min(255, filt))
                bank_out[i] = float(filt >> 2)

            if gate_left > 0:
                gate_left -= 1

        v.env_fixed = env
        v.env_stage = stage
        v.lfo_phase = lfo
        v.lfo_frozen = frozen
        v.lfo_mod = mod_held
        v.env_level = float(env_out[-1]) if n else v.env_level
        return env_out, bank_out

    def _render_voice(self, v: Voice, n: int) -> np.ndarray:
        """Render up to `n` mono samples for voice `v`, advancing its state.

        Two independent rendering paths:
          1. Synth wavetable instruments: envelope-driven filter-bank
             interpolation over the 64x128 OneFilter table.
          2. Sample/8SVX instruments: linear-interpolated playback of a
             (possibly per-note-sliced) waveform, with an optional
             seamless "hold" loop region and vibrato.
        Both paths deactivate the voice once its envelope/sample data can no
        longer contribute audible output.
        """
        inst = v.instrument
        assert inst is not None
        take = n
        out = np.zeros(n, dtype=np.float32)

        if inst.kind == "synth" and inst.filter_banks is not None:
            env, bank = self._sonix_env_step(v, take)
            assert bank is not None
            # Oscillator phase for this block: wraps every 128 samples
            # (the wavetable's cycle length) regardless of filter bank.
            positions = v.pos + v.step * np.arange(take, dtype=np.float64)
            idx = np.mod(positions.astype(np.int64), 128)
            # Filter bank interpolation: `bank` (from the envelope step) is
            # a continuous 0..63 index — blend the two neighbouring
            # precomputed OneFilter banks so the cutoff sweep is smooth
            # instead of stepping between 64 discrete tone colors.
            b0 = np.clip(bank.astype(np.int64), 0, 63)
            b1 = np.clip(b0 + 1, 0, 63)
            frac = (bank - b0).astype(np.float32)
            samples = (
                inst.filter_banks[b0, idx] * (1.0 - frac)
                + inst.filter_banks[b1, idx] * frac
            )
            amp = env if inst.vol_env else np.ones_like(env)
            out[:take] = samples * amp * v.vol * 1.4
            v.pos = float(positions[-1] + v.step)
            v.note_samples += take
            if v.samples_left > 0:
                v.samples_left = max(0, v.samples_left - take)
            if v.env_fixed <= 1.0 and (v.env_stage >= 3 or inst.env_levels[min(v.env_stage, 3)] == 0):
                v.active = False
            return out

        # Sample / 8SVX: oneshot, then seamless hold on the short end cycle
        wave = v.sample_wave if v.sample_wave is not None else inst.wave
        ls = v.sample_loop_start
        le = v.sample_loop_end
        wlen = len(wave)
        if wlen == 0:
            v.active = False
            return out

        env, _ = self._sonix_env_step(v, take)

        base_step = v.step
        steps = np.full(take, base_step, dtype=np.float64)
        if inst.vib_depth > 0 and inst.vib_rate > 0:
            # Vibrato: sinusoidally modulate the per-sample playback step
            # (pitch) once any initial delay has elapsed.
            vib_hz = 0.8 + (inst.vib_rate / 255.0) * 6.0
            depth = (inst.vib_depth / 128.0) * 0.015  # gentler pitch vib
            delay = v.vib_delay_left
            phase = v.vib_phase
            for i in range(take):
                if delay > 0:
                    delay -= 1
                else:
                    steps[i] = base_step * (1.0 + depth * np.sin(phase))
                    phase += (2.0 * np.pi * vib_hz) / self.sr
            v.vib_delay_left = delay
            v.vib_phase = phase

        # Integrate the (possibly vibrato-modulated) per-sample step into
        # absolute fractional playback positions for this block.
        positions = np.empty(take, dtype=np.float64)
        pos = v.pos
        for i in range(take):
            positions[i] = pos
            pos += steps[i]

        if le > ls:
            ll = float(le - ls)
            # Play [0..le) once; then hold by cycling the (seamless) end period
            idx_f = np.empty(take, dtype=np.float64)
            for i in range(take):
                p = positions[i]
                if p < le:
                    idx_f[i] = min(p, wlen - 1.001)
                else:
                    # Past the one-shot body: wrap position within the
                    # [ls, le) hold region (made click-free by the
                    # crossfade applied in `_start_voice`).
                    idx_f[i] = ls + (p - ls) % ll
            i0 = np.floor(idx_f).astype(np.int64)
            frac = (idx_f - i0).astype(np.float32)
            i1 = i0 + 1
            for i in range(take):
                if positions[i] >= le:
                    # In the hold region: wrap the upper interpolation
                    # neighbor back to `ls` at the loop boundary.
                    if i1[i] >= le:
                        i1[i] = ls
                    i0[i] = min(max(i0[i], ls), le - 1)
                else:
                    i0[i] = min(max(i0[i], 0), wlen - 1)
                    i1[i] = min(i1[i], wlen - 1)
            samples = wave[i0] * (1.0 - frac) + wave[i1] * frac
            out[:take] = samples * env * v.vol
            if positions[-1] >= le:
                v.pos = ls + (positions[-1] - ls) % ll
                v.in_hold = True
            else:
                v.pos = float(positions[-1] + steps[-1])
        else:
            # No hold region: simple one-shot playback that stops once the
            # position runs past the end of the wave.
            valid = positions < wlen
            take_valid = int(np.count_nonzero(valid))
            if take_valid:
                idx = positions[:take_valid].astype(np.int64)
                out[:take_valid] = wave[idx] * env[:take_valid] * v.vol
                v.pos = float(positions[take_valid - 1] + steps[take_valid - 1])
                take = take_valid
            else:
                v.active = False
                return out

        v.note_samples += take
        if v.samples_left > 0:
            v.samples_left = max(0, v.samples_left - take)
        if le <= ls and v.pos >= wlen:
            v.active = False
        elif v.env_fixed <= 1.0 and (
            v.env_stage >= 3 or (v.env_stage >= 2 and inst.env_levels[2] == 0)
        ):
            v.active = False
        return out

    def render_block(self, n: int) -> np.ndarray:
        """Render n stereo float32 samples, shape (n, 2). Paula L-R-R-L."""
        out = np.zeros((n, 2), dtype=np.float32)
        # Sequencer/render granularity: advance the beat clock and mix
        # voices in small chunks so envelopes/LFOs update finely enough,
        # rather than re-deriving beat position once for the whole block.
        grain = 128
        pos = 0
        while pos < n:
            g = min(grain, n - pos)
            self._advance_tracks(g / self.beat_samples)
            for v in self.voices:
                if v.active and v.instrument is not None:
                    mono = self._render_voice(v, g)
                    # Amiga Paula hardware wiring: channels 0 and 3 are
                    # hard-panned left, channels 1 and 2 hard-panned right
                    # (the classic "L-R-R-L" layout), via `_CHANNEL_PAN`.
                    side = _CHANNEL_PAN[v.channel & 3]
                    out[pos : pos + g, side] += mono
            pos += g
        out *= self.master
        np.clip(out, -1.0, 1.0, out=out)
        return out

    @property
    def finished(self) -> bool:
        """True once every track has run out of events and no voice is still sounding."""
        tracks_done = all(t.done or t.index >= len(t.events) for t in self.tracks)
        voices_idle = all(not v.active for v in self.voices)
        return tracks_done and voices_idle

    def render_all(self, max_seconds: float = 600.0) -> np.ndarray:
        """Render the entire song to a single (samples, 2) float32 array.

        Renders in fixed-size blocks until the sequencer + all voices
        report `finished`, or `max_seconds` is reached as a safety cap
        against runaway/never-ending scores. Trailing near-silence is then
        trimmed (keeping a short tail) so exported WAVs don't have long
        silent gaps at the end.
        """
        chunks: List[np.ndarray] = []
        block = 2048
        max_samples = int(max_seconds * self.sr)
        total = 0
        # Kick sequencer
        self._advance_tracks(0)
        while total < max_samples:
            chunks.append(self.render_block(block))
            total += block
            if self.finished:
                # Trailing silence drain
                if all(not v.active for v in self.voices):
                    break
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 2), dtype=np.float32)
        # Trim trailing silence: find the last sample where either
        # channel's magnitude exceeds a small threshold, then keep a
        # quarter-second of tail past it (natural decay/room) and drop
        # the rest of the rendered-ahead silent buffer.
        thresh = 1e-4
        energy = np.max(np.abs(audio), axis=1)
        nz = np.where(energy > thresh)[0]
        if len(nz):
            audio = audio[: nz[-1] + self.sr // 4]
        return audio


# ---------------------------------------------------------------------------
# Load score + instruments from disk
# ---------------------------------------------------------------------------


def load_song(smus_path: Path) -> Tuple[SmusScore, Dict[int, Instrument]]:
    """Parse a .smus file and load every instrument it references.

    Any instrument that fails to load (missing file, unsupported format,
    etc.) falls back to `default_instrument` so the song can still play,
    with a warning printed to stderr. Also guarantees register 0 always
    has some instrument bound, since it's the implicit default register.
    """
    score = parse_smus(smus_path)
    folder = smus_path.parent
    instruments: Dict[int, Instrument] = {}
    for reg, name in sorted(score.instruments.items()):
        try:
            instruments[reg] = load_instrument(folder, name)
            print(f"  [{reg:2d}] {name}: {instruments[reg].kind} "
                  f"({len(instruments[reg].wave)} samples)", file=sys.stderr)
        except Exception as e:
            print(f"  [{reg:2d}] {name}: FAILED ({e}) — using default", file=sys.stderr)
            instruments[reg] = default_instrument(name)
    # Ensure register 0 exists
    if 0 not in instruments:
        instruments[0] = default_instrument()
    return score, instruments


def write_wav(path: Path, audio: np.ndarray, sample_rate: int):
    """Write a float32 (-1..1) mono or stereo array to a 16-bit PCM WAV file."""
    pcm = np.clip(audio, -1, 1)
    if pcm.ndim == 1:
        pcm = pcm.reshape(-1, 1)
        channels = 1
    else:
        channels = pcm.shape[1]
    # Interleave stereo as L,R,L,R,...
    pcm16 = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.reshape(-1).tobytes())


def play_realtime(engine: SmusEngine):
    """Stream `engine`'s output live through the default audio device.

    Requires the optional `sounddevice` dependency. Uses an audio
    callback that pulls fresh blocks from `engine.render_block` on demand
    and stops the stream once the engine reports `finished`.
    """
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Realtime play needs sounddevice: pip install sounddevice")

    engine._advance_tracks(0)
    block = 1024

    def callback(outdata, frames, time_info, status):  # noqa: ARG001
        """sounddevice output callback: fill `outdata` with the next rendered block."""
        if status:
            print(status, file=sys.stderr)
        buf = engine.render_block(frames)
        outdata[:, :2] = buf
        if engine.finished:
            raise sd.CallbackStop

    print(f"Playing '{engine.score.name}' @ {engine.bpm:.1f} BPM - Ctrl+C to stop",
          file=sys.stderr)
    with sd.OutputStream(
        samplerate=engine.sr,
        channels=2,
        dtype="float32",
        blocksize=block,
        callback=callback,
    ):
        try:
            while not engine.finished:
                sd.sleep(100)
            sd.sleep(300)
        except KeyboardInterrupt:
            print("\nStopped.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, resolve the batch of .smus files to
    process, and either play or render (or both) each one in turn.

    Path/batch resolution: if `--all` is given or `path` is a directory,
    every `.smus`/`.SMUS` file directly inside that directory is
    processed (case-insensitively de-duplicated, since some filesystems —
    notably Windows — would otherwise match the same file twice via two
    differently-cased glob patterns); otherwise just the single given
    file is processed.

    Mode resolution: WAV rendering happens if `-o/--output` was given or
    `--all` is set; realtime playback happens if `-p/--play` was passed,
    or by default when exactly one file is being processed with no
    output/--all requested. Both can happen together (`-p` with `-o`
    renders then plays back the rendered audio).
    """
    ap = argparse.ArgumentParser(description="Aegis Sonix 2 / SMUS player & WAV converter")
    ap.add_argument("path", type=Path, help=".smus file or folder (with --all)")
    ap.add_argument("-o", "--output", type=Path, help="Output WAV file or directory")
    ap.add_argument("-p", "--play", action="store_true", help="Play after / instead of render")
    ap.add_argument("--all", action="store_true", help="Convert all .smus in a folder")
    ap.add_argument("-r", "--rate", type=int, default=44100, help="Sample rate (default 44100)")
    ap.add_argument("-v", "--volume", type=float, default=0.28, help="Master volume 0..1")
    args = ap.parse_args(argv)

    paths: List[Path]
    if args.all or args.path.is_dir():
        root = args.path if args.path.is_dir() else args.path.parent
        # Case-insensitive de-dupe (Windows globs *.smus and *.SMUS the same)
        seen = set()
        paths = []
        for p in sorted(root.glob("*.smus")) + sorted(root.glob("*.SMUS")):
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                paths.append(p)
        if not paths:
            sys.exit(f"No .smus files in {root}")
    else:
        paths = [args.path]

    # Default: realtime play for a single file; render if -o / --all
    do_play = args.play or (args.output is None and not args.all and len(paths) == 1)
    do_wav = args.output is not None or args.all

    for smus in paths:
        print(f"\n=== {smus.name} ===", file=sys.stderr)
        score, instruments = load_song(smus)
        print(f"  tempo={score.tempo} ({score.tempo/128:.1f} qpm)  "
              f"tracks={len(score.tracks)}  vol={score.volume}", file=sys.stderr)
        engine = SmusEngine(score, instruments, args.rate, args.volume)

        if do_wav:
            audio = engine.render_all()
            if args.output:
                out = args.output
                if out.is_dir() or args.all or len(paths) > 1:
                    out.mkdir(parents=True, exist_ok=True)
                    out = out / (smus.stem + ".wav")
            else:
                out = smus.with_suffix(".wav")
            write_wav(out, audio, args.rate)
            print(f"  wrote {out} ({len(audio)/args.rate:.1f}s)", file=sys.stderr)
            if args.play:
                try:
                    import sounddevice as sd
                    sd.play(audio, args.rate, blocking=True)
                except ImportError:
                    print("Install sounddevice for playback", file=sys.stderr)
        elif do_play:
            play_realtime(engine)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
