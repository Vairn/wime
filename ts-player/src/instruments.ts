import { i8ToF32, iterIffChunks, readCString, readU16BE, toI16, toI32 } from "./iff";

const FILTER_COEFFS = [
  0x8000, 0x7683, 0x6dba, 0x6597, 0x5e10, 0x5717, 0x50a2, 0x4aa8, 0x451f, 0x4000, 0x3b41,
  0x36dd, 0x32cb, 0x2f08, 0x2b8b, 0x2851, 0x2554, 0x228f, 0x2000, 0x1da0, 0x1b6e, 0x1965,
  0x1784, 0x15c5, 0x1428, 0x12aa, 0x1147, 0x1000, 0x0ed0, 0x0db7, 0x0cb2, 0x0bc2, 0x0ae2,
  0x0a14, 0x0955, 0x08a3, 0x0800, 0x0768, 0x06db, 0x0659, 0x05e1, 0x0571, 0x050a, 0x04aa,
  0x0451, 0x0400, 0x03b4, 0x036d, 0x032c, 0x02f0, 0x02b8, 0x0285, 0x0255, 0x0228, 0x0200,
  0x01da, 0x01b6, 0x0196, 0x0178, 0x015c, 0x0142, 0x012a, 0x0114, 0x0100,
];

export type InstrumentKind = "synth" | "sample" | "8svx";

export interface Instrument {
  name: string;
  kind: InstrumentKind;
  wave: Float32Array;
  loopStart: number;
  loopEnd: number;
  baseMidi: number;
  baseRate: number;
  volume: number;
  filterBanks: Float32Array[] | null;
  modTable: Float32Array | null;
  envLevels: [number, number, number, number];
  envRates: [number, number, number, number];
  fBase: number;
  fEnv: number;
  fMod: number;
  lfoRate: number;
  lfoInc: number;
  lfoEnable: boolean;
  lfoOneshot: boolean;
  volRaw: number;
  volEnv: boolean;
  volMod: number;
  pitchMod: number;
  ssOneshot: number;
  ssRepeat: number;
  ssLo: number;
  ssHi: number;
  ssData: Float32Array | null;
  vibDepth: number;
  vibRate: number;
  vibDelay: number;
}

function emptyInstrument(name: string, kind: InstrumentKind): Instrument {
  return {
    name,
    kind,
    wave: new Float32Array(128),
    loopStart: 0,
    loopEnd: 0,
    baseMidi: 60,
    baseRate: 8363,
    volume: 1,
    filterBanks: null,
    modTable: null,
    envLevels: [255, 255, 200, 0],
    envRates: [128, 128, 128, 64],
    fBase: 128,
    fEnv: 0,
    fMod: 0,
    lfoRate: 0,
    lfoInc: 0,
    lfoEnable: false,
    lfoOneshot: true,
    volRaw: 255,
    volEnv: true,
    volMod: 0,
    pitchMod: 0,
    ssOneshot: 0,
    ssRepeat: 0,
    ssLo: 0,
    ssHi: 0,
    ssData: null,
    vibDepth: 0,
    vibRate: 0,
    vibDelay: 0,
  };
}

export function sonixOneFilter(wave128: Uint8Array): Int8Array {
  const wave = new Int8Array(128);
  for (let i = 0; i < 128; i++) {
    let b = wave128[i]!;
    wave[i] = b >= 128 ? b - 256 : b;
  }
  const out = new Int8Array(64 * 128);
  let d3 = 0;
  let d4 = toI16(wave[127]! << 7);
  let oi = 0;
  for (let step = 0; step < 64; step++) {
    let d1 = FILTER_COEFFS[step]!;
    let d2 = (0x8000 - d1) & 0xffff;
    d2 = ((d2 * 0xe666) >>> 0) >>> 16;
    d1 >>= 1;
    for (let s = 0; s < 128; s++) {
      const d6 = toI16(toI16(wave[s]! << 7) - d4);
      let prod = toI32(toI16(d1) * d6);
      prod = toI32(prod << 2);
      d3 = toI16(d3 + (prod >> 16));
      d4 = toI16(d4 + d3);
      const d4u = d4 & 0xffff;
      const ror = ((d4u >> 7) | ((d4u & 0x7f) << 9)) & 0xffff;
      out[oi++] = ror & 0xff;
      const prod3 = toI32(toI16(d3) * toI16(d2));
      d3 = toI16(toI32(prod3 << 1) >> 16);
    }
  }
  return out;
}

export function sonixRateUnits(r: number): number {
  r &= 0xffff;
  if (r === 0) return 4000;
  const exp = 7 ^ ((r >> 5) & 7);
  const mant = (r & 0x1f) + 0x21;
  return mant << exp;
}

export function sampleOctaveForMidi(midi: number, lo: number, hi: number): number {
  const octv = 10 - Math.floor(midi / 12);
  return Math.max(lo, Math.min(hi, octv));
}

function load8svx(data: Uint8Array, name: string): Instrument {
  let oneshot = 0;
  let repeat = 0;
  let rate = 8363;
  let body = new Uint8Array(0);
  for (const { cid, body: chunk } of iterIffChunks(data)) {
    if (cid === "VHDR" && chunk.length >= 14) {
      oneshot = (chunk[0]! << 24) | (chunk[1]! << 16) | (chunk[2]! << 8) | chunk[3]!;
      oneshot >>>= 0;
      repeat =
        (((chunk[4]! << 24) | (chunk[5]! << 16) | (chunk[6]! << 8) | chunk[7]!) >>> 0);
      rate = readU16BE(chunk, 12) || 8363;
    } else if (cid === "BODY") {
      body = new Uint8Array(chunk);
    }
  }
  const wave = i8ToF32(body);
  const inst = emptyInstrument(name, "8svx");
  inst.wave = wave;
  inst.loopStart = oneshot;
  inst.loopEnd = repeat ? oneshot + repeat : 0;
  inst.baseRate = rate;
  return inst;
}

export function loadSs(
  data: Uint8Array,
  name: string,
  volume = 1,
  envLevels: [number, number, number, number] = [255, 255, 255, 0],
  envRates: [number, number, number, number] = [255, 255, 255, 128],
  vibDepth = 0,
  vibRate = 0,
  vibDelay = 0,
): Instrument {
  if (data.length < 64) throw new Error(`Truncated .ss: ${name}`);
  const oneshot = readU16BE(data, 0);
  const repeat = readU16BE(data, 2);
  let loOct = data[4]!;
  let hiOct = data[5]!;
  if (hiOct < loOct) hiOct = loOct;
  const payload = i8ToF32(data.subarray(0x3e));
  const mid = sampleOctaveForMidi(60, loOct, hiOct);
  const off = oneshot * ((1 << mid) - (1 << loOct));
  const ln = oneshot << mid;
  let wave = payload.subarray(off, off + ln);
  if (wave.length === 0) {
    wave = payload.subarray(0, Math.max(1, Math.min(payload.length, oneshot << loOct)));
  }
  const inst = emptyInstrument(name, "sample");
  inst.wave = wave.length ? new Float32Array(wave) : new Float32Array(1);
  inst.volume = volume;
  inst.envLevels = envLevels;
  inst.envRates = envRates;
  inst.ssOneshot = oneshot;
  inst.ssRepeat = repeat;
  inst.ssLo = loOct;
  inst.ssHi = hiOct;
  inst.ssData = payload;
  inst.vibDepth = vibDepth;
  inst.vibRate = vibRate;
  inst.vibDelay = vibDelay;
  return inst;
}

function bodyU16(body: Uint8Array, off: number): number {
  if (off + 2 > body.length) return 0;
  return readU16BE(body, off);
}

export function loadSynthInstr(data: Uint8Array, name: string): Instrument {
  if (data.length < 68 + 128) throw new Error("Truncated Synthesis instrument");
  const waveRaw = data.subarray(68, 196);
  const banksI8 = sonixOneFilter(waveRaw);
  const banks: Float32Array[] = [];
  for (let b = 0; b < 64; b++) {
    const row = new Float32Array(128);
    for (let s = 0; s < 128; s++) {
      let v = banksI8[b * 128 + s]!;
      row[s] = v / 128;
    }
    banks.push(row);
  }

  const body = data.subarray(32);
  const modRaw = body.subarray(0xa4, 0xa4 + 256);
  const modTable = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    let v = modRaw[i] ?? 0;
    if (v >= 128) v -= 256;
    modTable[i] = v / 128;
  }

  const volRaw = bodyU16(body, 0x1ac) & 0xff;
  const vol = 0.35 + 0.65 * (Math.max(volRaw, 1) / 255);
  const volEnv = bodyU16(body, 0x1ae) !== 0;
  const volMod = bodyU16(body, 0x1b0) & 0xff;
  const pitchMod = bodyU16(body, 0x1b4) & 0xff;
  const levels: [number, number, number, number] = [
    bodyU16(body, 0x1c6) & 0xff,
    bodyU16(body, 0x1c8) & 0xff,
    bodyU16(body, 0x1ca) & 0xff,
    bodyU16(body, 0x1cc) & 0xff,
  ];
  const rates: [number, number, number, number] = [
    bodyU16(body, 0x1ce),
    bodyU16(body, 0x1d0),
    bodyU16(body, 0x1d2),
    bodyU16(body, 0x1d4),
  ];

  const fBase = bodyU16(body, 0x1b6) & 0xff;
  const fEnv = bodyU16(body, 0x1b8) & 0xff;
  const fMod = bodyU16(body, 0x1ba) & 0xff;
  const lfoInc = bodyU16(body, 0x1bc) & 0xff;
  const lfoRate = bodyU16(body, 0x1c0) & 0xff;
  const lfoWord = bodyU16(body, 0x1be);
  const lfoSigned = lfoWord >= 0x8000 ? lfoWord - 0x10000 : lfoWord;
  const lfoEnable = lfoWord !== 0;
  const lfoOneshot = lfoSigned >= 0;

  const bank0 = Math.max(
    0,
    Math.min(63, ((255 - fBase) - ((255 * fEnv) >> 8)) >> 2),
  );

  const inst = emptyInstrument(name, "synth");
  inst.wave = banks[bank0]!.slice();
  inst.loopStart = 0;
  inst.loopEnd = 128;
  inst.baseRate = 16574.27;
  inst.volume = vol;
  inst.filterBanks = banks;
  inst.modTable = modTable;
  inst.envLevels = levels;
  inst.envRates = rates;
  inst.fBase = fBase;
  inst.fEnv = fEnv;
  inst.fMod = fMod;
  inst.lfoRate = lfoRate;
  inst.lfoInc = lfoInc;
  inst.lfoEnable = lfoEnable;
  inst.lfoOneshot = lfoOneshot;
  inst.volRaw = volRaw;
  inst.volEnv = volEnv;
  inst.volMod = volMod;
  inst.pitchMod = pitchMod;
  return inst;
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

function decodeAscii(data: Uint8Array, start: number, len: number): string {
  return readCString(data.subarray(start, start + len));
}

export async function loadSampledInstr(
  data: Uint8Array,
  name: string,
  folderUrl: string,
  fetchFile: (name: string) => Promise<Uint8Array>,
): Promise<Instrument> {
  const ssName = decodeAscii(data, 68, 24);
  if (!ssName) throw new Error(`No .ss name in ${name}`);
  const body = data.length >= 32 ? data.subarray(32) : data;
  const volWord = bodyU16(body, 0x48) || 0xc0;
  const volume = Math.max(volWord, 1) / 255;
  const levels: [number, number, number, number] = [
    bodyU16(body, 0x4a) & 0xff,
    bodyU16(body, 0x4c) & 0xff,
    bodyU16(body, 0x4e) & 0xff,
    bodyU16(body, 0x50) & 0xff,
  ];
  const rates: [number, number, number, number] = [
    bodyU16(body, 0x52),
    bodyU16(body, 0x54),
    bodyU16(body, 0x56),
    bodyU16(body, 0x58),
  ];
  const vibDepth = bodyU16(body, 0x5a) & 0xff;
  const vibRate = bodyU16(body, 0x5c) & 0xff;
  const vibDelay = bodyU16(body, 0x5e) & 0xff;

  const candidates = [`${ssName}.ss`, `${ssName}.SS`];
  let lastErr: unknown;
  for (const c of candidates) {
    try {
      const ss = await fetchFile(c);
      return loadSs(ss, name, volume, levels, rates, vibDepth, vibRate, vibDelay);
    } catch (e) {
      lastErr = e;
    }
  }
  // case-insensitive probe via catalog listing is handled by caller
  void folderUrl;
  throw lastErr ?? new Error(`Missing sample '${ssName}.ss' for ${name}`);
}

export function defaultInstrument(name = "default"): Instrument {
  const wave = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    const t = (i / 128) * Math.PI * 2;
    wave[i] = 0.4 * Math.sin(t) + 0.2 * Math.sin(2 * t);
  }
  const inst = emptyInstrument(name, "synth");
  inst.wave = wave;
  inst.loopEnd = 128;
  inst.baseRate = 16574.27;
  return inst;
}

export function detectAndLoadInstrument(
  data: Uint8Array,
  name: string,
): Instrument | "sampled" {
  const form = String.fromCharCode(data[0]!, data[1]!, data[2]!, data[3]!);
  const type = String.fromCharCode(data[8]!, data[9]!, data[10]!, data[11]!);
  if (form === "FORM" && type === "8SVX") return load8svx(data, name);
  if (form === "FORM" && type === "AIFF") throw new Error(`AIFF not supported: ${name}`);

  const hdr = String.fromCharCode(...data.subarray(0, Math.min(12, data.length)));
  if (data.length === 128 && hdr.startsWith("SampledSound")) return "sampled";
  if (
    data.length === 502 &&
    (hdr.startsWith("Synthesis") || (data[0] === 0 && data[1] === 0 && data[2] === 0 && data[3] === 0))
  ) {
    return loadSynthInstr(data, name);
  }
  if (data.length === 502) return loadSynthInstr(data, name);
  if (data.length === 128) return "sampled";
  throw new Error(`Unknown instrument format: ${name} (${data.length} bytes)`);
}

export async function loadInstrumentByName(
  folderUrl: string,
  name: string,
  fileIndex: Map<string, string>,
): Promise<Instrument> {
  const fetchFile = async (fileName: string): Promise<Uint8Array> => {
    const mapped = fileIndex.get(fileName.toLowerCase()) ?? fileName;
    return fetchBytes(`${folderUrl}/${encodeURIComponent(mapped).replace(/%2F/g, "/")}`);
  };

  // Find .instr (case-insensitive via index)
  const instrKey = `${name}.instr`.toLowerCase();
  let instrFile = fileIndex.get(instrKey);
  if (!instrFile) {
    // bare name
    for (const [k, v] of fileIndex) {
      if (k.replace(/\.[^.]+$/, "") === name.toLowerCase()) {
        instrFile = v;
        break;
      }
    }
  }
  if (!instrFile) throw new Error(`Instrument not found: ${name}`);

  const data = await fetchFile(instrFile);
  const loaded = detectAndLoadInstrument(data, name);
  if (loaded === "sampled") {
    return loadSampledInstr(data, name, folderUrl, async (ssName) => {
      const key = ssName.toLowerCase();
      const mapped = fileIndex.get(key);
      if (!mapped) {
        // try stem match
        const stem = ssName.replace(/\.ss$/i, "").toLowerCase();
        for (const [k, v] of fileIndex) {
          if (k.endsWith(".ss") && k.replace(/\.ss$/, "") === stem) return fetchFile(v);
        }
        throw new Error(`Missing ${ssName}`);
      }
      return fetchFile(mapped);
    });
  }
  return loaded;
}
