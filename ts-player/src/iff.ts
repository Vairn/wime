/** IFF FORM chunk helpers (big-endian). */

export function readU16BE(data: Uint8Array, offset: number): number {
  return (data[offset]! << 8) | data[offset + 1]!;
}

export function readU32BE(data: Uint8Array, offset: number): number {
  return (
    ((data[offset]! << 24) |
      (data[offset + 1]! << 16) |
      (data[offset + 2]! << 8) |
      data[offset + 3]!) >>>
    0
  );
}

export function readCString(data: Uint8Array, start = 0, maxLen?: number): string {
  const end = maxLen != null ? start + maxLen : data.length;
  let out = "";
  for (let i = start; i < end; i++) {
    const b = data[i]!;
    if (b === 0) break;
    out += String.fromCharCode(b);
  }
  return out.trim();
}

export function* iterIffChunks(
  data: Uint8Array,
  start = 12,
  end = data.length,
): Generator<{ cid: string; body: Uint8Array; offset: number }> {
  let pos = start;
  while (pos + 8 <= end) {
    const cid = String.fromCharCode(
      data[pos]!,
      data[pos + 1]!,
      data[pos + 2]!,
      data[pos + 3]!,
    );
    const size = readU32BE(data, pos + 4);
    const bodyStart = pos + 8;
    const bodyEnd = Math.min(bodyStart + size, end);
    yield { cid, body: data.subarray(bodyStart, bodyEnd), offset: pos };
    pos = bodyStart + size + (size & 1);
  }
}

export function i8ToF32(raw: Uint8Array): Float32Array {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    let v = raw[i]!;
    if (v >= 128) v -= 256;
    out[i] = v / 128;
  }
  return out;
}

export function toI16(x: number): number {
  x &= 0xffff;
  return x & 0x8000 ? x - 0x10000 : x;
}

export function toI32(x: number): number {
  x >>>= 0;
  return x & 0x80000000 ? x - 0x100000000 : x;
}
