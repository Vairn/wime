/**
 * IFF FORM helpers — Amiga Interchange File Format uses big-endian integers
 * and padded chunks. SMUS / 8SVX / instruments are all IFF-based.
 */

/** Read an unsigned 16-bit big-endian integer at `offset`. */
export function readU16BE(data: Uint8Array, offset: number): number {
  // High byte first, then low byte.
  return (data[offset]! << 8) | data[offset + 1]!;
}

/** Read an unsigned 32-bit big-endian integer at `offset` (>>> 0 keeps it unsigned). */
export function readU32BE(data: Uint8Array, offset: number): number {
  return (
    ((data[offset]! << 24) |
      (data[offset + 1]! << 16) |
      (data[offset + 2]! << 8) |
      data[offset + 3]!) >>>
    0
  );
}

/**
 * Read a C-style null-terminated Latin-1 string from `data`.
 * Stops at the first 0 byte (or `maxLen`), then trims whitespace.
 */
export function readCString(data: Uint8Array, start = 0, maxLen?: number): string {
  const end = maxLen != null ? start + maxLen : data.length;
  let out = "";
  for (let i = start; i < end; i++) {
    const b = data[i]!;
    if (b === 0) break; // NUL terminator
    out += String.fromCharCode(b);
  }
  return out.trim();
}

/**
 * Walk IFF chunks inside a FORM body.
 *
 * Layout of each chunk: 4-byte ID, 4-byte size (BE), then `size` bytes of data.
 * Odd-sized chunks are padded with one extra byte so the next chunk is word-aligned.
 *
 * @param start - Usually 12 (skip "FORM" + size + type)
 * @param end   - Exclusive end of the FORM body
 */
export function* iterIffChunks(
  data: Uint8Array,
  start = 12,
  end = data.length,
): Generator<{ cid: string; body: Uint8Array; offset: number }> {
  let pos = start;
  while (pos + 8 <= end) {
    // Chunk ID is four ASCII characters (e.g. "SHDR", "TRAK", "BODY").
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
    // Advance past body + optional pad byte for odd sizes.
    pos = bodyStart + size + (size & 1);
  }
}

/**
 * Convert unsigned bytes interpreted as signed int8 samples into float32 −1..1.
 * Amiga sample data is typically 8-bit signed PCM stored as raw bytes.
 */
export function i8ToF32(raw: Uint8Array): Float32Array {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    let v = raw[i]!;
    // Map 128..255 → −128..−1 so high bit means negative.
    if (v >= 128) v -= 256;
    out[i] = v / 128;
  }
  return out;
}

/**
 * Interpret a 16-bit word as signed (two's complement).
 * Used by the Sonix OneFilter fixed-point math.
 */
export function toI16(x: number): number {
  x &= 0xffff;
  return x & 0x8000 ? x - 0x10000 : x;
}

/**
 * Interpret a 32-bit word as signed (two's complement).
 * Used by the Sonix OneFilter multiply / shift chain.
 */
export function toI32(x: number): number {
  x >>>= 0;
  return x & 0x80000000 ? x - 0x100000000 : x;
}
