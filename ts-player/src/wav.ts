/**
 * WAV export helpers — encode float audio as downloadable 16-bit PCM WAV.
 */

/**
 * Encode interleaved stereo float32 samples (−1..1) as a RIFF/WAVE ArrayBuffer.
 *
 * Writes a standard 44-byte PCM header, then little-endian int16 L/R frames.
 */
export function encodeWav(interleaved: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples = Math.floor(interleaved.length / 2);
  const dataBytes = numSamples * 2 * 2; // stereo × 2 bytes per sample
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  /** Write an ASCII fourCC / string into the buffer at `offset`. */
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  // ---- RIFF header ----
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true); // file size minus 8
  writeStr(8, "WAVE");

  // ---- fmt  chunk (PCM) ----
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // chunk body size
  view.setUint16(20, 1, true); // format = PCM
  view.setUint16(22, 2, true); // channels = stereo
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2 * 2, true); // byte rate
  view.setUint16(32, 4, true); // block align (2 ch × 2 bytes)
  view.setUint16(34, 16, true); // bits per sample

  // ---- data chunk ----
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);

  // Convert floats → clamped int16, interleaved L,R,L,R,…
  let o = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < 2; ch++) {
      let s = interleaved[i * 2 + ch]!;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      view.setInt16(o, (s * 32767) | 0, true);
      o += 2;
    }
  }
  return buffer;
}

/**
 * Trigger a browser file download for `blob` under `filename`.
 * Creates a temporary object URL, clicks a hidden anchor, then revokes the URL.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
