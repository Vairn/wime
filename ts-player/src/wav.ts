/** Encode interleaved stereo float32 (−1..1) as a 16-bit PCM WAV ArrayBuffer. */
export function encodeWav(interleaved: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples = Math.floor(interleaved.length / 2);
  const dataBytes = numSamples * 2 * 2; // stereo int16
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 2, true); // stereo
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2 * 2, true); // byte rate
  view.setUint16(32, 4, true); // block align
  view.setUint16(34, 16, true); // bits
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);

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

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
