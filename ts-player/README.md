# WIME Sonix Tracker

Browser TypeScript port of `smus_player.py` — Aegis Sonix 2 / IFF-SMUS playback with a live tracker UI.

## Run

```bash
cd ts-player
npm install
npm run dev
```

Open the URL Vite prints (default http://127.0.0.1:5173). Pick a song and hit **Play**.

Song data is served from `../combined` at `/music`.

## What’s included

- SMUS score parser (SHDR / NAME / INS1 / TRAK)
- Synthesis `.instr` (OneFilter banks, ADSR, LFO/filter mod)
- SampledSound `.instr` + multi-octave `.ss`
- 8SVX instruments
- 4-voice Paula L-R-R-L stereo engine
- Tracker pattern view with scrolling playhead, per-channel scopes & VU

## Smoke test

```bash
npx vite-node src/smoke.ts
```
