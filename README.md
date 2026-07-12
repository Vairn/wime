# War in Middle-earth — Sonix SMUS

Tools for playing and rendering **Aegis Sonix 2 / IFF-SMUS** music from
*War in Middle Earth* (Melbourne House, 1989).

Scores and instruments live in `combined/` (merged from the two Amiga disks).

## Quick start (browser tracker)

```bash
cd ts-player
npm install
npm run dev
```

Open the URL Vite prints (usually http://127.0.0.1:5173). Pick a song, hit
**Play**, or **Export WAV** for a stereo 44.1 kHz download.

## Python player / WAV renderer

```bash
pip install numpy sounddevice   # sounddevice only needed for realtime play

python smus_player.py combined/Hob.Riven.smus           # play
python smus_player.py combined/Title.smus -o title.wav  # render WAV
python smus_player.py combined --all -o wav/            # batch folder
```

## Layout

| Path | Contents |
|------|----------|
| `ts-player/` | TypeScript web player — tracker UI, staff score, scopes, WAV export |
| `smus_player.py` | Python CLI player and WAV renderer |
| `combined/` | `.smus` scores + `.instr` / `.ss` / 8SVX instruments |
| `disk1/`, `disk2/` | Raw Amiga ADF extracts |
| `wav/` | Optional pre-rendered WAVs |
| `_ref/` | Sonix driver reference sources |

## What the engine supports

- SMUS scores (`SHDR` / `NAME` / `INS1` / `TRAK`)
- Synthesis instruments (OneFilter wavetable banks, ADSR, LFO / filter mod)
- SampledSound `.instr` + multi-octave `.ss`
- 8SVX samples
- 4-voice Amiga Paula panning (L-R-R-L)

## License / provenance

Game assets are from the original commercial release and are not redistributed
here as a claim of ownership. Code in this repo is for preservation and study.
