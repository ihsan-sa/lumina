# 3D Simulator Architecture

## Overview

The LUMINA simulator is a browser-based 3D visualization that renders a virtual venue with
light fixtures responding to music in real-time. It uses the exact same fixture control
protocol as physical hardware, ensuring that code tested in the simulator works on real
fixtures with zero changes to the control layer.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 3D rendering | Three.js via @react-three/fiber | Room model, fixture beams, light simulation |
| UI framework | React 18+ | Control panels, audio visualization |
| Audio visualization | Web Audio API | Real-time FFT for client-side spectrogram |
| Backend communication | WebSocket | Receive fixture commands from Python backend |
| Build tool | Vite | Fast dev server + production builds |
| Styling | Tailwind CSS | UI component styling |

## Architecture

```
Python Backend (lumina/app.py)
  |
  |-- WebSocket (port 8765)
  |     |-- Sends: FixtureCommand[] at 60fps
  |     |-- Sends: MusicState for UI display
  |     |-- Sends: Audio file data for browser playback
  |     |-- Receives: Transport controls, genre overrides, intensity
  |
Browser (simulator/)
  |-- Three.js Scene
  |     |-- Room.tsx: 3D room geometry (5m x 7m x 2.5m)
  |     |-- Fixture.tsx: Virtual fixtures with beam rendering
  |     |     |-- RGBW Par: Cone beam with color wash
  |     |     |-- RGB Strobe: Flash effect
  |     |     |-- UV Bar: UV glow effect
  |     |     |-- LED Bar: Linear wash
  |     |     |-- Laser: Beam lines
  |-- UI Overlay
  |     |-- ControlPanel.tsx: Mode selection, fixture config
  |     |-- AudioPlayer.tsx: Playback controls + waveform
  |     |-- Spectrogram.tsx: Real-time frequency display
  |-- Hooks
        |-- useWebSocket.ts: WebSocket connection management
        |-- useAudio.ts: Web Audio API integration
```

## Room Model

Default room dimensions match the target venue:
- **Size**: 5m x 7m x 2.5m ceiling (16ft x 23ft x 8ft)
- **Default fixture layout**: 15 fixtures (4 pars, 4 strobes, 3 UV bars, 2 LED bars, 2 lasers)
- **Coordinate system**: Y-up, origin at room center floor level

## Fixture Rendering

Each virtual fixture type renders differently:

| Type | Visual Representation |
|------|----------------------|
| RGBW Par | Cone-shaped volumetric beam with color and intensity |
| RGB Strobe | Bright flash overlay with bloom effect |
| UV Bar | Purple glow with fluorescence simulation |
| LED Bar | Linear color wash along bar length |
| Laser | Thin beam lines with scatter effect |

## WebSocket Protocol

The simulator connects to the Python backend via WebSocket and receives:

1. **Fixture commands**: Array of `FixtureCommand` objects at 60fps
2. **Music state**: Current `MusicState` for UI display (energy, segment, genre, etc.)
3. **Audio data**: Base64-encoded audio file for synchronized browser playback

The simulator sends:
1. **Transport controls**: Play, pause, seek
2. **Genre override**: User-selected genre profile
3. **Intensity**: Global intensity slider value
4. **Manual effects**: Blackout, strobe burst, UV flash

## Development

```bash
cd simulator
npm install
npm run dev    # Starts Vite dev server on http://localhost:5173
```

The simulator connects to the Python backend at `ws://localhost:8765`. Both must be running
for the full experience.
