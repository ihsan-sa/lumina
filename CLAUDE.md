# LUMINA — AI-Powered Intelligent Light Show System

## Project Overview

LUMINA is an AI-powered light show system for a basement/garage venue. It goes far beyond beat-syncing — it understands musical narrative (builds, drops, breakdowns, vocal energy, tension, release) and translates that into professional-grade lighting decisions in real time.

The system consists of custom PoE-networked light fixtures (strobes, RGB pars, UV bars; lasers and moving heads in later phases) controlled by a central AI host that analyzes music, generates lighting choreography, and optionally DJs entire sets.

**Target Genres:** Hip-hop, trap, R&B, techno, house, EDM — with 8 specific lighting profiles mapped to reference artists (Playboi Carti, The Weeknd, Stromae, Ninho, Jul, Kaaris, Fred again.., David Guetta, Armin van Buuren, Edward Maya, AyVe, Exetra Archive, Don Toliver, Travis Scott).

**Key Principle:** This is NOT a beat-sync system. The AI must understand song structure, genre conventions, emotional arc, and lighting design philosophy. A Carti mosh-pit track and a Stromae theatrical ballad are both "music" but demand completely opposite lighting.

---

## Architecture

### Three Layers

1. **Audio Intelligence Layer** — Processes music in real-time, extracts beats, segments, energy, drops, genre classification
2. **Light Show Brain** — Translates musical features into lighting decisions using genre-specific profiles (rule-based expert system + ML classification)
3. **Fixture Control Network** — PoE Ethernet delivering UDP commands to custom PCB-based fixtures

### Data Flow

```
Audio Source (Spotify/file/mic)
  → Audio Analysis Engine (librosa, madmom, essentia)
  → Feature Extraction (beats, energy, segments, genre, vocals)
  → AI Light Show Model (genre profiles + rule engine)
  → Command Generation (fixture-level RGBW + strobe + effects)
  → PoE Network (UDP unicast/multicast, 60fps target)
  → Fixture MCUs (STM32F4 + W5500)
  → LED/Laser Drivers (constant-current PWM)
```

### Three Operating Modes

- **Mode A (Live Listening):** User plays music from any source. System captures via loopback/line-in, analyzes in real-time (~100-200ms latency), generates light show live.
- **Mode B (AI DJ + Light Show):** User provides genre/mood/playlist. AI selects tracks, handles BPM matching, transitions, and pre-generates light shows for the full set.
- **Mode C (Queue Pre-Processing):** User controls playlist, AI pre-processes next song in queue (~3-5 min look-ahead). Analyzes structure, identifies key moments, pre-generates lighting timeline.

### Audio Capture Strategy

**Primary music source is streaming (Spotify/Apple Music).** These services do not expose raw audio via their APIs. The system captures audio via:

- **Production (Jetson):** PulseAudio/PipeWire monitor source — captures all system audio output as a loopback. The Jetson plays music through its audio output (connected to speakers/amp), and the monitor source taps that stream for analysis. This is Mode A's primary capture method.
- **Development (Windows):** Use WASAPI loopback capture (via `sounddevice` with WASAPI backend) to capture system audio output, or use local audio files (MP3/FLAC) as input to the analysis pipeline. The `scripts/capture_audio.py` utility should support both file input and system audio loopback, selected via config. WASAPI loopback is Windows-native and works reliably.
- **Alternative for Mode C:** If the user queues songs that are also available as local files, Mode C can pre-process directly from the file for higher quality analysis (no lossy loopback).

### Control Interface

The system is controlled via a **mobile-responsive web app** served directly by the Jetson host. During a party, the user opens `http://lumina.local` on their phone/tablet to:

- Select operating mode (A/B/C)
- View current song analysis (energy, segment, genre detected)
- Override genre profile if the AI gets it wrong
- Adjust global intensity / "party level"
- Trigger manual effects (blackout, strobe burst, color wash)
- Manage queue (Mode C)

The web UI is built in React (shared codebase with the simulator frontend) and communicates with the Python backend via WebSocket. The simulator 3D view is a dev-only feature; the party UI is a simplified control panel.

---

## Core Data Contracts

These two dataclasses are the most important interfaces in the system. Every module depends on them.

### MusicState (Audio → Lighting)

Produced by the audio engine at 60fps, consumed by the lighting engine:

```python
@dataclass
class MusicState:
    timestamp: float          # Current time in seconds
    bpm: float               # Current tempo
    beat_phase: float        # 0.0-1.0 position within current beat
    bar_phase: float         # 0.0-1.0 position within current bar
    is_beat: bool            # True on exact beat frames
    is_downbeat: bool        # True on bar downbeats
    energy: float            # 0.0-1.0 overall energy
    energy_derivative: float # Rising (+) or falling (-) energy
    segment: str             # "verse", "chorus", "drop", "breakdown", "intro", "outro", "bridge"
    genre_weights: dict[str, float]  # Profile name → weight (sum to 1.0)
    vocal_energy: float      # 0.0-1.0 vocal presence
    spectral_centroid: float # Brightness indicator
    sub_bass_energy: float   # Sub-bass (20-80Hz) energy
    onset_type: str | None   # "kick", "snare", "hihat", "clap", or None
    drop_probability: float  # 0.0-1.0 probability of drop in next 1-4 bars
```

### FixtureCommand (Lighting → Network)

Produced by the lighting engine, sent to fixtures via UDP at 60fps:

```python
@dataclass
class FixtureCommand:
    fixture_id: int       # Target fixture (1-255, 0 = broadcast)
    red: int              # 0-255
    green: int            # 0-255
    blue: int             # 0-255
    white: int            # 0-255
    strobe_rate: int      # 0 (off) to 255 (max rate, ~25Hz)
    strobe_intensity: int # 0-255
    special: int          # Fixture-type-specific (UV level, laser pattern, etc.)
```

Every fixture receives the same 8-byte command (1-byte fixture_id + 7 channel bytes) regardless of type. Fixtures ignore irrelevant fields (e.g., a UV bar only reads `special` as its intensity, ignores RGBW and strobe fields).

### UDP Packet Format (implemented in `lumina/control/protocol.py`)

Protocol constants:
- **Magic:** `0x4C55` ("LU" in ASCII)
- **Version:** `1`
- **Port:** `5150`
- **Max fixtures per packet:** `32`
- **Max packet size:** 265 bytes (9-byte header + 32 × 8-byte commands)

```
Header (9 bytes, little-endian):
  magic:         uint16  (0x4C55)
  version:       uint8   (1)
  packet_type:   uint8   (0x01=COMMAND, 0x10=DISCOVER_REQ, 0x11=DISCOVER_RESP, 0x20=HEARTBEAT, 0x30=CONFIG)
  sequence:      uint16  (wraps at 65535)
  timestamp_ms:  uint16  (wraps at 65535)
  fixture_count: uint8   (0-32)

Payload (fixture_count × 8 bytes each):
  fixture_id:       uint8
  red:              uint8
  green:            uint8
  blue:             uint8
  white:            uint8
  strobe_rate:      uint8
  strobe_intensity: uint8
  special:          uint8
```

---

## Repo Structure (Monorepo)

```
lumina/
├── CLAUDE.md                    # This file — project context for Claude Code
├── README.md                    # Public-facing project README
├── .claude/
│   ├── agents/                  # Subagent definitions
│   │   ├── audio-engineer.md
│   │   ├── lighting-designer.md
│   │   ├── simulator-dev.md
│   │   ├── firmware-engineer.md
│   │   └── protocol-engineer.md
│   └── settings.json
├── docs/
│   ├── project-plan.md          # Full project plan
│   ├── genre-lighting-profiles.md  # 8 genre lighting profiles deep-dive
│   ├── protocol-spec.md         # UDP fixture control protocol spec
│   ├── hardware/
│   │   ├── fixture-pcb.md       # PCB architecture and part selection
│   │   ├── poe-design.md        # PoE power delivery design
│   │   └── datasheets/          # Component datasheets (PDF)
│   └── architecture/
│       ├── audio-pipeline.md    # Audio analysis architecture
│       ├── ai-model.md          # Light show AI model design
│       └── simulator.md         # 3D simulator architecture
├── lumina/                      # Python package (import lumina.audio, lumina.lighting, etc.)
│   ├── __init__.py
│   ├── audio/                   # Audio analysis engine
│   │   ├── __init__.py
│   │   ├── beat_detector.py     # Beat/downbeat/bar detection
│   │   ├── onset_detector.py    # Transient detection (kick, snare, hi-hat)
│   │   ├── energy_tracker.py    # Energy envelope + derivative
│   │   ├── segment_classifier.py  # Verse/chorus/drop/breakdown classification
│   │   ├── genre_classifier.py  # Two-stage genre classification
│   │   ├── vocal_detector.py    # Language-agnostic vocal presence/energy
│   │   ├── drop_predictor.py    # 1-4 bar look-ahead drop prediction
│   │   └── source_separator.py  # Demucs/spleeter integration
│   ├── lighting/                # Light show AI engine
│   │   ├── __init__.py
│   │   ├── engine.py            # Main lighting decision engine
│   │   ├── profiles/            # Genre-specific lighting profiles
│   │   │   ├── base.py          # Base profile class
│   │   │   ├── rage_trap.py     # Profile 1: Carti, Travis Scott
│   │   │   ├── psych_rnb.py     # Profile 2: Don Toliver, Weeknd
│   │   │   ├── french_melodic.py  # Profile 3: Ninho, Jul
│   │   │   ├── french_hard.py   # Profile 4: Kaaris
│   │   │   ├── euro_alt.py      # Profile 5: AyVe, Exetra Archive
│   │   │   ├── theatrical.py    # Profile 6: Stromae
│   │   │   ├── festival_edm.py  # Profile 7: Guetta, Armin, Edward Maya
│   │   │   └── uk_bass.py       # Profile 8: Fred again..
│   │   ├── blender.py           # Profile blending with weights
│   │   └── transitions.py       # Cross-genre transition engine
│   ├── control/                 # Fixture network control
│   │   ├── __init__.py
│   │   ├── protocol.py          # UDP packet encoding/decoding
│   │   ├── fixture.py           # Fixture abstraction (RGBW, strobe, laser)
│   │   ├── discovery.py         # mDNS fixture discovery
│   │   └── network.py           # Network manager (send commands at 60fps)
│   ├── dj/                      # AI DJ engine (Phase 4)
│   │   ├── __init__.py
│   │   ├── mixer.py             # BPM matching, crossfading
│   │   ├── set_builder.py       # Set planning and energy arc
│   │   └── queue.py             # Queue management + pre-processing
│   ├── web/                     # Web server for mobile control UI
│   │   ├── __init__.py
│   │   ├── server.py            # Starlette HTTP + WebSocket server
│   │   └── static/              # Built React UI served as static files
│   └── app.py                   # Main application entry point
├── simulator/                   # Browser-based 3D simulator
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Room.tsx         # 3D room model
│   │   │   ├── Fixture.tsx      # Virtual fixture with beam rendering
│   │   │   ├── AudioPlayer.tsx  # Audio playback + waveform
│   │   │   ├── Spectrogram.tsx  # Real-time spectrogram
│   │   │   └── ControlPanel.tsx # Mode selection, fixture config
│   │   ├── hooks/
│   │   │   ├── useAudio.ts      # Web Audio API integration
│   │   │   └── useWebSocket.ts  # Connection to Python backend
│   │   └── types/
│   │       └── fixtures.ts      # Fixture type definitions
│   └── public/
├── firmware/                    # Fixture MCU firmware (STM32F4, C bare-metal)
│   ├── CMakeLists.txt           # Generated by CubeMX
│   ├── Core/                    # CubeMX-generated HAL init (DO NOT EDIT)
│   │   ├── Inc/
│   │   └── Src/
│   ├── Drivers/                 # STM32 HAL drivers (DO NOT EDIT)
│   ├── App/                     # Application code (Claude Code writes this)
│   │   ├── inc/
│   │   │   └── config.h         # Fixture ID, network config, pin mappings
│   │   └── src/
│   │       ├── app_main.c       # Main application loop
│   │       ├── w5500_driver.c   # W5500 SPI + UDP socket
│   │       ├── udp_listener.c   # LUMINA protocol parser
│   │       ├── led_controller.c # RGBW color mixing + strobe logic
│   │       ├── pwm_output.c     # HAL_TIM_PWM wrapper
│   │       └── discovery.c      # mDNS announcement + heartbeat
│   └── lumina_fixture.ioc       # CubeMX project file (owner configures)
├── tests/
│   ├── test_beat_detector.py
│   ├── test_genre_classifier.py
│   ├── test_protocol.py
│   ├── test_lighting_engine.py
│   └── conftest.py              # Shared fixtures (audio samples, etc.)
├── scripts/
│   ├── capture_audio.py         # Audio loopback capture utility
│   ├── fixture_tester.py        # Send test commands to fixtures
│   └── profile_demo.py          # Demo each lighting profile in simulator
├── pyproject.toml               # Python project config (dependencies, ruff, pytest)
└── .github/
    └── workflows/
        └── ci.yml               # Lint + test on PR
```

---

## Tech Stack

### Host Software (Python 3.12)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Audio analysis | librosa, madmom, essentia, aubio | Beat detection, features, classification |
| Source separation | demucs | Isolate vocals/drums/bass (GPU-accelerated) |
| ML framework | PyTorch | Genre classifier, drop predictor |
| Networking | asyncio + socket | UDP fixture control at 60fps |
| WebSocket server | websockets | Bridge to 3D simulator |
| Audio capture | sounddevice (WASAPI on Windows, PulseAudio on Linux) | Real-time audio input |
| Package management | pip + venv | Virtual environments, standard tooling |
| Linting/formatting | ruff | Linting + formatting (replaces black + isort + flake8) |
| Testing | pytest | Critical path testing |
| Type checking | mypy (strict mode) | Static type analysis |

### 3D Simulator (TypeScript + React)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 3D rendering | Three.js (via @react-three/fiber) | Room model, fixture beams, light simulation |
| UI framework | React 18+ | Control panels, audio visualization |
| Audio analysis (browser) | Web Audio API | Real-time FFT for client-side visualization |
| WebSocket client | native WebSocket | Receive fixture commands from Python backend |
| Build tool | Vite | Fast dev server + build |
| Styling | Tailwind CSS | UI components |

### Firmware (C, bare-metal on STM32)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| MCU | STM32F4 (STM32F407/F446) | ARM Cortex-M4 @ 168MHz, hardware FPU, advanced timers |
| Ethernet | W5500 SPI module | Hardware TCP/IP offload, UDP reception |
| LED control | STM32 TIM peripherals (hardware PWM) | Per-channel constant-current dimming, 20kHz |
| Peripheral init | STM32CubeMX (generates HAL code) | Clock tree, pin config, timer setup, SPI config |
| Application firmware | Written by Claude Code (on top of CubeMX output) | UDP listener, LED controller, strobe engine, discovery |
| Build system | CMake (generated by CubeMX) | Cross-compilation via arm-none-eabi-gcc |
| Debugging | STM32CubeIDE (debugger only) | Step-through, breakpoints, flash programming |
| Future wireless | ESP32-S3 (Phase 3+) | Battery-powered wireless accent fixtures |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Production host | NVIDIA Jetson Orin Nano | Central compute — 6-core ARM + 128-core CUDA GPU, runs all AI/audio |
| Dev machine | Laptop with NVIDIA RTX 4070 (Windows native) | Development and testing |
| Network | PoE+ managed switch (8-16 port, 802.3af/at) | Power + data to fixtures |
| Protocol | Custom UDP (lightweight, <1ms latency) | Fixture command delivery |
| Audio I/O | USB audio interface | Line-in capture for Mode A |

---

## Development vs Production Environment

### Development (Windows Native)

- **OS:** Windows 10/11
- **GPU:** NVIDIA RTX 4070 (CUDA via native Windows driver)
- **Python:** 3.12 installed from python.org (added to PATH)
- **Node.js:** 20 LTS (for simulator)
- **Claude Code:** Native Windows install (runs via Git Bash internally)
- **Architecture:** x86_64
- **Audio capture:** WASAPI loopback (Windows-native) for system audio, or local files

### Production (Jetson Orin Nano)

- **OS:** JetPack (Ubuntu-based, includes CUDA + TensorRT)
- **GPU:** 128-core NVIDIA Ampere (native CUDA, TensorRT for optimized inference)
- **CPU:** 6-core ARM Cortex-A78AE
- **Architecture:** aarch64 (ARM)
- **Memory:** 8GB unified (CPU + GPU shared)
- **Power:** ~7-15W
- **Audio capture:** PulseAudio/PipeWire monitor source for system audio loopback

### Cross-Platform Considerations

Code is developed on Windows x86_64 and deployed on Linux aarch64 (Jetson). This means:

- **No OS-specific code in the core logic.** Use `pathlib.Path` instead of hardcoded path separators. Use `os.environ` for platform-specific config.
- **Audio capture is platform-specific.** The `scripts/capture_audio.py` utility abstracts this: WASAPI loopback on Windows, PulseAudio monitor on Jetson/Linux. The rest of the audio pipeline receives a numpy array regardless of source.
- **No x86-specific dependencies.** All Python packages must have ARM wheels or compile from source on Jetson.
- **PyTorch:** Use `torch` with CUDA support on both platforms. On Jetson, install via NVIDIA's JetPack PyTorch wheels (not pip default).
- **Demucs:** Runs on GPU in both environments. On Jetson, verify model fits in 8GB unified memory.
- **TensorRT (production optimization):** Genre/segment classifiers can be exported to ONNX → TensorRT for faster inference on Jetson. Not needed during development.
- **Test on both.** CI runs on x86. Before deployment, test on Jetson to catch ARM-specific and Linux-specific issues (audio drivers, numpy SIMD).

### GPU Availability

Both development and production environments have NVIDIA GPUs with CUDA. This means:
- **Real-time source separation (demucs) is feasible** — GPU demucs runs at ~1-2x real-time, fast enough for Mode C pre-processing and potentially Mode A live.
- **ML inference is GPU-accelerated** — Genre classification, drop prediction, and future neural models all run on GPU.
- **Do NOT write CPU-only fallback paths** unless specifically needed. Assume CUDA is always available.

### Git Workflow

- Feature branches + PRs (GitHub flow)
  - Branch naming: `feature/<short-description>`, `fix/<short-description>`, `docs/<short-description>`
  - PRs require passing CI (lint + tests) before merge
  - Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)

---

## Code Style & Conventions

### Python

- **Formatter/Linter:** ruff (configured in pyproject.toml)
- **Type hints:** Required on all function signatures (parameters + return types)
- **Docstrings:** Required on all public modules, classes, and functions. Use Google-style docstrings:
  ```python
  def detect_beats(audio: np.ndarray, sr: int = 44100) -> list[float]:
      """Detect beat positions in an audio signal.

      Args:
          audio: Mono audio signal as numpy array.
          sr: Sample rate in Hz.

      Returns:
          List of beat timestamps in seconds.
      """
  ```
- **Imports:** Group as stdlib → third-party → local, enforced by ruff
- **Naming:**
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
  - Private members prefixed with `_`
- **Line length:** 100 characters max
- **No `print()` for logging** — use the `logging` module
- **Async:** Use `asyncio` for all I/O-bound operations (network, audio capture). The main control loop runs as an async event loop.

### TypeScript (Simulator)

- **Strict mode** enabled in tsconfig.json
- **Formatting:** Prettier
- **Linting:** ESLint with recommended rules
- **Components:** Functional components with hooks (no class components)
- **Naming:** `PascalCase` for components, `camelCase` for functions/variables

### C (Firmware)

- **Style:** Linux kernel style (K&R braces, tabs for indentation)
- **Naming:** `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for macros/constants
- **Header guards:** `#pragma once`
- **Comments:** Doxygen-style for public API functions

---

## Testing Strategy (Phase 1)

Tests are required for **critical paths only** in Phase 1. Write tests for:

- Beat detection accuracy (compare against known-BPM test tracks)
- Genre classification correctness (test against labeled audio samples)
- UDP protocol encoding/decoding (roundtrip tests)
- Lighting engine output (given specific audio features, verify expected fixture commands)

Use `pytest` with fixtures defined in `tests/conftest.py`. Store test audio samples in `tests/fixtures/audio/`.

Not required in Phase 1: UI tests, firmware tests, integration tests across the full pipeline.

---

## Genre Lighting Profiles (Summary)

The AI uses 8 genre-specific lighting profiles. Full details in `docs/genre-lighting-profiles.md`.

| # | Profile | Reference Artists | Key Principle |
|---|---------|------------------|---------------|
| 1 | Rage / Experimental Trap | Playboi Carti, Travis Scott | Extreme contrast — BLINDING or DARK, nothing in between |
| 2 | Psychedelic Trap / Dark R&B | Don Toliver, The Weeknd | Smooth and flowing — transitions over bars, not beats |
| 3 | French Rap (Melodic) | Ninho, Jul | Warm and colorful — hi-hat bounce drives light rhythm |
| 4 | French Rap (Hard) | Kaaris | Regimented power — every hit is deliberate like a punch |
| 5 | European Alt Hip-Hop | AyVe, Exetra Archive | Artistic restraint — visual silence makes hits impactful |
| 6 | Theatrical Electronic | Stromae | Storytelling — lights follow emotional arc, not just beats |
| 7 | Festival EDM / Trance | Guetta, Armin, Edward Maya | Build-drop cycle — everything serves tension or release |
| 8 | UK Bass / Dubstep / Grime | Fred again.. | Underground rave — raw, DIY, imperfect by design |

### Profile Blending

Tracks often span multiple profiles. The classifier outputs weighted blends:
```
Sao Paulo (Weeknd): { psych_rnb: 0.6, festival_edm: 0.2, rage_trap: 0.1, theatrical: 0.1 }
Victory Lap 5 (Fred again..): { uk_bass: 0.7, rage_trap: 0.2, festival_edm: 0.1 }
```

### Two-Stage Genre Classification

1. **Family Classification** → Hip-Hop/Rap, Electronic, Hybrid (3 classes)
2. **Profile Classification** → Specific profile within family (8+ classes)

### Multilingual Vocal Detection

The music library includes French, German, Portuguese, and English vocals. All vocal detection must be language-agnostic — use signal-level onset detection on isolated vocal tracks (via demucs), not language-specific models.

---

## Hardware (For Reference)

Claude Code primarily builds software, but needs hardware context for protocol design and fixture abstraction.

### Budget & Scope

- **Budget:** $500-$1000 for the first working setup (fixtures + PoE switch + enclosures; Jetson already owned)
- **First deployment:** 8-12 fixtures in a ~5m × 7m × 2.5m basement/garage room
- **Phase 1 fixtures:** RGB Strobes + RGBW Pars + UV Bars only. No lasers, no moving heads.

### Room Layout (Default for Simulator)

```
Room: 5m × 7m × 2.5m ceiling (16ft × 23ft × 8ft)

Suggested Phase 1 fixture placement (8-12 units):
- 4× RGBW Par: ceiling corners, angled down 45°, wide beam wash
- 2-4× RGB Strobe: ceiling center-line, aimed at dance area
- 2-4× UV Bar: wall-mounted at 2m height, facing inward
```

### Fixture Types

All fixture types receive the same 7-byte `FixtureCommand` (R, G, B, W, strobe_rate, strobe_intensity, special). Each type interprets the bytes differently:

| Fixture | R,G,B,W | strobe_rate | strobe_intensity | special | Phase |
|---------|---------|-------------|-----------------|---------|-------|
| RGB Strobe | Base color (overridden by strobe white) | 0-255 (off to ~25Hz) | 0-255 flash brightness | Unused (0) | 1 |
| RGBW Par | Color wash output | Ignored (0) | Ignored (0) | Master dimmer (0-255) | 1 |
| UV Bar | Ignored | Ignored | Ignored | UV intensity (0-255) | 1 |
| Laser Module | Ignored | Ignored | Ignored | Pattern ID (0-255) | 2+ |
| Moving Head | Color output | Ignored | Ignored | Gobo ID; pan/tilt via separate extended packet | 3+ |

### MCU: STM32F4 + W5500 Ethernet

**Primary MCU:** STM32F4 (e.g., STM32F407 or STM32F446) + W5500 SPI Ethernet module.

Chosen for superior hardware timer precision (critical for jitter-free PWM on high-power LEDs), deterministic interrupt handling, and no unnecessary WiFi/BT overhead. The W5500 offloads TCP/IP to hardware, keeping the MCU focused on LED control.

**Firmware Development Workflow:**
1. **STM32CubeMX** — Configure pin assignments, clock tree, timer peripherals, SPI (for W5500), and PWM channels. Generates HAL initialization boilerplate and CMake project structure.
2. **Claude Code** — Writes all application-level firmware on top of the CubeMX-generated code: UDP command parser, LED controller, strobe engine, color mixing, watchdog, discovery protocol.
3. **STM32CubeIDE** — Used only for hardware debugging (step-through, breakpoints, flash programming). Not used as a code editor.

**Future wireless option:** ESP32-S3 may be used for battery-powered wireless accent fixtures (Phase 3+). The UDP protocol is MCU-agnostic, so ESP32 fixtures would receive the same commands.

The fixture control protocol and software abstractions in the Python host are MCU-agnostic. Do not write MCU-specific code in the Python host — the protocol layer handles all abstraction.

### PoE Power Budget

- 802.3af: 13W per port (sufficient for RGBW pars, UV bars, lasers)
- 802.3at: 25W per port (required for 100W+ strobes)
- DC-DC: 48V PoE → 12V (LEDs) + 5V (MCU logic) + 3.3V (MCU core)

---

## Key Architectural Decisions

1. **Rule-based expert system first, neural model later.** Phase 1-2 use genre profiles + ML classifiers. Phase 4 trains end-to-end models on data generated by the rule system.
2. **GPU-first.** Both dev (RTX 4070) and production (Jetson Orin Nano) have NVIDIA CUDA GPUs. Use GPU for all ML inference, source separation (demucs), and future neural models. No CPU-only fallback paths needed.
3. **UDP, not DMX/ArtNet.** Custom lightweight UDP protocol for minimum latency. ArtNet compatibility can be added later as a bridge.
4. **60fps target refresh rate.** Fixture commands sent at 60fps (16.7ms interval). Audio analysis must keep up.
5. **Async everything.** The main control loop is an asyncio event loop. Audio capture, analysis, command generation, and network I/O all run as async tasks or in thread/process pools.
6. **Simulator mirrors production.** The 3D simulator uses the exact same fixture control protocol as physical hardware. Code that works in the simulator works on real fixtures with zero changes to the control layer.
7. **Extensible genre profiles.** Adding a new genre requires only: (a) a new profile class inheriting from `BaseProfile`, (b) labeled training examples, (c) optional simulator reference recordings. No architectural changes.
8. **Cross-platform: Windows → Linux/ARM.** Code develops on Windows x86_64 and deploys to Linux aarch64 (Jetson). Use `pathlib.Path` for all file paths (no hardcoded `/` or `\`). Avoid x86-specific or Windows-specific packages in the core logic. Audio capture is platform-specific and abstracted behind a common interface.

---

## What NOT To Do

- **Do not use DMX-specific libraries** (like python-dmx or OLA) — we have a custom protocol.
- **Do not hardcode genre behaviors** outside of profile classes — all genre logic lives in `lumina/lighting/profiles/`.
- **Do not use blocking I/O** in the main event loop — all network and audio operations must be async or offloaded to thread pools.
- **Do not train ML models in Phase 1** — use pre-trained models (madmom for beats, essentia for features) and rule-based logic for lighting decisions.
- **Do not assume English-only audio** — vocal detection must work across French, German, Portuguese, and English.
- **Do not build the DJ engine (Mode B) until Phase 4** — focus on Mode A (live) and Mode C (queue pre-processing) first.
- **Do not write firmware** — that is the hardware owner's domain. Claude Code writes the host-side protocol and provides firmware specs.
- **Do not write CPU-only fallback paths** — CUDA GPU is guaranteed on both dev (RTX 4070) and production (Jetson Orin Nano).
- **Do not use x86-only or Windows-only dependencies** in core logic — production runs on Linux aarch64 (Jetson). Verify ARM wheel availability for any new package.
- **Do not hardcode file paths with `\` or `/`** — use `pathlib.Path` for all file operations. Development is Windows, production is Linux.
- **Do not use `os.system()` or shell-specific commands** — use `subprocess.run()` or Python-native equivalents for cross-platform compatibility.

---

## Current Phase: Phase 1 — Foundation

### Phase 1 Software Goals (Weeks 1-4)
- [ ] Audio analysis pipeline: beat detection, energy tracking, onset detection
- [ ] Segment classifier: verse/chorus/drop/breakdown detection
- [ ] Genre classifier: two-stage family → profile classification
- [ ] Basic 3D simulator: room model, virtual fixtures, audio playback
- [ ] Fixture control protocol: UDP packet format design + reference implementation
- [ ] WebSocket bridge between Python backend and simulator frontend

### Phase 1 Hardware Goals (Parallel — Owner handles)
- [ ] Fixture PCB schematic (PoE PD + DC-DC + MCU + LED driver)
- [ ] Part selection and datasheet review
- [ ] LED and laser module selection and testing

---

## Reference Documents

- `docs/project-plan.md` — Full phased development plan
- `docs/genre-lighting-profiles.md` — Detailed lighting language for all 8 profiles
- `docs/protocol-spec.md` — UDP fixture control protocol specification
- `docs/architecture/audio-pipeline.md` — Audio analysis pipeline design
- `docs/architecture/ai-model.md` — AI model architecture (classification + rule engine)
- `docs/architecture/simulator.md` — 3D simulator design

---

## Environment Setup

### Prerequisites (Windows)

Install these first:
1. **Git for Windows** → https://git-scm.com/download/win (required by Claude Code)
2. **Python 3.12** → https://www.python.org/downloads/ (check "Add to PATH" during install)
3. **Node.js 20 LTS** → https://nodejs.org/ (for simulator)
4. **NVIDIA GPU driver** → Latest Game Ready or Studio driver (you already have this for the RTX 4070)
5. **Claude Code** → In PowerShell: `irm https://claude.ai/install.ps1 | iex`

### Development (Windows — PowerShell)

```powershell
# Clone repo
git clone <repo-url>
cd lumina

# Python virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA GeForce RTX 4070

# Simulator
cd simulator
npm install
npm run dev  # Starts Vite dev server

# Run tests (from repo root)
cd ..
pytest tests/

# Lint
ruff check lumina/
ruff format lumina/
mypy lumina/

# Launch Claude Code
claude
```

### Production (Jetson Orin Nano — Linux)

```bash
# Jetson comes with JetPack — Python 3.12 may need manual install
# Use NVIDIA's PyTorch wheels for Jetson (not pip default)
# See: https://developer.nvidia.com/embedded/downloads

# Clone and install
git clone <repo-url>
cd lumina
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Verify CUDA on Jetson
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: Orin (nvgpu)

# Run production mode
python -m lumina.app --mode live
```

