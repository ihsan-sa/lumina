# LUMINA Technical Documentation & Agent Context

Last updated: 2026-03-20

This is the single source of truth for agent-to-agent context transfer. Read this before
modifying any part of the codebase. For design philosophy and project-level rules, see `CLAUDE.md`.

---

## 1. Project Overview

LUMINA is an AI-powered light show system for a basement/garage venue (5m x 7m x 2.5m). It analyzes
music in real time and generates lighting choreography that understands song structure, genre
conventions, and emotional arc -- not just beats.

**Architecture**: Three-layer monolith (Python backend + React/Three.js simulator + future STM32
firmware). The backend analyzes audio, generates per-fixture commands at 60fps, and streams them
to connected clients (3D simulator or physical fixtures) via WebSocket and UDP.

**Current Phase**: Phase 1 -- Foundation. Rule-based lighting profiles with ML pipeline scaffolded.
All three operating modes work.

**Operating Modes**:
- `file` -- Offline analysis of an audio file, then playback at 60fps.
- `showcase` -- Server starts immediately with no audio for pattern showcase testing.
- `live` -- Real-time system audio capture via sounddevice, streaming analysis, live lighting.

**Key Principle**: This is NOT a beat-sync system. Each genre profile encodes a complete lighting
philosophy. A Carti mosh-pit track and a Stromae theatrical ballad produce completely different
lighting from the same audio features.

---

## 2. How to Run

### Prerequisites (Windows dev)

- Python 3.11+ (3.12 recommended)
- Node.js 20 LTS
- NVIDIA GPU with CUDA
- Git

### Setup

```bash
# Clone and install Python deps
cd C:\dev\lumina
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Simulator
cd simulator
npm install
```

### Running in File Mode (main workflow)

```bash
# Analyze a song and stream to simulator at 60fps
python -m lumina.app --mode file --file path/to/song.mp3

# With debug output (MusicState + lighting debug once per second)
python -m lumina.app --mode file --file song.mp3 --debug

# Force a specific genre profile
python -m lumina.app --mode file --file song.mp3 --genre rage_trap

# Also send UDP to physical fixtures
python -m lumina.app --mode file --file song.mp3 --udp-target 192.168.1.100:5150
```

**CLI flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `file` | `file`, `live`, or `showcase` |
| `--file` | None | Path to audio file (required for file mode) |
| `--host` | `0.0.0.0` | WebSocket server bind address |
| `--port` | `8765` | WebSocket server port |
| `--fps` | `60` | Output frame rate |
| `--sr` | `44100` | Audio sample rate |
| `--udp-target` | None | `IP:PORT` for physical fixture UDP output |
| `--debug` | false | Print MusicState + lighting debug once per second |
| `--genre` | None | Override genre classification with fixed profile |

### Running the Simulator

```bash
cd simulator
npm run dev
# Opens at http://localhost:5173
# Connects to backend WebSocket at ws://localhost:8765/ws
```

The simulator auto-connects, receives fixture layout, loads audio from `/audio` endpoint,
and syncs playback. Transport controls (play/pause/seek) work from the simulator UI.

### Running Tests

```bash
# All tests
pytest tests/

# Skip slow tests (audio file processing)
pytest tests/ -m "not slow"

# Specific test file
pytest tests/test_lighting_engine.py -v

# Lint
ruff check lumina/
ruff format lumina/
mypy lumina/
```

### ML Training Pipeline

```bash
# Install ML training deps
pip install -e ".[dev,ml_training]"

# Train (data must exist in data/features/aligned/*.parquet)
python -m lumina.ml.model.train

# Resume after stopping
python -m lumina.ml.model.train --resume

# Full options
python -m lumina.ml.model.train --resume --epochs 100 --batch-size 64 --lr 3e-4 --wandb
```

---

## 3. Data Flow -- End-to-End Pipeline

### File Mode Pipeline (`lumina/app.py` `_run_file_mode()`)

```
1. librosa.load(file)                              -> audio (float32 mono)
2. SourceSeparator.separate(audio, sr)              -> StemSet (drums, bass, vocals, other)
3. BeatDetector.analyze_offline(stems.drums)        -> list[BeatInfo]
4. OnsetDetector.analyze_offline(stems.drums)       -> list[OnsetEvent | None]
5. VocalDetector.analyze_offline(stems.vocals)      -> list[VocalFrame]
6. EnergyTracker.analyze_offline_with_bass_stem()   -> list[EnergyFrame]
7. DropPredictor.process_features(...)              -> list[DropFrame]
8. GenreClassifier.classify_file(...)               -> list[GenreFrame]   (locked per track)
9. StructuralAnalyzer.analyze(...)                  -> StructuralMap      (uses genre_family)
   -> .map_to_frames()                              -> list[SegmentFrame]
10. LayerTracker.analyze(stems)                     -> list[LayerFrame]   (resampled to fps)
11. MotifDetector.detect_macro_motifs(audio, beats) -> MotifTimeline
12. MotifDetector.detect_micro_patterns(stems.other)-> list[NotePattern]
13. ArcPlanner.plan(energy, layers, structure)      -> list[ArcFrame]
14. SongScore.build(layers, notes, arc, motifs)     -> list[ScoreFrame]
15. _assemble_music_state(all frame data)           -> list[MusicState]
```

Steps 3-6 run in parallel (different stem inputs). Steps 10-14 are the "extended analysis"
layer added after the core audio pipeline. Step 8 (genre) must run before step 9 (structure)
because electronic tracks use the EDM structural pass.

### Frame Output Loop

```
MusicState[frame_i]
  -> LightingEngine.generate(state)
     -> _select_profile(state.genre_weights) -> profile (e.g., RageTrapProfile)
     -> profile.generate(state)              -> list[FixtureCommand]
  -> app._apply_effects(commands)            -> list[FixtureCommand] (intensity/manual)
  -> server.state_queue.put(state, commands)
  -> [optional] UDP socket.sendto(encode_packet(commands))
```

### WebSocket Broadcast

```
server._broadcast_loop():
  state, commands = await state_queue.get()
  -> serialize fixture_commands message (JSON)
  -> serialize music_state message (JSON)
  -> send both to all connected clients
```

---

## 4. Core Data Contracts

### MusicState (22 fields)

Defined in `lumina/audio/models.py`. Produced at 60fps, consumed by lighting engine.

```python
@dataclass(slots=True)
class MusicState:
    # Core audio features (15 fields)
    timestamp: float = 0.0              # Current time in seconds
    bpm: float = 120.0                  # Current tempo estimate
    beat_phase: float = 0.0             # 0.0-1.0 position within current beat
    bar_phase: float = 0.0             # 0.0-1.0 position within current bar
    is_beat: bool = False               # True on exact beat frames
    is_downbeat: bool = False           # True on bar downbeats
    energy: float = 0.0                 # 0.0-1.0 overall energy level
    energy_derivative: float = 0.0      # Rising (+) or falling (-) energy
    segment: str = "verse"              # Section label
    genre_weights: dict[str, float]     # Profile -> weight (sum to 1.0)
    vocal_energy: float = 0.0           # 0.0-1.0 vocal presence
    spectral_centroid: float = 0.0      # Brightness indicator (Hz)
    sub_bass_energy: float = 0.0        # Sub-bass (20-80Hz) energy
    onset_type: str | None = None       # "kick", "snare", "hihat", "clap", or None
    drop_probability: float = 0.0       # 0.0-1.0 probability of drop in next 1-4 bars

    # Extended analysis (7 fields, defaults for backward compat)
    layer_count: int = 0                # Active stems (0-4)
    layer_mask: dict[str, float]        # Per-stem activity {"drums": 0.8, "bass": 0.3, ...}
    motif_id: int | None = None         # Which macro motif is playing (None if none)
    motif_repetition: int = 0           # How many times this motif has been heard
    notes_per_beat: int = 0             # Regular notes per beat (0 = no pattern)
    note_pattern_phase: float = 0.0     # Position in note cycle (0.0-1.0)
    headroom: float = 1.0              # Intensity budget for this moment (0.0-1.0)
```

**Segment labels (non-EDM)**: `verse`, `chorus`, `drop`, `breakdown`, `bridge`, `intro`, `outro`

**Segment labels (EDM)**: `build`, `drop`, `groove`, `breakdown`, `intro`, `outro`

### FixtureCommand (8 bytes)

Defined in `lumina/control/protocol.py`.

```python
@dataclass(slots=True)
class FixtureCommand:
    fixture_id: int = 0          # 1-255 unicast, 0 = broadcast
    red: int = 0                 # 0-255
    green: int = 0               # 0-255
    blue: int = 0                # 0-255
    white: int = 0               # 0-255
    strobe_rate: int = 0         # 0 = off, 255 = max ~25Hz
    strobe_intensity: int = 0    # 0-255
    special: int = 0             # Fixture-type-specific
```

**Channel interpretation by fixture type**:

| Fixture Type | R,G,B,W | strobe_rate | strobe_intensity | special |
|---|---|---|---|---|
| PAR | Color wash | Ignored | Ignored | Master dimmer 0-255 |
| STROBE | Strobe tint color | Flash rate | Flash brightness | Unused |
| LED_BAR | Color wash | Ignored | Ignored | Master dimmer 0-255 |
| LASER | Ignored | Ignored | Ignored | Pattern ID 0-255 |
| UV | Ignored | Ignored | Ignored | UV intensity 0-255 |

### ScoreFrame

Defined in `lumina/analysis/song_score.py`. Aggregates layer/motif/arc data per frame.

```python
@dataclass(slots=True)
class ScoreFrame:
    layer_count: int                  # Active stems 0-4
    layer_mask: dict[str, float]      # Per-stem activity
    motif_id: int | None              # Active motif (None if none)
    motif_repetition: int             # Repetition count of this motif
    notes_per_beat: int               # Regular note pattern (0 = none)
    note_pattern_phase: float         # Position in note cycle 0.0-1.0
    headroom: float                   # Intensity budget 0.0-1.0
```

### Color (float RGBW)

Defined in `lumina/lighting/profiles/base.py`. Used internally by profiles and patterns.

```python
@dataclass(slots=True)
class Color:
    r: float = 0.0    # 0.0-1.0
    g: float = 0.0
    b: float = 0.0
    w: float = 0.0
```

Constants: `BLACK = Color(0,0,0,0)`, `WHITE = Color(1,1,1,1)`, `RED = Color(1,0,0,0)`

---

## 5. Audio Pipeline

All audio modules live in `lumina/audio/`. Each produces per-frame output at the configured fps
(default 60).

| Module | File | Input | Output | Key Details |
|---|---|---|---|---|
| SourceSeparator | `source_separator.py` | audio, sr | `StemSet` (drums, bass, vocals, other) | Demucs htdemucs model, GPU, cached |
| BeatDetector | `beat_detector.py` | stems.drums | `list[BeatInfo]` | madmom RNN + DBN tracking, offline mode |
| OnsetDetector | `onset_detector.py` | stems.drums | `list[OnsetEvent \| None]` | Transient detection (kick, snare, hihat, clap) |
| EnergyTracker | `energy_tracker.py` | audio + stems.bass | `list[EnergyFrame]` | RMS energy, derivative, spectral centroid, sub-bass |
| VocalDetector | `vocal_detector.py` | stems.vocals | `list[VocalFrame]` | Language-agnostic vocal presence (signal-level) |
| DropPredictor | `drop_predictor.py` | frame features | `list[DropFrame]` | Rule-based look-ahead drop probability |
| GenreClassifier | `genre_classifier.py` | frame features + stems | `list[GenreFrame]` | Two-stage: family -> profile (rule-based) |
| StructuralAnalyzer | `structural_analyzer.py` | audio + all results | `StructuralMap` | MFCC self-similarity + EDM pass |
| SegmentClassifier | `segment_classifier.py` | frame features | `SegmentFrame` | Frame-by-frame (used in live mode) |

### Source Separation (Demucs)

`StemSet` dataclass holds 4 mono float32 stems:
- `drums` -- used by BeatDetector, OnsetDetector
- `bass` -- used by EnergyTracker for sub-bass
- `vocals` -- used by VocalDetector
- `other` -- used by MotifDetector for note patterns

Falls back to passthrough (full mix as all stems) if demucs import fails.

### Genre Classification (Two-Stage)

`lumina/audio/genre_classifier.py`

**Stage 1 -- Family** (3 classes): `hiphop_rap`, `electronic`, `hybrid`

**Stage 2 -- Profile** (8 profiles): `rage_trap`, `psych_rnb`, `french_melodic`, `french_hard`,
`euro_alt`, `theatrical`, `festival_edm`, `uk_bass`

Rule-based feature prototypes. Each profile has a characteristic feature signature. Classifier
computes soft similarity scores and normalizes to probability distribution. Output is locked
for the entire track (no per-frame changes).

Accepts `genre_override` parameter to bypass classification with a fixed profile.

### Structural Analysis

`lumina/audio/structural_analyzer.py`

**Default path** (non-EDM): MFCC recurrence matrix -> checkerboard novelty -> peak-pick boundaries
-> feature extraction -> clustering -> labeling via energy contrast.

**EDM path** (when `genre_family` is "electronic" or "hybrid"): `lumina/analysis/edm_structure.py`
`edm_structure_pass()` -- uses energy envelope and drop predictions for build/drop/groove/breakdown
labels.

Minimum section duration: 4.0 seconds. Sections are merged if shorter.

### Extended Analysis

| Module | File | Purpose |
|---|---|---|
| LayerTracker | `lumina/analysis/layer_tracker.py` | Count active stems per frame (0-4) from RMS envelopes |
| MotifDetector | `lumina/analysis/motif_detector.py` | Bar-level chroma+MFCC similarity + note-level IOI regularity |
| ArcPlanner | `lumina/analysis/arc_planner.py` | Headroom budgeting: significance = energy * sqrt(layers) |
| SongScore | `lumina/analysis/song_score.py` | Aggregates layers/motifs/arc into ScoreFrame, assigns motif patterns |

**LayerTracker**: RMS envelope per stem with hop=512. Stem is "active" if RMS > 15% of its peak.
EMA smoothing with alpha=0.05.

**MotifDetector**: Macro motifs use cosine similarity threshold 0.82 on bar-level chroma+MFCC
features. Micro patterns detect regular note sequences via spectral flux IOI regularity
(coefficient of variation < 0.30).

**ArcPlanner**: Per-section significance = mean_energy * sqrt(mean_layers). Percentile-ranked
across song, mapped to [0.15, 1.0] headroom. 1.5s transition smoothing between sections.

---

## 6. Lighting Engine

### Profile Dispatch (`lumina/lighting/engine.py`)

```
LightingEngine.generate(state):
  1. If pattern_override is set: run named pattern on all fixtures, bypass profiles
  2. Update LightingContext (segment tracking, bars_in_section, recent_max_intensity)
  3. _select_profile(state.genre_weights):
     a. If genre_override set (from UI): use that profile
     b. Find highest-weighted registered profile from genre_weights
     c. If weight >= 0.3: use it
     d. Otherwise: use "generic" fallback
  4. profile.generate(state) -> list[FixtureCommand]
  5. Build debug info (active fixtures, patterns used, dominant colors)
```

**Minimum genre weight**: 0.3 (constant `_MIN_GENRE_WEIGHT`)

### Profile Registration (`_PROFILE_REGISTRY`)

```python
{
    "rage_trap": RageTrapProfile,
    "psych_rnb": PsychRnbProfile,
    "festival_edm": FestivalEdmProfile,
    "french_melodic": FrenchMelodicProfile,
    "french_hard": FrenchHardProfile,
    "euro_alt": EuroAltProfile,
    "theatrical": TheatricalProfile,
    "uk_bass": UkBassProfile,
    "generic": GenericProfile,
}
```

### LightingContext (cross-frame state)

```python
@dataclass
class LightingContext:
    recent_patterns: deque[str]           # Last 16 patterns (contrast tracking)
    motif_visual_map: dict[int, str]      # motif_id -> pattern name
    motif_color_map: dict[int, int]       # motif_id -> color index
    bars_in_section: int                  # Bars since last segment change
    recent_max_intensity: float           # Decaying max (0.99 per frame)
    last_segment: str                     # Previous segment label
```

### Pattern System (`lumina/lighting/patterns.py`)

All patterns have the same signature:
```python
def pattern_fn(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    **kwargs,
) -> dict[int, FixtureCommand]
```

Patterns return partial command dicts (fixture_id -> command). Profiles merge multiple pattern
outputs with `_merge_commands()`.

### Fixture Command Routing (`route_command()`)

`lumina/lighting/profiles/base.py:route_command()` and `BaseProfile._cmd()` route channels
based on fixture type:
- **PAR**: RGBW color, `special` = master dimmer (auto-derived from intensity if not set)
- **STROBE**: RGBW color tint, `strobe_rate` and `strobe_intensity` control flash
- **LED_BAR**: Same as PAR (RGBW + dimmer)
- **LASER**: All RGBW/strobe zeroed, `special` = pattern ID
- **UV**: All RGBW zeroed, `special` = UV intensity

### BaseProfile Utilities

Key methods available to all profiles:

| Method | Purpose |
|---|---|
| `_cmd(fixture, color, intensity)` | Build routed FixtureCommand |
| `_blackout()` | All fixtures to black |
| `_all_color(color, intensity)` | Set all pars/bars to one color |
| `_chase(fixtures, phase, color, width)` | Chase pattern through fixture list |
| `_sweep_x(phase, color, width)` | Left-to-right sweep by x-position |
| `_sweep_y(phase, color, width)` | Front-to-back sweep by y-position |
| `_alternating(fixtures, a, b, phase)` | Even/odd color swap |
| `_focus_expand(phase, color)` | Center-out expansion |
| `_corner_isolation(corner_role, color)` | Light one corner only |
| `_merge_commands(*sources, base)` | Merge partial dicts into full list |
| `_strobe_on_beat(state, rate, intensity)` | Beat-synced strobe params |
| `_color_temperature(centroid, warm, cool)` | Lerp by spectral centroid |
| `_bass_saturate(sub_bass, color, boost)` | Deepen saturation from bass |
| `_layer_fixture_count(layers, energy, total)` | Map layers to fixture count |
| `_apply_note_pattern(state, fixtures, color)` | Cycle fixtures per note |

### BumpTracker (onset decay)

`lumina/lighting/profiles/base.py:BumpTracker`

Exponential decay after onset triggers. Each profile has its own decay rate.
Groups (e.g., "pars", "strobes") tracked independently.

```python
bump.trigger("pars", timestamp)
intensity = bump.get_intensity("pars", timestamp, peak=1.0, floor=0.0)
# Returns exp(-decay_rate * dt) from last trigger
```

---

## 7. All 9 Profiles

| # | Profile | File | Decay Rate | Philosophy | Key Colors | Segment Behavior |
|---|---|---|---|---|---|---|
| 1 | `rage_trap` | `rage_trap.py` | 20.0 | Extreme contrast -- BLINDING or DARK | Red, white only | Verse: dark red wash. Drop: strobe_burst + blackout. Build: stutter accelerating |
| 2 | `psych_rnb` | `psych_rnb.py` | 3.5 | Smooth and flowing -- transitions over bars | Purple, cyan, magenta, hot pink | Verse: breathe with phase offset. Chorus: alternate + color_split. No strobes ever |
| 3 | `french_melodic` | `french_melodic.py` | 12.0 | Warm palette, hi-hat-driven bounce | Gold, amber, sunset orange, coral | Verse: chase_bounce + flicker on bars. Chorus: chase_mirror + color_pop |
| 4 | `french_hard` | `french_hard.py` | 25.0 | Regimented symmetry, cold, deliberate | Ice white, steel blue | Verse: chase_mirror strict. Chorus: alternate (white vs black). Drop: blinder entry |
| 5 | `euro_alt` | `euro_alt.py` | 5.0 | Visual silence, gallery restraint | White only (warm/cool tones) | Verse: single spotlight. Chorus: 2-3 pars gradient. NO strobes ever |
| 6 | `theatrical` | `theatrical.py` | 3.0 | Storytelling -- vocal energy drives intensity | Per-segment palette changes | Verse: vocal-energy wash. Chorus: color_split gold. Drop: slow diverge bloom |
| 7 | `festival_edm` | `festival_edm.py` | 7.0 | Build-drop cycle -- tension or release | Blue->cyan->white builds, rainbow drops | Build: chase_lr accelerating + stutter. Drop: diverge + rainbow_roll + strobe |
| 8 | `uk_bass` | `uk_bass.py` | 8.0 | Underground rave -- raw, DIY, imperfect | Sodium amber, deep green, dirty pink | Verse: flicker. Build: stutter + converge. Drop: strobe_burst + random_scatter |
| 9 | `generic` | `generic.py` | 8.0 | Never ugly, never boring, never extreme | Blue/purple verse, warm chorus | Safe fallback. Moderate everything. Fixture count escalation by energy |

### Profile Override Properties

Each profile can override these for motif assignment:

- `motif_pattern_preferences` -- ordered list of pattern names (default: `["chase_lr", "alternate", "breathe", "converge"]`)
- `motif_color_palette` -- list of Colors to cycle through for motifs

### Profile Blending (`lumina/lighting/blender.py`)

When a track spans multiple genres, the `ProfileBlender` runs all profiles with weight >= 0.1
and blends their outputs proportionally:

- **RGBW channels**: Weighted average across all active profiles
- **Strobe**: Uses rate/intensity from the highest-weighted profile with strobe active (no blending)
- **Special byte**: Weighted average, clamped 0-255

Single active profile -> delegates directly (zero overhead). The engine enables blending
automatically when `len(genre_weights) > 1` and at least 2 profiles have weight >= 0.1.

### Cross-Genre Transitions (`lumina/lighting/transitions.py`)

When the dominant profile changes, `TransitionEngine` cross-fades between old and new outputs
over segment-aware durations:

| Segment | Duration | Rationale |
|---|---|---|
| `drop` | 0.1s | Drops should snap immediately |
| `breakdown` | 3.0s | Slow dissolve for atmospheric sections |
| `chorus` | 1.5s | Moderate blend for energy shifts |
| default | 2.0s | Safe default for other transitions |

Three easing curves: `linear`, `ease_in_out` (cubic), `crossfade` (sqrt / equal-power).
Strobe channels hard-switch at the midpoint rather than blending (blended strobe rates
look wrong visually).

---

## 8. 20 Patterns

All defined in `lumina/lighting/patterns.py` and registered in `PATTERN_REGISTRY`.

| Pattern | Description |
|---|---|
| `chase_lr` | Sequential left-to-right sweep, speed tied to bar_phase |
| `chase_bounce` | Ping-pong chase: L->R->L using triangle wave |
| `converge` | Outside-in: edge fixtures fire first, center follows |
| `diverge` | Center-out: center fixtures first, edges follow (bloom) |
| `alternate` | Even/odd fixtures swap colors on beat_phase |
| `random_scatter` | Deterministic pseudo-random scatter (hash-based, ~50ms quantized) |
| `breathe` | Sine-wave breathing, all fixtures rise and fall together |
| `strobe_burst` | All fixtures max strobe -- the nuclear option, use sparingly |
| `wash_hold` | Static color wash with gentle +-5% intensity drift |
| `color_split` | Left/right half split, complementary colors by x-position |
| `spotlight_isolate` | One fixture bright, all others dim/off |
| `stutter` | Rapid on/off at musical subdivisions (2/4/8 per beat) |
| `chase_mirror` | Symmetric L-R chase: left sweeps L->R, right sweeps R->L |
| `strobe_chase` | Sequential strobe rotation: one fixture at a time cycling |
| `lightning_flash` | Multi-flash with exponential aftershock decay (100%->60%->30%->0) |
| `color_pop` | Complementary color flash on beats against base wash |
| `rainbow_roll` | Each fixture gets different hue, rotating continuously |
| `flicker` | Per-fixture pseudo-random intensity jitter (fire/underground) |
| `gradient_y` | Front-to-back color gradient using fixture y-position |
| `blinder` | All fixtures max white + max strobe -- audience blinder |

Helper function `select_active_fixtures()` selects subsets based on energy level (low: 3,
mid: 8, high: all).

---

## 9. Fixture Layout

Default: 15 fixtures in `lumina/lighting/fixture_map.py`. Room: 5.0m (W) x 7.0m (D) x 2.5m (H).

| ID | Type | Name | Position (x,y,z) | Role | Groups |
|---|---|---|---|---|---|
| 1 | PAR | Par L1 | (0.0, 1.4, 2.0) | LEFT | par_left, par_all, left |
| 2 | PAR | Par L2 | (0.0, 2.8, 2.1) | LEFT | par_left, par_all, left |
| 3 | PAR | Par L3 | (0.0, 4.2, 2.2) | LEFT | par_left, par_all, left |
| 4 | PAR | Par L4 | (0.0, 5.6, 2.3) | LEFT | par_left, par_all, left |
| 5 | PAR | Par R1 | (5.0, 1.4, 2.0) | RIGHT | par_right, par_all, right |
| 6 | PAR | Par R2 | (5.0, 2.8, 2.1) | RIGHT | par_right, par_all, right |
| 7 | PAR | Par R3 | (5.0, 4.2, 2.2) | RIGHT | par_right, par_all, right |
| 8 | PAR | Par R4 | (5.0, 5.6, 2.3) | RIGHT | par_right, par_all, right |
| 9 | STROBE | Strobe FL | (0.3, 0.3, 2.4) | FRONT_LEFT | strobe_corners, strobe_left |
| 10 | STROBE | Strobe FR | (4.7, 0.3, 2.4) | FRONT_RIGHT | strobe_corners, strobe_right |
| 11 | STROBE | Strobe BL | (0.3, 6.7, 2.4) | BACK_LEFT | strobe_corners, strobe_left |
| 12 | STROBE | Strobe BR | (4.7, 6.7, 2.4) | BACK_RIGHT | strobe_corners, strobe_right |
| 13 | LED_BAR | Bar Front | (2.5, 2.33, 2.5) | CENTER | overhead, center |
| 14 | LED_BAR | Bar Rear | (2.5, 4.67, 2.5) | CENTER | overhead, center |
| 15 | LASER | Laser | (2.5, 7.0, 2.4) | BACK | laser, back |

**Summary**: 8 pars (4 left wall, 4 right wall), 4 corner strobes, 2 overhead LED bars, 1 rear laser.

**Spatial queries**: `FixtureMap` provides `by_type()`, `by_role()`, `by_group()`, `sorted_by_x()`,
`sorted_by_y()`, `left_side()`, `right_side()`, `front_half()`, `back_half()`, `get_left()`,
`get_right()`, `get_by_spatial_order()`.

---

## 10. WebSocket Protocol

Backend runs a Starlette ASGI app (`lumina/web/server.py`) with WebSocket at `/ws`,
HTTP health at `/health`, and audio file serving at `/audio`.

### Server -> Client Messages

**`fixture_layout`** (sent on connect):
```json
{
  "type": "fixture_layout",
  "fixtures": [
    {
      "fixture_id": 1,
      "fixture_type": "par",
      "position": [0.0, 1.4, 2.0],
      "role": "left",
      "name": "Par L1"
    }
  ]
}
```

**`playback_start`** (sent on connect if audio loaded):
```json
{
  "type": "playback_start",
  "filename": "song.mp3",
  "duration": 210.5,
  "audio_url": "/audio",
  "start_timestamp": 45.2
}
```

**`fixture_commands`** (60fps):
```json
{
  "type": "fixture_commands",
  "sequence": 12345,
  "timestamp_ms": 45200,
  "commands": [
    {"fixture_id": 1, "red": 255, "green": 0, "blue": 128, "white": 0,
     "strobe_rate": 0, "strobe_intensity": 0, "special": 200}
  ]
}
```

**`music_state`** (60fps):
```json
{
  "type": "music_state",
  "state": {
    "timestamp": 45.2,
    "bpm": 128.0,
    "beat_phase": 0.3,
    "bar_phase": 0.8,
    "is_beat": false,
    "is_downbeat": false,
    "energy": 0.72,
    "energy_derivative": 0.05,
    "segment": "drop",
    "genre_weights": {"festival_edm": 0.7, "uk_bass": 0.2},
    "vocal_energy": 0.1,
    "spectral_centroid": 3500.0,
    "sub_bass_energy": 0.6,
    "onset_type": "kick",
    "drop_probability": 0.1,
    "layer_count": 4,
    "layer_mask": {"drums": 0.9, "bass": 0.8, "vocals": 0.3, "other": 0.7},
    "motif_id": 2,
    "motif_repetition": 3,
    "notes_per_beat": 4,
    "note_pattern_phase": 0.75,
    "headroom": 0.95
  }
}
```

### Client -> Server Messages

| Type | Fields | Handler in `app.py _handle_transport()` |
|---|---|---|
| `transport` | `action`: "play"/"pause"/"seek", `position`: float | Play/pause/seek playback |
| `pattern_override` | `pattern`: string or null | Force pattern on all fixtures |
| `genre_override` | `profile`: string or null | Force genre profile (null = auto) |
| `intensity` | `value`: 0-100 | Set global intensity multiplier |
| `manual_effect` | `effect`: "blackout"/"strobe_burst"/"uv_flash" | Trigger 0.5s manual effect |
| `audio_loaded` | `filename`, `duration` | Acknowledge audio load (log only) |

### Audio Sync

Simulator auto-loads audio from `/audio` endpoint when `playback_start` is received. Drift
correction runs every 2 seconds (tolerance 0.3s). Late-joining clients receive current
`start_timestamp` to seek before playing.

---

## 11. UDP Fixture Protocol

Defined in `lumina/control/protocol.py`.

### Constants

| Constant | Value |
|---|---|
| Magic | `0x4C55` ("LU") |
| Version | `1` |
| Port | `5150` |
| Max fixtures per packet | `32` |
| Max packet size | 265 bytes (9 header + 32 x 8 payload) |

### Packet Format (little-endian)

```
Header (9 bytes):
  magic:         uint16  (0x4C55)
  version:       uint8   (1)
  packet_type:   uint8   (0x01=COMMAND, 0x10=DISCOVER_REQ, 0x11=DISCOVER_RESP, 0x20=HEARTBEAT, 0x30=CONFIG)
  sequence:      uint16  (wraps at 65535)
  timestamp_ms:  uint16  (wraps at 65535)
  fixture_count: uint8   (0-32)

Payload (fixture_count x 8 bytes each):
  fixture_id:       uint8
  red:              uint8
  green:            uint8
  blue:             uint8
  white:            uint8
  strobe_rate:      uint8
  strobe_intensity: uint8
  special:          uint8
```

### API

```python
from lumina.control.protocol import encode_packet, decode_packet, FixtureCommand, PacketType

# Encode
packet = encode_packet(commands, sequence=seq, timestamp_ms=ts)

# Decode
packet_type, sequence, timestamp_ms, commands = decode_packet(raw_bytes)
```

---

## 11b. Fixture Abstraction Layer (`lumina/control/fixture.py`)

Bridges the static layout metadata in `lumina/lighting/fixture_map.py` to the network control
layer. Contains two public types.

### `FixtureState` (dataclass)

Tracks the live output state of one physical fixture.

| Field | Type | Description |
|---|---|---|
| `fixture_id` | `int` | Fixture address (1-255), mirrors `info.fixture_id` |
| `info` | `FixtureInfo` | Static metadata (position, type, role, groups, name) |
| `last_command` | `FixtureCommand \| None` | Most recent command sent; `None` if never commanded |
| `online` | `bool` | `True` when seen within timeout window |
| `last_seen` | `float` | `time.monotonic()` of last heartbeat/discovery; `0.0` = never |
| `firmware_version` | `int` | Reported by fixture at discovery time; `0` if unknown |

Helper methods:

- `seconds_since_seen() -> float` -- elapsed seconds since last heartbeat; `inf` if never seen.
- `is_dark() -> bool` -- `True` if `last_command` is `None` or all channels are zero.

### `FixtureRegistry` (class)

Owns the full set of known fixtures; single source of truth for online status.

```python
from lumina.control.fixture import FixtureRegistry
from lumina.lighting.fixture_map import FixtureMap

fmap = FixtureMap()               # default 15-fixture layout
registry = FixtureRegistry(fmap)  # all fixtures start offline

# Discovery / heartbeat callback (from discovery.py)
registry.mark_seen(fixture_id=1, firmware_version=3)

# Lighting engine output -> registry -> network sender (each 60fps frame)
registry.apply_commands(commands)

# Periodic health sweep (run every ~1s from a background task)
registry.check_timeouts(timeout=10.0)

for state in registry.online_fixtures():
    ...
```

Key methods:

| Method | Description |
|---|---|
| `get(id) -> FixtureState \| None` | Look up by ID |
| `all_states() -> list[FixtureState]` | All fixtures, sorted by ID |
| `online_fixtures() -> list[FixtureState]` | Only `online=True` fixtures |
| `__len__() -> int` | Total registered fixture count |
| `update_command(cmd)` | Store last sent command; broadcast (ID 0) updates all |
| `mark_seen(id, firmware_version=0)` | Transition fixture to `online=True`, record timestamp |
| `check_timeouts(timeout=10.0)` | Mark fixtures offline if unseen longer than `timeout` seconds |
| `apply_commands(list[FixtureCommand])` | Batch `update_command` for a full lighting frame |
| `summary() -> str` | Multi-line human-readable status string for debug logging |

**Thread safety**: not thread-safe. All calls must come from the asyncio event loop that owns
the network manager.

**Broadcast semantics**: `update_command` with `fixture_id=0` writes the same command into
`last_command` for every fixture in the registry.

---

## 12. Simulator

### Architecture

React + Three.js + TypeScript. Vite dev server. Connects to Python backend via WebSocket.

Key files in `simulator/src/`:

| File | Purpose |
|---|---|
| `App.tsx` | Main component. Canvas + ControlPanel + WebSocket + audio hooks |
| `components/Room.tsx` | 3D room model (walls, floor, ceiling) |
| `components/Fixture.tsx` | Virtual fixture with beam rendering (SpotLight + volumetric cone) |
| `components/AudioPlayer.tsx` | Audio playback with waveform display |
| `components/ControlPanel.tsx` | Transport, pattern dropdown, genre dropdown, intensity slider, effects |
| `components/Spectrogram.tsx` | Real-time spectrogram visualization |
| `hooks/useWebSocket.ts` | WebSocket connection + message parsing |
| `hooks/useAudio.ts` | Web Audio API integration, `loadUrl()` for HTTP audio loading |
| `types/fixtures.ts` | TypeScript types for FixtureCommand, MusicState, messages |

### Fixture Rendering

Each fixture type renders differently in Three.js:
- **PAR**: SpotLight with volumetric cone, aimed downward at 45 degrees
- **STROBE**: Point light that flashes based on strobe_rate/intensity
- **LED_BAR**: Rectangular area light
- **LASER**: Line geometry (simplified)

Colors are converted from RGBW bytes to THREE.Color. White channel adds uniform brightness.

### Simulator Workflow

1. Connects to `ws://localhost:8765/ws`
2. Receives `fixture_layout` -- creates 3D fixture objects
3. Receives `playback_start` -- auto-loads audio from `/audio`, seeks to `start_timestamp`
4. Receives `fixture_commands` at 60fps -- updates fixture colors/intensities
5. Receives `music_state` at 60fps -- updates debug display
6. Sends transport/control messages from UI

---

## 13. ML Pipeline

### Overview

Video-based training pipeline that learns professional lighting decisions from concert footage.
All modules in `lumina/ml/`.

### Pipeline Steps (end-to-end)

1. **Download videos**: `lumina/ml/data/downloader.py` -- yt-dlp wrapper with genre search queries
2. **Catalog**: `lumina/ml/data/catalog.py` -- video metadata (genre, quality, camera type)
3. **Extract audio features**: `lumina/ml/audio/batch_analyzer.py` -- batch LUMINA audio analysis
4. **Extract video features**: Scene classification + lighting extraction + cut detection
5. **Align**: `lumina/ml/data/aligner.py` -- merge audio + video into Parquet training pairs
6. **Train**: `lumina/ml/model/train.py` -- multi-task transformer training with resume/shutdown
7. **Inference**: `lumina/ml/model/inference.py` -- sliding window, 10fps model -> 60fps output
8. **Integration**: `lumina/ml/integration/hybrid_engine.py` -- blend ML + rule-based

### Model Architecture (`lumina/ml/model/architecture.py`)

**LightingTransformer** -- genre-conditioned temporal fusion transformer.

```
Input per timestep:
  11 MusicState floats + 8-dim genre embedding + 8-dim segment embedding
  Context window: 40 frames (4 seconds at 10fps)

Encoder:
  Linear(~36 -> 128) -> 4x TransformerEncoderLayer(d_model=128, nhead=8, dim_ff=256)
  Causal attention mask

Decoder heads:
  Color:   Linear(128 -> 6) -> sigmoid  [hue, sat, sec_hue, diversity, temp, brightness]
  Spatial: Linear(128 -> 5) -> sigmoid  [left, center, right, symmetry, variance]
  Effect:  Linear(128 -> 3) -> sigmoid  [strobe, blackout, brightness_delta]

~400-500K parameters total
```

### Training (`lumina/ml/model/train.py`)

**Loss**: Multi-task weighted sum:
- Huber loss for color head (weight 1.0)
- MSE loss for spatial head (weight 0.5)
- BCE loss for strobe (weight 1.0) and blackout (weight 2.0)
- Temporal consistency loss (weight 0.1) -- penalizes frame-to-frame jitter

**Features**: Resume from checkpoint (`--resume`), graceful shutdown (Ctrl+C saves checkpoint),
per-epoch checkpoints to `data/models/checkpoints/`, optional wandb logging.

### Video Analysis Modules

| Module | File | Purpose |
|---|---|---|
| SceneClassifier | `lumina/ml/video/scene_classifier.py` | CLIP zero-shot: stage_view, crowd_view, led_screen, transition |
| LightingExtractor | `lumina/ml/video/lighting_extractor.py` | Per-frame HSV/spatial/temporal features from stage frames |
| CutDetector | `lumina/ml/video/cut_detector.py` | Frame-diff cut detection, unreliable frame masking |

### Integration (`lumina/ml/integration/`)

**HybridLightingEngine** (`hybrid_engine.py`): Wraps rule-based `LightingEngine` + ML
`LightingInferenceEngine`. Blends outputs at configurable weight. Confidence-based weight
reduction when ML output is uncertain. Automatic fallback to pure rule-based if ML
produces invalid output for > 2 seconds.

**IntentMapper** (`intent_mapper.py`): Converts `LightingIntent` (high-level predictions) to
per-fixture `FixtureCommand` using venue fixture layout.

### Data Locations

| What | Path |
|---|---|
| Aligned training data | `data/features/aligned/*.parquet` |
| Audio features | `data/features/audio/*.parquet` |
| Video lighting features | `data/features/lighting/*.parquet` |
| Downloaded videos | `data/videos/raw/{genre}/{video_id}/` |
| Video catalog | `data/videos/metadata/catalog.json` |
| Checkpoints | `data/models/checkpoints/` |
| Latest checkpoint | `data/models/checkpoints/latest_checkpoint.pt` |
| Best model | `data/models/checkpoints/best_model.pt` |

---

## 14. Known Issues / Limitations

### Unimplemented Features

| Feature | Status | Notes |
|---|---|---|
| DJ module (Mode B) | Empty `lumina/dj/` | Phase 4 planned |
| Mobile party UI | Not started | Separate from simulator (Phase 2+) |

### Known Bugs (not fixed)

| ID | Severity | Description |
|---|---|---|
| B7-partial | Info | `firmware/` directory does not exist (expected -- firmware is not written by Claude) |

### Previously Fixed Bugs

| ID | What was fixed |
|---|---|
| B1 | `genre_override`, `intensity`, `manual_effect` WebSocket handlers added to `_handle_transport()` |
| B2 | `uk_bass` profile implemented and registered |
| B3 | Mode radio buttons wired to onChange handlers |
| B4 | `asyncio.get_event_loop()` changed to `asyncio.get_running_loop()` |
| B5 | Extended MusicState fields (7) added to WebSocket broadcast + TypeScript types |
| B6 | Profile blending implemented via `ProfileBlender` + `TransitionEngine` |
| B8 | `pandas`, `pyarrow` added to pyproject.toml |
| B9 | Dataset/architecture schema aligned to 11 features (removed 5 unimplemented) |
| B10 | Training resume support added (checkpoints, graceful shutdown) |
| B11 | Scheduler state saved in checkpoints |
| B13 | Idle loop timestamp drift fixed (absolute timing) |
| B14 | `_handle_transport` seek crash fixed (stale `n=0` in showcase mode) |
| B15 | Missing control modules implemented: `fixture.py`, `discovery.py`, `network.py` |
| B16 | Profile blending + cross-genre transitions: `blender.py`, `transitions.py` |
| B17 | Live audio mode (Mode A) implemented with streaming analyzers |

---

## 15. Testing

### Test Organization

All tests in `tests/`. Uses pytest with pytest-asyncio.

| Test File | Tests | Module |
|---|---|---|
| `test_protocol.py` | UDP encode/decode roundtrip | `lumina/control/protocol.py` |
| `test_discovery.py` | Discovery service lifecycle, packet dispatch, heartbeat | `lumina/control/discovery.py` |
| `test_beat_detector.py` | Streaming/offline beat detection | `lumina/audio/beat_detector.py` |
| `test_energy_tracker.py` | Energy + derivative + spectral centroid | `lumina/audio/energy_tracker.py` |
| `test_onset_detector.py` | Transient detection | `lumina/audio/onset_detector.py` |
| `test_vocal_detector.py` | Vocal presence detection | `lumina/audio/vocal_detector.py` |
| `test_drop_predictor.py` | Drop probability prediction | `lumina/audio/drop_predictor.py` |
| `test_segment_classifier.py` | Frame-by-frame segment classification | `lumina/audio/segment_classifier.py` |
| `test_genre_classifier.py` | Two-stage genre classification | `lumina/audio/genre_classifier.py` |
| `test_structural_analyzer.py` | Offline structural analysis | `lumina/audio/structural_analyzer.py` |
| `test_source_separator.py` | Demucs mock separation | `lumina/audio/source_separator.py` |
| `test_fixture_map.py` | Fixture layout, spatial queries | `lumina/lighting/fixture_map.py` |
| `test_base_profile.py` | BaseProfile utilities, routing | `lumina/lighting/profiles/base.py` |
| `test_patterns.py` | All 20 pattern functions | `lumina/lighting/patterns.py` |
| `test_lighting_engine.py` | Engine dispatch, profile selection | `lumina/lighting/engine.py` |
| `test_rage_trap.py` | Rage trap profile (21 tests) | `lumina/lighting/profiles/rage_trap.py` |
| `test_psych_rnb.py` | Psych R&B profile | `lumina/lighting/profiles/psych_rnb.py` |
| `test_festival_edm.py` | Festival EDM profile | `lumina/lighting/profiles/festival_edm.py` |
| `test_generic_profile.py` | Generic fallback profile | `lumina/lighting/profiles/generic.py` |
| `test_french_melodic.py` | French melodic profile | `lumina/lighting/profiles/french_melodic.py` |
| `test_french_hard.py` | French hard profile | `lumina/lighting/profiles/french_hard.py` |
| `test_euro_alt.py` | Euro alt profile | `lumina/lighting/profiles/euro_alt.py` |
| `test_theatrical.py` | Theatrical profile | `lumina/lighting/profiles/theatrical.py` |
| `test_uk_bass.py` | UK bass profile | `lumina/lighting/profiles/uk_bass.py` |
| `test_profiles_spatial.py` | Spatial pattern tests across profiles | Multiple profiles |
| `test_layer_tracker.py` | Layer count + mask from stems | `lumina/analysis/layer_tracker.py` |
| `test_motif_detector.py` | Macro + micro motif detection | `lumina/analysis/motif_detector.py` |
| `test_arc_planner.py` | Headroom budgeting | `lumina/analysis/arc_planner.py` |
| `test_song_score.py` | Score frame aggregation | `lumina/analysis/song_score.py` |
| `test_blender.py` | Profile blending, weighted avg | `lumina/lighting/blender.py` |
| `test_transitions.py` | Easing functions, transition engine | `lumina/lighting/transitions.py` |
| `test_fixture_registry.py` | Fixture state, registry ops | `lumina/control/fixture.py` |
| `test_network_manager.py` | UDP network manager lifecycle | `lumina/control/network.py` |
| `test_app.py` | App integration tests | `lumina/app.py` |
| `test_server.py` | WebSocket server tests | `lumina/web/server.py` |

### Known Test Failures

- `tests/test_beat_detector.py` -- streaming/offline integration tests require madmom dependency
  (madmom needs Cython + C compiler, has no Python 3.11+ wheels)
- `tests/test_source_separator.py` -- mock separation tests (may fail depending on demucs install)

### Running

```bash
pytest tests/                          # All tests
pytest tests/ -m "not slow"            # Skip audio processing tests
pytest tests/test_patterns.py -v       # Specific file, verbose
pytest tests/ -k "test_rage"           # Filter by name
```

---

## 16. Dependencies

### Python (pyproject.toml)

**Core dependencies**:

| Package | Version | Purpose |
|---|---|---|
| librosa | >= 0.10 | Audio feature extraction (MFCCs, spectral, etc.) |
| aubio | >= 0.4 | Beat detection, onset detection |
| sounddevice | >= 0.4 | Audio capture (WASAPI on Windows, PulseAudio on Linux) |
| numpy | >= 1.26 | Array operations |
| scipy | >= 1.12 | Signal processing (find_peaks, etc.) |
| torch | >= 2.1 | ML framework (GPU-accelerated) |
| demucs | >= 4.0 | Source separation (GPU) |
| websockets | >= 12.0 | WebSocket client/server |
| starlette | >= 0.36 | ASGI web framework |
| uvicorn | >= 0.27 | ASGI server |
| pydantic | >= 2.5 | Data validation |

**Optional -- audio_full** (requires manual install):

| Package | Notes |
|---|---|
| madmom >= 0.16 | Requires Cython + C compiler. No Python 3.11+ wheels |
| essentia >= 2.1b6 | No Windows wheels. Install from conda-forge |

**Optional -- dev**:

| Package | Purpose |
|---|---|
| pytest >= 8.0 | Testing |
| pytest-asyncio >= 0.23 | Async test support |
| ruff >= 0.3 | Lint + format |
| mypy >= 1.8 | Type checking (strict mode) |
| pre-commit >= 3.6 | Git hooks |
| httpx >= 0.27 | HTTP testing |

**Optional -- ml_training**:

| Package | Purpose |
|---|---|
| opencv-python >= 4.8 | Video frame processing |
| transformers >= 4.30 | CLIP for scene classification |
| pillow >= 10.0 | Image processing |
| wandb >= 0.16 | Experiment tracking |
| yt-dlp >= 2024.1 | Video downloading |
| imageio-ffmpeg >= 0.5 | FFmpeg integration |
| pandas >= 2.0 | Parquet I/O |
| pyarrow >= 12.0 | Parquet backend |
| tqdm >= 4.66 | Progress bars |

### Simulator (simulator/package.json)

| Package | Version | Purpose |
|---|---|---|
| react | ^18.3.1 | UI framework |
| react-dom | ^18.3.1 | React DOM renderer |
| three | ^0.164.0 | 3D rendering |
| @react-three/fiber | ^8.16.0 | React-Three.js bridge |
| @react-three/drei | ^9.105.0 | Three.js helpers (OrbitControls, etc.) |
| vite | ^5.3.1 | Build tool + dev server |
| typescript | ^5.5.2 | Type checking |
| tailwindcss | ^3.4.4 | CSS framework |
| prettier | ^3.3.2 | Code formatting |
| eslint | ^8.57.0 | Linting |

---

## 17. Design Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| Phase 1 | Rule-based expert system first, neural model later | Genre profiles + ML classifiers are more controllable and debuggable |
| Phase 1 | GPU-first, no CPU fallbacks | Both dev (RTX 4070) and prod (Jetson) have CUDA GPUs |
| Phase 1 | Custom UDP, not DMX/ArtNet | Minimum latency for custom fixtures. ArtNet bridge can be added later |
| Phase 1 | 60fps target refresh rate | Professional-grade smoothness for lighting transitions |
| Phase 1 | Async everything | Main loop is asyncio event loop. Network/audio in thread pools |
| Phase 1 | Simulator mirrors production | Same fixture protocol -- code that works in simulator works on hardware |
| 2026-03-10 | Audio auto-sync between backend and simulator | `/audio` HTTP endpoint + `playback_start` WebSocket message + 2s drift correction |
| 2026-03-11 | ML training data from YouTube concert footage | Largest free dataset of concert lighting. yt-dlp for downloading |
| 2026-03-11 | Predict LightingIntent, not FixtureCommand | Video shows overall feel, not per-fixture state. Intent is a better abstraction |
| 2026-03-11 | Small transformer (~500K params) | Must run at 60fps on Jetson 8GB. Trains fast on limited data |
| 2026-03-11 | Hybrid blending (ML + rules) | Rule-based provides reliable baseline, ML adds nuance. Adjustable weight |
| 2026-03-11 | NUM_MUSIC_FEATURES = 11 (not 16) | 5 fields have no analyzer implementations yet. Will be re-added later |

---

## 18. Code Conventions

- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints required on all function signatures
- Google-style docstrings on all public classes/functions
- `pathlib.Path` for all file paths (cross-platform: Windows dev, Linux prod)
- `logging` module only (never `print()`)
- `asyncio` for all I/O-bound operations
- ruff for lint+format, mypy strict mode for type checking
- Line length: 100 characters max
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

### Pattern function signature (all 20):
```python
def pattern_fn(fixtures, state, timestamp, color, **kwargs) -> dict[int, FixtureCommand]
```

### Profile command builder:
```python
make_command(fixture, color, intensity=1.0, strobe_rate=0, strobe_intensity=0, special=None)
```

### MusicState field conventions:
- All extended fields (7) have safe defaults (0, None, 1.0) for backward compat
- `energy` must be declared as local variable in each profile method that uses it
- `_segment_start_time` (not `_verse_start_time`) in profiles -- reset on any segment change

---

## 19. Next-Gen Roadmap

See `docs/next-gen-plan.md` for the full critical assessment and feature roadmap.

**Key architectural weaknesses identified (2026-03-21):**

1. **W1**: Live mode has no source separation (degrades all stem-dependent analysis)
2. **W2**: No latency measurement or compensation
3. **W3**: Extended MusicState fields (7) are computed but ignored by most profiles
4. **W4**: No photosensitive safety limiter (strobe frequency/duty cycle unbound)
5. **W5**: File mode blocks until full analysis completes (bad UX)
6. **W6**: No show recording/playback
7. **W7**: No mobile party UI (only 3D simulator)
8. **W8**: No multi-song energy arc (each song independent)
9. **W9**: No fixture zones (entire room treated uniformly)
10. **W10**: ML pipeline exists but has zero training data

**Priority order:** F1 (safety) > F5 (use extended MusicState) > F4 (recording) > F2 (mobile UI) > F3 (progressive analysis) > F7 (streaming separation) > F8 (latency)
