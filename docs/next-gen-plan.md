# LUMINA Next-Gen Plan — Critical Assessment & Feature Roadmap

**Date:** 2026-03-21
**Scope:** Post-Phase 1 improvements — design weaknesses, missing capabilities, and the path to a production-ready system.

---

## Part 1: Critical Assessment of Current State

### What's Working Well

The Phase 1 foundation is solid. The audio pipeline (10 analyzers + extended analysis), 9 genre profiles with 20 patterns, profile blending/transitions, full UDP fixture protocol, 3D simulator, WebSocket bridge, and ML pipeline scaffolding are all implemented with 897+ passing tests.

### Design Weaknesses

These are not bugs — they are architectural gaps that will cause real problems at a live party.

#### W1. Live Mode Runs on Full Mix (No Source Separation)

**Problem:** File mode uses Demucs to separate drums/bass/vocals/other stems, then runs specialized analyzers on each stem. Live mode skips separation entirely and feeds the full mix to all analyzers.

**Impact:** Beat detection, onset classification, and vocal detection are all significantly worse on a full mix vs. isolated stems. The quality gap between file mode and live mode is huge — the lighting looks noticeably dumber in live mode.

**Root cause:** Streaming Demucs doesn't exist. The model needs ~10 seconds of context to produce good separations, making real-time use impractical at the per-frame level.

**Fix approach:** Implement a sliding-window source separation pipeline. Buffer 5-10 seconds of audio, run Demucs on GPU in a background thread, and feed separated stems to analyzers with a fixed latency offset. Accept ~5s latency for stem-aware analysis while keeping a fast path for beat/energy that runs on the mix.

#### W2. No Latency Compensation

**Problem:** Live mode has ~100-200ms latency from audio capture to light output, but there is no mechanism to measure, report, or compensate for it. Worse, if stem separation is added (W1), latency will jump to 5+ seconds.

**Impact:** Lights feel "late" — especially visible on transients (kicks, snare hits). The system doesn't know its own latency, so it can't tell the user.

**Fix approach:**
- Add a latency measurement system (timestamp audio capture → light command emission).
- For the fast path (~100ms): acceptable for wash/color changes, needs predictive look-ahead for transients.
- For the stem path (~5s): pre-buffer commands. The engine generates commands 5s ahead of playback, holds them in a timed queue, and releases them synchronized to the actual audio output.

#### W3. Profiles Don't Use Extended MusicState

**Problem:** The extended analysis pipeline (LayerTracker, MotifDetector, ArcPlanner, SongScore) computes 7 additional MusicState fields per frame: `layer_count`, `layer_mask`, `motif_id`, `motif_repetition`, `notes_per_beat`, `note_pattern_phase`, `headroom`. But most profiles completely ignore these fields.

**Impact:** The system does expensive analysis (chroma similarity, IOI regularity, headroom budgeting) whose results go unused. The lighting doesn't get smarter from knowing the song structure — it just reacts to energy/beats like any basic system.

**Fix approach:** Systematically integrate extended fields into all 9 profiles:
- **headroom**: Scale intensity ceiling per frame (prevents "everything at max" fatigue).
- **layer_count/layer_mask**: Control how many fixtures are active (sparse instrumentation = fewer lights).
- **motif_id/repetition**: Visual consistency — same riff gets same pattern. Repetition count drives evolution (first time = simple, fourth time = complex).
- **notes_per_beat/note_pattern_phase**: Drive chase/stutter timing to match melodic rhythm, not just beat grid.

#### W4. No Photosensitive Safety System

**Problem:** No limits on strobe frequency, duty cycle, or total flash exposure. A profile could theoretically drive all strobes at 25Hz continuously.

**Impact:** Photosensitive seizure risk. Also a legal liability in some jurisdictions (UK Ofcom guidelines, EU accessibility standards).

**Fix approach:** Implement a `SafetyLimiter` as a mandatory post-processing stage in the lighting engine:
- Max strobe frequency: 3Hz sustained, 10Hz for bursts < 1 second.
- Max strobe duty cycle: 20% over any 5-second window.
- Total brightness limiter: prevent all 15 fixtures at max for extended periods.
- Configurable "safety level" (party mode vs. accessible mode).
- Cannot be bypassed by profiles or manual effects.

#### W5. File Mode Blocks Until Full Analysis Completes

**Problem:** When a user loads a song in file mode, the entire audio analysis pipeline runs to completion before any lights turn on. For a 5-minute song with source separation + extended analysis, this can take 30-60+ seconds.

**Impact:** User starts a song at a party, stares at a black room for a minute. Terrible UX.

**Fix approach:** Progressive analysis with early playback:
1. Run fast analyzers first (energy, beat detection on mix) — takes ~5s.
2. Start playback with basic lighting immediately.
3. Run Demucs + stem-aware analysis in background.
4. Hot-swap to full-quality MusicState frames as they become available.
5. Show analysis progress in the web UI.

#### W6. No Show Recording/Playback

**Problem:** Can't record a generated light show and replay it later. If the system produces a perfect show for a specific song, there's no way to save it.

**Impact:** Every playback re-analyzes from scratch. Can't curate a library of "known-good" shows. Can't share shows between venues.

**Fix approach:**
- Record `(timestamp, list[FixtureCommand])` frames to a compact binary format.
- Save alongside audio file hash for verification.
- Playback mode: skip analysis, load recorded show, stream commands.
- Enable A/B comparison (rule-based vs. ML vs. recorded).

#### W7. No Mobile Party UI

**Problem:** The only UI is the 3D simulator — a development tool that requires a desktop browser and shows a 3D room model. There is no mobile-friendly control panel for live use.

**Impact:** At a party, the host needs quick access to: play/pause, intensity slider, genre override, manual effects, blackout. The 3D simulator is completely wrong for this context.

**Fix approach:** Build a separate React SPA (or route within the existing web server) optimized for mobile:
- Large touch targets, dark theme, minimal information.
- Mode selector, intensity slider, genre override dropdown.
- Big "BLACKOUT" and "STROBE" buttons.
- Current song info + energy meter.
- Fixture health overview (online/offline count).
- Same WebSocket connection as the simulator.

#### W8. No Multi-Song Energy Arc

**Problem:** Each song is analyzed independently. When songs transition (Mode C queue), the system has no awareness of the overall energy arc of the party/set.

**Impact:** Two high-energy songs in a row both hit 100% intensity from the start. No concept of "saving intensity headroom for the drop 3 songs from now." The party feels flat because there's no macro-level dynamics.

**Fix approach:**
- Build a `SetArcPlanner` that maintains state across songs.
- When songs are queued (Mode C), pre-analyze the full queue and plan a set-level energy arc.
- Map set position to a "global headroom" multiplier that the per-song ArcPlanner respects.
- Allow the user to mark "peak moment" songs in the queue.

#### W9. No Fixture Zones

**Problem:** Profiles treat all fixtures as one flat list sorted spatially. There's no concept of independent zones (e.g., dance floor vs. bar area vs. stage backdrop).

**Impact:** Can't have chill ambient lighting at the bar while the dance floor goes full strobe. Every area of the room gets the same treatment.

**Fix approach:**
- Extend `FixtureMap` with zone definitions (list of fixture IDs per zone).
- Each zone can have an independent profile assignment or intensity multiplier.
- Profiles receive zone-filtered fixture lists.
- Zone crossfades when people move between areas (optional, requires presence detection).

#### W10. ML Pipeline Has No Data

**Problem:** The full ML training pipeline exists (video download → feature extraction → alignment → transformer training → hybrid inference) but has never been run. No training data has been collected.

**Impact:** The ML pipeline is theoretical. The hybrid engine exists but will produce garbage until trained on real data.

**Fix approach:** This is not a code problem — it's a data collection effort:
1. Curate 50-100 concert videos per genre (400-800 total) using the existing downloader.
2. Run batch feature extraction (audio + video).
3. Manually review alignment quality on 10% sample.
4. Train initial model and evaluate against rule-based baseline.
5. The code is ready — the bottleneck is human curation and GPU hours.

---

## Part 2: Next-Gen Feature Roadmap

### Phase 2A: Production Hardening (Prerequisite for Live Use)

These must be done before the first real party.

| # | Feature | Depends On | Effort | Priority |
|---|---------|-----------|--------|----------|
| F1 | Safety limiter (W4) | None | 2-3 days | **CRITICAL** |
| F2 | Mobile party UI (W7) | None | 3-5 days | **CRITICAL** |
| F3 | Progressive file analysis (W5) | None | 2-3 days | HIGH |
| F4 | Show recording/playback (W6) | None | 2-3 days | HIGH |
| F5 | Integrate extended MusicState into profiles (W3) | None | 3-5 days | HIGH |
| F6 | Fixture health dashboard in UI | F2 | 1 day | MEDIUM |

### Phase 2B: Live Mode Quality (Make Live Mode Match File Mode)

| # | Feature | Depends On | Effort | Priority |
|---|---------|-----------|--------|----------|
| F7 | Sliding-window source separation (W1) | None | 5-7 days | HIGH |
| F8 | Latency measurement & compensation (W2) | None | 3-4 days | HIGH |
| F9 | Predictive transient look-ahead | F8 | 3-4 days | MEDIUM |
| F10 | Dual-path architecture (fast mix + slow stems) | F7, F8 | 3-4 days | HIGH |

### Phase 2C: Intelligence Upgrade

| # | Feature | Depends On | Effort | Priority |
|---|---------|-----------|--------|----------|
| F11 | Multi-song energy arc (W8) | None | 3-4 days | HIGH |
| F12 | Fixture zones (W9) | None | 3-5 days | MEDIUM |
| F13 | Effect sequencer (programmable effect chains) | None | 3-4 days | MEDIUM |
| F14 | Audience energy feedback loop (mic-based crowd detection) | None | 5-7 days | LOW |

### Phase 3: ML & Learning

| # | Feature | Depends On | Effort | Priority |
|---|---------|-----------|--------|----------|
| F15 | Training data collection & curation (W10) | None | 2-3 weeks | HIGH |
| F16 | Initial model training + evaluation | F15 | 1-2 weeks | HIGH |
| F17 | Online learning from user overrides | F16 | 1-2 weeks | MEDIUM |
| F18 | DMX/ArtNet bridge for off-the-shelf fixtures | None | 3-5 days | MEDIUM |

### Phase 4: AI DJ (Mode B)

| # | Feature | Depends On | Effort | Priority |
|---|---------|-----------|--------|----------|
| F19 | BPM matching & crossfade engine | None | 2-3 weeks | MEDIUM |
| F20 | Set planning with energy arc | F11 | 1-2 weeks | MEDIUM |
| F21 | Automated genre-aware transitions | F19 | 2-3 weeks | MEDIUM |

---

## Part 3: Detailed Design for Priority Features

### F1: Safety Limiter

```
Location: lumina/lighting/safety.py
Integration point: LightingEngine.generate() — wraps profile output

SafetyLimiter:
  - process(commands: list[FixtureCommand]) -> list[FixtureCommand]
  - Tracks rolling 5-second window of strobe events per fixture
  - Enforces:
    - Max sustained strobe: 3Hz (strobe_rate capped at ~77 = 3/25*255)
    - Max burst strobe: 10Hz for max 1 second, then 3s cooldown
    - Max simultaneous strobe fixtures: 8 of 15 (prevents full-room flash)
    - Total brightness ceiling: sum of all fixture intensities < 80% of max
  - Safety level enum: STANDARD, ACCESSIBLE, UNRESTRICTED
  - Cannot be overridden by profiles or manual effects
  - Logged warnings when limiting is applied
```

### F2: Mobile Party UI

```
Location: lumina/web/static/party/ (or separate route)
Stack: React (shared with simulator), Tailwind, mobile-first

Screens:
  1. Now Playing — song name, energy meter, segment indicator, BPM
  2. Controls — intensity slider, genre override, mode selector
  3. Effects — big buttons: BLACKOUT, STROBE BURST, UV FLASH, COLOR WASH
  4. Queue (Mode C) — drag-to-reorder, mark peak songs
  5. Status — fixture count, online/offline, network latency

Navigation: bottom tab bar, 5 tabs
WebSocket: same /ws endpoint, same message format
```

### F4: Show Recording

```
Location: lumina/recording/recorder.py, lumina/recording/player.py
Format: Custom binary (header + frame stream)

Header:
  magic: "LREC" (4 bytes)
  version: uint8
  audio_hash: sha256 (32 bytes) — for verification
  fps: uint8
  fixture_count: uint8
  duration_ms: uint32

Frame (per tick):
  timestamp_ms: uint32
  commands: fixture_count × 8 bytes (same as UDP payload)

File size: 60fps × 15 fixtures × 8 bytes = 7.2 KB/s = ~2.2 MB per 5-min song
Compression: optional zstd wrapper

API:
  recorder = ShowRecorder(audio_hash, fps, fixture_count)
  recorder.record_frame(timestamp, commands)
  recorder.save("show.lrec")

  player = ShowPlayer("show.lrec")
  for timestamp, commands in player.play():
      ...
```

### F5: Extended MusicState Integration

Each profile gets specific enhancements:

```
ALL PROFILES:
  - Multiply final intensity by state.headroom (prevents loudness fatigue)
  - Use _layer_fixture_count() for active fixture selection (already exists, unused)

RAGE_TRAP:
  - motif_repetition > 3: escalate to strobe_burst (recognizes "the part everyone knows")
  - layer_count < 2: single red spotlight (sparse = minimal)

PSYCH_RNB:
  - notes_per_beat drives breathe cycle speed
  - motif changes trigger slow color palette rotation

FESTIVAL_EDM:
  - layer_count drives build intensity (more layers = closer to drop)
  - motif_repetition resets on new section (fresh visual for new riff)

THEATRICAL:
  - layer_mask["vocals"] directly drives spotlight intensity
  - motif_id maps to distinct color temperature (each theme gets its own warmth)

UK_BASS:
  - notes_per_beat > 4: activate scatter pattern (fast amen breaks)
  - layer_count drop (e.g., 4→1): hard cut to single fixture
```

### F7: Sliding-Window Source Separation

```
Location: lumina/audio/streaming_separator.py

Architecture:
  - Ring buffer: 10 seconds of audio at 44100 Hz
  - Process trigger: every 2 seconds, submit last 10s to Demucs on GPU
  - Output: 4 stem buffers, each covering the most recent 10s
  - Latency: ~5 seconds (half the window, minus processing time)
  - Thread: dedicated background thread with GPU context

Interface:
  class StreamingSeparator:
    def __init__(self, sr: int = 44100, window_sec: float = 10.0):
    def feed(self, chunk: np.ndarray) -> None:
    def get_stems(self, duration_sec: float = 2.0) -> StemSet | None:
      """Returns separated stems for the most recent `duration_sec`.
      Returns None if not enough data has accumulated yet."""

Integration with live mode:
  - Fast path: beat/energy/onset run on raw mix (0 latency)
  - Slow path: vocal/genre/segment run on separated stems (~5s latency)
  - LightingEngine merges: fast path drives beat-sync, slow path drives color/profile selection
```

### F10: Dual-Path Architecture

```
Live mode data flow (revised):

Audio Input
  ├─→ Fast Path (0ms latency)
  │     ├── BeatDetector (mix)
  │     ├── EnergyTracker (mix)
  │     ├── OnsetDetector (mix)
  │     └── → MusicState (partial: beats, energy, onsets)
  │           └── → LightingEngine (beat-reactive: strobe, chase timing)
  │
  └─→ Slow Path (~5s latency via StreamingSeparator)
        ├── VocalDetector (vocals stem)
        ├── GenreClassifier (all stems)
        ├── SegmentClassifier (all stems)
        ├── DropPredictor (enhanced with stem data)
        └── → MusicState (full: adds vocals, genre, segment, drop)
              └── → LightingEngine (color, profile, intensity decisions)

Merge strategy:
  - Fast path output is the "base" — always rendered immediately
  - Slow path updates arrive ~5s behind but drive the "artistic" decisions
  - When slow path updates arrive, smoothly transition (don't jump)
  - If slow path fails or falls behind, fast path still works alone
```

---

## Part 4: Implementation Order Recommendation

```
Sprint 1 (Week 1-2):  F1 (Safety) + F5 (Extended MusicState in profiles) + F4 (Recording)
Sprint 2 (Week 3-4):  F2 (Mobile UI) + F3 (Progressive analysis) + F6 (Health dashboard)
Sprint 3 (Week 5-7):  F7 (Streaming separation) + F8 (Latency) + F10 (Dual path)
Sprint 4 (Week 8-9):  F11 (Multi-song arc) + F12 (Zones) + F13 (Effect sequencer)
Sprint 5 (Week 10+):  F15-F18 (ML training + DMX bridge)
```

Safety limiter (F1) is Sprint 1 because it's a liability. Extended MusicState integration (F5) is Sprint 1 because the expensive analysis is already running and going to waste. Show recording (F4) is Sprint 1 because it enables A/B testing for everything that follows.

---

## Part 5: Technical Debt to Address

| Item | Location | Issue | Fix |
|------|----------|-------|-----|
| DJ module is empty | `lumina/dj/` | Only `__init__.py` exists | Either remove or add "not implemented" marker |
| Genre classifier hand-tuned | `genre_classifier.py` | Feature prototypes are manually tuned constants | Collect real labeled data, train proper classifier (F15) |
| Live mode has no source separation | `app.py:_run_live_mode` | Runs all analyzers on full mix | F7 (streaming separation) |
| Hardcoded fixture layout | `fixture_map.py` | 15 fixtures, fixed positions | Make configurable via JSON file or web UI |
| No error recovery in live mode | `app.py:_run_live_mode` | Audio stream errors crash the app | Add reconnection logic for audio device |
| NetworkManager not used in app.py | `app.py` | App creates raw UDP socket instead of using NetworkManager | Refactor app.py to use NetworkManager |
| No graceful shutdown | `app.py` | Ctrl+C may leave WebSocket server dangling | Add signal handler, cleanup sequence |
