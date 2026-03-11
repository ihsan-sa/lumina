# LUMINA Project Plan

## Phase 1 — Foundation (Weeks 1-4)

Phase 1 establishes the core audio analysis pipeline, basic lighting engine, 3D simulator,
and fixture control protocol. No ML model training, no DJ engine, no physical hardware deployment.

### Software Goals

- [ ] **Audio analysis pipeline**: beat detection, energy tracking, onset detection
- [ ] **Segment classifier**: verse/chorus/drop/breakdown detection
- [ ] **Genre classifier**: two-stage family-to-profile classification
- [ ] **Basic 3D simulator**: room model, virtual fixtures, audio playback
- [ ] **Fixture control protocol**: UDP packet format design + reference implementation
- [ ] **WebSocket bridge**: Python backend to simulator frontend communication

### Hardware Goals (Parallel — Owner handles)

- [ ] Fixture PCB schematic (PoE PD + DC-DC + MCU + LED driver)
- [ ] Part selection and datasheet review
- [ ] LED and laser module selection and testing

### Operating Modes in Phase 1

| Mode | Description | Phase 1 Status |
|------|-------------|----------------|
| A (Live Listening) | Capture system audio, analyze in real-time | Partial — file input works, live capture stubbed |
| B (AI DJ) | AI selects tracks, handles transitions | Not started (Phase 4) |
| C (Queue Pre-Processing) | Pre-process next song in queue | Not started |

### Development Environment

- **Dev machine**: Windows with NVIDIA RTX 4070 (CUDA)
- **Production target**: NVIDIA Jetson Orin Nano (aarch64, JetPack)
- **Python**: 3.12, managed with venv
- **Simulator**: React + Three.js + Vite

### Success Criteria for Phase 1

1. Audio pipeline correctly detects beats, energy, onsets, segments, and genre for test tracks
2. Lighting engine produces genre-appropriate fixture commands from audio features
3. 3D simulator renders a room with virtual fixtures responding to music in real-time
4. UDP protocol encodes/decodes correctly (roundtrip tests pass)
5. WebSocket bridge connects Python backend to browser simulator

---

## Phase 2 — Profile Refinement & Blending

- Profile blending (weighted multi-profile output)
- Cross-genre transitions
- All 8 genre profiles fully tuned
- Global intensity control via web UI

## Phase 3 — Physical Hardware

- First fixture PCBs fabricated and tested
- PoE network deployed in venue
- Simulator-to-hardware handoff (zero code changes in control layer)
- ESP32-S3 wireless accent fixtures (optional)

## Phase 4 — AI DJ & ML Training

- Mode B (AI DJ) implementation
- Video-based ML training pipeline (concert footage analysis)
- Hybrid blending: ML model + rule-based profiles
- End-to-end neural lighting model

## Phase 5 — Production Polish

- Moving heads and lasers
- TensorRT optimization for Jetson inference
- Mobile control UI polish
- Multi-room support
