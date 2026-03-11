# Audio Analysis Pipeline Architecture

## Overview

The audio analysis pipeline is the first layer of LUMINA's three-layer architecture. It
processes music in real-time, extracting features that the lighting engine uses to make
decisions. The pipeline produces a `MusicState` object at 60fps.

## Data Flow

```
Audio Source (file, WASAPI loopback, PulseAudio monitor)
  -> Capture / Decode (numpy array, mono, 44100 Hz)
  -> Source Separation (demucs — vocals, drums, bass, other)
  -> Parallel Feature Extractors:
     |-> Beat Detector (BPM, beat phase, bar phase, downbeats)
     |-> Energy Tracker (energy envelope, derivative, sub-bass energy)
     |-> Onset Detector (kick, snare, hi-hat, clap transients)
     |-> Segment Classifier (verse, chorus, drop, breakdown, intro, outro, bridge)
     |-> Genre Classifier (two-stage: family -> profile weights)
     |-> Vocal Detector (vocal energy, language-agnostic)
     |-> Drop Predictor (1-4 bar look-ahead probability)
     |-> Spectral Features (centroid, brightness)
  -> MusicState Assembly (merge all features into single dataclass)
  -> Output at 60fps to Lighting Engine
```

## Key Components

### Audio Capture (`scripts/capture_audio.py`)

Platform-specific audio input abstracted behind a common interface:
- **Windows**: WASAPI loopback via `sounddevice`
- **Linux/Jetson**: PulseAudio/PipeWire monitor source via `sounddevice`
- **File input**: Direct decode via `librosa.load()` or `soundfile`

### Source Separation (`lumina/audio/source_separator.py`)

Uses **demucs** (GPU-accelerated) to separate audio into stems:
- Vocals, drums, bass, other
- Enables per-stem analysis (e.g., vocal energy from vocal stem only)
- Runs on GPU in both dev (RTX 4070) and production (Jetson Orin Nano)

### Beat Detection (`lumina/audio/beat_detector.py`)

- Uses madmom for beat and downbeat tracking
- Outputs BPM, beat positions, bar positions
- Computes beat_phase (0-1 within beat) and bar_phase (0-1 within bar)

### Energy Tracking (`lumina/audio/energy_tracker.py`)

- RMS energy envelope with smoothing
- Energy derivative (rising/falling detection)
- Sub-bass energy (20-80 Hz band isolation)

### Onset Detection (`lumina/audio/onset_detector.py`)

- Classifies transients: kick, snare, hi-hat, clap
- Works on drum stem from source separation for cleaner detection

### Segment Classification (`lumina/audio/segment_classifier.py`)

- Identifies song structure: verse, chorus, drop, breakdown, intro, outro, bridge
- Uses feature-based heuristics (energy contour, spectral changes, repetition)

### Genre Classification (`lumina/audio/genre_classifier.py`)

- Two-stage classification:
  1. Family: Hip-Hop/Rap, Electronic, Hybrid
  2. Profile: one of 8 genre profiles
- Outputs weighted dictionary (e.g., `{psych_rnb: 0.6, festival_edm: 0.3, ...}`)

### Vocal Detection (`lumina/audio/vocal_detector.py`)

- Language-agnostic vocal presence and energy detection
- Works on vocal stem from source separation
- Must handle French, German, Portuguese, English equally

### Drop Prediction (`lumina/audio/drop_predictor.py`)

- 1-4 bar look-ahead drop probability (0-1)
- Uses energy trajectory, spectral patterns, structural context

## MusicState Output

The pipeline assembles all features into a `MusicState` dataclass at 60fps:

```python
@dataclass
class MusicState:
    timestamp: float
    bpm: float
    beat_phase: float       # 0.0-1.0
    bar_phase: float        # 0.0-1.0
    is_beat: bool
    is_downbeat: bool
    energy: float           # 0.0-1.0
    energy_derivative: float
    segment: str
    genre_weights: dict[str, float]
    vocal_energy: float
    spectral_centroid: float
    sub_bass_energy: float
    onset_type: str | None
    drop_probability: float
```

## Performance Requirements

- **Latency**: <200ms end-to-end (Mode A live listening)
- **Throughput**: 60 MusicState outputs per second
- **GPU**: CUDA required for demucs and ML inference
- **Memory**: <4GB GPU memory for full pipeline
