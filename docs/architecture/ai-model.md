# AI Model Architecture

## Overview

LUMINA's lighting intelligence uses a hybrid approach: a rule-based expert system (Phase 1-2)
combined with an optional ML model trained on concert footage (Phase 4+). This document
covers both systems.

## Phase 1-2: Rule-Based Expert System

### Architecture

```
MusicState (from audio pipeline)
  -> Genre Weight Resolver (pick dominant profile or blend)
  -> Profile Engine (genre-specific rules)
  |    -> Pattern Selection (which lighting pattern for this segment/energy)
  |    -> Color Selection (palette lookup based on energy, segment, beat)
  |    -> Timing Engine (beat-sync, bar-sync, or arc-sync depending on profile)
  |    -> Spatial Mapper (which fixtures activate, symmetry/asymmetry)
  |    -> Effect Engine (strobe rate, UV intensity, special effects)
  -> Fixture Command Generator
  -> FixtureCommand[] output at 60fps
```

### Genre Profiles

Each of the 8 genre profiles (`lumina/lighting/profiles/`) is a class inheriting from
`BaseProfile` that implements:

- `select_pattern(state: MusicState) -> Pattern`: Choose lighting pattern
- `get_color(state: MusicState) -> RGBW`: Compute color for current moment
- `get_timing(state: MusicState) -> TimingMode`: Beat-sync, bar-sync, or free
- `get_spatial(state: MusicState) -> SpatialConfig`: Fixture activation map
- `get_effects(state: MusicState) -> EffectConfig`: Strobe, UV, special

### Pattern System

The lighting engine has ~20 built-in patterns (e.g., color wash, chase, strobe burst,
blackout, rainbow sweep, pulse). Profiles select and parameterize these patterns based on
the current musical context.

### Context Tracking

The engine maintains context across frames:
- **Contrast management**: Prevents staying at maximum intensity too long
- **Color history**: Avoids repeating the same color palette for too long
- **Arc planning**: Uses song structure to plan energy trajectory
- **Motif detection**: Recognizes recurring musical phrases for visual callbacks

## Phase 4+: ML Model (Planned)

### Training Data

Concert footage from YouTube analyzed via a video processing pipeline:
1. Extract frames at 10fps
2. Classify frames (stage view vs crowd vs LED screen) using CLIP
3. Extract lighting features (color, brightness, spatial distribution, strobes)
4. Align with LUMINA audio analysis (MusicState timeline)
5. Create training pairs: MusicState -> LightingIntent

### Model Architecture

Temporal Fusion Transformer variant (~500K parameters):
- **Input**: MusicState features + genre embedding + segment embedding (4-second context window)
- **Encoder**: 4x Transformer layers (d_model=128, nhead=8)
- **Decoder heads**: Color (6 outputs), Spatial (5 outputs), Effect (3 outputs)
- **Output**: `LightingIntent` — high-level lighting description, not per-fixture commands

### Hybrid Integration

```
MusicState
  |
  +-> Rule-Based Engine -> FixtureCommand[]
  |                              |
  +-> ML Model -> LightingIntent -> FixtureCommand[]
  |                                        |
  +-> Blender (configurable weight) -> Final FixtureCommand[]
```

The ML weight is adjustable (0.0 = pure rules, 1.0 = pure ML). Default starts at 0.3
for safe blending. Confidence-based adjustment reduces ML weight when model is uncertain.

## Two-Stage Genre Classification

The genre classifier (`lumina/audio/genre_classifier.py`) uses two stages:

1. **Family classification**: Hip-Hop/Rap, Electronic, Hybrid (3 classes)
2. **Profile classification**: Specific profile within family (8 classes)

Output is a weighted dictionary summing to 1.0, enabling smooth blending across profiles
when a track doesn't fit neatly into one category.

## Design Principles

1. **Music understanding, not beat-syncing**: The system must understand song structure,
   genre conventions, and emotional arc.
2. **Genre-appropriate responses**: A Carti mosh-pit track and a Stromae ballad demand
   completely different lighting, even at the same energy level.
3. **Contrast is king**: The most impactful lighting moments are defined by what comes
   before and after them, not by their absolute values.
4. **Less is more**: Restraint (darkness, single-color washes, stillness) is a valid and
   powerful lighting choice. Not every beat needs a light change.
