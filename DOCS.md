# LUMINA Technical Documentation & Agent Context

Last updated: 2026-03-11

---

## 1. Codebase Review — Critical Findings

### 1.1 Unhandled WebSocket Messages (CRITICAL)

The frontend sends 4 message types that the backend **completely ignores**:

| Frontend Message Type | What It Does | Backend Handler |
|---|---|---|
| `genre_override` | User selects a genre profile from dropdown | **NONE** — dropped silently |
| `intensity` | User adjusts intensity slider (0-100) | **NONE** — dropped silently |
| `manual_effect` | User clicks Blackout / Strobe / UV Flash | **NONE** — dropped silently |
| `audio_loaded` | Frontend notifies backend that audio loaded | **NONE** — dropped silently |

**Location**: `lumina/app.py:508-534` — `_handle_transport()` only handles `"transport"` and `"pattern_override"`.

**Impact**: The genre profile dropdown, intensity slider, and all three manual effect buttons in the simulator UI are **completely non-functional**. They send messages that vanish into the void.

**Fix required**: Add handlers in `_handle_transport()` for each message type:
- `genre_override` → call `self._engine.set_genre_override(profile)` (needs new engine method)
- `intensity` → apply global intensity multiplier to all fixture commands
- `manual_effect` → implement blackout (all zeros), strobe_burst (all strobes max), uv_flash (all UV bars max)

### 1.2 Missing `uk_bass` Lighting Profile (BUG)

The genre classifier (`lumina/audio/genre_classifier.py:48`) lists `uk_bass` as a valid profile and can assign it non-zero weight. However:
- No file `lumina/lighting/profiles/uk_bass.py` exists
- `uk_bass` is NOT in `_PROFILE_REGISTRY` in `lumina/lighting/engine.py:71-80`

**Impact**: When a track is classified as UK bass (e.g., Fred again..), the engine falls back to the generic profile instead of applying UK bass-specific lighting. This is a Phase 1 deliverable gap per CLAUDE.md.

### 1.3 Mode Radio Buttons Are Decorative Only (UI BUG)

In `simulator/src/components/ControlPanel.tsx:176-190`, the Mode radio buttons (Live A, Queue C, DJ B) have:
- No `onChange` handler
- No state management
- No WebSocket message sent to backend

They render but do absolutely nothing. The `name="mode"` attribute allows visual toggling but the selection is never communicated anywhere.

### 1.4 Missing Control Layer Modules

CLAUDE.md specifies these files in `lumina/control/`:
- `protocol.py` — **EXISTS** (UDP encoding/decoding)
- `fixture.py` — **MISSING** (fixture abstraction)
- `discovery.py` — **MISSING** (mDNS fixture discovery)
- `network.py` — **MISSING** (network manager, 60fps command sending)

The `fixture.py` functionality is partially covered by `lumina/lighting/fixture_map.py`. Discovery and network manager are unimplemented — acceptable for Phase 1 (simulator-only), but needed before physical fixtures.

### 1.5 Missing Scripts Directory

CLAUDE.md references `scripts/capture_audio.py`, `scripts/fixture_tester.py`, and `scripts/profile_demo.py`. The `scripts/` directory doesn't exist. Not blocking for Phase 1, but the planned utilities haven't been created.

### 1.6 Missing Documentation Directory

CLAUDE.md references `docs/project-plan.md`, `docs/genre-lighting-profiles.md`, `docs/protocol-spec.md`, and subdirectories `docs/hardware/` and `docs/architecture/`. The `docs/` directory doesn't exist.

### 1.7 Empty DJ Module

`lumina/dj/__init__.py` exists but is empty. Expected per Phase 1 scope (DJ engine is Phase 4).

### 1.8 MusicState WebSocket Serialization Gap

The `MusicState` dataclass (`lumina/audio/models.py`) has fields added since the original spec:
- `layer_count`, `layer_mask`, `motif_id`, `motif_repetition`
- `notes_per_beat`, `note_pattern_phase`, `headroom`

These are **not included** in the WebSocket broadcast (`lumina/web/server.py:286-306`). The frontend `MusicState` TypeScript interface (`simulator/src/types/fixtures.ts:14-30`) also doesn't include them. This means the simulator never receives these enhanced analysis fields.

### 1.9 Live Mode Not Implemented

`lumina/app.py:158` — `"live"` mode logs an error and returns. Only `"file"` and `"showcase"` modes work.

### 1.10 `asyncio.get_event_loop()` Deprecation Warning

`lumina/app.py:189` uses `asyncio.get_event_loop()` which is deprecated in Python 3.12+. Should use `asyncio.get_running_loop()` instead.

---

## 2. Architecture Assessment

### What's Working Well

- **Audio pipeline**: Beat detection, energy tracking, onset detection, source separation, genre classification, structural analysis, drop prediction, vocal detection — all implemented and tested.
- **Lighting engine**: Profile-based system with 7/8 genre profiles implemented (missing uk_bass). Pattern system with 20 patterns. Fixture map with 15 fixtures. Context tracking for contrast management.
- **Simulator**: 3D room rendering with 5 fixture types (par, strobe, UV, LED bar, laser). WebSocket communication works. Audio auto-play from backend works. Pattern showcase works.
- **Protocol**: UDP packet encoding/decoding with proper header format. Tested.
- **Analysis layer**: Layer tracking, motif detection, arc planning, song score — these provide rich context that the rule-based profiles can use.

### Architecture Gaps

1. **No global intensity control** — the `headroom` field from arc planning affects profile decisions, but there's no user-facing global dimmer.
2. **No profile blending** — the engine picks ONE profile exclusively. CLAUDE.md describes weighted blending across profiles, but `blender.py` and `transitions.py` don't exist.
3. **Single-profile selection** — if `genre_weights` is `{psych_rnb: 0.6, festival_edm: 0.3}`, only `psych_rnb` runs. The festival_edm influence is completely lost.

---

## 3. Video-Based ML Training Pipeline — Design Plan

### 3.1 Executive Summary

Build a pipeline that learns professional lighting decisions from concert/rave footage by:
1. Downloading concert videos at scale using yt-dlp
2. Extracting lighting states from video frames (color, brightness, strobe, spatial distribution)
3. Running audio through LUMINA's existing analysis pipeline to get MusicState
4. Training a model that maps MusicState → LightingState
5. Integrating the model as a hybrid blend with existing rule-based profiles

### 3.2 Data Collection Pipeline

**Tool**: `yt-dlp` (actively maintained, supports YouTube, Vimeo, Facebook, etc.)

**Target**: 50-100 hours of concert footage across all 8 genre profiles (~200-400 videos).

**Video Selection Criteria** (ranked by value):
1. **Best**: Fixed wide-angle shots of full stage with visible lighting rig (e.g., "full show" uploads, single-camera bootlegs)
2. **Good**: Multi-cam pro shots where most cuts show the stage (official concert films)
3. **Acceptable**: Festival livestream recordings with frequent stage views
4. **Avoid**: Music videos (post-produced, not real lighting), crowd-focused vlogs, clips under 2 minutes

**Search Strategy**:
```
Per genre profile, collect ~25-50 videos:
- rage_trap: "Playboi Carti concert full", "Travis Scott live show full set"
- psych_rnb: "The Weeknd concert full", "Don Toliver live"
- french_melodic: "Ninho concert complet", "Jul concert live"
- french_hard: "Kaaris concert live"
- euro_alt: "AyVe live concert", "Exetra Archive live"
- theatrical: "Stromae concert full", "Stromae live show"
- festival_edm: "Tomorrowland full set", "David Guetta live", "Armin van Buuren live"
- uk_bass: "Fred again live", "Boiler Room sets"
```

**Download Configuration**:
```python
# yt-dlp options
{
    "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",  # 720p sufficient, saves storage
    "merge_output_format": "mp4",
    "writeinfojson": True,      # Save metadata
    "writethumbnail": True,     # Save thumbnail for visual preview
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "0",  # Lossless for audio analysis
    }],
}
```

**Storage Estimate** (720p):
- Video: ~1GB/hour * 75 hours = ~75GB
- Audio (WAV): ~600MB/hour * 75 hours = ~45GB
- Extracted features: ~5GB
- Total: ~125GB (well within budget)

**Directory Structure**:
```
data/
  videos/
    raw/                    # Downloaded MP4 files
      {genre}/{video_id}/
        video.mp4
        audio.wav
        info.json           # yt-dlp metadata
    metadata/
      catalog.json          # Master catalog: video_id -> genre, artist, tags, quality score
  features/
    audio/                  # Pre-computed MusicState timelines
      {video_id}.parquet
    lighting/               # Extracted lighting features from video
      {video_id}.parquet
    aligned/                # Merged audio+lighting training pairs
      {video_id}.parquet
  models/
    checkpoints/
    final/
```

**Metadata Schema** (`catalog.json`):
```json
{
  "video_id": "dQw4w9WgXcQ",
  "genre_profile": "festival_edm",
  "artist": "David Guetta",
  "title": "Tomorrowland 2024 Full Set",
  "duration_s": 3600,
  "quality_score": 0.85,
  "camera_type": "multi_cam_pro",
  "venue_type": "festival_outdoor",
  "has_led_screens": true,
  "lighting_visibility": "high",
  "notes": ""
}
```

### 3.3 Video Analysis Pipeline — Extracting Lighting from Footage

This is the hardest part. Concert videos are noisy: camera cuts, crowd shots, LED screens, color grading, variable quality. The pipeline must be robust.

**Step 1: Frame Extraction**
- Extract frames at 10fps (not 60fps — lighting changes are visible at 10fps, saves 6x compute)
- Use FFmpeg: `ffmpeg -i video.mp4 -vf "fps=10" -q:v 2 frames/%06d.jpg`

**Step 2: Scene Classification (filter out non-stage frames)**

Train a lightweight binary classifier (or use CLIP zero-shot) to label each frame as:
- `stage_view` (useful — can extract lighting)
- `crowd_view` (discard for lighting extraction)
- `transition` (discard — camera movement, blur)
- `led_screen_closeup` (discard — LED screen colors aren't lighting)

**Implementation**: Use CLIP with prompts like "concert stage with lighting", "concert crowd", "LED screen closeup". No training needed initially.

```python
# CLIP-based scene classifier (zero-shot)
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "concert stage with theatrical lighting",
    "concert crowd audience",
    "LED screen or video display",
    "camera transition or blurry image",
]

def classify_frame(image: Image.Image) -> str:
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)[0]
    return labels[probs.argmax().item()]
```

**Step 3: Lighting Feature Extraction (from stage_view frames)**

For each stage-view frame, extract:

```python
@dataclass
class VideoLightingFrame:
    timestamp: float

    # Global metrics
    overall_brightness: float      # 0-1, mean luminance of stage area
    brightness_variance: float     # Spatial variance (low = wash, high = spots)

    # Color
    dominant_hue: float           # 0-360 degrees
    dominant_saturation: float    # 0-1
    secondary_hue: float          # 0-360 (if bimodal color distribution)
    color_temperature: float      # Warm vs cool (mapped from hue)
    color_diversity: float        # 0-1, how many distinct colors visible

    # Spatial
    left_brightness: float        # Left third of stage
    center_brightness: float      # Center third
    right_brightness: float       # Right third
    top_brightness: float         # Upper half
    bottom_brightness: float      # Lower half
    spatial_symmetry: float       # 0-1, how symmetric L vs R

    # Temporal (computed across frame pairs)
    brightness_delta: float       # Frame-to-frame brightness change
    is_strobe: bool              # Rapid brightness oscillation detected
    is_blackout: bool            # All regions below threshold
    color_change_rate: float     # Hue shift speed

    # Confidence
    scene_confidence: float       # How confident we are this is a stage view
```

**Extraction Algorithm**:
```python
import cv2
import numpy as np

def extract_lighting(frame: np.ndarray, prev_frame: np.ndarray | None) -> VideoLightingFrame:
    """Extract lighting features from a single video frame.

    Args:
        frame: BGR image (H, W, 3)
        prev_frame: Previous frame for temporal features (None for first frame)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Mask out very dark areas (< 15% brightness) — these are unlit
    bright_mask = v > 38  # 15% of 255

    # Overall brightness
    overall_brightness = float(np.mean(v)) / 255.0
    brightness_variance = float(np.std(v.astype(float))) / 255.0

    # Dominant color — histogram of hue channel in bright regions
    if bright_mask.any():
        hue_hist = cv2.calcHist([h], [0], bright_mask.astype(np.uint8) * 255, [36], [0, 180])
        dominant_hue = float(np.argmax(hue_hist)) * 10.0  # Convert to 0-360
        dominant_saturation = float(np.mean(s[bright_mask])) / 255.0
    else:
        dominant_hue = 0.0
        dominant_saturation = 0.0

    # Spatial distribution — divide frame into regions
    h_frame, w_frame = v.shape
    third_w = w_frame // 3
    half_h = h_frame // 2

    left_brightness = float(np.mean(v[:, :third_w])) / 255.0
    center_brightness = float(np.mean(v[:, third_w:2*third_w])) / 255.0
    right_brightness = float(np.mean(v[:, 2*third_w:])) / 255.0
    top_brightness = float(np.mean(v[:half_h, :])) / 255.0
    bottom_brightness = float(np.mean(v[half_h:, :])) / 255.0

    spatial_symmetry = 1.0 - abs(left_brightness - right_brightness)

    # Temporal features
    brightness_delta = 0.0
    is_strobe = False
    if prev_frame is not None:
        prev_v = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        brightness_delta = (float(np.mean(v)) - float(np.mean(prev_v))) / 255.0
        is_strobe = abs(brightness_delta) > 0.3  # Rapid change = strobe

    is_blackout = overall_brightness < 0.05

    # ... build and return VideoLightingFrame
```

**Step 4: Camera Cut Detection**

Detect scene changes to avoid extracting lighting during cuts:
```python
def detect_cuts(frames: list[np.ndarray], threshold: float = 30.0) -> list[int]:
    """Return indices where camera cuts occur."""
    cuts = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        mean_diff = np.mean(diff)
        if mean_diff > threshold:
            cuts.append(i)
    return cuts
```

Mark frames within 3 frames of a cut as unreliable and exclude from training data.

**Step 5: LED Screen Compensation**

Large LED screens on stage contaminate color readings. Strategy:
1. Detect high-saturation rectangular regions with text/graphics patterns
2. Mask these regions before color extraction
3. Use `scene_confidence` to downweight frames with large LED screen coverage

### 3.4 Audio-Visual Alignment

**Audio Extraction**: yt-dlp extracts audio to WAV. Run through LUMINA's existing pipeline:

```python
# For each video, generate complete MusicState timeline
from lumina.audio.beat_detector import BeatDetector
from lumina.audio.energy_tracker import EnergyTracker
# ... all analyzers

def analyze_video_audio(audio_path: Path, sr: int = 44100, fps: int = 10) -> list[MusicState]:
    """Run full LUMINA audio analysis on extracted video audio."""
    # Same pipeline as app.py _run_file_mode, but at 10fps to match video extraction rate
    ...
```

**Alignment**: Audio and video are inherently synced (same source file). The main concern is:
- A/V sync offset in the original video (usually <100ms, negligible at 10fps)
- Frame drops or variable frame rate — use FFmpeg's constant-rate extraction

**Training Pair Creation**:
For each 100ms timestep where we have both a valid `stage_view` frame and a `MusicState`:
```python
@dataclass
class TrainingPair:
    music_state: MusicState         # Audio features at this moment
    lighting_state: VideoLightingFrame  # Visual lighting at this moment
    genre_label: str                # Ground truth genre
    video_id: str                   # Source tracking
    confidence: float               # Scene classification confidence
```

Store as Parquet files for efficient training data loading.

### 3.5 ML Model Architecture

**Approach**: Genre-conditioned sequence-to-sequence model.

**Why sequence models**: Lighting is inherently temporal. A blackout is only dramatic because of what came before. A build means nothing without the eventual drop. The model must see context.

**Architecture: Temporal Fusion Transformer (TFT) variant**

```
Input (per timestep):
  - MusicState features (15-20 floats): energy, beat_phase, bar_phase, spectral_centroid,
    sub_bass, vocal_energy, drop_probability, is_beat, is_downbeat, energy_derivative, ...
  - Genre embedding (8-dim): learned embedding from genre_profile label
  - Segment embedding (8-dim): learned embedding from segment label
  - Positional encoding: bar_phase and beat_phase

Context window: 4 seconds (40 frames at 10fps)

Architecture:
  Encoder:
    - Input projection: Linear(~36 -> 128)
    - 4x Transformer encoder layers (d_model=128, nhead=8, dim_ff=256)
    - Causal attention mask (model only sees past + current, not future)

  Decoder heads (multi-task):
    - Color head: Linear(128 -> 6) -> sigmoid
      Outputs: dominant_hue (0-1 scaled), dominant_saturation, secondary_hue,
               color_diversity, color_temperature, overall_brightness
    - Spatial head: Linear(128 -> 5) -> sigmoid
      Outputs: left_brightness, center_brightness, right_brightness,
               spatial_symmetry, brightness_variance
    - Effect head: Linear(128 -> 3) -> sigmoid
      Outputs: strobe_probability, blackout_probability, brightness_delta_magnitude

Total parameters: ~500K (tiny — runs at 60fps easily on both RTX 4070 and Jetson)
```

**Why NOT predict FixtureCommand directly**: The video doesn't tell us per-fixture commands. It tells us the *overall visual feel* — dominant color, brightness distribution, strobes. The model predicts a high-level "lighting intent" that gets translated to FixtureCommand by a lightweight mapping layer.

**Mapping: Model Output -> FixtureCommand**

```python
@dataclass
class LightingIntent:
    """High-level lighting intent predicted by ML model."""
    dominant_color: tuple[float, float, float]  # HSV
    secondary_color: tuple[float, float, float]  # HSV
    overall_brightness: float
    color_diversity: float
    spatial_distribution: tuple[float, float, float]  # left, center, right
    spatial_symmetry: float
    strobe_active: bool
    strobe_intensity: float
    blackout: bool

def intent_to_commands(
    intent: LightingIntent,
    fixture_map: FixtureMap,
) -> list[FixtureCommand]:
    """Convert ML model's lighting intent to per-fixture commands."""
    # Map dominant_color -> par RGBW values
    # Map spatial_distribution -> per-fixture dimmer levels based on position
    # Map strobe_active -> strobe fixture commands
    # Map blackout -> all zeros
    ...
```

**Loss Functions**:
```python
# Multi-task loss with genre-aware weighting
loss = (
    w_color * huber_loss(pred_color, target_color)       # Smooth L1 for color
    + w_spatial * mse_loss(pred_spatial, target_spatial)   # MSE for spatial
    + w_strobe * bce_loss(pred_strobe, target_strobe)      # BCE for strobe detection
    + w_blackout * bce_loss(pred_blackout, target_blackout)
    + w_temporal * temporal_consistency_loss(pred_sequence)  # Penalize jitter
)

# Temporal consistency: penalize frame-to-frame output changes that are
# larger than the input changes warrant
def temporal_consistency_loss(predictions: torch.Tensor) -> torch.Tensor:
    diffs = predictions[:, 1:] - predictions[:, :-1]
    return torch.mean(diffs ** 2)
```

**Training Strategy**:
1. Pre-process all videos → Parquet training pairs
2. Train with AdamW, lr=1e-4, batch_size=32 (sequences of 40 frames)
3. Genre-stratified splits: 80% train, 10% val, 10% test
4. Train for ~50 epochs (small model + small dataset = fast convergence)
5. Estimated training time: ~2-4 hours on RTX 4070

### 3.6 Integration: Hybrid Blending

The ML model's output gets blended with the rule-based profile at a configurable weight:

```python
class HybridLightingEngine:
    """Blends ML model predictions with rule-based profile output."""

    def __init__(
        self,
        rule_engine: LightingEngine,
        ml_model: LightingMLModel,
        ml_weight: float = 0.3,  # Start conservative
    ):
        self._rule_engine = rule_engine
        self._ml_model = ml_model
        self._ml_weight = ml_weight

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        # Get rule-based commands
        rule_commands = self._rule_engine.generate(state)

        # Get ML intent and convert to commands
        intent = self._ml_model.predict(state)
        ml_commands = intent_to_commands(intent, self._rule_engine.fixture_map)

        # Blend per-fixture
        blended = []
        for rule_cmd, ml_cmd in zip(rule_commands, ml_commands):
            blended.append(blend_commands(rule_cmd, ml_cmd, self._ml_weight))

        return blended

    def set_ml_weight(self, weight: float) -> None:
        """Adjust ML influence. 0.0 = pure rules, 1.0 = pure ML."""
        self._ml_weight = max(0.0, min(1.0, weight))
```

**Confidence-based weight adjustment**: When the model is uncertain (low softmax confidence or high output variance), automatically reduce `ml_weight` toward 0 for that frame.

**Fallback**: If the ML model produces outputs outside expected ranges (e.g., NaN, all zeros for >2 seconds), the engine automatically falls back to pure rule-based for 5 seconds before retrying.

### 3.7 Infrastructure & Dependencies

**New Python packages needed**:
```toml
# In pyproject.toml [project.optional-dependencies]
ml_training = [
    "yt-dlp>=2024.1",           # Video downloading
    "opencv-python>=4.9",       # Video frame processing
    "transformers>=4.37",       # CLIP for scene classification
    "pyarrow>=15.0",            # Parquet I/O
    "pandas>=2.2",              # Data manipulation
    "wandb>=0.16",              # Experiment tracking (optional)
    "tqdm>=4.66",               # Progress bars
]
```

**GPU Memory Budget (RTX 4070, 12GB)**:
- CLIP scene classifier: ~400MB
- LUMINA audio pipeline (demucs): ~2GB
- Training model: ~200MB (tiny)
- Training data batch: ~1GB
- Total during training: ~4GB — plenty of headroom

**Jetson Orin Nano (8GB unified)**:
- Inference only — no CLIP or training needed
- Model inference: ~200MB
- Audio pipeline: ~2GB
- Total: ~2.5GB — fits easily

### 3.8 Implementation Phases

**Phase A: Data Collection (1-2 weeks)**
1. Build `lumina/ml/data/downloader.py` — yt-dlp wrapper with search + download + metadata
2. Build `lumina/ml/data/catalog.py` — catalog management, quality scoring
3. Collect 50-100 hours across 8 genres
4. Manual quality review: tag `lighting_visibility`, `camera_type`, `has_led_screens`

**Phase B: Feature Extraction (1-2 weeks)**
5. Build `lumina/ml/video/scene_classifier.py` — CLIP-based stage/crowd/screen filter
6. Build `lumina/ml/video/lighting_extractor.py` — per-frame lighting feature extraction
7. Build `lumina/ml/video/cut_detector.py` — camera cut detection
8. Build `lumina/ml/audio/batch_analyzer.py` — run LUMINA audio pipeline on all videos
9. Build `lumina/ml/data/aligner.py` — merge audio + video features into training pairs

**Phase C: Model Training (1-2 weeks)**
10. Build `lumina/ml/model/architecture.py` — transformer encoder + multi-task heads
11. Build `lumina/ml/model/dataset.py` — PyTorch Dataset from Parquet files
12. Build `lumina/ml/model/train.py` — training loop with W&B logging
13. Train, evaluate, iterate

**Phase D: Integration (1 week)**
14. Build `lumina/ml/model/inference.py` — real-time inference wrapper
15. Build `lumina/ml/integration/hybrid_engine.py` — hybrid blending with rule-based
16. Add ML weight slider to simulator UI
17. A/B test: rule-only vs hybrid on known tracks

### 3.9 Repo Structure Addition

```
lumina/
  ml/                           # New ML training pipeline
    __init__.py
    data/
      __init__.py
      downloader.py             # yt-dlp wrapper
      catalog.py                # Video catalog management
      aligner.py                # Audio-visual alignment
    video/
      __init__.py
      scene_classifier.py       # CLIP stage/crowd filter
      lighting_extractor.py     # Per-frame lighting extraction
      cut_detector.py           # Camera cut detection
    audio/
      __init__.py
      batch_analyzer.py         # Batch audio analysis
    model/
      __init__.py
      architecture.py           # Transformer model
      dataset.py                # PyTorch Dataset
      train.py                  # Training loop
      inference.py              # Real-time inference
    integration/
      __init__.py
      hybrid_engine.py          # Blending ML + rules
      intent_mapper.py          # LightingIntent -> FixtureCommand
```

---

## 4. Known Bugs & Technical Debt

| ID | Severity | Description | File | Status |
|---|---|---|---|---|
| B1 | Critical | `genre_override`, `intensity`, `manual_effect` WebSocket messages not handled | `lumina/app.py` | **Fixed** — added handlers for all 4 message types + `_apply_effects()` + `_generate_manual_effect()` |
| B2 | High | `uk_bass` profile not implemented — falls back to generic | `lumina/lighting/profiles/` | **Fixed** — `uk_bass.py` created, registered in `engine.py` |
| B3 | Medium | Mode radio buttons have no onChange handler | `simulator/ControlPanel.tsx` | **Fixed** — added onChange handlers that send transport messages |
| B4 | Low | `asyncio.get_event_loop()` deprecated in Python 3.12+ | `lumina/app.py:189` | **Fixed** — changed to `asyncio.get_running_loop()` |
| B5 | Low | New MusicState fields not sent over WebSocket | `lumina/web/server.py` | **Fixed** — added layer_count, layer_mask, motif_id, motif_repetition, notes_per_beat, note_pattern_phase, headroom to both server.py and fixtures.ts |
| B6 | Info | No profile blending — single profile selection only | `lumina/lighting/engine.py` | By design (Phase 2) |
| B7 | Info | `docs/`, `scripts/`, `firmware/` directories don't exist yet | Root | **Fixed** — created docs/ and scripts/ with all referenced files |

---

## 5. Design Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-11 | ML training data from YouTube scraping | Largest free dataset of concert lighting. For personal ML training use only. |
| 2026-03-11 | Hybrid blending (ML + rules) | Safe default — rule-based provides reliable baseline, ML adds nuance. Can tune weight. |
| 2026-03-11 | 50-100 hours initial dataset | Proof of concept scale. Can expand to 500+ hours if results are promising. |
| 2026-03-11 | 10fps video analysis (not 60fps) | Lighting changes visible at 10fps. 6x compute savings. Upscale to 60fps at inference via interpolation. |
| 2026-03-11 | Predict LightingIntent, not FixtureCommand | Video shows overall lighting feel, not per-fixture state. Intent is a better abstraction. |
| 2026-03-11 | Small transformer (~500K params) | Must run at 60fps on Jetson (8GB). Small model trains fast on limited data. |
| 2026-03-11 | CLIP for scene classification | Zero-shot, no training needed. Good enough for stage vs crowd vs LED screen. |
