"""
LUMINA — EDM Structural Analysis Pass

Replaces generic segment labels (verse/chorus/bridge) with EDM-specific
structural labels when genre_family == "electronic":
  - build:     Rising energy ramp, typically 8-32 bars before a drop
  - drop:      Energy spike following a build — the payoff moment
  - groove:    Sustained high-energy plateau after a drop
  - breakdown: Energy valley, low intensity, anticipation builder
  - intro:     Opening bars before first significant energy event
  - outro:     Closing bars after last significant energy event

The core insight: EDM structure is defined by energy contours, not by
melodic/harmonic content. A build and a groove can share identical chords
but have completely different energy trajectories. The default segment
classifier (trained on verse/chorus patterns) fails here because it
looks at timbre similarity, not energy derivatives.

Approach:
  1. Compute bar-level energy from the frame-level energy envelope
  2. Smooth the energy curve to remove per-beat variance
  3. Compute energy derivative (rate of change per bar)
  4. Classify bars into structural roles based on energy level + derivative
  5. Merge adjacent same-label bars into segments
  6. Apply minimum duration constraints and fix edge cases

Integration:
  Call edm_structure_pass() after beat/bar detection and energy extraction.
  Pass genre_family from the genre classifier. If genre_family != "electronic",
  this module returns None and the default analyzer labels are used.

Expected inputs (from existing pipeline):
  - energy_envelope: np.ndarray, frame-level energy at ~60fps (0.0 to 1.0)
  - beat_times: np.ndarray, beat timestamps in seconds
  - bar_times: np.ndarray, bar (downbeat) timestamps in seconds
  - sr: int, audio sample rate
  - hop_length: int, hop length used for energy computation
  - genre_family: str, from genre classifier ("electronic", "hiphop", "hybrid")
  - drop_probability: np.ndarray (optional), per-bar drop probability from predictor

Output:
  List[StructuralSegment] — labeled segments with bar-aligned boundaries
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EDMSection(Enum):
    """EDM structural section types."""
    INTRO = "intro"
    BUILD = "build"
    DROP = "drop"
    GROOVE = "groove"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


@dataclass
class StructuralSegment:
    """A labeled structural segment with bar-aligned boundaries."""
    label: str                  # EDM section type (from EDMSection)
    start_time: float           # Start time in seconds
    end_time: float             # End time in seconds
    start_bar: int              # Start bar index (0-based)
    end_bar: int                # End bar index (inclusive)
    energy_mean: float          # Mean energy across this segment
    energy_slope: float         # Energy derivative (+ = rising, - = falling)
    confidence: float           # Classification confidence 0.0 to 1.0
    bar_count: int = 0          # Number of bars in this segment

    def __post_init__(self):
        self.bar_count = self.end_bar - self.start_bar + 1

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self):
        return (f"[{self.start_time:6.2f}s - {self.end_time:6.2f}s] "
                f"{self.label:>10s} ({self.bar_count:2d} bars, "
                f"E={self.energy_mean:.2f}, slope={self.energy_slope:+.3f})")


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class EDMStructureConfig:
    """
    Tunable thresholds for EDM structure detection.
    Defaults calibrated for 126-142 BPM four-on-the-floor.
    """
    # Energy level thresholds (0.0 to 1.0 scale)
    high_energy_threshold: float = 0.65     # Above this = "high energy" (drop/groove)
    low_energy_threshold: float = 0.30      # Below this = "low energy" (breakdown)
    drop_spike_threshold: float = 0.15      # Min energy jump in 1-2 bars to qualify as drop

    # Energy derivative thresholds (per bar)
    build_slope_threshold: float = 0.015    # Positive slope needed to classify as build
    decay_slope_threshold: float = -0.02    # Negative slope for breakdown entry

    # Duration constraints (in bars)
    min_build_bars: int = 4                 # Minimum bars for a valid build
    min_drop_bars: int = 2                  # Minimum bars for a valid drop
    min_groove_bars: int = 4                # Minimum bars for a valid groove
    min_breakdown_bars: int = 4             # Minimum bars for a valid breakdown
    min_intro_bars: int = 2                 # Minimum bars for intro
    min_outro_bars: int = 2                 # Minimum bars for outro

    # Smoothing
    energy_smoothing_window: int = 4        # Bars to smooth over (median filter)
    derivative_window: int = 4              # Bars for derivative computation

    # Drop detection
    use_drop_probability: bool = True       # Use drop_probability signal if available
    drop_prob_threshold: float = 0.6        # Minimum drop_probability to boost detection

    # Intro/outro detection
    intro_max_energy: float = 0.40          # Max energy for intro classification
    outro_max_energy: float = 0.40          # Max energy for outro classification

    # Subgenre adjustments
    # Trance: longer builds, more gradual, lower drop spike threshold
    # Dubstep/UK bass: shorter builds, more extreme spikes
    # Electro house: standard
    subgenre_presets: dict = field(default_factory=lambda: {
        "festival_edm": {},  # use defaults
        "trance": {
            "min_build_bars": 8,
            "build_slope_threshold": 0.010,
            "drop_spike_threshold": 0.12,
        },
        "uk_bass": {
            "min_build_bars": 2,
            "drop_spike_threshold": 0.20,
            "high_energy_threshold": 0.60,
        },
        "theatrical_electronic": {
            "build_slope_threshold": 0.010,
            "low_energy_threshold": 0.25,
            "min_breakdown_bars": 8,
        },
    })

    def apply_subgenre(self, profile: str):
        """Override thresholds based on subgenre profile."""
        preset = self.subgenre_presets.get(profile, {})
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ---------------------------------------------------------------------------
#  Core Analysis Functions
# ---------------------------------------------------------------------------

def compute_bar_energy(
    energy_envelope: np.ndarray,
    bar_times: np.ndarray,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Aggregate frame-level energy into per-bar energy values.

    Takes the RMS or energy envelope computed at ~60fps and averages it
    over each bar interval. Returns one energy value per bar.

    Args:
        energy_envelope: Frame-level energy, shape (n_frames,), range [0, 1]
        bar_times: Bar (downbeat) timestamps in seconds, shape (n_bars,)
        sr: Audio sample rate
        hop_length: Hop length used to compute energy_envelope

    Returns:
        bar_energy: Per-bar mean energy, shape (n_bars - 1,)
    """
    frame_times = np.arange(len(energy_envelope)) * hop_length / sr
    n_bars = len(bar_times) - 1
    bar_energy = np.zeros(n_bars)

    for i in range(n_bars):
        start_t = bar_times[i]
        end_t = bar_times[i + 1]
        mask = (frame_times >= start_t) & (frame_times < end_t)
        if mask.any():
            bar_energy[i] = np.mean(energy_envelope[mask])
        elif i > 0:
            bar_energy[i] = bar_energy[i - 1]  # carry forward if no frames

    return bar_energy


def smooth_energy(bar_energy: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth bar-level energy using a median filter to remove per-beat noise
    while preserving sharp transitions (drops).

    Median filter is chosen over Gaussian/moving average because drops are
    step functions — we want to preserve the edge, not blur it.
    """
    if window < 2 or len(bar_energy) < window:
        return bar_energy.copy()

    smoothed = np.copy(bar_energy)
    half = window // 2

    for i in range(len(bar_energy)):
        start = max(0, i - half)
        end = min(len(bar_energy), i + half + 1)
        smoothed[i] = np.median(bar_energy[start:end])

    return smoothed


def compute_energy_derivative(
    bar_energy: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """
    Compute per-bar energy derivative using linear regression over a
    sliding window. More robust than simple differencing.

    Returns slope per bar (positive = energy rising, negative = falling).
    """
    n = len(bar_energy)
    derivative = np.zeros(n)
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        segment = bar_energy[start:end]

        if len(segment) < 2:
            derivative[i] = 0.0
            continue

        # Linear regression: slope of energy over the window
        x = np.arange(len(segment), dtype=float)
        x -= x.mean()
        y = segment - segment.mean()
        denom = np.sum(x ** 2)
        derivative[i] = np.sum(x * y) / denom if denom > 0 else 0.0

    return derivative


def detect_drop_points(
    bar_energy_smoothed: np.ndarray,
    bar_energy_raw: np.ndarray,
    derivative: np.ndarray,
    config: EDMStructureConfig,
    drop_probability: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Detect drop bar indices — moments where energy spikes after a ramp.

    Uses RAW (unsmoothed) energy for spike detection because the median
    filter blurs the exact drop edge. Uses smoothed derivative for
    build context.

    A drop is defined as:
      1. Raw energy at bar[i] is significantly higher than bar[i-1] or bar[i-2]
      2. The preceding bars had positive (rising) derivative (build context)
      3. Energy at bar[i] exceeds high_energy_threshold
      4. (Optional) drop_probability signal is above threshold

    Returns sorted list of bar indices identified as drop entry points.
    """
    drops = []
    n = len(bar_energy_raw)

    for i in range(2, n):
        # Condition 1: Energy spike (use RAW energy to preserve sharp edges)
        jump_1 = bar_energy_raw[i] - bar_energy_raw[i - 1]
        jump_2 = bar_energy_raw[i] - bar_energy_raw[max(0, i - 2)]
        spike = max(jump_1, jump_2 / 2)  # /2 because spread over 2 bars

        if spike < config.drop_spike_threshold:
            continue

        # Condition 2: Preceding bars were rising (build context)
        # Use smoothed derivative for more stable trend detection
        lookback = min(i, 6)
        preceding_slope = np.mean(derivative[max(0, i - lookback):i])
        if preceding_slope < config.build_slope_threshold * 0.3:
            # Still allow if there's a clear energy ramp in raw signal
            raw_ramp = bar_energy_raw[i - 1] - bar_energy_raw[max(0, i - lookback)]
            if raw_ramp < 0.1:
                continue

        # Condition 3: Energy is now high
        if bar_energy_raw[i] < config.high_energy_threshold * 0.8:
            continue

        # Condition 4 (optional): Drop probability signal
        confidence = 0.7
        if (drop_probability is not None and config.use_drop_probability
                and i < len(drop_probability)):
            if drop_probability[i] > config.drop_prob_threshold:
                confidence = min(0.95, confidence + 0.2)
            elif drop_probability[i] < config.drop_prob_threshold * 0.5:
                confidence *= 0.8

        # Suppress duplicates within 4 bars
        if drops and (i - drops[-1]) < 4:
            # Keep the higher-energy one
            if bar_energy_raw[i] > bar_energy_raw[drops[-1]]:
                drops[-1] = i
            continue

        drops.append(i)

    return drops


# ---------------------------------------------------------------------------
#  Bar-Level Classification
# ---------------------------------------------------------------------------

def classify_bars(
    bar_energy: np.ndarray,
    derivative: np.ndarray,
    drop_bars: List[int],
    config: EDMStructureConfig,
) -> List[str]:
    """
    Assign a structural label to each bar based on energy level, derivative,
    and detected drop points.

    Classification priority (highest to lowest):
      1. Drop: bars at detected drop points + immediate continuation
      2. Build: bars with sustained positive derivative preceding a drop
      3. Groove: high energy bars after a drop that aren't building
      4. Breakdown: low energy bars with flat/negative derivative
      5. Intro/Outro: low energy at start/end of track

    Returns list of label strings, one per bar.
    """
    n = len(bar_energy)
    labels = ["unknown"] * n

    # Pass 1: Mark drops and their immediate continuation
    # A drop typically sustains for 2-4 bars at peak energy
    for d in drop_bars:
        labels[d] = "drop"
        # Extend drop forward while energy stays near peak
        peak_e = bar_energy[d]
        for j in range(d + 1, min(d + config.min_drop_bars + 2, n)):
            if bar_energy[j] >= peak_e * 0.85:
                labels[j] = "drop"
            else:
                break

    # Pass 2: Mark builds (rising energy leading to drops)
    for d in drop_bars:
        # Walk backwards from drop to find build start
        build_end = d - 1
        if build_end < 0:
            continue

        build_start = build_end
        for j in range(build_end, max(0, d - 33), -1):  # max 32-bar build
            if labels[j] != "unknown":
                break
            if derivative[j] > config.build_slope_threshold * 0.3:
                build_start = j
            elif bar_energy[j] < config.low_energy_threshold:
                break
            else:
                # Allow a couple flat bars within a build (brief plateaus)
                if j < build_start - 2:
                    break
                build_start = j

        # Enforce minimum build length
        if (build_end - build_start + 1) >= config.min_build_bars:
            for j in range(build_start, build_end + 1):
                if labels[j] == "unknown":
                    labels[j] = "build"

    # Pass 3: Mark grooves (sustained high energy, not a build or drop)
    for i in range(n):
        if labels[i] != "unknown":
            continue
        if bar_energy[i] >= config.high_energy_threshold:
            # Allow slight negative derivative — energy often settles after a drop
            if derivative[i] < config.build_slope_threshold:
                labels[i] = "groove"

    # Pass 4: Mark breakdowns (low energy valleys)
    for i in range(n):
        if labels[i] != "unknown":
            continue
        if bar_energy[i] <= config.low_energy_threshold:
            labels[i] = "breakdown"

    # Pass 5: Classify remaining unknown bars by context
    for i in range(n):
        if labels[i] != "unknown":
            continue

        # Mid-energy, flat derivative — could be groove at lower intensity
        # or transition. Use neighboring context.
        if bar_energy[i] >= config.high_energy_threshold * 0.75:
            labels[i] = "groove"
        elif derivative[i] > config.build_slope_threshold * 0.5:
            labels[i] = "build"
        elif derivative[i] < config.decay_slope_threshold * 0.5:
            labels[i] = "breakdown"
        else:
            # Default to groove if energy is moderate, breakdown if low
            if bar_energy[i] > (config.high_energy_threshold + config.low_energy_threshold) / 2:
                labels[i] = "groove"
            else:
                labels[i] = "breakdown"

    # Pass 6: Intro and outro detection
    # Intro ends when energy starts rising consistently (build begins),
    # NOT just when it crosses an absolute threshold. This prevents the
    # early build ramp from being misclassified as intro.
    first_active = 0
    for i in range(n):
        # End intro if: energy is above threshold, OR derivative turns
        # consistently positive (build is starting even if energy is still low)
        if bar_energy[i] > config.intro_max_energy:
            first_active = i
            break
        if i >= 2 and derivative[i] > config.build_slope_threshold:
            # Check that this isn't a one-bar blip
            if i + 1 < n and derivative[i + 1] > config.build_slope_threshold * 0.5:
                first_active = i
                break
    if first_active >= config.min_intro_bars:
        for i in range(first_active):
            labels[i] = "intro"

    # Outro: low-energy bars at the end after the last high-energy event
    # Also check that derivative is negative or flat (energy fading out)
    last_active = n - 1
    for i in range(n - 1, -1, -1):
        if bar_energy[i] > config.outro_max_energy:
            last_active = i
            break
    if (n - 1 - last_active) >= config.min_outro_bars:
        for i in range(last_active + 1, n):
            labels[i] = "outro"

    return labels


# ---------------------------------------------------------------------------
#  Segment Merging & Cleanup
# ---------------------------------------------------------------------------

def merge_bar_labels(
    bar_labels: List[str],
    bar_times: np.ndarray,
    bar_energy: np.ndarray,
    derivative: np.ndarray,
    config: EDMStructureConfig,
) -> List[StructuralSegment]:
    """
    Merge consecutive bars with the same label into StructuralSegments.
    Apply minimum duration constraints and fix short orphan segments.
    """
    if not bar_labels:
        return []

    n = len(bar_labels)
    segments: List[StructuralSegment] = []

    # First pass: merge consecutive same-label bars
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or bar_labels[i] != bar_labels[seg_start]:
            label = bar_labels[seg_start]
            start_t = bar_times[seg_start]
            end_t = bar_times[min(i, len(bar_times) - 1)]
            seg_energy = bar_energy[seg_start:i]
            seg_deriv = derivative[seg_start:i]

            segments.append(StructuralSegment(
                label=label,
                start_time=start_t,
                end_time=end_t,
                start_bar=seg_start,
                end_bar=i - 1,
                energy_mean=float(np.mean(seg_energy)),
                energy_slope=float(np.mean(seg_deriv)),
                confidence=0.8,  # base confidence, refined below
            ))
            if i < n:
                seg_start = i

    # Second pass: absorb short segments into neighbors
    segments = _absorb_short_segments(segments, config)

    # Third pass: refine confidence scores
    for seg in segments:
        seg.confidence = _compute_confidence(seg, config)

    return segments


def _absorb_short_segments(
    segments: List[StructuralSegment],
    config: EDMStructureConfig,
) -> List[StructuralSegment]:
    """
    Absorb segments shorter than their minimum duration into adjacent segments.
    Iterate until stable.
    """
    min_bars = {
        "build": config.min_build_bars,
        "drop": config.min_drop_bars,
        "groove": config.min_groove_bars,
        "breakdown": config.min_breakdown_bars,
        "intro": config.min_intro_bars,
        "outro": config.min_outro_bars,
    }

    changed = True
    max_iterations = 10
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        new_segments = []

        for i, seg in enumerate(segments):
            minimum = min_bars.get(seg.label, 2)
            if seg.bar_count < minimum and len(segments) > 1:
                # Absorb into the neighbor with closer energy
                prev_seg = new_segments[-1] if new_segments else None
                next_seg = segments[i + 1] if i + 1 < len(segments) else None

                if prev_seg and next_seg:
                    prev_diff = abs(seg.energy_mean - prev_seg.energy_mean)
                    next_diff = abs(seg.energy_mean - next_seg.energy_mean)
                    target = prev_seg if prev_diff <= next_diff else None
                    if target is None:
                        # Merge forward — extend next segment's start
                        next_seg.start_time = seg.start_time
                        next_seg.start_bar = seg.start_bar
                        next_seg.bar_count = next_seg.end_bar - next_seg.start_bar + 1
                        changed = True
                        continue
                    else:
                        target.end_time = seg.end_time
                        target.end_bar = seg.end_bar
                        target.bar_count = target.end_bar - target.start_bar + 1
                        changed = True
                        continue
                elif prev_seg:
                    prev_seg.end_time = seg.end_time
                    prev_seg.end_bar = seg.end_bar
                    prev_seg.bar_count = prev_seg.end_bar - prev_seg.start_bar + 1
                    changed = True
                    continue
                elif next_seg:
                    next_seg.start_time = seg.start_time
                    next_seg.start_bar = seg.start_bar
                    next_seg.bar_count = next_seg.end_bar - next_seg.start_bar + 1
                    changed = True
                    continue

            new_segments.append(seg)

        segments = new_segments

    # Re-merge any adjacent segments that now share a label
    if segments:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.label == merged[-1].label:
                merged[-1].end_time = seg.end_time
                merged[-1].end_bar = seg.end_bar
                merged[-1].bar_count = merged[-1].end_bar - merged[-1].start_bar + 1
                merged[-1].energy_mean = (merged[-1].energy_mean + seg.energy_mean) / 2
                merged[-1].energy_slope = (merged[-1].energy_slope + seg.energy_slope) / 2
            else:
                merged.append(seg)
        segments = merged

    return segments


def _compute_confidence(seg: StructuralSegment, config: EDMStructureConfig) -> float:
    """Heuristic confidence based on how well the segment fits its label."""
    c = 0.5  # base

    if seg.label == "drop":
        # High energy + came after rising energy = confident drop
        if seg.energy_mean > config.high_energy_threshold:
            c += 0.3
        if seg.bar_count >= config.min_drop_bars:
            c += 0.1

    elif seg.label == "build":
        # Positive slope + increasing bars = confident build
        if seg.energy_slope > config.build_slope_threshold:
            c += 0.3
        if seg.bar_count >= 8:
            c += 0.1

    elif seg.label == "groove":
        if seg.energy_mean > config.high_energy_threshold:
            c += 0.2
        if abs(seg.energy_slope) < config.build_slope_threshold:
            c += 0.2

    elif seg.label == "breakdown":
        if seg.energy_mean < config.low_energy_threshold:
            c += 0.3
        if seg.bar_count >= config.min_breakdown_bars:
            c += 0.1

    elif seg.label in ("intro", "outro"):
        if seg.energy_mean < config.intro_max_energy:
            c += 0.3

    return min(c, 0.95)


# ---------------------------------------------------------------------------
#  Main Entry Point
# ---------------------------------------------------------------------------

def edm_structure_pass(
    energy_envelope: np.ndarray,
    beat_times: np.ndarray,
    bar_times: np.ndarray,
    sr: int,
    hop_length: int,
    genre_family: str,
    genre_profile: str = "",
    drop_probability: Optional[np.ndarray] = None,
    config: Optional[EDMStructureConfig] = None,
) -> Optional[List[StructuralSegment]]:
    """
    EDM-specific structural analysis pass.

    Call this after beat/bar detection and energy extraction. If genre_family
    is not "electronic", returns None (fall through to default analyzer).

    Args:
        energy_envelope: Frame-level energy at ~60fps, shape (n_frames,), [0,1]
        beat_times: Beat timestamps in seconds
        bar_times: Bar (downbeat) timestamps in seconds (n_bars + 1 values,
                   where bar_times[i] to bar_times[i+1] defines bar i)
        sr: Audio sample rate
        hop_length: Hop length used for energy_envelope
        genre_family: "electronic", "hiphop", or "hybrid"
        genre_profile: Specific profile for subgenre tuning (e.g. "festival_edm",
                       "trance", "uk_bass", "theatrical_electronic")
        drop_probability: Optional per-bar drop probability from predictor
        config: Override config (default uses EDMStructureConfig())

    Returns:
        List[StructuralSegment] if genre_family is "electronic" or "hybrid",
        None otherwise (signals caller to use default analyzer).
    """
    if genre_family not in ("electronic", "hybrid"):
        return None

    if config is None:
        config = EDMStructureConfig()

    # Apply subgenre-specific thresholds
    if genre_profile:
        config.apply_subgenre(genre_profile)

    if len(bar_times) < 4:
        # Not enough bars to analyze structure
        return [StructuralSegment(
            label="groove", start_time=bar_times[0], end_time=bar_times[-1],
            start_bar=0, end_bar=max(0, len(bar_times) - 2),
            energy_mean=float(np.mean(energy_envelope)),
            energy_slope=0.0, confidence=0.3,
        )]

    # Step 1: Compute bar-level energy
    bar_energy = compute_bar_energy(energy_envelope, bar_times, sr, hop_length)

    # Step 2: Smooth
    smoothed = smooth_energy(bar_energy, config.energy_smoothing_window)

    # Step 3: Compute derivative
    derivative = compute_energy_derivative(smoothed, config.derivative_window)

    # Step 4: Detect drop points (raw energy for spike detection, smoothed for context)
    drop_bars = detect_drop_points(smoothed, bar_energy, derivative, config, drop_probability)

    # Step 5: Classify each bar
    bar_labels = classify_bars(smoothed, derivative, drop_bars, config)

    # Step 6: Merge into segments
    segments = merge_bar_labels(bar_labels, bar_times, smoothed, derivative, config)

    return segments


# ---------------------------------------------------------------------------
#  Utility: Pretty-Print & Debug
# ---------------------------------------------------------------------------

def print_structure(segments: List[StructuralSegment], title: str = ""):
    """Print structural analysis results in a readable format."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    total_bars = sum(s.bar_count for s in segments)
    total_duration = segments[-1].end_time - segments[0].start_time if segments else 0

    for seg in segments:
        bar_pct = seg.bar_count / total_bars * 100 if total_bars > 0 else 0
        print(f"  {seg}  [{bar_pct:4.1f}%]  conf={seg.confidence:.2f}")

    print(f"\n  Total: {len(segments)} segments, {total_bars} bars, {total_duration:.1f}s")

    # Section distribution
    section_bars = {}
    for seg in segments:
        section_bars[seg.label] = section_bars.get(seg.label, 0) + seg.bar_count
    print(f"  Distribution: {', '.join(f'{k}={v}' for k, v in sorted(section_bars.items()))}")


def structure_to_timeline(
    segments: List[StructuralSegment],
    bar_times: np.ndarray,
) -> List[dict]:
    """
    Convert structural segments into a per-bar timeline suitable for
    the lighting engine.

    Returns list of dicts, one per bar:
        {
            "bar": int,
            "time": float,
            "label": str,
            "energy": float,
            "slope": float,
            "bars_into_section": int,      # how far into current section
            "bars_remaining_section": int,  # how many bars left in section
            "section_progress": float,      # 0.0 to 1.0 through current section
            "next_section": str | None,     # what comes next (for anticipation)
        }

    The lighting engine uses bars_remaining_section and next_section to
    implement anticipation — e.g., ramping strobe frequency during the
    last 4 bars of a build when next_section is "drop".
    """
    timeline = []
    n_bars = len(bar_times) - 1

    for seg_idx, seg in enumerate(segments):
        next_label = segments[seg_idx + 1].label if seg_idx + 1 < len(segments) else None

        for bar in range(seg.start_bar, seg.end_bar + 1):
            if bar >= n_bars:
                break

            bars_into = bar - seg.start_bar
            bars_remaining = seg.end_bar - bar
            progress = bars_into / max(1, seg.bar_count - 1)

            timeline.append({
                "bar": bar,
                "time": float(bar_times[bar]),
                "label": seg.label,
                "energy": float(seg.energy_mean),
                "slope": float(seg.energy_slope),
                "bars_into_section": bars_into,
                "bars_remaining_section": bars_remaining,
                "section_progress": round(progress, 3),
                "next_section": next_label,
            })

    return timeline


# ---------------------------------------------------------------------------
#  Integration Example (for reference — wire into your pipeline)
# ---------------------------------------------------------------------------

def example_integration():
    """
    Shows how to wire this into the existing LUMINA pipeline.
    NOT meant to be called directly — adapt to your actual code.
    """
    # In your analyze() function or equivalent:
    #
    # from edm_structure import edm_structure_pass, structure_to_timeline
    #
    # # After running beat/bar detection, energy extraction, genre classification:
    # segments = edm_structure_pass(
    #     energy_envelope=self.energy,           # your frame-level energy array
    #     beat_times=self.beat_times,             # from madmom/librosa
    #     bar_times=self.bar_times,               # from downbeat tracker
    #     sr=self.sr,
    #     hop_length=self.hop_length,
    #     genre_family=self.genre_result.family,  # "electronic" / "hiphop" / "hybrid"
    #     genre_profile=self.genre_result.profile, # "festival_edm", "rage_trap", etc.
    #     drop_probability=self.drop_prob,         # from your drop predictor (optional)
    # )
    #
    # if segments is not None:
    #     # EDM pass produced results — use these instead of default labels
    #     self.structural_segments = segments
    #     self.bar_timeline = structure_to_timeline(segments, self.bar_times)
    # else:
    #     # Not electronic — keep default verse/chorus labels
    #     pass
    #
    # # The lighting engine then reads self.bar_timeline to know:
    # #   - Current section type (build/drop/groove/breakdown)
    # #   - How far into the section we are (section_progress)
    # #   - What's coming next (next_section)
    # #   - How many bars until the section changes (bars_remaining_section)
    pass


if __name__ == "__main__":
    # Quick sanity test with synthetic data
    np.random.seed(42)  # reproducible
    print("Running synthetic EDM structure test...")

    # Simulate a typical EDM track: intro → build → drop → groove → breakdown → build → drop → outro
    # 128 BPM, 4/4 time, so 1 bar ≈ 1.875s (4 beats * 60/128)
    bpm = 128
    bar_duration = 4 * 60.0 / bpm
    n_bars = 64  # ~2 min track

    bar_times = np.array([i * bar_duration for i in range(n_bars + 1)])

    # Build synthetic energy contour
    # Bars 0-7:   intro (low, ~0.15)
    # Bars 8-23:  build (ramp 0.2 → 0.7)
    # Bars 24-25: drop (spike to 0.9)
    # Bars 26-39: groove (sustain 0.75-0.85)
    # Bars 40-47: breakdown (drop to 0.2)
    # Bars 48-55: build (ramp 0.25 → 0.75)
    # Bars 56-57: drop (spike to 0.95)
    # Bars 58-61: groove (0.8)
    # Bars 62-63: outro (fade to 0.1)

    bar_energy_truth = np.zeros(n_bars)
    bar_energy_truth[0:8] = np.linspace(0.10, 0.18, 8) + np.random.normal(0, 0.02, 8)
    bar_energy_truth[8:24] = np.linspace(0.20, 0.70, 16) + np.random.normal(0, 0.02, 16)
    bar_energy_truth[24:26] = [0.92, 0.89]
    bar_energy_truth[26:40] = np.linspace(0.82, 0.75, 14) + np.random.normal(0, 0.02, 14)
    bar_energy_truth[40:48] = np.linspace(0.65, 0.18, 8) + np.random.normal(0, 0.02, 8)
    bar_energy_truth[48:56] = np.linspace(0.22, 0.72, 8) + np.random.normal(0, 0.02, 8)
    bar_energy_truth[56:58] = [0.95, 0.93]
    bar_energy_truth[58:62] = [0.83, 0.80, 0.78, 0.76]
    bar_energy_truth[62:64] = [0.12, 0.06]
    bar_energy_truth = np.clip(bar_energy_truth, 0, 1)

    # Expand to frame-level energy (~60fps)
    sr = 22050
    hop_length = 512
    frames_per_bar = int(bar_duration * sr / hop_length)
    energy_envelope = np.zeros(n_bars * frames_per_bar)
    for i in range(n_bars):
        start_frame = i * frames_per_bar
        end_frame = (i + 1) * frames_per_bar
        base = bar_energy_truth[i]
        energy_envelope[start_frame:end_frame] = base + np.random.normal(0, 0.01, frames_per_bar)
    energy_envelope = np.clip(energy_envelope, 0, 1)

    # --- Debug: Run steps manually to inspect ---
    config = EDMStructureConfig()
    bar_energy = compute_bar_energy(energy_envelope, bar_times, sr, hop_length)
    smoothed = smooth_energy(bar_energy, config.energy_smoothing_window)
    derivative = compute_energy_derivative(smoothed, config.derivative_window)

    print("\n  Debug: Energy at expected drop points")
    for d_bar in [24, 56]:
        raw_jump = bar_energy[d_bar] - bar_energy[d_bar - 1]
        raw_jump2 = bar_energy[d_bar] - bar_energy[d_bar - 2]
        sm_jump = smoothed[d_bar] - smoothed[d_bar - 1]
        print(f"    Bar {d_bar}: raw_E={bar_energy[d_bar]:.3f}  "
              f"raw_jump1={raw_jump:+.3f}  raw_jump2={raw_jump2:+.3f}  "
              f"smooth_E={smoothed[d_bar]:.3f}  smooth_jump={sm_jump:+.3f}  "
              f"deriv={derivative[d_bar]:+.3f}")

    drop_bars = detect_drop_points(smoothed, bar_energy, derivative, config)
    print(f"\n  Debug: Detected drop bars: {drop_bars}")

    bar_labels = classify_bars(smoothed, derivative, drop_bars, config)
    print(f"  Debug: Bar labels at drops: ", end="")
    for d_bar in [24, 56]:
        print(f"bar[{d_bar}]={bar_labels[d_bar]}  ", end="")
    print()

    # --- Full pipeline ---
    segments = edm_structure_pass(
        energy_envelope=energy_envelope,
        beat_times=np.linspace(0, bar_times[-1], n_bars * 4 + 1),
        bar_times=bar_times,
        sr=sr,
        hop_length=hop_length,
        genre_family="electronic",
        genre_profile="festival_edm",
    )

    if segments:
        print_structure(segments, "Synthetic EDM Track — Structural Analysis")
        timeline = structure_to_timeline(segments, bar_times)
        print(f"\n  Timeline entries: {len(timeline)}")
        for entry in timeline:
            if entry["bars_remaining_section"] <= 1 and entry["next_section"]:
                print(f"    Bar {entry['bar']:3d} ({entry['time']:6.1f}s): "
                      f"{entry['label']:>10s} → next: {entry['next_section']}")

        # Validate expected structure
        labels_found = set(s.label for s in segments)
        expected = {"intro", "build", "drop", "groove", "breakdown", "outro"}
        missing = expected - labels_found
        if missing:
            print(f"\n  WARNING: Missing expected sections: {missing}")
        else:
            print(f"\n  PASS: All expected section types detected")
    else:
        print("ERROR: No segments returned")
