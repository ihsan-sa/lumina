"""Per-frame lighting feature extraction from concert video frames.

Extracts lighting characteristics from video frames using OpenCV and
numpy. Features include brightness, color, spatial distribution, and
temporal dynamics (strobe detection, blackouts, color change rate).

Designed to process ``stage_view`` frames identified by the scene
classifier. Outputs ``VideoLightingFrame`` dataclasses that become
training targets for the ML lighting model.

LED screen compensation: large high-saturation rectangular regions with
text/graphics patterns are detected and masked before color extraction
to avoid contaminating lighting readings with screen content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Thresholds
_BRIGHT_THRESHOLD = 38  # 15% of 255 — pixels below this are unlit
_BLACKOUT_THRESHOLD = 0.05  # Overall brightness below this = blackout
_STROBE_THRESHOLD = 0.3  # Frame-to-frame brightness delta for strobe
_LED_SCREEN_SAT_MIN = 180  # Minimum saturation for LED screen detection
_LED_SCREEN_AREA_MIN = 0.03  # Minimum contour area ratio to be an LED screen
_LED_SCREEN_AREA_MAX = 0.5  # Maximum area ratio (larger is likely the whole stage)


@dataclass(slots=True)
class VideoLightingFrame:
    """Lighting features extracted from a single video frame.

    Args:
        timestamp: Frame timestamp in seconds.
        overall_brightness: 0-1 mean luminance of stage area.
        brightness_variance: Spatial variance (low = wash, high = spots).
        dominant_hue: 0-360 degrees, dominant color hue.
        dominant_saturation: 0-1 saturation of dominant color.
        secondary_hue: 0-360 degrees, secondary color if bimodal.
        color_temperature: Warm (low) vs cool (high), mapped from hue.
        color_diversity: 0-1 how many distinct colors are visible.
        left_brightness: Brightness of left third of stage.
        center_brightness: Brightness of center third of stage.
        right_brightness: Brightness of right third of stage.
        top_brightness: Brightness of upper half.
        bottom_brightness: Brightness of lower half.
        spatial_symmetry: 0-1 how symmetric left vs right brightness.
        brightness_delta: Frame-to-frame brightness change.
        is_strobe: Rapid brightness oscillation detected.
        is_blackout: All regions below brightness threshold.
        color_change_rate: Hue shift speed (degrees per frame).
        scene_confidence: How confident we are this is a stage view.
    """

    timestamp: float

    # Global metrics
    overall_brightness: float
    brightness_variance: float

    # Color
    dominant_hue: float
    dominant_saturation: float
    secondary_hue: float
    color_temperature: float
    color_diversity: float

    # Spatial
    left_brightness: float
    center_brightness: float
    right_brightness: float
    top_brightness: float
    bottom_brightness: float
    spatial_symmetry: float

    # Temporal (computed across frame pairs)
    brightness_delta: float
    is_strobe: bool
    is_blackout: bool
    color_change_rate: float

    # Confidence
    scene_confidence: float


def _hue_to_color_temperature(hue: float) -> float:
    """Map hue angle to a warm-cool temperature value.

    Warm colors (reds, oranges, yellows: ~0-60, ~300-360) map to low
    values. Cool colors (blues, cyans: ~180-260) map to high values.

    Args:
        hue: Hue in degrees (0-360).

    Returns:
        0-1 temperature where 0 = warmest, 1 = coolest.
    """
    # Normalize hue to 0-360
    hue = hue % 360.0
    # Blue (240) is coolest, red/orange (0-30) is warmest
    # Use cosine mapping centered on blue=240
    angle_from_warm = min(hue, 360.0 - hue)  # Distance from 0/360 (red)
    return float(np.clip(angle_from_warm / 180.0, 0.0, 1.0))


def _compute_color_diversity(hue_hist: np.ndarray) -> float:
    """Compute color diversity from a hue histogram.

    Uses normalized entropy of the hue distribution. A uniform
    distribution (many colors) gives high diversity; a single peak
    (one dominant color) gives low diversity.

    Args:
        hue_hist: Histogram of hue values (non-negative counts).

    Returns:
        0-1 diversity score.
    """
    total = float(np.sum(hue_hist))
    if total < 1.0:
        return 0.0

    probs = hue_hist.flatten().astype(np.float64) / total
    probs = probs[probs > 0]

    entropy = -float(np.sum(probs * np.log2(probs)))
    # Normalize by max possible entropy (uniform over all bins)
    max_entropy = np.log2(len(hue_hist))
    if max_entropy < 1e-6:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _find_secondary_hue(
    hue_hist: np.ndarray,
    dominant_bin: int,
    exclusion_bins: int = 3,
) -> float:
    """Find the secondary hue peak excluding the dominant hue region.

    Args:
        hue_hist: 36-bin hue histogram.
        dominant_bin: Index of the dominant hue bin.
        exclusion_bins: Number of bins to exclude around the dominant peak.

    Returns:
        Secondary hue in degrees (0-360). Returns 0 if no secondary peak.
    """
    masked = hue_hist.copy().flatten()
    n_bins = len(masked)

    # Exclude bins around the dominant peak (wrapping)
    for offset in range(-exclusion_bins, exclusion_bins + 1):
        idx = (dominant_bin + offset) % n_bins
        masked[idx] = 0.0

    if float(np.sum(masked)) < 1.0:
        return 0.0

    secondary_bin = int(np.argmax(masked))
    return float(secondary_bin) * (360.0 / n_bins)


def detect_led_screens(frame: np.ndarray) -> np.ndarray:
    """Detect and create a mask for LED screen regions in the frame.

    LED screens are identified as large, high-saturation rectangular
    regions. The returned mask has 255 for non-screen areas (usable)
    and 0 for screen areas (to be excluded).

    Args:
        frame: BGR image (H, W, 3), uint8.

    Returns:
        Binary mask (H, W), uint8. 255 = usable area, 0 = LED screen.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # High saturation + high brightness = likely LED screen
    screen_mask = cv2.bitwise_and(
        (s > _LED_SCREEN_SAT_MIN).astype(np.uint8) * 255,
        (v > 200).astype(np.uint8) * 255,
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel)
    screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_OPEN, kernel)

    # Find contours and filter by area and aspect ratio
    contours, _ = cv2.findContours(
        screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = float(frame.shape[0] * frame.shape[1])
    usable_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    for contour in contours:
        area_ratio = cv2.contourArea(contour) / total_area
        if _LED_SCREEN_AREA_MIN < area_ratio < _LED_SCREEN_AREA_MAX:
            # Check rectangularity (LED screens tend to be rectangular)
            _, _, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            contour_area = cv2.contourArea(contour)
            if rect_area > 0 and contour_area / rect_area > 0.6:
                cv2.drawContours(usable_mask, [contour], -1, 0, cv2.FILLED)

    return usable_mask


def extract_lighting(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    timestamp: float = 0.0,
    scene_confidence: float = 1.0,
    compensate_led_screens: bool = True,
    prev_dominant_hue: float | None = None,
) -> VideoLightingFrame:
    """Extract lighting features from a single video frame.

    Args:
        frame: BGR image (H, W, 3), uint8.
        prev_frame: Previous frame for temporal features (None for first frame).
        timestamp: Frame timestamp in seconds.
        scene_confidence: Scene classification confidence (0-1).
        compensate_led_screens: Whether to detect and mask LED screen regions.
        prev_dominant_hue: Dominant hue from previous frame for color change rate.

    Returns:
        VideoLightingFrame with all extracted features.
    """
    # LED screen compensation (Step 5)
    if compensate_led_screens:
        usable_mask = detect_led_screens(frame)
    else:
        usable_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Mask out very dark areas (< 15% brightness) -- these are unlit
    bright_mask = v > _BRIGHT_THRESHOLD

    # Combine with LED screen mask
    combined_mask = cv2.bitwise_and(
        bright_mask.astype(np.uint8) * 255,
        usable_mask,
    )

    # Overall brightness
    overall_brightness = float(np.mean(v)) / 255.0
    brightness_variance = float(np.std(v.astype(np.float64))) / 255.0

    # Dominant color -- histogram of hue channel in bright regions
    n_hue_bins = 36
    if combined_mask.any():
        hue_hist = cv2.calcHist(
            [h], [0], combined_mask, [n_hue_bins], [0, 180]
        )
        dominant_bin = int(np.argmax(hue_hist))
        dominant_hue = float(dominant_bin) * (360.0 / n_hue_bins)
        dominant_saturation = float(np.mean(s[combined_mask > 0])) / 255.0
        secondary_hue = _find_secondary_hue(hue_hist, dominant_bin)
        color_diversity = _compute_color_diversity(hue_hist)
    else:
        dominant_hue = 0.0
        dominant_saturation = 0.0
        secondary_hue = 0.0
        color_diversity = 0.0

    color_temperature = _hue_to_color_temperature(dominant_hue)

    # Spatial distribution -- divide frame into regions
    h_frame, w_frame = v.shape
    third_w = w_frame // 3
    half_h = h_frame // 2

    left_brightness = float(np.mean(v[:, :third_w])) / 255.0
    center_brightness = float(np.mean(v[:, third_w : 2 * third_w])) / 255.0
    right_brightness = float(np.mean(v[:, 2 * third_w :])) / 255.0
    top_brightness = float(np.mean(v[:half_h, :])) / 255.0
    bottom_brightness = float(np.mean(v[half_h:, :])) / 255.0

    spatial_symmetry = 1.0 - abs(left_brightness - right_brightness)

    # Temporal features
    brightness_delta = 0.0
    is_strobe = False
    color_change_rate = 0.0

    if prev_frame is not None:
        prev_v = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        brightness_delta = (
            float(np.mean(v)) - float(np.mean(prev_v))
        ) / 255.0
        is_strobe = abs(brightness_delta) > _STROBE_THRESHOLD

    if prev_dominant_hue is not None:
        # Compute shortest angular distance between hues
        hue_diff = abs(dominant_hue - prev_dominant_hue)
        color_change_rate = min(hue_diff, 360.0 - hue_diff)

    is_blackout = overall_brightness < _BLACKOUT_THRESHOLD

    return VideoLightingFrame(
        timestamp=timestamp,
        overall_brightness=overall_brightness,
        brightness_variance=brightness_variance,
        dominant_hue=dominant_hue,
        dominant_saturation=dominant_saturation,
        secondary_hue=secondary_hue,
        color_temperature=color_temperature,
        color_diversity=color_diversity,
        left_brightness=left_brightness,
        center_brightness=center_brightness,
        right_brightness=right_brightness,
        top_brightness=top_brightness,
        bottom_brightness=bottom_brightness,
        spatial_symmetry=spatial_symmetry,
        brightness_delta=brightness_delta,
        is_strobe=is_strobe,
        is_blackout=is_blackout,
        color_change_rate=color_change_rate,
        scene_confidence=scene_confidence,
    )


def extract_lighting_sequence(
    frames: list[np.ndarray],
    timestamps: list[float],
    scene_confidences: list[float] | None = None,
    compensate_led_screens: bool = True,
) -> list[VideoLightingFrame]:
    """Extract lighting features from a sequence of video frames.

    Convenience function that processes a list of frames, automatically
    computing temporal features (brightness_delta, is_strobe,
    color_change_rate) from consecutive frame pairs.

    Args:
        frames: List of BGR images (H, W, 3), uint8.
        timestamps: Timestamp in seconds for each frame.
        scene_confidences: Per-frame scene classification confidence.
            Defaults to 1.0 for all frames if not provided.
        compensate_led_screens: Whether to mask LED screen regions.

    Returns:
        List of VideoLightingFrame, one per input frame.
    """
    if not frames:
        return []

    if scene_confidences is None:
        scene_confidences = [1.0] * len(frames)

    results: list[VideoLightingFrame] = []
    prev_frame: np.ndarray | None = None
    prev_hue: float | None = None

    for _i, (frame, ts, conf) in enumerate(
        zip(frames, timestamps, scene_confidences, strict=True)
    ):
        lighting = extract_lighting(
            frame=frame,
            prev_frame=prev_frame,
            timestamp=ts,
            scene_confidence=conf,
            compensate_led_screens=compensate_led_screens,
            prev_dominant_hue=prev_hue,
        )
        results.append(lighting)
        prev_frame = frame
        prev_hue = lighting.dominant_hue

    return results
