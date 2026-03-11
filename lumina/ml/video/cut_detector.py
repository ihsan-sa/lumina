"""Camera cut detection for concert video frames.

Detects scene changes (camera cuts) in video frame sequences using
frame differencing. Frames near cuts produce unreliable lighting data
due to motion blur, exposure changes, and mixed-scene content, so they
are marked for exclusion from training data.

A cut is detected when the mean absolute pixel difference between
consecutive frames exceeds a configurable threshold. Frames within
3 frames of a detected cut are flagged as unreliable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default threshold for mean pixel difference to detect a cut
_DEFAULT_CUT_THRESHOLD = 30.0

# Frames within this many frames of a cut are marked unreliable
_CUT_UNRELIABLE_RADIUS = 3


@dataclass(slots=True)
class CutDetectionResult:
    """Result of camera cut detection on a frame sequence.

    Args:
        cut_indices: List of frame indices where cuts were detected.
        unreliable_mask: Boolean array (one per frame). True = frame is
            within ``_CUT_UNRELIABLE_RADIUS`` of a cut and should be
            excluded from training data.
        frame_diffs: Mean absolute difference for each consecutive frame
            pair. Length is ``len(frames) - 1``.
    """

    cut_indices: list[int]
    unreliable_mask: list[bool]
    frame_diffs: list[float]


def detect_cuts(
    frames: list[np.ndarray],
    threshold: float = _DEFAULT_CUT_THRESHOLD,
) -> list[int]:
    """Return indices where camera cuts occur.

    Computes the mean absolute pixel difference between consecutive
    frames. A difference exceeding the threshold indicates a camera cut.

    Args:
        frames: List of BGR images (H, W, 3), uint8.
        threshold: Mean pixel difference threshold to trigger a cut.
            Higher values detect only hard cuts; lower values also
            catch dissolves and fast pans.

    Returns:
        List of frame indices where cuts were detected.
    """
    cuts: list[int] = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        mean_diff = float(np.mean(diff))
        if mean_diff > threshold:
            cuts.append(i)
    return cuts


def detect_cuts_with_reliability(
    frames: list[np.ndarray],
    threshold: float = _DEFAULT_CUT_THRESHOLD,
    unreliable_radius: int = _CUT_UNRELIABLE_RADIUS,
) -> CutDetectionResult:
    """Detect camera cuts and mark nearby frames as unreliable.

    Extends ``detect_cuts`` by also computing a per-frame reliability
    mask. Frames within ``unreliable_radius`` frames of any detected
    cut are marked as unreliable and should be excluded from training
    data, since they may contain motion blur, exposure transitions, or
    mixed-scene content.

    Args:
        frames: List of BGR images (H, W, 3), uint8.
        threshold: Mean pixel difference threshold for cut detection.
        unreliable_radius: Number of frames around each cut to mark
            as unreliable. Default is 3 per DOCS.md specification.

    Returns:
        CutDetectionResult with cut indices, unreliable mask, and
        per-frame difference values.
    """
    n_frames = len(frames)

    if n_frames == 0:
        return CutDetectionResult(
            cut_indices=[],
            unreliable_mask=[],
            frame_diffs=[],
        )

    if n_frames == 1:
        return CutDetectionResult(
            cut_indices=[],
            unreliable_mask=[False],
            frame_diffs=[],
        )

    # Compute frame-to-frame differences
    frame_diffs: list[float] = []
    cuts: list[int] = []

    for i in range(1, n_frames):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        mean_diff = float(np.mean(diff))
        frame_diffs.append(mean_diff)
        if mean_diff > threshold:
            cuts.append(i)

    # Build unreliable mask: True for frames within radius of any cut
    unreliable = [False] * n_frames
    for cut_idx in cuts:
        start = max(0, cut_idx - unreliable_radius)
        end = min(n_frames, cut_idx + unreliable_radius + 1)
        for j in range(start, end):
            unreliable[j] = True

    logger.info(
        "Detected %d cuts in %d frames, %d frames marked unreliable",
        len(cuts),
        n_frames,
        sum(unreliable),
    )

    return CutDetectionResult(
        cut_indices=cuts,
        unreliable_mask=unreliable,
        frame_diffs=frame_diffs,
    )
