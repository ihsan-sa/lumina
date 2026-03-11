"""Tests for the LayerTracker analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from lumina.analysis.layer_tracker import LayerFrame, LayerTracker, _rms_envelope
from lumina.audio.source_separator import StemSet


def _make_stems(
    sr: int = 44100,
    duration: float = 2.0,
    drums_active: bool = True,
    bass_active: bool = True,
    vocals_active: bool = False,
    other_active: bool = False,
) -> StemSet:
    """Create synthetic stems with known active/inactive states."""
    n = int(sr * duration)
    active_signal = np.random.randn(n).astype(np.float32) * 0.5
    silent = np.zeros(n, dtype=np.float32)
    return StemSet(
        drums=active_signal.copy() if drums_active else silent.copy(),
        bass=active_signal.copy() if bass_active else silent.copy(),
        vocals=active_signal.copy() if vocals_active else silent.copy(),
        other=active_signal.copy() if other_active else silent.copy(),
        sample_rate=sr,
    )


class TestRmsEnvelope:
    """Tests for the RMS envelope helper."""

    def test_silent_audio_gives_zero_rms(self) -> None:
        audio = np.zeros(44100, dtype=np.float32)
        env = _rms_envelope(audio, 512)
        assert np.all(env == 0.0)

    def test_loud_audio_gives_positive_rms(self) -> None:
        audio = np.ones(44100, dtype=np.float32)
        env = _rms_envelope(audio, 512)
        assert np.all(env > 0.0)

    def test_output_length(self) -> None:
        audio = np.random.randn(44100).astype(np.float32)
        env = _rms_envelope(audio, 512)
        assert len(env) == 44100 // 512


class TestLayerTracker:
    """Tests for the LayerTracker."""

    def test_all_stems_active(self) -> None:
        stems = _make_stems(drums_active=True, bass_active=True,
                            vocals_active=True, other_active=True)
        tracker = LayerTracker(sr=44100)
        frames = tracker.analyze(stems)
        assert len(frames) > 0
        # After smoothing settles, all 4 should be active
        last_frame = frames[-1]
        assert last_frame.active_count == 4

    def test_no_stems_active(self) -> None:
        stems = _make_stems(drums_active=False, bass_active=False,
                            vocals_active=False, other_active=False)
        tracker = LayerTracker(sr=44100)
        frames = tracker.analyze(stems)
        assert len(frames) > 0
        for f in frames:
            assert f.active_count == 0

    def test_partial_stems(self) -> None:
        stems = _make_stems(drums_active=True, bass_active=False,
                            vocals_active=True, other_active=False)
        tracker = LayerTracker(sr=44100)
        frames = tracker.analyze(stems)
        last_frame = frames[-1]
        assert last_frame.active_count == 2

    def test_layer_mask_has_all_stems(self) -> None:
        stems = _make_stems()
        tracker = LayerTracker(sr=44100)
        frames = tracker.analyze(stems)
        for f in frames:
            assert "drums" in f.layer_mask
            assert "bass" in f.layer_mask
            assert "vocals" in f.layer_mask
            assert "other" in f.layer_mask

    def test_resample_to_fps(self) -> None:
        stems = _make_stems(duration=2.0)
        tracker = LayerTracker(sr=44100)
        frames = tracker.analyze(stems)
        resampled = tracker.resample_to_fps(frames, target_n=120, fps=60)
        assert len(resampled) == 120

    def test_layer_change_detection(self) -> None:
        """Build stems where drums start active, then cut to silence midway."""
        sr = 44100
        n = sr * 2  # 2 seconds
        half = n // 2
        drums = np.concatenate([
            np.random.randn(half).astype(np.float32) * 0.5,
            np.zeros(half, dtype=np.float32),
        ])
        silent = np.zeros(n, dtype=np.float32)
        stems = StemSet(drums=drums, bass=silent, vocals=silent,
                        other=silent, sample_rate=sr)

        tracker = LayerTracker(sr=sr)
        frames = tracker.analyze(stems)

        # Should have at least one "drop_drums" event
        changes = [f.layer_change for f in frames if f.layer_change is not None]
        assert any("drums" in c for c in changes)
