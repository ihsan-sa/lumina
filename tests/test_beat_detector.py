"""Tests for the LUMINA beat detection module.

Fast tests (no @slow marker) test the phase tracking math without madmom.
Slow tests run the full madmom pipeline on synthetic audio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina.audio.beat_detector import BeatDetector, BeatInfo


# ── Helpers ───────────────────────────────────────────────────────────


def make_click_track(
    bpm: float,
    duration: float,
    sr: int = 44100,
    accent_every: int = 4,
) -> np.ndarray:
    """Generate a synthetic click track at a given BPM.

    Args:
        bpm: Tempo in beats per minute.
        duration: Length in seconds.
        sr: Sample rate.
        accent_every: Accent (louder click) every N beats (for downbeat).

    Returns:
        Mono float32 audio array normalized to [-1, 1].
    """
    num_samples = int(sr * duration)
    audio = np.zeros(num_samples, dtype=np.float32)
    beat_interval = 60.0 / bpm
    click_len = int(0.01 * sr)  # 10ms click

    beat_idx = 0
    while True:
        sample_pos = int(beat_idx * beat_interval * sr)
        if sample_pos + click_len > num_samples:
            break
        amp = 0.9 if beat_idx % accent_every == 0 else 0.5
        audio[sample_pos : sample_pos + click_len] = amp
        beat_idx += 1

    return audio


# ── BeatInfo dataclass ────────────────────────────────────────────────


class TestBeatInfo:
    def test_fields(self) -> None:
        info = BeatInfo(bpm=128.0, beat_phase=0.5, bar_phase=0.25, is_beat=True, is_downbeat=False)
        assert info.bpm == 128.0
        assert info.beat_phase == 0.5
        assert info.bar_phase == 0.25
        assert info.is_beat is True
        assert info.is_downbeat is False

    def test_equality(self) -> None:
        a = BeatInfo(bpm=120.0, beat_phase=0.0, bar_phase=0.0, is_beat=True, is_downbeat=True)
        b = BeatInfo(bpm=120.0, beat_phase=0.0, bar_phase=0.0, is_beat=True, is_downbeat=True)
        assert a == b


# ── Phase tracking math (no madmom) ──────────────────────────────────


class TestPhaseTracking:
    """Test _update_tracking + _get_frame_info with known beat times."""

    def _make_detector(self, bpm: float = 120.0, beats_per_bar: int = 4) -> BeatDetector:
        return BeatDetector(sr=44100, fps=100, beats_per_bar=beats_per_bar)

    def test_beat_phase_at_beat(self) -> None:
        """Phase should be ~0.0 exactly at a beat time."""
        det = self._make_detector(bpm=120.0)
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        det._update_tracking(beats, bpm=120.0)

        # Query at exact beat time (2.5s = next beat after ref)
        info = det._get_frame_info(2.5)
        assert info.beat_phase == pytest.approx(0.0, abs=0.01)

    def test_beat_phase_mid_beat(self) -> None:
        """Phase should be ~0.5 halfway between beats."""
        det = self._make_detector(bpm=120.0)
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        det._update_tracking(beats, bpm=120.0)

        info = det._get_frame_info(2.25)
        assert info.beat_phase == pytest.approx(0.5, abs=0.01)

    def test_beat_phase_quarter(self) -> None:
        """Phase should be ~0.25 a quarter into a beat."""
        det = self._make_detector(bpm=120.0)
        beats = np.array([0.0, 0.5, 1.0])
        det._update_tracking(beats, bpm=120.0)

        info = det._get_frame_info(1.125)
        assert info.beat_phase == pytest.approx(0.25, abs=0.01)

    def test_bpm_estimation_from_beats(self) -> None:
        """BPM should be computed from inter-beat intervals."""
        det = self._make_detector()
        # 140 BPM = 60/140 ≈ 0.4286s per beat
        interval = 60.0 / 140.0
        beats = np.array([i * interval for i in range(8)])
        det._update_tracking(beats)

        assert det._bpm == pytest.approx(140.0, abs=1.0)

    def test_bpm_explicit_override(self) -> None:
        """Explicit bpm parameter should override estimation."""
        det = self._make_detector()
        beats = np.array([0.0, 0.5, 1.0])  # would estimate 120 BPM
        det._update_tracking(beats, bpm=128.0)

        assert det._bpm == pytest.approx(128.0, abs=0.01)

    def test_bar_phase_at_downbeat(self) -> None:
        """Bar phase should be 0.0 at a downbeat."""
        det = self._make_detector(beats_per_bar=4)
        # 4 beats: first is downbeat (index 0)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        # Next downbeat is at 2.0s (4 beats after first downbeat)
        info = det._get_frame_info(2.0)
        assert info.bar_phase == pytest.approx(0.0, abs=0.02)

    def test_bar_phase_at_beat_2(self) -> None:
        """Bar phase should be 0.25 at the second beat of a bar."""
        det = self._make_detector(beats_per_bar=4)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        # Beat 2 of next bar at 2.5s
        info = det._get_frame_info(2.5)
        assert info.bar_phase == pytest.approx(0.25, abs=0.02)

    def test_bar_phase_at_beat_3(self) -> None:
        """Bar phase should be 0.5 at the third beat."""
        det = self._make_detector(beats_per_bar=4)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        info = det._get_frame_info(3.0)
        assert info.bar_phase == pytest.approx(0.5, abs=0.02)

    def test_bar_phase_at_beat_4(self) -> None:
        """Bar phase should be 0.75 at the fourth beat."""
        det = self._make_detector(beats_per_bar=4)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        info = det._get_frame_info(3.5)
        assert info.bar_phase == pytest.approx(0.75, abs=0.02)

    def test_is_beat_fires_once_per_beat(self) -> None:
        """is_beat should be True for exactly one frame per beat period."""
        det = self._make_detector(bpm=120.0)
        beats = np.array([0.0, 0.5, 1.0])
        det._update_tracking(beats, bpm=120.0)

        # Prime tracker (first call establishes prev_abs_beat)
        det._get_frame_info(0.9)

        # Scan 1 second at 100fps — expect 2 beats (at 1.0s and 1.5s)
        beat_count = 0
        for i in range(100):
            t = 1.0 + i / 100.0
            info = det._get_frame_info(t)
            if info.is_beat:
                beat_count += 1

        assert beat_count == 2

    def test_is_downbeat_fires_every_4_beats(self) -> None:
        """is_downbeat should fire every beats_per_bar beats."""
        det = self._make_detector(beats_per_bar=4)
        beats = np.array([i * 0.5 for i in range(8)])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        # Prime tracker
        det._get_frame_info(3.9)

        downbeat_count = 0
        beat_count = 0
        # Scan 4 seconds (8 beats) — expect 2 downbeats
        for i in range(400):
            t = 4.0 + i / 100.0
            info = det._get_frame_info(t)
            if info.is_beat:
                beat_count += 1
            if info.is_downbeat:
                downbeat_count += 1

        assert beat_count == 8
        assert downbeat_count == 2

    def test_no_beats_returns_defaults(self) -> None:
        """Before any beats detected, return sensible defaults."""
        det = self._make_detector()
        info = det._get_frame_info(1.0)
        assert info.bpm == 120.0
        assert info.beat_phase == 0.0
        assert info.bar_phase == 0.0
        assert info.is_beat is False
        assert info.is_downbeat is False

    def test_beat_phase_always_in_range(self) -> None:
        """beat_phase should always be in [0.0, 1.0)."""
        det = self._make_detector()
        beats = np.array([0.0, 0.5, 1.0])
        det._update_tracking(beats, bpm=120.0)

        for i in range(500):
            t = i * 0.013  # arbitrary timestamps
            info = det._get_frame_info(t)
            assert 0.0 <= info.beat_phase < 1.0, f"beat_phase={info.beat_phase} at t={t}"

    def test_bar_phase_always_in_range(self) -> None:
        """bar_phase should always be in [0.0, 1.0)."""
        det = self._make_detector(beats_per_bar=4)
        beats = np.array([i * 0.5 for i in range(8)])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        for i in range(500):
            t = i * 0.017
            info = det._get_frame_info(t)
            assert 0.0 <= info.bar_phase < 1.0, f"bar_phase={info.bar_phase} at t={t}"

    def test_bpm_smoothing_on_update(self) -> None:
        """Subsequent _update_tracking calls should smooth BPM."""
        det = self._make_detector()
        beats1 = np.array([0.0, 0.5, 1.0])
        det._update_tracking(beats1, bpm=120.0)
        assert det._bpm == pytest.approx(120.0)

        # Second update with different BPM — should be smoothed
        beats2 = np.array([2.0, 2.4, 2.8])  # 150 BPM
        det._update_tracking(beats2, bpm=150.0)
        # smoothing=0.3: result = 0.3*120 + 0.7*150 = 36 + 105 = 141
        assert det._bpm == pytest.approx(141.0, abs=0.1)

    def test_reset_clears_state(self) -> None:
        """reset() should clear all tracking state."""
        det = self._make_detector()
        beats = np.array([0.0, 0.5, 1.0])
        det._update_tracking(beats, bpm=128.0)
        assert det._has_beats is True

        det.reset()
        assert det._has_beats is False
        assert det._bpm == 120.0
        info = det._get_frame_info(0.5)
        assert info.is_beat is False

    def test_3_4_time_signature(self) -> None:
        """Should work with 3/4 time (waltz)."""
        det = self._make_detector(beats_per_bar=3)
        beats = np.array([i * 0.5 for i in range(6)])
        det._update_tracking(beats, bpm=120.0, first_downbeat_idx=0)

        # Prime tracker
        det._get_frame_info(2.9)

        # Count downbeats over 3 seconds (6 beats = 2 bars in 3/4)
        downbeat_count = 0
        for i in range(300):
            t = 3.0 + i / 100.0
            info = det._get_frame_info(t)
            if info.is_downbeat:
                downbeat_count += 1

        assert downbeat_count == 2

    def test_continuous_beat_numbering_across_updates(self) -> None:
        """Beat numbering should be continuous across tracking updates."""
        det = self._make_detector(beats_per_bar=4)
        beats1 = np.array([0.0, 0.5, 1.0, 1.5])
        det._update_tracking(beats1, bpm=120.0, first_downbeat_idx=0)

        # Iterate to establish prev_abs_beat
        for i in range(200):
            det._get_frame_info(2.0 + i / 100.0)

        # Second update: new beats continuing the sequence
        beats2 = np.array([4.0, 4.5, 5.0, 5.5])
        det._update_tracking(beats2, bpm=120.0)

        # Beats should continue without double-firing or gaps
        beat_times = []
        for i in range(200):
            t = 4.0 + i / 100.0
            info = det._get_frame_info(t)
            if info.is_beat:
                beat_times.append(t)

        # Should have ~4 beats in 2 seconds at 120 BPM
        assert len(beat_times) == 4
        # Spacing should be ~0.5s
        for j in range(1, len(beat_times)):
            spacing = beat_times[j] - beat_times[j - 1]
            assert spacing == pytest.approx(0.5, abs=0.02)


# ── BPM computation ──────────────────────────────────────────────────


class TestBpmComputation:
    def test_basic_bpm(self) -> None:
        det = BeatDetector()
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        assert det._compute_bpm(beats) == pytest.approx(120.0, abs=0.1)

    def test_single_beat_returns_default(self) -> None:
        det = BeatDetector()
        beats = np.array([1.0])
        assert det._compute_bpm(beats) == 120.0  # default

    def test_empty_beats_returns_default(self) -> None:
        det = BeatDetector()
        beats = np.array([])
        assert det._compute_bpm(beats) == 120.0

    def test_filters_outlier_intervals(self) -> None:
        """Outlier intervals outside min/max BPM should be filtered."""
        det = BeatDetector(min_bpm=60.0, max_bpm=200.0)
        # Most intervals are 0.5s (120 BPM), one outlier at 5.0s
        beats = np.array([0.0, 0.5, 1.0, 1.5, 6.5, 7.0, 7.5])
        bpm = det._compute_bpm(beats)
        assert bpm == pytest.approx(120.0, abs=1.0)

    def test_various_tempos(self) -> None:
        for target_bpm in [80.0, 100.0, 128.0, 140.0, 174.0]:
            det = BeatDetector()
            interval = 60.0 / target_bpm
            beats = np.array([i * interval for i in range(16)])
            assert det._compute_bpm(beats) == pytest.approx(target_bpm, abs=0.5)


# ── Streaming process_chunk ──────────────────────────────────────────


class TestProcessChunk:
    """Test process_chunk without madmom (short audio, no analysis trigger)."""

    def test_returns_frames_for_chunk(self) -> None:
        """process_chunk should return the right number of frames."""
        det = BeatDetector(sr=44100, fps=100)
        # 0.5 seconds of audio = 50 frames at 100fps
        chunk = np.zeros(22050, dtype=np.float32)
        results = det.process_chunk(chunk)
        assert len(results) == 50

    def test_returns_beat_info_type(self) -> None:
        det = BeatDetector(sr=44100, fps=100)
        chunk = np.zeros(4410, dtype=np.float32)
        results = det.process_chunk(chunk)
        assert all(isinstance(r, BeatInfo) for r in results)

    def test_multiple_chunks_accumulate(self) -> None:
        """Buffer should grow across multiple process_chunk calls."""
        det = BeatDetector(sr=44100, fps=100)
        chunk = np.zeros(4410, dtype=np.float32)  # 0.1s

        det.process_chunk(chunk)
        assert len(det._buffer) == 4410

        det.process_chunk(chunk)
        assert len(det._buffer) == 8820

    def test_buffer_capped_at_max(self) -> None:
        """Buffer should not exceed max_buffer_seconds."""
        det = BeatDetector(sr=44100, fps=100)
        det._max_buffer_seconds = 1.0  # 1 second cap

        # Feed 2 seconds of audio
        chunk = np.zeros(88200, dtype=np.float32)
        det.process_chunk(chunk)

        max_samples = int(44100 * 1.0)
        assert len(det._buffer) <= max_samples

    def test_default_bpm_before_analysis(self) -> None:
        """Before enough audio for analysis, BPM should be default."""
        det = BeatDetector(sr=44100, fps=100)
        chunk = np.zeros(4410, dtype=np.float32)  # 0.1s — too short for analysis
        results = det.process_chunk(chunk)

        for info in results:
            assert info.bpm == 120.0


# ── Integration tests with madmom (slow) ─────────────────────────────


@pytest.mark.slow
class TestStreamingIntegration:
    """Integration tests that run the full madmom pipeline."""

    def test_detects_beats_120bpm(self) -> None:
        """Streaming should detect ~120 BPM from a click track."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)

        audio = make_click_track(bpm=120.0, duration=6.0, sr=44100)

        # Feed in 1-second chunks
        all_results: list[BeatInfo] = []
        chunk_size = 44100
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            results = det.process_chunk(chunk)
            all_results.extend(results)

        # After 6 seconds, BPM should be close to 120
        final_bpm = all_results[-1].bpm
        assert final_bpm == pytest.approx(120.0, abs=5.0), f"Got BPM={final_bpm}"

    def test_streaming_finds_beats(self) -> None:
        """Streaming should fire is_beat at regular intervals."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)

        audio = make_click_track(bpm=120.0, duration=8.0, sr=44100)

        all_results: list[BeatInfo] = []
        chunk_size = 44100
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            all_results.extend(det.process_chunk(chunk))

        # Count beats in the last 4 seconds (after analysis has stabilized)
        late_results = all_results[400:]  # from t=4s onward
        beat_count = sum(1 for r in late_results if r.is_beat)

        # 4 seconds at 120 BPM = 8 beats, allow some tolerance
        assert 6 <= beat_count <= 10, f"Expected ~8 beats, got {beat_count}"

    def test_streaming_beat_phase_wraps(self) -> None:
        """beat_phase should wrap from near 1.0 to near 0.0 at beats."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)

        audio = make_click_track(bpm=120.0, duration=6.0, sr=44100)

        all_results: list[BeatInfo] = []
        chunk_size = 44100
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            all_results.extend(det.process_chunk(chunk))

        # After stabilization, check phase ranges
        late_results = all_results[400:]
        phases = [r.beat_phase for r in late_results]

        assert min(phases) < 0.1, "Phase should approach 0.0"
        assert max(phases) > 0.9, "Phase should approach 1.0"


@pytest.mark.slow
class TestOfflineIntegration:
    """Integration tests for analyze_offline with madmom."""

    def test_offline_120bpm(self) -> None:
        """Offline analysis should detect ~120 BPM."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)
        audio = make_click_track(bpm=120.0, duration=8.0, sr=44100)

        results = det.analyze_offline(audio)

        assert len(results) > 0
        assert results[len(results) // 2].bpm == pytest.approx(120.0, abs=5.0)

    def test_offline_finds_beats(self) -> None:
        """Offline analysis should find beats at expected positions."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)
        audio = make_click_track(bpm=120.0, duration=6.0, sr=44100)

        results = det.analyze_offline(audio)
        beat_frames = [i for i, r in enumerate(results) if r.is_beat]

        # 6 seconds at 120 BPM = ~12 beats
        assert 8 <= len(beat_frames) <= 16, f"Expected ~12 beats, got {len(beat_frames)}"

        # Check spacing: beats should be ~50 frames apart (0.5s × 100fps)
        spacings = np.diff(beat_frames)
        median_spacing = np.median(spacings)
        assert median_spacing == pytest.approx(50, abs=5)

    def test_offline_finds_downbeats(self) -> None:
        """Offline analysis should identify downbeats."""
        det = BeatDetector(sr=44100, fps=100, beats_per_bar=4, min_bpm=60, max_bpm=200)
        audio = make_click_track(bpm=120.0, duration=8.0, sr=44100, accent_every=4)

        results = det.analyze_offline(audio)
        downbeat_frames = [i for i, r in enumerate(results) if r.is_downbeat]

        # 8 seconds, 4/4 time, 120 BPM = ~4 bars = ~4 downbeats (+ possible edge ones)
        assert len(downbeat_frames) >= 2, f"Expected ≥2 downbeats, got {len(downbeat_frames)}"

    def test_offline_bar_phase_range(self) -> None:
        """bar_phase should span [0, 1) across all frames."""
        det = BeatDetector(sr=44100, fps=100, beats_per_bar=4, min_bpm=60, max_bpm=200)
        audio = make_click_track(bpm=120.0, duration=8.0, sr=44100)

        results = det.analyze_offline(audio)
        bar_phases = [r.bar_phase for r in results]

        assert all(0.0 <= bp < 1.0 for bp in bar_phases), "bar_phase out of range"
        assert min(bar_phases) < 0.1
        assert max(bar_phases) > 0.8

    def test_offline_different_tempo(self) -> None:
        """Should correctly detect 140 BPM."""
        det = BeatDetector(sr=44100, fps=100, min_bpm=60, max_bpm=200)
        audio = make_click_track(bpm=140.0, duration=8.0, sr=44100)

        results = det.analyze_offline(audio)
        mid_bpm = results[len(results) // 2].bpm
        assert mid_bpm == pytest.approx(140.0, abs=5.0), f"Got BPM={mid_bpm}"
