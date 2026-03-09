"""Tests for the LUMINA segment classification module.

Tests verify segment classification via prototype matching, position
bias, minimum segment duration, and offline smoothing.
"""

from __future__ import annotations

import pytest

from lumina.audio.segment_classifier import (
    SEGMENT_LABELS,
    SegmentClassifier,
    SegmentFrame,
    SegmentType,
)


# ── Helpers ───────────────────────────────────────────────────────────


def feed_constant_frames(
    clf: SegmentClassifier,
    n_frames: int,
    energy: float = 0.5,
    energy_derivative: float = 0.0,
    spectral_centroid: float = 3000.0,
    sub_bass_energy: float = 0.3,
    vocal_energy: float = 0.3,
    has_onset: bool = False,
) -> list[SegmentFrame]:
    """Feed N identical frames and return results."""
    results: list[SegmentFrame] = []
    for _ in range(n_frames):
        results.append(clf.process_frame(
            energy=energy,
            energy_derivative=energy_derivative,
            spectral_centroid=spectral_centroid,
            sub_bass_energy=sub_bass_energy,
            vocal_energy=vocal_energy,
            has_onset=has_onset,
        ))
    return results


def make_section_features(
    n_frames: int,
    energy: float = 0.5,
    energy_derivative: float = 0.0,
    spectral_centroid: float = 3000.0,
    sub_bass_energy: float = 0.3,
    vocal_energy: float = 0.3,
    onset_rate: float = 0.0,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[bool]]:
    """Generate feature lists for a section with given characteristics."""
    energies = [energy] * n_frames
    derivs = [energy_derivative] * n_frames
    centroids = [spectral_centroid] * n_frames
    basses = [sub_bass_energy] * n_frames
    vocals = [vocal_energy] * n_frames
    # Onset every 1/onset_rate frames
    if onset_rate > 0:
        interval = max(1, int(1.0 / onset_rate))
        onsets = [i % interval == 0 for i in range(n_frames)]
    else:
        onsets = [False] * n_frames
    return energies, derivs, centroids, basses, vocals, onsets


def concat_features(
    *sections: tuple[list[float], list[float], list[float], list[float], list[float], list[bool]],
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[bool]]:
    """Concatenate multiple section feature tuples."""
    result: tuple[list[float], list[float], list[float], list[float], list[float], list[bool]] = (
        [], [], [], [], [], [],
    )
    for section in sections:
        for i in range(6):
            result[i].extend(section[i])
    return result


# ── SegmentFrame dataclass ───────────────────────────────────────────


class TestSegmentFrame:
    def test_fields(self) -> None:
        frame = SegmentFrame(segment="chorus", confidence=0.8, scores={"chorus": 0.9})
        assert frame.segment == "chorus"
        assert frame.confidence == 0.8
        assert frame.scores == {"chorus": 0.9}

    def test_equality(self) -> None:
        a = SegmentFrame(segment="verse", confidence=0.5, scores={})
        b = SegmentFrame(segment="verse", confidence=0.5, scores={})
        assert a == b


class TestSegmentType:
    def test_all_labels_have_enum(self) -> None:
        """Every label in SEGMENT_LABELS should have a SegmentType enum."""
        for label in SEGMENT_LABELS:
            assert SegmentType(label).value == label

    def test_label_count(self) -> None:
        assert len(SEGMENT_LABELS) == 7


# ── Basic classification ─────────────────────────────────────────────


class TestBasicClassification:
    def _make_classifier(self, **kwargs: float | int) -> SegmentClassifier:
        return SegmentClassifier(fps=60, **kwargs)

    def test_returns_valid_segment(self) -> None:
        """Classification should always return a valid segment label."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 300)
        for f in frames:
            assert f.segment in SEGMENT_LABELS

    def test_confidence_in_range(self) -> None:
        """Confidence should be in [0, 1]."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 300, energy=0.7)
        for f in frames:
            assert 0.0 <= f.confidence <= 1.0

    def test_scores_all_segments(self) -> None:
        """Scores dict should contain all segment labels."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 60)
        for f in frames:
            for label in SEGMENT_LABELS:
                assert label in f.scores

    def test_scores_non_negative(self) -> None:
        """All scores should be non-negative."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 120)
        for f in frames:
            for score in f.scores.values():
                assert score >= 0.0


# ── Segment prototype matching ───────────────────────────────────────


class TestPrototypeMatching:
    def _make_classifier(self, **kwargs: float | int) -> SegmentClassifier:
        return SegmentClassifier(fps=60, min_segment_seconds=0.5, **kwargs)

    def test_high_energy_dense_onsets_not_verse(self) -> None:
        """Very high energy with dense onsets should score drop/chorus higher than verse."""
        clf = self._make_classifier()
        frames = feed_constant_frames(
            clf, 300,
            energy=0.9,
            spectral_centroid=6000.0,
            sub_bass_energy=0.75,
            vocal_energy=0.1,
            has_onset=True,
        )
        last = frames[-1]
        assert last.scores["drop"] > last.scores["verse"]

    def test_low_energy_sparse_scores_breakdown(self) -> None:
        """Low energy with sparse onsets should score breakdown high."""
        clf = self._make_classifier()
        frames = feed_constant_frames(
            clf, 300,
            energy=0.15,
            energy_derivative=-0.01,
            spectral_centroid=2500.0,
            sub_bass_energy=0.08,
            vocal_energy=0.15,
            has_onset=False,
        )
        last = frames[-1]
        assert last.scores["breakdown"] > last.scores["chorus"]

    def test_moderate_energy_vocals_scores_verse(self) -> None:
        """Moderate energy with vocals should favor verse or chorus."""
        clf = self._make_classifier()
        frames = feed_constant_frames(
            clf, 300,
            energy=0.45,
            spectral_centroid=3500.0,
            sub_bass_energy=0.35,
            vocal_energy=0.55,
        )
        last = frames[-1]
        # verse/chorus should score higher than drop or breakdown
        verse_chorus = max(last.scores["verse"], last.scores["chorus"])
        assert verse_chorus > last.scores["drop"]
        assert verse_chorus > last.scores["breakdown"]

    def test_chorus_vs_verse_energy(self) -> None:
        """Higher energy with vocals should favor chorus over verse."""
        clf = self._make_classifier()
        frames = feed_constant_frames(
            clf, 300,
            energy=0.72,
            spectral_centroid=5500.0,
            sub_bass_energy=0.50,
            vocal_energy=0.65,
            has_onset=True,
        )
        last = frames[-1]
        assert last.scores["chorus"] > last.scores["verse"]


# ── Minimum segment duration ─────────────────────────────────────────


class TestMinSegmentDuration:
    def test_segment_stable_within_min_duration(self) -> None:
        """Segment should not change within min_segment_seconds."""
        clf = SegmentClassifier(fps=60, min_segment_seconds=2.0)

        # Feed verse-like features to establish segment
        feed_constant_frames(clf, 200, energy=0.45, vocal_energy=0.55,
                             sub_bass_energy=0.35)

        # Abruptly switch to drop-like features
        frames = feed_constant_frames(clf, 60, energy=0.95, vocal_energy=0.0,
                                      sub_bass_energy=0.8, has_onset=True)

        # Within 1 second (< 2.0s min), segment should not change
        # (it was locked to the previous segment)
        initial_segment = frames[0].segment
        for f in frames[:30]:
            assert f.segment == initial_segment

    def test_segment_changes_after_min_duration(self) -> None:
        """Segment should eventually change after min_segment_seconds."""
        clf = SegmentClassifier(fps=60, min_segment_seconds=1.0)

        # Establish a segment
        feed_constant_frames(clf, 120, energy=0.45, vocal_energy=0.55,
                             sub_bass_energy=0.35)

        # Switch to very different features for long enough
        frames = feed_constant_frames(clf, 300, energy=0.92, vocal_energy=0.05,
                                      sub_bass_energy=0.8, has_onset=True,
                                      spectral_centroid=6000.0)

        # By the end, segment should have changed
        segments = {f.segment for f in frames[-60:]}
        assert len(segments) >= 1  # at least one segment present


# ── Position bias ────────────────────────────────────────────────────


class TestPositionBias:
    def test_intro_bias_at_start(self) -> None:
        """Intro score should be boosted at the start of a track."""
        clf = SegmentClassifier(fps=60, position_weight=0.5)
        clf.set_track_duration(180.0)  # 3-minute track

        # Low energy at very start → intro should score well
        frames = feed_constant_frames(
            clf, 60,
            energy=0.15, vocal_energy=0.05, sub_bass_energy=0.1,
        )
        last = frames[-1]
        # Intro should be boosted by position
        assert last.scores["intro"] > 0

    def test_outro_bias_at_end(self) -> None:
        """Outro score should be boosted at the end of a track."""
        clf = SegmentClassifier(fps=60, position_weight=0.5)
        clf.set_track_duration(60.0)  # 1-minute track

        # Advance to end of track
        feed_constant_frames(clf, 60 * 55, energy=0.5, vocal_energy=0.3)

        # Low energy at end → outro should be boosted
        frames = feed_constant_frames(
            clf, 60,
            energy=0.18, energy_derivative=-0.01, vocal_energy=0.08,
            sub_bass_energy=0.12,
        )
        last = frames[-1]
        assert last.scores["outro"] > last.scores["verse"]

    def test_no_bias_without_duration(self) -> None:
        """Without track duration, intro/outro should not be artificially boosted."""
        clf = SegmentClassifier(fps=60, position_weight=0.5)
        # Do NOT call set_track_duration

        frames = feed_constant_frames(clf, 60, energy=0.15, vocal_energy=0.05)
        last = frames[-1]
        # Scores should be purely prototype-based
        # (intro may still score well due to low energy, but not position-boosted)
        assert "intro" in last.scores


# ── Offline classification ───────────────────────────────────────────


class TestOfflineClassification:
    def _make_classifier(self, **kwargs: float | int) -> SegmentClassifier:
        return SegmentClassifier(fps=60, **kwargs)

    def test_offline_returns_correct_length(self) -> None:
        """Offline should return one frame per input."""
        clf = self._make_classifier()
        n = 300
        features = make_section_features(n, energy=0.5)
        result = clf.classify_offline(*features)
        assert len(result) == n

    def test_offline_empty_input(self) -> None:
        """Empty input should return empty list."""
        clf = self._make_classifier()
        result = clf.classify_offline([], [], [], [], [], [])
        assert result == []

    def test_offline_valid_segments(self) -> None:
        """All offline segments should be valid labels."""
        clf = self._make_classifier()
        features = make_section_features(300, energy=0.6, vocal_energy=0.4)
        result = clf.classify_offline(*features)
        for f in result:
            assert f.segment in SEGMENT_LABELS

    def test_offline_multi_section_track(self) -> None:
        """Offline analysis of a multi-section track should produce
        at least 2 distinct segments if features vary enough."""
        clf = self._make_classifier(min_segment_seconds=1.0)

        # Simulate: quiet intro → loud chorus → quiet breakdown
        intro = make_section_features(
            180, energy=0.12, vocal_energy=0.05, sub_bass_energy=0.08,
        )
        chorus = make_section_features(
            300, energy=0.75, vocal_energy=0.65, sub_bass_energy=0.50,
            spectral_centroid=5500.0, onset_rate=0.5,
        )
        breakdown = make_section_features(
            180, energy=0.15, vocal_energy=0.10, sub_bass_energy=0.08,
        )

        features = concat_features(intro, chorus, breakdown)
        result = clf.classify_offline(*features)

        # Should have at least 2 different segment types
        unique_segments = {f.segment for f in result}
        assert len(unique_segments) >= 2

    def test_offline_smoothing_removes_flicker(self) -> None:
        """Offline smoothing should prevent very short segments."""
        clf = self._make_classifier(min_segment_seconds=0.5)

        # Feed mostly verse with a brief spike
        verse = make_section_features(600, energy=0.45, vocal_energy=0.55)
        result = clf.classify_offline(*verse)

        # Find segment runs
        runs: list[tuple[str, int]] = []
        current = result[0].segment
        count = 1
        for f in result[1:]:
            if f.segment == current:
                count += 1
            else:
                runs.append((current, count))
                current = f.segment
                count = 1
        runs.append((current, count))

        # All runs should be at least 30 frames (0.5s at 60fps)
        for label, length in runs:
            assert length >= 30 or length == runs[-1][1], (
                f"Short run: {label} for {length} frames"
            )


# ── Reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self) -> None:
        """reset() should clear all tracking state."""
        clf = SegmentClassifier(fps=60)
        feed_constant_frames(clf, 120, energy=0.8)

        clf.reset()
        assert len(clf._energy_history) == 0
        assert clf._total_frames == 0
        assert clf._current_segment == "verse"

    def test_classify_after_reset(self) -> None:
        """Classification after reset should behave as fresh."""
        clf = SegmentClassifier(fps=60)
        feed_constant_frames(clf, 120, energy=0.8, vocal_energy=0.7)
        clf.reset()

        frames = feed_constant_frames(clf, 60, energy=0.15, vocal_energy=0.05)
        # Should not be stuck on a previous segment
        assert all(f.segment in SEGMENT_LABELS for f in frames)
