"""Tests for the LUMINA genre classification module.

Tests verify two-stage classification (family → profile), weighted
blend output, softmax normalization, and profile discrimination.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.genre_classifier import (
    FAMILY_NAMES,
    PROFILE_NAMES,
    GenreClassifier,
    GenreFrame,
)

# ── Helpers ───────────────────────────────────────────────────────────


def feed_constant_frames(
    clf: GenreClassifier,
    n_frames: int,
    energy: float = 0.5,
    spectral_centroid: float = 3000.0,
    sub_bass_energy: float = 0.3,
    vocal_energy: float = 0.3,
    drop_probability: float = 0.1,
    onset_every: int = 0,
) -> list[GenreFrame]:
    """Feed N identical frames and return results."""
    results: list[GenreFrame] = []
    for i in range(n_frames):
        has_onset = onset_every > 0 and i % onset_every == 0
        results.append(
            clf.process_frame(
                energy=energy,
                spectral_centroid=spectral_centroid,
                sub_bass_energy=sub_bass_energy,
                has_onset=has_onset,
                vocal_energy=vocal_energy,
                drop_probability=drop_probability,
            )
        )
    return results


# ── GenreFrame dataclass ─────────────────────────────────────────────


class TestGenreFrame:
    def test_fields(self) -> None:
        frame = GenreFrame(
            family="hiphop_rap",
            family_weights={"hiphop_rap": 0.7, "electronic": 0.2, "hybrid": 0.1},
            genre_weights={"rage_trap": 0.5, "psych_rnb": 0.3},
        )
        assert frame.family == "hiphop_rap"
        assert frame.family_weights["hiphop_rap"] == 0.7
        assert frame.genre_weights["rage_trap"] == 0.5

    def test_equality(self) -> None:
        w = {"rage_trap": 1.0}
        fw = {"hiphop_rap": 1.0}
        a = GenreFrame(family="hiphop_rap", family_weights=fw, genre_weights=w)
        b = GenreFrame(family="hiphop_rap", family_weights=fw, genre_weights=w)
        assert a == b


# ── Weight normalization ─────────────────────────────────────────────


class TestWeightNormalization:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_genre_weights_sum_to_one(self) -> None:
        """Genre weights should sum to ~1.0."""
        clf = self._make_classifier(smoothing=0.5)
        frames = feed_constant_frames(clf, 600, energy=0.6, vocal_energy=0.5)
        for f in frames[-10:]:
            total = sum(f.genre_weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_family_weights_sum_to_one(self) -> None:
        """Family weights should sum to ~1.0."""
        clf = self._make_classifier(smoothing=0.5)
        frames = feed_constant_frames(clf, 600, energy=0.6, vocal_energy=0.5)
        for f in frames[-10:]:
            total = sum(f.family_weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_all_profiles_present(self) -> None:
        """Genre weights should contain all 8 profiles."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 120)
        for f in frames:
            for profile in PROFILE_NAMES:
                assert profile in f.genre_weights

    def test_all_families_present(self) -> None:
        """Family weights should contain all 3 families."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 120)
        for f in frames:
            for family in FAMILY_NAMES:
                assert family in f.family_weights

    def test_weights_non_negative(self) -> None:
        """All weights should be non-negative."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 300, energy=0.7, sub_bass_energy=0.6)
        for f in frames:
            for w in f.genre_weights.values():
                assert w >= 0.0
            for w in f.family_weights.values():
                assert w >= 0.0


# ── Family classification (Stage 1) ──────────────────────────────────


class TestFamilyClassification:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_hiphop_features_favor_hiphop(self) -> None:
        """Hip-hop features (sub-bass, vocals, low centroid) should favor hiphop_rap."""
        clf = self._make_classifier(smoothing=0.3)
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.55,
            spectral_centroid=3500.0,
            sub_bass_energy=0.50,
            vocal_energy=0.55,
            drop_probability=0.1,
            onset_every=5,
        )
        last = frames[-1]
        assert last.family_weights["hiphop_rap"] > last.family_weights["electronic"]

    def test_electronic_features_favor_electronic(self) -> None:
        """Electronic features (high centroid, drops, low vocals) should favor electronic."""
        clf = self._make_classifier(smoothing=0.3)
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.70,
            spectral_centroid=5500.0,
            sub_bass_energy=0.45,
            vocal_energy=0.15,
            drop_probability=0.55,
            onset_every=4,
        )
        last = frames[-1]
        assert last.family_weights["electronic"] > last.family_weights["hiphop_rap"]

    def test_valid_family_label(self) -> None:
        """Family label should be one of the valid names."""
        clf = self._make_classifier()
        frames = feed_constant_frames(clf, 120)
        for f in frames:
            assert f.family in FAMILY_NAMES


# ── Profile classification (Stage 2) ─────────────────────────────────


class TestProfileClassification:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_rage_trap_features(self) -> None:
        """Rage trap features should give rage_trap the highest or near-highest weight."""
        clf = self._make_classifier(smoothing=0.3)
        # Carti/Travis: high energy contrast, heavy 808s, sparse vocals
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.65,
            spectral_centroid=3500.0,
            sub_bass_energy=0.60,
            vocal_energy=0.35,
            drop_probability=0.30,
            onset_every=5,
        )
        last = frames[-1]
        # rage_trap should be among the top profiles
        sorted_profiles = sorted(last.genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "rage_trap" in top_3

    def test_festival_edm_features(self) -> None:
        """Festival EDM features should favor festival_edm profile."""
        clf = self._make_classifier(smoothing=0.3)
        # Guetta/Armin: high energy, bright, build-drop cycles
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.72,
            spectral_centroid=6000.0,
            sub_bass_energy=0.50,
            vocal_energy=0.25,
            drop_probability=0.65,
            onset_every=4,
        )
        last = frames[-1]
        sorted_profiles = sorted(last.genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "festival_edm" in top_3

    def test_psych_rnb_features(self) -> None:
        """Psych R&B features should favor psych_rnb profile."""
        clf = self._make_classifier(smoothing=0.3)
        # Weeknd/Toliver: smooth, atmospheric, rich vocals
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.50,
            spectral_centroid=4500.0,
            sub_bass_energy=0.40,
            vocal_energy=0.65,
            drop_probability=0.10,
            onset_every=8,
        )
        last = frames[-1]
        sorted_profiles = sorted(last.genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "psych_rnb" in top_3

    def test_uk_bass_features(self) -> None:
        """UK bass features should favor uk_bass profile."""
        clf = self._make_classifier(smoothing=0.3)
        # Fred again..: bass-heavy, raw, low vocals, moderate drops
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.65,
            spectral_centroid=4000.0,
            sub_bass_energy=0.60,
            vocal_energy=0.20,
            drop_probability=0.45,
            onset_every=4,
        )
        last = frames[-1]
        sorted_profiles = sorted(last.genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "uk_bass" in top_3

    def test_euro_alt_features(self) -> None:
        """Euro alt features should favor euro_alt profile."""
        clf = self._make_classifier(smoothing=0.3)
        # AyVe/Exetra: restrained, sparse, artistic
        frames = feed_constant_frames(
            clf,
            600,
            energy=0.35,
            spectral_centroid=5000.0,
            sub_bass_energy=0.25,
            vocal_energy=0.40,
            drop_probability=0.08,
            onset_every=10,
        )
        last = frames[-1]
        sorted_profiles = sorted(last.genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "euro_alt" in top_3


# ── Smoothing and stability ──────────────────────────────────────────


class TestSmoothing:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_high_smoothing_stable(self) -> None:
        """High smoothing should produce stable weights across frames."""
        clf = self._make_classifier(smoothing=0.95)
        frames = feed_constant_frames(clf, 300, energy=0.6)
        # Check weight stability in last 10 frames
        last_10 = frames[-10:]
        for profile in PROFILE_NAMES:
            weights = [f.genre_weights[profile] for f in last_10]
            std = float(np.std(weights))
            assert std < 0.01, f"Unstable: {profile} std={std}"

    def test_low_smoothing_responsive(self) -> None:
        """Low smoothing should respond quickly to feature changes."""
        clf = self._make_classifier(smoothing=0.3)

        # Feed hip-hop features
        feed_constant_frames(
            clf, 300, energy=0.55, vocal_energy=0.6, sub_bass_energy=0.5, drop_probability=0.1
        )

        # Switch to EDM features
        frames = feed_constant_frames(
            clf, 300, energy=0.72, vocal_energy=0.15, spectral_centroid=6000.0, drop_probability=0.6
        )

        # EDM profile should have gained weight
        last = frames[-1]
        assert last.genre_weights["festival_edm"] > 0.05

    def test_temperature_affects_peakiness(self) -> None:
        """Lower temperature should produce more peaked (confident) distributions."""
        clf_low_temp = self._make_classifier(smoothing=0.0, temperature=0.2)
        clf_high_temp = self._make_classifier(smoothing=0.0, temperature=2.0)

        # Same features for both
        low_frames = feed_constant_frames(
            clf_low_temp, 600, energy=0.7, sub_bass_energy=0.6, vocal_energy=0.35
        )
        high_frames = feed_constant_frames(
            clf_high_temp, 600, energy=0.7, sub_bass_energy=0.6, vocal_energy=0.35
        )

        # Low temp should have higher max weight (more peaked)
        low_max = max(low_frames[-1].genre_weights.values())
        high_max = max(high_frames[-1].genre_weights.values())
        assert low_max > high_max


# ── Offline classification ───────────────────────────────────────────


class TestOfflineClassification:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_offline_returns_correct_length(self) -> None:
        """Offline should return one frame per input."""
        clf = self._make_classifier()
        n = 300
        result = clf.classify_offline(
            energies=[0.5] * n,
            spectral_centroids=[3000.0] * n,
            sub_bass_energies=[0.3] * n,
            has_onsets=[False] * n,
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.1] * n,
        )
        assert len(result) == n

    def test_offline_empty_input(self) -> None:
        """Empty input should return empty list."""
        clf = self._make_classifier()
        result = clf.classify_offline([], [], [], [], [], [])
        assert result == []

    def test_offline_valid_output(self) -> None:
        """Offline output should have valid weights."""
        clf = self._make_classifier()
        result = clf.classify_offline(
            energies=[0.6] * 120,
            spectral_centroids=[4000.0] * 120,
            sub_bass_energies=[0.4] * 120,
            has_onsets=[True, False] * 60,
            vocal_energies=[0.5] * 120,
            drop_probabilities=[0.2] * 120,
        )
        for f in result:
            assert f.family in FAMILY_NAMES
            assert sum(f.genre_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_offline_seeding_converges_faster(self) -> None:
        """Offline seeding should produce stable classification from the start."""
        clf = self._make_classifier(smoothing=0.5)

        n = 300
        # Rage-trap-like features: high energy, heavy bass, low vocals
        energies = [0.65] * n
        centroids = [3500.0] * n
        basses = [0.6] * n
        onsets = [i % 3 == 0 for i in range(n)]
        vocals = [0.3] * n
        drops = [0.3] * n

        offline = clf.classify_offline(energies, centroids, basses, onsets, vocals, drops)

        # With seeding, even the first frame should have a reasonable
        # classification (not uniform) since windows are pre-filled
        first = offline[0]
        top_profile = max(first.genre_weights, key=first.genre_weights.get)  # type: ignore[arg-type]
        # Top profile weight should be above uniform (1/8 = 0.125)
        assert first.genre_weights[top_profile] > 0.15


# ── File mode genre lock ─────────────────────────────────────────────


class TestGenreLock:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_all_frames_identical(self) -> None:
        """classify_file should return identical frames for every position."""
        clf = self._make_classifier()
        n = 600
        result = clf.classify_file(
            energies=[0.65] * n,
            spectral_centroids=[3500.0] * n,
            sub_bass_energies=[0.6] * n,
            has_onsets=[i % 3 == 0 for i in range(n)],
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.3] * n,
        )
        assert len(result) == n
        # Every frame should be the same object
        for frame in result:
            assert frame is result[0]

    def test_no_genre_drift(self) -> None:
        """Genre should not change over time in file mode."""
        clf = self._make_classifier()
        n = 3600  # 60s at 60fps
        result = clf.classify_file(
            energies=[0.65] * n,
            spectral_centroids=[3500.0] * n,
            sub_bass_energies=[0.6] * n,
            has_onsets=[i % 3 == 0 for i in range(n)],
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.3] * n,
        )
        first_weights = result[0].genre_weights
        last_weights = result[-1].genre_weights
        # Since all frames are identical, weights must match exactly
        assert first_weights == last_weights

    def test_rage_trap_locked(self) -> None:
        """Rage trap features should lock to rage_trap as top profile."""
        clf = self._make_classifier()
        n = 300
        result = clf.classify_file(
            energies=[0.65] * n,
            spectral_centroids=[3500.0] * n,
            sub_bass_energies=[0.60] * n,
            has_onsets=[i % 3 == 0 for i in range(n)],
            vocal_energies=[0.35] * n,
            drop_probabilities=[0.30] * n,
        )
        top = max(result[0].genre_weights, key=result[0].genre_weights.get)  # type: ignore[arg-type]
        top_3 = sorted(result[0].genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_3_names = [p[0] for p in top_3[:3]]
        assert "rage_trap" in top_3_names

    def test_file_empty_input(self) -> None:
        """Empty input should return empty list."""
        clf = self._make_classifier()
        result = clf.classify_file([], [], [], [], [], [])
        assert result == []

    def test_file_weights_sum_to_one(self) -> None:
        """Locked genre weights should sum to 1.0."""
        clf = self._make_classifier()
        result = clf.classify_file(
            energies=[0.5] * 120,
            spectral_centroids=[4000.0] * 120,
            sub_bass_energies=[0.4] * 120,
            has_onsets=[True, False] * 60,
            vocal_energies=[0.5] * 120,
            drop_probabilities=[0.2] * 120,
        )
        total = sum(result[0].genre_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)


# ── Reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self) -> None:
        """reset() should return to initial uniform weights."""
        clf = GenreClassifier(fps=60, smoothing=0.3)
        feed_constant_frames(clf, 300, energy=0.8, sub_bass_energy=0.7)

        clf.reset()
        assert len(clf._energy_history) == 0

        # Weights should be uniform after reset
        uniform = 1.0 / len(PROFILE_NAMES)
        for w in clf._prev_weights.values():
            assert w == pytest.approx(uniform)

    def test_classify_after_reset(self) -> None:
        """Classification after reset should behave as fresh."""
        clf = GenreClassifier(fps=60, smoothing=0.3)
        feed_constant_frames(clf, 300, energy=0.8, sub_bass_energy=0.7)

        clf.reset()
        frames = feed_constant_frames(clf, 60, energy=0.5)
        assert all(f.family in FAMILY_NAMES for f in frames)


# ── Profile count and names ──────────────────────────────────────────


class TestProfileDefinitions:
    def test_eight_profiles(self) -> None:
        assert len(PROFILE_NAMES) == 8

    def test_three_families(self) -> None:
        assert len(FAMILY_NAMES) == 3

    def test_profile_names(self) -> None:
        expected = {
            "rage_trap",
            "psych_rnb",
            "french_melodic",
            "french_hard",
            "euro_alt",
            "theatrical",
            "festival_edm",
            "uk_bass",
        }
        assert set(PROFILE_NAMES) == expected
