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
from lumina.audio.source_separator import StemSet

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


# ── Stem-based genre classification ──────────────────────────────────


def _make_stems(
    sr: int = 44100,
    duration: float = 10.0,
    drums_amp: float = 0.1,
    bass_amp: float = 0.1,
    vocals_amp: float = 0.1,
    other_amp: float = 0.1,
    drum_regularity: bool = True,
) -> StemSet:
    """Create synthetic StemSet with controlled stem amplitudes.

    Args:
        sr: Sample rate.
        duration: Duration in seconds.
        drums_amp: RMS amplitude of drums stem.
        bass_amp: RMS amplitude of bass stem.
        vocals_amp: RMS amplitude of vocals stem.
        other_amp: RMS amplitude of other stem.
        drum_regularity: If True, create regular drum impulses; if False, random.

    Returns:
        StemSet with synthetic signals.
    """
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)

    # Drums: impulse train (regular or irregular)
    drums = np.zeros(n_samples, dtype=np.float32)
    if drum_regularity:
        # Regular impulses every 0.5s (120 BPM)
        interval = int(sr * 0.5)
        for i in range(0, n_samples, interval):
            end = min(i + int(sr * 0.02), n_samples)
            drums[i:end] = 1.0
    else:
        # Irregular impulses at random intervals
        pos = 0
        while pos < n_samples:
            end = min(pos + int(sr * 0.02), n_samples)
            drums[pos:end] = 1.0
            pos += int(sr * rng.uniform(0.2, 1.0))
    # Scale to target RMS
    drums_rms = float(np.sqrt(np.mean(drums**2)))
    if drums_rms > 1e-10:
        drums = drums * (drums_amp / drums_rms)

    # Bass: low-frequency sine (80 Hz)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    bass = np.sin(2 * np.pi * 80 * t).astype(np.float32)
    bass_rms = float(np.sqrt(np.mean(bass**2)))
    if bass_rms > 1e-10:
        bass = bass * (bass_amp / bass_rms)

    # Vocals: noise bursts with gaps (simulating speech phrasing)
    vocals = np.zeros(n_samples, dtype=np.float32)
    phrase_samples = int(sr * 1.5)  # 1.5s phrase
    gap_samples = int(sr * 0.5)     # 0.5s gap
    cycle = phrase_samples + gap_samples
    pos = 0
    while pos < n_samples:
        end = min(pos + phrase_samples, n_samples)
        vocals[pos:end] = rng.standard_normal(end - pos).astype(np.float32)
        pos += cycle
    # Scale to target RMS (including gaps)
    vocals_rms = float(np.sqrt(np.mean(vocals**2)))
    if vocals_rms > 1e-10:
        vocals = vocals * (vocals_amp / vocals_rms)

    # Other: higher-frequency content (synths/pads — sine at 1000 Hz + harmonics)
    other = (
        np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
    ).astype(np.float32)
    other_rms = float(np.sqrt(np.mean(other**2)))
    if other_rms > 1e-10:
        other = other * (other_amp / other_rms)

    return StemSet(
        drums=drums,
        bass=bass,
        vocals=vocals,
        other=other,
        sample_rate=sr,
    )


class TestStemGenreClassification:
    def _make_classifier(self, **kwargs: float | int) -> GenreClassifier:
        return GenreClassifier(fps=60, **kwargs)

    def test_electronic_high_other_stem(self) -> None:
        """High other stem + regular drums -> electronic family."""
        stems = _make_stems(
            other_amp=0.5, vocals_amp=0.1, drums_amp=0.3,
            drum_regularity=True,
        )
        stem_features = GenreClassifier._compute_stem_features(stems)
        family = GenreClassifier._classify_family_from_stems(stem_features)
        assert family["electronic"] > family["hiphop_rap"]

    def test_hiphop_high_vocal_stem(self) -> None:
        """High vocal stem + low other + irregular drums -> hiphop family."""
        stems = _make_stems(
            other_amp=0.1, vocals_amp=0.4, drums_amp=0.2,
            drum_regularity=False,
        )
        stem_features = GenreClassifier._compute_stem_features(stems)
        family = GenreClassifier._classify_family_from_stems(stem_features)
        assert family["hiphop_rap"] > family["electronic"]

    def test_electronic_not_misclassified(self) -> None:
        """Features mimicking Stereo Love + electronic stems -> NOT hiphop."""
        clf = self._make_classifier()
        n = 300
        # Moderate spectral features (ambiguous full-mix)
        stems = _make_stems(
            other_amp=0.5, vocals_amp=0.1, drums_amp=0.3,
            drum_regularity=True, duration=10.0,
        )
        result = clf.classify_file(
            energies=[0.55] * n,
            spectral_centroids=[4000.0] * n,
            sub_bass_energies=[0.35] * n,
            has_onsets=[i % 4 == 0 for i in range(n)],
            vocal_energies=[0.15] * n,
            drop_probabilities=[0.3] * n,
            stems=stems,
        )
        top_family = result[0].family
        assert top_family != "hiphop_rap"

    def test_festival_edm_with_stems(self) -> None:
        """High other + very regular drums + low vocals -> festival_edm in top 3."""
        clf = self._make_classifier()
        n = 300
        stems = _make_stems(
            other_amp=0.6, vocals_amp=0.08, drums_amp=0.35,
            drum_regularity=True,
        )
        result = clf.classify_file(
            energies=[0.70] * n,
            spectral_centroids=[6000.0] * n,
            sub_bass_energies=[0.45] * n,
            has_onsets=[i % 4 == 0 for i in range(n)],
            vocal_energies=[0.15] * n,
            drop_probabilities=[0.60] * n,
            stems=stems,
        )
        sorted_profiles = sorted(
            result[0].genre_weights.items(), key=lambda x: x[1], reverse=True
        )
        top_3 = [p[0] for p in sorted_profiles[:3]]
        assert "festival_edm" in top_3

    def test_stem_features_computation(self) -> None:
        """Stem feature ratios should match known synthetic signals."""
        stems = _make_stems(
            drums_amp=0.2, bass_amp=0.15, vocals_amp=0.4, other_amp=0.5,
            drum_regularity=True,
        )
        features = GenreClassifier._compute_stem_features(stems)

        # Other should have highest ratio
        assert features["stem_other_ratio"] > features["stem_drum_ratio"]
        assert features["stem_vocal_ratio"] > features["stem_drum_ratio"]
        # All ratios should be in [0, 1]
        for key in ("stem_vocal_ratio", "stem_other_ratio", "stem_drum_ratio"):
            assert 0.0 <= features[key] <= 1.0
        # Drum regularity should be high for regular impulses
        assert features["drum_regularity"] > 0.3
        # Bass character should be low (80 Hz sine)
        assert features["bass_character"] < 0.5
        # Onset regularity should be high for regular drums
        assert features["onset_regularity"] > 0.3
        # Debug keys should be present
        assert "_vocal_confidence" in features
        assert "_drum_reg_autocorr" in features
        assert "_drum_reg_ioi" in features

    def test_vocal_confidence_reduces_bleed(self) -> None:
        """When vocal and other stems are correlated, vocal_ratio is reduced."""
        sr = 44100
        n_samples = sr * 10
        t = np.linspace(0, 10, n_samples, dtype=np.float32)

        # Shared signal simulating instrument bleed into vocal stem
        shared = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        # Other = shared + unique content
        other = shared + np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.2
        # Vocal = shared + tiny noise (high correlation with other)
        rng = np.random.default_rng(42)
        vocals = shared + rng.standard_normal(n_samples).astype(np.float32) * 0.02

        stems = StemSet(
            drums=np.zeros(n_samples, dtype=np.float32),
            bass=np.sin(2 * np.pi * 80 * t).astype(np.float32) * 0.2,
            vocals=vocals,
            other=other,
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        # Vocal confidence should be reduced due to high correlation
        assert features["_vocal_confidence"] < 0.8
        # Vocal ratio should be lower than the raw RMS ratio
        raw_vocal_rms = float(np.sqrt(np.mean(vocals**2)))
        full = stems.drums + stems.bass + vocals + other
        full_rms = float(np.sqrt(np.mean(full**2)))
        raw_ratio = raw_vocal_rms / max(full_rms, 1e-10)
        assert features["stem_vocal_ratio"] < raw_ratio

    def test_vocal_confidence_high_for_real_vocals(self) -> None:
        """Independent vocal and other stems should have high vocal confidence."""
        stems = _make_stems(
            vocals_amp=0.4, other_amp=0.3, drums_amp=0.2,
        )
        features = GenreClassifier._compute_stem_features(stems)
        # Noise (vocals) and sine (other) are uncorrelated → confidence ≈ 1.0
        assert features["_vocal_confidence"] > 0.9

    def test_bass_character_low_for_sub_bass(self) -> None:
        """Bass character should be low for a pure sub-bass signal (80 Hz)."""
        sr = 44100
        n_samples = sr * 10
        t = np.linspace(0, 10, n_samples, dtype=np.float32)

        stems = StemSet(
            drums=np.zeros(n_samples, dtype=np.float32),
            bass=np.sin(2 * np.pi * 80 * t).astype(np.float32) * 0.3,
            vocals=np.zeros(n_samples, dtype=np.float32),
            other=np.zeros(n_samples, dtype=np.float32),
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        # 80 Hz centroid / 500 Hz ≈ 0.16 — should be well under 0.3
        assert features["bass_character"] < 0.3

    def test_bass_character_not_stuck_at_one(self) -> None:
        """Bass character should NOT clip to 1.0 for typical bass content."""
        sr = 44100
        n_samples = sr * 10
        t = np.linspace(0, 10, n_samples, dtype=np.float32)

        # Bass with harmonics (80 Hz fundamental + overtones up to 400 Hz)
        bass = (
            np.sin(2 * np.pi * 80 * t)
            + 0.5 * np.sin(2 * np.pi * 160 * t)
            + 0.25 * np.sin(2 * np.pi * 240 * t)
            + 0.12 * np.sin(2 * np.pi * 320 * t)
        ).astype(np.float32) * 0.3

        stems = StemSet(
            drums=np.zeros(n_samples, dtype=np.float32),
            bass=bass,
            vocals=np.zeros(n_samples, dtype=np.float32),
            other=np.zeros(n_samples, dtype=np.float32),
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        # Even with harmonics, centroid should be well below 500 Hz → < 1.0
        assert features["bass_character"] < 0.6

    def test_drum_regularity_high_for_four_on_floor(self) -> None:
        """Regular 128 BPM kicks should produce high drum regularity."""
        sr = 44100
        n_samples = sr * 10
        drums = np.zeros(n_samples, dtype=np.float32)
        # 128 BPM = beat every 0.469s
        interval = int(sr * 60 / 128)
        for pos in range(0, n_samples, interval):
            end = min(pos + int(sr * 0.015), n_samples)
            drums[pos:end] = 1.0

        stems = StemSet(
            drums=drums,
            bass=np.zeros(n_samples, dtype=np.float32),
            vocals=np.zeros(n_samples, dtype=np.float32),
            other=np.zeros(n_samples, dtype=np.float32),
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        assert features["drum_regularity"] > 0.6

    def test_no_stems_backward_compat(self) -> None:
        """classify_file without stems should still work correctly."""
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
        assert len(result) == n
        total = sum(result[0].genre_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)
        assert result[0].family in FAMILY_NAMES

    def test_sustained_instrument_reduces_confidence(self) -> None:
        """Sustained instrument in vocal stem should reduce vocal confidence."""
        sr = 44100
        n_samples = sr * 10
        t = np.linspace(0, 10, n_samples, dtype=np.float32)

        # Vocal stem = continuous sine (sustained, no pauses — like accordion)
        vocals = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.4
        # Other stem = weak content
        other = np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.05

        stems = StemSet(
            drums=np.zeros(n_samples, dtype=np.float32),
            bass=np.sin(2 * np.pi * 80 * t).astype(np.float32) * 0.1,
            vocals=vocals,
            other=other,
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        # Silence fraction ≈ 0 → confidence should be reduced
        assert features["_vocal_confidence"] < 0.5

    def test_drum_regularity_with_subdivisions(self) -> None:
        """Regular kicks + quieter hi-hats should still show high regularity."""
        sr = 44100
        n_samples = sr * 10
        drums = np.zeros(n_samples, dtype=np.float32)
        # 128 BPM kicks every 0.469s
        kick_interval = int(sr * 60 / 128)
        for pos in range(0, n_samples, kick_interval):
            end = min(pos + int(sr * 0.015), n_samples)
            drums[pos:end] = 1.0
        # Add 16th-note hi-hats (4× rate, quieter)
        hihat_interval = kick_interval // 4
        for pos in range(0, n_samples, hihat_interval):
            end = min(pos + int(sr * 0.005), n_samples)
            drums[pos:end] += 0.2  # much quieter than kicks

        stems = StemSet(
            drums=drums,
            bass=np.zeros(n_samples, dtype=np.float32),
            vocals=np.zeros(n_samples, dtype=np.float32),
            other=np.zeros(n_samples, dtype=np.float32),
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        assert features["drum_regularity"] > 0.6

    def test_ioi_finds_peaks_in_clean_kicks(self) -> None:
        """Clean kick impulses at regular intervals should have high IOI regularity."""
        sr = 44100
        n_samples = sr * 10
        drums = np.zeros(n_samples, dtype=np.float32)
        # 120 BPM = every 0.5s
        interval = int(sr * 0.5)
        expected_peaks = 0
        for pos in range(0, n_samples, interval):
            end = min(pos + int(sr * 0.02), n_samples)
            drums[pos:end] = 1.0
            expected_peaks += 1

        stems = StemSet(
            drums=drums,
            bass=np.zeros(n_samples, dtype=np.float32),
            vocals=np.zeros(n_samples, dtype=np.float32),
            other=np.zeros(n_samples, dtype=np.float32),
            sample_rate=sr,
        )

        features = GenreClassifier._compute_stem_features(stems)
        assert features["_drum_reg_ioi"] > 0.7
        # Autocorrelation should also be high
        assert features["_drum_reg_autocorr"] > 0.5

    def test_passthrough_stems_fallback(self) -> None:
        """When all stems are identical (demucs fallback), should still work."""
        sr = 44100
        n_samples = sr * 5
        rng = np.random.default_rng(99)
        audio = rng.standard_normal(n_samples).astype(np.float32) * 0.3

        # Passthrough: all stems = full mix
        stems = StemSet(
            drums=audio.copy(),
            bass=audio.copy(),
            vocals=audio.copy(),
            other=audio.copy(),
            sample_rate=sr,
        )
        clf = self._make_classifier()
        n = 300
        result = clf.classify_file(
            energies=[0.5] * n,
            spectral_centroids=[4000.0] * n,
            sub_bass_energies=[0.3] * n,
            has_onsets=[False] * n,
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.1] * n,
            stems=stems,
        )
        assert len(result) == n
        total = sum(result[0].genre_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_decision_tree_electronic_regular_drums(self) -> None:
        """Electronic + regular drums → festival_edm wins decisively."""
        clf = self._make_classifier()
        n = 300
        stems = _make_stems(
            other_amp=0.5, vocals_amp=0.08, drums_amp=0.3,
            drum_regularity=True,
        )
        result = clf.classify_file(
            energies=[0.65] * n,
            spectral_centroids=[5500.0] * n,
            sub_bass_energies=[0.40] * n,
            has_onsets=[i % 4 == 0 for i in range(n)],
            vocal_energies=[0.15] * n,
            drop_probabilities=[0.50] * n,
            stems=stems,
        )
        w = result[0].genre_weights
        # Decision tree should give festival_edm 0.45 — no longer ~0.18 uniform
        assert w["festival_edm"] > 0.35
        top = max(w, key=w.get)  # type: ignore[arg-type]
        assert top == "festival_edm"

    def test_decision_tree_hiphop_rage_trap(self) -> None:
        """Hiphop + confident vocals + dark bass → rage_trap wins.

        Uses real Cocaine Nose stem features.
        """
        # Cocaine Nose: family=hiphop, vocal_conf=1.00, bass_char=0.29
        features = {
            "stem_other_ratio": 0.27,
            "stem_vocal_ratio": 0.35,
            "_vocal_confidence": 1.00,
            "drum_regularity": 0.46,
            "bass_character": 0.29,
            "_rms_bass": 0.23,
            "_rms_drums": 0.12,
        }
        family_weights = {"hiphop_rap": 0.55, "electronic": 0.20, "hybrid": 0.25}
        w = GenreClassifier._classify_profile_from_stems(features, family_weights)
        assert w["rage_trap"] > 0.35
        top = max(w, key=w.get)  # type: ignore[arg-type]
        assert top == "rage_trap"

    def test_decision_tree_hiphop_psych_rnb(self) -> None:
        """Hiphop + low other ratio + warm bass → psych_rnb wins.

        Uses real São Paulo stem features.
        """
        # São Paulo: family=hiphop, vocal_conf=0.88, bass_char=0.42, other=0.13
        features = {
            "stem_other_ratio": 0.13,
            "stem_vocal_ratio": 0.30,
            "_vocal_confidence": 0.88,
            "drum_regularity": 0.47,
            "bass_character": 0.42,
            "_rms_bass": 0.20,
            "_rms_drums": 0.15,
        }
        family_weights = {"hiphop_rap": 0.60, "electronic": 0.15, "hybrid": 0.25}
        w = GenreClassifier._classify_profile_from_stems(features, family_weights)
        assert w["psych_rnb"] > 0.35
        top = max(w, key=w.get)  # type: ignore[arg-type]
        assert top == "psych_rnb"

    def test_bass_dominance_boosts_hiphop_family(self) -> None:
        """Bass-dominant stems should push family toward hiphop_rap.

        Uses real Cocaine Nose stem ratios.
        """
        # Cocaine Nose: bass_rms=0.23, drum_rms=0.12 → bass dominant
        features = {
            "stem_other_ratio": 0.27,
            "stem_vocal_ratio": 0.35,
            "drum_regularity": 0.46,
            "_rms_bass": 0.23,
            "_rms_drums": 0.12,
        }
        family = GenreClassifier._classify_family_from_stems(features)
        assert family["hiphop_rap"] > family["electronic"]

    def test_genre_override(self) -> None:
        """genre_override should lock to the specified profile at 1.0."""
        clf = self._make_classifier()
        n = 120
        result = clf.classify_file(
            energies=[0.5] * n,
            spectral_centroids=[4000.0] * n,
            sub_bass_energies=[0.3] * n,
            has_onsets=[False] * n,
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.1] * n,
            genre_override="rage_trap",
        )
        assert len(result) == n
        w = result[0].genre_weights
        assert w["rage_trap"] == 1.0
        assert w["festival_edm"] == 0.0
        assert result[0].family == "hiphop_rap"
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_genre_override_electronic_profile(self) -> None:
        """genre_override with electronic profile should set electronic family."""
        clf = self._make_classifier()
        n = 60
        result = clf.classify_file(
            energies=[0.5] * n,
            spectral_centroids=[4000.0] * n,
            sub_bass_energies=[0.3] * n,
            has_onsets=[False] * n,
            vocal_energies=[0.3] * n,
            drop_probabilities=[0.1] * n,
            genre_override="festival_edm",
        )
        assert result[0].genre_weights["festival_edm"] == 1.0
        assert result[0].family == "electronic"
