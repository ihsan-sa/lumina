"""Two-stage genre classification with weighted profile blends.

Classifies audio into LUMINA's 8 lighting profiles via a two-stage
approach:

**Stage 1 — Family Classification (3 classes):**
- Hip-Hop/Rap: Strong sub-bass, rhythmic kicks/snares, vocal presence.
- Electronic: Synthetic textures, high spectral centroid, build-drop patterns.
- Hybrid: Mixed characteristics from both families.

**Stage 2 — Profile Classification (8 profiles):**
Each family narrows the profile candidates, then spectral/rhythmic
features determine the specific profile blend.

**Design (Phase 1 — rule-based):**
Uses hand-tuned feature prototypes rather than trained ML models.
Each profile has a characteristic feature signature. The classifier
computes soft similarity scores against all profiles and normalizes
to a probability distribution (weights summing to 1.0).

This approach works on the multilingual music library (French, German,
Portuguese, English) because it uses signal-level features only — no
language-specific processing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Profile definitions ──────────────────────────────────────────────

PROFILE_NAMES = (
    "rage_trap",
    "psych_rnb",
    "french_melodic",
    "french_hard",
    "euro_alt",
    "theatrical",
    "festival_edm",
    "uk_bass",
)

FAMILY_NAMES = ("hiphop_rap", "electronic", "hybrid")

# Family membership: which profiles belong to which family
# A profile can appear in multiple families with different base weights
_FAMILY_PROFILES: dict[str, list[str]] = {
    "hiphop_rap": ["rage_trap", "psych_rnb", "french_melodic", "french_hard", "euro_alt"],
    "electronic": ["theatrical", "festival_edm", "uk_bass"],
    "hybrid": [
        "rage_trap",
        "psych_rnb",
        "euro_alt",
        "theatrical",
        "festival_edm",
        "uk_bass",
    ],
}

# ── Feature prototypes per profile ───────────────────────────────────
# Each profile has a characteristic feature vector.
# Values are (ideal_value, tolerance) for Gaussian similarity scoring.
# Features:
#   energy_mean: average energy level
#   energy_variance: how much energy fluctuates (drops = high variance)
#   spectral_centroid_norm: brightness (0-1, normalized from Hz)
#   sub_bass_ratio: sub-bass energy relative to total
#   onset_density: how many percussive events per second
#   vocal_ratio: vocal presence level
#   drop_tendency: how often drops/builds occur (from drop predictor)

_PROFILE_PROTOTYPES: dict[str, dict[str, tuple[float, float]]] = {
    "rage_trap": {
        # Carti, Travis Scott: VIOLENT dynamics, distorted highs, sparse/processed vocals
        # Key separators vs french_hard: high variance, high centroid, very low vocals
        "energy_mean": (0.60, 0.20),
        "energy_variance": (0.35, 0.12),  # VERY high, tight — defining feature
        "spectral_centroid_norm": (0.50, 0.20),  # distorted, aggressive production
        "sub_bass_ratio": (0.55, 0.20),
        "onset_density": (0.50, 0.20),  # rapid hi-hats, aggressive drums
        "vocal_ratio": (0.20, 0.15),  # VERY low, tight — sparse/processed
        "drop_tendency": (0.35, 0.20),
    },
    "psych_rnb": {
        # Don Toliver, Weeknd: smooth, atmospheric, moderate energy, rich vocals
        # Key separators: very low variance, very high vocals, sparse onsets
        "energy_mean": (0.45, 0.15),
        "energy_variance": (0.08, 0.08),  # very smooth
        "spectral_centroid_norm": (0.45, 0.20),
        "sub_bass_ratio": (0.35, 0.20),
        "onset_density": (0.20, 0.15),  # sparse, laid-back
        "vocal_ratio": (0.70, 0.15),  # VERY high, tight — vocals dominate
        "drop_tendency": (0.10, 0.10),
    },
    "french_melodic": {
        # Ninho, Jul: warm, colorful, hi-hat bounce, melodic vocals
        # Key separators: very high onset density (hi-hats), high vocals, low variance
        "energy_mean": (0.48, 0.15),
        "energy_variance": (0.10, 0.08),  # consistent vibe
        "spectral_centroid_norm": (0.42, 0.20),
        "sub_bass_ratio": (0.35, 0.15),
        "onset_density": (0.55, 0.15),  # HIGH, tight — hi-hat bounce is defining
        "vocal_ratio": (0.62, 0.15),  # high melodic vocals
        "drop_tendency": (0.08, 0.08),
    },
    "french_hard": {
        # Kaaris: heavy, deliberate, consistently aggressive, DARK
        # Key separators vs rage_trap: low variance, very low centroid, very high bass
        "energy_mean": (0.68, 0.12),  # consistently high
        "energy_variance": (0.12, 0.10),  # LOW — deliberate, regimented
        "spectral_centroid_norm": (0.28, 0.12),  # VERY low, tight — dark production
        "sub_bass_ratio": (0.70, 0.15),  # VERY high, tight — 808s dominate
        "onset_density": (0.30, 0.15),  # moderate — deliberate hits
        "vocal_ratio": (0.50, 0.20),  # clear aggressive rap
        "drop_tendency": (0.15, 0.12),
    },
    "euro_alt": {
        # AyVe, Exetra Archive: restrained, sparse, artistic
        # Key separators: very low energy, very sparse onsets
        "energy_mean": (0.32, 0.12),  # LOW, tight — restraint is defining
        "energy_variance": (0.12, 0.10),
        "spectral_centroid_norm": (0.52, 0.20),
        "sub_bass_ratio": (0.22, 0.15),
        "onset_density": (0.18, 0.12),  # VERY low, tight — sparse
        "vocal_ratio": (0.38, 0.20),
        "drop_tendency": (0.08, 0.08),
    },
    "theatrical": {
        # Stromae: dynamic storytelling, emotional arc, vocal-driven
        # Key separators: high variance + high vocals (unlike rage_trap which has low vocals)
        "energy_mean": (0.50, 0.20),
        "energy_variance": (0.28, 0.15),  # high — follows emotional arc
        "spectral_centroid_norm": (0.52, 0.20),
        "sub_bass_ratio": (0.28, 0.15),  # low — not bass-driven
        "onset_density": (0.35, 0.20),
        "vocal_ratio": (0.60, 0.15),  # high — vocals carry the story
        "drop_tendency": (0.22, 0.15),
    },
    "festival_edm": {
        # Guetta, Armin, Edward Maya: build-drop cycles, bright synths, euphoric
        # Key separators: very high centroid, very high drop tendency
        "energy_mean": (0.68, 0.15),
        "energy_variance": (0.32, 0.12),  # VERY high, tight — extreme build-drop
        "spectral_centroid_norm": (0.62, 0.15),  # HIGH, tight — bright synths
        "sub_bass_ratio": (0.48, 0.20),
        "onset_density": (0.48, 0.20),
        "vocal_ratio": (0.28, 0.20),  # low — instrumental focused
        "drop_tendency": (0.70, 0.15),  # VERY high, tight — this IS the genre
    },
    "uk_bass": {
        # Fred again..: raw, underground, bass-heavy, chaotic energy
        # Key separators vs french_hard: higher centroid, higher onsets, higher drops
        "energy_mean": (0.62, 0.15),
        "energy_variance": (0.22, 0.12),
        "spectral_centroid_norm": (0.38, 0.15),
        "sub_bass_ratio": (0.62, 0.15),  # very high bass
        "onset_density": (0.52, 0.15),  # high — frantic
        "vocal_ratio": (0.22, 0.15),  # low — sample-based
        "drop_tendency": (0.50, 0.15),  # high
    },
}

# Family prototypes for Stage 1
_FAMILY_PROTOTYPES: dict[str, dict[str, tuple[float, float]]] = {
    "hiphop_rap": {
        "energy_mean": (0.55, 0.25),
        "spectral_centroid_norm": (0.35, 0.20),
        "sub_bass_ratio": (0.45, 0.25),
        "onset_density": (0.40, 0.25),
        "vocal_ratio": (0.50, 0.25),
        "drop_tendency": (0.15, 0.15),
    },
    "electronic": {
        "energy_mean": (0.65, 0.20),
        "spectral_centroid_norm": (0.55, 0.20),
        "sub_bass_ratio": (0.45, 0.25),
        "onset_density": (0.45, 0.25),
        "vocal_ratio": (0.30, 0.25),
        "drop_tendency": (0.45, 0.25),
    },
    "hybrid": {
        "energy_mean": (0.55, 0.25),
        "spectral_centroid_norm": (0.45, 0.25),
        "sub_bass_ratio": (0.45, 0.25),
        "onset_density": (0.40, 0.25),
        "vocal_ratio": (0.40, 0.25),
        "drop_tendency": (0.25, 0.20),
    },
}


@dataclass(slots=True)
class GenreFrame:
    """Genre classification for a single time frame.

    Args:
        family: Top-level family classification.
        family_weights: Family name -> confidence weight (sum to 1.0).
        genre_weights: Profile name -> weight (sum to 1.0).
    """

    family: str
    family_weights: dict[str, float]
    genre_weights: dict[str, float]


class GenreClassifier:
    """Two-stage genre classifier producing weighted profile blends.

    Stage 1 classifies into family (hiphop_rap, electronic, hybrid).
    Stage 2 classifies into specific lighting profiles within the
    detected family, producing a weighted blend.

    Operates on a rolling window of audio features for stability.
    Classification updates are smoothed via EMA to prevent rapid
    flickering between genres.

    Args:
        fps: Input frame rate.
        window_seconds: Feature averaging window size.
        smoothing: EMA smoothing for genre weights (0=none, 1=frozen).
        temperature: Softmax temperature for weight distribution.
            Lower = more peaky (confident), higher = more uniform.
    """

    def __init__(
        self,
        fps: int = 60,
        window_seconds: float = 30.0,
        smoothing: float = 0.995,
        temperature: float = 0.3,
    ) -> None:
        self._fps = fps
        self._window_size = max(1, int(fps * window_seconds))
        self._smoothing = smoothing
        self._temperature = max(0.01, temperature)

        self.reset()

    def reset(self) -> None:
        """Reset all internal state."""
        self._energy_history: deque[float] = deque(maxlen=self._window_size)
        self._centroid_history: deque[float] = deque(maxlen=self._window_size)
        self._bass_history: deque[float] = deque(maxlen=self._window_size)
        self._onset_history: deque[float] = deque(maxlen=self._window_size)
        self._vocal_history: deque[float] = deque(maxlen=self._window_size)
        self._drop_history: deque[float] = deque(maxlen=self._window_size)

        # Initialize to uniform weights
        n = len(PROFILE_NAMES)
        uniform = 1.0 / n
        self._prev_weights: dict[str, float] = {p: uniform for p in PROFILE_NAMES}
        self._prev_family_weights: dict[str, float] = {
            f: 1.0 / len(FAMILY_NAMES) for f in FAMILY_NAMES
        }

    # ── Public API ────────────────────────────────────────────────

    def process_frame(
        self,
        energy: float,
        spectral_centroid: float,
        sub_bass_energy: float,
        has_onset: bool,
        vocal_energy: float,
        drop_probability: float,
    ) -> GenreFrame:
        """Process a single frame and return genre classification.

        Args:
            energy: 0.0-1.0 overall energy.
            spectral_centroid: Brightness in Hz.
            sub_bass_energy: 0.0-1.0 sub-bass level.
            has_onset: Whether an onset was detected.
            vocal_energy: 0.0-1.0 vocal presence.
            drop_probability: 0.0-1.0 drop prediction.

        Returns:
            GenreFrame with family and weighted profile blend.
        """
        # Store features
        self._energy_history.append(energy)
        centroid_norm = min(1.0, spectral_centroid / 10000.0)
        self._centroid_history.append(centroid_norm)
        self._bass_history.append(sub_bass_energy)
        self._onset_history.append(1.0 if has_onset else 0.0)
        self._vocal_history.append(vocal_energy)
        self._drop_history.append(drop_probability)

        # Compute windowed features
        features = self._compute_features()

        # Stage 1: Family classification
        family_weights = self._classify_family(features)

        # Stage 2: Profile classification (informed by family)
        genre_weights = self._classify_profiles(features, family_weights)

        # Smooth via EMA
        smoothed_family = self._smooth_weights(family_weights, self._prev_family_weights)
        smoothed_genre = self._smooth_weights(genre_weights, self._prev_weights)

        self._prev_family_weights = smoothed_family
        self._prev_weights = smoothed_genre

        # Determine top family
        top_family = max(smoothed_family, key=smoothed_family.get)  # type: ignore[arg-type]

        return GenreFrame(
            family=top_family,
            family_weights=smoothed_family,
            genre_weights=smoothed_genre,
        )

    def classify_offline(
        self,
        energies: list[float],
        spectral_centroids: list[float],
        sub_bass_energies: list[float],
        has_onsets: list[bool],
        vocal_energies: list[float],
        drop_probabilities: list[float],
    ) -> list[GenreFrame]:
        """Classify genre for an entire track (offline mode).

        Args:
            energies: Energy per frame.
            spectral_centroids: Centroid per frame.
            sub_bass_energies: Sub-bass per frame.
            has_onsets: Onset flags per frame.
            vocal_energies: Vocal energy per frame.
            drop_probabilities: Drop probability per frame.

        Returns:
            List of GenreFrame, one per input frame.
        """
        self.reset()
        n = len(energies)
        if n == 0:
            return []

        # Seed feature windows with global averages so classification
        # starts with reasonable context instead of building from zero.
        # This prevents the first ~30s from drifting before enough data
        # accumulates in the rolling window.
        seed_count = min(n, self._window_size)
        avg_energy = float(np.mean(energies[:seed_count]))
        avg_centroid = float(
            np.mean([min(1.0, c / 10000.0) for c in spectral_centroids[:seed_count]])
        )
        avg_bass = float(np.mean(sub_bass_energies[:seed_count]))
        avg_onset = float(np.mean([1.0 if o else 0.0 for o in has_onsets[:seed_count]]))
        avg_vocal = float(np.mean(vocal_energies[:seed_count]))
        avg_drop = float(np.mean(drop_probabilities[:seed_count]))

        # Pre-fill windows with global averages (half the window size
        # to allow real data to quickly dominate)
        prefill = self._window_size // 2
        for _ in range(prefill):
            self._energy_history.append(avg_energy)
            self._centroid_history.append(avg_centroid)
            self._bass_history.append(avg_bass)
            self._onset_history.append(avg_onset)
            self._vocal_history.append(avg_vocal)
            self._drop_history.append(avg_drop)

        # Run a single classification pass with seeded data to initialize
        # the EMA weights before processing actual frames
        seed_features = self._compute_features()
        seed_family = self._classify_family(seed_features)
        seed_genre = self._classify_profiles(seed_features, seed_family)
        self._prev_family_weights = seed_family
        self._prev_weights = seed_genre

        return [
            self.process_frame(
                energy=energies[i],
                spectral_centroid=spectral_centroids[i],
                sub_bass_energy=sub_bass_energies[i],
                has_onset=has_onsets[i],
                vocal_energy=vocal_energies[i],
                drop_probability=drop_probabilities[i],
            )
            for i in range(n)
        ]

    def classify_file(
        self,
        energies: list[float],
        spectral_centroids: list[float],
        sub_bass_energies: list[float],
        has_onsets: list[bool],
        vocal_energies: list[float],
        drop_probabilities: list[float],
    ) -> list[GenreFrame]:
        """Classify genre for a file with a single locked result.

        Computes aggregate features from the first 30s of the track,
        classifies once, and returns the same GenreFrame for every frame.
        No EMA, no rolling window, no drift.

        Args:
            energies: Energy per frame.
            spectral_centroids: Centroid per frame.
            sub_bass_energies: Sub-bass per frame.
            has_onsets: Onset flags per frame.
            vocal_energies: Vocal energy per frame.
            drop_probabilities: Drop probability per frame.

        Returns:
            List of identical GenreFrame, one per input frame.
        """
        n = len(energies)
        if n == 0:
            return []

        # Use first 30s of features (or all if shorter)
        window = min(n, self._fps * 30)
        e_slice = energies[:window]
        c_slice = spectral_centroids[:window]
        b_slice = sub_bass_energies[:window]
        o_slice = has_onsets[:window]
        v_slice = vocal_energies[:window]
        d_slice = drop_probabilities[:window]

        # Compute aggregate features (same keys as _compute_features)
        features: dict[str, float] = {
            "energy_mean": float(np.mean(e_slice)),
            "energy_variance": float(np.var(e_slice)),
            "spectral_centroid_norm": float(
                np.mean([min(1.0, c / 10000.0) for c in c_slice])
            ),
            "sub_bass_ratio": float(np.mean(b_slice)),
            "onset_density": float(np.mean([1.0 if o else 0.0 for o in o_slice])),
            "vocal_ratio": float(np.mean(v_slice)),
            "drop_tendency": float(np.mean(d_slice)),
        }

        # Classify once
        family_weights = self._classify_family(features)
        genre_weights = self._classify_profiles(features, family_weights)
        top_family = max(family_weights, key=family_weights.get)  # type: ignore[arg-type]

        locked_frame = GenreFrame(
            family=top_family,
            family_weights=family_weights,
            genre_weights=genre_weights,
        )

        logger.info(
            "Genre locked: %s (top profile: %s @ %.1f%%)",
            top_family,
            max(genre_weights, key=genre_weights.get),  # type: ignore[arg-type]
            max(genre_weights.values()) * 100,
        )

        return [locked_frame] * n

    # ── Internal ──────────────────────────────────────────────────

    def _compute_features(self) -> dict[str, float]:
        """Compute windowed feature statistics.

        Returns:
            Feature dict matching prototype keys.
        """
        energies = list(self._energy_history) if self._energy_history else [0.0]
        return {
            "energy_mean": float(np.mean(energies)),
            "energy_variance": float(np.var(energies)),
            "spectral_centroid_norm": (
                float(np.mean(self._centroid_history)) if self._centroid_history else 0.0
            ),
            "sub_bass_ratio": (float(np.mean(self._bass_history)) if self._bass_history else 0.0),
            "onset_density": (float(np.mean(self._onset_history)) if self._onset_history else 0.0),
            "vocal_ratio": (float(np.mean(self._vocal_history)) if self._vocal_history else 0.0),
            "drop_tendency": (float(np.mean(self._drop_history)) if self._drop_history else 0.0),
        }

    def _classify_family(self, features: dict[str, float]) -> dict[str, float]:
        """Stage 1: Classify into family using softmax over similarities.

        Args:
            features: Current windowed features.

        Returns:
            Family weights summing to 1.0.
        """
        raw_scores: dict[str, float] = {}
        for family, prototype in _FAMILY_PROTOTYPES.items():
            score = self._gaussian_similarity(features, prototype)
            raw_scores[family] = score

        return self._softmax(raw_scores)

    def _classify_profiles(
        self,
        features: dict[str, float],
        family_weights: dict[str, float],
    ) -> dict[str, float]:
        """Stage 2: Classify into profiles, modulated by family weights.

        Profiles within the detected family get a boost. Profiles outside
        the family are penalized but not zeroed (allows cross-family blends).

        Args:
            features: Current windowed features.
            family_weights: Stage 1 family classification weights.

        Returns:
            Profile weights summing to 1.0.
        """
        raw_scores: dict[str, float] = {}

        for profile, prototype in _PROFILE_PROTOTYPES.items():
            # Base similarity score
            score = self._gaussian_similarity(features, prototype)

            # Family boost: multiply by the family weight that contains this profile
            family_boost = 0.0
            for family, profiles in _FAMILY_PROFILES.items():
                if profile in profiles:
                    family_boost = max(family_boost, family_weights.get(family, 0.0))

            # Blend: 70% feature similarity + 30% family membership
            modulated = 0.7 * score + 0.3 * family_boost
            raw_scores[profile] = modulated

        return self._softmax(raw_scores)

    @staticmethod
    def _gaussian_similarity(
        features: dict[str, float],
        prototype: dict[str, tuple[float, float]],
    ) -> float:
        """Compute Gaussian similarity between features and a prototype.

        Args:
            features: Current feature values.
            prototype: (center, tolerance) per feature.

        Returns:
            Average Gaussian similarity (0-1).
        """
        total = 0.0
        count = 0
        for feat_name, (center, tolerance) in prototype.items():
            value = features.get(feat_name, 0.0)
            diff = value - center
            score = float(np.exp(-(diff**2) / (2 * tolerance**2)))
            total += score
            count += 1
        return total / count if count > 0 else 0.0

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        """Apply temperature-scaled softmax to normalize scores to [0,1].

        Args:
            scores: Raw similarity scores per class.

        Returns:
            Normalized weights summing to 1.0.
        """
        if not scores:
            return {}

        values = np.array(list(scores.values()))
        # Temperature scaling
        scaled = values / self._temperature

        # Numerical stability: subtract max
        scaled = scaled - np.max(scaled)
        exp_values = np.exp(scaled)
        total = float(np.sum(exp_values))

        if total < 1e-20:
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        result: dict[str, float] = {}
        for i, key in enumerate(scores):
            result[key] = float(exp_values[i]) / total

        return result

    def _smooth_weights(
        self,
        new_weights: dict[str, float],
        prev_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply EMA smoothing to weight dictionary.

        Args:
            new_weights: Freshly computed weights.
            prev_weights: Previous smoothed weights.

        Returns:
            Smoothed weights summing to ~1.0.
        """
        smoothed: dict[str, float] = {}
        for key in new_weights:
            prev = prev_weights.get(key, new_weights[key])
            smoothed[key] = self._smoothing * prev + (1 - self._smoothing) * new_weights[key]

        # Re-normalize to ensure sum = 1.0
        total = sum(smoothed.values())
        if total > 1e-10:
            return {k: v / total for k, v in smoothed.items()}
        return new_weights
