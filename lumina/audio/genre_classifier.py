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

from lumina.audio.source_separator import StemSet

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
        stems: StemSet | None = None,
        genre_override: str | None = None,
    ) -> list[GenreFrame]:
        """Classify genre for a file with a single locked result.

        Computes aggregate features from the first 30s of the track,
        classifies once, and returns the same GenreFrame for every frame.
        No EMA, no rolling window, no drift.

        When ``stems`` is provided, uses demucs stem energy ratios for
        family classification and a decision tree for profile selection.

        When ``genre_override`` is provided, skips all classification and
        locks to the specified profile at weight 1.0.

        Args:
            energies: Energy per frame.
            spectral_centroids: Centroid per frame.
            sub_bass_energies: Sub-bass per frame.
            has_onsets: Onset flags per frame.
            vocal_energies: Vocal energy per frame.
            drop_probabilities: Drop probability per frame.
            stems: Optional demucs stem separation for stem-based features.
            genre_override: Optional profile name to lock to (skips classification).

        Returns:
            List of identical GenreFrame, one per input frame.
        """
        n = len(energies)
        if n == 0:
            return []

        # Genre override — skip all classification
        if genre_override is not None:
            if genre_override not in PROFILE_NAMES:
                logger.warning("Unknown genre override: %s", genre_override)
            genre_weights = {p: 0.0 for p in PROFILE_NAMES}
            genre_weights[genre_override] = 1.0
            # Determine primary family from profile
            family = "hybrid"
            for fam, profiles in _FAMILY_PROFILES.items():
                if genre_override in profiles:
                    family = fam
                    break
            family_weights = {f: 0.0 for f in FAMILY_NAMES}
            family_weights[family] = 1.0
            locked_frame = GenreFrame(
                family=family,
                family_weights=family_weights,
                genre_weights=genre_weights,
            )
            logger.info("Genre override: %s (family: %s)", genre_override, family)
            return [locked_frame] * n

        # Use first 30s of features (or all if shorter)
        window = min(n, self._fps * 30)
        e_slice = energies[:window]
        c_slice = spectral_centroids[:window]
        b_slice = sub_bass_energies[:window]
        o_slice = has_onsets[:window]
        v_slice = vocal_energies[:window]
        d_slice = drop_probabilities[:window]

        # Compute aggregate spectral features (same keys as _compute_features)
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

        # Check if stems are usable (not passthrough fallback where all stems
        # are identical, which happens when demucs fails)
        use_stems = False
        if stems is not None:
            if not np.allclose(stems.drums, stems.vocals, atol=1e-6):
                use_stems = True
            else:
                logger.info("Stems appear to be passthrough (all identical), "
                            "falling back to spectral-only classification")

        if use_stems:
            assert stems is not None
            stem_features = self._compute_stem_features(stems)

            # Stage 1: Family from stems (more discriminating)
            family_weights = self._classify_family_from_stems(stem_features)

            # Stage 2: Decision-tree profile classification
            genre_weights = self._classify_profile_from_stems(
                stem_features, family_weights
            )

            logger.info(
                "Stem features: other=%.2f vocal=%.2f(conf=%.2f) "
                "drum_reg=%.2f(ac=%.2f,ioi=%.2f) bass=%.2f",
                stem_features["stem_other_ratio"],
                stem_features["stem_vocal_ratio"],
                stem_features["_vocal_confidence"],
                stem_features["drum_regularity"],
                stem_features["_drum_reg_autocorr"],
                stem_features["_drum_reg_ioi"],
                stem_features["bass_character"],
            )
        else:
            # No stems — original spectral-only path
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

    # ── Stem-based classification ────────────────────────────────

    @staticmethod
    def _compute_stem_features(stems: StemSet) -> dict[str, float]:
        """Compute classification features from demucs stem separation.

        Args:
            stems: Four-stem separation result from demucs.

        Returns:
            Dict with stem-derived features plus debug keys (``_`` prefix):
            - stem_vocal_ratio: RMS(vocals) / RMS(full_mix), adjusted by confidence
            - stem_other_ratio: RMS(other) / RMS(full_mix)
            - stem_drum_ratio: RMS(drums) / RMS(full_mix)
            - drum_regularity: max(autocorrelation, IOI) regularity (0-1)
            - bass_character: spectral centroid of bass below 500 Hz (0-1)
            - onset_regularity: IOI-based onset regularity (0-1)
            - _vocal_confidence: confidence that vocal stem is real voice
            - _drum_reg_autocorr: autocorrelation-based drum regularity
            - _drum_reg_ioi: IOI-based drum regularity
        """
        sr = stems.sample_rate

        # RMS of each stem
        def _rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x**2)))

        rms_drums = _rms(stems.drums)
        rms_bass = _rms(stems.bass)
        rms_vocals = _rms(stems.vocals)
        rms_other = _rms(stems.other)

        # Full mix = sum of all stems
        full_mix = stems.drums + stems.bass + stems.vocals + stems.other
        rms_full = max(_rms(full_mix), 1e-10)

        stem_vocal_ratio = min(rms_vocals / rms_full, 1.0)
        stem_other_ratio = min(rms_other / rms_full, 1.0)
        stem_drum_ratio = min(rms_drums / rms_full, 1.0)

        logger.info(
            "STEM drum_rms=%.4f vocal_rms=%.4f other_rms=%.4f bass_rms=%.4f",
            rms_drums, rms_vocals, rms_other, rms_bass,
        )

        # ── Vocal confidence ──
        # Demucs puts accordion/violin/synth leads into the vocal stem
        # when their harmonics resemble voice. Three independent signals
        # multiplied together to detect non-vocal content:
        # 1. Silence fraction — voice has pauses, sustained instruments don't
        # 2. Vocal dominance — very high ratio means instrument in vocal stem
        # 3. Correlation — high correlation means instrument bleed
        vocal_confidence = 1.0
        silence_fraction = -1.0  # sentinel for logging
        vocal_other_corr = 0.0
        vocal_dominance = rms_vocals / max(rms_vocals + rms_other, 1e-10)
        analysis_len = min(sr * 30, len(stems.vocals), len(stems.other))
        if analysis_len > sr:
            vocal_clip = stems.vocals[:analysis_len]
            other_clip = stems.other[:analysis_len]

            # Signal 1: Silence fraction of vocal stem
            # Voice has frequent pauses (>25% silence); sustained instruments
            # like accordion play near-continuously (<10%)
            hop_conf = 1024  # ~23ms windows
            n_conf_frames = analysis_len // hop_conf
            if n_conf_frames > 10:
                frame_energies = np.array(
                    [_rms(vocal_clip[i * hop_conf : (i + 1) * hop_conf])
                     for i in range(n_conf_frames)]
                )
                max_frame_e = float(np.max(frame_energies))
                if max_frame_e > 1e-10:
                    silence_threshold = 0.15 * max_frame_e
                    silence_fraction = float(
                        np.mean(frame_energies < silence_threshold)
                    )
                    # Low silence fraction (<0.15) → likely sustained instrument
                    if silence_fraction < 0.15:
                        vocal_confidence *= max(
                            0.4, silence_fraction / 0.15
                        )

            # Signal 2: Vocal dominance ratio
            # Very high dominance means demucs put most melodic energy in
            # vocal stem — suspicious for real vocals which have backing
            if vocal_dominance > 0.65:
                vocal_confidence *= max(
                    0.5, 1.0 - (vocal_dominance - 0.65) / 0.30 * 0.5
                )

            # Signal 3: Correlation (supplementary for instrument bleed)
            window_size = sr * 5
            n_corr_windows = max(1, analysis_len // window_size)
            correlations: list[float] = []
            for i in range(n_corr_windows):
                start = i * window_size
                end = min(start + window_size, analysis_len)
                v = vocal_clip[start:end]
                o = other_clip[start:end]
                if np.std(v) > 1e-10 and np.std(o) > 1e-10:
                    corr = float(np.abs(np.corrcoef(v, o)[0, 1]))
                    correlations.append(corr)
            vocal_other_corr = (
                float(np.mean(correlations)) if correlations else 0.0
            )
            if vocal_other_corr > 0.6:
                vocal_confidence *= max(
                    0.3, 1.0 - (vocal_other_corr - 0.6) / 0.4 * 0.7
                )

            # Floor
            vocal_confidence = max(0.3, vocal_confidence)

        logger.info(
            "STEM vocal_dominance=%.3f silence_fraction=%.3f "
            "vocal_other_corr=%.3f",
            vocal_dominance, silence_fraction, vocal_other_corr,
        )

        stem_vocal_ratio *= vocal_confidence

        # ── Drum regularity ──
        # Two methods: autocorrelation and IOI (inter-onset interval),
        # take max for robustness. Operates on the RMS envelope directly
        # (not diff-based onset envelope) for less noisy periodic detection.
        drum_reg_autocorr = 0.0
        drum_reg_ioi = 0.0
        n_drum_peaks = 0
        env_max = 0.0
        drum_clip = stems.drums[: sr * 30]
        if len(drum_clip) >= sr * 4:
            hop = 512  # ~11.6ms at 44100
            n_env_frames = len(drum_clip) // hop
            if n_env_frames > 2:
                # RMS energy envelope (NOT diff-based — less noisy)
                env = np.array(
                    [_rms(drum_clip[i * hop : (i + 1) * hop])
                     for i in range(n_env_frames)]
                )
                env_max = float(np.max(env))

                # Method 1: Autocorrelation on RMS envelope directly
                if len(env) > 1 and env_max > 1e-10:
                    centered = env - np.mean(env)
                    n_fft = 2 ** int(np.ceil(np.log2(len(centered) * 2)))
                    fft_env = np.fft.rfft(centered, n=n_fft)
                    autocorr = np.fft.irfft(fft_env * np.conj(fft_env))
                    autocorr = autocorr[: len(centered)]
                    if autocorr[0] > 1e-10:
                        autocorr = autocorr / autocorr[0]
                    # BPM range 60-200 → lag in envelope frames
                    env_fps = sr / hop
                    lag_min = max(1, int(env_fps * 60 / 200))  # 200 BPM
                    lag_max = min(
                        len(autocorr) - 1, int(env_fps * 60 / 60)
                    )  # 60 BPM
                    if lag_max > lag_min:
                        region = autocorr[lag_min : lag_max + 1]
                        drum_reg_autocorr = float(
                            np.clip(np.max(region), 0.0, 1.0)
                        )
                        best_lag = lag_min + int(np.argmax(region))
                        best_bpm = (
                            env_fps * 60 / best_lag if best_lag > 0 else 0
                        )
                        logger.debug(
                            "STEM autocorr: peak=%.3f lag=%d (%.1f BPM)",
                            drum_reg_autocorr, best_lag, best_bpm,
                        )

                # Method 2: IOI with absolute threshold + min distance
                if env_max > 1e-10:
                    peak_threshold = 0.3 * env_max
                    min_distance = 20  # ~230ms at hop=512, sr=44100
                    peaks: list[int] = []
                    i = 0
                    while i < len(env):
                        if env[i] > peak_threshold:
                            # Find local max in this above-threshold region
                            j = i
                            while j < len(env) and env[j] > peak_threshold:
                                j += 1
                            peak_pos = i + int(np.argmax(env[i:j]))
                            peaks.append(peak_pos)
                            i = peak_pos + min_distance  # skip ahead
                        else:
                            i += 1
                    n_drum_peaks = len(peaks)
                    if len(peaks) >= 3:
                        intervals = np.diff(peaks).astype(float)
                        mean_interval = float(np.mean(intervals))
                        if mean_interval > 0:
                            cv = float(np.std(intervals) / mean_interval)
                            drum_reg_ioi = float(
                                np.clip(1.0 - cv, 0.0, 1.0)
                            )
                            logger.debug(
                                "STEM IOI: n_peaks=%d mean_interval=%.1f "
                                "cv=%.3f regularity=%.3f",
                                len(peaks), mean_interval, cv, drum_reg_ioi,
                            )

        drum_regularity = max(drum_reg_autocorr, drum_reg_ioi)

        logger.info(
            "STEM drum_reg_autocorr=%.3f drum_reg_ioi=%.3f "
            "vocal_confidence=%.3f",
            drum_reg_autocorr, drum_reg_ioi, vocal_confidence,
        )
        logger.info(
            "STEM drum_peaks=%d drum_peak_amp=%.4f",
            n_drum_peaks, env_max,
        )

        # ── Bass character ──
        # Spectral centroid of bass stem limited to frequencies below 500 Hz.
        # Uses windowed FFT (8192-sample windows ≈ 186ms) for stable estimates.
        # The old code used a single FFT over the full spectrum — the centroid
        # was pulled above 500 Hz by harmonics/leakage, clipping to 1.0.
        bass_clip = stems.bass[: sr * 30]
        bass_character = 0.3  # default
        if len(bass_clip) > 0:
            win_size = min(8192, len(bass_clip))
            n_bass_windows = max(1, len(bass_clip) // win_size)
            centroids: list[float] = []
            for i in range(n_bass_windows):
                start = i * win_size
                chunk = bass_clip[start : start + win_size]
                if len(chunk) < win_size:
                    break
                spectrum = np.abs(np.fft.rfft(chunk))
                freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
                # Only consider frequencies below 500 Hz
                mask = freqs <= 500.0
                low_spectrum = spectrum[mask]
                low_freqs = freqs[mask]
                total_mag = float(np.sum(low_spectrum))
                if total_mag > 1e-10:
                    centroid_hz = float(
                        np.sum(low_freqs * low_spectrum) / total_mag
                    )
                    centroids.append(centroid_hz)
            if centroids:
                avg_centroid = float(np.mean(centroids))
                bass_character = float(np.clip(avg_centroid / 500.0, 0.0, 1.0))
                logger.debug(
                    "STEM bass centroid=%.1f Hz -> character=%.3f",
                    avg_centroid, bass_character,
                )

        # ── Onset regularity (reuses IOI from drum analysis) ──
        onset_regularity = drum_reg_ioi if drum_reg_ioi > 0 else 0.5

        return {
            "stem_vocal_ratio": stem_vocal_ratio,
            "stem_other_ratio": stem_other_ratio,
            "stem_drum_ratio": stem_drum_ratio,
            "drum_regularity": drum_regularity,
            "bass_character": bass_character,
            "onset_regularity": onset_regularity,
            "_vocal_confidence": vocal_confidence,
            "_drum_reg_autocorr": drum_reg_autocorr,
            "_drum_reg_ioi": drum_reg_ioi,
            "_rms_bass": rms_bass,
            "_rms_drums": rms_drums,
        }

    @staticmethod
    def _classify_family_from_stems(
        stem_features: dict[str, float],
    ) -> dict[str, float]:
        """Stage 1 family classification using stem energy ratios.

        Uses soft scoring based on stem ratios rather than spectral prototypes.
        More discriminating for electronic vs hip-hop separation.

        Args:
            stem_features: Features from _compute_stem_features().

        Returns:
            Family weights summing to 1.0.
        """
        other_ratio = stem_features["stem_other_ratio"]
        vocal_ratio = stem_features["stem_vocal_ratio"]
        drum_regularity = stem_features["drum_regularity"]
        rms_bass = stem_features.get("_rms_bass", 0.0)
        rms_drums = stem_features.get("_rms_drums", 0.0)

        # Electronic requires high other_ratio (synths/pads) as a gate;
        # drum_regularity only adds bonus.  Without this gating, a track
        # with moderate drum regularity but low synth energy could be
        # mis-classified as electronic.
        other_gate = min(other_ratio / 0.35, 1.0)
        electronic = other_gate * (
            0.5 + min(drum_regularity / 0.7, 1.0) * 0.5
        )
        hiphop = (
            min(vocal_ratio / 0.25, 1.0) * 0.5
            + max(0.0, 1.0 - other_ratio / 0.35) * 0.5
        )

        # Bass dominance boost: bass-heavy tracks (808s, sub-bass) are a
        # strong hip-hop signal.  Without this, tracks like Cocaine Nose
        # (bass_rms >> drum_rms) land in hybrid instead of hiphop.
        if rms_bass > rms_drums * 0.8:
            hiphop += 0.3

        hybrid = 1.0 - abs(electronic - hiphop)

        # Softmax normalization
        scores = np.array([hiphop, electronic, hybrid])
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / 0.3)  # temperature = 0.3
        total = float(np.sum(exp_scores))
        if total < 1e-20:
            return {f: 1.0 / 3 for f in FAMILY_NAMES}

        return {
            "hiphop_rap": float(exp_scores[0]) / total,
            "electronic": float(exp_scores[1]) / total,
            "hybrid": float(exp_scores[2]) / total,
        }

    @staticmethod
    def _classify_profile_from_stems(
        stem_features: dict[str, float],
        family_weights: dict[str, float],
    ) -> dict[str, float]:
        """Stage 2: Decision-tree profile classification from stem features.

        Uses hard splits on stem features rather than prototype distance,
        producing decisive winners (0.40-0.50) instead of uniform blends.

        Args:
            stem_features: Features from _compute_stem_features().
            family_weights: Family weights from _classify_family_from_stems().

        Returns:
            Profile weights summing to 1.0.
        """
        top_family = max(family_weights, key=family_weights.get)  # type: ignore[arg-type]
        drum_reg = stem_features["drum_regularity"]
        vocal_conf = stem_features["_vocal_confidence"]
        bass_char = stem_features["bass_character"]
        other_ratio = stem_features["stem_other_ratio"]
        rms_bass = stem_features.get("_rms_bass", 0.0)
        rms_drums = stem_features.get("_rms_drums", 0.0)

        if top_family == "electronic":
            if drum_reg > 0.5:
                weights = {
                    "festival_edm": 0.45, "uk_bass": 0.20,
                    "theatrical": 0.10, "euro_alt": 0.07,
                    "rage_trap": 0.05, "psych_rnb": 0.05,
                    "french_hard": 0.05, "french_melodic": 0.03,
                }
            else:
                weights = {
                    "theatrical": 0.40, "festival_edm": 0.20,
                    "uk_bass": 0.10, "euro_alt": 0.10,
                    "psych_rnb": 0.05, "rage_trap": 0.05,
                    "french_melodic": 0.05, "french_hard": 0.05,
                }
        elif top_family == "hiphop_rap":
            if vocal_conf > 0.8 and bass_char < 0.35:
                weights = {
                    "rage_trap": 0.45, "french_hard": 0.20,
                    "psych_rnb": 0.10, "french_melodic": 0.05,
                    "euro_alt": 0.05, "theatrical": 0.05,
                    "festival_edm": 0.05, "uk_bass": 0.05,
                }
            elif other_ratio < 0.2:
                weights = {
                    "psych_rnb": 0.45, "french_melodic": 0.20,
                    "rage_trap": 0.10, "theatrical": 0.05,
                    "euro_alt": 0.05, "french_hard": 0.05,
                    "festival_edm": 0.05, "uk_bass": 0.05,
                }
            else:
                weights = {
                    "euro_alt": 0.35, "french_melodic": 0.20,
                    "psych_rnb": 0.15, "rage_trap": 0.10,
                    "french_hard": 0.05, "theatrical": 0.05,
                    "festival_edm": 0.05, "uk_bass": 0.05,
                }
        else:  # hybrid
            if rms_bass > rms_drums * 0.8:
                # Bass-dominant → likely hip-hop that didn't score high enough
                weights = {
                    "rage_trap": 0.40, "french_hard": 0.20,
                    "psych_rnb": 0.10, "uk_bass": 0.10,
                    "euro_alt": 0.05, "theatrical": 0.05,
                    "festival_edm": 0.05, "french_melodic": 0.05,
                }
            elif drum_reg > 0.5:
                weights = {
                    "festival_edm": 0.35, "uk_bass": 0.25,
                    "theatrical": 0.10, "rage_trap": 0.10,
                    "euro_alt": 0.05, "psych_rnb": 0.05,
                    "french_melodic": 0.05, "french_hard": 0.05,
                }
            else:
                weights = {
                    "euro_alt": 0.30, "psych_rnb": 0.20,
                    "theatrical": 0.15, "rage_trap": 0.10,
                    "french_melodic": 0.10, "uk_bass": 0.05,
                    "festival_edm": 0.05, "french_hard": 0.05,
                }

        return weights

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
