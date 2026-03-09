"""Demucs-based source separation for isolating drums, bass, vocals, and other.

Separates audio into four stems using Meta's Hybrid Transformer Demucs
(htdemucs) model. The isolated stems improve downstream analysis:
- Drums stem → BeatDetector, OnsetDetector (cleaner transients)
- Vocals stem → VocalDetector (no instrumental bleed)
- Bass stem → EnergyTracker sub-bass (already band-limited, no FFT needed)

Uses the demucs 4.0.x API: ``demucs.pretrained.get_model`` to load the
model and ``demucs.apply.apply_model`` to run separation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Module-level cache (same pattern as beat_detector._beat_nn_cache)
_separator_cache: SourceSeparator | None = None


@dataclass(slots=True)
class StemSet:
    """Four-stem separation result.

    Args:
        drums: Mono float32 drum stem.
        bass: Mono float32 bass stem.
        vocals: Mono float32 vocal stem.
        other: Mono float32 other/residual stem.
        sample_rate: Sample rate of all stems.
    """

    drums: np.ndarray
    bass: np.ndarray
    vocals: np.ndarray
    other: np.ndarray
    sample_rate: int


def _make_passthrough_stems(audio: np.ndarray, sr: int) -> StemSet:
    """Return full mix as all stems (fallback when separation fails).

    Args:
        audio: Mono float32 audio.
        sr: Sample rate.

    Returns:
        StemSet with all stems set to the full mix.
    """
    return StemSet(
        drums=audio.copy(),
        bass=audio.copy(),
        vocals=audio.copy(),
        other=audio.copy(),
        sample_rate=sr,
    )


class SourceSeparator:
    """Demucs source separator with GPU acceleration.

    Args:
        model_name: Demucs model name (default: htdemucs).
        device: Torch device string. None = auto-detect (CUDA if available).
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model: object | None = None
        self._sources: list[str] | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the demucs model on first use."""
        if self._model is not None:
            return

        from demucs.pretrained import get_model  # type: ignore[import-untyped]

        logger.info(
            "Loading demucs model '%s' on %s",
            self._model_name,
            self._device,
        )
        model = get_model(self._model_name)
        self._sources = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        model.to(self._device)
        self._model = model

    def separate(self, audio: np.ndarray, sr: int = 44100) -> StemSet:
        """Separate audio into four stems.

        Args:
            audio: Mono float32 audio, shape ``(N,)``.
            sr: Sample rate in Hz.

        Returns:
            StemSet with drums, bass, vocals, other stems.
        """
        if len(audio) < sr:
            logger.warning("Audio shorter than 1s (%d samples), skipping separation", len(audio))
            return _make_passthrough_stems(audio, sr)

        try:
            self._ensure_loaded()
            return self._run_separation(audio, sr)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during demucs separation — falling back to full mix")
            torch.cuda.empty_cache()
            return _make_passthrough_stems(audio, sr)

    def _run_separation(self, audio: np.ndarray, sr: int) -> StemSet:
        """Run demucs separation.

        Args:
            audio: Mono float32 audio.
            sr: Sample rate.

        Returns:
            StemSet with separated stems.
        """
        from demucs.apply import apply_model  # type: ignore[import-untyped]

        n_samples = len(audio)

        # Demucs expects (batch, channels, samples) — mono duplicated to stereo
        mix = torch.from_numpy(audio).float().unsqueeze(0).expand(2, -1).unsqueeze(0)
        # mix shape: (1, 2, N)

        # apply_model returns (batch, sources, channels, samples)
        with torch.no_grad():
            estimates = apply_model(
                self._model,
                mix.to(self._device),
                device=self._device,
            )
        # estimates shape: (1, n_sources, 2, N)

        # Map source index to name
        sources = self._sources or ["drums", "bass", "other", "vocals"]
        stem_names = ("drums", "bass", "vocals", "other")
        stems: dict[str, np.ndarray] = {}

        for name in stem_names:
            if name in sources:
                idx = sources.index(name)
                # (1, 2, N) -> mean over channels -> (N,) mono
                stem_tensor = estimates[0, idx]  # (2, N)
                mono = stem_tensor.mean(dim=0).cpu().numpy().astype(np.float32)
                # Truncate or pad to match input length exactly
                if len(mono) > n_samples:
                    mono = mono[:n_samples]
                elif len(mono) < n_samples:
                    mono = np.pad(mono, (0, n_samples - len(mono)))
                stems[name] = mono
            else:
                logger.warning("Stem '%s' not found in model sources %s", name, sources)
                stems[name] = audio.copy()

        return StemSet(
            drums=stems["drums"],
            bass=stems["bass"],
            vocals=stems["vocals"],
            other=stems["other"],
            sample_rate=sr,
        )


def get_separator(
    model_name: str = "htdemucs",
    device: str | None = None,
) -> SourceSeparator:
    """Get or create the cached SourceSeparator instance.

    Args:
        model_name: Demucs model name.
        device: Torch device string.

    Returns:
        Cached SourceSeparator instance.
    """
    global _separator_cache
    if _separator_cache is None:
        _separator_cache = SourceSeparator(model_name=model_name, device=device)
    return _separator_cache
