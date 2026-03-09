"""Tests for the LUMINA source separation module.

Tests verify the StemSet data contract, passthrough fallback, and
separator interface without requiring the actual demucs model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lumina.audio.source_separator import (
    SourceSeparator,
    StemSet,
    _make_passthrough_stems,
    get_separator,
)


# ── StemSet dataclass ────────────────────────────────────────────────


class TestStemSet:
    def test_fields(self) -> None:
        drums = np.zeros(100, dtype=np.float32)
        bass = np.ones(100, dtype=np.float32)
        vocals = np.zeros(100, dtype=np.float32)
        other = np.zeros(100, dtype=np.float32)
        s = StemSet(drums=drums, bass=bass, vocals=vocals, other=other, sample_rate=44100)
        assert s.sample_rate == 44100
        np.testing.assert_array_equal(s.bass, bass)

    def test_stem_lengths_match(self) -> None:
        n = 44100
        audio = np.random.randn(n).astype(np.float32)
        s = _make_passthrough_stems(audio, 44100)
        assert len(s.drums) == n
        assert len(s.bass) == n
        assert len(s.vocals) == n
        assert len(s.other) == n


# ── Passthrough fallback ─────────────────────────────────────────────


class TestPassthrough:
    def test_passthrough_copies_audio(self) -> None:
        """Passthrough should return copies, not references."""
        audio = np.random.randn(1000).astype(np.float32)
        s = _make_passthrough_stems(audio, 44100)
        # Modify original — stems should be unaffected
        original_drums = s.drums.copy()
        audio[0] = 999.0
        np.testing.assert_array_equal(s.drums, original_drums)

    def test_passthrough_preserves_sr(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        s = _make_passthrough_stems(audio, 22050)
        assert s.sample_rate == 22050


# ── Short audio fallback ─────────────────────────────────────────────


class TestShortAudioFallback:
    def test_short_audio_returns_passthrough(self) -> None:
        """Audio shorter than 1s should skip separation."""
        sep = SourceSeparator(device="cpu")
        short_audio = np.zeros(100, dtype=np.float32)
        result = sep.separate(short_audio, sr=44100)
        assert len(result.drums) == 100
        assert result.sample_rate == 44100


# ── Separator cache ──────────────────────────────────────────────────


class TestSeparatorCache:
    def test_get_separator_returns_instance(self) -> None:
        """get_separator should return a SourceSeparator."""
        import lumina.audio.source_separator as mod

        # Reset cache
        mod._separator_cache = None
        sep = get_separator(device="cpu")
        assert isinstance(sep, SourceSeparator)

    def test_get_separator_caches(self) -> None:
        """Repeated calls should return the same instance."""
        import lumina.audio.source_separator as mod

        mod._separator_cache = None
        sep1 = get_separator(device="cpu")
        sep2 = get_separator(device="cpu")
        assert sep1 is sep2
        # Clean up
        mod._separator_cache = None


# ── Mock separation ──────────────────────────────────────────────────


class TestMockSeparation:
    def _make_mock_separator(self) -> SourceSeparator:
        """Create a SourceSeparator with a mocked demucs model.

        Patches ``demucs.apply.apply_model`` so the real model isn't needed.
        The mock returns a tensor of shape (1, 4, 2, N) matching htdemucs
        output format: sources=['drums', 'bass', 'other', 'vocals'].
        """
        n = 44100 * 3  # 3 seconds
        sep = SourceSeparator(device="cpu")

        # Set model and sources as if _ensure_loaded() ran
        mock_model = MagicMock()
        sep._model = mock_model
        sep._sources = ["drums", "bass", "other", "vocals"]

        # Build fake apply_model output: (1, 4, 2, N)
        fake_output = torch.randn(1, 4, 2, n)
        self._patch = patch(
            "demucs.apply.apply_model",
            return_value=fake_output,
        )
        return sep

    def _start_patch(self) -> SourceSeparator:
        sep = self._make_mock_separator()
        self._patch.start()
        return sep

    def _stop_patch(self) -> None:
        self._patch.stop()

    def test_separation_returns_correct_length(self) -> None:
        """Separated stems should match input length."""
        sep = self._start_patch()
        try:
            audio = np.random.randn(44100 * 3).astype(np.float32)
            result = sep.separate(audio, sr=44100)

            assert len(result.drums) == len(audio)
            assert len(result.bass) == len(audio)
            assert len(result.vocals) == len(audio)
            assert len(result.other) == len(audio)
        finally:
            self._stop_patch()

    def test_separation_returns_float32(self) -> None:
        """All stems should be float32."""
        sep = self._start_patch()
        try:
            audio = np.random.randn(44100 * 3).astype(np.float32)
            result = sep.separate(audio, sr=44100)

            assert result.drums.dtype == np.float32
            assert result.bass.dtype == np.float32
            assert result.vocals.dtype == np.float32
            assert result.other.dtype == np.float32
        finally:
            self._stop_patch()

    def test_separation_mono_output(self) -> None:
        """All stems should be 1D (mono)."""
        sep = self._start_patch()
        try:
            audio = np.random.randn(44100 * 3).astype(np.float32)
            result = sep.separate(audio, sr=44100)

            assert result.drums.ndim == 1
            assert result.bass.ndim == 1
            assert result.vocals.ndim == 1
            assert result.other.ndim == 1
        finally:
            self._stop_patch()
