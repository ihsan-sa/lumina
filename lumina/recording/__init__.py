"""LUMINA show recording and playback module.

Provides compact binary recording of fixture command streams and deterministic
replay without re-analysis. Recordings are tied to a specific audio file via
SHA256 hash for integrity verification.
"""

from lumina.recording.player import ShowPlayer
from lumina.recording.recorder import ShowRecorder, hash_audio_file

__all__ = ["ShowPlayer", "ShowRecorder", "hash_audio_file"]
