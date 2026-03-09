"""Core data contracts for the LUMINA audio analysis pipeline.

These dataclasses define the interface between the audio engine and
the lighting engine. MusicState is produced at 60fps by the audio
pipeline and consumed by the lighting decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MusicState:
    """Complete musical state at a single point in time.

    Produced by the audio analysis engine at 60fps, consumed by the
    lighting engine to generate fixture commands.

    Args:
        timestamp: Current time in seconds.
        bpm: Current tempo estimate.
        beat_phase: 0.0-1.0 position within current beat.
        bar_phase: 0.0-1.0 position within current bar.
        is_beat: True on exact beat frames.
        is_downbeat: True on bar downbeats.
        energy: 0.0-1.0 overall energy level.
        energy_derivative: Rising (+) or falling (-) energy.
        segment: Song section identifier.
        genre_weights: Profile name to weight mapping (sum to 1.0).
        vocal_energy: 0.0-1.0 vocal presence level.
        spectral_centroid: Brightness indicator (Hz).
        sub_bass_energy: Sub-bass (20-80Hz) energy level.
        onset_type: Detected transient type, or None.
        drop_probability: 0.0-1.0 probability of drop in next 1-4 bars.
    """

    timestamp: float = 0.0
    bpm: float = 120.0
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    is_beat: bool = False
    is_downbeat: bool = False
    energy: float = 0.0
    energy_derivative: float = 0.0
    segment: str = "verse"
    genre_weights: dict[str, float] = field(default_factory=dict)
    vocal_energy: float = 0.0
    spectral_centroid: float = 0.0
    sub_bass_energy: float = 0.0
    onset_type: str | None = None
    drop_probability: float = 0.0
