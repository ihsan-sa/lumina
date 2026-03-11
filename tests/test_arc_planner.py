"""Tests for the ArcPlanner analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from lumina.analysis.arc_planner import ArcFrame, ArcPlanner
from lumina.analysis.layer_tracker import LayerFrame
from lumina.audio.energy_tracker import EnergyFrame
from lumina.audio.structural_analyzer import Section, StructuralMap


def _make_energy_frames(n: int, energy: float = 0.5) -> list[EnergyFrame]:
    return [
        EnergyFrame(
            energy=energy,
            energy_derivative=0.0,
            spectral_centroid=2000.0,
            sub_bass_energy=0.3,
        )
        for _ in range(n)
    ]


def _make_layer_frames(n: int, count: int = 2) -> list[LayerFrame]:
    return [
        LayerFrame(active_count=count, layer_mask={}, layer_change=None)
        for _ in range(n)
    ]


def _make_structural_map(sections: list[tuple[float, float, str]]) -> StructuralMap:
    duration = max(e for _, e, _ in sections) if sections else 0.0
    return StructuralMap(
        sections=[
            Section(
                start_time=s, end_time=e, segment_type=t,
                confidence=0.8, features={},
            )
            for s, e, t in sections
        ],
        duration=duration,
    )


class TestArcPlanner:
    """Tests for headroom budgeting."""

    def test_single_section_gets_full_headroom(self) -> None:
        planner = ArcPlanner(fps=60)
        energy = _make_energy_frames(600, energy=0.7)
        layers = _make_layer_frames(600, count=3)
        smap = _make_structural_map([(0.0, 10.0, "verse")])

        result = planner.plan(energy, layers, smap)
        assert len(result) == 600
        # Single section → normalized to 1.0
        assert result[-1].headroom == 1.0

    def test_intro_gets_lower_headroom_than_drop(self) -> None:
        planner = ArcPlanner(fps=60)
        n = 1200  # 20 seconds at 60fps

        # Intro: low energy, 1 layer. Drop: high energy, 4 layers.
        energy = []
        layers = []
        for i in range(n):
            t = i / 60.0
            if t < 10.0:
                energy.append(EnergyFrame(energy=0.2, energy_derivative=0.0,
                                          spectral_centroid=1000.0, sub_bass_energy=0.1))
                layers.append(LayerFrame(active_count=1, layer_mask={}, layer_change=None))
            else:
                energy.append(EnergyFrame(energy=0.9, energy_derivative=0.0,
                                          spectral_centroid=4000.0, sub_bass_energy=0.8))
                layers.append(LayerFrame(active_count=4, layer_mask={}, layer_change=None))

        smap = _make_structural_map([
            (0.0, 10.0, "intro"),
            (10.0, 20.0, "drop"),
        ])

        result = planner.plan(energy, layers, smap)
        # Intro headroom should be lower than drop headroom
        intro_hr = result[300].headroom  # t=5s
        drop_hr = result[900].headroom  # t=15s
        assert intro_hr < drop_hr

    def test_headroom_floor(self) -> None:
        """Headroom never goes below 0.15."""
        planner = ArcPlanner(fps=60)
        energy = _make_energy_frames(600, energy=0.01)
        layers = _make_layer_frames(600, count=0)
        smap = _make_structural_map([
            (0.0, 5.0, "intro"),
            (5.0, 10.0, "drop"),
        ])

        # Make drop section much higher energy
        for i in range(300, 600):
            energy[i] = EnergyFrame(energy=1.0, energy_derivative=0.0,
                                    spectral_centroid=5000.0, sub_bass_energy=1.0)

        result = planner.plan(energy, layers, smap)
        for frame in result:
            assert frame.headroom >= 0.15

    def test_empty_input(self) -> None:
        planner = ArcPlanner(fps=60)
        result = planner.plan([], [], StructuralMap(sections=[], duration=0.0))
        assert result == []
