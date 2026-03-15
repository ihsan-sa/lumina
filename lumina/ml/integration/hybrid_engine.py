"""Hybrid lighting engine blending ML model predictions with rule-based profiles.

The HybridLightingEngine wraps the existing rule-based LightingEngine
and an ML-based LightingInferenceEngine, blending their outputs at a
configurable weight.  This allows gradual adoption of the ML model
while maintaining the reliable rule-based baseline.

Confidence-based weight adjustment: when the model output is uncertain
(low variance, all-zero, or NaN), the engine automatically reduces
ml_weight toward 0 for that frame.

Automatic fallback: if the ML model produces invalid output (NaN or
all-zeros for >2 seconds), the engine falls back to pure rule-based
for 5 seconds before retrying.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.engine import LightingEngine
from lumina.ml.integration.intent_mapper import intent_to_commands
from lumina.ml.model.inference import LightingInferenceEngine

logger = logging.getLogger(__name__)

# Duration thresholds for automatic fallback.
_INVALID_OUTPUT_THRESHOLD_S = 2.0  # If bad output for this long, trigger fallback.
_FALLBACK_DURATION_S = 5.0  # Stay in fallback mode for this long.


def _clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """Clamp an integer to [lo, hi].

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))


def _blend_commands(
    rule_cmd: FixtureCommand,
    ml_cmd: FixtureCommand,
    ml_weight: float,
) -> FixtureCommand:
    """Blend two fixture commands by weighted average.

    Continuous channels (RGB, white, strobe_intensity, special) are
    linearly interpolated.  strobe_rate uses the ML value if ml_weight
    > 0.5, otherwise the rule value (rate doesn't blend well linearly).

    Args:
        rule_cmd: Command from the rule-based engine.
        ml_cmd: Command from the ML model.
        ml_weight: Blend factor (0.0 = pure rules, 1.0 = pure ML).

    Returns:
        Blended FixtureCommand.
    """
    rw = 1.0 - ml_weight
    mw = ml_weight

    def _lerp(a: int, b: int) -> int:
        return _clamp(round(a * rw + b * mw))

    return FixtureCommand(
        fixture_id=rule_cmd.fixture_id,
        red=_lerp(rule_cmd.red, ml_cmd.red),
        green=_lerp(rule_cmd.green, ml_cmd.green),
        blue=_lerp(rule_cmd.blue, ml_cmd.blue),
        white=_lerp(rule_cmd.white, ml_cmd.white),
        strobe_rate=ml_cmd.strobe_rate if ml_weight > 0.5 else rule_cmd.strobe_rate,
        strobe_intensity=_lerp(rule_cmd.strobe_intensity, ml_cmd.strobe_intensity),
        special=_lerp(rule_cmd.special, ml_cmd.special),
    )


class HybridLightingEngine:
    """Blends ML model predictions with rule-based profile output.

    The engine runs both the rule-based LightingEngine and the ML
    LightingInferenceEngine, then blends their outputs per-fixture.

    Features:
      - Configurable ml_weight (0.0 = pure rules, 1.0 = pure ML).
      - Confidence-based weight adjustment per frame.
      - Automatic fallback to rules on model failure.

    Args:
        rule_engine: The existing rule-based LightingEngine.
        ml_engine: The ML inference engine.
        ml_weight: Initial blend weight (default 0.3 — conservative).
    """

    def __init__(
        self,
        rule_engine: LightingEngine,
        ml_engine: LightingInferenceEngine,
        ml_weight: float = 0.3,
    ) -> None:
        self._rule_engine = rule_engine
        self._ml_engine = ml_engine
        self._ml_weight = max(0.0, min(1.0, ml_weight))

        # Fallback state tracking.
        self._invalid_output_start: float | None = None
        self._fallback_until: float = 0.0
        self._consecutive_zero_frames: int = 0

    @classmethod
    def create(
        cls,
        rule_engine: LightingEngine,
        checkpoint_path: Path | str,
        ml_weight: float = 0.3,
    ) -> HybridLightingEngine:
        """Convenience factory that loads the ML model from a checkpoint.

        Args:
            rule_engine: The existing rule-based LightingEngine.
            checkpoint_path: Path to a trained model checkpoint.
            ml_weight: Initial ML blend weight.

        Returns:
            Initialized HybridLightingEngine.
        """
        ml_engine = LightingInferenceEngine.from_checkpoint(checkpoint_path)
        return cls(rule_engine=rule_engine, ml_engine=ml_engine, ml_weight=ml_weight)

    @property
    def ml_weight(self) -> float:
        """Current ML blend weight."""
        return self._ml_weight

    @property
    def fixture_map(self):
        """Delegate to rule engine's fixture map."""
        return self._rule_engine.fixture_map

    def set_ml_weight(self, weight: float) -> None:
        """Adjust ML influence.

        Args:
            weight: New blend weight. 0.0 = pure rules, 1.0 = pure ML.
        """
        self._ml_weight = max(0.0, min(1.0, weight))
        logger.info("ML weight set to %.2f", self._ml_weight)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate fixture commands by blending ML and rule-based output.

        Args:
            state: Current MusicState from the audio pipeline.

        Returns:
            List of FixtureCommand, one per fixture in the map.
        """
        # Always generate rule-based commands (they're our safety net).
        rule_commands = self._rule_engine.generate(state)

        # If ML weight is zero, skip ML entirely.
        if self._ml_weight < 0.001:
            return rule_commands

        # Check if we're in fallback mode.
        now = time.monotonic()
        if now < self._fallback_until:
            return rule_commands

        # Get ML prediction.
        try:
            intent = self._ml_engine.predict(state)
        except Exception:
            logger.exception("ML model prediction failed, using rules only")
            self._trigger_fallback(now)
            return rule_commands

        # Validate ML output.
        if not self._validate_intent(intent, now):
            return rule_commands

        # Convert intent to fixture commands.
        ml_commands = intent_to_commands(intent, self._rule_engine.fixture_map)

        # Compute effective weight with confidence adjustment.
        effective_weight = self._compute_effective_weight(intent)

        # Blend per-fixture.
        blended: list[FixtureCommand] = []
        for rule_cmd, ml_cmd in zip(rule_commands, ml_commands, strict=True):
            blended.append(_blend_commands(rule_cmd, ml_cmd, effective_weight))

        return blended

    def _validate_intent(self, intent, now: float) -> bool:
        """Check if the ML output is valid, tracking invalid duration.

        Triggers fallback if invalid output persists for too long.

        Args:
            intent: LightingIntent from the ML model.
            now: Current monotonic time.

        Returns:
            True if the intent is valid, False if we should use rules only.
        """
        import math

        # Check for NaN in any field.
        values = [
            intent.overall_brightness,
            intent.color_diversity,
            intent.spatial_symmetry,
            intent.strobe_intensity,
            *intent.dominant_color,
            *intent.secondary_color,
            *intent.spatial_distribution,
        ]
        has_nan = any(math.isnan(v) for v in values)

        # Check for all-zero output.
        is_all_zero = (
            intent.overall_brightness < 0.001
            and all(v < 0.001 for v in intent.spatial_distribution)
            and not intent.blackout  # Blackout is intentional zero.
        )

        if has_nan or is_all_zero:
            if self._invalid_output_start is None:
                self._invalid_output_start = now

            invalid_duration = now - self._invalid_output_start
            if invalid_duration > _INVALID_OUTPUT_THRESHOLD_S:
                logger.warning(
                    "ML model produced invalid output for %.1fs, triggering fallback",
                    invalid_duration,
                )
                self._trigger_fallback(now)
                return False

            # Invalid but not long enough to trigger fallback — still skip this frame.
            return False
        else:
            # Valid output — reset invalid tracking.
            self._invalid_output_start = None
            return True

    def _trigger_fallback(self, now: float) -> None:
        """Enter fallback mode (pure rules) for _FALLBACK_DURATION_S.

        Args:
            now: Current monotonic time.
        """
        self._fallback_until = now + _FALLBACK_DURATION_S
        self._invalid_output_start = None
        logger.info("Fallback mode active for %.0fs", _FALLBACK_DURATION_S)

    def _compute_effective_weight(self, intent) -> float:
        """Adjust ml_weight based on model output confidence.

        Reduces weight when the model output looks uncertain:
        - Very low brightness with no blackout intent
        - Extreme spatial asymmetry without clear musical cause

        Args:
            intent: LightingIntent from the ML model.

        Returns:
            Effective ML weight for this frame (0.0 to ml_weight).
        """
        confidence = 1.0

        # Low brightness without blackout reduces confidence.
        if intent.overall_brightness < 0.05 and not intent.blackout:
            confidence *= 0.3

        # Very high strobe intensity without clear trigger.
        if intent.strobe_intensity > 0.9:
            confidence *= 0.8

        return self._ml_weight * confidence
