"""Audio-visual alignment for creating ML training pairs.

Merges MusicState features (from audio analysis) with VideoLightingFrame
features (from video frame extraction) into aligned TrainingPair records.
Aligned data is saved as Parquet files to `data/features/aligned/{video_id}.parquet`.

This module handles timestamp alignment between 10fps video frames and
audio analysis output, as specified in DOCS.md Section 3.4.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from lumina.ml.video.lighting_extractor import VideoLightingFrame

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingPair:
    """A single aligned audio-visual training pair.

    Merges a MusicState (audio features) with a VideoLightingFrame
    (visual lighting features) at the same timestamp. Used for training
    the ML model that maps audio -> lighting intent.

    Follows the schema from DOCS.md Section 3.4.

    Args:
        timestamp: Aligned timestamp in seconds.
        video_id: Source video identifier.
        genre_label: Ground truth genre profile name.
        confidence: Scene classification confidence (0-1).

        Audio features (from MusicState):
        bpm: Current tempo.
        beat_phase: 0-1 position within current beat.
        bar_phase: 0-1 position within current bar.
        is_beat: True on exact beat frames.
        is_downbeat: True on bar downbeats.
        energy: 0-1 overall energy level.
        energy_derivative: Rising (+) or falling (-) energy.
        segment: Song section identifier.
        vocal_energy: 0-1 vocal presence level.
        spectral_centroid: Brightness indicator (Hz).
        sub_bass_energy: Sub-bass (20-80Hz) energy level.
        onset_type: Detected transient type, or None.
        drop_probability: 0-1 probability of drop in next 1-4 bars.

        Lighting features (from VideoLightingFrame):
        overall_brightness: 0-1 mean luminance.
        brightness_variance: Spatial brightness variance.
        dominant_hue: 0-360 degrees.
        dominant_saturation: 0-1.
        secondary_hue: 0-360 degrees.
        color_temperature: Warm vs cool.
        color_diversity: 0-1 distinct colors.
        left_brightness: Left third brightness.
        center_brightness: Center third brightness.
        right_brightness: Right third brightness.
        top_brightness: Upper half brightness.
        bottom_brightness: Lower half brightness.
        spatial_symmetry: 0-1 L/R symmetry.
        brightness_delta: Frame-to-frame change.
        is_strobe: Strobe detected.
        is_blackout: Blackout detected.
        color_change_rate: Hue shift speed.
        scene_confidence: Stage view confidence.
    """

    # Identity.
    timestamp: float = 0.0
    video_id: str = ""
    genre_label: str = ""
    confidence: float = 0.0

    # Audio features (from MusicState).
    bpm: float = 120.0
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    is_beat: bool = False
    is_downbeat: bool = False
    energy: float = 0.0
    energy_derivative: float = 0.0
    segment: str = "verse"
    vocal_energy: float = 0.0
    spectral_centroid: float = 0.0
    sub_bass_energy: float = 0.0
    onset_type: str | None = None
    drop_probability: float = 0.0

    # Lighting features (from VideoLightingFrame).
    overall_brightness: float = 0.0
    brightness_variance: float = 0.0
    dominant_hue: float = 0.0
    dominant_saturation: float = 0.0
    secondary_hue: float = 0.0
    color_temperature: float = 0.0
    color_diversity: float = 0.0
    left_brightness: float = 0.0
    center_brightness: float = 0.0
    right_brightness: float = 0.0
    top_brightness: float = 0.0
    bottom_brightness: float = 0.0
    spatial_symmetry: float = 0.0
    brightness_delta: float = 0.0
    is_strobe: bool = False
    is_blackout: bool = False
    color_change_rate: float = 0.0
    scene_confidence: float = 0.0


class AudioVisualAligner:
    """Aligns audio MusicState features with video VideoLightingFrame features.

    Creates TrainingPair records by matching timestamps between audio analysis
    output (MusicState at each timestep) and video lighting extraction output
    (VideoLightingFrame at 10fps). Saves aligned data as Parquet files.

    Audio and video are inherently synced from the same source file.
    The main alignment concern is matching frame timestamps: video frames
    are extracted at a constant 10fps, while audio features may be at
    different rates. This aligner uses nearest-neighbor matching with a
    configurable tolerance.

    Args:
        data_root: Root data directory (default: project_root/data).
        max_time_drift_s: Maximum allowed time difference between matched
            audio and video timestamps. Pairs outside this tolerance
            are discarded.
        min_scene_confidence: Minimum scene_confidence for a video frame
            to be included in training pairs.
    """

    def __init__(
        self,
        data_root: Path | None = None,
        max_time_drift_s: float = 0.1,
        min_scene_confidence: float = 0.5,
    ) -> None:
        if data_root is None:
            self._data_root = Path(__file__).resolve().parents[3] / "data"
        else:
            self._data_root = Path(data_root)

        self._aligned_dir = self._data_root / "features" / "aligned"
        self._audio_features_dir = self._data_root / "features" / "audio"
        self._lighting_features_dir = self._data_root / "features" / "lighting"
        self._max_time_drift_s = max_time_drift_s
        self._min_scene_confidence = min_scene_confidence

    @property
    def aligned_dir(self) -> Path:
        """Return the aligned features output directory."""
        return self._aligned_dir

    def align_from_lists(
        self,
        video_id: str,
        genre_label: str,
        music_states: list[Any],
        lighting_frames: list[VideoLightingFrame],
    ) -> list[TrainingPair]:
        """Align MusicState and VideoLightingFrame lists by timestamp.

        For each VideoLightingFrame with sufficient scene_confidence, finds
        the nearest MusicState by timestamp within max_time_drift_s and
        creates a TrainingPair.

        Args:
            video_id: Source video identifier.
            genre_label: Ground truth genre profile name.
            music_states: List of MusicState objects (from audio analysis).
                Must have a 'timestamp' attribute.
            lighting_frames: List of VideoLightingFrame objects (from video
                extraction at 10fps).

        Returns:
            List of aligned TrainingPair objects.
        """
        if not music_states or not lighting_frames:
            logger.warning(
                "Empty input for video %s: %d music states, %d lighting frames",
                video_id,
                len(music_states),
                len(lighting_frames),
            )
            return []

        # Sort both lists by timestamp for efficient alignment.
        music_states_sorted = sorted(music_states, key=lambda s: s.timestamp)
        lighting_sorted = sorted(lighting_frames, key=lambda f: f.timestamp)

        # Build array of audio timestamps for nearest-neighbor search.
        audio_timestamps = [s.timestamp for s in music_states_sorted]

        pairs: list[TrainingPair] = []
        audio_idx = 0

        for frame in lighting_sorted:
            # Skip low-confidence frames (not stage views).
            if frame.scene_confidence < self._min_scene_confidence:
                continue

            # Advance audio index to nearest timestamp.
            audio_idx = self._find_nearest(
                audio_timestamps, frame.timestamp, audio_idx
            )
            if audio_idx is None:
                continue

            drift = abs(audio_timestamps[audio_idx] - frame.timestamp)
            if drift > self._max_time_drift_s:
                continue

            state = music_states_sorted[audio_idx]
            pair = self._create_pair(
                video_id=video_id,
                genre_label=genre_label,
                music_state=state,
                lighting_frame=frame,
            )
            pairs.append(pair)

        logger.info(
            "Aligned %d training pairs for video %s "
            "(from %d audio states, %d video frames)",
            len(pairs),
            video_id,
            len(music_states),
            len(lighting_frames),
        )
        return pairs

    def align_from_parquet(
        self,
        video_id: str,
        genre_label: str,
    ) -> list[TrainingPair]:
        """Align pre-computed audio and lighting features from Parquet files.

        Reads audio features from `data/features/audio/{video_id}.parquet`
        and lighting features from `data/features/lighting/{video_id}.parquet`,
        then aligns by timestamp.

        Args:
            video_id: Source video identifier.
            genre_label: Ground truth genre profile name.

        Returns:
            List of aligned TrainingPair objects.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error(
                "pandas is not installed. Install with: "
                "pip install 'lumina[ml_training]' or pip install pandas"
            )
            return []

        audio_path = self._audio_features_dir / f"{video_id}.parquet"
        lighting_path = self._lighting_features_dir / f"{video_id}.parquet"

        if not audio_path.exists():
            logger.error("Audio features not found: %s", audio_path)
            return []
        if not lighting_path.exists():
            logger.error("Lighting features not found: %s", lighting_path)
            return []

        audio_df = pd.read_parquet(audio_path)
        lighting_df = pd.read_parquet(lighting_path)

        logger.info(
            "Loaded %d audio rows and %d lighting rows for %s",
            len(audio_df),
            len(lighting_df),
            video_id,
        )

        # Filter lighting frames by scene confidence.
        if "scene_confidence" in lighting_df.columns:
            lighting_df = lighting_df[
                lighting_df["scene_confidence"] >= self._min_scene_confidence
            ]

        if audio_df.empty or lighting_df.empty:
            logger.warning("No data after filtering for video %s", video_id)
            return []

        # Perform nearest-neighbor merge on timestamp.
        pairs = self._merge_dataframes(
            audio_df=audio_df,
            lighting_df=lighting_df,
            video_id=video_id,
            genre_label=genre_label,
        )

        logger.info(
            "Aligned %d training pairs from Parquet for video %s",
            len(pairs),
            video_id,
        )
        return pairs

    def save_aligned(
        self,
        video_id: str,
        pairs: list[TrainingPair],
    ) -> Path | None:
        """Save aligned training pairs to a Parquet file.

        Output path: `data/features/aligned/{video_id}.parquet`

        Args:
            video_id: Source video identifier (used as filename).
            pairs: List of TrainingPair objects to save.

        Returns:
            Path to the saved Parquet file, or None on failure.
        """
        if not pairs:
            logger.warning("No training pairs to save for video %s", video_id)
            return None

        try:
            import pandas as pd
        except ImportError:
            logger.error(
                "pandas is not installed. Install with: "
                "pip install 'lumina[ml_training]' or pip install pandas"
            )
            return None

        self._aligned_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._aligned_dir / f"{video_id}.parquet"

        records = [asdict(pair) for pair in pairs]
        df = pd.DataFrame(records)

        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(
            "Saved %d aligned pairs to %s (%.1f MB)",
            len(pairs),
            output_path,
            output_path.stat().st_size / (1024 * 1024),
        )
        return output_path

    def load_aligned(self, video_id: str) -> list[TrainingPair]:
        """Load previously saved aligned training pairs from Parquet.

        Args:
            video_id: Source video identifier.

        Returns:
            List of TrainingPair objects, or empty list on failure.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is not installed.")
            return []

        path = self._aligned_dir / f"{video_id}.parquet"
        if not path.exists():
            logger.warning("No aligned data found at %s", path)
            return []

        df = pd.read_parquet(path)
        pairs: list[TrainingPair] = []
        pair_fields = {f.name for f in fields(TrainingPair)}

        for _, row in df.iterrows():
            kwargs = {k: v for k, v in row.to_dict().items() if k in pair_fields}
            pairs.append(TrainingPair(**kwargs))

        logger.info("Loaded %d aligned pairs from %s", len(pairs), path)
        return pairs

    def align_and_save(
        self,
        video_id: str,
        genre_label: str,
        music_states: list[Any] | None = None,
        lighting_frames: list[VideoLightingFrame] | None = None,
    ) -> Path | None:
        """Align features and save to Parquet in one step.

        If music_states and lighting_frames are provided, aligns from lists.
        Otherwise, reads from pre-computed Parquet files.

        Args:
            video_id: Source video identifier.
            genre_label: Ground truth genre profile name.
            music_states: Optional list of MusicState objects.
            lighting_frames: Optional list of VideoLightingFrame objects.

        Returns:
            Path to saved Parquet file, or None on failure.
        """
        if music_states is not None and lighting_frames is not None:
            pairs = self.align_from_lists(
                video_id=video_id,
                genre_label=genre_label,
                music_states=music_states,
                lighting_frames=lighting_frames,
            )
        else:
            pairs = self.align_from_parquet(
                video_id=video_id,
                genre_label=genre_label,
            )

        return self.save_aligned(video_id, pairs)

    def _create_pair(
        self,
        video_id: str,
        genre_label: str,
        music_state: Any,
        lighting_frame: VideoLightingFrame,
    ) -> TrainingPair:
        """Create a TrainingPair from a MusicState and VideoLightingFrame.

        Args:
            video_id: Source video identifier.
            genre_label: Ground truth genre profile.
            music_state: MusicState object (from lumina.audio.models).
            lighting_frame: VideoLightingFrame with extracted lighting.

        Returns:
            Merged TrainingPair.
        """
        return TrainingPair(
            # Identity.
            timestamp=lighting_frame.timestamp,
            video_id=video_id,
            genre_label=genre_label,
            confidence=lighting_frame.scene_confidence,
            # Audio features from MusicState.
            bpm=music_state.bpm,
            beat_phase=music_state.beat_phase,
            bar_phase=music_state.bar_phase,
            is_beat=music_state.is_beat,
            is_downbeat=music_state.is_downbeat,
            energy=music_state.energy,
            energy_derivative=music_state.energy_derivative,
            segment=music_state.segment,
            vocal_energy=music_state.vocal_energy,
            spectral_centroid=music_state.spectral_centroid,
            sub_bass_energy=music_state.sub_bass_energy,
            onset_type=music_state.onset_type,
            drop_probability=music_state.drop_probability,
            # Lighting features from VideoLightingFrame.
            overall_brightness=lighting_frame.overall_brightness,
            brightness_variance=lighting_frame.brightness_variance,
            dominant_hue=lighting_frame.dominant_hue,
            dominant_saturation=lighting_frame.dominant_saturation,
            secondary_hue=lighting_frame.secondary_hue,
            color_temperature=lighting_frame.color_temperature,
            color_diversity=lighting_frame.color_diversity,
            left_brightness=lighting_frame.left_brightness,
            center_brightness=lighting_frame.center_brightness,
            right_brightness=lighting_frame.right_brightness,
            top_brightness=lighting_frame.top_brightness,
            bottom_brightness=lighting_frame.bottom_brightness,
            spatial_symmetry=lighting_frame.spatial_symmetry,
            brightness_delta=lighting_frame.brightness_delta,
            is_strobe=lighting_frame.is_strobe,
            is_blackout=lighting_frame.is_blackout,
            color_change_rate=lighting_frame.color_change_rate,
            scene_confidence=lighting_frame.scene_confidence,
        )

    @staticmethod
    def _find_nearest(
        sorted_timestamps: list[float],
        target: float,
        start_idx: int,
    ) -> int | None:
        """Find the index of the nearest timestamp using linear scan.

        Assumes sorted_timestamps is sorted ascending. Starts scanning from
        start_idx for efficiency when called sequentially with increasing
        target values.

        Args:
            sorted_timestamps: Sorted list of timestamps.
            target: Target timestamp to match.
            start_idx: Index to start scanning from.

        Returns:
            Index of the nearest timestamp, or None if list is empty.
        """
        if not sorted_timestamps:
            return None

        n = len(sorted_timestamps)
        idx = min(start_idx, n - 1)

        # Scan forward while next timestamp is closer.
        while idx < n - 1 and abs(sorted_timestamps[idx + 1] - target) <= abs(
            sorted_timestamps[idx] - target
        ):
            idx += 1

        return idx

    def _merge_dataframes(
        self,
        audio_df: Any,
        lighting_df: Any,
        video_id: str,
        genre_label: str,
    ) -> list[TrainingPair]:
        """Merge audio and lighting DataFrames by nearest timestamp.

        Uses pandas merge_asof for efficient nearest-neighbor join on
        the timestamp column.

        Args:
            audio_df: DataFrame with audio MusicState features.
            lighting_df: DataFrame with VideoLightingFrame features.
            video_id: Source video identifier.
            genre_label: Ground truth genre profile.

        Returns:
            List of TrainingPair objects from the merged data.
        """
        import pandas as pd

        # Ensure both are sorted by timestamp.
        audio_df = audio_df.sort_values("timestamp").reset_index(drop=True)
        lighting_df = lighting_df.sort_values("timestamp").reset_index(drop=True)

        # Use merge_asof for nearest-neighbor join within tolerance.
        merged = pd.merge_asof(
            lighting_df,
            audio_df,
            on="timestamp",
            tolerance=self._max_time_drift_s,
            direction="nearest",
            suffixes=("_light", "_audio"),
        )

        # Drop rows where no audio match was found (NaN from merge_asof).
        merged = merged.dropna(subset=["bpm"])

        pairs: list[TrainingPair] = []
        pair_field_names = {f.name for f in fields(TrainingPair)}

        for _, row in merged.iterrows():
            kwargs: dict[str, Any] = {
                "video_id": video_id,
                "genre_label": genre_label,
            }

            # Map merged columns to TrainingPair fields.
            # Lighting columns may have _light suffix from merge.
            # Audio columns may have _audio suffix from merge.
            row_dict = row.to_dict()
            for field_name in pair_field_names:
                if field_name in ("video_id", "genre_label"):
                    continue

                # Try exact match first, then suffixed versions.
                if field_name in row_dict and not pd.isna(row_dict[field_name]):
                    kwargs[field_name] = row_dict[field_name]
                elif f"{field_name}_light" in row_dict and not pd.isna(
                    row_dict[f"{field_name}_light"]
                ):
                    kwargs[field_name] = row_dict[f"{field_name}_light"]
                elif f"{field_name}_audio" in row_dict and not pd.isna(
                    row_dict[f"{field_name}_audio"]
                ):
                    kwargs[field_name] = row_dict[f"{field_name}_audio"]

            pairs.append(TrainingPair(**kwargs))

        return pairs
