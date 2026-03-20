"""Video catalog management for the ML training data pipeline.

Manages a catalog.json file tracking all downloaded concert videos with
metadata, quality scores, and filtering capabilities. The catalog schema
follows DOCS.md Section 3.2.

Catalog entries track:
- Video identity (video_id, artist, title)
- Genre classification (genre_profile)
- Quality metadata (quality_score, camera_type, venue_type)
- Lighting-specific tags (has_led_screens, lighting_visibility)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraType(StrEnum):
    """Camera setup type for quality assessment."""

    FIXED_WIDE = "fixed_wide"
    MULTI_CAM_PRO = "multi_cam_pro"
    FESTIVAL_STREAM = "festival_stream"
    SINGLE_CAM_BOOTLEG = "single_cam_bootleg"
    UNKNOWN = "unknown"


class VenueType(StrEnum):
    """Venue type classification."""

    FESTIVAL_OUTDOOR = "festival_outdoor"
    FESTIVAL_INDOOR = "festival_indoor"
    ARENA = "arena"
    CLUB = "club"
    WAREHOUSE = "warehouse"
    STUDIO = "studio"
    UNKNOWN = "unknown"


class LightingVisibility(StrEnum):
    """How well stage lighting is visible in the footage."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class CatalogEntry:
    """A single video entry in the catalog.

    Follows the metadata schema from DOCS.md Section 3.2.

    Args:
        video_id: Platform video identifier (e.g. YouTube ID).
        genre_profile: Genre profile name (e.g. "rage_trap", "festival_edm").
        artist: Primary artist name.
        title: Video title.
        duration_s: Video duration in seconds.
        quality_score: Overall quality score 0.0-1.0 for training suitability.
        camera_type: Camera setup classification.
        venue_type: Venue type classification.
        has_led_screens: Whether large LED screens are visible on stage.
        lighting_visibility: How well stage lighting is visible.
        notes: Free-form notes about the video.
    """

    video_id: str
    genre_profile: str
    artist: str = ""
    title: str = ""
    duration_s: float = 0.0
    quality_score: float = 0.0
    camera_type: str = CameraType.UNKNOWN.value
    venue_type: str = VenueType.UNKNOWN.value
    has_led_screens: bool = False
    lighting_visibility: str = LightingVisibility.MEDIUM.value
    notes: str = ""

    def compute_quality_score(self) -> float:
        """Compute a quality score based on video attributes.

        Scoring factors (from DOCS.md Section 3.2 video selection criteria):
        - Camera type: fixed wide (best) > multi-cam pro > festival stream
        - Lighting visibility: high > medium > low
        - Duration: 5-60 min is ideal; very short or very long penalized
        - LED screens: penalizes (contaminates color readings)

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0

        # Camera type scoring (0-0.35).
        camera_scores: dict[str, float] = {
            CameraType.FIXED_WIDE.value: 0.35,
            CameraType.SINGLE_CAM_BOOTLEG.value: 0.30,
            CameraType.MULTI_CAM_PRO.value: 0.25,
            CameraType.FESTIVAL_STREAM.value: 0.20,
            CameraType.UNKNOWN.value: 0.15,
        }
        score += camera_scores.get(self.camera_type, 0.15)

        # Lighting visibility scoring (0-0.35).
        visibility_scores: dict[str, float] = {
            LightingVisibility.HIGH.value: 0.35,
            LightingVisibility.MEDIUM.value: 0.20,
            LightingVisibility.LOW.value: 0.10,
            LightingVisibility.NONE.value: 0.0,
        }
        score += visibility_scores.get(self.lighting_visibility, 0.10)

        # Duration scoring (0-0.20): ideal is 5-60 minutes.
        if self.duration_s > 0:
            minutes = self.duration_s / 60.0
            if 5.0 <= minutes <= 60.0:
                score += 0.20
            elif 2.0 <= minutes < 5.0 or 60.0 < minutes <= 120.0:
                score += 0.10
            else:
                score += 0.05

        # LED screen penalty (0-0.10 bonus for no screens).
        if not self.has_led_screens:
            score += 0.10

        self.quality_score = round(min(1.0, max(0.0, score)), 2)
        return self.quality_score


@dataclass
class Catalog:
    """In-memory representation of the full video catalog.

    Args:
        entries: List of catalog entries.
        version: Catalog schema version.
    """

    entries: list[CatalogEntry] = field(default_factory=list)
    version: str = "1.0"


class CatalogManager:
    """Manages the video catalog JSON file on disk.

    The catalog is stored at `data/videos/metadata/catalog.json` and tracks
    all downloaded concert videos with their metadata, quality scores, and
    genre assignments.

    Args:
        data_root: Root data directory (default: project_root/data).
    """

    def __init__(self, data_root: Path | None = None) -> None:
        if data_root is None:
            self._data_root = Path(__file__).resolve().parents[3] / "data"
        else:
            self._data_root = Path(data_root)

        self._catalog_path = self._data_root / "videos" / "metadata" / "catalog.json"
        self._catalog = self._load()

    @property
    def catalog_path(self) -> Path:
        """Return the path to the catalog JSON file."""
        return self._catalog_path

    @property
    def entries(self) -> list[CatalogEntry]:
        """Return all catalog entries."""
        return self._catalog.entries

    def _load(self) -> Catalog:
        """Load the catalog from disk, or create an empty one.

        Returns:
            Loaded or empty Catalog instance.
        """
        if not self._catalog_path.exists():
            logger.info("No catalog found at %s, starting empty", self._catalog_path)
            return Catalog()

        try:
            raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
            entries = [CatalogEntry(**entry) for entry in raw.get("entries", [])]
            version = raw.get("version", "1.0")
            logger.info("Loaded catalog with %d entries", len(entries))
            return Catalog(entries=entries, version=version)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to parse catalog at %s: %s", self._catalog_path, e)
            return Catalog()

    def save(self) -> None:
        """Save the catalog to disk as JSON."""
        self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self._catalog.version,
            "entries": [asdict(entry) for entry in self._catalog.entries],
        }
        self._catalog_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved catalog with %d entries to %s",
            len(self._catalog.entries),
            self._catalog_path,
        )

    def add_entry(self, entry: CatalogEntry, auto_score: bool = True) -> None:
        """Add a new entry to the catalog.

        If an entry with the same video_id already exists, it is updated.

        Args:
            entry: The catalog entry to add.
            auto_score: If True, compute quality_score from attributes.
        """
        if auto_score:
            entry.compute_quality_score()

        # Update existing or append new.
        for i, existing in enumerate(self._catalog.entries):
            if existing.video_id == entry.video_id:
                self._catalog.entries[i] = entry
                logger.info("Updated catalog entry: %s", entry.video_id)
                return

        self._catalog.entries.append(entry)
        logger.info("Added catalog entry: %s (%s)", entry.video_id, entry.genre_profile)

    def remove_entry(self, video_id: str) -> bool:
        """Remove an entry by video_id.

        Args:
            video_id: The video ID to remove.

        Returns:
            True if an entry was removed, False if not found.
        """
        original_len = len(self._catalog.entries)
        self._catalog.entries = [
            e for e in self._catalog.entries if e.video_id != video_id
        ]
        removed = len(self._catalog.entries) < original_len
        if removed:
            logger.info("Removed catalog entry: %s", video_id)
        return removed

    def get_entry(self, video_id: str) -> CatalogEntry | None:
        """Look up a single entry by video_id.

        Args:
            video_id: The video ID to find.

        Returns:
            The CatalogEntry if found, else None.
        """
        for entry in self._catalog.entries:
            if entry.video_id == video_id:
                return entry
        return None

    def query_by_genre(self, genre_profile: str) -> list[CatalogEntry]:
        """Return all entries matching a genre profile.

        Args:
            genre_profile: Genre profile name (e.g. "rage_trap").

        Returns:
            List of matching CatalogEntry objects.
        """
        return [
            e for e in self._catalog.entries if e.genre_profile == genre_profile
        ]

    def query_by_min_quality(self, min_score: float) -> list[CatalogEntry]:
        """Return entries with quality_score >= min_score.

        Args:
            min_score: Minimum quality score threshold (0.0-1.0).

        Returns:
            List of entries meeting the quality threshold.
        """
        return [
            e for e in self._catalog.entries if e.quality_score >= min_score
        ]

    def query_by_genre_and_quality(
        self,
        genre_profile: str,
        min_score: float = 0.0,
    ) -> list[CatalogEntry]:
        """Return entries matching genre and minimum quality.

        Args:
            genre_profile: Genre profile name.
            min_score: Minimum quality score threshold.

        Returns:
            List of matching entries sorted by quality_score descending.
        """
        matches = [
            e
            for e in self._catalog.entries
            if e.genre_profile == genre_profile and e.quality_score >= min_score
        ]
        return sorted(matches, key=lambda e: e.quality_score, reverse=True)

    def genre_summary(self) -> dict[str, int]:
        """Return count of entries per genre profile.

        Returns:
            Dict mapping genre_profile name to entry count.
        """
        counts: dict[str, int] = {}
        for entry in self._catalog.entries:
            counts[entry.genre_profile] = counts.get(entry.genre_profile, 0) + 1
        return counts

    def total_duration_hours(self) -> float:
        """Return total duration of all cataloged videos in hours.

        Returns:
            Total duration in hours.
        """
        total_s = sum(e.duration_s for e in self._catalog.entries)
        return total_s / 3600.0

    def add_from_info_json(
        self,
        info_path: Path,
        genre_profile: str,
        camera_type: str = CameraType.UNKNOWN.value,
        venue_type: str = VenueType.UNKNOWN.value,
        has_led_screens: bool = False,
        lighting_visibility: str = LightingVisibility.MEDIUM.value,
        notes: str = "",
    ) -> CatalogEntry | None:
        """Create a catalog entry from a yt-dlp info.json file.

        Extracts video_id, title, artist (uploader/channel), and duration
        from the yt-dlp metadata, then adds to the catalog.

        Args:
            info_path: Path to the yt-dlp info.json file.
            genre_profile: Genre profile to assign.
            camera_type: Camera type classification.
            venue_type: Venue type classification.
            has_led_screens: Whether LED screens are visible.
            lighting_visibility: Lighting visibility level.
            notes: Free-form notes.

        Returns:
            The created CatalogEntry, or None on failure.
        """
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read info.json at %s: %s", info_path, e)
            return None

        video_id = info.get("id", info_path.parent.name)
        artist = info.get("channel") or info.get("uploader") or ""
        title = info.get("title", "")
        duration = info.get("duration", 0)

        entry = CatalogEntry(
            video_id=str(video_id),
            genre_profile=genre_profile,
            artist=str(artist),
            title=str(title),
            duration_s=float(duration) if duration else 0.0,
            camera_type=camera_type,
            venue_type=venue_type,
            has_led_screens=has_led_screens,
            lighting_visibility=lighting_visibility,
            notes=notes,
        )
        self.add_entry(entry, auto_score=True)
        return entry
