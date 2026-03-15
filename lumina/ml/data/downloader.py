"""yt-dlp wrapper for downloading concert videos for ML training data.

Downloads concert footage organized by genre profile, extracts audio as WAV,
and saves metadata JSON. Videos are stored in the structure:
    data/videos/raw/{genre}/{video_id}/video.mp4, audio.wav, info.json

This module is part of the ML data collection pipeline (Phase A) described
in DOCS.md Section 3.2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumina.ml.data.catalog import CameraType, CatalogEntry, CatalogManager, LightingVisibility, VenueType

logger = logging.getLogger(__name__)

# Default search queries per genre profile, from DOCS.md Section 3.2.
GENRE_SEARCH_QUERIES: dict[str, list[str]] = {
    "rage_trap": [
        "Playboi Carti concert full",
        "Travis Scott live show full set",
    ],
    "psych_rnb": [
        "The Weeknd concert full",
        "Don Toliver live",
    ],
    "french_melodic": [
        "Ninho concert complet",
        "Jul concert live",
    ],
    "french_hard": [
        "Kaaris concert live",
    ],
    "euro_alt": [
        "AyVe live concert",
        "Exetra Archive live",
    ],
    "theatrical": [
        "Stromae concert full",
        "Stromae live show",
    ],
    "festival_edm": [
        "Tomorrowland full set",
        "David Guetta live",
        "Armin van Buuren live",
    ],
    "uk_bass": [
        "Fred again live",
        "Boiler Room sets",
    ],
}

def _find_ffmpeg() -> str | None:
    """Locate an FFmpeg binary, preferring imageio-ffmpeg's bundled copy.

    yt-dlp requires the binary to be named 'ffmpeg' or 'ffmpeg.exe'. If the
    imageio-ffmpeg binary has a version-stamped name, a copy named 'ffmpeg.exe'
    is created alongside it (one-time operation).

    Returns:
        Path to the directory containing a 'ffmpeg.exe', or None if not found.
    """
    try:
        import shutil

        import imageio_ffmpeg

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        ffmpeg_std = exe.parent / "ffmpeg.exe"
        if not ffmpeg_std.exists():
            shutil.copy2(exe, ffmpeg_std)
        return str(exe.parent)
    except Exception:
        pass
    return None


# yt-dlp download options from DOCS.md Section 3.2.
_ffmpeg_dir = _find_ffmpeg()
DEFAULT_YTDLP_OPTS: dict[str, Any] = {
    "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "merge_output_format": "mp4",
    "writeinfojson": True,
    "writethumbnail": True,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        },
    ],
    **({"ffmpeg_location": _ffmpeg_dir} if _ffmpeg_dir else {}),
}


@dataclass
class DownloadResult:
    """Result of a single video download attempt.

    Args:
        video_id: The platform video identifier (e.g. YouTube ID).
        success: Whether the download completed successfully.
        video_path: Path to the downloaded MP4 file, if successful.
        audio_path: Path to the extracted WAV file, if successful.
        info_path: Path to the yt-dlp metadata JSON, if successful.
        error: Error message if download failed.
    """

    video_id: str
    success: bool
    video_path: Path | None = None
    audio_path: Path | None = None
    info_path: Path | None = None
    error: str | None = None


@dataclass
class GenreDownloadPlan:
    """Plan for downloading videos for a specific genre profile.

    Args:
        genre_profile: The genre profile name (e.g. "rage_trap").
        queries: Search queries to use for finding videos.
        max_results_per_query: Maximum number of results to download per query.
        max_duration_s: Skip videos longer than this (seconds).
        min_duration_s: Skip videos shorter than this (seconds).
    """

    genre_profile: str
    queries: list[str] = field(default_factory=list)
    max_results_per_query: int = 10
    max_duration_s: int = 7200
    min_duration_s: int = 120


class VideoDownloader:
    """Downloads concert videos using yt-dlp and organizes by genre profile.

    Videos are saved to `data/videos/raw/{genre}/{video_id}/` with:
    - video.mp4: The concert footage at 720p max
    - audio.wav: Extracted lossless audio for analysis
    - info.json: yt-dlp metadata (title, duration, uploader, etc.)

    Args:
        data_root: Root data directory (default: project_root/data).
        ytdlp_opts: Override default yt-dlp options.
    """

    def __init__(
        self,
        data_root: Path | None = None,
        ytdlp_opts: dict[str, Any] | None = None,
    ) -> None:
        if data_root is None:
            self._data_root = Path(__file__).resolve().parents[3] / "data"
        else:
            self._data_root = Path(data_root)

        self._raw_dir = self._data_root / "videos" / "raw"
        self._ytdlp_opts = ytdlp_opts or dict(DEFAULT_YTDLP_OPTS)
        self._catalog = CatalogManager(self._data_root)

    @property
    def data_root(self) -> Path:
        """Return the root data directory."""
        return self._data_root

    @property
    def raw_dir(self) -> Path:
        """Return the raw video storage directory."""
        return self._raw_dir

    def download_video(
        self,
        url: str,
        genre_profile: str,
    ) -> DownloadResult:
        """Download a single video by URL.

        Downloads the video at 720p max, extracts audio as WAV, and saves
        yt-dlp metadata JSON. All files are stored in:
            data/videos/raw/{genre_profile}/{video_id}/

        Args:
            url: Video URL (YouTube, Vimeo, etc.).
            genre_profile: Genre profile to file under (e.g. "rage_trap").

        Returns:
            DownloadResult with paths to downloaded files or error info.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error(
                "yt-dlp is not installed. Install with: "
                "pip install 'lumina[ml_training]' or pip install yt-dlp"
            )
            return DownloadResult(
                video_id="unknown",
                success=False,
                error="yt-dlp not installed",
            )

        # Extract video ID before downloading to set up output directory.
        video_id = self._extract_video_id(url)
        if video_id is None:
            # Use yt-dlp to extract info first.
            video_id = self._extract_id_via_ytdlp(url)
            if video_id is None:
                return DownloadResult(
                    video_id="unknown",
                    success=False,
                    error=f"Could not extract video ID from URL: {url}",
                )

        output_dir = self._raw_dir / genre_profile / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / "video.mp4"
        audio_path = output_dir / "audio.wav"
        info_path = output_dir / "info.json"

        # Configure yt-dlp output template.
        opts = dict(self._ytdlp_opts)
        opts["outtmpl"] = str(output_dir / "video.%(ext)s")

        # Keep the video file after audio extraction.
        opts["keepvideo"] = True

        # Set up postprocessors: extract audio to WAV alongside the video.
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            },
        ]

        logger.info(
            "Downloading video %s for genre '%s' to %s",
            video_id,
            genre_profile,
            output_dir,
        )

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

            if info is None:
                return DownloadResult(
                    video_id=video_id,
                    success=False,
                    error="yt-dlp returned no info",
                )

            # Save sanitized metadata.
            metadata = self._sanitize_info(info)
            info_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Verify expected files exist. yt-dlp may name files differently.
            actual_video = self._find_file(output_dir, ".mp4")
            actual_audio = self._find_file(output_dir, ".wav")

            if actual_video and actual_video != video_path:
                actual_video.rename(video_path)
            if actual_audio and actual_audio != audio_path:
                actual_audio.rename(audio_path)

            final_video = video_path if video_path.exists() else None
            final_audio = audio_path if audio_path.exists() else None

            logger.info(
                "Download complete: video=%s, audio=%s",
                final_video is not None,
                final_audio is not None,
            )

            # Auto-catalog the downloaded video.
            if info_path.exists():
                self._catalog.add_from_info_json(
                    info_path,
                    genre_profile=genre_profile,
                    camera_type=CameraType.UNKNOWN.value,
                    venue_type=VenueType.UNKNOWN.value,
                    lighting_visibility=LightingVisibility.MEDIUM.value,
                )
                self._catalog.save()
                logger.info("Cataloged %s under '%s'", video_id, genre_profile)

            return DownloadResult(
                video_id=video_id,
                success=True,
                video_path=final_video,
                audio_path=final_audio,
                info_path=info_path if info_path.exists() else None,
            )

        except Exception as e:
            logger.exception("Failed to download %s: %s", url, e)
            return DownloadResult(
                video_id=video_id,
                success=False,
                error=str(e),
            )

    def search_and_download(
        self,
        plan: GenreDownloadPlan,
    ) -> list[DownloadResult]:
        """Search for videos matching a genre plan and download them.

        Uses yt-dlp's search functionality to find concert videos matching
        the plan's queries, then downloads each result.

        Args:
            plan: Download plan specifying genre, queries, and limits.

        Returns:
            List of DownloadResult for each attempted download.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp is not installed.")
            return []

        queries = plan.queries or GENRE_SEARCH_QUERIES.get(plan.genre_profile, [])
        if not queries:
            logger.warning(
                "No search queries for genre profile '%s'", plan.genre_profile
            )
            return []

        results: list[DownloadResult] = []

        for query in queries:
            search_query = f"ytsearch{plan.max_results_per_query}:{query}"
            logger.info(
                "Searching: '%s' (max %d results)",
                query,
                plan.max_results_per_query,
            )

            try:
                extract_opts: dict[str, Any] = {
                    "quiet": True,
                    "extract_flat": True,
                    "skip_download": True,
                }
                with yt_dlp.YoutubeDL(extract_opts) as ydl:
                    search_results = ydl.extract_info(search_query, download=False)

                if search_results is None or "entries" not in search_results:
                    logger.warning("No search results for query: '%s'", query)
                    continue

                entries = list(search_results["entries"])
                logger.info("Found %d results for '%s'", len(entries), query)

                for entry in entries:
                    if entry is None:
                        continue

                    duration = entry.get("duration")
                    if duration is not None:
                        if duration < plan.min_duration_s:
                            logger.debug(
                                "Skipping %s: too short (%ds < %ds)",
                                entry.get("id", "?"),
                                duration,
                                plan.min_duration_s,
                            )
                            continue
                        if duration > plan.max_duration_s:
                            logger.debug(
                                "Skipping %s: too long (%ds > %ds)",
                                entry.get("id", "?"),
                                duration,
                                plan.max_duration_s,
                            )
                            continue

                    video_url = entry.get("url") or entry.get("webpage_url")
                    if video_url is None:
                        video_id = entry.get("id")
                        if video_id:
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                        else:
                            continue

                    # Check if already downloaded.
                    vid = entry.get("id", "")
                    existing_dir = self._raw_dir / plan.genre_profile / vid
                    if existing_dir.exists() and (existing_dir / "video.mp4").exists():
                        logger.info("Already downloaded: %s, skipping", vid)
                        results.append(
                            DownloadResult(
                                video_id=vid,
                                success=True,
                                video_path=existing_dir / "video.mp4",
                                audio_path=existing_dir / "audio.wav",
                                info_path=existing_dir / "info.json",
                            )
                        )
                        continue

                    result = self.download_video(video_url, plan.genre_profile)
                    results.append(result)

            except Exception as e:
                logger.exception("Search failed for query '%s': %s", query, e)

        logger.info(
            "Genre '%s': %d/%d downloads successful",
            plan.genre_profile,
            sum(1 for r in results if r.success),
            len(results),
        )
        return results

    def download_all_genres(
        self,
        max_results_per_query: int = 10,
        max_duration_s: int = 7200,
        min_duration_s: int = 120,
        genres: list[str] | None = None,
    ) -> dict[str, list[DownloadResult]]:
        """Download videos for all (or selected) genre profiles.

        Creates a GenreDownloadPlan for each profile using the default
        search queries from GENRE_SEARCH_QUERIES.

        Args:
            max_results_per_query: Max results per search query.
            max_duration_s: Skip videos longer than this (seconds).
            min_duration_s: Skip videos shorter than this (seconds).
            genres: Specific genre profiles to download. None = all genres.

        Returns:
            Dict mapping genre profile name to list of DownloadResult.
        """
        target_genres = genres or list(GENRE_SEARCH_QUERIES.keys())
        all_results: dict[str, list[DownloadResult]] = {}

        for genre in target_genres:
            if genre not in GENRE_SEARCH_QUERIES:
                logger.warning("Unknown genre profile: '%s', skipping", genre)
                continue

            plan = GenreDownloadPlan(
                genre_profile=genre,
                queries=GENRE_SEARCH_QUERIES[genre],
                max_results_per_query=max_results_per_query,
                max_duration_s=max_duration_s,
                min_duration_s=min_duration_s,
            )
            all_results[genre] = self.search_and_download(plan)

        total = sum(len(v) for v in all_results.values())
        successful = sum(
            sum(1 for r in v if r.success) for v in all_results.values()
        )
        logger.info("All genres complete: %d/%d successful", successful, total)

        return all_results

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """Extract video ID from common URL formats.

        Args:
            url: Video URL.

        Returns:
            Video ID string, or None if not parseable.
        """
        import re

        # YouTube patterns.
        patterns = [
            r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"(?:embed/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_id_via_ytdlp(url: str) -> str | None:
        """Use yt-dlp to extract the video ID without downloading.

        Args:
            url: Video URL.

        Returns:
            Video ID string, or None on failure.
        """
        try:
            import yt_dlp

            opts: dict[str, Any] = {"quiet": True, "skip_download": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and "id" in info:
                    return str(info["id"])
        except Exception as e:
            logger.debug("Could not extract ID via yt-dlp: %s", e)
        return None

    @staticmethod
    def _sanitize_info(info: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable fields from yt-dlp info dict.

        Args:
            info: Raw yt-dlp info dictionary.

        Returns:
            Sanitized dictionary safe for JSON serialization.
        """
        safe_keys = {
            "id",
            "title",
            "description",
            "upload_date",
            "uploader",
            "uploader_id",
            "channel",
            "channel_id",
            "duration",
            "view_count",
            "like_count",
            "categories",
            "tags",
            "thumbnail",
            "webpage_url",
            "original_url",
            "fps",
            "width",
            "height",
            "resolution",
            "filesize_approx",
        }
        return {k: v for k, v in info.items() if k in safe_keys}

    @staticmethod
    def _find_file(directory: Path, suffix: str) -> Path | None:
        """Find first file with given suffix in a directory.

        Args:
            directory: Directory to search.
            suffix: File extension including dot (e.g. ".mp4").

        Returns:
            Path to matching file, or None if not found.
        """
        for path in directory.iterdir():
            if path.suffix.lower() == suffix.lower() and path.is_file():
                return path
        return None


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download concert videos for ML training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download <url> --genre <profile>
    dl_parser = subparsers.add_parser("download", help="Download a single video by URL.")
    dl_parser.add_argument("url", help="Video URL (YouTube, etc.)")
    dl_parser.add_argument("--genre", required=True, choices=list(GENRE_SEARCH_QUERIES.keys()), help="Genre profile to file under.")

    # list — show catalog summary
    subparsers.add_parser("list", help="Show catalog summary (genres, video counts, total hours).")

    # search --genre <profile> [--genre ...] --max-results N
    search_parser = subparsers.add_parser("search", help="Search and download videos for one or more genres.")
    search_parser.add_argument("--genre", dest="genres", action="append", choices=list(GENRE_SEARCH_QUERIES.keys()), help="Genre(s) to download. Repeat for multiple. Default: all.")
    search_parser.add_argument("--max-results", type=int, default=5, help="Max results per search query (default: 5).")
    search_parser.add_argument("--min-duration", type=int, default=120, help="Minimum video duration in seconds (default: 120).")
    search_parser.add_argument("--max-duration", type=int, default=7200, help="Maximum video duration in seconds (default: 7200).")

    args = parser.parse_args()
    downloader = VideoDownloader()

    if args.command == "list":
        cat = CatalogManager()
        summary = cat.genre_summary()
        total_h = cat.total_duration_hours()
        if not summary:
            print("Catalog is empty. Run 'search' to download videos.")
        else:
            print(f"{'Genre':<20} {'Videos':>7}")
            print("-" * 30)
            for genre, count in sorted(summary.items()):
                print(f"  {genre:<18} {count:>7}")
            print("-" * 30)
            print(f"  {'TOTAL':<18} {sum(summary.values()):>7}  ({total_h:.1f}h)")

    if args.command == "download":
        result = downloader.download_video(args.url, args.genre)
        if result.success:
            print(f"Downloaded: {result.video_path}")
        else:
            print(f"Failed: {result.error}", file=sys.stderr)
            sys.exit(1)

    if args.command == "search":
        results = downloader.download_all_genres(
            max_results_per_query=args.max_results,
            min_duration_s=args.min_duration,
            max_duration_s=args.max_duration,
            genres=args.genres,
        )
        total = sum(len(v) for v in results.values())
        successful = sum(sum(1 for r in v if r.success) for v in results.values())
        print(f"\nDone: {successful}/{total} videos downloaded successfully.")
