"""Aim detection pipeline for elimination review.

This module implements an end-to-end pipeline that ingests a gameplay video,
identifies elimination events, extracts clips, computes aim related features, and
produces CSV/HTML reports with suspicion scores.

The implementation focuses on providing a modular, extendable architecture that
can integrate computer vision, audio, and OCR detectors while remaining usable
on systems where those dependencies are unavailable. When specialised models or
pre-computed detections are missing, the pipeline gracefully falls back to
placeholder heuristics so that the full reporting stack continues to operate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional at runtime
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None  # type: ignore


@dataclass
class PipelineConfig:
    """Configuration for the aim detection pipeline."""

    video_path: Path
    output_dir: Path
    gpu_device: Optional[int] = None
    pre_window: float = 10.0
    post_window: float = 2.0
    frame_sample_rate: int = 1
    suspicion_threshold: float = 0.8
    min_reaction_ms: float = 40.0
    chunk_duration: Optional[float] = None
    preprocessed_dir: Optional[Path] = None
    debug: bool = False


@dataclass
class VideoMetadata:
    """Stores key metadata about the video."""

    fps: float
    width: int
    height: int
    duration: float


@dataclass
class DetectionSignal:
    """A raw signal from any elimination detector."""

    timestamp: float
    confidence: float
    detector: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EliminationEvent:
    """Represents a fused elimination event."""

    elim_id: str
    timestamp: float
    confidence: float
    signals: List[DetectionSignal]


@dataclass
class EliminationFeatures:
    """Feature vector computed around an elimination event."""

    elim_id: str
    timestamp: float
    clip_file: Optional[str] = None
    reaction_ms: Optional[float] = None
    dwell_ms: Optional[float] = None
    micro_adjusts_per_s: Optional[float] = None
    snap_magnitude_px: Optional[float] = None
    snap_frames: Optional[int] = None
    headshot: Optional[bool] = None
    shots_count: Optional[int] = None
    headshot_ratio: Optional[float] = None
    aim_variance: Optional[float] = None
    muzzle_flash_count: Optional[int] = None
    suspicion_score: Optional[float] = None
    flags: List[str] = field(default_factory=list)


class FFprobe:
    """Utility wrapper around ffprobe for querying video metadata."""

    @staticmethod
    def probe(video_path: Path) -> VideoMetadata:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        logging.debug("Running ffprobe: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - tool failure
            raise RuntimeError("ffprobe is required to read video metadata") from exc

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        avg_frame_rate = stream.get("avg_frame_rate", "0/1")
        if avg_frame_rate and "/" in avg_frame_rate:
            num, den = avg_frame_rate.split("/")
            fps = float(num) / max(float(den), 1.0)
        else:
            fps = float(avg_frame_rate)

        duration = float(fmt.get("duration", 0.0))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        metadata = VideoMetadata(fps=fps, width=width, height=height, duration=duration)
        logging.debug("Parsed video metadata: %s", metadata)
        return metadata


class BaseDetector:
    """Interface for elimination detection modules."""

    name: str = "base"

    def detect(
        self, config: PipelineConfig, metadata: VideoMetadata
    ) -> List[DetectionSignal]:  # pragma: no cover - base class method
        raise NotImplementedError


class PreprocessedDetector(BaseDetector):
    """Loads elimination candidates from preprocessed JSON/CSV files."""

    name = "preprocessed"

    def detect(self, config: PipelineConfig, metadata: VideoMetadata) -> List[DetectionSignal]:
        if not config.preprocessed_dir:
            return []

        directory = config.preprocessed_dir
        signals: List[DetectionSignal] = []

        json_path = directory / "eliminations.json"
        csv_path = directory / "eliminations.csv"

        if json_path.exists():
            logging.info("Loading elimination timestamps from %s", json_path)
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for entry in payload:
                timestamp = float(entry.get("timestamp", 0.0))
                confidence = float(entry.get("confidence", 0.5))
                signals.append(
                    DetectionSignal(timestamp=timestamp, confidence=confidence, detector=self.name, data=entry)
                )
        elif csv_path.exists():
            logging.info("Loading elimination timestamps from %s", csv_path)
            with csv_path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        timestamp = float(line.split(",")[0])
                    except ValueError:
                        logging.debug("Skipping malformed line %d in %s", idx, csv_path)
                        continue
                    signals.append(
                        DetectionSignal(
                            timestamp=timestamp,
                            confidence=0.5,
                            detector=self.name,
                            data={"source": "csv"},
                        )
                    )
        else:
            logging.debug("No preprocessed elimination files located in %s", directory)
        return signals


class HeuristicKillFeedDetector(BaseDetector):
    """Placeholder detector using rough heuristics on optional OCR dumps."""

    name = "kill_feed"

    def detect(self, config: PipelineConfig, metadata: VideoMetadata) -> List[DetectionSignal]:
        if not config.preprocessed_dir:
            logging.debug("Kill feed detector skipped (no preprocessed dir).")
            return []
        kill_feed_file = config.preprocessed_dir / "kill_feed.json"
        if not kill_feed_file.exists():
            logging.debug("Kill feed detector skipped (missing %s).", kill_feed_file)
            return []

        with kill_feed_file.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        signals: List[DetectionSignal] = []
        for entry in entries:
            phrase = entry.get("text", "").lower()
            if any(keyword in phrase for keyword in ["eliminated", "you eliminated", "knocked"]):
                timestamp = float(entry.get("timestamp", 0.0))
                confidence = float(entry.get("confidence", 0.7))
                signals.append(
                    DetectionSignal(
                        timestamp=timestamp,
                        confidence=confidence,
                        detector=self.name,
                        data=entry,
                    )
                )
        return signals


class HeuristicAudioDetector(BaseDetector):
    """Audio based heuristic using precomputed spectral peaks when available."""

    name = "audio"

    def detect(self, config: PipelineConfig, metadata: VideoMetadata) -> List[DetectionSignal]:
        if not config.preprocessed_dir:
            logging.debug("Audio detector skipped (no preprocessed dir).")
            return []
        peaks_file = config.preprocessed_dir / "audio_peaks.json"
        if not peaks_file.exists():
            logging.debug("Audio detector skipped (missing %s).", peaks_file)
            return []

        with peaks_file.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        signals: List[DetectionSignal] = []
        for entry in entries:
            if entry.get("label") != "elimination":
                continue
            timestamp = float(entry.get("timestamp", 0.0))
            confidence = float(entry.get("confidence", 0.6))
            signals.append(
                DetectionSignal(
                    timestamp=timestamp,
                    confidence=confidence,
                    detector=self.name,
                    data=entry,
                )
            )
        return signals


class DetectionFusion:
    """Fuse detection signals into consolidated elimination events."""

    def __init__(self, tolerance: float = 0.75):
        self.tolerance = tolerance

    def fuse(self, signals: Sequence[DetectionSignal]) -> List[EliminationEvent]:
        if not signals:
            return []
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        events: List[EliminationEvent] = []
        current_group: List[DetectionSignal] = []
        group_start: Optional[float] = None

        for signal in sorted_signals:
            if group_start is None:
                group_start = signal.timestamp
            if signal.timestamp - group_start <= self.tolerance:
                current_group.append(signal)
            else:
                events.extend(self._finalise_group(current_group))
                current_group = [signal]
                group_start = signal.timestamp
        if current_group:
            events.extend(self._finalise_group(current_group))
        return events

    def _finalise_group(self, group: List[DetectionSignal]) -> List[EliminationEvent]:
        if not group:
            return []
        timestamp = sum(signal.timestamp for signal in group) / len(group)
        confidence = max(signal.confidence for signal in group)
        elim_id = f"{len(group):04d}-{abs(hash(timestamp)) % 1_000_000:06d}"
        event = EliminationEvent(elim_id=elim_id, timestamp=timestamp, confidence=confidence, signals=list(group))
        logging.debug("Fused event %s with %d signals", event.elim_id, len(group))
        return [event]


class ClipExtractor:
    """Handles extraction of short video clips around elimination times."""

    def __init__(self, config: PipelineConfig, metadata: VideoMetadata):
        self.config = config
        self.metadata = metadata

    def extract_clip(self, event: EliminationEvent, clips_dir: Path) -> Optional[Path]:
        start = max(event.timestamp - self.config.pre_window, 0.0)
        duration = self.config.pre_window + self.config.post_window
        clips_dir.mkdir(parents=True, exist_ok=True)
        out_path = clips_dir / f"{event.elim_id}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(self.config.video_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(out_path),
        ]
        if self.config.gpu_device is not None:
            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                str(self.config.gpu_device),
                "-ss",
                f"{start:.3f}",
                "-i",
                str(self.config.video_path),
                "-t",
                f"{duration:.3f}",
                "-c",
                "copy",
                str(out_path),
            ]
        logging.debug("Extracting clip with command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - ffmpeg failure
            logging.warning("Failed to extract clip for %s: %s", event.elim_id, exc)
            return None
        return out_path


class PreprocessedFeatureLoader:
    """Loads precomputed features when present."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.index: Dict[str, Dict[str, Any]] = {}
        if directory.exists():
            cache_file = directory / "aim_features.json"
            if cache_file.exists():
                try:
                    with cache_file.open("r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    for entry in payload:
                        elim_id = str(entry.get("elim_id"))
                        if elim_id:
                            self.index[elim_id] = entry
                except json.JSONDecodeError:
                    logging.warning("Failed to parse %s", cache_file)

    def get(self, elim_id: str) -> Optional[Dict[str, Any]]:
        return self.index.get(elim_id)


class AimFeatureExtractor:
    """Computes aim-related features for an elimination event."""

    def __init__(self, config: PipelineConfig, metadata: VideoMetadata):
        self.config = config
        self.metadata = metadata
        self.preprocessed_loader = (
            PreprocessedFeatureLoader(config.preprocessed_dir) if config.preprocessed_dir else None
        )

    def extract(self, event: EliminationEvent) -> EliminationFeatures:
        features = EliminationFeatures(elim_id=event.elim_id, timestamp=event.timestamp)
        if self.preprocessed_loader:
            cached = self.preprocessed_loader.get(event.elim_id)
            if cached:
                logging.debug("Using preprocessed features for %s", event.elim_id)
                self._populate_from_mapping(features, cached)
                return features

        reaction_ms, dwell_ms = self._estimate_timing_from_signals(event.signals)
        snap_magnitude_px = self._estimate_snap_from_signals(event.signals)
        micro_adjusts = self._estimate_micro_adjusts(event.signals)

        features.reaction_ms = reaction_ms
        features.dwell_ms = dwell_ms
        features.micro_adjusts_per_s = micro_adjusts
        features.snap_magnitude_px = snap_magnitude_px
        features.snap_frames = None
        features.headshot = self._infer_headshot(event.signals)
        features.shots_count = self._infer_shots(event.signals)
        features.headshot_ratio = None
        features.aim_variance = None
        features.muzzle_flash_count = None
        return features

    def _populate_from_mapping(self, features: EliminationFeatures, mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            if hasattr(features, key):
                setattr(features, key, value)

    def _estimate_timing_from_signals(self, signals: Sequence[DetectionSignal]) -> Tuple[Optional[float], Optional[float]]:
        timestamps = [signal.timestamp for signal in signals if signal.detector == "audio"]
        if not timestamps:
            return None, None
        spread = max(timestamps) - min(timestamps)
        reaction_ms = max(0.0, 1000.0 * (spread / max(len(timestamps), 1)))
        dwell_ms = 1000.0 * spread
        return reaction_ms, dwell_ms

    def _estimate_snap_from_signals(self, signals: Sequence[DetectionSignal]) -> Optional[float]:
        if np is None:
            return None
        magnitudes = [signal.data.get("snap_magnitude", 0.0) for signal in signals if "snap_magnitude" in signal.data]
        if not magnitudes:
            return None
        return float(np.mean(magnitudes))

    def _estimate_micro_adjusts(self, signals: Sequence[DetectionSignal]) -> Optional[float]:
        counts = [signal.data.get("micro_adjusts", 0) for signal in signals if "micro_adjusts" in signal.data]
        if not counts:
            return None
        duration = max(self.config.pre_window, 1.0)
        return sum(counts) / duration

    def _infer_headshot(self, signals: Sequence[DetectionSignal]) -> Optional[bool]:
        for signal in signals:
            if "headshot" in signal.data:
                return bool(signal.data.get("headshot"))
        return None

    def _infer_shots(self, signals: Sequence[DetectionSignal]) -> Optional[int]:
        shot_counts = [signal.data.get("shots", 1) for signal in signals if "shots" in signal.data]
        if not shot_counts:
            return None
        return int(max(shot_counts))


class SuspicionScorer:
    """Combine features into a suspicion score and flags."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def score(self, features: Iterable[EliminationFeatures]) -> List[EliminationFeatures]:
        scored: List[EliminationFeatures] = []
        reaction_values: List[float] = []
        for feat in features:
            if feat.reaction_ms is not None:
                reaction_values.append(feat.reaction_ms)

        reaction_std = statistics.pstdev(reaction_values) if len(reaction_values) > 1 else None
        for feat in features:
            score, flags = self._score_single(feat, reaction_std)
            feat.suspicion_score = score
            feat.flags = flags
            scored.append(feat)
        return scored

    def _score_single(
        self, feat: EliminationFeatures, global_reaction_std: Optional[float]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        flags: List[str] = []

        if feat.reaction_ms is not None:
            if feat.reaction_ms < self.config.min_reaction_ms:
                score += 0.4
                flags.append("fast_reaction")
            score += max(0.0, (self.config.min_reaction_ms - feat.reaction_ms) / 100.0)

        if feat.snap_magnitude_px is not None:
            normalized_snap = min(feat.snap_magnitude_px / 300.0, 1.0)
            if normalized_snap > 0.5:
                score += 0.3 * normalized_snap
                flags.append("large_snap")

        if feat.micro_adjusts_per_s is not None and feat.micro_adjusts_per_s < 1.0:
            score += 0.1
            flags.append("low_micro_adjusts")

        if global_reaction_std is not None and global_reaction_std < 20:
            score += 0.2
            flags.append("low_reaction_variance")

        if feat.headshot:
            score += 0.05
            flags.append("headshot")

        score = min(score, 1.0)
        return score, flags


class ReportWriter:
    """Generates CSV/HTML reports from scored features."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def write(self, features: Sequence[EliminationFeatures]) -> Tuple[Optional[Path], Optional[Path]]:
        if not features:
            logging.warning("No elimination features available to write reports.")
            return None, None

        reports_dir = self.config.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "elimination_report.csv"
        html_path = reports_dir / "elimination_report.html"

        rows = [dataclasses.asdict(feature) for feature in features]
        for row in rows:
            flags = row.get("flags") or []
            row["flags"] = ",".join(flags)

        if pd is not None:
            frame = pd.DataFrame(rows)
            frame.sort_values(by="suspicion_score", ascending=False, inplace=True)
            frame.to_csv(csv_path, index=False)
            html_content = self._build_html(frame)
            html_path.write_text(html_content, encoding="utf-8")
        else:
            import csv

            headers = sorted(rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            html_path.write_text("<html><body><p>Pandas not installed; CSV only.</p></body></html>", encoding="utf-8")
        logging.info("Reports written to %s and %s", csv_path, html_path)
        return csv_path, html_path

    def _build_html(self, frame: "pd.DataFrame") -> str:  # type: ignore[name-defined]
        top = frame.head(20)
        table_html = top.to_html(index=False, escape=True)
        summary = f"<p>Generated at {datetime.utcnow().isoformat()}Z</p>"
        html = f"""
        <html>
        <head>
            <title>Aim Suspicion Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
                th {{ background: #f0f0f0; }}
            </style>
        </head>
        <body>
            <h1>Aim Suspicion Report</h1>
            {summary}
            <h2>Top 20 Suspicious Eliminations</h2>
            {table_html}
        </body>
        </html>
        """
        return html


class AimDetectionPipeline:
    """Main orchestration class for the aim detection pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metadata: Optional[VideoMetadata] = None
        self.detectors: List[BaseDetector] = [
            PreprocessedDetector(),
            HeuristicKillFeedDetector(),
            HeuristicAudioDetector(),
        ]
        self.fusion = DetectionFusion()
        self.scorer = SuspicionScorer(config)

    def run(self) -> None:
        logging.info("Starting aim detection pipeline for %s", self.config.video_path)
        self.metadata = FFprobe.probe(self.config.video_path)
        all_signals = self._run_detectors()
        events = self.fusion.fuse(all_signals)
        logging.info("Detected %d elimination events", len(events))

        clip_extractor = ClipExtractor(self.config, self.metadata)
        feature_extractor = AimFeatureExtractor(self.config, self.metadata)
        clips_dir = self.config.output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        features: List[EliminationFeatures] = []
        for idx, event in enumerate(events, start=1):
            elim_id = f"{idx:04d}"
            event.elim_id = elim_id
            clip_path = clip_extractor.extract_clip(event, clips_dir)
            extracted = feature_extractor.extract(event)
            if clip_path:
                extracted.clip_file = str(Path("clips") / clip_path.name)
            features.append(extracted)

        scored_features = self.scorer.score(features)
        scored_features.sort(key=lambda f: f.suspicion_score or 0.0, reverse=True)
        ReportWriter(self.config).write(scored_features)
        logging.info("Pipeline completed. Processed %d eliminations.", len(scored_features))

    def _run_detectors(self) -> List[DetectionSignal]:
        signals: List[DetectionSignal] = []
        for detector in self.detectors:
            try:
                detector_signals = detector.detect(self.config, self.metadata or VideoMetadata(60, 1920, 1080, 0))
                logging.info("%s detector produced %d signals", detector.name, len(detector_signals))
                signals.extend(detector_signals)
            except Exception as exc:  # pragma: no cover - detectors may fail unexpectedly
                logging.warning("Detector %s failed: %s", detector.name, exc)
                if self.config.debug:
                    raise
        return signals


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Detect suspicious aim around eliminations in gameplay videos.")
    parser.add_argument("--video", dest="video", required=True, help="Path to the gameplay video (mp4)")
    parser.add_argument("--out", dest="out", required=True, help="Output directory for clips and reports")
    parser.add_argument("--gpu", dest="gpu", type=int, default=None, help="GPU device index for decoding")
    parser.add_argument("--pre", dest="pre", type=float, default=10.0, help="Seconds to include before elimination")
    parser.add_argument("--post", dest="post", type=float, default=2.0, help="Seconds to include after elimination")
    parser.add_argument(
        "--min_reaction_ms",
        dest="min_reaction_ms",
        type=float,
        default=40.0,
        help="Reaction time threshold considered suspicious",
    )
    parser.add_argument(
        "--suspicion_thresh",
        dest="suspicion_thresh",
        type=float,
        default=0.8,
        help="Threshold above which eliminations are flagged",
    )
    parser.add_argument(
        "--frame-sample",
        dest="frame_sample",
        type=int,
        default=1,
        help="Sample every Nth frame during analysis",
    )
    parser.add_argument(
        "--chunk-duration",
        dest="chunk_duration",
        type=float,
        default=None,
        help="Optional chunk duration (seconds) for long videos",
    )
    parser.add_argument(
        "--preprocessed-dir",
        dest="preprocessed_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed detections/features",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Raise exceptions instead of swallowing them",
    )

    args = parser.parse_args(argv)
    config = PipelineConfig(
        video_path=Path(args.video),
        output_dir=Path(args.out),
        gpu_device=args.gpu,
        pre_window=args.pre,
        post_window=args.post,
        frame_sample_rate=max(1, args.frame_sample),
        suspicion_threshold=args.suspicion_thresh,
        min_reaction_ms=args.min_reaction_ms,
        chunk_duration=args.chunk_duration,
        preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None,
        debug=args.debug,
    )
    return config


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    configure_logging(config.debug)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = AimDetectionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
