"""Gameplay elimination detection and aim heuristics pipeline.

This script scans a gameplay video, detects potential elimination moments using
simple visual heuristics, extracts highlight clips, computes lightweight aim
metrics, and produces CSV/HTML reports ranking suspicious eliminations.

It intentionally avoids heavyweight ML dependencies so that it can operate on a
vanilla Python + OpenCV stack while still yielding useful, inspectable output.
The heuristics are basic but provide a concrete baseline that can be extended
with richer detectors in future iterations.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    video_path: Path
    output_dir: Path
    pre_window: float = 10.0
    post_window: float = 2.0
    frame_stride: int = 1
    min_event_gap: float = 1.5
    detection_std_factor: float = 2.5
    min_reaction_ms: float = 40.0
    center_ratio: float = 0.18
    debug: bool = False


@dataclass
class VideoMetadata:
    """Lightweight container describing the input video."""

    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


@dataclass
class FrameScore:
    """Score captured for a processed frame."""

    timestamp: float
    combined_score: float
    global_score: float
    center_score: float
    color_score: float
    center_brightness: float


@dataclass
class EliminationEvent:
    """Represents a detected elimination."""

    elim_id: str
    timestamp: float
    score: float
    source: FrameScore


@dataclass
class EliminationFeatures:
    """Aim heuristics computed for an elimination."""

    elim_id: str
    timestamp: float
    clip_file: Optional[str]
    reaction_ms: Optional[float]
    dwell_ms: Optional[float]
    micro_adjusts_per_s: Optional[float]
    snap_magnitude_px: Optional[float]
    snap_frames: Optional[int]
    headshot: Optional[bool]
    shots_count: Optional[int]
    headshot_ratio: Optional[float]
    aim_variance: Optional[float]
    muzzle_flash_count: Optional[int]
    suspicion_score: float
    flags: List[str]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _ensure_positive(value: float, fallback: float) -> float:
    return value if value and value > 0 else fallback


def load_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = _ensure_positive(cap.get(cv2.CAP_PROP_FPS), 30.0)
    width = int(_ensure_positive(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0))
    height = int(_ensure_positive(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0))
    frame_count_val = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = 0.0
    if frame_count_val > 0:
        duration = frame_count_val / fps
    cap.release()

    return VideoMetadata(
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count_val,
        duration=duration,
    )


def _center_slice(height: int, width: int, ratio: float) -> tuple[int, int, int, int]:
    half = int(min(height, width) * max(min(ratio, 0.8), 0.02) * 0.5)
    half = max(4, half)
    cy = height // 2
    cx = width // 2
    y0 = max(0, cy - half)
    y1 = min(height, cy + half)
    x0 = max(0, cx - half)
    x1 = min(width, cx + half)
    return y0, y1, x0, x1


# ---------------------------------------------------------------------------
# Detection pass
# ---------------------------------------------------------------------------

def collect_frame_scores(
    video_path: Path, metadata: VideoMetadata, config: PipelineConfig
) -> tuple[List[FrameScore], VideoMetadata]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = metadata.fps
    prev_gray: Optional[np.ndarray] = None
    scores: List[FrameScore] = []
    frame_idx = 0
    processed_frames = 0

    width = metadata.width
    height = metadata.height

    y0 = y1 = x0 = x1 = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if width == 0 or height == 0:
            height, width = frame.shape[:2]
            y0, y1, x0, x1 = _center_slice(height, width, config.center_ratio)
        elif processed_frames == 0:
            y0, y1, x0, x1 = _center_slice(height, width, config.center_ratio)

        if frame_idx % max(config.frame_stride, 1) != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            processed_frames += 1
            continue

        diff = cv2.absdiff(gray, prev_gray)
        center_diff = diff[y0:y1, x0:x1]
        global_score = float(np.mean(diff))
        center_score = float(np.mean(center_diff))

        red_channel = frame[:, :, 2]
        red_center = red_channel[y0:y1, x0:x1]
        bright_red = float(np.mean(red_center > 200)) * 255.0

        center_patch = gray[y0:y1, x0:x1]
        center_brightness = float(np.mean(center_patch))

        combined = global_score * 0.5 + center_score * 0.35 + bright_red * 0.15
        timestamp = (frame_idx - 1) / fps

        scores.append(
            FrameScore(
                timestamp=timestamp,
                combined_score=combined,
                global_score=global_score,
                center_score=center_score,
                color_score=bright_red,
                center_brightness=center_brightness,
            )
        )

        prev_gray = gray
        processed_frames += 1

    cap.release()

    duration = metadata.duration
    if processed_frames > 0:
        last_timestamp = (frame_idx - 1) / fps
        duration = max(duration, last_timestamp)

    updated_metadata = VideoMetadata(
        fps=metadata.fps,
        width=width,
        height=height,
        frame_count=frame_idx,
        duration=duration,
    )
    logging.debug("Collected %d frame scores", len(scores))
    return scores, updated_metadata


def detect_eliminations(
    scores: Sequence[FrameScore], config: PipelineConfig
) -> List[EliminationEvent]:
    if not scores:
        return []

    values = np.array([s.combined_score for s in scores], dtype=float)
    baseline = float(np.mean(values)) if values.size else 0.0
    stdev = float(np.std(values)) if values.size else 0.0
    threshold = baseline + config.detection_std_factor * stdev
    if not math.isfinite(threshold) or threshold <= baseline:
        threshold = baseline + max(5.0, baseline * 0.5)

    candidates = [s for s in scores if s.combined_score >= threshold]
    candidates.sort(key=lambda s: s.timestamp)

    merged: List[FrameScore] = []
    for cand in candidates:
        if not merged:
            merged.append(cand)
            continue
        if cand.timestamp - merged[-1].timestamp < config.min_event_gap:
            if cand.combined_score > merged[-1].combined_score:
                merged[-1] = cand
            continue
        merged.append(cand)

    events = [
        EliminationEvent(
            elim_id=f"{idx + 1:04d}",
            timestamp=score.timestamp,
            score=score.combined_score,
            source=score,
        )
        for idx, score in enumerate(merged)
    ]

    logging.info("Detected %d elimination candidates", len(events))
    return events


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_clip(
    video_path: Path,
    output_path: Path,
    start: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.warning("ffmpeg clipping failed; falling back to OpenCV writer")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Unable to open video for clip fallback: %s", video_path)
        return

    fps = _ensure_positive(cap.get(cv2.CAP_PROP_FPS), 30.0)
    start_frame = max(int(start * fps), 0)
    end_frame = start_frame + int(duration * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(_ensure_positive(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 640))
    height = int(_ensure_positive(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()


def _compute_frame_statistics(
    video_path: Path,
    metadata: VideoMetadata,
    event: EliminationEvent,
    config: PipelineConfig,
) -> List[dict[str, float]]:
    fps = metadata.fps
    start_time = max(event.timestamp - config.pre_window, 0.0)
    end_time = min(event.timestamp + config.post_window, metadata.duration or event.timestamp + config.post_window)

    start_frame = max(int(start_time * fps), 0)
    total_frames = int((end_time - start_time) * fps) + 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    y0, y1, x0, x1 = _center_slice(metadata.height, metadata.width, config.center_ratio)

    frame_stats: List[dict[str, float]] = []
    prev_center_mean: Optional[float] = None
    prev_center_patch: Optional[np.ndarray] = None

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if metadata.width == 0 or metadata.height == 0:
            metadata.width = frame.shape[1]
            metadata.height = frame.shape[0]
            y0, y1, x0, x1 = _center_slice(metadata.height, metadata.width, config.center_ratio)

        center_patch = gray[y0:y1, x0:x1]
        if center_patch.size == 0:
            y0, y1, x0, x1 = _center_slice(gray.shape[0], gray.shape[1], config.center_ratio)
            center_patch = gray[y0:y1, x0:x1]

        current_ts = start_time + idx / fps
        center_mean = float(np.mean(center_patch))
        center_std = float(np.std(center_patch))
        derivative = 0.0 if prev_center_mean is None else center_mean - prev_center_mean
        local_diff = 0.0
        if prev_center_patch is not None and center_patch.size == prev_center_patch.size:
            local_diff = float(np.mean(cv2.absdiff(center_patch, prev_center_patch)))

        red_patch = frame[y0:y1, x0:x1, 2]
        red_mean = float(np.mean(red_patch))
        global_mean = float(np.mean(gray))

        frame_stats.append(
            {
                "timestamp": current_ts,
                "center_mean": center_mean,
                "center_std": center_std,
                "derivative": derivative,
                "local_diff": local_diff,
                "red_mean": red_mean,
                "global_mean": global_mean,
            }
        )

        prev_center_mean = center_mean
        prev_center_patch = center_patch.copy()

    cap.release()
    return frame_stats


def compute_features(
    video_path: Path,
    metadata: VideoMetadata,
    event: EliminationEvent,
    config: PipelineConfig,
    clip_path: Optional[Path],
) -> EliminationFeatures:
    frame_stats = _compute_frame_statistics(video_path, metadata, event, config)

    if clip_path is not None:
        try:
            clip_file = str(clip_path.relative_to(config.output_dir))
        except ValueError:
            clip_file = str(clip_path)
    else:
        clip_file = None

    if not frame_stats:
        return EliminationFeatures(
            elim_id=event.elim_id,
            timestamp=event.timestamp,
            clip_file=clip_file,
            reaction_ms=None,
            dwell_ms=None,
            micro_adjusts_per_s=None,
            snap_magnitude_px=None,
            snap_frames=None,
            headshot=None,
            shots_count=None,
            headshot_ratio=None,
            aim_variance=None,
            muzzle_flash_count=None,
            suspicion_score=0.0,
            flags=["no_data"],
        )

    pre_frames = [fs for fs in frame_stats if fs["timestamp"] <= event.timestamp]
    post_frames = [fs for fs in frame_stats if fs["timestamp"] >= event.timestamp]

    baseline_source = [fs["center_mean"] for fs in pre_frames if fs["timestamp"] < event.timestamp - 0.2]
    if not baseline_source:
        baseline_source = [fs["center_mean"] for fs in frame_stats[: max(1, len(frame_stats) // 4)]]

    baseline_mean = statistics.fmean(baseline_source)
    baseline_std = statistics.pstdev(baseline_source) if len(baseline_source) > 1 else 1.0
    reaction_threshold = baseline_mean + max(5.0, baseline_std * 1.5)

    first_above = next((fs for fs in pre_frames if fs["center_mean"] >= reaction_threshold), None)
    reaction_ms: Optional[float] = None
    if first_above:
        reaction_ms = max(0.0, (event.timestamp - first_above["timestamp"]) * 1000.0)

    dwell_ms: Optional[float] = None
    if first_above:
        dwell_start = first_above["timestamp"]
        for fs in reversed(pre_frames):
            if fs["timestamp"] <= first_above["timestamp"]:
                if fs["center_mean"] >= reaction_threshold:
                    dwell_start = fs["timestamp"]
                else:
                    break
        dwell_ms = max(0.0, (event.timestamp - dwell_start) * 1000.0)

    adjust_threshold = max(2.0, baseline_std)
    micro_adjusts = 0
    prev_sign: Optional[int] = None
    for fs in pre_frames:
        derivative = fs["derivative"]
        if abs(derivative) < adjust_threshold:
            continue
        sign = 1 if derivative > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            micro_adjusts += 1
        prev_sign = sign

    pre_duration = max(config.pre_window, 0.5)
    micro_adjusts_per_s = micro_adjusts / pre_duration

    max_derivative = max(abs(fs["derivative"]) for fs in pre_frames) if pre_frames else 0.0
    snap_magnitude_px = (max_derivative / 255.0) * (metadata.width or 1920)
    snap_frames = sum(1 for fs in pre_frames if abs(fs["derivative"]) > adjust_threshold * 2)

    muzzle_window_start = event.timestamp - 0.2
    muzzle_window_end = event.timestamp + 0.05
    muzzle_flash_count = sum(
        1
        for fs in frame_stats
        if muzzle_window_start <= fs["timestamp"] <= muzzle_window_end
        and fs["local_diff"] > adjust_threshold
    )

    post_window_end = event.timestamp + 0.4
    post_slice = [fs for fs in post_frames if fs["timestamp"] <= post_window_end]
    if not post_slice:
        post_slice = post_frames[:1]

    red_values = [fs["red_mean"] for fs in post_slice]
    red_baseline = statistics.fmean([fs["red_mean"] for fs in pre_frames[-5:]]) if len(pre_frames) >= 5 else statistics.fmean(red_values)
    headshot_threshold = red_baseline + 15.0
    headshot_frames = [fs for fs in post_slice if fs["red_mean"] >= headshot_threshold]
    headshot = bool(headshot_frames)
    headshot_ratio = len(headshot_frames) / max(len(post_slice), 1)

    shots_count = sum(
        1
        for fs in frame_stats
        if event.timestamp - 0.3 <= fs["timestamp"] <= event.timestamp + 0.3
        and fs["local_diff"] > adjust_threshold
    )

    aim_variance = statistics.pvariance([fs["center_mean"] for fs in pre_frames]) if len(pre_frames) > 1 else 0.0

    suspicion_score = 0.0
    flags: List[str] = []

    if reaction_ms is not None and reaction_ms < config.min_reaction_ms:
        suspicion_score += 0.4
        flags.append("fast_reaction")
    if dwell_ms is not None and dwell_ms < 120.0:
        suspicion_score += 0.2
        flags.append("short_dwell")
    if snap_magnitude_px is not None and snap_magnitude_px > (metadata.width or 1920) * 0.15:
        suspicion_score += 0.2
        flags.append("snap_motion")
    if micro_adjusts_per_s < 0.8:
        suspicion_score += 0.1
        flags.append("low_micro_adjusts")
    if headshot:
        suspicion_score += 0.1
        flags.append("headshot")

    suspicion_score = min(suspicion_score, 1.0)

    return EliminationFeatures(
        elim_id=event.elim_id,
        timestamp=event.timestamp,
        clip_file=clip_file,
        reaction_ms=reaction_ms,
        dwell_ms=dwell_ms,
        micro_adjusts_per_s=micro_adjusts_per_s,
        snap_magnitude_px=snap_magnitude_px,
        snap_frames=snap_frames,
        headshot=headshot,
        shots_count=shots_count,
        headshot_ratio=headshot_ratio,
        aim_variance=aim_variance,
        muzzle_flash_count=muzzle_flash_count,
        suspicion_score=suspicion_score,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _features_to_dict(feature: EliminationFeatures) -> dict[str, object]:
    data = dataclasses.asdict(feature)
    data["flags"] = ",".join(feature.flags)
    return data


def write_csv(features: Sequence[EliminationFeatures], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_features_to_dict(f) for f in features]
    fieldnames = list(rows[0].keys()) if rows else [
        "elim_id",
        "timestamp",
        "clip_file",
        "reaction_ms",
        "dwell_ms",
        "micro_adjusts_per_s",
        "snap_magnitude_px",
        "snap_frames",
        "headshot",
        "shots_count",
        "headshot_ratio",
        "aim_variance",
        "muzzle_flash_count",
        "suspicion_score",
        "flags",
    ]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_html(features: Sequence[EliminationFeatures], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in features:
        clip_link = (
            f'<a href="{f.clip_file}">{f.clip_file}</a>' if f.clip_file else ""
        )
        flags = ", ".join(f.flags) if f.flags else ""
        rows.append(
            "<tr>"
            f"<td>{f.elim_id}</td>"
            f"<td>{f.timestamp:.2f}</td>"
            f"<td>{clip_link}</td>"
            f"<td>{'' if f.reaction_ms is None else f'{f.reaction_ms:.1f}'}</td>"
            f"<td>{'' if f.dwell_ms is None else f'{f.dwell_ms:.1f}'}</td>"
            f"<td>{f.micro_adjusts_per_s:.2f if f.micro_adjusts_per_s is not None else ''}</td>"
            f"<td>{f.snap_magnitude_px:.1f if f.snap_magnitude_px is not None else ''}</td>"
            f"<td>{f.snap_frames if f.snap_frames is not None else ''}</td>"
            f"<td>{'yes' if f.headshot else 'no' if f.headshot is not None else ''}</td>"
            f"<td>{f.shots_count if f.shots_count is not None else ''}</td>"
            f"<td>{f.headshot_ratio:.2f if f.headshot_ratio is not None else ''}</td>"
            f"<td>{f.aim_variance:.2f if f.aim_variance is not None else ''}</td>"
            f"<td>{f.muzzle_flash_count if f.muzzle_flash_count is not None else ''}</td>"
            f"<td>{f.suspicion_score:.2f}</td>"
            f"<td>{flags}</td>"
            "</tr>"
        )

    header = """
    <tr>
        <th>Elimination</th>
        <th>Timestamp (s)</th>
        <th>Clip</th>
        <th>Reaction (ms)</th>
        <th>Dwell (ms)</th>
        <th>Micro adjusts / s</th>
        <th>Snap magnitude (px)</th>
        <th>Snap frames</th>
        <th>Headshot</th>
        <th>Shots</th>
        <th>Headshot ratio</th>
        <th>Aim variance</th>
        <th>Muzzle flashes</th>
        <th>Suspicion</th>
        <th>Flags</th>
    </tr>
    """

    html = (
        "<html><head><meta charset='utf-8'><title>Elimination report</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;background:#111;color:#eee;}"
        "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #444;padding:6px;}"
        "th{background:#222;}tr:nth-child(even){background:#181818;}a{color:#6cf;}</style>"
        "</head><body>"
        "<h1>Elimination review report</h1>"
        "<p>Generated by build_aim_detection_pipeline.py</p>"
        "<table>" + header + "".join(rows) + "</table>"
        "</body></html>"
    )

    output_path.write_text(html)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> List[EliminationFeatures]:
    logging.info("Loading video metadata for %s", config.video_path)
    metadata = load_video_metadata(config.video_path)

    logging.info("Collecting frame scores")
    scores, metadata = collect_frame_scores(config.video_path, metadata, config)

    logging.info("Detecting elimination events")
    events = detect_eliminations(scores, config)

    features: List[EliminationFeatures] = []

    if not events:
        logging.warning("No elimination events detected; reports will be empty")

    for event in events:
        logging.info("Processing elimination %s at %.2fs", event.elim_id, event.timestamp)
        clip_start = max(event.timestamp - config.pre_window, 0.0)
        clip_duration = config.pre_window + config.post_window
        clip_dir = config.output_dir / "clips"
        clip_path = clip_dir / f"{event.elim_id}.mp4"
        _extract_clip(config.video_path, clip_path, clip_start, clip_duration)

        feature = compute_features(
            config.video_path,
            metadata,
            event,
            config,
            clip_path,
        )
        features.append(feature)

    features.sort(key=lambda f: f.suspicion_score, reverse=True)
    return features


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Detect eliminations and compute aim heuristics from a gameplay video",
    )
    parser.add_argument("--video", dest="video", required=True, help="Path to the gameplay video")
    parser.add_argument("--out", dest="output", required=True, help="Directory for pipeline outputs")
    parser.add_argument("--pre", dest="pre", type=float, default=10.0, help="Seconds to include before elimination")
    parser.add_argument("--post", dest="post", type=float, default=2.0, help="Seconds to include after elimination")
    parser.add_argument("--stride", dest="stride", type=int, default=1, help="Frame stride for detection pass")
    parser.add_argument("--min-gap", dest="min_gap", type=float, default=1.5, help="Minimum seconds between eliminations")
    parser.add_argument("--std-factor", dest="std_factor", type=float, default=2.5, help="Std deviation multiplier for detection threshold")
    parser.add_argument("--min-reaction", dest="min_reaction", type=float, default=40.0, help="Suspicious reaction time threshold in ms")
    parser.add_argument("--center-ratio", dest="center_ratio", type=float, default=0.18, help="Fraction of frame treated as center patch")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    config = PipelineConfig(
        video_path=Path(args.video).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        pre_window=max(0.1, args.pre),
        post_window=max(0.1, args.post),
        frame_stride=max(1, args.stride),
        min_event_gap=max(0.2, args.min_gap),
        detection_std_factor=max(0.5, args.std_factor),
        min_reaction_ms=max(1.0, args.min_reaction),
        center_ratio=max(0.02, min(args.center_ratio, 0.8)),
        debug=args.debug,
    )
    return config


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    setup_logging(config.debug)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    features = run_pipeline(config)
    report_csv = config.output_dir / "reports" / "elimination_report.csv"
    report_html = config.output_dir / "reports" / "elimination_report.html"

    write_csv(features, report_csv)
    write_html(features, report_html)

    logging.info("Wrote %s", report_csv)
    logging.info("Wrote %s", report_html)


if __name__ == "__main__":
    main()
