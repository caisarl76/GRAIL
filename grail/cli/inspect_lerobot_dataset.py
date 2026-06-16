from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


REQUIRED_SONIC_VLA_COLUMNS = (
    "observation.state",
    "action.wbc",
    "action.motion_token",
    "action.hand_primitive",
)


@dataclass
class DatasetInspection:
    root: Path
    episodes: int
    episode_frames: int
    info_total_episodes: Optional[int]
    info_total_frames: Optional[int]
    video_keys: list[str]
    checked_episodes: int
    missing_parquet: list[Path] = field(default_factory=list)
    missing_videos: list[Path] = field(default_factory=list)
    extra_parquet: list[Path] = field(default_factory=list)
    extra_videos: list[Path] = field(default_factory=list)
    parquet_rows_checked: int = 0
    parquet_read_status: str = "not requested"
    first_columns: list[str] = field(default_factory=list)
    timestamp_checks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.errors
            and not self.missing_parquet
            and not self.missing_videos
            and not self.extra_parquet
            and not self.extra_videos
        )


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_episode_path(pattern: str, episode_index: int, chunks_size: int) -> str:
    episode_chunk = episode_index // chunks_size
    return pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _get_video_keys(info: dict) -> list[str]:
    video_keys = list(info.get("video_keys", []))
    if video_keys:
        return video_keys
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in ("video", "image")
    ]


def _read_parquet_table(path: Path) -> tuple[Optional[Any], Optional[int], list[str], Optional[str]]:
    try:
        import pandas as pd
    except Exception as exc:
        return None, None, [], f"skipped: pandas unavailable ({type(exc).__name__}: {exc})"

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return None, None, [], f"failed: {path} ({type(exc).__name__}: {exc})"
    return frame, len(frame), list(frame.columns), None


def _table_columns(table: Any) -> list[str]:
    if hasattr(table, "columns"):
        return list(table.columns)
    return list(table.keys())


def _column_array(table: Any, key: str) -> np.ndarray:
    column = table[key]
    if hasattr(column, "to_numpy"):
        values = column.to_numpy()
    else:
        values = np.asarray(column)
    if values.dtype == object:
        values = np.asarray([np.asarray(value).reshape(-1)[0] for value in values])
    return np.asarray(values).reshape(-1)


def validate_episode_table(
    table: Any,
    parquet_path: Path,
    episode_index: int,
    episode_length: int,
    fps: int,
    tolerance_s: float = 1e-4,
) -> list[str]:
    columns = set(_table_columns(table))
    errors = []

    missing_required = [key for key in REQUIRED_SONIC_VLA_COLUMNS if key not in columns]
    if missing_required:
        errors.append(f"{parquet_path} missing required columns: {missing_required}")

    if "episode_index" in columns:
        episode_indices = _column_array(table, "episode_index").astype(np.int64)
        if len(episode_indices) != episode_length:
            errors.append(
                f"{parquet_path} episode_index length {len(episode_indices)} != {episode_length}"
            )
        elif not np.all(episode_indices == episode_index):
            observed = sorted(set(int(value) for value in episode_indices[:10]))
            errors.append(
                f"{parquet_path} has episode_index values {observed}, expected only {episode_index}"
            )

    if "frame_index" in columns:
        frame_indices = _column_array(table, "frame_index").astype(np.int64)
        expected = np.arange(episode_length, dtype=np.int64)
        if len(frame_indices) != episode_length or not np.array_equal(frame_indices, expected):
            errors.append(
                f"{parquet_path} frame_index is not 0..{episode_length - 1}; "
                f"first values={frame_indices[:5].tolist()}"
            )

    if "timestamp" in columns:
        timestamps = _column_array(table, "timestamp").astype(np.float64)
        expected = np.arange(episode_length, dtype=np.float64) / float(fps)
        if len(timestamps) != episode_length:
            errors.append(f"{parquet_path} timestamp length {len(timestamps)} != {episode_length}")
        elif not np.all(np.isfinite(timestamps)):
            errors.append(f"{parquet_path} timestamp contains non-finite values")
        else:
            max_error = float(np.max(np.abs(timestamps - expected))) if len(timestamps) else 0.0
            if max_error > tolerance_s:
                reset_indices = np.where(np.diff(timestamps) < -tolerance_s)[0]
                reset_hint = ""
                if len(reset_indices):
                    index = int(reset_indices[0])
                    reset_hint = f"; reset near rows {index}->{index + 1}: {timestamps[index:index + 2].tolist()}"
                errors.append(
                    f"{parquet_path} timestamp max error {max_error:.6f}s exceeds "
                    f"tolerance {tolerance_s:.6f}s{reset_hint}"
                )

    return errors


def inspect_dataset(
    root: Path,
    max_episodes: int = 3,
    read_parquet: bool = True,
    require_readable_parquet: bool = False,
) -> DatasetInspection:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot episodes file: {episodes_path}")

    info = _load_json(info_path)
    episodes = _load_jsonl(episodes_path)
    chunks_size = int(info.get("chunks_size", 1000))
    fps = int(info.get("fps", 50))
    data_pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_pattern = info.get(
        "video_path",
        "videos/{video_key}/episode_{episode_index:06d}.mp4",
    )
    video_keys = _get_video_keys(info)

    result = DatasetInspection(
        root=root,
        episodes=len(episodes),
        episode_frames=sum(int(episode.get("length", 0)) for episode in episodes),
        info_total_episodes=info.get("total_episodes"),
        info_total_frames=info.get("total_frames"),
        video_keys=video_keys,
        checked_episodes=min(len(episodes), max_episodes if max_episodes > 0 else len(episodes)),
    )

    if result.info_total_episodes is not None and result.info_total_episodes != result.episodes:
        result.errors.append(
            f"info total_episodes={result.info_total_episodes}, episodes.jsonl has {result.episodes}"
        )
    if result.info_total_frames is not None and result.info_total_frames != result.episode_frames:
        result.errors.append(
            f"info total_frames={result.info_total_frames}, episodes.jsonl sums to {result.episode_frames}"
        )

    expected_parquet = {
        root / _format_episode_path(data_pattern, int(episode["episode_index"]), chunks_size)
        for episode in episodes
    }
    actual_parquet = set((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else set()
    result.extra_parquet = sorted(actual_parquet - expected_parquet)

    expected_videos = set()
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        for video_key in video_keys:
            expected_videos.add(
                root
                / video_pattern.format(
                    video_key=video_key,
                    episode_chunk=episode_index // chunks_size,
                    episode_index=episode_index,
                )
            )
    actual_videos = set((root / "videos").rglob("*.mp4")) if (root / "videos").is_dir() else set()
    result.extra_videos = sorted(actual_videos - expected_videos)

    for episode in episodes[: result.checked_episodes]:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        parquet_path = root / _format_episode_path(data_pattern, episode_index, chunks_size)
        if not parquet_path.is_file():
            result.missing_parquet.append(parquet_path)
        elif read_parquet:
            table, row_count, columns, status = _read_parquet_table(parquet_path)
            if status is not None:
                result.parquet_read_status = status
                if require_readable_parquet or status.startswith("failed:"):
                    result.errors.append(status)
            else:
                result.parquet_read_status = "ok"
                result.parquet_rows_checked += row_count or 0
                if row_count != episode_length:
                    result.errors.append(
                        f"{parquet_path} has {row_count} rows, episode length is {episode_length}"
                    )
                if table is not None:
                    table_errors = validate_episode_table(
                        table,
                        parquet_path=parquet_path,
                        episode_index=episode_index,
                        episode_length=episode_length,
                        fps=fps,
                    )
                    result.errors.extend(table_errors)
                    result.timestamp_checks += 1
                if not result.first_columns:
                    result.first_columns = columns

        for video_key in video_keys:
            rel_video = video_pattern.format(
                video_key=video_key,
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
            video_path = root / rel_video
            if not video_path.is_file():
                result.missing_videos.append(video_path)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a LeRobot v2.1 dataset on disk.")
    parser.add_argument("--root", required=True, help="LeRobot dataset root")
    parser.add_argument("--max-episodes", type=int, default=3, help="Episodes to check; 0 checks all")
    parser.add_argument("--no-read-parquet", action="store_true", help="Only check parquet file existence")
    parser.add_argument(
        "--require-readable-parquet",
        action="store_true",
        help="Fail if pandas/pyarrow cannot read checked parquet files",
    )
    return parser


def _print_result(result: DatasetInspection) -> None:
    print(f"root: {result.root}")
    print(f"episodes: {result.episodes}")
    print(f"frames: {result.episode_frames}")
    print(f"info_total_episodes: {result.info_total_episodes}")
    print(f"info_total_frames: {result.info_total_frames}")
    print(f"video_keys: {result.video_keys}")
    print(f"checked_episodes: {result.checked_episodes}")
    print(f"missing_parquet: {len(result.missing_parquet)}")
    print(f"missing_videos: {len(result.missing_videos)}")
    print(f"extra_parquet: {len(result.extra_parquet)}")
    print(f"extra_videos: {len(result.extra_videos)}")
    print(f"parquet_read_status: {result.parquet_read_status}")
    if result.parquet_rows_checked:
        print(f"parquet_rows_checked: {result.parquet_rows_checked}")
    if result.timestamp_checks:
        print(f"timestamp_checks: {result.timestamp_checks}")
    if result.first_columns:
        print(f"first_columns: {result.first_columns}")
    for error in result.errors:
        print(f"ERROR: {error}")
    for path in result.missing_parquet[:5]:
        print(f"MISSING_PARQUET: {path}")
    for path in result.missing_videos[:5]:
        print(f"MISSING_VIDEO: {path}")
    for path in result.extra_parquet[:5]:
        print(f"EXTRA_PARQUET: {path}")
    for path in result.extra_videos[:5]:
        print(f"EXTRA_VIDEO: {path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = inspect_dataset(
        Path(args.root),
        max_episodes=args.max_episodes,
        read_parquet=not args.no_read_parquet,
        require_readable_parquet=args.require_readable_parquet,
    )
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
