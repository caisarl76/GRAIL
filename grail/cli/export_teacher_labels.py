from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from grail.vla.lerobot_schema import HAND_PRIMITIVE_DIM, MOTION_TOKEN_DIM
from grail.vla.motion_library import load_motion_episode

META_ACTION_DIM = MOTION_TOKEN_DIM + HAND_PRIMITIVE_DIM


def _record_frame_index(record: dict) -> int:
    if "frame_index" in record:
        return int(record["frame_index"])
    if "step" in record:
        return int(record["step"])
    raise ValueError(f"teacher record is missing frame_index: {record}")


def _record_motion_key(record: dict) -> str:
    motion_key = record.get("motion_key")
    if not motion_key:
        raise ValueError(f"teacher record is missing motion_key: {record}")
    return str(motion_key)


def _record_actions(record: dict) -> np.ndarray:
    actions = np.asarray(record.get("actions"), dtype=np.float64).reshape(-1)
    if actions.shape != (META_ACTION_DIM,):
        raise ValueError(f"teacher actions must have shape ({META_ACTION_DIM},), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("teacher actions contain non-finite values")
    return actions


def _load_debug_records(debug_log: Path) -> list[dict]:
    records = json.loads(debug_log.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"teacher debug log must contain a JSON list: {debug_log}")
    return records


def _expected_motion_frames(motion_lib: Path, motion_key: str) -> int:
    return load_motion_episode(motion_lib / "robot" / f"{motion_key}.pkl").num_frames


def _resample_labels_to_motion_frames(
    labels: np.ndarray,
    *,
    motion_key: str,
    expected_frames: int,
) -> np.ndarray:
    if labels.shape[0] == expected_frames:
        return labels
    if labels.shape[0] < expected_frames:
        raise ValueError(
            f"{motion_key} teacher labels shorter than motion_lib: "
            f"labels={labels.shape[0]}, motion_lib={expected_frames}"
        )
    indices = np.rint(np.linspace(0, labels.shape[0] - 1, expected_frames)).astype(np.int64)
    return labels[indices]


def _first_done_index(done_flags: dict[int, bool]) -> Optional[int]:
    done_indices = [index for index, flag in done_flags.items() if flag]
    return min(done_indices) if done_indices else None


def export_teacher_labels(
    debug_log: str | Path,
    output: str | Path,
    *,
    motion_lib: Optional[str | Path] = None,
    overwrite: bool = False,
    resample_to_motion_lib: bool = False,
    truncate_at_done: bool = False,
    skip_unknown_motions: bool = False,
    dedupe_overflow_envs: bool = False,
) -> int:
    """Convert SONIC teacher debug records into per-motion 66-D label arrays."""

    debug_path = Path(debug_log)
    output_path = Path(output)
    motion_lib_path = Path(motion_lib) if motion_lib is not None else None

    grouped: dict[str, dict[int, np.ndarray]] = {}
    done_flags: dict[str, dict[int, bool]] = {}
    for record in _load_debug_records(debug_path):
        motion_key = _record_motion_key(record)
        frame_index = _record_frame_index(record)
        if frame_index < 0:
            raise ValueError(f"negative frame_index for {motion_key}: {frame_index}")
        grouped.setdefault(motion_key, {})
        done_flags.setdefault(motion_key, {})
        if frame_index in grouped[motion_key]:
            # Overflow envs (num_envs > motions in the shard) replay an existing
            # motion, duplicating its rows. Keep the first (real) env's stream.
            if dedupe_overflow_envs:
                continue
            raise ValueError(f"duplicate frame_index {frame_index} for {motion_key}")
        grouped[motion_key][frame_index] = _record_actions(record)
        done_flags[motion_key][frame_index] = bool(record.get("done", False))

    if not grouped:
        raise ValueError(f"teacher debug log has no records: {debug_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    exported = 0
    for motion_key, frames in sorted(grouped.items()):
        if (
            skip_unknown_motions
            and motion_lib_path is not None
            and not (motion_lib_path / "robot" / f"{motion_key}.pkl").exists()
        ):
            continue

        if truncate_at_done:
            cutoff = _first_done_index(done_flags[motion_key])
            if cutoff is not None:
                frames = {index: frames[index] for index in frames if index <= cutoff}

        expected_indices = list(range(len(frames)))
        actual_indices = sorted(frames)
        if actual_indices != expected_indices:
            raise ValueError(
                f"{motion_key} frame indices must be contiguous from 0; got {actual_indices[:5]}"
            )

        labels = np.stack([frames[index] for index in expected_indices], axis=0)
        if motion_lib_path is not None:
            expected_frames = _expected_motion_frames(motion_lib_path, motion_key)
            if labels.shape[0] != expected_frames:
                if not resample_to_motion_lib:
                    raise ValueError(
                        f"{motion_key} frame count mismatch: labels={labels.shape[0]}, "
                        f"motion_lib={expected_frames}"
                    )
                labels = _resample_labels_to_motion_frames(
                    labels,
                    motion_key=motion_key,
                    expected_frames=expected_frames,
                )

        output_file = output_path / f"{motion_key}.npy"
        if output_file.exists() and not overwrite:
            raise FileExistsError(f"Teacher label already exists: {output_file}")
        np.save(output_file, labels.astype(np.float32))
        exported += 1

    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-motion 66-D SONIC teacher labels from token_debug.json records."
    )
    parser.add_argument("--debug-log", required=True, help="SONIC debug JSON from eval callback")
    parser.add_argument("--output", required=True, help="Output directory for <motion_key>.npy labels")
    parser.add_argument(
        "--motion-lib",
        help="Optional GRAIL motion library root; validates label frame counts against robot/*.pkl",
    )
    parser.add_argument(
        "--resample-to-motion-lib",
        action="store_true",
        help=(
            "When --motion-lib is set, downsample a longer SONIC eval/debug stream to the "
            "motion-lib frame count. Shorter streams still fail as incomplete."
        ),
    )
    parser.add_argument(
        "--truncate-at-done",
        action="store_true",
        help=(
            "Truncate each motion's teacher stream at its first done=True frame. Required for "
            "multi-env batches, where short motions auto-reset and replay garbage after time-out."
        ),
    )
    parser.add_argument(
        "--skip-unknown-motions",
        action="store_true",
        help=(
            "When --motion-lib is set, skip motion keys with no robot/<key>.pkl (overflow/padding "
            "envs) instead of failing."
        ),
    )
    parser.add_argument(
        "--dedupe-overflow-envs",
        action="store_true",
        help=(
            "Drop duplicate (motion_key, frame_index) rows from overflow envs that replay an "
            "existing motion when num_envs exceeds the motions in a shard. Keeps the first env."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing label files")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exported = export_teacher_labels(
        args.debug_log,
        args.output,
        motion_lib=args.motion_lib,
        overwrite=args.overwrite,
        resample_to_motion_lib=args.resample_to_motion_lib,
        truncate_at_done=args.truncate_at_done,
        skip_unknown_motions=args.skip_unknown_motions,
        dedupe_overflow_envs=args.dedupe_overflow_envs,
    )
    print(f"Exported teacher labels: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
