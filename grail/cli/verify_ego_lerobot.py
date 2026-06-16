from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from grail.vla.ego_frames import EgoFrameError, find_ego_frame_source, load_ego_frames
from grail.vla.export_lerobot import write_episode_to_exporter
from grail.vla.lerobot_schema import DEFAULT_TASK
from grail.vla.motion_library import load_motion_episode
from grail.vla.render_jobs import RenderJob, build_render_jobs
from grail.vla.token_labels import TokenLabelError, attach_motion_tokens, load_motion_tokens


@dataclass
class VerificationResult:
    planned: int
    completed_sources: int
    missing_sources: int
    verified: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verified > 0 and not self.errors


class InMemoryEpisodeExporter:
    """Small LeRobot-like sink used to validate row conversion without disk IO."""

    def __init__(self) -> None:
        self.current_frames: list[dict] = []
        self.episodes: list[list[dict]] = []

    def add_frame(self, row: dict) -> None:
        _validate_sonic_vla_row(row)
        self.current_frames.append(row)

    def save_episode(self) -> None:
        if not self.current_frames:
            raise ValueError("cannot save an empty episode")
        self.episodes.append(self.current_frames)
        self.current_frames = []


def _validate_sonic_vla_row(row: dict) -> None:
    if "timestamp" in row:
        timestamp = np.asarray(row["timestamp"])
        if timestamp.shape != (1,) or timestamp.dtype != np.float32:
            raise ValueError(f"timestamp must have shape (1,) and dtype float32, got {timestamp.shape} {timestamp.dtype}")
        if not np.all(np.isfinite(timestamp)):
            raise ValueError("timestamp must be finite")

    image = np.asarray(row["observation.images.ego_view"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"ego image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"ego image must be uint8, got {image.dtype}")

    state = np.asarray(row["observation.state"])
    action = np.asarray(row["action.wbc"])
    token = np.asarray(row["action.motion_token"])
    hand_primitive = np.asarray(row["action.hand_primitive"])
    if state.shape != action.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} vs {action.shape}")
    if state.ndim != 1 or state.shape[0] != 43:
        raise ValueError(f"observation.state must have shape (43,), got {state.shape}")
    if token.shape != (64,):
        raise ValueError(f"action.motion_token must have shape (64,), got {token.shape}")
    if hand_primitive.shape != (2,):
        raise ValueError(f"action.hand_primitive must have shape (2,), got {hand_primitive.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("observation.state contains non-finite values")
    if not np.all(np.isfinite(action)):
        raise ValueError("action.wbc contains non-finite values")
    if not np.all(np.isfinite(token)):
        raise ValueError("action.motion_token contains non-finite values")
    if not np.all(np.isfinite(hand_primitive)):
        raise ValueError("action.hand_primitive contains non-finite values")


def _completed_render_jobs(jobs: Sequence[RenderJob], ego_frame_root: Path) -> tuple[list[RenderJob], int]:
    completed = []
    missing = 0
    for job in jobs:
        try:
            find_ego_frame_source(ego_frame_root, job.motion_key)
        except EgoFrameError:
            missing += 1
        else:
            completed.append(job)
    return completed, missing


def verify_rendered_jobs(
    jobs: Sequence[RenderJob],
    ego_frame_root: Path,
    token_label_root: Optional[Path] = None,
    task: str = DEFAULT_TASK,
    allow_empty_token_labels: bool = False,
    max_episodes: int = 5,
    require_all_rendered: bool = False,
    max_errors: int = 5,
) -> VerificationResult:
    completed, missing = _completed_render_jobs(jobs, ego_frame_root)
    result = VerificationResult(
        planned=len(jobs),
        completed_sources=len(completed),
        missing_sources=missing,
        verified=0,
    )

    if require_all_rendered and missing:
        result.errors.append("not all planned episodes have rendered ego sources")
        return result
    if not completed:
        result.errors.append("no rendered ego sources were found")
        return result
    if token_label_root is None and not allow_empty_token_labels:
        result.errors.append("token labels are required unless --allow-empty-token-labels is set")
        return result

    jobs_to_verify = completed if max_episodes == 0 else completed[:max_episodes]
    exporter = InMemoryEpisodeExporter()
    for job in jobs_to_verify:
        try:
            episode = load_motion_episode(job.robot_path)
            if token_label_root is not None:
                tokens = load_motion_tokens(
                    token_label_root,
                    job.motion_key,
                    expected_frames=episode.num_frames,
                )
                episode = attach_motion_tokens(episode, tokens)

            frame_source = find_ego_frame_source(ego_frame_root, job.motion_key)
            frames = load_ego_frames(frame_source, expected_frames=episode.num_frames)
            write_episode_to_exporter(exporter, episode, frames, task=task)
        except (EgoFrameError, TokenLabelError, ValueError, KeyError) as exc:
            result.errors.append(f"{job.motion_key}: {exc}")
            if len(result.errors) >= max_errors:
                break
        else:
            result.verified += 1

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that rendered GRAIL ego-view episodes can be converted into "
            "SONIC/GR00T LeRobot VLA rows."
        )
    )
    parser.add_argument("--motion-lib", required=True, help="GRAIL motion library root with robot/*.pkl")
    parser.add_argument("--ego-frame-root", required=True, help="Directory containing rendered ego mp4/npy frames")
    parser.add_argument("--token-labels", help="Optional directory/file with SONIC teacher motion-token labels")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="Number of completed episodes to decode and verify; use 0 for all completed episodes",
    )
    parser.add_argument(
        "--allow-empty-token-labels",
        action="store_true",
        help="Allow zero motion-token labels while verifying video/proprio conversion only",
    )
    parser.add_argument(
        "--require-all-rendered",
        action="store_true",
        help="Fail unless every planned motion has a matching rendered ego source",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Maximum per-episode conversion errors to report before stopping",
    )
    return parser


def _print_result(result: VerificationResult) -> None:
    print(f"Planned episodes: {result.planned}")
    print(f"Completed ego sources: {result.completed_sources}")
    print(f"Missing ego sources: {result.missing_sources}")
    print(f"Verified convertible episodes: {result.verified}")
    for error in result.errors:
        print(f"ERROR: {error}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be >= 0")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")

    jobs = build_render_jobs(args.motion_lib, args.ego_frame_root, args.shard_index, args.num_shards)
    try:
        result = verify_rendered_jobs(
            jobs,
            ego_frame_root=Path(args.ego_frame_root),
            token_label_root=Path(args.token_labels) if args.token_labels else None,
            task=args.task,
            allow_empty_token_labels=args.allow_empty_token_labels,
            max_episodes=args.max_episodes,
            require_all_rendered=args.require_all_rendered,
            max_errors=args.max_errors,
        )
    except (EgoFrameError, TokenLabelError, ValueError, KeyError) as exc:
        result = VerificationResult(
            planned=len(jobs),
            completed_sources=0,
            missing_sources=0,
            verified=0,
            errors=[str(exc)],
        )

    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
