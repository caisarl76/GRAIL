#!/usr/bin/env python3
"""Plan and run ego-view rendering for GRAIL motion libraries.

The local, testable part of this module builds deterministic render plans and
per-motion manifests. The Isaac Sim render loop is intentionally isolated so
importing this module does not require an Isaac/RTX environment.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from grail.visualization.batch_render_replay import (
    build_render_plan as build_replay_render_plan,
)
from grail.visualization.batch_render_replay import render_all as render_replay_all
from grail.visualization.prepare_vis_shard import convert_motion_lib_to_trajectories
from grail.vla.ego_camera import (
    DEFAULT_EGO_CAMERA_POSITION_OFFSET,
    DEFAULT_EGO_CAMERA_TARGET_OFFSET,
    DEFAULT_EGO_CAMERA_UP_AXIS,
    DEFAULT_OAK_D_COLOR_DFOV_DEG,
    DEFAULT_OAK_D_COLOR_FOCAL_LENGTH,
    DEFAULT_OAK_D_COLOR_HFOV_DEG,
    DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
    DEFAULT_OAK_D_COLOR_MODEL,
    DEFAULT_OAK_D_COLOR_VFOV_DEG,
    EgoCameraSpec,
    vertical_fov_from_pinhole,
)
from grail.vla.motion_library import load_motion_episode
from grail.vla.render_jobs import RenderJob, build_render_jobs


PathLike = Union[str, Path]


@dataclass(frozen=True)
class EgoRenderConfig:
    width: int = 640
    height: int = 480
    position_offset: Tuple[float, float, float] = DEFAULT_EGO_CAMERA_POSITION_OFFSET
    target_offset: Tuple[float, float, float] = DEFAULT_EGO_CAMERA_TARGET_OFFSET
    up_axis: Tuple[float, float, float] = DEFAULT_EGO_CAMERA_UP_AXIS

    @property
    def camera_spec(self) -> EgoCameraSpec:
        return EgoCameraSpec(
            position_offset=self.position_offset,
            target_offset=self.target_offset,
            up_axis=self.up_axis,
        )


def parse_resolution(resolution: str) -> Tuple[int, int]:
    try:
        width_text, height_text = resolution.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must use WIDTHxHEIGHT format, got {resolution!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution width and height must be positive")
    return width, height


def build_ego_render_plan(
    motion_lib: PathLike,
    output_dir: PathLike,
    config: EgoRenderConfig = EgoRenderConfig(),
    shard_index: int = 0,
    num_shards: int = 1,
    max_motions: int = 0,
    skip_existing: bool = False,
) -> list[RenderJob]:
    """Build deterministic ego-render jobs from a released GRAIL motion library."""

    del config  # Reserved for future camera-dependent planning; keep API stable.
    jobs = build_render_jobs(motion_lib, output_dir, shard_index=shard_index, num_shards=num_shards)
    if skip_existing:
        jobs = [job for job in jobs if not job.video_path.exists()]
    if max_motions > 0:
        jobs = jobs[:max_motions]
    return jobs


def manifest_for_job(job: RenderJob, config: EgoRenderConfig = EgoRenderConfig()) -> dict:
    episode = load_motion_episode(job.robot_path)
    rendered_vfov_deg = vertical_fov_from_pinhole(
        width=config.width,
        height=config.height,
        focal_length=DEFAULT_OAK_D_COLOR_FOCAL_LENGTH,
        horizontal_aperture=DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
    )
    return {
        "motion_key": job.motion_key,
        "status": "planned",
        "fps": float(episode.fps),
        "num_frames": episode.num_frames,
        "robot_path": str(job.robot_path),
        "object_path": str(episode.object_path) if episode.object_path else None,
        "object_usd_path": str(episode.object_usd_path) if episode.object_usd_path else None,
        "video_path": str(job.video_path),
        "camera": {
            "mode": "root_relative_ego",
            "resolution": [config.height, config.width],
            "width": config.width,
            "height": config.height,
            "position_offset": list(config.position_offset),
            "target_offset": list(config.target_offset),
            "up_axis": list(config.up_axis),
            "intrinsics": {
                "model": DEFAULT_OAK_D_COLOR_MODEL,
                "dfov_deg": DEFAULT_OAK_D_COLOR_DFOV_DEG,
                "hfov_deg": DEFAULT_OAK_D_COLOR_HFOV_DEG,
                "vfov_deg": DEFAULT_OAK_D_COLOR_VFOV_DEG,
                "rendered_vfov_deg": rendered_vfov_deg,
                "horizontal_aperture": DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
                "focal_length": DEFAULT_OAK_D_COLOR_FOCAL_LENGTH,
            },
        },
        "training_observations": [
            "observation.images.ego_view",
            "robot_proprioception",
        ],
    }


def write_planned_manifests(
    jobs: Sequence[RenderJob],
    config: EgoRenderConfig = EgoRenderConfig(),
) -> int:
    for job in jobs:
        job.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with job.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_for_job(job, config), f, indent=2)
            f.write("\n")
    return len(jobs)


def write_motion_key_list(jobs: Sequence[RenderJob], output_path: PathLike) -> int:
    """Write planned motion keys in render order, one per line."""

    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(f"{job.motion_key}\n")
    return len(jobs)


def render_all(
    jobs: Sequence[RenderJob],
    config: EgoRenderConfig = EgoRenderConfig(),
    headless: bool = True,
    quat_convention: str = "xyzw",
) -> None:
    """Render selected motion jobs from a root-relative robot ego camera."""

    if not jobs:
        return

    motion_roots = {job.robot_path.parent.parent.resolve() for job in jobs}
    if len(motion_roots) != 1:
        raise ValueError("All ego render jobs must come from the same motion library")
    output_dirs = {job.video_path.parent.resolve() for job in jobs}
    if len(output_dirs) != 1:
        raise ValueError("All ego render jobs must write to the same output directory")

    motion_root = next(iter(motion_roots))
    output_dir = next(iter(output_dirs))
    motion_filter = {job.motion_key for job in jobs}
    shard_dir = Path(tempfile.mkdtemp(prefix=f"ego_vis_shard_{motion_root.name}_"))

    print(f"Preparing ego replay shard: {shard_dir}", flush=True)
    convert_motion_lib_to_trajectories(
        data_dir=str(motion_root),
        shard_dir=str(shard_dir),
        max_motions=0,
        motion_filter=motion_filter,
        quat_convention=quat_convention,
    )
    plan, stats, total_keys, successful_keys = build_replay_render_plan(
        str(shard_dir),
        str(motion_root / "object_usd"),
        str(output_dir),
        skip_existing=False,
    )
    print(
        "Ego replay plan: "
        f"{len(plan)} renderable / {successful_keys} successful / {total_keys} total "
        f"(skipped={stats.get('skipped', 0)} missing_traj={stats.get('missing_traj', 0)} "
        f"missing_usd={stats.get('missing_usd', 0)})",
        flush=True,
    )
    render_replay_all(
        plan,
        resolution=(config.width, config.height),
        headless=headless,
        start_frame_skip=0,
        ego_camera_spec=config.camera_spec,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch render GRAIL ego-view videos")
    parser.add_argument("--motion_lib", "--motion-lib", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument(
        "--camera_position_offset",
        "--camera-position-offset",
        type=float,
        nargs=3,
        default=list(DEFAULT_EGO_CAMERA_POSITION_OFFSET),
    )
    parser.add_argument(
        "--camera_target_offset",
        "--camera-target-offset",
        type=float,
        nargs=3,
        default=list(DEFAULT_EGO_CAMERA_TARGET_OFFSET),
    )
    parser.add_argument(
        "--camera_up_axis",
        "--camera-up-axis",
        type=float,
        nargs=3,
        default=list(DEFAULT_EGO_CAMERA_UP_AXIS),
    )
    parser.add_argument("--shard_index", "--shard-index", type=int, default=0)
    parser.add_argument("--num_shards", "--num-shards", type=int, default=1)
    parser.add_argument("--max_motions", "--max-motions", type=int, default=0)
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument("--write_manifests", "--write-manifests", action="store_true")
    parser.add_argument(
        "--write_motion_keys",
        "--write-motion-keys",
        default=None,
        help="Write selected motion keys to this text file before rendering.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--quat_convention",
        "--quat-convention",
        choices=("auto", "wxyz", "xyzw"),
        default="xyzw",
        help="Input root_rot convention. Public GRAIL/HF data uses xyzw; retarget outputs often use wxyz.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> EgoRenderConfig:
    width, height = parse_resolution(args.resolution)
    return EgoRenderConfig(
        width=width,
        height=height,
        position_offset=tuple(args.camera_position_offset),
        target_offset=tuple(args.camera_target_offset),
        up_axis=tuple(args.camera_up_axis),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    jobs = build_ego_render_plan(
        args.motion_lib,
        args.output_dir,
        config=config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_motions=args.max_motions,
        skip_existing=args.skip_existing,
    )

    print(f"Motion library: {args.motion_lib}")
    print(f"Ego render plan: {len(jobs)} motions")
    print(f"Camera: {config.width}x{config.height} {asdict(config)}")
    for job in jobs[:10]:
        print(f"  {job.motion_key} -> {job.video_path}")
    if len(jobs) > 10:
        print(f"  ... and {len(jobs) - 10} more")

    if args.write_manifests:
        written = write_planned_manifests(jobs, config)
        print(f"Wrote {written} manifest(s)")

    if args.write_motion_keys:
        written = write_motion_key_list(jobs, args.write_motion_keys)
        print(f"Wrote {written} motion key(s): {args.write_motion_keys}")

    if args.dry_run:
        return 0
    if not jobs:
        print("Nothing to render.")
        return 0

    render_all(jobs, config=config, headless=args.headless, quat_convention=args.quat_convention)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
