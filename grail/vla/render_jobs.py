from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RenderJob:
    motion_key: str
    robot_path: Path
    video_path: Path
    manifest_path: Path


def _missing_robot_dir_message(motion_lib: Path) -> str:
    robot_dir = motion_lib / "robot"
    return (
        "Expected a GRAIL motion library root with robot/*.pkl, "
        f"but missing: {robot_dir}\n"
        f"Passed motion_lib: {motion_lib}\n"
        "Expected layout:\n"
        "  <motion_lib>/robot/*.pkl\n"
        "  <motion_lib>/objects/*.pkl\n"
        "  <motion_lib>/object_usd/*.usd or *.usda\n"
        "In the container, locate candidate roots with:\n"
        "  for root in /workspace/grail /data; do "
        '[ -d "$root" ] && find "$root" -path \'*/robot/*.pkl\' -print; '
        "done | head"
    )


def build_render_jobs(
    motion_lib: PathLike,
    output_dir: PathLike,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[RenderJob]:
    """Build deterministic, modulo-sharded ego render jobs."""

    motion_lib = Path(motion_lib)
    output_dir = Path(output_dir)
    robot_dir = motion_lib / "robot"
    if not robot_dir.is_dir():
        raise FileNotFoundError(_missing_robot_dir_message(motion_lib))
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    robot_paths = sorted(robot_dir.glob("*.pkl"))
    selected = [path for idx, path in enumerate(robot_paths) if idx % num_shards == shard_index]
    return [
        RenderJob(
            motion_key=path.stem,
            robot_path=path,
            video_path=output_dir / f"{path.stem}.mp4",
            manifest_path=output_dir / f"{path.stem}.json",
        )
        for path in selected
    ]
