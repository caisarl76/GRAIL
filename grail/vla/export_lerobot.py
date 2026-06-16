from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np

from grail.vla.ego_frames import EgoFrameError, find_ego_frame_source, load_ego_frames
from grail.vla.episode import VLAEpisode
from grail.vla.lerobot_schema import DEFAULT_TASK, frame_to_sonic_vla_dict
from grail.vla.motion_library import load_motion_episode
from grail.vla.render_jobs import RenderJob
from grail.vla.token_labels import TokenLabelError, attach_motion_tokens, load_motion_tokens


def _as_image_list(image_frames: Iterable[np.ndarray], expected_frames: int) -> list[np.ndarray]:
    images = [np.asarray(image) for image in image_frames]
    if len(images) != expected_frames:
        raise ValueError(f"image frame count {len(images)} does not match episode frames {expected_frames}")
    for index, image in enumerate(images):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"image frame {index} must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            images[index] = image.astype(np.uint8)
    return images


def write_episode_to_exporter(
    exporter,
    episode: VLAEpisode,
    image_frames: Iterable[np.ndarray],
    task: str = DEFAULT_TASK,
) -> None:
    """Write one loaded GRAIL episode into a LeRobot-like exporter object.

    The exporter only needs the same two methods used by SONIC's
    ``Gr00tDataExporter``: ``add_frame(dict)`` and ``save_episode()``. This keeps
    the conversion logic testable without importing LeRobot or IsaacSim.
    """

    images = _as_image_list(image_frames, episode.num_frames)
    for frame, image in zip(episode.frames, images):
        row = frame_to_sonic_vla_dict(frame, task=task)
        row["observation.images.ego_view"] = image
        row.pop("timestamp", None)
        exporter.add_frame(row)
    exporter.save_episode()


def create_gr00t_data_exporter(
    save_root: Union[str, Path],
    fps: int,
    task: str,
    script_config: Optional[dict] = None,
    *,
    overwrite_existing: bool = False,
):
    """Create SONIC's LeRobot exporter with imports isolated to runtime.

    This function intentionally imports ``gear_sonic`` lazily because local unit
    tests and CPU-only development environments usually do not have LeRobot,
    IsaacSim, or the SONIC data-collection extras installed.
    """

    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_g1_robot_model,
        get_modality_config_sonic_vla,
    )

    robot_model = get_g1_robot_model()
    return Gr00tDataExporter.create(
        save_root=save_root,
        fps=fps,
        features=get_features_sonic_vla(robot_model),
        modality_config=get_modality_config_sonic_vla(robot_model),
        task=task,
        script_config=script_config or {},
        overwrite_existing=overwrite_existing,
    )


ExporterFactory = Callable[[Union[str, Path], int, str, Optional[dict]], object]


def _lock_path_for_output(output_dir: Union[str, Path]) -> Path:
    output_path = Path(output_dir)
    return output_path.parent / f".{output_path.name}.grail_export.lock"


@contextmanager
def _exclusive_export_lock(output_dir: Union[str, Path]):
    lock_path = _lock_path_for_output(output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another export process is already writing to {Path(output_dir)}. "
            f"Lock file exists: {lock_path}"
        ) from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(f"pid={os.getpid()}\n")
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def select_rendered_jobs(
    jobs: Sequence[RenderJob],
    ego_frame_root: Union[str, Path],
    only_rendered: bool = False,
    max_episodes: Optional[int] = None,
    require_all_rendered: bool = False,
) -> tuple[list[RenderJob], int]:
    """Select export jobs while optionally filtering to completed ego renders."""

    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be >= 0")

    selected = []
    missing = 0
    root = Path(ego_frame_root)
    for job in jobs:
        try:
            find_ego_frame_source(root, job.motion_key)
        except EgoFrameError:
            missing += 1
            if not only_rendered:
                selected.append(job)
        else:
            selected.append(job)

    if require_all_rendered and missing:
        raise FileNotFoundError(f"Missing rendered ego sources: {missing}")
    if max_episodes not in (None, 0):
        selected = selected[:max_episodes]
    return selected, missing


def export_rendered_jobs(
    jobs: Sequence[RenderJob],
    ego_frame_root: Union[str, Path],
    output_dir: Union[str, Path],
    token_label_root: Optional[Union[str, Path]] = None,
    task: str = DEFAULT_TASK,
    allow_empty_token_labels: bool = False,
    append_existing: bool = False,
    overwrite_existing: bool = False,
    exporter_factory: ExporterFactory = create_gr00t_data_exporter,
) -> int:
    """Export already-rendered ego frames and token labels into LeRobot rows.

    Isaac Sim rendering is intentionally outside this function. It consumes the
    renderer output and writes episodes through SONIC's exporter lifecycle:
    create exporter once, add frames, save one episode per motion.
    """

    if append_existing and overwrite_existing:
        raise ValueError("append_existing and overwrite_existing are mutually exclusive")

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not (append_existing or overwrite_existing):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_path}. "
            "Use --append-existing to resume/append, --overwrite-existing to replace it, "
            "or choose a fresh --output path."
        )

    with _exclusive_export_lock(output_dir):
        exporter = None
        exporter_fps = None
        exported = 0
        empty_existing_output = output_path.exists() and not any(output_path.iterdir())
        for job in jobs:
            episode = load_motion_episode(job.robot_path)
            if token_label_root is None:
                if not allow_empty_token_labels:
                    raise TokenLabelError(
                        "token labels are required unless allow_empty_token_labels is true"
                    )
            else:
                tokens = load_motion_tokens(
                    token_label_root,
                    job.motion_key,
                    expected_frames=episode.num_frames,
                )
                episode = attach_motion_tokens(episode, tokens)

            frame_source = find_ego_frame_source(ego_frame_root, job.motion_key)
            frames = load_ego_frames(frame_source, expected_frames=episode.num_frames)

            episode_fps = int(round(episode.fps))
            if exporter is None:
                if exporter_factory is create_gr00t_data_exporter:
                    exporter = create_gr00t_data_exporter(
                        output_dir,
                        episode_fps,
                        task,
                        {"source": "grail_ego_vla"},
                        overwrite_existing=overwrite_existing or empty_existing_output,
                    )
                else:
                    exporter = exporter_factory(
                        output_dir, episode_fps, task, {"source": "grail_ego_vla"}
                    )
                exporter_fps = episode_fps
            elif exporter_fps != episode_fps:
                raise ValueError(
                    f"All exported episodes must share one FPS; got {episode_fps} after {exporter_fps}"
                )

            write_episode_to_exporter(exporter, episode, frames, task=task)
            exported += 1

    return exported
