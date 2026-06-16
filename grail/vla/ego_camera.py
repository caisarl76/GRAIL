from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from grail.vla.episode import VLAEpisode, VLAFrame


Vector3 = Tuple[float, float, float]

DEFAULT_EGO_CAMERA_POSITION_OFFSET: Vector3 = (0.25, 0.0, 0.55)
DEFAULT_EGO_CAMERA_TARGET_OFFSET: Vector3 = (0.846, 0.0, -0.301)
DEFAULT_EGO_CAMERA_UP_AXIS: Vector3 = (0.0, 0.0, 1.0)

DEFAULT_OAK_D_COLOR_MODEL = "Luxonis OAK-D RGB IMX378"
DEFAULT_OAK_D_COLOR_DFOV_DEG = 81.0
DEFAULT_OAK_D_COLOR_HFOV_DEG = 69.0
DEFAULT_OAK_D_COLOR_VFOV_DEG = 55.0
DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE = 10.0


def focal_length_from_horizontal_fov(horizontal_aperture: float, horizontal_fov_deg: float) -> float:
    if horizontal_aperture <= 0.0:
        raise ValueError("horizontal_aperture must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    return horizontal_aperture / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def vertical_fov_from_pinhole(
    *,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if focal_length <= 0.0:
        raise ValueError("focal_length must be positive")
    if horizontal_aperture <= 0.0:
        raise ValueError("horizontal_aperture must be positive")
    vertical_aperture = horizontal_aperture * (float(height) / float(width))
    return math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal_length)))


DEFAULT_OAK_D_COLOR_FOCAL_LENGTH = focal_length_from_horizontal_fov(
    DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
    DEFAULT_OAK_D_COLOR_HFOV_DEG,
)


@dataclass(frozen=True)
class EgoCameraSpec:
    """Camera offsets expressed in the robot root frame."""

    position_offset: Vector3 = DEFAULT_EGO_CAMERA_POSITION_OFFSET
    target_offset: Vector3 = DEFAULT_EGO_CAMERA_TARGET_OFFSET
    up_axis: Vector3 = DEFAULT_EGO_CAMERA_UP_AXIS


@dataclass(frozen=True)
class EgoCameraView:
    frame_index: int
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray


def _as_vector3(name: str, value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector, got shape {array.shape}")
    return array


def normalize_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError(f"root quaternion must be a finite 4-vector, got shape {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("root quaternion norm must be non-zero")
    return quat / norm


def rotation_matrix_from_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_ego_camera_view(frame: VLAFrame, spec: EgoCameraSpec = EgoCameraSpec()) -> EgoCameraView:
    root_position = _as_vector3("root_position", frame.root_position)
    position_offset = _as_vector3("position_offset", spec.position_offset)
    target_offset = _as_vector3("target_offset", spec.target_offset)
    up_axis = _as_vector3("up_axis", spec.up_axis)

    local_direction = target_offset - position_offset
    if float(np.linalg.norm(local_direction)) <= 1e-12:
        raise ValueError("target_offset must differ from position_offset")

    rotation = rotation_matrix_from_quaternion_wxyz(frame.root_quaternion_wxyz)
    eye = root_position + rotation @ position_offset
    target = root_position + rotation @ target_offset
    up = rotation @ up_axis
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-12:
        raise ValueError("up_axis must be non-zero")
    up = up / up_norm

    world_direction = target - eye
    forward_norm = float(np.linalg.norm(world_direction))
    if forward_norm <= 1e-12:
        raise ValueError("camera view direction must be non-zero")
    forward = world_direction / forward_norm
    if float(np.linalg.norm(np.cross(forward, up))) <= 1e-8:
        raise ValueError("camera up_axis must not be parallel to the view direction")

    return EgoCameraView(
        frame_index=frame.frame_index,
        eye=eye,
        target=target,
        up=up,
    )


def camera_views_for_episode(
    episode: VLAEpisode,
    spec: EgoCameraSpec = EgoCameraSpec(),
) -> list[EgoCameraView]:
    return [compute_ego_camera_view(frame, spec) for frame in episode.frames]
