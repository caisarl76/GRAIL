from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np

from grail.vla.episode import VLAEpisode, VLAFrame


class MotionLibraryError(ValueError):
    """Raised when a GRAIL motion-library file cannot be converted."""


PathLike = Union[str, Path]


def _single_entry(data: dict, motion_key: str, path: Path) -> dict:
    if not isinstance(data, dict):
        raise MotionLibraryError(f"Expected dictionary in {path}, got {type(data).__name__}")
    if motion_key in data:
        return data[motion_key]
    if len(data) == 1:
        return next(iter(data.values()))
    raise MotionLibraryError(f"Cannot identify motion entry for {motion_key} in {path}")


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise MotionLibraryError(f"Quaternion must end with 4 values, got shape {quat.shape}")
    return quat[..., [3, 0, 1, 2]]


def _as_frame_object_array(array: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise MotionLibraryError(
            f"{name} must have shape (T, {width}) or (T, 1, {width}); got {array.shape}"
        )
    return array


def _require_equal_length(name: str, array: np.ndarray, num_frames: int) -> None:
    if len(array) != num_frames:
        raise MotionLibraryError(f"{name} has {len(array)} frames, expected {num_frames}")


def _motion_root_from_robot_path(robot_path: Path) -> Path:
    if robot_path.parent.name != "robot":
        raise MotionLibraryError(f"Expected path under robot directory, got {robot_path}")
    root = robot_path.parent.parent
    if not (root / "robot").is_dir():
        raise MotionLibraryError(f"Missing robot directory under {root}")
    return root


def _get_optional_frame_array(entry: dict, key: str, frame_index: int) -> Optional[np.ndarray]:
    if key not in entry:
        return None
    return np.asarray(entry[key][frame_index], dtype=np.float32)


def load_motion_episode(robot_path: PathLike, quat_convention: str = "xyzw") -> VLAEpisode:
    """Load one released-style GRAIL motion into schema-neutral frame records.

    Released GRAIL motion libraries store body DOFs in ``robot/<key>.pkl`` and
    hand DOFs separately as ``hand_dof_pos``. This loader combines them into a
    43-DOF joint vector so downstream LeRobot export can use a single state/action
    field before the IsaacSim ego renderer is connected.
    """

    robot_path = Path(robot_path)
    root = _motion_root_from_robot_path(robot_path)
    if not robot_path.is_file():
        raise MotionLibraryError(f"Robot trajectory file not found: {robot_path}")

    motion_key = robot_path.stem
    robot_data = _single_entry(joblib.load(robot_path), motion_key, robot_path)
    object_path = root / "objects" / f"{motion_key}.pkl"
    object_data = None
    if object_path.is_file():
        object_data = _single_entry(joblib.load(object_path), motion_key, object_path)

    meta_path = root / "meta" / f"{motion_key}.pkl"
    meta = joblib.load(meta_path) if meta_path.is_file() else {}
    object_usd_path = root / "object_usd" / f"{motion_key}.usd"

    dof = np.asarray(robot_data["dof"], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] not in (29, 43):
        raise MotionLibraryError(f"robot dof must have shape (T, 29) or (T, 43); got {dof.shape}")
    num_frames = len(dof)

    if dof.shape[1] == 29:
        hand = np.asarray(robot_data.get("hand_dof_pos", np.zeros((num_frames, 14))), dtype=np.float32)
        if hand.shape != (num_frames, 14):
            raise MotionLibraryError(f"hand_dof_pos must have shape ({num_frames}, 14); got {hand.shape}")
        joint_position = np.concatenate([dof, hand], axis=1)
    else:
        joint_position = dof

    root_position = np.asarray(robot_data["root_trans_offset"], dtype=np.float32)
    root_quat = np.asarray(robot_data["root_rot"], dtype=np.float32)
    _require_equal_length("root_trans_offset", root_position, num_frames)
    _require_equal_length("root_rot", root_quat, num_frames)

    if quat_convention == "xyzw":
        root_quat = _xyzw_to_wxyz(root_quat)
    elif quat_convention != "wxyz":
        raise MotionLibraryError(f"quat_convention must be 'xyzw' or 'wxyz', got {quat_convention}")

    fps = float(robot_data.get("fps", 25.0))
    object_position = np.zeros((num_frames, 3), dtype=np.float32)
    object_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_frames, 1))
    if object_data is not None:
        object_position = _as_frame_object_array(object_data["root_pos"], 3, "object root_pos")
        object_quat = _xyzw_to_wxyz(_as_frame_object_array(object_data["root_quat"], 4, "object root_quat"))
        _require_equal_length("object root_pos", object_position, num_frames)
        _require_equal_length("object root_quat", object_quat, num_frames)

    frames = []
    for idx in range(num_frames):
        frames.append(
            VLAFrame(
                frame_index=idx,
                timestamp=idx / fps,
                joint_position=joint_position[idx],
                root_position=root_position[idx],
                root_quaternion_wxyz=root_quat[idx],
                object_position=object_position[idx],
                object_quaternion_wxyz=object_quat[idx],
                smpl_joints=_get_optional_frame_array(robot_data, "smpl_joints", idx),
                smpl_pose=_get_optional_frame_array(robot_data, "pose_aa", idx),
            )
        )

    return VLAEpisode(
        motion_key=motion_key,
        fps=fps,
        frames=frames,
        robot_path=robot_path,
        object_path=object_path if object_path.is_file() else None,
        object_usd_path=object_usd_path if object_usd_path.is_file() else None,
        meta=meta,
    )
