from __future__ import annotations

import numpy as np

from grail.vla.ego_camera import rotation_matrix_from_quaternion_wxyz
from grail.vla.episode import VLAFrame

DEFAULT_TASK = "perform the demonstrated manipulation task"
MOTION_TOKEN_DIM = 64
HAND_PRIMITIVE_DIM = 2
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _fixed_vector(value, length: int, dtype, name: str, truncate: bool = False) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=dtype)
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape == (length,):
        return array
    if truncate and array.size >= length:
        return array[:length]
    raise ValueError(f"{name} must flatten to ({length},), got {array.shape}")


def _root_projected_gravity(root_quaternion_wxyz: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_quaternion_wxyz(root_quaternion_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return rotation.T @ gravity_world


def _quat_to_rot6d(quaternion_wxyz: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
    return rotation[:, :2].T.reshape(6).astype(np.float32)


def _hand_joints(joint_position: np.ndarray, start: int) -> np.ndarray:
    if joint_position.shape[0] >= start + 7:
        return joint_position[start : start + 7].astype(np.float32)
    return np.zeros(7, dtype=np.float32)


def frame_to_sonic_vla_dict(frame: VLAFrame, task: str = DEFAULT_TASK) -> dict:
    """Convert one neutral VLA frame into Sonic VLA feature keys."""

    token = frame.motion_token
    if token is None:
        token = np.zeros(MOTION_TOKEN_DIM, dtype=np.float64)
    token = np.asarray(token, dtype=np.float64)
    if token.shape != (MOTION_TOKEN_DIM,):
        raise ValueError(f"motion_token must have shape ({MOTION_TOKEN_DIM},), got {token.shape}")
    if not np.all(np.isfinite(token)):
        raise ValueError("motion_token must contain only finite values")

    hand_primitive = frame.hand_primitive
    if hand_primitive is None:
        hand_primitive = np.zeros(HAND_PRIMITIVE_DIM, dtype=np.float64)
    hand_primitive = np.asarray(hand_primitive, dtype=np.float64)
    if hand_primitive.shape != (HAND_PRIMITIVE_DIM,):
        raise ValueError(
            f"hand_primitive must have shape ({HAND_PRIMITIVE_DIM},), got {hand_primitive.shape}"
        )
    if not np.all(np.isfinite(hand_primitive)):
        raise ValueError("hand_primitive must contain only finite values")

    joint_position = np.asarray(frame.joint_position, dtype=np.float64)
    root_quaternion = np.asarray(frame.root_quaternion_wxyz, dtype=np.float64)
    smpl_joints = _fixed_vector(frame.smpl_joints, 72, np.float32, "smpl_joints")
    smpl_pose = _fixed_vector(frame.smpl_pose, 63, np.float32, "smpl_pose", truncate=True)
    return {
        "timestamp": np.array([frame.timestamp], dtype=np.float32),
        "task": task,
        "observation.state": joint_position,
        "observation.eef_state": np.zeros(14, dtype=np.float64),
        "action.wbc": joint_position.copy(),
        "observation.root_orientation": root_quaternion,
        "observation.projected_gravity": _root_projected_gravity(root_quaternion),
        "observation.cpp_rotation_offset": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "observation.init_base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "teleop.delta_heading": np.zeros(1, dtype=np.float64),
        "action.motion_token": token,
        "action.hand_primitive": hand_primitive,
        "teleop.smpl_joints": smpl_joints,
        "teleop.smpl_pose": smpl_pose,
        "teleop.body_quat_w": root_quaternion.astype(np.float32),
        "teleop.target_body_orientation": _quat_to_rot6d(root_quaternion),
        "teleop.left_hand_joints": _hand_joints(joint_position, 29),
        "teleop.right_hand_joints": _hand_joints(joint_position, 36),
        "teleop.smpl_frame_index": np.array([frame.frame_index], dtype=np.int64),
        "teleop.left_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.right_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.stream_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_movement": np.zeros(3, dtype=np.float32),
        "teleop.planner_facing": np.zeros(3, dtype=np.float32),
        "teleop.planner_speed": np.zeros(1, dtype=np.float32),
        "teleop.planner_height": np.zeros(1, dtype=np.float32),
        "teleop.vr_3pt_position": np.zeros(9, dtype=np.float32),
        "teleop.vr_3pt_orientation": np.tile(IDENTITY_ROT6D, 3),
    }
