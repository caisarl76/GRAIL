from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VLAFrame:
    frame_index: int
    timestamp: float
    joint_position: np.ndarray
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    object_position: np.ndarray
    object_quaternion_wxyz: np.ndarray
    smpl_joints: Optional[np.ndarray] = None
    smpl_pose: Optional[np.ndarray] = None
    motion_token: Optional[np.ndarray] = None
    hand_primitive: Optional[np.ndarray] = None


@dataclass(frozen=True)
class VLAEpisode:
    motion_key: str
    fps: float
    frames: list[VLAFrame]
    robot_path: Optional[Path]
    object_path: Optional[Path]
    object_usd_path: Optional[Path]
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)
