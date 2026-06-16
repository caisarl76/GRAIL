from pathlib import Path

import joblib
import numpy as np
import pytest

from grail.vla.motion_library import MotionLibraryError, load_motion_episode


def _write_motion(root: Path, key: str = "pickup_table__can__000") -> Path:
    for subdir in ["robot", "objects", "meta", "object_usd"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    robot = {
        key: {
            "dof": np.zeros((3, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((3, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
            "fps": 25.0,
            "hand_dof_pos": np.ones((3, 14), dtype=np.float32),
            "hand_action_left": np.zeros(3, dtype=np.float32),
            "hand_action_right": np.zeros(3, dtype=np.float32),
            "smpl_joints": np.zeros((3, 24, 3), dtype=np.float32),
            "pose_aa": np.zeros((3, 30, 3), dtype=np.float32),
        }
    }
    objects = {
        key: {
            "root_pos": np.zeros((3, 1, 3), dtype=np.float32),
            "root_quat": np.tile(
                np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), (3, 1, 1)
            ),
            "fps": 25.0,
        }
    }

    joblib.dump(robot, root / "robot" / f"{key}.pkl")
    joblib.dump(objects, root / "objects" / f"{key}.pkl")
    joblib.dump({"object_name": "can"}, root / "meta" / f"{key}.pkl")
    (root / "object_usd" / f"{key}.usd").write_text("#usda 1.0\n", encoding="utf-8")
    return root / "robot" / f"{key}.pkl"


def test_load_motion_episode_shapes(tmp_path):
    motion_path = _write_motion(tmp_path)

    episode = load_motion_episode(motion_path)

    assert episode.motion_key == "pickup_table__can__000"
    assert episode.fps == 25.0
    assert episode.num_frames == 3
    assert episode.frames[0].joint_position.shape == (43,)
    assert episode.frames[0].object_position.shape == (3,)
    assert episode.frames[0].object_quaternion_wxyz.shape == (4,)
    assert episode.object_usd_path.name == "pickup_table__can__000.usd"


def test_missing_robot_dir_raises(tmp_path):
    with pytest.raises(MotionLibraryError, match="robot directory"):
        load_motion_episode(tmp_path / "robot" / "missing.pkl")
