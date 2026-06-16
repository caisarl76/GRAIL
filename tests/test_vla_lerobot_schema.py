import numpy as np

from grail.vla.episode import VLAFrame
from grail.vla.lerobot_schema import DEFAULT_TASK, frame_to_sonic_vla_dict


def test_frame_to_sonic_vla_dict_defaults():
    frame = VLAFrame(
        frame_index=0,
        timestamp=0.0,
        joint_position=np.ones(43, dtype=np.float32),
        root_position=np.zeros(3, dtype=np.float32),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        object_position=np.zeros(3, dtype=np.float32),
        object_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    row = frame_to_sonic_vla_dict(frame)

    assert row["task"] == DEFAULT_TASK
    assert row["timestamp"].shape == (1,)
    assert row["timestamp"].dtype == np.float32
    assert np.allclose(row["timestamp"], [0.0])
    assert row["observation.state"].shape == (43,)
    assert row["action.wbc"].shape == (43,)
    assert row["action.motion_token"].shape == (64,)
    assert row["action.hand_primitive"].shape == (2,)
    assert np.allclose(row["action.motion_token"], 0.0)
    assert np.allclose(row["action.hand_primitive"], 0.0)


def test_frame_to_sonic_vla_dict_contains_required_sonic_vla_features():
    frame = VLAFrame(
        frame_index=3,
        timestamp=0.12,
        joint_position=np.arange(43, dtype=np.float32),
        root_position=np.zeros(3, dtype=np.float32),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        object_position=np.zeros(3, dtype=np.float32),
        object_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        smpl_joints=np.ones((24, 3), dtype=np.float32),
        smpl_pose=np.ones((30, 3), dtype=np.float32),
    )

    row = frame_to_sonic_vla_dict(frame)

    expected_shapes = {
        "observation.state": (43,),
        "observation.eef_state": (14,),
        "action.wbc": (43,),
        "observation.root_orientation": (4,),
        "observation.projected_gravity": (3,),
        "observation.cpp_rotation_offset": (4,),
        "observation.init_base_quat": (4,),
        "teleop.delta_heading": (1,),
        "action.motion_token": (64,),
        "action.hand_primitive": (2,),
        "teleop.smpl_joints": (72,),
        "teleop.smpl_pose": (63,),
        "teleop.body_quat_w": (4,),
        "teleop.target_body_orientation": (6,),
        "teleop.left_hand_joints": (7,),
        "teleop.right_hand_joints": (7,),
        "teleop.smpl_frame_index": (1,),
        "teleop.left_wrist_joints": (3,),
        "teleop.right_wrist_joints": (3,),
        "teleop.stream_mode": (1,),
        "teleop.planner_mode": (1,),
        "teleop.planner_movement": (3,),
        "teleop.planner_facing": (3,),
        "teleop.planner_speed": (1,),
        "teleop.planner_height": (1,),
        "teleop.vr_3pt_position": (9,),
        "teleop.vr_3pt_orientation": (18,),
    }
    for key, shape in expected_shapes.items():
        assert row[key].shape == shape

    assert row["teleop.smpl_frame_index"].dtype == np.int64
    assert row["teleop.stream_mode"].dtype == np.int32
    assert np.allclose(row["teleop.smpl_joints"], 1.0)
    assert np.allclose(row["teleop.smpl_pose"], 1.0)
    assert np.allclose(row["teleop.left_hand_joints"], np.arange(29, 36))
    assert np.allclose(row["teleop.right_hand_joints"], np.arange(36, 43))


def test_frame_to_sonic_vla_dict_does_not_export_privileged_object_pose():
    frame = VLAFrame(
        frame_index=0,
        timestamp=0.0,
        joint_position=np.zeros(43, dtype=np.float32),
        root_position=np.zeros(3, dtype=np.float32),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        object_position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        object_quaternion_wxyz=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )

    row = frame_to_sonic_vla_dict(frame)

    forbidden_fragments = ("object", "obj")
    assert all(
        not any(fragment in key.lower() for fragment in forbidden_fragments)
        for key in row
    )
