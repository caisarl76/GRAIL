import torch


def test_static_pose_aa_body_is_filled_from_raw_robot_dof():
    from gear_sonic.utils.motion_lib.motion_lib_base import (
        _apply_raw_dof_to_static_pose_aa,
    )

    pose_aa = torch.zeros(2, 5, 3)
    pose_aa[:, 0, 2] = torch.tensor([0.25, -0.5])
    raw_dof = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4, 9.0],
            [-0.5, 0.6, -0.7, 0.8, 8.0],
        ]
    )
    dof_axis = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    filled, did_fill = _apply_raw_dof_to_static_pose_aa(pose_aa, raw_dof, dof_axis)

    assert did_fill
    torch.testing.assert_close(filled[:, 0], pose_aa[:, 0])
    torch.testing.assert_close(filled[:, 1:5], raw_dof[:, :4, None] * dof_axis)


def test_nonstatic_pose_aa_body_is_not_overwritten_by_raw_robot_dof():
    from gear_sonic.utils.motion_lib.motion_lib_base import (
        _apply_raw_dof_to_static_pose_aa,
    )

    pose_aa = torch.zeros(1, 3, 3)
    pose_aa[:, 1, 0] = 0.25
    raw_dof = torch.tensor([[1.0, 2.0]])
    dof_axis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    filled, did_fill = _apply_raw_dof_to_static_pose_aa(pose_aa, raw_dof, dof_axis)

    assert not did_fill
    torch.testing.assert_close(filled, pose_aa)
