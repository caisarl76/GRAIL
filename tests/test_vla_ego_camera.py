from __future__ import annotations

import numpy as np
import pytest

import grail.vla.ego_camera as ego_camera
from grail.vla.ego_camera import (
    EgoCameraSpec,
    compute_ego_camera_view,
)
from grail.vla.episode import VLAFrame


def _frame(root_position=(0.0, 0.0, 0.0), root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return VLAFrame(
        frame_index=4,
        timestamp=0.16,
        joint_position=np.zeros(2),
        root_position=np.asarray(root_position, dtype=np.float64),
        root_quaternion_wxyz=np.asarray(root_quaternion_wxyz, dtype=np.float64),
        object_position=np.zeros(3),
        object_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_compute_ego_camera_view_uses_local_offsets_for_identity_root_pose():
    spec = EgoCameraSpec(
        position_offset=(0.25, 0.0, 0.55),
        target_offset=(0.846, 0.0, -0.301),
        up_axis=(0.0, 0.0, 1.0),
    )

    view = compute_ego_camera_view(_frame(root_position=(1.0, 2.0, 0.75)), spec)

    assert view.frame_index == 4
    np.testing.assert_allclose(view.eye, [1.25, 2.0, 1.30])
    np.testing.assert_allclose(view.target, [1.846, 2.0, 0.449])
    np.testing.assert_allclose(view.up, [0.0, 0.0, 1.0])


def test_default_ego_camera_view_points_down_into_manipulation_workspace():
    view = compute_ego_camera_view(_frame())
    ray = view.target - view.eye
    pitch_degrees = np.degrees(np.arctan2(ray[2], np.linalg.norm(ray[:2])))

    assert pitch_degrees == pytest.approx(-55.0, abs=0.02)


def test_default_oak_d_color_intrinsics_match_official_horizontal_fov():
    assert hasattr(ego_camera, "DEFAULT_OAK_D_COLOR_FOCAL_LENGTH")
    focal_length = ego_camera.focal_length_from_horizontal_fov(
        ego_camera.DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
        ego_camera.DEFAULT_OAK_D_COLOR_HFOV_DEG,
    )

    assert ego_camera.DEFAULT_OAK_D_COLOR_HFOV_DEG == 69.0
    assert ego_camera.DEFAULT_OAK_D_COLOR_VFOV_DEG == 55.0
    assert ego_camera.DEFAULT_OAK_D_COLOR_FOCAL_LENGTH == pytest.approx(focal_length)
    assert ego_camera.vertical_fov_from_pinhole(
        width=640,
        height=480,
        focal_length=ego_camera.DEFAULT_OAK_D_COLOR_FOCAL_LENGTH,
        horizontal_aperture=ego_camera.DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE,
    ) == pytest.approx(ego_camera.DEFAULT_OAK_D_COLOR_VFOV_DEG, abs=0.6)


def test_compute_ego_camera_view_rotates_offsets_by_root_quaternion():
    yaw_90_wxyz = (
        np.cos(np.pi / 4.0),
        0.0,
        0.0,
        np.sin(np.pi / 4.0),
    )
    spec = EgoCameraSpec(
        position_offset=(1.0, 0.0, 0.0),
        target_offset=(2.0, 0.0, 0.0),
        up_axis=(0.0, 0.0, 1.0),
    )

    view = compute_ego_camera_view(
        _frame(root_position=(1.0, 2.0, 0.0), root_quaternion_wxyz=yaw_90_wxyz),
        spec,
    )

    np.testing.assert_allclose(view.eye, [1.0, 3.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(view.target, [1.0, 4.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(view.up, [0.0, 0.0, 1.0], atol=1e-7)


def test_compute_ego_camera_view_rejects_degenerate_view_direction():
    spec = EgoCameraSpec(position_offset=(0.0, 0.0, 0.0), target_offset=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="target_offset"):
        compute_ego_camera_view(_frame(), spec)
