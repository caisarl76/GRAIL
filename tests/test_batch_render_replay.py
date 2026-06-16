from __future__ import annotations

import numpy as np
import pytest

import grail.visualization.batch_render_replay as batch_render_replay
import grail.vla.ego_camera as ego_camera
from grail.visualization.batch_render_replay import (
    _ego_camera_eye_target,
    _prepare_rgb_frame,
)
from grail.vla.ego_camera import EgoCameraSpec


def test_prepare_rgb_frame_rejects_flat_renderer_output_with_actionable_error():
    with pytest.raises(RuntimeError) as exc_info:
        _prepare_rgb_frame(np.array([0], dtype=np.uint8), "pickup_table__can__000")

    message = str(exc_info.value)
    assert "pickup_table__can__000" in message
    assert "invalid shape (1,)" in message
    assert "libXt.so.6" in message
    assert "libGLU.so.1" in message


def test_prepare_rgb_frame_accepts_batched_rgba_and_strips_alpha():
    rgba = np.zeros((1, 4, 5, 4), dtype=np.uint8)
    rgba[..., 0] = 255

    rgb = _prepare_rgb_frame(rgba, "pickup_table__can__000")

    assert rgb.shape == (4, 5, 3)
    assert np.all(rgb[..., 0] == 255)


def test_ego_camera_eye_target_uses_transformed_robot_root_pose():
    spec = EgoCameraSpec(
        position_offset=(0.25, 0.0, 0.55),
        target_offset=(0.846, 0.0, -0.301),
    )

    eye, target = _ego_camera_eye_target(
        root_position=np.array([1.0, 2.0, 0.75], dtype=np.float32),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ego_camera_spec=spec,
        frame_index=7,
    )

    np.testing.assert_allclose(eye, [1.25, 2.0, 1.30])
    np.testing.assert_allclose(target, [1.846, 2.0, 0.449])


def test_ego_camera_pinhole_kwargs_use_oak_d_color_intrinsics():
    assert hasattr(batch_render_replay, "_ego_camera_pinhole_kwargs")
    kwargs = batch_render_replay._ego_camera_pinhole_kwargs()

    assert kwargs["focal_length"] == pytest.approx(ego_camera.DEFAULT_OAK_D_COLOR_FOCAL_LENGTH)
    assert kwargs["horizontal_aperture"] == ego_camera.DEFAULT_OAK_D_COLOR_HORIZONTAL_APERTURE
    assert kwargs["focus_distance"] == 100.0
    assert kwargs["clipping_range"] == (0.1, 500.0)
