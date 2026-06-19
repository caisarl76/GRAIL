from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "scripts/eval/compare_sonic_teacher_debug.py"
    spec = importlib.util.spec_from_file_location("compare_sonic_teacher_debug", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_joint_error_summary_reports_mse_mae_and_ranking():
    module = _load_compare_module()
    rows = [
        {"joint_error": np.asarray([1.0, -2.0, 0.5])},
        {"joint_error": np.asarray([3.0, 0.0, -0.5])},
    ]

    summary = module.per_joint_error_summary(rows, joint_names=["a", "b", "c"])

    assert [row["joint_name"] for row in summary] == ["a", "b", "c"]
    assert summary[0]["mse"] == 5.0
    assert summary[0]["mae"] == 2.0
    assert summary[0]["rmse"] == np.sqrt(5.0)
    assert summary[1]["mse"] == 2.0
    assert summary[2]["mse"] == 0.25


def test_motion_file_dof_target_defaults_to_mujoco_to_isaaclab_order():
    module = _load_compare_module()
    raw_mujoco = np.arange(29, dtype=np.float64)

    reordered = module.motion_file_dof_target(raw_mujoco, motion_dof_order="mujoco")

    np.testing.assert_array_equal(
        reordered,
        raw_mujoco[np.asarray(module.G1_MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)],
    )
    np.testing.assert_array_equal(
        module.motion_file_dof_target(raw_mujoco, motion_dof_order="isaaclab"),
        raw_mujoco,
    )


def test_motion_file_dof_at_eval_frame_interpolates_raw_motion_fps():
    module = _load_compare_module()
    motion = {
        "fps": 25,
        "dof": np.stack(
            [
                np.zeros(29, dtype=np.float64),
                np.ones(29, dtype=np.float64) * 10.0,
            ]
        ),
    }

    target = module.motion_file_dof_at_eval_frame(
        motion,
        target_frame=1,
        motion_dof_order="isaaclab",
        motion_timebase="eval-time",
        eval_fps=50.0,
        motion_file_fps=None,
    )

    np.testing.assert_allclose(target, np.ones(29, dtype=np.float64) * 5.0)


def test_motion_file_array_at_eval_frame_supports_nearest_discrete_labels():
    module = _load_compare_module()
    motion = {
        "fps": 25,
        "hand_action_left": np.asarray([[-1.0], [1.0]]),
    }

    target = module.motion_file_array_at_eval_frame(
        motion,
        "hand_action_left",
        target_frame=1,
        motion_timebase="eval-time",
        eval_fps=50.0,
        motion_file_fps=None,
        interpolation="nearest",
    )

    np.testing.assert_array_equal(target, np.asarray([-1.0]))


def test_hand_comparison_accepts_column_vector_primitive_labels(tmp_path):
    module = _load_compare_module()
    motion_key = "motion_0"
    joblib = __import__("joblib")
    joblib.dump(
        {
            motion_key: {
                "fps": 25,
                "dof": np.zeros((2, 29), dtype=np.float64),
                "hand_action_left": np.asarray([[-1.0], [1.0]], dtype=np.float64),
                "hand_action_right": np.asarray([[1.0], [-1.0]], dtype=np.float64),
                "hand_dof_pos": np.zeros((2, 14), dtype=np.float64),
            }
        },
        tmp_path / f"{motion_key}.pkl",
    )
    records = [
        {
            "env_index": 0,
            "step": 0,
            "frame_index": 0,
            "motion_key": motion_key,
            "hand_primitive": [-1.0, 1.0],
            "env_action_hand": [0.0] * 14,
        }
    ]

    rows = module.hand_comparison(
        records,
        motion_dir=tmp_path,
        env_index=0,
        frame_offset=0,
        include_done=False,
        motion_timebase="eval-time",
        eval_fps=50.0,
        motion_file_fps=None,
    )

    assert len(rows["primitive"]) == 1
    assert rows["primitive"][0]["action_mse"] == 0.0
    assert rows["primitive"][0]["sign_accuracy"] == 1.0


def test_hand_comparison_reports_primitive_sign_accuracy(tmp_path):
    module = _load_compare_module()
    motion_key = "motion_0"
    joblib = __import__("joblib")
    joblib.dump(
        {
            motion_key: {
                "fps": 25,
                "dof": np.zeros((2, 29), dtype=np.float64),
                "hand_action_left": np.asarray([-1.0, -1.0], dtype=np.float64),
                "hand_action_right": np.asarray([1.0, 1.0], dtype=np.float64),
                "hand_dof_pos": np.zeros((2, 14), dtype=np.float64),
            }
        },
        tmp_path / f"{motion_key}.pkl",
    )
    records = [
        {
            "env_index": 0,
            "step": 0,
            "frame_index": 0,
            "motion_key": motion_key,
            "hand_primitive": [0.1, 0.2],
        }
    ]

    rows = module.hand_comparison(
        records,
        motion_dir=tmp_path,
        env_index=0,
        frame_offset=0,
        include_done=False,
        motion_timebase="eval-time",
        eval_fps=50.0,
        motion_file_fps=None,
    )

    assert rows["primitive"][0]["sign_accuracy"] == 0.5
    np.testing.assert_array_equal(rows["primitive"][0]["sign_match"], np.asarray([0.0, 1.0]))
