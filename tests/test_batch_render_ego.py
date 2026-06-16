from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from grail.visualization.batch_render_ego import (
    EgoRenderConfig,
    build_ego_render_plan,
    main,
    manifest_for_job,
    render_all,
    write_planned_manifests,
)


def _write_motion(root: Path, key: str = "pickup_table__can__000") -> None:
    for subdir in ["robot", "objects", "meta", "object_usd"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    robot = {
        key: {
            "dof": np.zeros((3, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((3, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
            "fps": 25.0,
            "hand_dof_pos": np.zeros((3, 14), dtype=np.float32),
        }
    }
    objects = {
        key: {
            "root_pos": np.zeros((3, 1, 3), dtype=np.float32),
            "root_quat": np.tile(
                np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), (3, 1, 1)
            ),
        }
    }
    joblib.dump(robot, root / "robot" / f"{key}.pkl")
    joblib.dump(objects, root / "objects" / f"{key}.pkl")
    joblib.dump({"object_name": "can"}, root / "meta" / f"{key}.pkl")
    (root / "object_usd" / f"{key}.usd").write_text("#usda 1.0\n", encoding="utf-8")


def test_build_ego_render_plan_and_manifest_use_motion_metadata(tmp_path):
    _write_motion(tmp_path)
    config = EgoRenderConfig(width=640, height=480)

    jobs = build_ego_render_plan(tmp_path, tmp_path / "ego", config=config)
    manifest = manifest_for_job(jobs[0], config)

    assert len(jobs) == 1
    assert manifest["motion_key"] == "pickup_table__can__000"
    assert manifest["status"] == "planned"
    assert manifest["fps"] == 25.0
    assert manifest["num_frames"] == 3
    assert manifest["video_path"].endswith("pickup_table__can__000.mp4")
    assert manifest["camera"]["mode"] == "root_relative_ego"
    assert manifest["camera"]["resolution"] == [480, 640]
    assert manifest["camera"]["position_offset"] == [0.25, 0.0, 0.55]
    assert manifest["camera"]["target_offset"] == [0.846, 0.0, -0.301]
    assert manifest["camera"]["intrinsics"]["model"] == "Luxonis OAK-D RGB IMX378"
    assert manifest["camera"]["intrinsics"]["hfov_deg"] == 69.0
    assert manifest["camera"]["intrinsics"]["vfov_deg"] == 55.0


def test_write_planned_manifests_creates_one_json_per_job(tmp_path):
    _write_motion(tmp_path)
    config = EgoRenderConfig(width=320, height=240)
    jobs = build_ego_render_plan(tmp_path, tmp_path / "ego", config=config)

    written = write_planned_manifests(jobs, config)

    assert written == 1
    data = json.loads((tmp_path / "ego" / "pickup_table__can__000.json").read_text())
    assert data["status"] == "planned"
    assert data["camera"]["resolution"] == [240, 320]


def test_main_dry_run_writes_manifests_without_isaac_import(tmp_path, capsys):
    _write_motion(tmp_path)

    rc = main(
        [
            "--motion_lib",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "ego"),
            "--resolution",
            "640x480",
            "--dry_run",
            "--write_manifests",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "Ego render plan: 1 motions" in captured.out
    assert (tmp_path / "ego" / "pickup_table__can__000.json").is_file()


def test_main_dry_run_writes_motion_key_list_without_isaac_import(tmp_path):
    _write_motion(tmp_path)
    motion_keys = tmp_path / "ego" / "motion_keys.txt"

    rc = main(
        [
            "--motion_lib",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "ego"),
            "--dry_run",
            "--write_motion_keys",
            str(motion_keys),
        ]
    )

    assert rc == 0
    assert motion_keys.read_text(encoding="utf-8") == "pickup_table__can__000\n"


def test_render_all_converts_selected_jobs_and_delegates_to_ego_replay(tmp_path, monkeypatch):
    _write_motion(tmp_path, "pickup_table__can__000")
    output_dir = tmp_path / "ego"
    jobs = build_ego_render_plan(tmp_path, output_dir, max_motions=1)
    config = EgoRenderConfig(width=320, height=240)
    calls = {}

    def fake_convert_motion_lib_to_trajectories(**kwargs):
        calls["convert"] = kwargs
        return sorted(kwargs["motion_filter"])

    def fake_build_replay_render_plan(shard_dir, object_usd_dir, output_dir_arg, **kwargs):
        calls["plan"] = {
            "shard_dir": shard_dir,
            "object_usd_dir": object_usd_dir,
            "output_dir": output_dir_arg,
            "kwargs": kwargs,
        }
        return [("plan", "pickup_table__can__000")], {"skipped": 0}, 1, 1

    def fake_render_replay_all(plan, **kwargs):
        calls["render"] = {"plan": plan, "kwargs": kwargs}

    monkeypatch.setattr(
        "grail.visualization.batch_render_ego.convert_motion_lib_to_trajectories",
        fake_convert_motion_lib_to_trajectories,
    )
    monkeypatch.setattr(
        "grail.visualization.batch_render_ego.build_replay_render_plan",
        fake_build_replay_render_plan,
    )
    monkeypatch.setattr(
        "grail.visualization.batch_render_ego.render_replay_all",
        fake_render_replay_all,
    )

    render_all(jobs, config=config, headless=False)

    assert calls["convert"]["data_dir"] == str(tmp_path)
    assert calls["convert"]["motion_filter"] == {"pickup_table__can__000"}
    assert calls["convert"]["quat_convention"] == "xyzw"
    assert calls["plan"]["object_usd_dir"] == str(tmp_path / "object_usd")
    assert calls["plan"]["output_dir"] == str(output_dir)
    assert calls["render"]["plan"] == [("plan", "pickup_table__can__000")]
    assert calls["render"]["kwargs"]["resolution"] == (320, 240)
    assert calls["render"]["kwargs"]["headless"] is False
    assert calls["render"]["kwargs"]["ego_camera_spec"] == config.camera_spec


def test_main_render_path_calls_render_all(tmp_path, monkeypatch):
    _write_motion(tmp_path)
    calls = {}

    def fake_render_all(jobs, *, config, headless, quat_convention):
        calls["jobs"] = jobs
        calls["config"] = config
        calls["headless"] = headless
        calls["quat_convention"] = quat_convention

    monkeypatch.setattr("grail.visualization.batch_render_ego.render_all", fake_render_all)

    rc = main(
        [
            "--motion_lib",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "ego"),
            "--resolution",
            "320x240",
            "--max_motions",
            "1",
        ]
    )

    assert rc == 0
    assert [job.motion_key for job in calls["jobs"]] == ["pickup_table__can__000"]
    assert calls["config"].width == 320
    assert calls["config"].height == 240
    assert calls["config"].target_offset == (0.846, 0.0, -0.301)
    assert calls["headless"] is True
    assert calls["quat_convention"] == "xyzw"
