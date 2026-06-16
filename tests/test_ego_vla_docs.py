from __future__ import annotations

import os
import subprocess
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_motion(root: Path, key: str = "pickup_table__can__000") -> None:
    for subdir in ["robot", "objects", "meta", "object_usd"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    robot = {
        key: {
            "dof": np.zeros((2, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
            "fps": 25.0,
            "hand_dof_pos": np.zeros((2, 14), dtype=np.float32),
        }
    }
    objects = {
        key: {
            "root_pos": np.zeros((2, 1, 3), dtype=np.float32),
            "root_quat": np.tile(
                np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), (2, 1, 1)
            ),
        }
    }
    joblib.dump(robot, root / "robot" / f"{key}.pkl")
    joblib.dump(objects, root / "objects" / f"{key}.pkl")
    joblib.dump({"object_name": "can"}, root / "meta" / f"{key}.pkl")
    (root / "object_usd" / f"{key}.usd").write_text("#usda 1.0\n", encoding="utf-8")


def test_ego_vla_pipeline_docs_capture_gpu_split_and_script_entrypoint():
    doc = (REPO_ROOT / "docs/source/ego_vla_pipeline.md").read_text()
    assert "RTX" in doc
    assert "H100" in doc
    assert "--ego-frame-root" in doc
    assert "--render-only" in doc
    assert "--camera-target-offset" in doc
    assert "--with-third-eye-comparison" in doc
    assert "side-by-side" in doc
    assert "OAK" in doc
    assert "69 deg" in doc
    assert "-55.0 deg" in doc
    assert "fixed relative to the robot root" in doc
    assert "grail.visualization.batch_render_ego" in doc
    assert "scripts/docker/render_ego_lerobot.sh" in doc

    index = (REPO_ROOT / "docs/source/index.rst").read_text()
    assert "ego_vla_pipeline" in index


def test_render_ego_lerobot_wrapper_has_help_and_print_command_mode():
    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"

    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ego-view LeRobot" in help_result.stdout
    assert "RTX" in help_result.stdout
    assert "H100" in help_result.stdout
    assert "--with-third-eye-comparison" in help_result.stdout

    print_result = subprocess.run(
        [
            "bash",
            str(script),
            "--motion-lib",
            "/data/pickup_table",
            "--output",
            "/data/lerobot",
            "--ego-frame-root",
            "/data/ego_frames",
            "--num-shards",
            "8",
            "--shard-index",
            "3",
            "--dry-run",
            "--print-command",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "scripts/docker/run_grail.sh" in print_result.stdout
    assert "python -m grail.cli.export_ego_lerobot" in print_result.stdout
    assert "--ego-frame-root /data/ego_frames" in print_result.stdout
    assert "--num-shards 8" in print_result.stdout

    render_result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--motion-lib",
            "/data/pickup_table",
            "--output",
            "/data/ego_frames",
            "--num-shards",
            "4",
            "--shard-index",
            "1",
            "--camera-target-offset",
            "1.0",
            "0.0",
            "0.0",
            "--skip-existing",
            "--dry-run",
            "--print-command",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "python -m grail.visualization.batch_render_ego" in render_result.stdout
    assert "--motion_lib /data/pickup_table" in render_result.stdout
    assert "--output_dir /data/ego_frames" in render_result.stdout
    assert "--camera_target_offset 1.0 0.0 0.0" in render_result.stdout
    assert "--skip_existing" in render_result.stdout


def test_render_pnp_dual_gpu_script_prints_two_gpu_jobs():
    script = REPO_ROOT / "scripts/docker/render_pnp_ego_dual_gpu.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--data-root",
            "/data/grail",
            "--output-root",
            "/data/ego_frames",
            "--log-root",
            "/tmp/grail_logs",
            "--resolution",
            "640x480",
            "--print-command",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "/data/grail/pickup_table" in result.stdout
    assert "/data/grail/pickup_ground" in result.stdout
    assert "/data/ego_frames/pnp_table" in result.stdout
    assert "/data/ego_frames/pnp_ground" in result.stdout
    assert "--skip-existing" in result.stdout


def test_render_pnp_dual_gpu_preflights_both_motion_libraries_before_launch(tmp_path: Path):
    _write_motion(tmp_path / "pickup_table")
    script = REPO_ROOT / "scripts/docker/render_pnp_ego_dual_gpu.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "ego_frames"),
            "--log-root",
            str(tmp_path / "logs"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Missing motion library for pnp_ground" in result.stderr
    assert "Started pnp_table" not in result.stdout


def test_render_ego_lerobot_wrapper_prints_comparison_pipeline():
    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["GRAIL_RENDER_EGO_USE_CONDA"] = "0"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--with-third-eye-comparison",
            "--motion-lib",
            "/data/pickup_table",
            "--output",
            "/data/ego_frames",
            "--side-by-side-output",
            "/data/ego_compare",
            "--resolution",
            "640x480",
            "--max-motions",
            "2",
            "--dry-run",
            "--print-command",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "python -m grail.visualization.batch_render_ego" in result.stdout
    assert "--write_motion_keys" in result.stdout
    assert "python -m grail.visualization.prepare_vis_shard" in result.stdout
    assert "python -u -m grail.visualization.batch_render_replay" in result.stdout
    assert "ffmpeg" in result.stdout
    assert "/data/ego_compare" in result.stdout


def test_render_ego_lerobot_accepts_singular_write_manifest_alias():
    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["GRAIL_RENDER_EGO_USE_CONDA"] = "0"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--motion-lib",
            "/data/pickup_table",
            "--output",
            "/data/ego_frames",
            "--write-manifest",
            "--dry-run",
            "--print-command",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--write_manifests" in result.stdout


def test_render_ego_lerobot_wrapper_runs_directly_inside_container(tmp_path: Path):
    _write_motion(tmp_path)
    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["GRAIL_RENDER_EGO_USE_CONDA"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--motion-lib",
            str(tmp_path),
            "--output",
            str(tmp_path / "ego"),
            "--dry-run",
            "--write-manifests",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Ego render plan: 1 motions" in result.stdout
    assert "scripts/docker/run_grail.sh" not in result.stdout
    assert (tmp_path / "ego" / "pickup_table__can__000.json").is_file()


def test_render_ego_lerobot_comparison_dry_run_writes_matching_motion_keys(tmp_path: Path):
    _write_motion(tmp_path)
    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["GRAIL_RENDER_EGO_USE_CONDA"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--with-third-eye-comparison",
            "--motion-lib",
            str(tmp_path),
            "--output",
            str(tmp_path / "ego"),
            "--side-by-side-output",
            str(tmp_path / "compare"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[1/3] Rendering ego-view videos and writing motion key list" in result.stdout
    assert "Dry run complete" in result.stdout
    assert (tmp_path / "ego" / "pickup_table__can__000.json").is_file()
    assert (tmp_path / "compare" / "motion_keys.txt").read_text() == "pickup_table__can__000\n"


def test_render_ego_lerobot_direct_print_uses_sonic_conda_when_not_active(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda = bin_dir / "conda"
    conda.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    conda.chmod(0o755)

    script = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("CONDA_DEFAULT_ENV", None)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--render-only",
            "--motion-lib",
            "/data/grail/pickup_table",
            "--output",
            "/data/ego_frames/pickup_table",
            "--dry-run",
            "--print-command",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "conda run -n sonic --no-capture-output python -m grail.visualization.batch_render_ego" in result.stdout
