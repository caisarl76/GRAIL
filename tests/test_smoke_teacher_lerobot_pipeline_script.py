from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/eval/smoke_teacher_lerobot_pipeline.sh"


def _make_motion_lib(root: Path, *, num_motions: int = 4) -> Path:
    motion_lib = root / "pickup_ground"
    for child in ("robot", "objects", "object_usd", "bps"):
        (motion_lib / child).mkdir(parents=True)
    for i in range(num_motions):
        (motion_lib / "robot" / f"pickup_ground__obj_{i:03d}__000.pkl").touch()
    return motion_lib


def test_smoke_pipeline_export_uses_rendered_subset_not_full_library_requirement(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path)
    output_root = tmp_path / "smoke"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-dir",
            str(motion_lib),
            "--bps-dir",
            str(motion_lib / "bps"),
            "--output-root",
            str(output_root),
            "--num-episodes",
            "2",
            "--num-envs",
            "1",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--only-rendered" in result.stdout
    assert "--max-episodes 2" in result.stdout
    assert "--resolution 640x480" in result.stdout
    assert "--require-all-rendered" not in result.stdout
    assert "++manager_env.config.per_rank_motion_keys_file=" in result.stdout

    keys = (output_root / "requested_motion_keys.txt").read_text(encoding="utf-8").splitlines()
    assert keys == ["pickup_ground__obj_000__000", "pickup_ground__obj_001__000"]


def test_smoke_pipeline_rejects_resolution_that_does_not_match_lerobot_schema(
    tmp_path: Path,
):
    motion_lib = _make_motion_lib(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-dir",
            str(motion_lib),
            "--bps-dir",
            str(motion_lib / "bps"),
            "--resolution",
            "320x240",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SONIC VLA expects observation.images.ego_view shape (480, 640, 3)" in result.stderr
