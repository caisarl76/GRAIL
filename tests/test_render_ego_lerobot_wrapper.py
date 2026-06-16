from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/docker/render_ego_lerobot.sh"


def _print_command(script_args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GRAIL_RENDER_EGO_DIRECT"] = "1"
    env["GRAIL_RENDER_EGO_USE_CONDA"] = "0"
    return subprocess.run(
        ["bash", str(SCRIPT), *script_args, "--print-command"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_render_ego_lerobot_passes_export_selection_flags(tmp_path: Path):
    result = _print_command(
        [
            "--motion-lib",
            str(tmp_path / "lib"),
            "--output",
            str(tmp_path / "out"),
            "--ego-frame-root",
            str(tmp_path / "ego"),
            "--token-labels",
            str(tmp_path / "labels"),
            "--only-rendered",
            "--require-all-rendered",
            "--max-episodes",
            "5",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "grail.cli.export_ego_lerobot" in out
    assert "--only-rendered" in out
    assert "--require-all-rendered" in out
    assert "--max-episodes 5" in out
