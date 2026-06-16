from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_visualize_script_checks_system_ffmpeg_before_postprocess():
    script = (REPO_ROOT / "grail/visualization/scripts/visualize.sh").read_text()

    assert "command -v ffmpeg" in script
    assert "INSTALL_SYSTEM_DEPS=1 bash scripts/setup/install_env_sonic.sh" in script
