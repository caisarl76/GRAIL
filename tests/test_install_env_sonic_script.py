from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _copy_install_script(tmp_path: Path, *, with_overrides: bool = True) -> Path:
    script = tmp_path / "scripts/setup/install_env_sonic.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts/setup/install_env_sonic.sh", script)

    (tmp_path / "imports/GMR/general_motion_retargeting").mkdir(parents=True)
    (tmp_path / "imports/SONIC/gear_sonic/data").mkdir(parents=True)
    if with_overrides:
        (tmp_path / "grail/retargeting/gmr_overrides").mkdir(parents=True)
    return script


def test_install_env_sonic_uses_apk_when_apt_get_is_not_runnable(tmp_path: Path):
    script = _copy_install_script(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"

    _write_executable(
        bin_dir / "apt-get",
        """#!/usr/bin/env bash
printf 'apt-get %s\\n' "$*" >> "$COMMAND_LOG"
if [[ "$1" == "--version" ]]; then
    exit 127
fi
exit 64
""",
    )
    _write_executable(
        bin_dir / "apk",
        """#!/usr/bin/env bash
printf 'apk %s\\n' "$*" >> "$COMMAND_LOG"
exit 0
""",
    )
    _write_executable(
        bin_dir / "id",
        """#!/usr/bin/env bash
if [[ "$1" == "-u" ]]; then
    echo 0
else
    /usr/bin/id "$@"
fi
""",
    )
    _write_executable(
        bin_dir / "conda",
        """#!/usr/bin/env bash
if [[ "$1" == "shell.bash" && "$2" == "hook" ]]; then
    cat <<'HOOK'
conda() {
    if [[ "$1" == "env" && "$2" == "list" ]]; then
        printf 'sonic\\n'
    elif [[ "$1" == "activate" ]]; then
        return 0
    else
        return 0
    fi
}
HOOK
    exit 0
fi
exit 0
""",
    )
    for command in ("rsync", "pip", "python", "git", "git-lfs"):
        _write_executable(
            bin_dir / command,
            f"""#!/usr/bin/env bash
printf '{command} %s\\n' "$*" >> "$COMMAND_LOG"
exit 0
""",
        )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["COMMAND_LOG"] = str(command_log)
    env["BOOTSTRAP_SONIC"] = "0"
    env["PULL_LFS"] = "0"
    env["INSTALL_SYSTEM_DEPS"] = "1"
    env["GRAIL_SONIC_ENV"] = "sonic"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text()
    assert "apt-get --version" in commands
    assert "apt-get update" not in commands
    assert "apk update" in commands
    assert "apk add --no-cache" in commands
    assert "vulkan-loader" in commands
    assert "libxt" in commands
    assert "mesa-glu" in commands
    assert "ffmpeg" in commands
    assert "git-lfs" in commands
    assert "rsync" in commands


def test_install_env_sonic_creates_env_from_conda_forge_only():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert (
        'conda create -y -n "${ENV_NAME}" '
        '-c conda-forge --override-channels python=3.11'
    ) in script
    assert "repo.anaconda.com" not in script


def test_install_env_sonic_installs_isaacsim_renderer_system_libraries():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert "libxt6" in script
    assert "libglu1-mesa" in script
    assert "libxt " in script
    assert "mesa-glu" in script


def test_install_env_sonic_installs_ffmpeg_for_visualization_postprocess():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()
    apt_pkgs = re.search(r"APT_PKGS=\(\n(?P<body>.*?)\n    \)", script, re.DOTALL)
    apk_pkgs = re.search(r"APK_PKGS=\(\n(?P<body>.*?)\n    \)", script, re.DOTALL)

    assert apt_pkgs is not None
    assert apk_pkgs is not None
    assert re.search(r"\bffmpeg\b", apt_pkgs.group("body"))
    assert re.search(r"\bffmpeg\b", apk_pkgs.group("body"))


def test_install_env_sonic_links_gear_sonic_to_imports_sonic_models():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert '"${REPO_ROOT}/imports/SONIC/models"' in script
    assert 'ln -sfn ../models "${GEAR_SONIC}/models"' in script
    assert 'ln -sfn ../../../models "${GEAR_SONIC}/models"' not in script
    assert "imports/SONIC/models/" in script


def test_install_env_sonic_skips_missing_gmr_overrides(tmp_path: Path):
    script = _copy_install_script(tmp_path, with_overrides=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"

    _write_executable(
        bin_dir / "conda",
        """#!/usr/bin/env bash
if [[ "$1" == "shell.bash" && "$2" == "hook" ]]; then
    cat <<'HOOK'
conda() {
    if [[ "$1" == "env" && "$2" == "list" ]]; then
        printf 'sonic\\n'
    elif [[ "$1" == "activate" ]]; then
        return 0
    else
        return 0
    fi
}
HOOK
    exit 0
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "rsync",
        """#!/usr/bin/env bash
printf 'rsync %s\\n' "$*" >> "$COMMAND_LOG"
exit 23
""",
    )
    for command in ("pip", "python", "git", "git-lfs"):
        _write_executable(
            bin_dir / command,
            f"""#!/usr/bin/env bash
printf '{command} %s\\n' "$*" >> "$COMMAND_LOG"
exit 0
""",
        )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["COMMAND_LOG"] = str(command_log)
    env["BOOTSTRAP_SONIC"] = "0"
    env["PULL_LFS"] = "0"
    env["INSTALL_SYSTEM_DEPS"] = "0"
    env["GRAIL_SONIC_ENV"] = "sonic"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert ">>> [skip GMR overrides]" in result.stdout
    assert "rsync" not in command_log.read_text()


def test_install_env_sonic_prevents_unpinned_retargeting_dependency_drift():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert 'pip install --no-deps -e "${GMR_DIR}"' in script
    assert "pip install --no-deps mink" in script
    assert "'qpsolvers[proxqp]'" in script
    assert "'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git'" in script
    assert "pip install --no-deps \\\n    'smplx @ git+https://github.com/ZhengyiLuo/smplx.git@master'" in script
    assert "pip install --no-deps \\\n    'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git'" in script


def test_install_env_sonic_installs_open3d_for_motion_lib_import():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert "'open3d==0.19.0'" in script
    assert "import open3d" in script
    assert "open3d: OK" in script


def test_install_env_sonic_installs_lerobot_export_dependencies():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    assert "gear_sonic[training,data_collection]" in script
    assert "GIT_LFS_SKIP_SMUDGE=1 pip install -e" in script
    assert "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset" in script
    assert "from gear_sonic.data.exporter import Gr00tDataExporter" in script
    assert "LeRobot data exporter: OK" in script


def test_install_env_sonic_restores_runtime_pins_before_verifying_install():
    script = (REPO_ROOT / "scripts/setup/install_env_sonic.sh").read_text()

    repair_index = script.index(">>> Restoring Isaac/SONIC compatibility pins")
    verify_index = script.index(">>> Verifying install")

    assert repair_index < verify_index
    for requirement in (
        "'numpy==1.26.4'",
        "'packaging==23.0'",
        "'psutil==5.9.8'",
        "'click==8.4.1'",
        "'daqp==0.7.2'",
        "'opencv-python==4.11.0.86'",
        "'torchaudio==2.7.0'",
    ):
        assert requirement in script
    assert "import importlib.metadata as md" in script
    assert '"click": "8.4.1"' in script
    assert "pinned packages: OK" in script
    assert "import wandb" in script
    assert "wandb: OK" in script
