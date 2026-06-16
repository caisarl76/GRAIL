from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_fake_docker(
    tmp_path: Path,
    fake_docker: str,
    args: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    docker_script = bin_dir / "docker"
    docker_script.write_text(fake_docker)
    docker_script.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["DOCKER_LOG"] = str(docker_log)
    env["GRAIL_CACHE_DIR"] = str(tmp_path / "grail-cache")
    env["GRAIL_HF_CACHE"] = str(tmp_path / "hf-cache")
    env.pop("DISPLAY", None)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/docker/run_grail.sh"), *(args or [])],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, docker_log


def test_run_grail_creates_new_container_with_bin_bash(tmp_path: Path):
    fake_docker = """#!/usr/bin/env bash
set -eo pipefail
printf '%s\\n' "$*" >> "$DOCKER_LOG"

if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    exit 1
fi

if [[ "$1" == "build" ]]; then
    exit 0
fi

if [[ "$1" == "inspect" ]]; then
    exit 1
fi

if [[ "$1" == "run" ]]; then
    exit 0
fi

echo "unexpected docker command: $*" >&2
exit 64
"""

    result, docker_log = _run_with_fake_docker(tmp_path, fake_docker)

    assert result.returncode == 0, result.stderr
    docker_commands = docker_log.read_text()
    assert "image inspect grail-fixed:latest" in docker_commands
    assert (
        "build --build-arg GRAIL_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
        in docker_commands
    )
    assert " -t grail-fixed:latest -" in docker_commands
    assert docker_commands.rstrip().endswith(" grail-fixed:latest /bin/bash")
    assert not docker_commands.rstrip().endswith(" grail-fixed:latest bash")


def test_run_grail_rebuild_refreshes_default_fixed_image(tmp_path: Path):
    fake_docker = """#!/usr/bin/env bash
set -eo pipefail
printf '%s\\n' "$*" >> "$DOCKER_LOG"
REMOVED_MARKER="${DOCKER_LOG}.removed"

if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    exit 0
fi

if [[ "$1" == "build" || "$1" == "run" ]]; then
    exit 0
fi

if [[ "$1" == "rm" ]]; then
    touch "$REMOVED_MARKER"
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "grail-sonic" ]]; then
    if [[ -f "$REMOVED_MARKER" ]]; then
        exit 1
    fi
    exit 0
fi

echo "unexpected docker command: $*" >&2
exit 64
"""

    result, docker_log = _run_with_fake_docker(tmp_path, fake_docker, ["--rebuild"])

    assert result.returncode == 0, result.stderr
    docker_commands = docker_log.read_text()
    assert "image inspect grail-fixed:latest" not in docker_commands
    assert (
        "build --build-arg GRAIL_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
        in docker_commands
    )
    assert "rm -f grail-sonic" in docker_commands
    assert docker_commands.rstrip().endswith(" grail-fixed:latest /bin/bash")


def test_run_grail_reports_stale_usr_bin_bash_container(tmp_path: Path):
    fake_docker = """#!/usr/bin/env bash
set -eo pipefail
printf '%s\\n' "$*" >> "$DOCKER_LOG"

if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "grail-sonic" ]]; then
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{.State.Running}}" ]]; then
    echo false
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{json .Config.Cmd}}" ]]; then
    echo '["/usr/bin/bash"]'
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{.Config.Image}}" ]]; then
    echo docker.io/nvgrail/grail:latest
    exit 0
fi

if [[ "$1" == "start" ]]; then
    echo "exec /usr/bin/bash: no such file or directory" >&2
    exit 127
fi

echo "unexpected docker command: $*" >&2
exit 64
"""

    result, docker_log = _run_with_fake_docker(tmp_path, fake_docker)

    assert result.returncode != 0
    assert "created with an invalid shell command" in result.stderr
    assert "bash scripts/docker/run_grail.sh --rebuild" in result.stderr
    assert "start -ai grail-sonic" not in docker_log.read_text()


def test_run_grail_reports_stale_bin_bash_base_image_container(tmp_path: Path):
    fake_docker = """#!/usr/bin/env bash
set -eo pipefail
printf '%s\\n' "$*" >> "$DOCKER_LOG"

if [[ "$1" == "inspect" && "$2" == "grail-sonic" ]]; then
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{.State.Running}}" ]]; then
    echo false
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{json .Config.Cmd}}" ]]; then
    echo '["/bin/bash"]'
    exit 0
fi

if [[ "$1" == "inspect" && "$2" == "--format={{.Config.Image}}" ]]; then
    echo docker.io/nvgrail/grail:latest
    exit 0
fi

if [[ "$1" == "start" ]]; then
    echo "exec /bin/bash: no such file or directory" >&2
    exit 127
fi

echo "unexpected docker command: $*" >&2
exit 64
"""

    result, docker_log = _run_with_fake_docker(tmp_path, fake_docker)

    assert result.returncode != 0
    assert "created with an invalid shell command" in result.stderr
    assert "docker.io/nvgrail/grail:latest" in result.stderr
    assert "bash scripts/docker/run_grail.sh --rebuild" in result.stderr
    assert "start -ai grail-sonic" not in docker_log.read_text()


def test_fixed_dockerfile_uses_glibc_cuda_base_and_installs_miniforge():
    dockerfile = (REPO_ROOT / "Dockerfile.grail-fixed").read_text()

    assert "ARG GRAIL_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04" in dockerfile
    assert "apt-get update -qq" in dockerfile
    assert "apt-get install -y --no-install-recommends" in dockerfile
    assert "libvulkan1 vulkan-tools mesa-vulkan-drivers" in dockerfile
    assert "libxt6" in dockerfile
    assert "libglu1-mesa" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "git-lfs" in dockerfile
    assert "rsync" in dockerfile
    assert "MINIFORGE_URL" in dockerfile
    assert "github.com/conda-forge/miniforge/releases/latest/download" in dockerfile
    assert "bash /tmp/miniforge.sh -b -p /root/miniconda3" in dockerfile
    assert "conda config --system --add channels conda-forge" in dockerfile
    assert "conda config --system --set channel_priority strict" in dockerfile
    assert "conda --version" in dockerfile
    assert "conda clean -afy" in dockerfile
    assert "repo.anaconda.com" not in dockerfile
    assert "MINICONDA_URL" not in dockerfile
    assert "apk" not in dockerfile
    assert "gcompat" not in dockerfile
    assert "coreutils" not in dockerfile
