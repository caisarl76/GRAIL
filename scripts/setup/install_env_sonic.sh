#!/bin/bash
# Install / augment the `sonic` conda env used by GRAIL retargeting + SONIC training.
#
# Usage:
#   bash scripts/setup/install_env_sonic.sh              # default env 'sonic'
#   GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
#   BOOTSTRAP_SONIC=0 bash scripts/setup/install_env_sonic.sh         # skip Isaac Sim/Lab
#                                                                      (assume already installed)
#   INSTALL_SYSTEM_DEPS=0 bash scripts/setup/install_env_sonic.sh      # skip apt step
#   PULL_LFS=0 bash scripts/setup/install_env_sonic.sh                 # skip git-lfs pull
#
# What this script does, in order:
#   -1. (INSTALL_SYSTEM_DEPS=1 — default when apt/apk+sudo/root are available)
#       Install vulkan/GUI libs + git-lfs via apt or apk. Uses sudo if needed;
#       no-op if we're neither root nor have sudo.
#   0. (BOOTSTRAP_SONIC=1 — default) Create the conda env with Python 3.11,
#      pip-install Isaac Sim 5.1.0 (`isaacsim[all,extscache]`), clone Isaac
#      Lab v2.3.2 to $ISAAC_LAB_DIR (default: ~/IsaacLab), run
#      `./isaaclab.sh --install all`, pip install the core `isaaclab`
#      editable, and install `vector_quantize_pytorch`. Set BOOTSTRAP_SONIC=0
#      to skip when you already have an env with IsaacLab/IsaacSim installed
#      (e.g. gearenv).
#   1. Applies optional NVIDIA GMR overrides from grail/retargeting/gmr_overrides/
#      when present (idempotent — safe to rerun).
#   2. Symlinks data/motion_lib_genhoi + models into imports/SONIC/gear_sonic/.
#   3. pip install --no-deps -e imports/GMR + install
#      imports/SONIC/gear_sonic[training,data_collection]
#      + GRAIL package (editable) + huggingface_hub.
#   4. pip install retargeting-specific deps (smplx, mujoco, pxr, trimesh, ...).
#   5. Sanity-imports the top-level modules.
#   6. (PULL_LFS=1 — default when git-lfs is on PATH) git-lfs pull on
#      imports/SONIC so the robot mesh STLs + policy ONNX materialize.

set -eo pipefail

ENV_NAME="${GRAIL_SONIC_ENV:-sonic}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GMR_DIR="${REPO_ROOT}/imports/GMR"
OVERRIDES="${REPO_ROOT}/grail/retargeting/gmr_overrides"
BOOTSTRAP_SONIC="${BOOTSTRAP_SONIC:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
PULL_LFS="${PULL_LFS:-1}"
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/IsaacLab}"
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1.0}"
ISAAC_LAB_TAG="${ISAAC_LAB_TAG:-v2.3.2}"

echo ">>> Target conda env: ${ENV_NAME}"
echo ">>> Repo root:        ${REPO_ROOT}"
echo ">>> Bootstrap mode:   ${BOOTSTRAP_SONIC} (1=install Isaac Sim/Lab, 0=assume present)"

# --- Step -1: system deps (Vulkan/GUI/git-lfs) --------------------------
# Idempotent: re-installs are a fast pass. Skipped entirely when no supported
# package manager can execute or when we can't elevate.
can_run_command() {
    command -v "$1" &>/dev/null && "$1" --version &>/dev/null
}

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
    APT_PKGS=(
        libvulkan1 vulkan-tools mesa-vulkan-drivers
        libxcb-xfixes0 libxcb-cursor0 libxrandr2 libxi6 libxcursor1
        libxt6 libxtst6 libxss1 libxrender1 libgl1 libegl1 libglu1-mesa
        ffmpeg git-lfs rsync
    )
    APK_PKGS=(
        vulkan-loader vulkan-tools mesa-vulkan-swrast
        libxcb xcb-util-cursor libxrandr libxi libxcursor
        libxt libxtst libxscrnsaver libxrender mesa-gl mesa-egl mesa-glu
        ffmpeg git-lfs rsync
    )

    if can_run_command apt-get; then
        if [[ "$(id -u)" -eq 0 ]]; then
            APT_CMD=(apt-get)
        elif command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
            APT_CMD=(sudo apt-get)
        else
            APT_CMD=()
            echo ">>> [skip apt] not root and no passwordless sudo; install these manually if missing:"
            echo "    ${APT_PKGS[*]}"
        fi
        if [[ ${#APT_CMD[@]} -gt 0 ]]; then
            echo ">>> Installing system deps via ${APT_CMD[*]} (Vulkan, GUI, git-lfs)"
            "${APT_CMD[@]}" update -qq
            "${APT_CMD[@]}" install -y --no-install-recommends "${APT_PKGS[@]}" | tail -3
        fi
    elif can_run_command apk; then
        if [[ "$(id -u)" -eq 0 ]]; then
            APK_CMD=(apk)
        elif command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
            APK_CMD=(sudo apk)
        else
            APK_CMD=()
            echo ">>> [skip apk] not root and no passwordless sudo; install these manually if missing:"
            echo "    ${APK_PKGS[*]}"
        fi
        if [[ ${#APK_CMD[@]} -gt 0 ]]; then
            echo ">>> Installing system deps via ${APK_CMD[*]} (Vulkan, GUI, git-lfs)"
            "${APK_CMD[@]}" update
            "${APK_CMD[@]}" add --no-cache "${APK_PKGS[@]}" | tail -3
        fi
    else
        echo ">>> [skip system deps] no runnable apt-get or apk found; install these manually if missing:"
        echo "    apt: ${APT_PKGS[*]}"
        echo "    apk: ${APK_PKGS[*]}"
    fi
fi

# --- Step 0: bootstrap the env + Isaac Sim + Isaac Lab ------------------
eval "$(conda shell.bash hook)"

if [[ "${BOOTSTRAP_SONIC}" == "1" ]]; then
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo ">>> Creating conda env '${ENV_NAME}' with Python 3.11"
        # Avoid Anaconda default-channel ToS prompts in non-interactive containers.
        conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.11
    fi
    conda activate "${ENV_NAME}"

    echo ">>> Upgrading pip"
    pip install --upgrade pip

    if ! python -c "import isaacsim" 2>/dev/null; then
        echo ">>> Installing Isaac Sim ${ISAAC_SIM_VERSION} (~6 GB download)"
        pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
            --extra-index-url https://pypi.nvidia.com
    fi

    # Accept EULA non-interactively on first import. The Kit kernel checks
    # for the literal file <isaacsim_pkg>/kit/EULA_ACCEPTED before showing
    # its interactive prompt — write it directly so this works in non-TTY
    # builds (CI, Docker image bakes) where stdin is closed and the
    # `python -c "import isaacsim"` workaround silently fails.
    export OMNI_KIT_ACCEPT_EULA=Yes
    ISAACSIM_PKG=$(python -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))")
    echo "yes" > "${ISAACSIM_PKG}/kit/EULA_ACCEPTED"
    python -c "import isaacsim" >/dev/null

    # Pre-install flatdict without build isolation. flatdict 4.0.1 (pinned by
    # Isaac Lab core) has a legacy setup.py that imports pkg_resources, which
    # setuptools 81+ removed. PEP 517 build isolation installs the latest
    # setuptools, so the wheel build fails. Pin setuptools<81 in the env
    # first, then build flatdict against it.
    pip install 'setuptools<81' 'wheel==0.44.0' 'packaging==23.0'
    pip install 'flatdict==4.0.1' --no-build-isolation

    if [[ ! -d "${ISAAC_LAB_DIR}" ]]; then
        echo ">>> Cloning Isaac Lab ${ISAAC_LAB_TAG} to ${ISAAC_LAB_DIR}"
        git clone --depth 1 --branch "${ISAAC_LAB_TAG}" \
            https://github.com/isaac-sim/IsaacLab.git "${ISAAC_LAB_DIR}"
    fi

    if ! python -c "import isaaclab" 2>/dev/null; then
        echo ">>> Running ./isaaclab.sh --install all (~10-15 min, ~4 GB)"
        (cd "${ISAAC_LAB_DIR}" && ./isaaclab.sh --install all)
        # Isaac Lab's --install flag sometimes skips the core `isaaclab`
        # package when a transitive dep (e.g., flatdict) failed during the
        # first pass. Install it explicitly to be safe.
        pip install -e "${ISAAC_LAB_DIR}/source/isaaclab"
    fi

    # Required by some gear_sonic configs.
    pip install vector_quantize_pytorch
else
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "ERROR: conda env '${ENV_NAME}' does not exist and BOOTSTRAP_SONIC=0." >&2
        echo "       Either unset BOOTSTRAP_SONIC (default bootstrap) or create the env first." >&2
        exit 1
    fi
    conda activate "${ENV_NAME}"
fi

if [[ ! -d "${GMR_DIR}/general_motion_retargeting" ]]; then
    echo "ERROR: ${GMR_DIR} is empty." >&2
    echo "       Run: git submodule update --init imports/GMR" >&2
    exit 1
fi

# --- Step 1: apply optional NVIDIA GMR overrides ------------------------
if [[ -d "${OVERRIDES}" ]]; then
    echo ">>> Applying GMR overrides: ${OVERRIDES} -> ${GMR_DIR}"
    rsync -a --exclude='README.md' "${OVERRIDES}/" "${GMR_DIR}/"
else
    echo ">>> [skip GMR overrides] optional directory not found: ${OVERRIDES}"
fi

# --- Step 2: surface data/ and models/ into the SONIC submodule ---------
# imports/SONIC/gear_sonic/ is the cwd for training scripts; it expects
# data/motion_lib_genhoi/... and models/... to resolve from there.
GEAR_SONIC="${REPO_ROOT}/imports/SONIC/gear_sonic"
mkdir -p "${REPO_ROOT}/data/motion_lib_genhoi" "${REPO_ROOT}/imports/SONIC/models"
ln -sfn ../../../../data/motion_lib_genhoi "${GEAR_SONIC}/data/motion_lib_genhoi"
ln -sfn ../models "${GEAR_SONIC}/models"
echo ">>> Linked ${GEAR_SONIC}/data/motion_lib_genhoi -> repo data/"
echo ">>> Linked ${GEAR_SONIC}/models -> imports/SONIC/models/"

# --- Step 3: editable installs ------------------------------------------
echo ">>> pip install -e imports/GMR (--no-deps; deps pinned below)"
pip install --no-deps -e "${GMR_DIR}"

echo ">>> pip install -e imports/SONIC/gear_sonic[training,data_collection] + huggingface_hub"
# data_collection provides the LeRobot v2.1 exporter stack used by
# grail.cli.export_ego_lerobot. Skip LeRobot's LFS test artifacts during the
# git dependency clone; they are not needed for dataset writing.
GIT_LFS_SKIP_SMUDGE=1 pip install -e "${GEAR_SONIC}[training,data_collection]"
pip install huggingface_hub

echo ">>> pip install -e . (grail, --no-deps)"
# --no-deps: grail's setup.cfg has unpinned numpy/opencv-python, which resolve
# to numpy 2.x + opencv 4.13 and break gear_sonic (numpy==1.26.4), isaaclab-rl
# (numpy<2), and isaacsim-kernel (numpy==1.26.0). The sonic env only consumes
# grail.retargeting; its real deps (smplx, scipy, mujoco, mink, trimesh, pxr,
# isaaclab, gmr) are installed by other steps in this script.
pip install --no-deps -e "${REPO_ROOT}"

# --- Step 4: retargeting-specific deps ----------------------------------
echo ">>> pip install retargeting deps"
pip install \
    'numpy==1.26.4' \
    'smplx @ git+https://github.com/vchoutas/smplx' \
    joblib \
    trimesh \
    usd-core \
    'scipy==1.15.3' \
    rich \
    tqdm \
    mujoco \
    'qpsolvers[proxqp]' \
    loop_rate_limiters \
    natsort \
    'redis[hiredis]' \
    'imageio[ffmpeg]' \
    protobuf

# GMR needs mink, but current mink metadata asks for qpsolvers[daqp] and pulls
# daqp>=0.8.2. Isaac Lab 2.3.2 pins daqp==0.7.2, and GRAIL retargeting uses
# proxqp through qpsolvers, so install mink without letting it rewrite daqp.
pip install --no-deps mink

pip install \
    'simple-raycaster @ git+https://github.com/Agent-3154/simple-raycaster.git@197daa6dcb146c5ce3e675a173328e17df6b9777'

# --- Step 4b: SONIC training/eval-callback deps -------------------------
# smpl_sim is a non-PyPI package providing compute_metrics_lite, used by the
# SONIC eval-watcher's im_eval callback (gear_sonic/trl/callbacks/im_eval_callback.py).
# Without it, eval `python eval_agent_trl.py` crashes at metrics computation
# and no rendered videos get uploaded to wandb.
#
# Sourced from ZhengyiLuo's SMPLSim repo. Its non-trivial deps (numpy-stl, vtk,
# easydict, gymnasium, mediapy, torchgeometry) aren't pulled in by
# pip-from-git automatically because the package's setup.py doesn't always
# install_requires them — list them explicitly.
pip install \
    'numpy==1.26.4' numpy-stl easydict gymnasium mediapy torchgeometry vtk

# gear_sonic.utils.motion_lib.torch_humanoid_batch imports open3d during
# IsaacLab checkpoint eval. It is currently listed in SONIC's sim requirements
# but not in the training extra, so install it explicitly for pnp eval.
pip install 'open3d==0.19.0'

# SMPLSim's setup metadata advertises a broad numpy dependency and pulls numpy
# 2.x in fresh envs. Install its runtime deps explicitly above, then keep the
# editable package boundary dependency-free.
pip install --no-deps \
    'smplx @ git+https://github.com/ZhengyiLuo/smplx.git@master'
pip install --no-deps \
    'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git'

# Re-run a final repair pass because Isaac Lab/Isaac Sim, GEAR-SONIC, and the
# retargeting stack carry incompatible metadata ranges. The actual runtime
# target is NumPy 1.26.x for Isaac/SONIC ABI compatibility, with 1.26.4 chosen
# because gear_sonic pins it exactly.
echo ">>> Restoring Isaac/SONIC compatibility pins"
pip install \
    'numpy==1.26.4' \
    'packaging==23.0' \
    'psutil==5.9.8' \
    'click==8.4.1' \
    'daqp==0.7.2' \
    'opencv-python==4.11.0.86' \
    'sympy==1.13.3'
# Keep the PyTorch binary trio ABI-matched. Installing only torchaudio can
# upgrade torch while leaving torchvision pinned to an older wheel, which makes
# `import torchvision` fail with missing custom ops such as torchvision::nms.
pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.7.0' \
    'torchvision==0.22.0' \
    'torchaudio==2.7.0'

# --- Step 6: git-lfs pull for SONIC assets ------------------------------
# Mesh STLs + policy ONNX files are LFS-tracked. Without this pull, the
# preflight check fails at the size check (pointer files are <1 KB).
if [[ "${PULL_LFS}" == "1" ]] && command -v git-lfs &>/dev/null; then
    echo ">>> git lfs install + pull in imports/SONIC"
    # Avoid git's "dubious ownership" refusal when running as root inside a
    # container against a bind-mounted host repo (different UIDs).
    git config --global --add safe.directory "${REPO_ROOT}" 2>/dev/null || true
    git config --global --add safe.directory "${REPO_ROOT}/imports/SONIC" 2>/dev/null || true
    # --skip-repo: set up global LFS filters only. Without it, `git lfs install`
    # aborts with exit 2 when an identical pre-push hook already exists in the
    # cwd repo (idempotency foot-gun under `set -e`). `git lfs pull` below
    # works regardless since SONIC already has the hook.
    git lfs install --skip-repo
    (cd "${REPO_ROOT}/imports/SONIC" && git lfs pull) | tail -3 || \
        echo "  [WARN] git lfs pull failed — run manually: cd imports/SONIC && git lfs pull"
elif [[ "${PULL_LFS}" == "1" ]]; then
    echo ">>> [skip git-lfs pull] git-lfs not on PATH — install it and run:"
    echo "    cd imports/SONIC && git lfs pull"
fi

# --- Sanity checks -------------------------------------------------------
echo ">>> Verifying install"
python - <<'PY'
import importlib.metadata as md

expected_exact = {
    "numpy": "1.26.4",
    "packaging": "23.0",
    "psutil": "5.9.8",
    "click": "8.4.1",
    "daqp": "0.7.2",
    "opencv-python": "4.11.0.86",
    "sympy": "1.13.3",
}

for package, expected in expected_exact.items():
    actual = md.version(package)
    if actual != expected:
        raise SystemExit(f"{package}=={actual}; expected {expected}")

torch = md.version("torch")
if torch.split("+", 1)[0] != "2.7.0":
    raise SystemExit(f"torch=={torch}; expected 2.7.0")

torchvision = md.version("torchvision")
if torchvision.split("+", 1)[0] != "0.22.0":
    raise SystemExit(f"torchvision=={torchvision}; expected 0.22.0")

torchaudio = md.version("torchaudio")
if torchaudio.split("+", 1)[0] != "2.7.0":
    raise SystemExit(f"torchaudio=={torchaudio}; expected 2.7.0")

import torch as _torch  # noqa: F401
import torchvision as _torchvision  # noqa: F401

print("  pinned packages: OK")
PY
python -c "import general_motion_retargeting as gmr; print(f'  GMR: {gmr.__file__}')"
python -c "from grail.retargeting.retarget import main; print('  grail.retargeting.retarget: OK')"
python -c "import smplx, mujoco; print('  smplx, mujoco: OK')"
python -c "import open3d; print('  open3d: OK')"
python -c "import wandb; print('  wandb: OK')"
python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from gear_sonic.data.exporter import Gr00tDataExporter

print("  LeRobot data exporter: OK")
PY
if [[ "${BOOTSTRAP_SONIC}" == "1" ]]; then
    OMNI_KIT_ACCEPT_EULA=Yes python -c "import isaaclab, isaacsim; print('  isaaclab + isaacsim: OK')"
fi

echo ""
echo "Setup complete. Quick start:"
echo "  bash grail/retargeting/scripts/retarget_pipeline.sh <data_dir> <output_folder>"
echo ""
echo "Full preflight:"
echo "  OMNI_KIT_ACCEPT_EULA=Yes python imports/SONIC/check_environment.py --training"
