#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMNI_KIT_ACCEPT_EULA=Yes

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/test_sonic_pnp_checkpoint.sh --task pnp_table --data-dir <motion-lib> [options]

Runs a small IsaacLab evaluation of a released GRAIL/GEAR-SONIC PnP checkpoint.
The policy MLP outputs a 66-D meta-action; actions[:64] are the SONIC latent/token
residual and actions[64:] are the two hand primitive actions.

Options:
  --task pnp_table|pnp_ground       Task/checkpoint bundle to evaluate.
  --data-dir PATH                  Motion library root with robot/, objects/, object_usd/.
  --bps-dir PATH                   BPS directory. Defaults to <data-dir>/bps.
  --bps-dim N                      Fallback BPS width when --bps-dir is missing.
                                   Defaults to 10, matching released PnP checkpoints.
  --checkpoint PATH                Checkpoint path relative to imports/SONIC or absolute.
                                   Defaults to models/<task>/last.pt.
  --output-root PATH               Output root for metrics, renders, and token debug JSON.
                                   Defaults to /workspace/grail/data/sonic_eval if present,
                                   otherwise ./data/sonic_eval.
  --motion-keys-file PATH          Text file with one motion key per line. Use this to make
                                   teacher inference match a rendered episode subset exactly.
  --gpu ID                         Sets CUDA_VISIBLE_DEVICES for the eval process.
  --num-envs N                     Number of Isaac envs. Default: 4.
  --max-unique-motions N|all       Limit loaded motions for a fast smoke test. Use "all"
                                   for full teacher-label generation. Default: 16.
  --motion-shard-rank N            Motion shard rank for parallel label generation. Default: 0.
  --motion-shard-world-size N      Number of motion shards. Default: 1.
  --disable-early-terminations     Disable non-timeout terminations so teacher-label
                                   extraction covers the full motion duration.
  --render                         Also save third-person render videos.
  --dry-run                        Print the command without running Isaac.
  -h, --help                       Show this help.

Examples:
  SNAP=$(find /root/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Locomanipulation-GRAIL/snapshots \
    -mindepth 1 -maxdepth 1 -type d | head -1)

  bash scripts/eval/test_sonic_pnp_checkpoint.sh \
    --task pnp_table \
    --data-dir "$SNAP/data/pickup_table" \
    --gpu 0 \
    --num-envs 4 \
    --max-unique-motions 16 \
    --render
EOF
}

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    cd "${script_dir}/../.." && pwd -P
}

abs_existing_dir() {
    local path="$1"
    cd "$path" && pwd -P
}

quote_command() {
    local -a cmd=("$@")
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

TASK="pnp_table"
DATA_DIR=""
BPS_DIR=""
BPS_DIM="10"
CHECKPOINT=""
OUTPUT_ROOT=""
MOTION_KEYS_FILE=""
GPU_ID=""
NUM_ENVS="4"
MAX_UNIQUE_MOTIONS="16"
MOTION_SHARD_RANK="0"
MOTION_SHARD_WORLD_SIZE="1"
DISABLE_EARLY_TERMINATIONS="0"
RENDER="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --bps-dir)
            BPS_DIR="$2"
            shift 2
            ;;
        --bps-dim)
            BPS_DIM="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --motion-keys-file)
            MOTION_KEYS_FILE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-unique-motions)
            MAX_UNIQUE_MOTIONS="$2"
            shift 2
            ;;
        --motion-shard-rank)
            MOTION_SHARD_RANK="$2"
            shift 2
            ;;
        --motion-shard-world-size)
            MOTION_SHARD_WORLD_SIZE="$2"
            shift 2
            ;;
        --disable-early-terminations)
            DISABLE_EARLY_TERMINATIONS="1"
            shift
            ;;
        --render)
            RENDER="1"
            shift
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$TASK" != "pnp_table" && "$TASK" != "pnp_ground" ]]; then
    echo "--task must be pnp_table or pnp_ground, got: $TASK" >&2
    exit 2
fi

if [[ -z "$DATA_DIR" ]]; then
    echo "--data-dir is required" >&2
    usage >&2
    exit 2
fi

REPO_ROOT="$(repo_root)"
SONIC_ROOT="${REPO_ROOT}/imports/SONIC"

for required in "$DATA_DIR/robot" "$DATA_DIR/objects" "$DATA_DIR/object_usd"; do
    if [[ ! -d "$required" ]]; then
        echo "Missing required motion-library directory: $required" >&2
        exit 2
    fi
done

DATA_DIR="$(abs_existing_dir "$DATA_DIR")"

if [[ -z "$BPS_DIR" ]]; then
    BPS_DIR="${DATA_DIR}/bps"
fi
if [[ ! -d "$BPS_DIR" ]]; then
    echo "BPS dir not found: $BPS_DIR; SONIC will run with zero ${BPS_DIM}-D BPS observations." >&2
else
    BPS_DIR="$(abs_existing_dir "$BPS_DIR")"
fi

if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="models/${TASK}/last.pt"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ -d "/workspace/grail" ]]; then
        OUTPUT_ROOT="/workspace/grail/data/sonic_eval"
    else
        OUTPUT_ROOT="${REPO_ROOT}/data/sonic_eval"
    fi
fi

TASK_OUT="${OUTPUT_ROOT}/${TASK}"
METRICS_OUT="${TASK_OUT}/metrics"
RENDER_OUT="${TASK_OUT}/renderings"
DEBUG_LOG="${TASK_OUT}/token_debug.json"

mkdir -p "$TASK_OUT" "$METRICS_OUT"
if [[ "$RENDER" == "1" ]]; then
    mkdir -p "$RENDER_OUT"
fi

cmd=(
    python -u gear_sonic/eval_agent_trl.py
    "+checkpoint=${CHECKPOINT}"
    "+headless=True"
    "++eval_callbacks=im_eval"
    "++run_eval_loop=False"
    "++num_envs=${NUM_ENVS}"
    "++eval_output_dir=${METRICS_OUT}"
    "++manager_env.config.debug_state_log=${DEBUG_LOG}"
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot"
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects"
    "++manager_env.config.object_usd_path=${DATA_DIR}/object_usd"
    "++manager_env.config.action_transform_module_cfg=models/sonic_manipulation_base/model_config.yaml"
    "++manager_env.config.action_transform_module_checkpoint=models/sonic_manipulation_base/last.pt"
    "++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}"
    "++manager_env.commands.motion.motion_lib_cfg.bps_dim=${BPS_DIM}"
    "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=${MOTION_SHARD_RANK}"
    "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=${MOTION_SHARD_WORLD_SIZE}"
    "++manager_env.commands.motion.motion_lib_cfg.load_unique_motions=true"
    "++manager_env.commands.motion.motion_lib_cfg.interleave_by_object=true"
)

if [[ -n "$MOTION_KEYS_FILE" ]]; then
    cmd+=("++manager_env.config.per_rank_motion_keys_file=${MOTION_KEYS_FILE}")
fi

if [[ "$MAX_UNIQUE_MOTIONS" != "all" ]]; then
    cmd+=("++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=${MAX_UNIQUE_MOTIONS}")
fi

if [[ "$DISABLE_EARLY_TERMINATIONS" == "1" ]]; then
    cmd+=(
        "++manager_env.terminations.anchor_pos=null"
        "++manager_env.terminations.anchor_ori_full=null"
        "++manager_env.terminations.ee_body_pos=null"
        "++manager_env.terminations.object_pos_deviation=null"
        "++manager_env.terminations.object_z_pos_deviation=null"
        "++manager_env.terminations.grasp_failure_after_contact=null"
        "++manager_env.terminations.hand_table_contact_termination=null"
    )
fi

if [[ "$RENDER" == "1" ]]; then
    cmd+=(
        "++manager_env.config.render_results=True"
        "++manager_env.config.save_rendering_dir=${RENDER_OUT}"
        "++manager_env.config.env_spacing=10.0"
        "~manager_env/recorders=empty"
        "+manager_env/recorders=render"
    )
fi

if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
fi

printf 'cd %q\n' "$SONIC_ROOT"
quote_command "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

cd "$SONIC_ROOT"
"${cmd[@]}"

python - "$DEBUG_LOG" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing token debug log: {path}")

records = json.loads(path.read_text())
if not records:
    raise SystemExit(f"empty token debug log: {path}")

actions = [record.get("actions") for record in records if record.get("actions") is not None]
if not actions:
    raise SystemExit(f"debug log contains no action records: {path}")

bad_dims = [len(action) for action in actions if len(action) != 66]
if bad_dims:
    raise SystemExit(f"expected 66-D meta-actions, saw dimensions: {sorted(set(bad_dims))}")

flat = [value for action in actions for value in action]
if any(not math.isfinite(float(value)) for value in flat):
    raise SystemExit("meta-actions contain NaN or Inf")

tokens = [value for action in actions for value in action[:64]]
hands = [value for action in actions for value in action[64:]]
token_abs_max = max(abs(float(value)) for value in tokens)
hand_abs_max = max(abs(float(value)) for value in hands)
token_rms = math.sqrt(sum(float(value) ** 2 for value in tokens) / len(tokens))

print(f"token_debug_records: {len(actions)}")
print(f"meta_action_dim: 66")
print(f"token_residual_dim: 64")
print(f"hand_primitive_dim: 2")
print(f"token_residual_abs_max: {token_abs_max:.6g}")
print(f"token_residual_rms: {token_rms:.6g}")
print(f"hand_primitive_abs_max: {hand_abs_max:.6g}")
if token_abs_max < 1e-8:
    print("WARNING: token residuals are numerically near zero; inspect metrics/render before trusting behavior.")
PY
