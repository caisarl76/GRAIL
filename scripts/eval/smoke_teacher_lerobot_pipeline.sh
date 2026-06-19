#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke for the GRAIL -> SONIC teacher-label -> LeRobot path.
# Intended to run inside the grail-sonic container from /workspace/grail.

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/smoke_teacher_lerobot_pipeline.sh --data-dir <motion-lib> --bps-dir <bps-dir> [options]

Options:
  --task pnp_ground|pnp_table     Task/checkpoint bundle. Default: pnp_ground.
  --data-dir PATH                 Motion library root with robot/, objects/, object_usd/.
  --bps-dir PATH                  BPS directory for SONIC eval.
  --output-root PATH              Root for smoke artifacts. Default: data/teacher_verify/teacher_lerobot_smoke_<timestamp>.
  --gpu ID                        CUDA_VISIBLE_DEVICES for teacher-label eval. Default: 0.
  --num-episodes N                Number of episodes to export/check. Default: 10.
  --num-envs N                    Motions per teacher-label eval launch. Default: 5.
  --resolution WIDTHxHEIGHT       Ego render resolution. Default: 640x480.
                                  Must match SONIC VLA LeRobot image schema.
  --object-init-z-offset VALUE    Extra pnp_ground object init offset. Default: -0.13. Use "none" to disable.
  --batch-size N                  Smoke-training batch size. Default: 2.
  --num-batches N                 Smoke-training batches. Default: 2.
  --device DEVICE                 Smoke-training torch device. Default: cpu.
  --dry-run                       Print/plan commands without GPU render/eval/training execution.
  -h, --help                      Show this help.

Outputs under --output-root:
  eval_shards/                    Per-rank teacher eval debug logs and metrics.
  teacher_labels/*.npy            (frames, 66) [motion_token, hand_primitive].
  ego_frames/*.mp4                Rendered ego-view videos.
  lerobot/                        Exported local LeRobot dataset.
EOF
}

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

activate_conda_if_needed() {
    local env_name="${GRAIL_SONIC_ENV:-sonic}"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$env_name"
    fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

TASK="pnp_ground"
DATA_DIR=""
BPS_DIR=""
OUTPUT_ROOT=""
GPU_ID="0"
NUM_EPISODES="10"
NUM_ENVS="5"
RESOLUTION="640x480"
OBJECT_INIT_Z_OFFSET="-0.13"
BATCH_SIZE="2"
NUM_BATCHES="2"
DEVICE="cpu"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --bps-dir) BPS_DIR="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --object-init-z-offset) OBJECT_INIT_Z_OFFSET="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-batches) NUM_BATCHES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "--data-dir is required" >&2; usage >&2; exit 2
fi
if [[ -z "$BPS_DIR" ]]; then
    echo "--bps-dir is required" >&2; usage >&2; exit 2
fi
if [[ ! -d "$DATA_DIR/robot" ]]; then
    echo "Missing motion robot dir: $DATA_DIR/robot" >&2
    exit 2
fi
if [[ ! "$NUM_EPISODES" =~ ^[0-9]+$ || "$NUM_EPISODES" -le 0 ]]; then
    echo "--num-episodes must be a positive integer" >&2
    exit 2
fi
if [[ ! "$NUM_ENVS" =~ ^[0-9]+$ || "$NUM_ENVS" -le 0 ]]; then
    echo "--num-envs must be a positive integer" >&2
    exit 2
fi
if [[ "$RESOLUTION" != "640x480" ]]; then
    echo "--resolution must be 640x480 for LeRobot export; SONIC VLA expects observation.images.ego_view shape (480, 640, 3)" >&2
    exit 2
fi
NUM_MOTIONS="$(find "$DATA_DIR/robot" -maxdepth 1 -name '*.pkl' | wc -l)"
if [[ "$NUM_MOTIONS" -eq 0 ]]; then
    echo "No motions found under $DATA_DIR/robot/*.pkl" >&2
    exit 2
fi
if [[ "$NUM_EPISODES" -gt "$NUM_MOTIONS" ]]; then
    echo "--num-episodes (${NUM_EPISODES}) exceeds motion count (${NUM_MOTIONS})" >&2
    exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ -d "/workspace/grail" ]]; then
        OUTPUT_ROOT="/workspace/grail/data/teacher_verify/teacher_lerobot_smoke_$(date +%Y%m%d_%H%M%S)"
    else
        OUTPUT_ROOT="${REPO_ROOT}/data/teacher_verify/teacher_lerobot_smoke_$(date +%Y%m%d_%H%M%S)"
    fi
fi

LABEL_ROOT="${OUTPUT_ROOT%/}/teacher_labels"
EVAL_ROOT="${OUTPUT_ROOT%/}/eval_shards"
EGO_ROOT="${OUTPUT_ROOT%/}/ego_frames"
LEROBOT_ROOT="${OUTPUT_ROOT%/}/lerobot"
REQUESTED_KEYS_FILE="${OUTPUT_ROOT%/}/requested_motion_keys.txt"

WORLD_SIZE=$(((NUM_EPISODES + NUM_ENVS - 1) / NUM_ENVS))
MOTIONS_PER_SHARD=$(((NUM_EPISODES + WORLD_SIZE - 1) / WORLD_SIZE))
LABEL_SHARDS="$WORLD_SIZE"
PLANNED_LABELS="$NUM_EPISODES"

mkdir -p "$OUTPUT_ROOT"
find "$DATA_DIR/robot" -maxdepth 1 -name '*.pkl' -printf '%f\n' \
    | sed 's/\.pkl$//' \
    | sort \
    | head -n "$NUM_EPISODES" > "$REQUESTED_KEYS_FILE"
cat > "${OUTPUT_ROOT%/}/paths.env" <<EOF
OUTPUT_ROOT=${OUTPUT_ROOT}
LABEL_ROOT=${LABEL_ROOT}
EVAL_ROOT=${EVAL_ROOT}
EGO_ROOT=${EGO_ROOT}
LEROBOT_ROOT=${LEROBOT_ROOT}
DATA_DIR=${DATA_DIR}
BPS_DIR=${BPS_DIR}
REQUESTED_KEYS_FILE=${REQUESTED_KEYS_FILE}
EOF

echo "[plan] task=${TASK} episodes=${NUM_EPISODES} label_envs=${NUM_ENVS} world_size=${WORLD_SIZE} label_shards=${LABEL_SHARDS} planned_labels~=${PLANNED_LABELS}"
if [[ "$PLANNED_LABELS" -ne "$NUM_EPISODES" ]]; then
    echo "[note] label generation will produce ${PLANNED_LABELS} labels; LeRobot export is capped to ${NUM_EPISODES} episodes."
fi

cd "$REPO_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then
    activate_conda_if_needed
fi

label_cmd=(
    bash scripts/eval/generate_teacher_labels.sh
    --task "$TASK"
    --data-dir "$DATA_DIR"
    --bps-dir "$BPS_DIR"
    --output-root "$EVAL_ROOT"
    --label-root "$LABEL_ROOT"
    --motion-keys-file "$REQUESTED_KEYS_FILE"
    --gpu "$GPU_ID"
    --num-envs "$NUM_ENVS"
    --num-shards "$WORLD_SIZE"
    --shard-start 0
    --shard-end "$LABEL_SHARDS"
)
if [[ "$OBJECT_INIT_Z_OFFSET" != "none" ]]; then
    label_cmd+=("++manager_env.commands.motion.object_init_z_offset=${OBJECT_INIT_Z_OFFSET}")
fi

echo "[1/4] teacher labels"
if [[ "$DRY_RUN" == "1" ]]; then
    "${label_cmd[@]}" --dry-run
else
    "${label_cmd[@]}"
    python - "$LABEL_ROOT" "$NUM_EPISODES" "$REQUESTED_KEYS_FILE" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
expected = int(sys.argv[2])
requested = [line.strip() for line in Path(sys.argv[3]).read_text().splitlines() if line.strip()]
files = sorted(root.glob("*.npy"))
print("teacher_label_files:", len(files))
if len(files) < expected:
    raise SystemExit(f"expected at least {expected} teacher label files, got {len(files)}")
missing = [key for key in requested[:expected] if not (root / f"{key}.npy").is_file()]
if missing:
    raise SystemExit(f"missing requested teacher labels: {missing[:10]}")
for key in requested[:expected]:
    path = root / f"{key}.npy"
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 66:
        raise SystemExit(f"bad teacher label shape for {path}: {arr.shape}")
    if not np.isfinite(arr).all():
        raise SystemExit(f"non-finite teacher labels in {path}")
sample = root / f"{requested[0]}.npy"
print("teacher_label_sample:", sample.name, np.load(sample).shape)
PY
fi

render_cmd=(
    env GRAIL_RENDER_EGO_DIRECT=1 GRAIL_RENDER_EGO_USE_CONDA=0
    bash scripts/docker/render_ego_lerobot.sh
    --render-only
    --motion-lib "$DATA_DIR"
    --output "$EGO_ROOT"
    --max-motions "$NUM_EPISODES"
    --resolution "$RESOLUTION"
    --skip-existing
    --write-manifests
)

echo "[2/4] ego render"
if [[ "$DRY_RUN" == "1" ]]; then
    "${render_cmd[@]}" --dry-run --print-command
else
    "${render_cmd[@]}"
    rendered_count="$(find "$EGO_ROOT" -maxdepth 1 -name '*.mp4' | wc -l)"
    echo "ego_render_videos: ${rendered_count}"
    if [[ "$rendered_count" -lt "$NUM_EPISODES" ]]; then
        echo "expected at least ${NUM_EPISODES} ego render videos, got ${rendered_count}" >&2
        exit 1
    fi
fi

export_cmd=(
    env GRAIL_RENDER_EGO_DIRECT=1 GRAIL_RENDER_EGO_USE_CONDA=0
    bash scripts/docker/render_ego_lerobot.sh
    --motion-lib "$DATA_DIR"
    --output "$LEROBOT_ROOT"
    --ego-frame-root "$EGO_ROOT"
    --token-labels "$LABEL_ROOT"
    --only-rendered
    --max-episodes "$NUM_EPISODES"
    --overwrite-existing
)

echo "[3/4] LeRobot export"
if [[ "$DRY_RUN" == "1" ]]; then
    "${export_cmd[@]}" --dry-run --print-command
else
    "${export_cmd[@]}"
fi

train_cmd=(
    python -m grail.cli.smoke_lerobot_training
    --root "$LEROBOT_ROOT"
    --repo-id tmp/grail_vla_smoke
    --batch-size "$BATCH_SIZE"
    --num-batches "$NUM_BATCHES"
    --device "$DEVICE"
)

echo "[4/4] training smoke"
if [[ "$DRY_RUN" == "1" ]]; then
    quote_cmd "${train_cmd[@]}"
else
    "${train_cmd[@]}"
fi

echo "[done] output_root=${OUTPUT_ROOT}"
