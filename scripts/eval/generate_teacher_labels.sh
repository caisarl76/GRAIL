#!/usr/bin/env bash
set -euo pipefail

# Batched stage-1 teacher-label generation for the GRAIL -> LeRobot distillation
# dataset. For each motion shard (rank), this runs one IsaacLab eval that labels
# --num-envs motions per launch (with early terminations disabled so every motion
# rolls to its own time-out), then exports per-motion 66-D SONIC teacher labels.
#
# The eval rollout is delegated to scripts/eval/test_sonic_pnp_checkpoint.sh so the
# Hydra overrides stay defined in one place. Export uses --truncate-at-done so a
# short motion's post-time-out replay is dropped, and --resample-to-motion-lib so
# the ~50 Hz teacher stream is downsampled to the GRAIL ~25 Hz frame count.

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/generate_teacher_labels.sh --task pnp_ground --data-dir <motion-lib> \
    --label-root <teacher-labels-out> [options]

Options:
  --task pnp_table|pnp_ground   Task/checkpoint bundle to evaluate.
  --data-dir PATH               Motion library root with robot/, objects/, object_usd/.
  --bps-dir PATH                BPS directory. Defaults to <data-dir>/bps.
  --output-root PATH            Root for per-rank eval output (token_debug.json, metrics).
                                Each rank writes <output-root>/rank_XXXX/<task>/.
  --label-root PATH             Output directory for <motion_key>.npy teacher labels.
  --gpu ID                      CUDA_VISIBLE_DEVICES for the eval process.
  --num-envs N                  Motions labeled per Isaac launch (one batch). Default: 16.
                                Per-object USD collision meshes are static, so a single
                                eval batch is the hard ceiling: each shard must hold
                                <= num_envs motions (multi-batch shards are unreliable).
  --num-shards N                Total motion shards (world size). Default: auto =
                                ceil(motion_count / num_envs), i.e. one batch per shard.
                                An explicit value is rejected if it would exceed num_envs
                                motions per shard.
  --shard-start N               First rank to run (inclusive). Default: 0.
  --shard-end N                 One past the last rank to run (exclusive). Default: --num-shards.
  --skip-existing               Skip ranks that already wrote a .teacher_labels.done marker.
  --dry-run                     Print per-rank eval + export commands without running.
  -h, --help                    Show this help.

Outputs:
  <label-root>/<motion_key>.npy            (frames, 66) teacher labels
  <output-root>/rank_XXXX/<task>/token_debug.json
  <output-root>/rank_XXXX/<task>/.teacher_labels.done   (success marker)
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/test_sonic_pnp_checkpoint.sh"

TASK="pnp_ground"
DATA_DIR=""
BPS_DIR=""
OUTPUT_ROOT=""
LABEL_ROOT=""
GPU_ID=""
NUM_ENVS="16"
NUM_SHARDS=""
SHARD_START="0"
SHARD_END=""
SKIP_EXISTING="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --bps-dir) BPS_DIR="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --label-root) LABEL_ROOT="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        --shard-start) SHARD_START="$2"; shift 2 ;;
        --shard-end) SHARD_END="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="1"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "--data-dir is required" >&2; usage >&2; exit 2
fi
if [[ -z "$LABEL_ROOT" ]]; then
    echo "--label-root is required" >&2; usage >&2; exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ -d "/workspace/grail" ]]; then
        OUTPUT_ROOT="/workspace/grail/data/${TASK}_teacher/sonic_eval_shards"
    else
        OUTPUT_ROOT="${REPO_ROOT}/data/${TASK}_teacher/sonic_eval_shards"
    fi
fi

# Per-object USD collision meshes are assigned once at scene creation, so each eval
# process may only run a single batch: motions-per-shard must be <= num_envs. Derive
# world_size from the motion count to guarantee one batch per shard.
NUM_MOTIONS="$(find "$DATA_DIR/robot" -maxdepth 1 -name '*.pkl' | wc -l)"
if [[ "$NUM_MOTIONS" -eq 0 ]]; then
    echo "No motions found under $DATA_DIR/robot/*.pkl" >&2
    exit 2
fi

if [[ -z "$NUM_SHARDS" ]]; then
    NUM_SHARDS=$(((NUM_MOTIONS + NUM_ENVS - 1) / NUM_ENVS))
fi

per_shard=$(((NUM_MOTIONS + NUM_SHARDS - 1) / NUM_SHARDS))
if [[ "$per_shard" -gt "$NUM_ENVS" ]]; then
    echo "Refusing: ${NUM_MOTIONS} motions / ${NUM_SHARDS} shards = ${per_shard} motions per shard > ${NUM_ENVS} envs." >&2
    echo "Multi-batch shards are unreliable (static per-object USD meshes). Raise --num-shards or --num-envs." >&2
    exit 2
fi

echo "[plan] ${NUM_MOTIONS} motions, ${NUM_ENVS} envs -> ${NUM_SHARDS} shards (<= ${per_shard} motions/shard, one batch each)"

if [[ -z "$SHARD_END" ]]; then
    SHARD_END="$NUM_SHARDS"
fi

for rank in $(seq "$SHARD_START" $(("$SHARD_END" - 1))); do
    pad="$(printf '%04d' "$rank")"
    rank_out="${OUTPUT_ROOT}/rank_${pad}"
    debug_log="${rank_out}/${TASK}/token_debug.json"
    done_marker="${rank_out}/${TASK}/.teacher_labels.done"

    if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
        echo "[skip] rank ${rank}: ${done_marker} exists"
        continue
    fi

    echo "===== rank ${rank}/${NUM_SHARDS} ====="

    eval_cmd=(
        bash "$EVAL_SCRIPT"
        --task "$TASK"
        --data-dir "$DATA_DIR"
        --output-root "$rank_out"
        --num-envs "$NUM_ENVS"
        --max-unique-motions all
        --motion-shard-world-size "$NUM_SHARDS"
        --motion-shard-rank "$rank"
        --disable-early-terminations
    )
    [[ -n "$BPS_DIR" ]] && eval_cmd+=(--bps-dir "$BPS_DIR")
    [[ -n "$GPU_ID" ]] && eval_cmd+=(--gpu "$GPU_ID")
    [[ "$DRY_RUN" == "1" ]] && eval_cmd+=(--dry-run)

    export_cmd=(
        env "PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/imports/SONIC"
        python -m grail.cli.export_teacher_labels
        --debug-log "$debug_log"
        --output "$LABEL_ROOT"
        --motion-lib "$DATA_DIR"
        --resample-to-motion-lib
        --truncate-at-done
        --skip-unknown-motions
        --dedupe-overflow-envs
        --overwrite
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        "${eval_cmd[@]}"
        printf '%q ' "${export_cmd[@]}"; printf '\n'
        continue
    fi

    "${eval_cmd[@]}"
    ( cd "$REPO_ROOT" && "${export_cmd[@]}" )
    mkdir -p "$(dirname "$done_marker")"
    touch "$done_marker"
    echo "[done] rank ${rank}: labels -> ${LABEL_ROOT}"
done
