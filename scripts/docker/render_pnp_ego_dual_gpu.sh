#!/usr/bin/env bash
# Render all GRAIL pick-up ego-view videos for the SONIC pnp_table/pnp_ground tasks.
#
# Run this from inside the GRAIL container so CUDA_VISIBLE_DEVICES pins each
# Isaac Sim render process to the intended physical GPU.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/docker/render_pnp_ego_dual_gpu.sh --data-root PATH [options]

Options:
  --data-root PATH              Directory containing pickup_table/ and pickup_ground/.
  --output-root PATH            Output root for ego videos (default: data/ego_frames).
  --log-root PATH               Log root for render stdout/stderr (default: logs/ego_render).
  --resolution WIDTHxHEIGHT     Ego render resolution (default: 640x480).
  --table-gpu N                 Physical GPU for pnp_table / pickup_table (default: 0).
  --ground-gpu N                Physical GPU for pnp_ground / pickup_ground (default: 1).
  --camera-position-offset X Y Z
                               Root-frame camera position (default: 0.25 0.0 0.55).
  --camera-target-offset X Y Z Root-frame look-at target (default: 0.846 0.0 -0.301).
  --dry-run                     Plan jobs without rendering.
  --print-command               Print the two launch commands without running.
  -h, --help                    Show this help.

Outputs:
  <output-root>/pnp_table/*.mp4
  <output-root>/pnp_ground/*.mp4
  <output-root>/pnp_table/*.json
  <output-root>/pnp_ground/*.json

Notes:
  GRAIL data directories are usually named pickup_table and pickup_ground.
  SONIC checkpoints/configs are usually named pnp_table and pnp_ground.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RENDER_SCRIPT="${SCRIPT_DIR}/render_ego_lerobot.sh"

DATA_ROOT=""
OUTPUT_ROOT="data/ego_frames"
LOG_ROOT="logs/ego_render"
RESOLUTION="640x480"
TABLE_GPU="0"
GROUND_GPU="1"
DRY_RUN=0
PRINT_COMMAND=0
CAMERA_POSITION=(0.25 0.0 0.55)
CAMERA_TARGET=(0.846 0.0 -0.301)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-root)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --log-root)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        --resolution)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            RESOLUTION="$2"
            shift 2
            ;;
        --table-gpu)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            TABLE_GPU="$2"
            shift 2
            ;;
        --ground-gpu)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            GROUND_GPU="$2"
            shift 2
            ;;
        --camera-position-offset|--camera_position_offset)
            if [[ $# -lt 4 ]]; then
                echo "Missing three values for $1" >&2
                exit 2
            fi
            CAMERA_POSITION=("$2" "$3" "$4")
            shift 4
            ;;
        --camera-target-offset|--camera_target_offset)
            if [[ $# -lt 4 ]]; then
                echo "Missing three values for $1" >&2
                exit 2
            fi
            CAMERA_TARGET=("$2" "$3" "$4")
            shift 4
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --print-command)
            PRINT_COMMAND=1
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

if [[ -z "${DATA_ROOT}" ]]; then
    echo "--data-root is required" >&2
    usage >&2
    exit 2
fi

if [[ "${RESOLUTION}" != *x* ]]; then
    echo "--resolution must use WIDTHxHEIGHT" >&2
    exit 2
fi

resolve_motion_lib() {
    local label="$1"
    local grail_dir="$2"
    local preferred="${DATA_ROOT%/}/${grail_dir}"
    local fallback="${DATA_ROOT%/}/${label}"

    if [[ -d "${preferred}/robot" || "${PRINT_COMMAND}" -eq 1 ]]; then
        printf "%s" "${preferred}"
        return
    fi
    if [[ -d "${fallback}/robot" ]]; then
        printf "%s" "${fallback}"
        return
    fi

    echo "Missing motion library for ${label}. Expected ${preferred}/robot or ${fallback}/robot" >&2
    echo "Download the missing GRAIL task data before running the dual-GPU launcher." >&2
    exit 1
}

print_job_command() {
    local gpu="$1"
    shift
    printf "CUDA_VISIBLE_DEVICES=%q GRAIL_RENDER_EGO_DIRECT=1 " "${gpu}"
    printf "%q " "$@"
    printf "\n"
}

launch_task() {
    local label="$1"
    local motion_lib="$2"
    local gpu="$3"
    local output_dir="${OUTPUT_ROOT%/}/${label}"
    local log_path="${LOG_ROOT%/}/${label}.gpu${gpu}.log"
    local command=(
        bash "${RENDER_SCRIPT}"
        --render-only
        --motion-lib "${motion_lib}"
        --output "${output_dir}"
        --resolution "${RESOLUTION}"
        --camera-position-offset "${CAMERA_POSITION[@]}"
        --camera-target-offset "${CAMERA_TARGET[@]}"
        --write-manifests
        --skip-existing
    )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        command+=(--dry-run)
    fi

    if [[ "${PRINT_COMMAND}" -eq 1 ]]; then
        print_job_command "${gpu}" "${command[@]}"
        return
    fi

    mkdir -p "${output_dir}" "$(dirname "${log_path}")"

    (
        cd "${REPO_ROOT}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GRAIL_RENDER_EGO_DIRECT=1
        "${command[@]}"
    ) > "${log_path}" 2>&1 &

    local pid=$!
    PIDS+=("${pid}")
    LABELS+=("${label}")
    LOGS+=("${log_path}")
    echo "Started ${label} on GPU ${gpu}: pid=${pid}, log=${log_path}"
}

if [[ "${PRINT_COMMAND}" -ne 1 ]]; then
    if [[ ! -f "/.dockerenv" && "${GRAIL_RENDER_EGO_DIRECT:-0}" != "1" ]]; then
        echo "Run this script inside the GRAIL container for reliable GPU pinning." >&2
        echo "Enter the container with: bash scripts/docker/run_grail.sh" >&2
        exit 2
    fi
fi

PIDS=()
LABELS=()
LOGS=()

PNP_TABLE_MOTION_LIB="$(resolve_motion_lib "pnp_table" "pickup_table")"
PNP_GROUND_MOTION_LIB="$(resolve_motion_lib "pnp_ground" "pickup_ground")"

launch_task "pnp_table" "${PNP_TABLE_MOTION_LIB}" "${TABLE_GPU}"
launch_task "pnp_ground" "${PNP_GROUND_MOTION_LIB}" "${GROUND_GPU}"

if [[ "${PRINT_COMMAND}" -eq 1 ]]; then
    exit 0
fi

status=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "Completed ${LABELS[$i]}: ${LOGS[$i]}"
    else
        echo "FAILED ${LABELS[$i]}: ${LOGS[$i]}" >&2
        status=1
    fi
done

exit "${status}"
