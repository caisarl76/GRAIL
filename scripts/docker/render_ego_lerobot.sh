#!/usr/bin/env bash
# Run the GRAIL ego-view LeRobot conversion entrypoint inside the standard
# GRAIL Docker container.

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/docker/render_ego_lerobot.sh --motion-lib PATH --output PATH [options]

Options:
  --render-only                Run ego-view rendering/planning instead of LeRobot export.
  --motion-lib PATH             GRAIL task motion-library root, for example data/pickup_table.
  --output PATH                 Output root for render jobs or LeRobot records.
  --ego-frame-root PATH         Existing rendered ego-frame root for offline LeRobot export.
  --task TEXT                   LeRobot task label.
  --token-labels PATH           Directory/file containing SONIC teacher motion-token labels.
  --allow-empty-token-labels    Permit missing teacher token labels during bring-up.
  --only-rendered               Export only motions that have a rendered ego-frame source.
  --require-all-rendered        Fail if any planned motion is missing a rendered ego source.
  --max-episodes N              Limit exported episodes after rendered-source filtering.
  --append-existing             Append to an existing LeRobot output instead of failing.
  --overwrite-existing          Replace an existing LeRobot output before writing.
  --resolution WIDTHxHEIGHT     Ego render resolution for --render-only.
  --camera-position-offset X Y Z
                               Camera position offset in the robot root frame for --render-only.
  --camera-target-offset X Y Z Camera look-at target offset in the robot root frame for --render-only.
  --camera-up-axis X Y Z       Camera up axis in the robot root frame for --render-only.
  --max-motions N               Limit ego render plan size for --render-only.
  --skip-existing               Skip ego videos that already exist for --render-only.
  --with-third-eye-comparison   Also render third-eye videos for the same motion keys and compose side-by-side diagnostics.
  --third-eye-output PATH       Third-eye intermediate video directory for comparison mode.
  --side-by-side-output PATH    Side-by-side diagnostic video directory for comparison mode.
  --num-shards N                Total render/export shards.
  --shard-index N               Current shard index.
  --write-manifests            Write planned ego render manifests for --render-only.
  --dry-run                     Plan jobs without rendering/exporting.
  --print-command               Print the Docker command instead of running it.
  -h, --help                    Show this help.

GPU placement:
  Use an RTX GPU host for Isaac Sim ego-view LeRobot rendering and sim checks.
  Use an H100 server for GR00T/VLA training after the LeRobot dataset exists.

Execution:
  On the host, this script enters the GRAIL Docker container via run_grail.sh.
  Inside an existing container, or with GRAIL_RENDER_EGO_DIRECT=1, it runs the
  Python entrypoint directly and does not call Docker.
  Direct/container mode uses 'conda run -n sonic' unless CONDA_DEFAULT_ENV is
  already sonic, conda is unavailable, or GRAIL_RENDER_EGO_USE_CONDA=0.
  Set GRAIL_SONIC_ENV to use a different conda environment name.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_grail.sh"
ORIGINAL_ARGS=("$@")

MOTION_LIB=""
OUTPUT=""
RENDER_ONLY=0
PRINT_COMMAND=0
DRY_RUN=0
RESOLUTION="640x480"
WITH_THIRD_EYE_COMPARISON=0
THIRD_EYE_OUTPUT=""
SIDE_BY_SIDE_OUTPUT=""
WRITE_MANIFESTS=0
EXPORT_ARGS=()
RENDER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --render-only)
            RENDER_ONLY=1
            shift
            ;;
        --with-third-eye-comparison)
            WITH_THIRD_EYE_COMPARISON=1
            shift
            ;;
        --third-eye-output)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            THIRD_EYE_OUTPUT="$2"
            shift 2
            ;;
        --side-by-side-output)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            SIDE_BY_SIDE_OUTPUT="$2"
            shift 2
            ;;
        --motion-lib)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            MOTION_LIB="$2"
            EXPORT_ARGS+=("--motion-lib" "$2")
            RENDER_ARGS+=("--motion_lib" "$2")
            shift 2
            ;;
        --output)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            OUTPUT="$2"
            EXPORT_ARGS+=("--output" "$2")
            RENDER_ARGS+=("--output_dir" "$2")
            shift 2
            ;;
        --num-shards)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            EXPORT_ARGS+=("--num-shards" "$2")
            RENDER_ARGS+=("--num_shards" "$2")
            shift 2
            ;;
        --shard-index)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            EXPORT_ARGS+=("--shard-index" "$2")
            RENDER_ARGS+=("--shard_index" "$2")
            shift 2
            ;;
        --ego-frame-root|--task|--token-labels)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            EXPORT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --resolution)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            RESOLUTION="$2"
            RENDER_ARGS+=("--resolution" "$2")
            shift 2
            ;;
        --camera-position-offset|--camera_position_offset)
            if [[ $# -lt 4 ]]; then
                echo "Missing three values for $1" >&2
                exit 2
            fi
            RENDER_ARGS+=("--camera_position_offset" "$2" "$3" "$4")
            shift 4
            ;;
        --camera-target-offset|--camera_target_offset)
            if [[ $# -lt 4 ]]; then
                echo "Missing three values for $1" >&2
                exit 2
            fi
            RENDER_ARGS+=("--camera_target_offset" "$2" "$3" "$4")
            shift 4
            ;;
        --camera-up-axis|--camera_up_axis)
            if [[ $# -lt 4 ]]; then
                echo "Missing three values for $1" >&2
                exit 2
            fi
            RENDER_ARGS+=("--camera_up_axis" "$2" "$3" "$4")
            shift 4
            ;;
        --max-motions)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            RENDER_ARGS+=("--max_motions" "$2")
            shift 2
            ;;
        --skip-existing|--skip_existing)
            RENDER_ARGS+=("--skip_existing")
            shift
            ;;
        --max-episodes)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            EXPORT_ARGS+=("--max-episodes" "$2")
            shift 2
            ;;
        --only-rendered|--require-all-rendered|--append-existing|--overwrite-existing)
            EXPORT_ARGS+=("$1")
            shift
            ;;
        --allow-empty-token-labels|--dry-run)
            if [[ "$1" == "--dry-run" ]]; then
                DRY_RUN=1
                EXPORT_ARGS+=("--dry-run")
                RENDER_ARGS+=("--dry_run")
            else
                EXPORT_ARGS+=("$1")
            fi
            shift
            ;;
        --write-manifest|--write-manifests)
            WRITE_MANIFESTS=1
            RENDER_ARGS+=("--write_manifests")
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

if [[ -z "${MOTION_LIB}" || -z "${OUTPUT}" ]]; then
    echo "--motion-lib and --output are required" >&2
    usage >&2
    exit 2
fi

if [[ "${WITH_THIRD_EYE_COMPARISON}" -eq 1 && "${RENDER_ONLY}" -ne 1 ]]; then
    echo "--with-third-eye-comparison requires --render-only" >&2
    exit 2
fi

if [[ "${RENDER_ONLY}" -eq 1 ]]; then
    COMMAND=(python -m grail.visualization.batch_render_ego "${RENDER_ARGS[@]}")
else
    COMMAND=(python -m grail.cli.export_ego_lerobot "${EXPORT_ARGS[@]}")
fi

RUN_DIRECT=0
if [[ "${GRAIL_RENDER_EGO_DIRECT:-0}" == "1" || -f "/.dockerenv" ]]; then
    RUN_DIRECT=1
fi

DIRECT_PREFIX=()
DIRECT_COMMAND=("${COMMAND[@]}")
if [[ "${RUN_DIRECT}" -eq 1 && "${GRAIL_RENDER_EGO_USE_CONDA:-1}" != "0" ]]; then
    SONIC_ENV="${GRAIL_SONIC_ENV:-sonic}"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${SONIC_ENV}" ]] && command -v conda >/dev/null 2>&1; then
        DIRECT_PREFIX=(conda run -n "${SONIC_ENV}" --no-capture-output)
        DIRECT_COMMAND=(conda run -n "${SONIC_ENV}" --no-capture-output "${COMMAND[@]}")
    fi
fi

print_shell_command() {
    printf "%q " "$@"
    printf "\n"
}

print_direct_command() {
    if [[ "${#DIRECT_PREFIX[@]}" -gt 0 ]]; then
        print_shell_command "${DIRECT_PREFIX[@]}" "$@"
    else
        print_shell_command "$@"
    fi
}

run_direct_command() {
    if [[ "${#DIRECT_PREFIX[@]}" -gt 0 ]]; then
        "${DIRECT_PREFIX[@]}" "$@"
    else
        "$@"
    fi
}

resolution_width() {
    printf "%s" "${RESOLUTION%x*}"
}

resolution_height() {
    printf "%s" "${RESOLUTION#*x}"
}

print_comparison_plan() {
    local key_file="${SIDE_BY_SIDE_OUTPUT%/}/motion_keys.txt"
    local shard_dir="<temp_third_eye_shard>"
    local motion_keys="<motion_keys_from_ego_render>"
    local width
    local height
    width="$(resolution_width)"
    height="$(resolution_height)"

    local comparison_extra_args=(--write_motion_keys "${key_file}")
    if [[ "${WRITE_MANIFESTS}" -eq 0 ]]; then
        comparison_extra_args+=(--write_manifests)
    fi

    print_direct_command python -m grail.visualization.batch_render_ego \
        "${RENDER_ARGS[@]}" \
        "${comparison_extra_args[@]}"
    print_direct_command python -m grail.visualization.prepare_vis_shard \
        --data_dir "${MOTION_LIB}" \
        --shard_dir "${shard_dir}" \
        --motion_keys "${motion_keys}" \
        --quat_convention xyzw
    print_direct_command python -u -m grail.visualization.batch_render_replay \
        --shard_dir "${shard_dir}" \
        --traj_dir "${shard_dir}/trajectories" \
        --object_usd_dir "${MOTION_LIB%/}/object_usd" \
        --output_dir "${THIRD_EYE_OUTPUT}" \
        --resolution "${RESOLUTION}" \
        --camera_offset 1.5 -1.5 1.0 \
        --camera_target 0.0 0.0 0.8 \
        --start_frame_skip 0 \
        --headless
    print_shell_command ffmpeg -y -loglevel error \
        -i "${OUTPUT%/}/<motion_key>.mp4" \
        -i "${THIRD_EYE_OUTPUT%/}/<motion_key>.mp4" \
        -filter_complex "[0:v]scale=${width}:${height},setsar=1[left];[1:v]scale=${width}:${height},setsar=1[right];[left][right]hstack=inputs=2[v]" \
        -map "[v]" \
        -an \
        -c:v libx264 \
        -preset veryfast \
        -crf 23 \
        "${SIDE_BY_SIDE_OUTPUT%/}/<motion_key>.mp4"
}

run_comparison_mode() {
    if [[ "${RESOLUTION}" != *x* ]]; then
        echo "--resolution must use WIDTHxHEIGHT for comparison mode" >&2
        exit 2
    fi

    THIRD_EYE_OUTPUT="${THIRD_EYE_OUTPUT:-${OUTPUT%/}_third_eye}"
    SIDE_BY_SIDE_OUTPUT="${SIDE_BY_SIDE_OUTPUT:-${OUTPUT%/}_side_by_side}"

    if [[ "${PRINT_COMMAND}" -eq 1 ]]; then
        print_comparison_plan
        exit 0
    fi

    mkdir -p "${THIRD_EYE_OUTPUT}" "${SIDE_BY_SIDE_OUTPUT}"

    local key_file="${SIDE_BY_SIDE_OUTPUT%/}/motion_keys.txt"
    local shard_dir
    shard_dir="$(mktemp -d "${TMPDIR:-/tmp}/grail_compare_shard_XXXXXX")"
    local width
    local height
    width="$(resolution_width)"
    height="$(resolution_height)"

    local ego_command=(
        python -m grail.visualization.batch_render_ego
        "${RENDER_ARGS[@]}"
        --write_motion_keys "${key_file}"
    )
    if [[ "${WRITE_MANIFESTS}" -eq 0 ]]; then
        ego_command+=(--write_manifests)
    fi

    if [[ "${DRY_RUN}" -ne 1 ]] && ! command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg is required for --with-third-eye-comparison. Install ffmpeg before running comparison renders." >&2
        exit 127
    fi

    echo "[1/3] Rendering ego-view videos and writing motion key list"
    run_direct_command "${ego_command[@]}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Dry run complete. Comparison rendering and side-by-side composition were not executed."
        exit 0
    fi

    if [[ ! -s "${key_file}" ]]; then
        echo "No motion keys were written: ${key_file}" >&2
        exit 1
    fi

    local motion_keys
    motion_keys="$(paste -sd, "${key_file}")"
    if [[ -z "${motion_keys}" ]]; then
        echo "No motion keys selected for comparison rendering." >&2
        exit 1
    fi

    echo "[2/3] Rendering third-eye videos for the same motion keys"
    run_direct_command python -m grail.visualization.prepare_vis_shard \
        --data_dir "${MOTION_LIB}" \
        --shard_dir "${shard_dir}" \
        --motion_keys "${motion_keys}" \
        --quat_convention xyzw
    run_direct_command python -u -m grail.visualization.batch_render_replay \
        --shard_dir "${shard_dir}" \
        --traj_dir "${shard_dir}/trajectories" \
        --object_usd_dir "${MOTION_LIB%/}/object_usd" \
        --output_dir "${THIRD_EYE_OUTPUT}" \
        --resolution "${RESOLUTION}" \
        --camera_offset 1.5 -1.5 1.0 \
        --camera_target 0.0 0.0 0.8 \
        --start_frame_skip 0 \
        --headless

    echo "[3/3] Composing side-by-side comparison videos"
    local count=0
    local motion_key
    while IFS= read -r motion_key; do
        [[ -z "${motion_key}" ]] && continue
        local ego_video="${OUTPUT%/}/${motion_key}.mp4"
        local third_eye_video="${THIRD_EYE_OUTPUT%/}/${motion_key}.mp4"
        local side_by_side_video="${SIDE_BY_SIDE_OUTPUT%/}/${motion_key}.mp4"
        if [[ ! -f "${ego_video}" ]]; then
            echo "Missing ego-view video: ${ego_video}" >&2
            exit 1
        fi
        if [[ ! -f "${third_eye_video}" ]]; then
            echo "Missing third-eye video: ${third_eye_video}" >&2
            exit 1
        fi
        ffmpeg -y -loglevel error \
            -i "${ego_video}" \
            -i "${third_eye_video}" \
            -filter_complex "[0:v]scale=${width}:${height},setsar=1[left];[1:v]scale=${width}:${height},setsar=1[right];[left][right]hstack=inputs=2[v]" \
            -map "[v]" \
            -an \
            -c:v libx264 \
            -preset veryfast \
            -crf 23 \
            "${side_by_side_video}"
        count=$((count + 1))
    done < "${key_file}"

    echo "Comparison render complete: ${count} side-by-side video(s) in ${SIDE_BY_SIDE_OUTPUT}"
}

if [[ "${WITH_THIRD_EYE_COMPARISON}" -eq 1 ]]; then
    if [[ "${RUN_DIRECT}" -ne 1 ]]; then
        DOCKER_COMMAND=(bash "${RUNNER}" -- bash scripts/docker/render_ego_lerobot.sh "${ORIGINAL_ARGS[@]}")
        if [[ "${PRINT_COMMAND}" -eq 1 ]]; then
            print_shell_command "${DOCKER_COMMAND[@]}"
            exit 0
        fi
        exec "${DOCKER_COMMAND[@]}"
    fi
    run_comparison_mode
fi

if [[ "${PRINT_COMMAND}" -eq 1 ]]; then
    if [[ "${RUN_DIRECT}" -eq 1 ]]; then
        printf "%q " "${DIRECT_COMMAND[@]}"
    else
        DOCKER_COMMAND=(bash "${RUNNER}" -- "${COMMAND[@]}")
        printf "%q " "${DOCKER_COMMAND[@]}"
    fi
    printf "\n"
    exit 0
fi

if [[ "${RUN_DIRECT}" -eq 1 ]]; then
    exec "${DIRECT_COMMAND[@]}"
fi

DOCKER_COMMAND=(bash "${RUNNER}" -- "${COMMAND[@]}")
exec "${DOCKER_COMMAND[@]}"
