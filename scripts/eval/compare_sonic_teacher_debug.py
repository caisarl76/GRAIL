#!/usr/bin/env python3
"""Compare SONIC teacher debug logs with reference motion data.

This script has two checks:
1. rollout state vs. reference state already recorded in token_debug.json;
2. final applied body action vs. the next reference frame in a selected space.
   By default this compares in Isaac's normalized joint-position action space,
   using action_offset_body/action_scale_body recorded by the debug logger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]

G1_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_HAND_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

HAND_PRIMITIVE_NAMES = ["left_hand_action", "right_hand_action"]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records") or payload.get("steps") or payload.get("debug")
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a JSON list of debug records")
    return [rec for rec in records if isinstance(rec, dict)]


def _load_single_motion(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not load to a dict")

    if "dof" in payload:
        return payload

    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict) and "dof" in only_value:
            return only_value

    raise ValueError(f"{path} does not contain a recognizable motion dict with 'dof'")


def _motion_path(motion_dir: Path, motion_key: str) -> Path:
    direct = motion_dir / f"{motion_key}.pkl"
    if direct.exists():
        return direct
    matches = sorted(motion_dir.rglob(f"{motion_key}.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {motion_key}.pkl under {motion_dir}")


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def state_divergence(
    records: list[dict[str, Any]],
    *,
    env_index: int,
    root_threshold: float,
    joint_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    first_bad = None
    for rec in records:
        if int(rec.get("env_index", -1)) != env_index:
            continue
        required = ("root_pos", "ref_root_pos", "joint_pos", "ref_joint_pos")
        if any(key not in rec for key in required):
            continue

        root_err = _l2(
            np.asarray(rec["root_pos"], dtype=np.float64),
            np.asarray(rec["ref_root_pos"], dtype=np.float64),
        )
        joint_err = _l2(
            np.asarray(rec["joint_pos"], dtype=np.float64),
            np.asarray(rec["ref_joint_pos"], dtype=np.float64),
        )
        row = {
            "step": int(rec.get("step", -1)),
            "frame_index": int(rec.get("frame_index", -1)),
            "motion_key": str(rec.get("motion_key", "")),
            "root_l2": root_err,
            "joint_l2": joint_err,
            "reward": rec.get("reward"),
            "done": bool(rec.get("done", False)),
        }
        rows.append(row)
        if first_bad is None and (root_err > root_threshold or joint_err > joint_threshold):
            first_bad = row

    return rows, first_bad


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def motion_file_dof_target(
    raw_dof: np.ndarray,
    *,
    motion_dof_order: str,
) -> np.ndarray:
    """Return a motion-file DOF vector in eval/action body-joint order."""

    raw_dof = np.asarray(raw_dof, dtype=np.float64)
    if motion_dof_order == "isaaclab":
        return raw_dof.copy()
    if motion_dof_order != "mujoco":
        raise ValueError(f"Unsupported motion DOF order: {motion_dof_order}")
    if raw_dof.shape[0] < len(G1_MUJOCO_TO_ISAACLAB_DOF):
        return raw_dof.copy()
    indices = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
    return raw_dof[indices]


def _raw_motion_frame(
    motion: dict[str, Any],
    *,
    target_frame: int,
    motion_timebase: str,
    eval_fps: float,
    motion_file_fps: float | None,
) -> float | None:
    if motion_timebase == "raw-frame":
        return float(target_frame)
    if motion_timebase != "eval-time":
        raise ValueError(f"Unsupported motion timebase: {motion_timebase}")

    raw_fps = motion_file_fps
    if raw_fps is None:
        raw_fps = float(motion.get("fps") or 0.0)
    if raw_fps <= 0.0:
        raise ValueError(
            "Motion-file fps is missing; pass --motion-file-fps or use "
            "--motion-timebase raw-frame"
        )
    if eval_fps <= 0.0:
        raise ValueError("--eval-fps must be positive")
    return target_frame * raw_fps / eval_fps


def motion_file_array_at_eval_frame(
    motion: dict[str, Any],
    key: str,
    *,
    target_frame: int,
    motion_timebase: str,
    eval_fps: float,
    motion_file_fps: float | None,
    interpolation: str,
) -> np.ndarray | float | None:
    if key not in motion:
        return None
    values = np.asarray(motion[key], dtype=np.float64)
    raw_frame = _raw_motion_frame(
        motion,
        target_frame=target_frame,
        motion_timebase=motion_timebase,
        eval_fps=eval_fps,
        motion_file_fps=motion_file_fps,
    )
    if raw_frame is None or raw_frame < 0.0 or raw_frame > len(values) - 1:
        return None
    if interpolation == "nearest":
        return values[int(np.round(raw_frame))]
    if interpolation != "linear":
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    lo = int(np.floor(raw_frame))
    hi = min(lo + 1, len(values) - 1)
    alpha = raw_frame - lo
    return (1.0 - alpha) * values[lo] + alpha * values[hi]


def motion_file_dof_at_eval_frame(
    motion: dict[str, Any],
    *,
    target_frame: int,
    motion_dof_order: str,
    motion_timebase: str,
    eval_fps: float,
    motion_file_fps: float | None,
) -> np.ndarray | None:
    raw_target = motion_file_array_at_eval_frame(
        motion,
        "dof",
        target_frame=target_frame,
        motion_timebase=motion_timebase,
        eval_fps=eval_fps,
        motion_file_fps=motion_file_fps,
        interpolation="linear",
    )
    if raw_target is None:
        return None
    return motion_file_dof_target(raw_target, motion_dof_order=motion_dof_order)


def per_joint_error_summary(
    rows: list[dict[str, Any]],
    *,
    joint_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    errors = np.stack([np.asarray(row["joint_error"], dtype=np.float64) for row in rows])
    if errors.ndim != 2:
        raise ValueError("joint_error rows must be 1-D vectors")

    names = joint_names or G1_BODY_JOINT_NAMES
    summary: list[dict[str, Any]] = []
    for joint_index in range(errors.shape[1]):
        joint_errors = errors[:, joint_index]
        mse = float(np.mean(np.square(joint_errors)))
        mae = float(np.mean(np.abs(joint_errors)))
        summary.append(
            {
                "joint_index": joint_index,
                "joint_name": names[joint_index]
                if joint_index < len(names)
                else f"joint_{joint_index:02d}",
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": mae,
                "max_abs": float(np.max(np.abs(joint_errors))),
                "count": int(joint_errors.shape[0]),
            }
        )
    return summary


def _error_row(
    *,
    rec: dict[str, Any],
    target_frame: int,
    motion_key: str,
    compare_space: str,
    action_source: str,
    target_source: str,
    action: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    joint_error = action - target
    abs_err = np.abs(joint_error)
    return {
        "step": int(rec.get("step", -1)),
        "frame_index": int(rec.get("frame_index", -1)),
        "target_frame": target_frame,
        "motion_key": motion_key,
        "compare_space": compare_space,
        "action_source": action_source,
        "target_source": target_source,
        "action_l2": _l2(action, target),
        "action_mse": float(np.mean(np.square(joint_error))),
        "action_mean_abs": float(abs_err.mean()),
        "action_max_abs": float(abs_err.max()),
        "joint_error": joint_error,
        "done": bool(rec.get("done", False)),
    }


def _records_by_frame(
    records: list[dict[str, Any]],
    *,
    env_index: int,
    include_done: bool,
) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for rec in records:
        if int(rec.get("env_index", -1)) != env_index:
            continue
        if not include_done and bool(rec.get("done", False)):
            continue
        if "frame_index" not in rec:
            continue
        by_frame[int(rec["frame_index"])] = rec
    return by_frame


def action_comparison(
    records: list[dict[str, Any]],
    *,
    motion_dir: Path | None,
    env_index: int,
    frame_offset: int,
    include_done: bool,
    action_threshold: float,
    compare_space: str,
    action_source: str,
    target_source: str,
    motion_dof_order: str,
    motion_timebase: str,
    eval_fps: float,
    motion_file_fps: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    first_bad = None
    debug_ref_by_frame = _records_by_frame(
        records, env_index=env_index, include_done=include_done
    )
    action_key = "env_action_body"
    joint_target_key = "env_action_body_joint_target"
    if action_source == "zero":
        action_key = "zero_residual_body_action"
        joint_target_key = "zero_residual_body_joint_target"
    elif action_source == "forced":
        action_key = "forced_reference_body_action"
        joint_target_key = "forced_reference_body_joint_target"
    elif action_source == "motion":
        action_key = "forced_motion_body_action"
        joint_target_key = "forced_motion_body_joint_target"
    elif action_source != "actor":
        raise ValueError(f"Unsupported action source: {action_source}")

    for rec in records:
        if int(rec.get("env_index", -1)) != env_index:
            continue
        if not include_done and bool(rec.get("done", False)):
            continue
        if action_key not in rec:
            continue
        if compare_space == "action" and (
            "action_offset_body" not in rec or "action_scale_body" not in rec
        ):
            continue
        if compare_space == "dof" and (
            joint_target_key not in rec
            and ("action_offset_body" not in rec or "action_scale_body" not in rec)
        ):
            continue

        target_frame = int(rec.get("frame_index", -1)) + frame_offset
        motion_key = str(rec.get("motion_key", ""))

        env_action = np.asarray(rec[action_key], dtype=np.float64)
        if target_source == "motion-file":
            if motion_dir is None:
                raise ValueError("--motion-dir is required when --target-source=motion-file")
            if not motion_key:
                continue
            if motion_key not in cache:
                cache[motion_key] = _load_single_motion(_motion_path(motion_dir, motion_key))
            target_dof = motion_file_dof_at_eval_frame(
                cache[motion_key],
                target_frame=target_frame,
                motion_dof_order=motion_dof_order,
                motion_timebase=motion_timebase,
                eval_fps=eval_fps,
                motion_file_fps=motion_file_fps,
            )
            if target_dof is None:
                continue
            target_dof = target_dof[: env_action.shape[0]]
        elif target_source == "debug-ref":
            target_rec = debug_ref_by_frame.get(target_frame)
            if target_rec is None or "ref_joint_pos" not in target_rec:
                continue
            target_dof = np.asarray(target_rec["ref_joint_pos"], dtype=np.float64)[
                : env_action.shape[0]
            ]
        else:
            raise ValueError(f"Unsupported target source: {target_source}")

        if compare_space == "action":
            if "action_offset_body" not in rec or "action_scale_body" not in rec:
                continue
            action = env_action
            offset = np.asarray(rec["action_offset_body"], dtype=np.float64)
            scale = np.asarray(rec["action_scale_body"], dtype=np.float64)
            if action.shape != offset.shape or action.shape != scale.shape:
                continue
            if np.any(scale == 0):
                continue
            target = (target_dof - offset) / scale
        elif compare_space == "dof":
            if joint_target_key in rec:
                action = np.asarray(rec[joint_target_key], dtype=np.float64)
            elif "action_offset_body" in rec and "action_scale_body" in rec:
                offset = np.asarray(rec["action_offset_body"], dtype=np.float64)
                scale = np.asarray(rec["action_scale_body"], dtype=np.float64)
                if env_action.shape != offset.shape or env_action.shape != scale.shape:
                    continue
                action = env_action * scale + offset
            else:
                continue
            target = target_dof[: action.shape[0]]
        else:
            raise ValueError(f"Unsupported compare space: {compare_space}")

        if target.shape != action.shape:
            continue

        row = _error_row(
            rec=rec,
            target_frame=target_frame,
            motion_key=motion_key,
            compare_space=compare_space,
            action_source=action_source,
            target_source=target_source,
            action=action,
            target=target,
        )
        rows.append(row)
        if first_bad is None and row["action_l2"] > action_threshold:
            first_bad = {key: value for key, value in row.items() if key != "joint_error"}

    return rows, first_bad


def hand_comparison(
    records: list[dict[str, Any]],
    *,
    motion_dir: Path,
    env_index: int,
    frame_offset: int,
    include_done: bool,
    motion_timebase: str,
    eval_fps: float,
    motion_file_fps: float | None,
) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, dict[str, Any]] = {}
    primitive_rows: list[dict[str, Any]] = []
    dof_rows: list[dict[str, Any]] = []

    for rec in records:
        if int(rec.get("env_index", -1)) != env_index:
            continue
        if not include_done and bool(rec.get("done", False)):
            continue
        motion_key = str(rec.get("motion_key", ""))
        if not motion_key:
            continue
        target_frame = int(rec.get("frame_index", -1)) + frame_offset
        if motion_key not in cache:
            cache[motion_key] = _load_single_motion(_motion_path(motion_dir, motion_key))
        motion = cache[motion_key]

        if "hand_primitive" in rec:
            left = motion_file_array_at_eval_frame(
                motion,
                "hand_action_left",
                target_frame=target_frame,
                motion_timebase=motion_timebase,
                eval_fps=eval_fps,
                motion_file_fps=motion_file_fps,
                interpolation="nearest",
            )
            right = motion_file_array_at_eval_frame(
                motion,
                "hand_action_right",
                target_frame=target_frame,
                motion_timebase=motion_timebase,
                eval_fps=eval_fps,
                motion_file_fps=motion_file_fps,
                interpolation="nearest",
            )
            if left is not None and right is not None:
                action = np.asarray(rec["hand_primitive"], dtype=np.float64)
                target = np.asarray([left, right], dtype=np.float64).reshape(-1)
                if action.shape == target.shape:
                    row = _error_row(
                        rec=rec,
                        target_frame=target_frame,
                        motion_key=motion_key,
                        compare_space="hand_primitive",
                        action_source="actor",
                        target_source="motion-file",
                        action=action,
                        target=target,
                    )
                    sign_match = (action >= 0.0) == (target >= 0.0)
                    row["sign_match"] = sign_match.astype(np.float64)
                    row["sign_accuracy"] = float(sign_match.mean())
                    primitive_rows.append(row)

        if "env_action_hand" in rec:
            target_hand_dof = motion_file_array_at_eval_frame(
                motion,
                "hand_dof_pos",
                target_frame=target_frame,
                motion_timebase=motion_timebase,
                eval_fps=eval_fps,
                motion_file_fps=motion_file_fps,
                interpolation="linear",
            )
            if target_hand_dof is not None:
                action = np.asarray(rec["env_action_hand"], dtype=np.float64)
                target = np.asarray(target_hand_dof, dtype=np.float64)[: action.shape[0]]
                if action.shape == target.shape:
                    dof_rows.append(
                        _error_row(
                            rec=rec,
                            target_frame=target_frame,
                            motion_key=motion_key,
                            compare_space="hand_dof",
                            action_source="actor",
                            target_source="motion-file",
                            action=action,
                            target=target,
                        )
                    )

    return {"primitive": primitive_rows, "dof": dof_rows}


def _print_metric_summary(name: str, rows: list[dict[str, Any]], key: str) -> None:
    if not rows:
        print(f"{name}: no rows")
        return
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    print(
        f"{name}: count={len(rows)} mean={values.mean():.6f} "
        f"p95={np.percentile(values, 95):.6f} max={values.max():.6f}"
    )


def _print_per_joint_mse(
    name: str,
    rows: list[dict[str, Any]],
    *,
    top_joints: int,
    joint_names: list[str] | None = None,
) -> None:
    summary = per_joint_error_summary(rows, joint_names=joint_names)
    if not summary:
        print(f"{name}: no rows")
        return
    ranked = sorted(summary, key=lambda row: row["mse"], reverse=True)
    limit = min(top_joints, len(ranked))
    print(f"{name}: count={summary[0]['count']} joints={len(summary)} top={limit}")
    print("  idx joint_name                       mse        rmse       mae        max_abs")
    for row in ranked[:limit]:
        print(
            f"  {row['joint_index']:02d} {row['joint_name']:<30} "
            f"{row['mse']:.8f} {row['rmse']:.6f} {row['mae']:.6f} {row['max_abs']:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-debug", type=Path, required=True)
    parser.add_argument(
        "--motion-dir",
        type=Path,
        help="Motion robot directory. Required only when --target-source=motion-file.",
    )
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument(
        "--frame-offsets",
        type=str,
        help="Comma-separated offsets for offline t->t+k loss, e.g. 0,1,2,5,10.",
    )
    parser.add_argument("--root-threshold", type=float, default=0.1)
    parser.add_argument("--joint-threshold", type=float, default=2.0)
    parser.add_argument("--action-threshold", type=float, default=1.0)
    parser.add_argument(
        "--compare-space",
        choices=("action", "dof"),
        default="action",
        help=(
            "Compare in normalized action space by default. Use 'dof' to compare "
            "postprocessed joint targets against raw reference DOF positions."
        ),
    )
    parser.add_argument(
        "--action-source",
        choices=("actor", "zero", "forced", "motion"),
        default="actor",
        help=(
            "Compare the actual actor-residual decoded action, the base ATM "
            "zero-residual decoded action, an eval-only forced reference action, "
            "or a forced motion-file action label."
        ),
    )
    parser.add_argument(
        "--target-source",
        choices=("motion-file", "debug-ref"),
        default="motion-file",
        help=(
            "Use raw motion .pkl DOF targets, or the evaluator's logged ref_joint_pos "
            "from token_debug.json. debug-ref is the safer check for eval-time phase/offset."
        ),
    )
    parser.add_argument(
        "--motion-dof-order",
        choices=("mujoco", "isaaclab"),
        default="mujoco",
        help=(
            "Order of raw motion-file dof targets before comparing with logged action "
            "targets. Released PnP .pkl files use Mujoco order."
        ),
    )
    parser.add_argument(
        "--motion-timebase",
        choices=("eval-time", "raw-frame"),
        default="eval-time",
        help=(
            "How to map eval frame_index to raw motion-file DOF frames. eval-time "
            "uses motion fps and --eval-fps; raw-frame preserves the old direct index."
        ),
    )
    parser.add_argument(
        "--eval-fps",
        type=float,
        default=50.0,
        help="Eval/debug frame rate used for frame_index when --motion-timebase=eval-time.",
    )
    parser.add_argument(
        "--motion-file-fps",
        type=float,
        help="Override raw motion-file fps when the .pkl does not contain an fps field.",
    )
    parser.add_argument(
        "--per-joint-mse",
        action="store_true",
        help="Print per-joint MSE/RMSE/MAE ranked by MSE.",
    )
    parser.add_argument(
        "--top-joints",
        type=int,
        default=len(G1_BODY_JOINT_NAMES),
        help="Number of highest-MSE joints to print when --per-joint-mse is set.",
    )
    parser.add_argument(
        "--hand-report",
        action="store_true",
        help=(
            "Also compare logged hand_primitive/env_action_hand against motion-file "
            "hand_action_left/right and hand_dof_pos."
        ),
    )
    parser.add_argument("--include-done", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.token_debug)

    state_rows, first_state_bad = state_divergence(
        records,
        env_index=args.env_index,
        root_threshold=args.root_threshold,
        joint_threshold=args.joint_threshold,
    )
    print(f"token_debug: {args.token_debug}")
    _print_metric_summary("state/root_l2", state_rows, "root_l2")
    _print_metric_summary("state/joint_l2", state_rows, "joint_l2")
    print(f"state/first_over_threshold: {first_state_bad}")

    offsets = _parse_int_list(args.frame_offsets) if args.frame_offsets else [args.frame_offset]
    any_action_rows = False
    first_action_bad = None
    for frame_offset in offsets:
        action_rows, first_action_bad = action_comparison(
            records,
            motion_dir=args.motion_dir,
            env_index=args.env_index,
            frame_offset=frame_offset,
            include_done=args.include_done,
            action_threshold=args.action_threshold,
            compare_space=args.compare_space,
            action_source=args.action_source,
            target_source=args.target_source,
            motion_dof_order=args.motion_dof_order,
            motion_timebase=args.motion_timebase,
            eval_fps=args.eval_fps,
            motion_file_fps=args.motion_file_fps,
        )
        if action_rows:
            any_action_rows = True
            metric_prefix = (
                f"action/{args.action_source}_{args.compare_space}_body"
                f"_vs_{args.target_source}_offset_{frame_offset:+d}"
            )
            _print_metric_summary(f"{metric_prefix}_l2", action_rows, "action_l2")
            _print_metric_summary(
                f"{metric_prefix}_mean_abs", action_rows, "action_mean_abs"
            )
            _print_metric_summary(f"{metric_prefix}_mse", action_rows, "action_mse")
            if args.per_joint_mse:
                _print_per_joint_mse(
                    f"{metric_prefix}_per_joint_mse",
                    action_rows,
                    top_joints=args.top_joints,
                    joint_names=G1_BODY_JOINT_NAMES,
                )
            print(f"action/first_over_threshold_offset_{frame_offset:+d}: {first_action_bad}")

            for window_name, start, end in (
                ("early_0_30", 0, 30),
                ("visible_200_250", 200, 250),
            ):
                window_rows = [
                    row for row in action_rows if start <= int(row["step"]) <= end
                ]
                _print_metric_summary(
                    f"{metric_prefix}_{window_name}_mean_abs",
                    window_rows,
                    "action_mean_abs",
                )
        else:
            has_raw_actions = any("actions" in rec for rec in records)
            has_env_actions = any("env_action_body" in rec for rec in records)
            has_requested_actions = any(
                {
                    "actor": "env_action_body",
                    "zero": "zero_residual_body_action",
                    "forced": "forced_reference_body_action",
                    "motion": "forced_motion_body_action",
                }[args.action_source]
                in rec
                for rec in records
            )
            if not has_requested_actions and args.action_source == "zero":
                reason = (
                    "missing zero_residual_body_action; rerun eval with "
                    "++manager_env.config.debug_zero_residual_action=true"
                )
            elif not has_requested_actions and args.action_source == "forced":
                reason = (
                    "missing forced_reference_body_action; rerun eval with "
                    "++manager_env.config.debug_force_reference_body_action_steps=1"
                )
            elif not has_requested_actions and args.action_source == "motion":
                reason = (
                    "missing forced_motion_body_action; rerun eval with "
                    "++manager_env.config.debug_force_motion_body_action_steps=1 and "
                    "ensure motion files contain an 'action' field"
                )
            elif not has_env_actions:
                reason = "missing env_action_body"
            elif args.target_source == "debug-ref":
                reason = "missing ref_joint_pos at requested offset"
            elif args.compare_space == "action":
                reason = "missing action_offset_body/action_scale_body for action-space comparison"
            else:
                reason = (
                    "missing "
                    + (
                        {
                        "actor": "env_action_body_joint_target",
                        "zero": "zero_residual_body_joint_target",
                        "forced": "forced_reference_body_joint_target",
                        "motion": "forced_motion_body_joint_target",
                    }[args.action_source]
                    )
                    + " or action offset/scale"
                )
            if has_raw_actions and not has_env_actions:
                reason += " (raw 'actions' are actor meta-actions/residuals, not final body actions)"
            print(
                f"action/{args.action_source}_{args.compare_space}_body"
                f"_vs_{args.target_source}_offset_{frame_offset:+d}: no rows, {reason}"
            )

        if args.hand_report:
            if args.motion_dir is None:
                raise ValueError("--motion-dir is required when --hand-report is set")
            hand_rows = hand_comparison(
                records,
                motion_dir=args.motion_dir,
                env_index=args.env_index,
                frame_offset=frame_offset,
                include_done=args.include_done,
                motion_timebase=args.motion_timebase,
                eval_fps=args.eval_fps,
                motion_file_fps=args.motion_file_fps,
            )
            for hand_kind, joint_names in (
                ("primitive", HAND_PRIMITIVE_NAMES),
                ("dof", G1_HAND_JOINT_NAMES),
            ):
                rows = hand_rows[hand_kind]
                metric_prefix = (
                    f"action/actor_hand_{hand_kind}"
                    f"_vs_motion-file_offset_{frame_offset:+d}"
                )
                if rows:
                    any_action_rows = True
                    _print_metric_summary(f"{metric_prefix}_l2", rows, "action_l2")
                    _print_metric_summary(
                        f"{metric_prefix}_mean_abs", rows, "action_mean_abs"
                    )
                    _print_metric_summary(f"{metric_prefix}_mse", rows, "action_mse")
                    if hand_kind == "primitive" and "sign_accuracy" in rows[0]:
                        _print_metric_summary(
                            f"{metric_prefix}_sign_accuracy", rows, "sign_accuracy"
                        )
                    if args.per_joint_mse:
                        _print_per_joint_mse(
                            f"{metric_prefix}_per_joint_mse",
                            rows,
                            top_joints=args.top_joints,
                            joint_names=joint_names,
                        )
                else:
                    missing = "hand_action_left/right" if hand_kind == "primitive" else "hand_dof_pos"
                    print(f"{metric_prefix}: no rows, missing {missing} or logged hand actions")

    if any_action_rows:
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
