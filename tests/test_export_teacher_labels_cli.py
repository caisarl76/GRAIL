from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from grail.cli.export_teacher_labels import export_teacher_labels, main


def _write_debug_log(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _actions(value: float) -> list[float]:
    return [value] * 64 + [value + 1.0, value + 2.0]


def _motion_token(value: float) -> list[float]:
    return [value] * 64


def _record(
    motion_key: str,
    frame_index: int,
    action_value: float,
    *,
    token_value: float | None = None,
    done: bool | None = None,
) -> dict:
    record = {
        "motion_key": motion_key,
        "frame_index": frame_index,
        "actions": _actions(action_value),
        "motion_token": _motion_token(action_value if token_value is None else token_value),
    }
    if done is not None:
        record["done"] = done
    return record


def _write_motion(root: Path, key: str, frames: int) -> None:
    for subdir in ("robot", "objects"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    robot = {
        key: {
            "dof": np.zeros((frames, 29), dtype=np.float32),
            "hand_dof_pos": np.zeros((frames, 14), dtype=np.float32),
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": np.tile(
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                (frames, 1),
            ),
            "fps": 25.0,
        }
    }
    objects = {
        key: {
            "root_pos": np.zeros((frames, 1, 3), dtype=np.float32),
            "root_quat": np.tile(
                np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
                (frames, 1, 1),
            ),
        }
    }
    joblib.dump(robot, root / "robot" / f"{key}.pkl")
    joblib.dump(objects, root / "objects" / f"{key}.pkl")


def test_export_teacher_labels_writes_per_motion_66d_arrays(tmp_path: Path):
    debug_log = tmp_path / "token_debug.json"
    output = tmp_path / "labels"
    _write_debug_log(
        debug_log,
        [
            _record("pickup_table__can__000", 1, 0.2),
            _record("pickup_table__can__000", 0, 0.1),
            _record("pickup_table__cup__000", 0, 0.3),
        ],
    )

    exported = export_teacher_labels(debug_log, output)

    assert exported == 2
    can = np.load(output / "pickup_table__can__000.npy")
    cup = np.load(output / "pickup_table__cup__000.npy")
    assert can.shape == (2, 66)
    assert cup.shape == (1, 66)
    assert np.allclose(can[0, :64], 0.1)
    assert np.allclose(can[0, 64:], [1.1, 2.1])
    assert np.allclose(can[1, 64:], [1.2, 2.2])


def test_export_teacher_labels_saves_final_motion_token_not_latent_residual(tmp_path: Path):
    debug_log = tmp_path / "token_debug.json"
    output = tmp_path / "labels"
    _write_debug_log(
        debug_log,
        [
            _record(
                "pickup_ground__can__000",
                0,
                action_value=0.1,
                token_value=0.9,
            )
        ],
    )

    export_teacher_labels(debug_log, output)

    labels = np.load(output / "pickup_ground__can__000.npy")
    assert labels.shape == (1, 66)
    assert np.allclose(labels[0, :64], 0.9)
    assert np.allclose(labels[0, 64:], [1.1, 2.1])


def test_export_teacher_labels_rejects_missing_final_motion_token_by_default(tmp_path: Path):
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            {
                "motion_key": "pickup_ground__can__000",
                "frame_index": 0,
                "actions": _actions(0.1),
            }
        ],
    )

    with pytest.raises(ValueError, match="final motion_token"):
        export_teacher_labels(debug_log, tmp_path / "labels")


def test_export_teacher_labels_rejects_frame_count_mismatch_with_motion_lib(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=3)
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record(key, 0, 0.1),
            _record(key, 1, 0.2),
        ],
    )

    with pytest.raises(ValueError, match="frame count"):
        export_teacher_labels(debug_log, tmp_path / "labels", motion_lib=motion_lib)


def test_export_teacher_labels_resamples_longer_debug_stream_to_motion_frames(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=3)
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record(key, 0, 0.0),
            _record(key, 1, 1.0),
            _record(key, 2, 2.0),
            _record(key, 3, 3.0),
            _record(key, 4, 4.0),
        ],
    )

    exported = export_teacher_labels(
        debug_log,
        tmp_path / "labels",
        motion_lib=motion_lib,
        resample_to_motion_lib=True,
    )

    labels = np.load(tmp_path / "labels" / f"{key}.npy")
    assert exported == 1
    assert labels.shape == (3, 66)
    assert np.allclose(labels[:, 0], [0.0, 2.0, 4.0])


def test_export_teacher_labels_rejects_short_debug_stream_when_resampling(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=3)
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record(key, 0, 0.0),
            _record(key, 1, 1.0),
        ],
    )

    with pytest.raises(ValueError, match="shorter than motion_lib"):
        export_teacher_labels(
            debug_log,
            tmp_path / "labels",
            motion_lib=motion_lib,
            resample_to_motion_lib=True,
        )


def test_export_teacher_labels_truncates_at_first_done(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=3)
    debug_log = tmp_path / "token_debug.json"
    # 5 real eval frames (done on the last), then replay garbage after timeout.
    _write_debug_log(
        debug_log,
        [
            _record(key, 0, 0.0, done=False),
            _record(key, 1, 1.0, done=False),
            _record(key, 2, 2.0, done=False),
            _record(key, 3, 3.0, done=False),
            _record(key, 4, 4.0, done=True),
            _record(key, 5, 99.0, done=False),
            _record(key, 6, 99.0, done=False),
            _record(key, 7, 99.0, done=False),
        ],
    )

    exported = export_teacher_labels(
        debug_log,
        tmp_path / "labels",
        motion_lib=motion_lib,
        resample_to_motion_lib=True,
        truncate_at_done=True,
    )

    labels = np.load(tmp_path / "labels" / f"{key}.npy")
    assert exported == 1
    assert labels.shape == (3, 66)
    # Truncated to frames 0..4 then resampled 5 -> 3 = indices [0, 2, 4].
    assert np.allclose(labels[:, 0], [0.0, 2.0, 4.0])


def test_export_teacher_labels_truncates_each_motion_independently(tmp_path: Path):
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, "pickup_table__can__000", frames=2)
    _write_motion(motion_lib, "pickup_table__cup__000", frames=2)
    debug_log = tmp_path / "token_debug.json"
    # Two motions sharing a batch; "cup" is shorter and replays garbage after its done.
    _write_debug_log(
        debug_log,
        [
            _record("pickup_table__can__000", 0, 0.0, done=False),
            _record("pickup_table__cup__000", 0, 10.0, done=False),
            _record("pickup_table__can__000", 1, 1.0, done=True),
            _record("pickup_table__cup__000", 1, 11.0, done=True),
            _record("pickup_table__cup__000", 2, 99.0, done=False),
            _record("pickup_table__can__000", 2, 99.0, done=False),
        ],
    )

    exported = export_teacher_labels(
        debug_log,
        tmp_path / "labels",
        motion_lib=motion_lib,
        resample_to_motion_lib=True,
        truncate_at_done=True,
    )

    assert exported == 2
    can = np.load(tmp_path / "labels" / "pickup_table__can__000.npy")
    cup = np.load(tmp_path / "labels" / "pickup_table__cup__000.npy")
    assert can.shape == (2, 66)
    assert cup.shape == (2, 66)
    assert np.allclose(can[:, 0], [0.0, 1.0])
    assert np.allclose(cup[:, 0], [10.0, 11.0])


def test_export_teacher_labels_skips_unknown_motion_keys(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=2)
    debug_log = tmp_path / "token_debug.json"
    # "motion_417" is an overflow/padding env with no .pkl in the library.
    _write_debug_log(
        debug_log,
        [
            _record(key, 0, 0.0, done=False),
            _record(key, 1, 1.0, done=True),
            _record("motion_417", 0, 5.0, done=True),
        ],
    )

    exported = export_teacher_labels(
        debug_log,
        tmp_path / "labels",
        motion_lib=motion_lib,
        resample_to_motion_lib=True,
        truncate_at_done=True,
        skip_unknown_motions=True,
    )

    assert exported == 1
    assert (tmp_path / "labels" / f"{key}.npy").exists()
    assert not (tmp_path / "labels" / "motion_417.npy").exists()


def test_export_teacher_labels_rejects_unknown_motion_key_by_default(tmp_path: Path):
    key = "pickup_table__can__000"
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, key, frames=2)
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record("motion_417", 0, 5.0, done=True),
        ],
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        export_teacher_labels(
            debug_log,
            tmp_path / "labels",
            motion_lib=motion_lib,
            resample_to_motion_lib=True,
        )


def test_export_teacher_labels_dedupes_overflow_env_replays(tmp_path: Path):
    # 2 motions in >2 envs: the overflow env replays "can", emitting duplicate
    # (motion_key, frame_index) rows. Keep the first (real) env's stream.
    motion_lib = tmp_path / "motion_lib"
    _write_motion(motion_lib, "pickup_table__can__000", frames=2)
    _write_motion(motion_lib, "pickup_table__cup__000", frames=2)
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record("pickup_table__can__000", 0, 0.0, done=False),
            _record("pickup_table__cup__000", 0, 10.0, done=False),
            _record("pickup_table__can__000", 0, 99.0, done=False),
            _record("pickup_table__can__000", 1, 1.0, done=True),
            _record("pickup_table__cup__000", 1, 11.0, done=True),
            _record("pickup_table__can__000", 1, 99.0, done=True),
        ],
    )

    exported = export_teacher_labels(
        debug_log,
        tmp_path / "labels",
        motion_lib=motion_lib,
        resample_to_motion_lib=True,
        truncate_at_done=True,
        dedupe_overflow_envs=True,
    )

    assert exported == 2
    can = np.load(tmp_path / "labels" / "pickup_table__can__000.npy")
    assert can.shape == (2, 66)
    # First (real) env kept; overflow replay (99.0) dropped.
    assert np.allclose(can[:, 0], [0.0, 1.0])


def test_export_teacher_labels_rejects_duplicate_frame_index_by_default(tmp_path: Path):
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [
            _record("pickup_table__can__000", 0, 0.0),
            _record("pickup_table__can__000", 0, 0.0),
        ],
    )

    with pytest.raises(ValueError, match="duplicate frame_index"):
        export_teacher_labels(debug_log, tmp_path / "labels")


def test_export_teacher_labels_cli_reports_count(tmp_path: Path, capsys):
    debug_log = tmp_path / "token_debug.json"
    _write_debug_log(
        debug_log,
        [_record("pickup_table__can__000", 0, 0.1)],
    )

    rc = main(["--debug-log", str(debug_log), "--output", str(tmp_path / "labels")])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Exported teacher labels: 1" in captured.out
