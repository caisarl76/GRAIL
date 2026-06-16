from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from grail.cli.verify_ego_lerobot import main


def _write_motion(root: Path, key: str = "pickup_table__can__000", frames: int = 2) -> None:
    for subdir in ["robot", "objects", "meta", "object_usd"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    robot = {
        key: {
            "dof": np.zeros((frames, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": np.tile(
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                (frames, 1),
            ),
            "fps": 25.0,
            "hand_dof_pos": np.zeros((frames, 14), dtype=np.float32),
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
    joblib.dump({"object_name": "can"}, root / "meta" / f"{key}.pkl")
    (root / "object_usd" / f"{key}.usd").write_text("#usda 1.0\n", encoding="utf-8")


def test_verify_cli_accepts_rendered_episode_and_converts_rows(tmp_path, capsys):
    _write_motion(tmp_path)
    ego_root = tmp_path / "ego_frames"
    ego_root.mkdir()
    np.save(
        ego_root / "pickup_table__can__000.npy",
        np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    rc = main(
        [
            "--motion-lib",
            str(tmp_path),
            "--ego-frame-root",
            str(ego_root),
            "--allow-empty-token-labels",
            "--max-episodes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "Planned episodes: 1" in captured.out
    assert "Completed ego sources: 1" in captured.out
    assert "Verified convertible episodes: 1" in captured.out


def test_verify_cli_can_require_all_rendered_sources(tmp_path, capsys):
    _write_motion(tmp_path)

    rc = main(
        [
            "--motion-lib",
            str(tmp_path),
            "--ego-frame-root",
            str(tmp_path / "ego_frames"),
            "--allow-empty-token-labels",
            "--require-all-rendered",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "Missing ego sources: 1" in captured.out
