from __future__ import annotations

import numpy as np
import pytest

from grail.vla.ego_frames import EgoFrameError, find_ego_frame_source, load_ego_frames


def test_load_ego_frames_from_npy_stack(tmp_path):
    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    frames[1, 0, 0, 0] = 255
    frame_path = tmp_path / "pickup_table__can__000.npy"
    np.save(frame_path, frames)

    loaded = load_ego_frames(frame_path, expected_frames=2)

    assert len(loaded) == 2
    assert loaded[0].shape == (4, 5, 3)
    assert loaded[1][0, 0, 0] == 255


def test_load_ego_frames_from_directory_of_npy_images(tmp_path):
    frame_dir = tmp_path / "pickup_table__can__000"
    frame_dir.mkdir()
    np.save(frame_dir / "000001.npy", np.full((2, 2, 3), 7, dtype=np.uint8))
    np.save(frame_dir / "000000.npy", np.full((2, 2, 3), 3, dtype=np.uint8))

    loaded = load_ego_frames(frame_dir, expected_frames=2)

    assert [int(frame[0, 0, 0]) for frame in loaded] == [3, 7]


def test_find_ego_frame_source_prefers_motion_npy_stack(tmp_path):
    np.save(tmp_path / "pickup_table__can__000.npy", np.zeros((1, 2, 2, 3), dtype=np.uint8))

    source = find_ego_frame_source(tmp_path, "pickup_table__can__000")

    assert source == tmp_path / "pickup_table__can__000.npy"


def test_load_ego_frames_rejects_wrong_frame_count(tmp_path):
    frame_path = tmp_path / "frames.npy"
    np.save(frame_path, np.zeros((1, 2, 2, 3), dtype=np.uint8))

    with pytest.raises(EgoFrameError, match="expected 2"):
        load_ego_frames(frame_path, expected_frames=2)
