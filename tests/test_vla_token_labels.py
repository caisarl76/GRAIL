import numpy as np
import pytest

from grail.vla.episode import VLAEpisode, VLAFrame
from grail.vla.token_labels import (
    TokenLabelError,
    attach_motion_tokens,
    load_motion_tokens,
    validate_motion_tokens,
)


def _episode(num_frames: int = 2) -> VLAEpisode:
    frames = [
        VLAFrame(
            frame_index=i,
            timestamp=float(i),
            joint_position=np.zeros(43, dtype=np.float32),
            root_position=np.zeros(3, dtype=np.float32),
            root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            object_position=np.zeros(3, dtype=np.float32),
            object_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        for i in range(num_frames)
    ]
    return VLAEpisode(
        motion_key="pickup_table__can__000",
        fps=25.0,
        frames=frames,
        robot_path=None,
        object_path=None,
        object_usd_path=None,
    )


def test_attach_motion_tokens_sets_one_token_per_frame():
    tokens = np.ones((2, 64), dtype=np.float32)

    episode = attach_motion_tokens(_episode(), tokens)

    assert episode.frames[0].motion_token.shape == (64,)
    assert episode.frames[0].hand_primitive.shape == (2,)
    assert episode.frames[1].motion_token.shape == (64,)
    assert episode.frames[1].hand_primitive.shape == (2,)
    assert np.allclose(episode.frames[0].motion_token, 1.0)
    assert np.allclose(episode.frames[0].hand_primitive, 0.0)


def test_attach_motion_tokens_splits_66d_meta_actions():
    meta_actions = np.zeros((2, 66), dtype=np.float32)
    meta_actions[:, :64] = 0.25
    meta_actions[:, 64:] = np.array([[-1.0, 1.0], [0.5, -0.5]], dtype=np.float32)

    episode = attach_motion_tokens(_episode(), meta_actions)

    assert episode.frames[0].motion_token.shape == (64,)
    assert episode.frames[0].hand_primitive.shape == (2,)
    assert np.allclose(episode.frames[0].motion_token, 0.25)
    assert np.allclose(episode.frames[0].hand_primitive, [-1.0, 1.0])
    assert np.allclose(episode.frames[1].hand_primitive, [0.5, -0.5])


def test_validate_motion_tokens_rejects_wrong_frame_count():
    tokens = np.ones((1, 64), dtype=np.float32)

    with pytest.raises(TokenLabelError, match="expected 2"):
        validate_motion_tokens(tokens, expected_frames=2)


def test_validate_motion_tokens_rejects_nan():
    tokens = np.ones((2, 64), dtype=np.float32)
    tokens[0, 0] = np.nan

    with pytest.raises(TokenLabelError, match="finite"):
        validate_motion_tokens(tokens, expected_frames=2)


def test_load_motion_tokens_resolves_motion_key_from_directory(tmp_path):
    tokens = np.ones((2, 64), dtype=np.float32)
    np.save(tmp_path / "pickup_table__can__000.npy", tokens)

    loaded = load_motion_tokens(tmp_path, "pickup_table__can__000", expected_frames=2)

    assert loaded.shape == (2, 64)
    assert loaded.dtype == np.float64


def test_load_motion_tokens_reads_npz_tokens_key(tmp_path):
    tokens = np.full((2, 64), 0.25, dtype=np.float32)
    token_path = tmp_path / "pickup_table__can__000.npz"
    np.savez(token_path, tokens=tokens)

    loaded = load_motion_tokens(token_path, expected_frames=2)

    assert np.allclose(loaded, 0.25)


def test_load_motion_tokens_reads_split_npz_motion_token_and_hand_primitive(tmp_path):
    token_path = tmp_path / "pickup_table__can__000.npz"
    np.savez(
        token_path,
        **{
            "action.motion_token": np.full((2, 64), 0.25, dtype=np.float32),
            "action.hand_primitive": np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32),
        },
    )

    loaded = load_motion_tokens(token_path, expected_frames=2)

    assert loaded.shape == (2, 66)
    assert np.allclose(loaded[:, :64], 0.25)
    assert np.allclose(loaded[:, 64:], [[1.0, -1.0], [0.5, -0.5]])
