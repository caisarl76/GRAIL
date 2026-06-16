import numpy as np
import pytest

from grail.cli.smoke_lerobot_training import (
    REQUIRED_BATCH_KEYS,
    run_torch_training_step,
    validate_training_batch,
)


def _valid_batch():
    return {
        "observation.images.ego_view": np.zeros((2, 480, 640, 3), dtype=np.uint8),
        "observation.state": np.ones((2, 43), dtype=np.float32),
        "action.wbc": np.ones((2, 43), dtype=np.float32),
        "action.motion_token": np.zeros((2, 64), dtype=np.float32),
        "action.hand_primitive": np.zeros((2, 2), dtype=np.float32),
        "timestamp": np.array([0.0, 0.02], dtype=np.float32),
        "frame_index": np.array([0, 1], dtype=np.int64),
        "episode_index": np.array([0, 0], dtype=np.int64),
    }


def test_validate_training_batch_accepts_sonic_vla_shapes():
    report = validate_training_batch(_valid_batch())

    assert report["observation.images.ego_view"] == (2, 480, 640, 3)
    assert report["observation.state"] == (2, 43)
    assert report["action.motion_token"] == (2, 64)
    assert report["action.hand_primitive"] == (2, 2)


def test_validate_training_batch_reports_missing_required_key():
    batch = _valid_batch()
    batch.pop("action.motion_token")

    with pytest.raises(ValueError, match="Missing required batch keys"):
        validate_training_batch(batch)


def test_validate_training_batch_rejects_bad_motion_token_shape():
    batch = _valid_batch()
    batch["action.motion_token"] = np.zeros((2, 63), dtype=np.float32)

    with pytest.raises(ValueError, match="action.motion_token"):
        validate_training_batch(batch)


def test_validate_training_batch_rejects_bad_hand_primitive_shape():
    batch = _valid_batch()
    batch["action.hand_primitive"] = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="action.hand_primitive"):
        validate_training_batch(batch)


def test_required_batch_keys_stay_focused_on_training_inputs():
    assert REQUIRED_BATCH_KEYS == (
        "observation.images.ego_view",
        "observation.state",
        "action.wbc",
        "action.motion_token",
        "action.hand_primitive",
        "timestamp",
        "frame_index",
        "episode_index",
    )


def test_run_torch_training_step_returns_finite_loss():
    pytest.importorskip("torch")

    loss = run_torch_training_step(_valid_batch(), device="cpu")

    assert np.isfinite(loss)
    assert loss >= 0.0
