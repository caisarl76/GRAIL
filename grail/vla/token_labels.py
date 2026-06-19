from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import numpy as np

from grail.vla.episode import VLAEpisode
from grail.vla.lerobot_schema import HAND_PRIMITIVE_DIM, MOTION_TOKEN_DIM


class TokenLabelError(ValueError):
    """Raised when SONIC motion-token labels are missing or malformed."""


PathLike = Union[str, Path]
META_ACTION_DIM = MOTION_TOKEN_DIM + HAND_PRIMITIVE_DIM


def validate_motion_tokens(
    tokens: np.ndarray,
    expected_frames: Optional[int] = None,
    token_dim: Optional[int] = None,
) -> np.ndarray:
    """Return SONIC teacher labels as float64 after checking shape and finiteness.

    Accepted label widths are:
    - 64: final ATM motion token only; hand primitives default to zeros.
    - 66: final ATM motion token plus 2 applied hand primitive values.
    """

    token_array = np.asarray(tokens, dtype=np.float64)
    valid_dims = (token_dim,) if token_dim is not None else (MOTION_TOKEN_DIM, META_ACTION_DIM)
    if token_array.ndim != 2 or token_array.shape[1] not in valid_dims:
        expected = " or ".join(str(dim) for dim in valid_dims)
        raise TokenLabelError(
            f"motion tokens must have shape (T, {expected}), got {token_array.shape}"
        )
    if expected_frames is not None and token_array.shape[0] != expected_frames:
        raise TokenLabelError(
            f"motion tokens have {token_array.shape[0]} frames, expected {expected_frames}"
        )
    if not np.all(np.isfinite(token_array)):
        raise TokenLabelError("motion tokens must contain only finite values")
    return token_array


def _resolve_token_path(source: Path, motion_key: Optional[str]) -> Path:
    if source.is_file():
        return source
    if motion_key is None:
        raise TokenLabelError(f"motion_key is required when token source is a directory: {source}")

    candidates = [
        source / f"{motion_key}.npy",
        source / f"{motion_key}.npz",
        source / motion_key / "tokens.npy",
        source / motion_key / "motion_tokens.npy",
        source / motion_key / "tokens.npz",
        source / motion_key / "motion_tokens.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TokenLabelError(f"No token labels found for motion {motion_key} under {source}")


def _read_token_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        archive = np.load(path)
        if "action.motion_token" in archive and "action.hand_primitive" in archive:
            tokens = archive["action.motion_token"]
            hand = archive["action.hand_primitive"]
            return np.concatenate([tokens, hand], axis=1)
        for key in ("meta_actions", "action.meta_action", "actions", "tokens", "motion_tokens", "action.motion_token", "token_state"):
            if key in archive:
                return archive[key]
        if len(archive.files) == 1:
            return archive[archive.files[0]]
        raise TokenLabelError(
            f"Token npz file {path} must contain one array or a recognized token key"
        )
    raise TokenLabelError(f"Unsupported token-label file type: {path}")


def load_motion_tokens(
    source: PathLike,
    motion_key: Optional[str] = None,
    expected_frames: Optional[int] = None,
    token_dim: Optional[int] = None,
) -> np.ndarray:
    """Load SONIC teacher motion-token labels from npy/npz file or directory."""

    path = _resolve_token_path(Path(source), motion_key)
    return validate_motion_tokens(
        _read_token_file(path),
        expected_frames=expected_frames,
        token_dim=token_dim,
    )


def attach_motion_tokens(episode: VLAEpisode, tokens: np.ndarray) -> VLAEpisode:
    """Return a copy of an episode with one SONIC token vector attached per frame."""

    token_array = validate_motion_tokens(tokens, expected_frames=episode.num_frames)
    motion_tokens = token_array[:, :MOTION_TOKEN_DIM]
    if token_array.shape[1] == META_ACTION_DIM:
        hand_primitives = token_array[:, MOTION_TOKEN_DIM:]
    else:
        hand_primitives = np.zeros((episode.num_frames, HAND_PRIMITIVE_DIM), dtype=np.float64)
    frames = [
        replace(
            frame,
            motion_token=motion_tokens[frame_index],
            hand_primitive=hand_primitives[frame_index],
        )
        for frame_index, frame in enumerate(episode.frames)
    ]
    return replace(episode, frames=frames)
