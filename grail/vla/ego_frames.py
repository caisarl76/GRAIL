from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


PathLike = Union[str, Path]


class EgoFrameError(ValueError):
    """Raised when rendered ego-view frames cannot be loaded or validated."""


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv")


def _as_rgb_uint8(frame: np.ndarray, source: Path) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise EgoFrameError(f"ego frame from {source} must have shape (H, W, 3/4), got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _check_frame_count(frames: list[np.ndarray], expected_frames: Optional[int], source: Path) -> None:
    if expected_frames is not None and len(frames) != expected_frames:
        raise EgoFrameError(f"ego frames from {source} have {len(frames)} frames, expected {expected_frames}")


def _load_npy_frames(path: Path) -> list[np.ndarray]:
    array = np.load(path)
    if array.ndim == 4:
        return [_as_rgb_uint8(frame, path) for frame in array]
    if array.ndim == 3:
        return [_as_rgb_uint8(array, path)]
    raise EgoFrameError(f"ego frame npy file {path} must have shape (T, H, W, 3) or (H, W, 3)")


def _load_image_file(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise EgoFrameError(f"Reading image frames requires Pillow: {path}") from exc

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_video_frames(path: Path) -> list[np.ndarray]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise EgoFrameError(f"Reading video ego frames requires imageio: {path}") from exc

    array = iio.imread(path)
    if array.ndim != 4:
        raise EgoFrameError(f"ego video {path} must decode to shape (T, H, W, 3/4), got {array.shape}")
    return [_as_rgb_uint8(frame, path) for frame in array]


def find_ego_frame_source(frame_root: PathLike, motion_key: str) -> Path:
    """Resolve the rendered ego-frame source for a motion key.

    The first implementation supports renderer-friendly file layouts:
    ``<root>/<motion_key>.npy`` for a frame stack, ``<root>/<motion_key>/`` for
    per-frame arrays/images, or ``<root>/<motion_key>.mp4`` for rendered video.
    """

    root = Path(frame_root)
    if root.is_file():
        return root

    candidates = [
        root / f"{motion_key}.npy",
        root / motion_key,
        root / f"{motion_key}.mp4",
        root / f"{motion_key}.mov",
        root / f"{motion_key}.mkv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise EgoFrameError(f"No ego-frame source found for motion {motion_key} under {root}")


def load_ego_frames(source: PathLike, expected_frames: Optional[int] = None) -> list[np.ndarray]:
    """Load rendered ego-view RGB frames from a npy stack, frame directory, or video."""

    path = Path(source)
    if not path.exists():
        raise EgoFrameError(f"Ego-frame source not found: {path}")

    if path.is_dir():
        frame_paths = sorted(
            [
                candidate
                for candidate in path.iterdir()
                if candidate.suffix.lower() == ".npy" or candidate.suffix.lower() in _IMAGE_SUFFIXES
            ]
        )
        if not frame_paths:
            raise EgoFrameError(f"No supported ego frame files found in {path}")
        frames = []
        for frame_path in frame_paths:
            if frame_path.suffix.lower() == ".npy":
                loaded = _load_npy_frames(frame_path)
                if len(loaded) != 1:
                    raise EgoFrameError(f"Per-frame npy file must contain one image: {frame_path}")
                frames.append(loaded[0])
            else:
                frames.append(_load_image_file(frame_path))
        _check_frame_count(frames, expected_frames, path)
        return frames

    suffix = path.suffix.lower()
    if suffix == ".npy":
        frames = _load_npy_frames(path)
    elif suffix in _IMAGE_SUFFIXES:
        frames = [_load_image_file(path)]
    elif suffix in _VIDEO_SUFFIXES:
        frames = _load_video_frames(path)
    else:
        raise EgoFrameError(f"Unsupported ego-frame source type: {path}")

    _check_frame_count(frames, expected_frames, path)
    return frames
