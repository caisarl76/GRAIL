from __future__ import annotations

import argparse
import inspect
from itertools import islice
from pathlib import Path
from typing import Any, Optional

import numpy as np


REQUIRED_BATCH_KEYS = (
    "observation.images.ego_view",
    "observation.state",
    "action.wbc",
    "action.motion_token",
    "action.hand_primitive",
    "timestamp",
    "frame_index",
    "episode_index",
)


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _is_finite(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.float()).all().item())
    except Exception:
        pass
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return True
    return bool(np.isfinite(array.astype(np.float64)).all())


def _require_last_dim(batch: dict[str, Any], key: str, expected: int) -> None:
    shape = _shape(batch[key])
    if not shape or shape[-1] != expected:
        raise ValueError(f"{key} must end with dimension {expected}, got shape {shape}")


def validate_training_batch(batch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
    if missing:
        raise ValueError(f"Missing required batch keys: {missing}")

    _require_last_dim(batch, "observation.state", 43)
    _require_last_dim(batch, "action.wbc", 43)
    _require_last_dim(batch, "action.motion_token", 64)
    _require_last_dim(batch, "action.hand_primitive", 2)

    image_shape = _shape(batch["observation.images.ego_view"])
    if len(image_shape) < 3 or 3 not in image_shape[-3:]:
        raise ValueError(
            "observation.images.ego_view must include an RGB channel dimension; "
            f"got shape {image_shape}"
        )

    for key in REQUIRED_BATCH_KEYS:
        if not _is_finite(batch[key]):
            raise ValueError(f"{key} contains non-finite values")

    return {key: _shape(batch[key]) for key in REQUIRED_BATCH_KEYS}


def _as_2d_float_tensor(value: Any, device: str):
    import torch

    tensor = torch.as_tensor(value, device=device).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(-1, tensor.shape[-1])


def run_torch_training_step(batch: dict[str, Any], device: str = "cpu") -> float:
    import torch

    validate_training_batch(batch)

    state = _as_2d_float_tensor(batch["observation.state"], device)
    motion_token = _as_2d_float_tensor(batch["action.motion_token"], device)
    hand_primitive = _as_2d_float_tensor(batch["action.hand_primitive"], device)
    target = torch.cat([motion_token, hand_primitive], dim=-1)
    if state.shape[0] != target.shape[0]:
        raise ValueError(f"state batch {state.shape[0]} does not match target batch {target.shape[0]}")

    image = torch.as_tensor(batch["observation.images.ego_view"], device=device).float()
    if image.ndim >= 4 and image.shape[0] == state.shape[0]:
        image_feature = image.reshape(image.shape[0], -1).mean(dim=1, keepdim=True) / 255.0
    else:
        image_feature = torch.zeros((state.shape[0], 1), device=device)

    inputs = torch.cat([state, image_feature], dim=-1)
    model = torch.nn.Linear(inputs.shape[-1], target.shape[-1], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _dataset_kwargs(dataset_cls, repo_id: str, root: Path) -> dict[str, Any]:
    signature = inspect.signature(dataset_cls)
    kwargs: dict[str, Any] = {"repo_id": repo_id, "root": root}
    if "local_files_only" in signature.parameters:
        kwargs["local_files_only"] = True
    return kwargs


def load_lerobot_dataset(root: Path, repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(**_dataset_kwargs(LeRobotDataset, repo_id, root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a converted LeRobot dataset for GR00T/VLA training: "
            "load dataset, decode batches, validate SONIC VLA keys, and run a tiny torch step."
        )
    )
    parser.add_argument("--root", required=True, help="LeRobot dataset root")
    parser.add_argument("--repo-id", default="tmp/grail_vla_smoke", help="Local LeRobot repo id")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Torch device for the tiny optimization step")
    parser.add_argument("--no-torch-step", action="store_true", help="Only decode and validate batches")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    dataset = load_lerobot_dataset(root, args.repo_id)

    import torch
    from torch.utils.data import DataLoader

    print(f"root: {root}")
    print(f"frames: {len(dataset)}")
    if hasattr(dataset, "num_episodes"):
        print(f"episodes: {dataset.num_episodes}")
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "video_keys"):
        print(f"video_keys: {dataset.meta.video_keys}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    checked = 0
    last_loss = None
    with torch.set_grad_enabled(not args.no_torch_step):
        for batch_index, batch in enumerate(islice(loader, args.num_batches)):
            shape_report = validate_training_batch(batch)
            print(f"batch {batch_index}: {shape_report}")
            if not args.no_torch_step:
                last_loss = run_torch_training_step(batch, device=args.device)
                print(f"batch {batch_index} training_smoke_loss: {last_loss:.6f}")
            checked += 1

    if checked == 0:
        raise ValueError("No batches were produced by the DataLoader")
    print(f"checked_batches: {checked}")
    if last_loss is not None:
        print(f"last_training_smoke_loss: {last_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
