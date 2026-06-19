from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/eval/generate_teacher_labels.sh"


def _make_motion_lib(root: Path, *, num_motions: int = 40) -> Path:
    motion_lib = root / "pickup_ground"
    for child in ("robot", "objects", "object_usd", "bps"):
        (motion_lib / child).mkdir(parents=True)
    for i in range(num_motions):
        (motion_lib / "robot" / f"pickup_ground__obj_{i:03d}__000.pkl").touch()
    return motion_lib


def _run_dry(script_args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        ["bash", str(SCRIPT), *script_args, "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_generate_teacher_labels_computes_one_batch_per_shard(tmp_path: Path):
    # 40 motions with 16 envs => ceil(40/16) = 3 shards, each <= 16 motions = one batch.
    motion_lib = _make_motion_lib(tmp_path, num_motions=40)
    output_root = tmp_path / "sonic_eval_shards"
    label_root = tmp_path / "teacher_labels"

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(output_root),
            "--label-root",
            str(label_root),
            "--gpu",
            "1",
            "--num-envs",
            "16",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout

    # world_size derived so each shard holds <= num_envs motions (single batch).
    assert "++num_envs=16" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=3" in out
    assert "++manager_env.terminations.ee_body_pos=null" in out

    for rank in range(3):
        pad = f"{rank:04d}"
        assert f"++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank={rank}" in out
        assert f"rank_{pad}" in out
    assert "motion_shard_rank=3" not in out

    # Export step hardened for multi-env batches.
    assert "export_teacher_labels" in out
    assert "--resample-to-motion-lib" in out
    assert "--truncate-at-done" in out
    assert "--skip-unknown-motions" in out
    assert "--dedupe-overflow-envs" in out
    assert f"--output {label_root}" in out


def test_generate_teacher_labels_respects_shard_range(tmp_path: Path):
    # 40 motions, 8 envs => 5 shards; run only ranks 2..3.
    motion_lib = _make_motion_lib(tmp_path, num_motions=40)

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(tmp_path / "shards"),
            "--label-root",
            str(tmp_path / "labels"),
            "--num-envs",
            "8",
            "--shard-start",
            "2",
            "--shard-end",
            "4",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "motion_shard_world_size=5" in out
    assert "motion_shard_rank=2" in out
    assert "motion_shard_rank=3" in out
    # End is exclusive; rank 4 and ranks below the start are not run.
    assert "motion_shard_rank=4" not in out
    assert "motion_shard_rank=0" not in out
    assert "motion_shard_rank=1" not in out


def test_generate_teacher_labels_passes_extra_hydra_overrides(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path, num_motions=4)

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(tmp_path / "shards"),
            "--label-root",
            str(tmp_path / "labels"),
            "--num-envs",
            "4",
            "++manager_env.commands.motion.object_init_z_offset=-0.13",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "++manager_env.commands.motion.object_init_z_offset=-0.13" in result.stdout


def test_generate_teacher_labels_splits_explicit_motion_keys_by_rank(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path, num_motions=6)
    keys_file = tmp_path / "requested_motion_keys.txt"
    keys_file.write_text(
        "\n".join(
            [
                "pickup_ground__obj_000__000",
                "pickup_ground__obj_001__000",
                "pickup_ground__obj_002__000",
                "pickup_ground__obj_003__000",
                "pickup_ground__obj_004__000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "shards"

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(output_root),
            "--label-root",
            str(tmp_path / "labels"),
            "--motion-keys-file",
            str(keys_file),
            "--num-envs",
            "2",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "[plan] 5 motions, 2 envs -> 3 shards" in out
    assert "++manager_env.config.per_rank_motion_keys_file=" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=" in out

    assert (
        output_root / "rank_0000" / "pnp_ground" / "motion_keys.txt"
    ).read_text(encoding="utf-8").splitlines() == [
        "pickup_ground__obj_000__000",
    ]
    assert (
        output_root / "rank_0001" / "pnp_ground" / "motion_keys.txt"
    ).read_text(encoding="utf-8").splitlines() == [
        "pickup_ground__obj_001__000",
        "pickup_ground__obj_002__000",
    ]
    assert (
        output_root / "rank_0002" / "pnp_ground" / "motion_keys.txt"
    ).read_text(encoding="utf-8").splitlines() == [
        "pickup_ground__obj_003__000",
        "pickup_ground__obj_004__000",
    ]


def test_generate_teacher_labels_rejects_oversubscribed_shard_override(tmp_path: Path):
    # 40 motions / 2 shards = 20 motions per shard > 4 envs => multi-batch, must be refused.
    motion_lib = _make_motion_lib(tmp_path, num_motions=40)

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(tmp_path / "shards"),
            "--label-root",
            str(tmp_path / "labels"),
            "--num-envs",
            "4",
            "--num-shards",
            "2",
        ]
    )

    assert result.returncode != 0
    assert "per shard" in (result.stderr + result.stdout)


def test_generate_teacher_labels_skips_completed_ranks(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path, num_motions=40)
    output_root = tmp_path / "shards"
    done_marker = output_root / "rank_0000" / "pnp_ground" / ".teacher_labels.done"
    done_marker.parent.mkdir(parents=True)
    done_marker.touch()

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(output_root),
            "--label-root",
            str(tmp_path / "labels"),
            "--num-envs",
            "16",
            "--skip-existing",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "motion_shard_rank=0" not in out
    assert "motion_shard_rank=1" in out
