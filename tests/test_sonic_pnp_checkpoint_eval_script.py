from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/eval/test_sonic_pnp_checkpoint.sh"


def _make_motion_lib(root: Path, *, with_bps: bool = True) -> Path:
    motion_lib = root / "pickup_table"
    for child in ("robot", "objects", "object_usd"):
        (motion_lib / child).mkdir(parents=True)
    if with_bps:
        (motion_lib / "bps").mkdir()
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


def test_sonic_pnp_checkpoint_eval_dry_run_builds_table_render_command(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path)
    output_root = tmp_path / "eval_out"

    result = _run_dry(
        [
            "--task",
            "pnp_table",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(output_root),
            "--motion-keys-file",
            str(tmp_path / "motion_keys.txt"),
            "--gpu",
            "1",
            "--num-envs",
            "3",
            "--max-unique-motions",
            "7",
            "--motion-shard-rank",
            "2",
            "--motion-shard-world-size",
            "5",
            "--render",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in out
    assert "python -u gear_sonic/eval_agent_trl.py" in out
    assert "+checkpoint=models/pnp_table/last.pt" in out
    assert "++eval_callbacks=im_eval" in out
    assert "++run_eval_loop=False" in out
    assert "++num_envs=3" in out
    assert f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_lib / 'robot'}" in out
    assert (
        f"++manager_env.commands.motion.motion_lib_cfg.object_motion_file={motion_lib / 'objects'}"
        in out
    )
    assert f"++manager_env.config.object_usd_path={motion_lib / 'object_usd'}" in out
    assert f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={motion_lib / 'bps'}" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=7" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=2" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=5" in out
    assert f"++manager_env.config.debug_state_log={output_root / 'pnp_table' / 'token_debug.json'}" in out
    assert (
        "++manager_env.config.action_transform_module_cfg="
        "models/sonic_manipulation_base/model_config.yaml"
    ) in out
    assert (
        "++manager_env.config.action_transform_module_checkpoint="
        "models/sonic_manipulation_base/last.pt"
    ) in out
    assert f"++manager_env.config.per_rank_motion_keys_file={tmp_path / 'motion_keys.txt'}" in out
    assert "++manager_env.config.render_results=True" in out
    assert f"++manager_env.config.save_rendering_dir={output_root / 'pnp_table' / 'renderings'}" in out
    assert "~manager_env/recorders=empty" in out
    assert "+manager_env/recorders=render" in out


def test_sonic_pnp_checkpoint_eval_sets_noninteractive_eval_env():
    script = SCRIPT.read_text()

    assert "export HYDRA_FULL_ERROR=1" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "export WANDB_MODE=offline" in script
    assert "export OMNI_KIT_ACCEPT_EULA=Yes" in script


def test_sonic_pnp_checkpoint_eval_dry_run_warns_but_sets_ground_bps_path(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path, with_bps=False)
    output_root = tmp_path / "eval_out"

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(output_root),
            "--max-unique-motions",
            "5",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "BPS dir not found" in result.stderr
    out = result.stdout
    assert "+checkpoint=models/pnp_ground/last.pt" in out
    assert f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={motion_lib / 'bps'}" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.bps_dim=10" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1" in out
    assert "++manager_env.config.render_results=True" not in out
    assert f"++manager_env.config.debug_state_log={output_root / 'pnp_ground' / 'token_debug.json'}" in out


def test_sonic_pnp_checkpoint_eval_all_motions_omits_cap(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path)

    result = _run_dry(
        [
            "--task",
            "pnp_table",
            "--data-dir",
            str(motion_lib),
            "--max-unique-motions",
            "all",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions" not in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0" in out
    assert "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1" in out


def test_sonic_pnp_checkpoint_eval_can_disable_early_terminations(tmp_path: Path):
    motion_lib = _make_motion_lib(tmp_path)

    result = _run_dry(
        [
            "--task",
            "pnp_ground",
            "--data-dir",
            str(motion_lib),
            "--output-root",
            str(tmp_path / "eval_out"),
            "--disable-early-terminations",
        ]
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "++manager_env.terminations.anchor_pos=null" in out
    assert "++manager_env.terminations.anchor_ori_full=null" in out
    assert "++manager_env.terminations.ee_body_pos=null" in out
    assert "++manager_env.terminations.object_pos_deviation=null" in out
    assert "++manager_env.terminations.object_z_pos_deviation=null" in out
    assert "++manager_env.terminations.grasp_failure_after_contact=null" in out
    assert "++manager_env.terminations.hand_table_contact_termination=null" in out


def test_motion_lib_base_has_configured_bps_dim_fallback_for_missing_bps_dir():
    source = (REPO_ROOT / "imports/SONIC/gear_sonic/utils/motion_lib/motion_lib_base.py").read_text()

    assert 'motion_lib_cfg.get("bps_dim", 0)' in source
    assert "BPS obs will be zeros with dim={self._bps_dim}" in source


def test_im_eval_debug_records_are_per_env_teacher_label_records():
    source = (
        REPO_ROOT / "imports/SONIC/gear_sonic/trl/callbacks/im_eval_callback.py"
    ).read_text()

    assert "for env_idx in range" in source
    assert '"motion_key": motion_key' in source
    assert '"frame_index": int(self.curr_steps)' in source
