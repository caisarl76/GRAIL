# Teacher Label Generation Handover

This page is the teammate handover for generating SONIC teacher labels from
GRAIL motion libraries, attaching them to ego-view LeRobot data, and verifying
the result before GR00T/VLA training.

## What This Produces

Teacher label generation converts closed-loop GEAR-SONIC teacher rollouts into
one NumPy file per GRAIL motion:

```text
<label-root>/<motion_key>.npy
shape: (frames, 66)
layout: [motion_token(64), hand_primitive(2)]
```

The first 64 values must come from `token_debug.json["motion_token"]`, which is
the final action-transform-module token produced by SONIC. Do not use
`token_debug.json["actions"][:64]` for distillation labels; those values are
actor residual/debug values in the current eval path.

The downstream LeRobot exporter writes these labels as:

- `action.motion_token`: 64-D supervised token target.
- `action.hand_primitive`: 2-D applied hand primitive target.

## Repository Locations

Run all commands from the GRAIL repository root unless the command explicitly
changes directory:

```bash
cd /home/jihun/work/GRAIL
```

Important paths:

| Path | Purpose |
| --- | --- |
| `imports/SONIC` | Vendored GEAR-SONIC release tree used for eval, training, and deploy code. |
| `imports/SONIC/gear_sonic/eval_agent_trl.py` | SONIC eval entrypoint used by teacher generation. |
| `imports/SONIC/models/pnp_table/last.pt` | Released tabletop pickup teacher actor checkpoint. |
| `imports/SONIC/models/pnp_ground/last.pt` | Released ground pickup teacher actor checkpoint. |
| `imports/SONIC/models/sonic_manipulation_base/last.pt` | ATM checkpoint whose final `motion_token` space is used for labels. |
| `scripts/eval/test_sonic_pnp_checkpoint.sh` | Single strict or diagnostic SONIC eval wrapper. |
| `scripts/eval/generate_teacher_labels.sh` | Production label-generation wrapper. |
| `grail/cli/export_teacher_labels.py` | Converts `token_debug.json` into per-motion `.npy` labels. |
| `scripts/eval/smoke_teacher_lerobot_pipeline.sh` | End-to-end small smoke: teacher labels, ego render, LeRobot export, training batch. |

## Environment Setup

Use the repository-provided SONIC environment setup when the `sonic` conda env
or IsaacLab stack is missing:

```bash
bash scripts/setup/install_env_sonic.sh
```

This creates the default `sonic` conda env, installs Isaac Sim/Lab, installs
`imports/SONIC/gear_sonic[training,data_collection]`, installs GRAIL editable,
and pulls SONIC LFS assets when `git-lfs` is available.

Useful setup variants:

```bash
GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
BOOTSTRAP_SONIC=0 bash scripts/setup/install_env_sonic.sh
PULL_LFS=0 bash scripts/setup/install_env_sonic.sh
```

Download only the SONIC checkpoints from the project checkpoint script:

```bash
bash scripts/setup/download_checkpoints.sh \
  --skip-gem-smpl --skip-gem-soma \
  --skip-foundationpose --skip-hunyuan3d
```

Preflight the expected files:

```bash
test -d imports/SONIC/gear_sonic
test -f imports/SONIC/models/pnp_table/last.pt
test -f imports/SONIC/models/pnp_ground/last.pt
test -f imports/SONIC/models/sonic_manipulation_base/last.pt
test -f imports/SONIC/models/sonic_manipulation_base/model_config.yaml
OMNI_KIT_ACCEPT_EULA=Yes python imports/SONIC/check_environment.py --training
```

If `export_teacher_labels` reports missing final `motion_token`, the SONIC eval
tree is not recording the debug fields required by this pipeline. Use this
GRAIL checkout's `imports/SONIC` tree, where the eval callback records
`motion_token` and `hand_primitive` into `token_debug.json`.

## Motion Library Inputs

Teacher generation expects a complete GRAIL motion-library root:

```text
<motion-lib>/
  robot/*.pkl
  objects/*.pkl
  object_usd/*.usd
  bps/                 # preferred for production; script can warn and use zero BPS if absent
```

The released Hugging Face dataset snapshot normally contains:

```bash
SNAPSHOT_ROOT="$(find \
  /home/jihun/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Locomanipulation-GRAIL/snapshots \
  /root/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Locomanipulation-GRAIL/snapshots \
  -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"

export PNP_TABLE_DATA="$SNAPSHOT_ROOT/data/pickup_table"
export PNP_GROUND_DATA="$SNAPSHOT_ROOT/data/pickup_ground"

for dir in "$PNP_TABLE_DATA" "$PNP_GROUND_DATA"; do
  test -d "$dir/robot"
  test -d "$dir/objects"
  test -d "$dir/object_usd"
done
```

Use `--task pnp_table` with `pickup_table` data and `--task pnp_ground` with
`pickup_ground` data.

## Required Behavior Gate

Before generating full labels, run a strict closed-loop teacher eval without
disabled terminations. This checks that the teacher actually performs the task
before the label job forces every motion to run to its full horizon.

Example for ground pickup:

```bash
conda activate sonic

bash scripts/eval/test_sonic_pnp_checkpoint.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/teacher_verify/strict_pnp_ground \
  --gpu 0 \
  --num-envs 4 \
  --max-unique-motions 16 \
  --render
```

Example for tabletop pickup:

```bash
conda activate sonic

bash scripts/eval/test_sonic_pnp_checkpoint.sh \
  --task pnp_table \
  --data-dir "$PNP_TABLE_DATA" \
  --bps-dir "$PNP_TABLE_DATA/bps" \
  --output-root data/teacher_verify/strict_pnp_table \
  --gpu 0 \
  --num-envs 4 \
  --max-unique-motions 16 \
  --render
```

Check the outputs:

```text
data/teacher_verify/strict_<task>/<task>/metrics/metrics_eval.json
data/teacher_verify/strict_<task>/<task>/token_debug.json
data/teacher_verify/strict_<task>/<task>/renderings/
```

Acceptance criteria:

- The rendered rollout shows grasp/lift behavior for the selected task.
- Metrics are not obviously failed.
- The script prints nonzero `token_debug_records`.
- The script prints `motion_token_dim: 64`.
- There is no failure about missing final `motion_token`.

Only after this strict gate passes should full label generation use
`--disable-early-terminations`, which the production wrapper applies
internally.

## Production Label Generation

The production wrapper is:

```bash
bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/pnp_ground_teacher/sonic_eval_shards \
  --label-root data/teacher_labels/pnp_ground \
  --gpu 0 \
  --num-envs 16 \
  --skip-existing
```

For tabletop:

```bash
bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_table \
  --data-dir "$PNP_TABLE_DATA" \
  --bps-dir "$PNP_TABLE_DATA/bps" \
  --output-root data/pnp_table_teacher/sonic_eval_shards \
  --label-root data/teacher_labels/pnp_table \
  --gpu 0 \
  --num-envs 16 \
  --skip-existing
```

For the current ground-pickup smoke path, pass the known object-height override
if strict eval confirms it is still needed:

```bash
++manager_env.commands.motion.object_init_z_offset=-0.13
```

Add it at the end of the `generate_teacher_labels.sh` command.

### Sharding Model

`generate_teacher_labels.sh` derives:

```text
num_shards = ceil(number_of_motions / num_envs)
```

Each rank runs one Isaac launch and labels at most `num_envs` motions. This is
intentional: per-object USD collision meshes are static after scene creation,
so multi-batch shards are unreliable. If an explicit `--num-shards` would put
more motions in a shard than `--num-envs`, the wrapper refuses to run.

To resume or distribute work:

```bash
# GPU 0, ranks 0..49
bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/pnp_ground_teacher/sonic_eval_shards \
  --label-root data/teacher_labels/pnp_ground \
  --gpu 0 \
  --num-envs 16 \
  --shard-start 0 \
  --shard-end 50 \
  --skip-existing

# GPU 1, ranks 50..99
bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/pnp_ground_teacher/sonic_eval_shards \
  --label-root data/teacher_labels/pnp_ground \
  --gpu 1 \
  --num-envs 16 \
  --shard-start 50 \
  --shard-end 100 \
  --skip-existing
```

`--skip-existing` checks the per-rank marker:

```text
<output-root>/rank_XXXX/<task>/.teacher_labels.done
```

It does not inspect label files directly.

### Exact Episode Subsets

When label generation must match a rendered subset exactly, create a motion-key
file and pass it to the wrapper:

```bash
find "$PNP_GROUND_DATA/robot" -maxdepth 1 -name '*.pkl' -printf '%f\n' \
  | sed 's/\.pkl$//' \
  | sort \
  | head -20 > data/teacher_verify/requested_motion_keys.txt

bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/teacher_verify/eval_shards \
  --label-root data/teacher_verify/teacher_labels \
  --motion-keys-file data/teacher_verify/requested_motion_keys.txt \
  --gpu 0 \
  --num-envs 5
```

With `--motion-keys-file`, each rank writes its own
`rank_XXXX/<task>/motion_keys.txt` and the underlying SONIC eval runs with
`motion_shard_world_size=1` plus an explicit motion-key filter.

## What The Wrapper Runs Internally

For each rank, the wrapper calls `scripts/eval/test_sonic_pnp_checkpoint.sh`,
which runs from `imports/SONIC`:

```text
python -u gear_sonic/eval_agent_trl.py
  +checkpoint=models/<task>/last.pt
  ++manager_env.config.action_transform_module_cfg=models/sonic_manipulation_base/model_config.yaml
  ++manager_env.config.action_transform_module_checkpoint=models/sonic_manipulation_base/last.pt
  ++manager_env.config.debug_state_log=<rank>/<task>/token_debug.json
  ++manager_env.config.debug_zero_residual_action=true
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<motion-lib>/robot
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=<motion-lib>/objects
  ++manager_env.config.object_usd_path=<motion-lib>/object_usd
```

Then it exports labels with:

```text
python -m grail.cli.export_teacher_labels
  --debug-log <rank>/<task>/token_debug.json
  --output <label-root>
  --motion-lib <motion-lib>
  --resample-to-motion-lib
  --truncate-at-done
  --skip-unknown-motions
  --dedupe-overflow-envs
  --overwrite
```

Those export flags are part of the production path:

- `--truncate-at-done` drops replay after a short motion times out.
- `--resample-to-motion-lib` converts the teacher stream to the GRAIL motion
  frame count.
- `--skip-unknown-motions` ignores overflow/padding env keys not present in
  `robot/*.pkl`.
- `--dedupe-overflow-envs` keeps the first stream when extra envs duplicate a
  motion.

## Label Verification

First, dry-run the command to check sharding and Hydra overrides:

```bash
bash scripts/eval/generate_teacher_labels.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root /tmp/grail_teacher_dry/eval_shards \
  --label-root /tmp/grail_teacher_dry/labels \
  --num-envs 16 \
  --dry-run
```

The dry run should show:

- `++manager_env.config.action_transform_module_checkpoint=models/sonic_manipulation_base/last.pt`
- `++manager_env.config.debug_state_log=.../token_debug.json`
- `--resample-to-motion-lib`
- `--truncate-at-done`
- `--dedupe-overflow-envs`

After generation, run a shape/count check:

```bash
python - "$PNP_GROUND_DATA" data/teacher_labels/pnp_ground <<'PY'
from pathlib import Path
import sys
import joblib
import numpy as np

motion_lib = Path(sys.argv[1])
label_root = Path(sys.argv[2])
motion_keys = sorted(path.stem for path in (motion_lib / "robot").glob("*.pkl"))
labels = sorted(label_root.glob("*.npy"))
print("motions:", len(motion_keys))
print("labels:", len(labels))

missing = [key for key in motion_keys if not (label_root / f"{key}.npy").is_file()]
if missing:
    raise SystemExit(f"missing labels: {missing[:10]}")

for key in motion_keys:
    path = label_root / f"{key}.npy"
    arr = np.load(path)
    motion = joblib.load(motion_lib / "robot" / f"{key}.pkl")
    if "dof" not in motion:
        motion = next(iter(motion.values()))
    frames = int(np.asarray(motion["dof"]).shape[0])
    if arr.shape != (frames, 66):
        raise SystemExit(f"{path}: label shape {arr.shape}, expected ({frames}, 66)")
    if not np.isfinite(arr).all():
        raise SystemExit(f"{path}: non-finite values")

print("teacher labels: OK")
PY
```

## Ego Frames And LeRobot Export

Teacher labels are only useful for VLA training after ego frames have been
rendered and rows are exported to the SONIC/GR00T LeRobot schema.

Render ego frames on an RTX/Isaac-capable host:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --render-only \
  --motion-lib "$PNP_GROUND_DATA" \
  --output data/ego_frames/pnp_ground \
  --resolution 640x480 \
  --write-manifests \
  --skip-existing
```

Optional visual diagnostic:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --render-only \
  --with-third-eye-comparison \
  --motion-lib "$PNP_GROUND_DATA" \
  --output data/ego_frames/pnp_ground \
  --side-by-side-output data/ego_compare/pnp_ground \
  --resolution 640x480 \
  --max-motions 5 \
  --write-manifests
```

Export LeRobot rows from existing ego frames and labels:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --motion-lib "$PNP_GROUND_DATA" \
  --ego-frame-root data/ego_frames/pnp_ground \
  --output data/lerobot_v2/pnp_ground_labeled \
  --token-labels data/teacher_labels/pnp_ground \
  --only-rendered \
  --overwrite-existing
```

During renderer bring-up only, `--allow-empty-token-labels` can check camera
and row formatting before labels exist. Do not use unlabeled episodes for final
training.

## LeRobot Verification

Verify that rendered episodes plus labels convert into valid SONIC VLA rows:

```bash
python -m grail.cli.verify_ego_lerobot \
  --motion-lib "$PNP_GROUND_DATA" \
  --ego-frame-root data/ego_frames/pnp_ground \
  --token-labels data/teacher_labels/pnp_ground \
  --require-all-rendered \
  --max-episodes 10
```

Expected output has `Verified convertible episodes` greater than zero and no
`ERROR:` lines.

Smoke-test the resulting local LeRobot dataset:

```bash
python -m grail.cli.smoke_lerobot_training \
  --root data/lerobot_v2/pnp_ground_labeled \
  --repo-id tmp/grail_vla_smoke \
  --batch-size 2 \
  --num-batches 2 \
  --device cpu
```

This loads the dataset, checks required batch keys and dimensions, decodes ego
images, and runs a tiny torch optimization step against motion-token and hand
targets.

## Full End-To-End Smoke

For a small end-to-end check, use:

```bash
bash scripts/eval/smoke_teacher_lerobot_pipeline.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root data/teacher_verify/teacher_lerobot_smoke_pnp_ground \
  --gpu 0 \
  --num-episodes 10 \
  --num-envs 5 \
  --device cpu
```

Dry-run first when checking paths:

```bash
bash scripts/eval/smoke_teacher_lerobot_pipeline.sh \
  --task pnp_ground \
  --data-dir "$PNP_GROUND_DATA" \
  --bps-dir "$PNP_GROUND_DATA/bps" \
  --output-root /tmp/grail_teacher_lerobot_smoke \
  --gpu 0 \
  --num-episodes 10 \
  --num-envs 5 \
  --dry-run
```

The smoke creates:

```text
<output-root>/requested_motion_keys.txt
<output-root>/eval_shards/
<output-root>/teacher_labels/*.npy
<output-root>/ego_frames/*.mp4
<output-root>/lerobot/
```

## Optional MuJoCo Replay Check

After a labeled LeRobot dataset exists, one episode can be extracted and replayed
through the MuJoCo deploy path:

```bash
bash scripts/eval/replay_lerobot_teacher_mujoco.sh \
  --dataset-root data/lerobot_v2/pnp_ground_labeled \
  --motion-key <motion_key> \
  --motion-root "$SNAPSHOT_ROOT/data" \
  --sonic-root "$PWD/imports/SONIC" \
  --output-root outputs/teacher_token_replay \
  --overwrite
```

The command prepares:

- `<output-root>/<motion_key>/<motion_key>.npy`
- `<output-root>/<motion_key>/deploy_reference/`
- printed terminal commands for MuJoCo sim and teacher-token replay.

Important caveat: teacher-token replay uses a decoder ONNX. The decoder must be
exported from the same `sonic_manipulation_base` ATM that produced the labels;
see `docs/teacher_decoder_onnx_notes.md`. A stale deploy decoder can make
correct labels look wrong.

## Unit Tests To Run After Pipeline Changes

Run these before trusting changes to the label/export code:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
  tests/test_generate_teacher_labels_script.py \
  tests/test_export_teacher_labels_cli.py \
  tests/test_vla_export_lerobot.py \
  tests/test_verify_ego_lerobot_cli.py \
  tests/test_smoke_lerobot_training_cli.py \
  tests/test_replay_lerobot_teacher_mujoco_cli.py \
  -q
```

These tests cover sharding, export hardening, label format, LeRobot row
conversion, dataset smoke loading, and replay command construction. They do not
replace GPU/Isaac behavior verification.

## Common Failures

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `teacher record is missing final motion_token` | Running against a SONIC eval tree without the debug `motion_token` field. | Use this repository's `imports/SONIC` tree and rerun eval. |
| `labels shorter than motion_lib` | Eval ended before the motion horizon. | Confirm strict behavior, then use the production wrapper with disabled non-timeout terminations. |
| Duplicate `frame_index` errors | Overflow env replay duplicated a motion. | Use the wrapper defaults, which pass `--dedupe-overflow-envs`. |
| Missing BPS warning | `<motion-lib>/bps` not present. | For production, pass a real `--bps-dir`. For smoke only, understand that zero BPS changes observations. |
| LeRobot export fails on missing labels | Label root does not contain `<motion_key>.npy` for each rendered motion. | Use `--motion-keys-file` for exact subsets or regenerate missing labels. |
| MuJoCo replay motion is meaningless | Decoder ONNX does not match the ATM label space. | Export/use the decoder from `imports/SONIC/models/sonic_manipulation_base/last.pt`. |

