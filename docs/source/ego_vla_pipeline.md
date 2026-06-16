# Ego-View VLA Pipeline

This page tracks the repository-local path for turning released GRAIL motions
into an ego-view LeRobot dataset that can be used for GR00T/VLA training and
then checked in simulation before real robot deployment.

## Scope

GRAIL release data gives the motion side of the problem: robot trajectories,
object trajectories, object assets, metadata, and source HOI videos. It does
not ship a ready-made LeRobot dataset with robot ego-view RGB frames and
per-step VLA action rows. The converter code under `grail/vla/` is the bridge:

1. Load the GRAIL motion library.
2. Shard episodes into render/export jobs.
3. Render robot ego-view RGB frames in Isaac Sim.
4. Attach SONIC teacher motion-token labels where available.
5. Write the episode rows with the SONIC VLA LeRobot schema.

The current checked-in implementation covers steps 1, 2, schema conversion,
token-label validation, offline LeRobot export from pre-rendered ego frames,
ego-render planning/manifests, and the Isaac Sim RGB capture loop behind
`grail.visualization.batch_render_ego`.

## GPU Placement

Use different machines for rendering and training:

- RTX GPU server: Isaac Sim ego-view rendering, visual smoke checks, and
  simulation verification. Isaac Sim rendering requires RTX ray-tracing
  support; do not plan the render stage on H100/A100-only hosts.
- H100 GPU server: GR00T-N1.7 or other VLA training, large-scale data loading,
  and distillation training once the LeRobot dataset already exists.

This keeps the expensive H100 training box focused on model work and avoids a
renderer that cannot use that hardware class.

## Docker Entry Point

Use the wrapper to run the converter through the same GRAIL container launcher
used by the rest of the repo:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --motion-lib /data/grail/pickup_table \
  --output /data/lerobot/pickup_table \
  --num-shards 8 \
  --shard-index 0 \
  --dry-run
```

On an RTX Isaac Sim node, plan ego-view rendering with `--render-only`. The
current local path writes per-motion render manifests without importing Isaac;
the full render loop lives behind `grail.visualization.batch_render_ego`:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --render-only \
  --motion-lib /data/grail/pickup_table \
  --output /data/ego_frames/pickup_table \
  --resolution 640x480 \
  --num-shards 8 \
  --shard-index 0 \
  --dry-run \
  --write-manifests
```

The default simulated ego camera uses the same OAK-style image size as the
SONIC camera server (`640x480`) and OAK-D RGB IMX378 color-camera intrinsics
from Luxonis (`81 deg / 69 deg / 55 deg` DFOV/HFOV/VFOV). The virtual camera
pitch is fixed relative to the robot root: the default ray from
`(0.25, 0.0, 0.55)` to `(0.846, 0.0, -0.301)` looks down into the manipulation
workspace at about `-55.0 deg`. This is the steeper test mount used to examine
whether objects stay visible in the robot ego camera. This renderer does not
actuate a neck or recompute the camera pose from G1 waist/head links. The
released SONIC OAK driver reads the physical camera as mounted on the robot; it
does not publish a calibrated extrinsic angle. To tune the virtual mount against
real OAK footage, override the offsets explicitly:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --render-only \
  --motion-lib /data/grail/pickup_table \
  --output /data/ego_frames/pickup_table \
  --resolution 640x480 \
  --camera-position-offset 0.25 0.0 0.55 \
  --camera-target-offset 0.846 0.0 -0.301 \
  --write-manifests
```

To diagnose whether the robot can actually see the target object, render both
the ego camera and a third-eye verification view for the same selected episodes:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --render-only \
  --with-third-eye-comparison \
  --motion-lib /data/grail/pickup_table \
  --output /data/ego_frames/pickup_table \
  --side-by-side-output /data/ego_compare/pickup_table \
  --resolution 640x480 \
  --max-motions 5 \
  --write-manifests
```

This produces ego-view videos in `--output`, third-eye intermediate videos in
`<output>_third_eye` unless `--third-eye-output` is provided, and side-by-side
diagnostic videos in `--side-by-side-output`. The wrapper writes
`motion_keys.txt` into the side-by-side output directory so both render passes
use the exact same episodes.

For full training-data rendering, use the dual-GPU launcher from inside the
GRAIL container. It renders the GRAIL `pickup_table` motion library as
`pnp_table` on GPU 0 and `pickup_ground` as `pnp_ground` on GPU 1:

```bash
bash scripts/docker/render_pnp_ego_dual_gpu.sh \
  --data-root /data/grail \
  --output-root /data/ego_frames \
  --log-root /data/logs/ego_render \
  --resolution 640x480
```

On a cluster login node, inspect the command without launching Docker:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --motion-lib /data/grail/pickup_table \
  --output /data/lerobot/pickup_table \
  --num-shards 8 \
  --shard-index 0 \
  --dry-run \
  --print-command
```

For real export runs, pass teacher token labels once the SONIC teacher labeling
job is available:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --motion-lib /data/grail/pickup_table \
  --output /data/lerobot/pickup_table \
  --token-labels /data/sonic_tokens/pickup_table \
  --num-shards 8 \
  --shard-index 0
```

During early renderer bring-up only, `--allow-empty-token-labels` can be used
to verify camera rendering and row formatting before teacher labels are ready.
Do not use unlabeled episodes for final VLA training.

After ego frames have been rendered, run the offline LeRobot export path with
`--ego-frame-root`:

```bash
bash scripts/docker/render_ego_lerobot.sh \
  --motion-lib /data/grail/pickup_table \
  --ego-frame-root /data/ego_frames/pickup_table \
  --output /data/lerobot/pickup_table \
  --token-labels /data/sonic_tokens/pickup_table \
  --num-shards 8 \
  --shard-index 0
```

The current pure-Python exporter accepts ego frames as `<motion_key>.npy`,
`<motion_key>/000000.npy` frame directories, image frame directories, or videos
once the runtime image/video dependencies are installed.

## LeRobot Row Shape

The converter targets the SONIC VLA feature shape from
`imports/SONIC/gear_sonic/data/features_sonic_vla.py`. The repo-local boundary
currently includes:

- `observation.images.ego_view`: RGB ego camera frame.
- Robot proprioception fields from the retargeted G1 trajectory, including
  joint state and base/eef orientation fields required by the SONIC VLA schema.
- `action.motion_token`: SONIC teacher token label for the frame. This is the
  supervised action target, not an observation.
- `task`: Natural-language task string.

Object trajectories and object USD assets are used for Isaac Sim replay and
ego-view rendering only. Do not export privileged object position/quaternion
fields as GR00T observations for real-robot training.

The nursing-home tasks should be split into separate datasets or task labels:

- Approach and pick a drink can from a table.
- Pick a box from the ground and place it on a shelf.
- Pick trash from the ground and place it in a trash bin.
- Collect clothes on a bed and place them in a laundry basket.

Keep task labels concrete and consistent because they become part of the VLA
conditioning data.
