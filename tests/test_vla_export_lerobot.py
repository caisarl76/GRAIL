import numpy as np
import joblib

from grail.vla.episode import VLAEpisode, VLAFrame
from grail.vla.export_lerobot import export_rendered_jobs, select_rendered_jobs, write_episode_to_exporter
from grail.vla.render_jobs import build_render_jobs


class FakeExporter:
    def __init__(self):
        self.frames = []
        self.saved = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved = True


def _episode() -> VLAEpisode:
    frame = VLAFrame(
        frame_index=0,
        timestamp=0.0,
        joint_position=np.ones(43, dtype=np.float32),
        root_position=np.zeros(3, dtype=np.float32),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        object_position=np.zeros(3, dtype=np.float32),
        object_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        motion_token=np.full(64, 0.5, dtype=np.float32),
        hand_primitive=np.array([-1.0, 1.0], dtype=np.float32),
    )
    return VLAEpisode(
        motion_key="pickup_table__can__000",
        fps=25.0,
        frames=[frame],
        robot_path=None,
        object_path=None,
        object_usd_path=None,
    )


def test_write_episode_to_exporter_adds_image_and_saves():
    exporter = FakeExporter()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    write_episode_to_exporter(exporter, _episode(), [image], task="pick up the can")

    assert exporter.saved is True
    assert len(exporter.frames) == 1
    row = exporter.frames[0]
    assert "timestamp" not in row
    assert row["task"] == "pick up the can"
    assert row["observation.images.ego_view"].shape == (480, 640, 3)
    assert row["action.motion_token"].shape == (64,)
    assert row["action.hand_primitive"].shape == (2,)
    assert np.allclose(row["action.motion_token"], 0.5)
    assert np.allclose(row["action.hand_primitive"], [-1.0, 1.0])


def _write_motion(root, key: str = "pickup_table__can__000"):
    for subdir in ["robot", "objects", "meta"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    robot = {
        key: {
            "dof": np.zeros((2, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
            "fps": 25.0,
            "hand_dof_pos": np.zeros((2, 14), dtype=np.float32),
        }
    }
    objects = {
        key: {
            "root_pos": np.zeros((2, 1, 3), dtype=np.float32),
            "root_quat": np.tile(
                np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), (2, 1, 1)
            ),
        }
    }
    joblib.dump(robot, root / "robot" / f"{key}.pkl")
    joblib.dump(objects, root / "objects" / f"{key}.pkl")
    joblib.dump({"object_name": "can"}, root / "meta" / f"{key}.pkl")


def test_export_rendered_jobs_attaches_loaded_frames_and_tokens(tmp_path):
    _write_motion(tmp_path)
    frames_root = tmp_path / "ego_frames"
    token_root = tmp_path / "tokens"
    frames_root.mkdir()
    token_root.mkdir()
    np.save(frames_root / "pickup_table__can__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    labels = np.zeros((2, 66), dtype=np.float32)
    labels[:, :64] = 0.75
    labels[:, 64:] = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    np.save(token_root / "pickup_table__can__000.npy", labels)
    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")
    created = []

    def exporter_factory(save_root, fps, task, script_config=None):
        exporter = FakeExporter()
        exporter.save_root = save_root
        exporter.fps = fps
        exporter.task = task
        exporter.script_config = script_config
        created.append(exporter)
        return exporter

    exported = export_rendered_jobs(
        jobs,
        ego_frame_root=frames_root,
        output_dir=tmp_path / "lerobot",
        token_label_root=token_root,
        task="pick up the can",
        exporter_factory=exporter_factory,
    )

    assert exported == 1
    assert len(created) == 1
    assert created[0].fps == 25
    assert created[0].saved is True
    assert len(created[0].frames) == 2
    assert created[0].frames[0]["task"] == "pick up the can"
    assert created[0].frames[0]["observation.images.ego_view"].shape == (8, 8, 3)
    assert np.allclose(created[0].frames[0]["action.motion_token"], 0.75)
    assert np.allclose(created[0].frames[0]["action.hand_primitive"], [1.0, -1.0])
    assert np.allclose(created[0].frames[1]["action.hand_primitive"], [0.5, -0.5])


def test_export_rendered_jobs_refuses_existing_output_without_append_or_overwrite(tmp_path):
    _write_motion(tmp_path)
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    np.save(frames_root / "pickup_table__can__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    output_dir = tmp_path / "lerobot"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("previous export", encoding="utf-8")
    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")

    try:
        export_rendered_jobs(
            jobs,
            ego_frame_root=frames_root,
            output_dir=output_dir,
            allow_empty_token_labels=True,
            exporter_factory=lambda *args: FakeExporter(),
        )
    except FileExistsError as exc:
        assert "--append-existing" in str(exc)
        assert "--overwrite-existing" in str(exc)
    else:
        raise AssertionError("expected existing output directory failure")


def test_export_rendered_jobs_can_explicitly_append_to_existing_output(tmp_path):
    _write_motion(tmp_path)
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    np.save(frames_root / "pickup_table__can__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    output_dir = tmp_path / "lerobot"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("previous export", encoding="utf-8")
    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")

    exported = export_rendered_jobs(
        jobs,
        ego_frame_root=frames_root,
        output_dir=output_dir,
        allow_empty_token_labels=True,
        append_existing=True,
        exporter_factory=lambda *args: FakeExporter(),
    )

    assert exported == 1


def test_export_rendered_jobs_refuses_existing_export_lock(tmp_path):
    _write_motion(tmp_path)
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    np.save(frames_root / "pickup_table__can__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    output_dir = tmp_path / "lerobot"
    (tmp_path / ".lerobot.grail_export.lock").write_text("locked", encoding="utf-8")
    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")

    try:
        export_rendered_jobs(
            jobs,
            ego_frame_root=frames_root,
            output_dir=output_dir,
            allow_empty_token_labels=True,
            exporter_factory=lambda *args: FakeExporter(),
        )
    except FileExistsError as exc:
        assert "already writing" in str(exc)
    else:
        raise AssertionError("expected export lock failure")


def test_select_rendered_jobs_limits_to_completed_sources(tmp_path):
    _write_motion(tmp_path, key="pickup_table__can__000")
    _write_motion(tmp_path, key="pickup_table__cup__000")
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    np.save(frames_root / "pickup_table__cup__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))

    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")
    selected, missing = select_rendered_jobs(jobs, frames_root, only_rendered=True)

    assert [job.motion_key for job in selected] == ["pickup_table__cup__000"]
    assert missing == 1


def test_select_rendered_jobs_applies_max_episodes_after_completed_filter(tmp_path):
    _write_motion(tmp_path, key="pickup_table__can__000")
    _write_motion(tmp_path, key="pickup_table__cup__000")
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    np.save(frames_root / "pickup_table__can__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    np.save(frames_root / "pickup_table__cup__000.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))

    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")
    selected, missing = select_rendered_jobs(jobs, frames_root, only_rendered=True, max_episodes=1)

    assert len(selected) == 1
    assert selected[0].motion_key == "pickup_table__can__000"
    assert missing == 0


def test_select_rendered_jobs_can_require_all_rendered_sources(tmp_path):
    _write_motion(tmp_path, key="pickup_table__can__000")
    frames_root = tmp_path / "ego_frames"
    frames_root.mkdir()
    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego")

    try:
        select_rendered_jobs(jobs, frames_root, only_rendered=True, require_all_rendered=True)
    except FileNotFoundError as exc:
        assert "Missing rendered ego sources: 1" in str(exc)
    else:
        raise AssertionError("expected missing rendered source failure")
