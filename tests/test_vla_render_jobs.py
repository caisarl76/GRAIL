import pytest

from grail.vla.render_jobs import build_render_jobs


def test_build_render_jobs_shards_sorted_motion_files(tmp_path):
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    for name in ["c.pkl", "a.pkl", "b.pkl"]:
        (robot_dir / name).write_text("x", encoding="utf-8")

    jobs = build_render_jobs(tmp_path, output_dir=tmp_path / "ego", shard_index=1, num_shards=2)

    assert [job.motion_key for job in jobs] == ["b"]
    assert jobs[0].video_path == tmp_path / "ego" / "b.mp4"


def test_build_render_jobs_missing_robot_dir_explains_expected_motion_lib_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected a GRAIL motion library root"):
        build_render_jobs(tmp_path / "missing_pickup_table", output_dir=tmp_path / "ego")


def test_build_render_jobs_missing_robot_dir_recommends_missing_root_safe_find(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        build_render_jobs(tmp_path / "missing_pickup_table", output_dir=tmp_path / "ego")

    message = str(exc_info.value)
    assert "for root in /workspace/grail /data" in message
    assert '[ -d "$root" ]' in message
