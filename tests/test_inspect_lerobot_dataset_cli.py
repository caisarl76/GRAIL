import json

import numpy as np

from grail.cli.inspect_lerobot_dataset import inspect_dataset, main, validate_episode_table


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_dataset(root, *, include_video=True, total_episodes=1):
    _write_json(
        root / "meta" / "info.json",
        {
            "total_episodes": total_episodes,
            "total_frames": 2,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "video_keys": ["observation.images.ego_view"],
        },
    )
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "length": 2}],
    )
    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"placeholder")
    if include_video:
        video_path = root / "videos" / "observation.images.ego_view" / "episode_000000.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"placeholder")


def test_inspect_dataset_accepts_existing_files_without_reading_parquet(tmp_path):
    _write_dataset(tmp_path)

    result = inspect_dataset(tmp_path, read_parquet=False)

    assert result.ok is True
    assert result.episodes == 1
    assert result.episode_frames == 2
    assert result.video_keys == ["observation.images.ego_view"]
    assert result.missing_parquet == []
    assert result.missing_videos == []


def test_inspect_dataset_reports_missing_video(tmp_path):
    _write_dataset(tmp_path, include_video=False)

    result = inspect_dataset(tmp_path, read_parquet=False)

    assert result.ok is False
    assert len(result.missing_videos) == 1
    assert "observation.images.ego_view" in str(result.missing_videos[0])


def test_inspect_dataset_reports_extra_unreferenced_files(tmp_path):
    _write_dataset(tmp_path)
    extra_parquet = tmp_path / "data" / "chunk-000" / "episode_000001.parquet"
    extra_parquet.write_bytes(b"extra")
    extra_video = tmp_path / "videos" / "observation.images.ego_view" / "episode_000001.mp4"
    extra_video.write_bytes(b"extra")

    result = inspect_dataset(tmp_path, read_parquet=False)

    assert result.ok is False
    assert result.extra_parquet == [extra_parquet]
    assert result.extra_videos == [extra_video]


def test_inspect_dataset_reports_metadata_count_mismatch(tmp_path):
    _write_dataset(tmp_path, total_episodes=2)

    result = inspect_dataset(tmp_path, read_parquet=False)

    assert result.ok is False
    assert result.errors == ["info total_episodes=2, episodes.jsonl has 1"]


def test_inspect_dataset_cli_returns_nonzero_for_missing_video(tmp_path, capsys):
    _write_dataset(tmp_path, include_video=False)

    exit_code = main(["--root", str(tmp_path), "--no-read-parquet", "--max-episodes", "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing_videos: 1" in captured.out


def test_validate_episode_table_reports_timestamp_reset(tmp_path):
    table = {
        "episode_index": np.full((2,), 261),
        "frame_index": np.array([0, 1]),
        "timestamp": np.array([9.96, 0.0]),
    }

    errors = validate_episode_table(
        table,
        parquet_path=tmp_path / "episode_000261.parquet",
        episode_index=261,
        episode_length=2,
        fps=25,
    )

    assert any("timestamp" in error for error in errors)


def test_validate_episode_table_reports_missing_hand_primitive(tmp_path):
    table = {
        "episode_index": np.zeros((2,), dtype=np.int64),
        "frame_index": np.array([0, 1]),
        "timestamp": np.array([0.0, 0.04]),
        "observation.state": np.zeros((2, 43), dtype=np.float32),
        "action.wbc": np.zeros((2, 43), dtype=np.float32),
        "action.motion_token": np.zeros((2, 64), dtype=np.float32),
    }

    errors = validate_episode_table(
        table,
        parquet_path=tmp_path / "episode_000000.parquet",
        episode_index=0,
        episode_length=2,
        fps=25,
    )

    assert any("action.hand_primitive" in error for error in errors)
