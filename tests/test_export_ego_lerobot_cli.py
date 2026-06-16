from grail.cli.export_ego_lerobot import build_parser


def test_cli_parser_defaults():
    args = build_parser().parse_args(["--motion-lib", "data/pickup_table", "--output", "outputs/ds"])

    assert args.motion_lib == "data/pickup_table"
    assert args.output == "outputs/ds"
    assert args.shard_index == 0
    assert args.num_shards == 1
    assert args.task == "perform the demonstrated manipulation task"
    assert args.allow_empty_token_labels is False
    assert args.append_existing is False
    assert args.overwrite_existing is False
    assert args.ego_frame_root is None


def test_cli_parser_accepts_rendered_frame_inputs():
    args = build_parser().parse_args(
        [
            "--motion-lib",
            "data/pickup_table",
            "--output",
            "outputs/ds",
            "--ego-frame-root",
            "outputs/ego_frames",
            "--token-labels",
            "outputs/tokens",
            "--append-existing",
        ]
    )

    assert args.ego_frame_root == "outputs/ego_frames"
    assert args.token_labels == "outputs/tokens"
    assert args.append_existing is True
