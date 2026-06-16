from __future__ import annotations

import argparse
from typing import Optional

from grail.vla.export_lerobot import export_rendered_jobs, select_rendered_jobs
from grail.vla.lerobot_schema import DEFAULT_TASK
from grail.vla.render_jobs import build_render_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GRAIL motion library into an ego-view LeRobot dataset."
    )
    parser.add_argument("--motion-lib", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ego-frame-root")
    parser.add_argument("--token-labels")
    parser.add_argument("--allow-empty-token-labels", action="store_true")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to an existing LeRobot output directory instead of requiring a fresh output",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing LeRobot output directory before writing",
    )
    parser.add_argument(
        "--only-rendered",
        action="store_true",
        help="Export only motions with completed ego-frame sources under --ego-frame-root",
    )
    parser.add_argument(
        "--require-all-rendered",
        action="store_true",
        help="Fail if any planned motion is missing a rendered ego-frame source",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Limit exported episodes after optional rendered-source filtering; use 0 for no limit",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = build_render_jobs(args.motion_lib, args.output, args.shard_index, args.num_shards)
    missing = None
    if args.ego_frame_root and (args.only_rendered or args.require_all_rendered or args.max_episodes is not None):
        jobs, missing = select_rendered_jobs(
            jobs,
            args.ego_frame_root,
            only_rendered=args.only_rendered,
            max_episodes=args.max_episodes,
            require_all_rendered=args.require_all_rendered,
        )
    if args.dry_run:
        print(f"Planned {len(jobs)} render jobs")
        if missing is not None:
            print(f"Missing rendered ego sources: {missing}")
        return 0
    if args.ego_frame_root:
        exported = export_rendered_jobs(
            jobs,
            ego_frame_root=args.ego_frame_root,
            output_dir=args.output,
            token_label_root=args.token_labels,
            task=args.task,
            allow_empty_token_labels=args.allow_empty_token_labels,
            append_existing=args.append_existing,
            overwrite_existing=args.overwrite_existing,
        )
        print(f"Exported {exported} LeRobot episode(s)")
        return 0
    raise NotImplementedError("Full render/export path will be enabled after ego renderer lands.")


if __name__ == "__main__":
    raise SystemExit(main())
