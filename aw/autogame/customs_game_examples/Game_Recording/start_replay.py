#!/usr/bin/env python3
"""无需 Launcher，选择一条 Game_Recording 历史记录并回放。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    default_records = (
        REPO_ROOT
        / "aw"
        / "autogame"
        / "Game_Recording_records"
    )
    parser = argparse.ArgumentParser(
        description="弹窗选择 Game_Recording 历史记录并回放。",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=default_records,
        help="录制记录根目录",
    )
    parser.add_argument(
        "--video-so",
        default="auto",
        help=(
            "HOS 投屏 SO：auto 动态尝试全部候选（默认）；"
            "latest 只强制最新候选；reuse 复用设备现有版本；"
            "也可填写完整 SO 文件名"
        ),
    )
    parser.add_argument(
        "--touch-backend",
        choices=("hos", "sendevent"),
        default="hos",
        help="触控后端：hos（默认）或 sendevent",
    )
    parser.add_argument(
        "--sendevent-device",
        default="",
        help="手动指定触摸设备，例如 event2；留空自动探测",
    )
    parser.add_argument("--sendevent-max-x", type=int, default=None)
    parser.add_argument("--sendevent-max-y", type=int, default=None)
    args = parser.parse_args(argv)
    args.video_so = str(args.video_so or "").strip()
    if not args.video_so:
        parser.error("--video-so 不能为空")
    manual = (
        bool(str(args.sendevent_device or "").strip()),
        args.sendevent_max_x is not None,
        args.sendevent_max_y is not None,
    )
    if any(manual) and not all(manual):
        parser.error(
            "手动配置 sendevent 时必须同时提供设备、max-x 和 max-y"
        )
    if all(manual) and args.touch_backend != "sendevent":
        parser.error("sendevent 设备参数只能和 --touch-backend sendevent 一起使用")
    if args.sendevent_max_x is not None and args.sendevent_max_x <= 0:
        parser.error("--sendevent-max-x 必须大于 0")
    if args.sendevent_max_y is not None and args.sendevent_max_y <= 0:
        parser.error("--sendevent-max-y 必须大于 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    from aw.autogame.customs_examples.Game_Recording.resource.replay_app import run

    return int(
        run(
            records_root=args.records.expanduser().resolve(),
            video_so=args.video_so,
            touch_backend=args.touch_backend,
            sendevent_device=args.sendevent_device,
            sendevent_max_x=args.sendevent_max_x,
            sendevent_max_y=args.sendevent_max_y,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
