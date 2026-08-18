#!/usr/bin/env python3
"""无需 Launcher，直接启动华为 HOS 键盘录制。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    default_output = (
        REPO_ROOT
        / "aw"
        / "autogame"
        / "customs_examples"
        / "Game_Recording"
        / "records"
    )
    parser = argparse.ArgumentParser(
        description="读取 Game_Recording/info.py，按 q 开始、按 e 停止录制。",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="录制结果目录")
    parser.add_argument("--fps", type=float, default=15.0, help="本地 MP4 帧率，默认 15")
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from aw.autogame.customs_examples.Game_Recording.resource.app import run

    return run(output_root=args.output.expanduser().resolve(), fps=args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
