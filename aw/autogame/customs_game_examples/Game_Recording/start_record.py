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
    output_root = args.output.expanduser().resolve()
    from aw.autogame.customs_examples.Game_Recording.resource.runtime_log import (
        RuntimeLogCapture,
    )

    with RuntimeLogCapture(output_root) as runtime_log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True,
        )
        print(f"[Game Recording] 运行日志：{runtime_log.path}", flush=True)
        from aw.autogame.customs_examples.Game_Recording.resource.app import run

        try:
            return run(
                output_root=output_root,
                fps=args.fps,
                runtime_log_path=runtime_log.path,
            )
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(f"[Game Recording] 启动或运行失败：{exc}", file=sys.stderr, flush=True)
            raise
        except Exception:
            logging.exception("Game Recording 未处理异常")
            raise
        finally:
            # logging handler 持有 TeeTextIO，需在日志文件关闭前先收尾。
            logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
