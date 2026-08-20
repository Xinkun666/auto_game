#!/usr/bin/env python3
"""无需 Launcher，直接启动华为 HOS 键盘录制。"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def _stop_cleanly_on_termination_signals():
    """将 IDE 停止、终端关闭等终止信号转为可收尾的 SystemExit。"""
    previous_handlers = {}

    def request_exit(signum, _frame):
        raise SystemExit(128 + int(signum))

    for signal_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_exit)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


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
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="录制批次根目录，每次启动会在其下创建时间目录",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="本地 MP4 帧率，默认 15")
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
    parser.add_argument(
        "--sendevent-max-x",
        type=int,
        default=None,
        help="手动指定触摸面板 ABS X 最大值",
    )
    parser.add_argument(
        "--sendevent-max-y",
        type=int,
        default=None,
        help="手动指定触摸面板 ABS Y 最大值",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    args.video_so = str(args.video_so or "").strip()
    if not args.video_so:
        parser.error("--video-so 不能为空")
    sendevent_manual = (
        bool(str(args.sendevent_device or "").strip()),
        args.sendevent_max_x is not None,
        args.sendevent_max_y is not None,
    )
    if any(sendevent_manual) and not all(sendevent_manual):
        parser.error(
            "手动配置 sendevent 时必须同时提供 "
            "--sendevent-device、--sendevent-max-x 和 --sendevent-max-y"
        )
    if all(sendevent_manual) and args.touch_backend != "sendevent":
        parser.error("sendevent 设备参数只能和 --touch-backend sendevent 一起使用")
    if args.sendevent_max_x is not None and args.sendevent_max_x <= 0:
        parser.error("--sendevent-max-x 必须大于 0")
    if args.sendevent_max_y is not None and args.sendevent_max_y <= 0:
        parser.error("--sendevent-max-y 必须大于 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    records_root = args.output.expanduser().resolve()
    from aw.autogame.customs_examples.Game_Recording.resource.runtime_log import (
        HilogCapture,
        RuntimeLogCapture,
        create_run_directory,
        save_run_summary,
    )

    started_at = datetime.now().astimezone()
    run_dir = create_run_directory(records_root, now=started_at)
    exit_code = 1
    outcome = "failed"
    run_error = ""
    try:
        with (
            _stop_cleanly_on_termination_signals(),
            RuntimeLogCapture(run_dir) as runtime_log,
            HilogCapture(run_dir) as hilog,
        ):
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                force=True,
            )
            print(f"[Game Recording] 本次记录目录：{run_dir}", flush=True)
            print(f"[Game Recording] 运行日志：{runtime_log.path}", flush=True)
            print(f"[Game Recording] hilog 实时抓取：{hilog.path}", flush=True)
            if hilog.start_error:
                print(
                    f"[Game Recording] hilog 抓取启动失败：{hilog.start_error}",
                    file=sys.stderr,
                    flush=True,
                )
            from aw.autogame.customs_examples.Game_Recording.resource.app import run

            try:
                app_exit_code = int(
                    run(
                        output_root=run_dir,
                        fps=args.fps,
                        runtime_log_path=runtime_log.path,
                        hilog_capture=hilog,
                        video_so=args.video_so,
                        touch_backend=args.touch_backend,
                        sendevent_device=args.sendevent_device,
                        sendevent_max_x=args.sendevent_max_x,
                        sendevent_max_y=args.sendevent_max_y,
                    )
                )
                if (run_dir / "hos_disconnect.json").is_file():
                    exit_code = 1
                    run_error = "HOS disconnect"
                else:
                    exit_code = app_exit_code
                outcome = "success" if exit_code == 0 else "failed"
                return exit_code
            except SystemExit as exc:
                if exc.code is None:
                    exit_code = 0
                else:
                    exit_code = int(exc.code) if isinstance(exc.code, int) else 1
                outcome = "success" if exit_code == 0 else "failed"
                run_error = "" if exit_code == 0 else str(exc)
                if exit_code != 0:
                    print(
                        f"[Game Recording] 启动或运行失败：{exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                raise
            except Exception as exc:
                run_error = str(exc)
                logging.exception("Game Recording 未处理异常")
                raise
            finally:
                # logging handler 持有 TeeTextIO，需在日志文件关闭前先收尾。
                logging.shutdown()
    except KeyboardInterrupt:
        exit_code = 130
        run_error = "KeyboardInterrupt"
        raise
    except BaseException as exc:
        if outcome == "failed" and not run_error:
            run_error = str(exc) or exc.__class__.__name__
        raise
    finally:
        summary_path = save_run_summary(
            run_dir=run_dir,
            started_at=started_at,
            outcome=outcome,
            exit_code=exit_code,
            error=run_error,
        )
        print(f"[Game Recording] 运行摘要：{summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
