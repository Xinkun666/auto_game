#!/usr/bin/env python3
"""对已手动到达的当前房屋执行一次南大原流程房型匹配。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "aw" / "autogame" / "temp" / "results" / "room_match_once"
)
DETAILS_FILENAME = "匹配详情.json"
SCREEN_RECORDER_BUNDLE = "com.huawei.hmos.screenrecorder"
SCREEN_RECORDER_ABILITY = (
    "com.huawei.hmos.screenrecorder.ServiceExtAbility"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "手动走到门前后，从 HOScrcpy 取当前画面，按南大原方案执行门框取景、"
            "必要时后拉和动态抬头，并进行 building/door frame/window + "
            "DINOv3/MLP 房型匹配；只处理当前一栋房屋，不执行进屋回放。"
        )
    )
    parser.add_argument(
        "--expected-room",
        default="",
        help="可选的真实 room_id；填写后结果 JSON 会直接给出 correct",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        default="",
        help=(
            "结果保存根目录；每次运行会在其中新建 YYYYMMDD_HHMMSS 目录，"
            "默认写入 room_match_once 结果目录"
        ),
    )
    parser.add_argument(
        "--room-library",
        default="",
        help="包含 rooms/ 的 room_library 根目录",
    )
    parser.add_argument("--dino-model-dir", default="", help="DINOv3 模型目录")
    parser.add_argument("--mlp-model-path", default="", help="MLP pkl 模型路径")
    parser.add_argument("--device", default="", help="可选 HOS/HDC 设备 SN")
    parser.add_argument("--ip", default="", help="HOScrcpy SDK IP")
    parser.add_argument("--port", type=int, default=0, help="HOScrcpy/HDC 端口")
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="包含首次模型预热的总超时秒数，默认 900",
    )
    return parser


def _set_if_value(name: str, value) -> None:
    text = str(value or "").strip()
    if text:
        os.environ[name] = text


def _create_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate = base_dir / f"{timestamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"无法创建本次匹配结果目录: {base_dir}")


def _configure_environment(args: argparse.Namespace) -> Path:
    os.chdir(REPO_ROOT)
    os.environ["TARGET_PROJECT_CASE"] = "Auto_PUBG_ALL"
    os.environ["TARGET_GAME_CASE"] = "room_match_once"
    os.environ["AUTOGAME_TEST_PROFILE"] = "function"
    os.environ["AUTOGAME_PRESERVE_GAME_PROCESS"] = "1"
    os.environ["AUTOGAME_DISABLE_SAVE_FRAMES"] = "1"
    os.environ["AUTOGAME_VIS_MODE"] = "launcher"
    _set_if_value("AUTOGAME_EXPECTED_ROOM_ID", args.expected_room)
    _set_if_value("AUTOGAME_NANDA_ROOM_LIBRARY", args.room_library)
    _set_if_value("AUTOGAME_NANDA_DINO_MODEL_DIR", args.dino_model_dir)
    _set_if_value("AUTOGAME_NANDA_MLP_MODEL_PATH", args.mlp_model_path)
    _set_if_value("AUTOGAME_HOSCRCPY_SN", args.device)
    _set_if_value("AUTOGAME_HOSCRCPY_IP", args.ip)
    _set_if_value("AUTOGAME_HOSCRCPY_PORT", args.port)

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR
    )
    run_dir = _create_run_dir(output_root)
    os.environ["AUTOGAME_ROOM_MATCH_OUTPUT_DIR"] = str(run_dir)
    os.environ.pop("AUTOGAME_ROOM_MATCH_OUTPUT", None)
    return run_dir / DETAILS_FILENAME


def _read_result(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_screen_recorder_command(command: str, action: str) -> None:
    from aw.autogame.tools.ProcessUtils import (
        hdc_command_args,
        hidden_subprocess_kwargs,
    )

    command_args = hdc_command_args(command)
    if not command_args:
        raise RuntimeError(f"无法生成{action}录屏的 HDC 命令")

    try:
        completed = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{action}录屏命令执行失败: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"{action}录屏命令返回码 {completed.returncode}{suffix}"
        )


def _start_screen_recording(run_dir: Path) -> str:
    recording_filename = f"{run_dir.name}.mp4"
    command = (
        f"hdc shell aa start -b {SCREEN_RECORDER_BUNDLE} "
        f"-a {SCREEN_RECORDER_ABILITY} "
        f'--ps "CustomizedFileName" "{recording_filename}"'
    )
    _run_screen_recorder_command(command, "开启")
    return recording_filename


def _stop_screen_recording() -> None:
    command = (
        f"hdc shell aa start -b {SCREEN_RECORDER_BUNDLE} "
        f"-a {SCREEN_RECORDER_ABILITY}"
    )
    _run_screen_recorder_command(command, "关闭")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于 0")
    output = _configure_environment(args)

    from aw.autogame.stream_client.stream_client import (
        HOSScrcpyStreamClient,
        global_buffer,
    )
    from aw.autogame.tools.GameFrameWorker import FrameWorker

    logging.basicConfig(level=logging.INFO)
    client = HOSScrcpyStreamClient(global_buffer)
    worker = FrameWorker(
        global_buffer,
        driver=None,
        logger=logging.getLogger("RoomMatchOnce"),
        stream_client=client,
    )
    # 服务器脚本不需要实时可视化子进程；结果会保留查询帧。
    worker._start_visualizer_process = lambda *args, **kwargs: None

    stop_requested = threading.Event()

    def monitor() -> None:
        deadline = time.monotonic() + float(args.timeout)
        while worker.running and not stop_requested.wait(0.2):
            if time.monotonic() < deadline:
                continue
            worker.mark_failed(
                "room_match_once_timeout",
                f"单次房型匹配超过 {args.timeout:g} 秒",
                result_path=str(output),
            )
            worker.stop()
            break
        client.stop()

    monitor_thread = None
    recording_attempted = False
    execution_error = None
    interrupted = False
    try:
        recording_attempted = True
        recording_filename = _start_screen_recording(output.parent)
        print(f"录屏已开启: {recording_filename}")

        worker.start()
        monitor_thread = threading.Thread(
            target=monitor,
            name="RoomMatchOnceMonitor",
            daemon=True,
        )
        monitor_thread.start()
        client.run()
    except KeyboardInterrupt:
        print("收到中断，停止单次房型匹配。")
        interrupted = True
    except Exception as exc:
        execution_error = exc
    finally:
        stop_requested.set()
        client.stop()
        worker.stop()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        if recording_attempted:
            try:
                _stop_screen_recording()
                print("录屏已关闭。")
            except Exception as exc:
                print(f"关闭录屏失败: {exc}")
                if execution_error is None and not interrupted:
                    execution_error = exc

    if interrupted:
        return 130
    if execution_error is not None:
        print(f"单次房型匹配执行失败: {execution_error}")
        return 1

    payload = _read_result(output)
    status = payload.get("status")
    print(f"结果目录: {output.parent}")
    print(f"详细结果: {output}")
    if status == "matched":
        print(
            f"匹配结果: room_id={payload.get('room_id')}, "
            f"score={payload.get('score')}, correct={payload.get('correct')}"
        )
        return 0
    if status == "no_match":
        print(
            f"匹配结果: NO_MATCH, reason={payload.get('no_match_reason')}, "
            f"top_candidates={payload.get('top_candidates')}"
        )
        return 2
    print(f"匹配失败: {payload.get('error') or worker.failure_reason or '未知错误'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
