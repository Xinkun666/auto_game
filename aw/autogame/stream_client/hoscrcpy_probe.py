import argparse
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


if __package__ in (None, ""):
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "aw" / "autogame").exists():
            sys.path.insert(0, str(parent))
            break

from aw.autogame.stream_client.stream_client import HOSScrcpyStreamClient
from aw.autogame.tools.Utils import resolve_tmp_frames_dir


class HOScrcpyProbeError(RuntimeError):
    """Raised when the HOScrcpy probe cannot start or verify the stream."""


class HOScrcpyStreamTimeout(HOScrcpyProbeError):
    """Raised when the HOScrcpy probe starts but no decoded frame arrives."""


@dataclass(frozen=True)
class HOScrcpyProbeResult:
    elapsed_seconds: float
    frame_count: int
    width: Optional[int]
    height: Optional[int]
    saved_frame_path: Optional[Path]


class ProbeFrameBuffer:
    def __init__(self):
        self.frame_count = 0
        self._frames = queue.Queue(maxsize=1)

    def push(self, frame):
        self.frame_count += 1
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def wait_for_frame(self, timeout: float, client=None):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            client_error = getattr(client, "_last_error", None)
            if client_error is not None:
                raise HOScrcpyProbeError("HOScrcpy 流启动失败: %s" % client_error) from client_error

            main_thread = getattr(client, "main_thread", None)
            client_running = bool(getattr(client, "running", False))
            if main_thread is not None and not main_thread.is_alive() and not client_running:
                raise HOScrcpyProbeError("HOScrcpy 流线程已退出，但没有收到首帧")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HOScrcpyStreamTimeout("等待 HOScrcpy 首帧超时: %.1fs" % float(timeout))

            try:
                return self._frames.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue


def _default_save_frame_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return resolve_tmp_frames_dir() / "hoscrcpy_probe" / ("hoscrcpy_first_frame_%s.jpg" % timestamp)


def _frame_size(frame):
    size = getattr(frame, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    return None, None


def _save_frame(frame, save_frame_path: Optional[Path]) -> Optional[Path]:
    if save_frame_path is None:
        return None
    if not hasattr(frame, "save"):
        raise HOScrcpyProbeError("HOScrcpy 首帧不是可保存的图像对象: %r" % (type(frame),))
    save_frame_path = Path(save_frame_path)
    save_frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(str(save_frame_path), "JPEG")
    return save_frame_path


def probe_hoscrcpy_stream(
    timeout: float = 15.0,
    save_frame_path: Optional[Path] = None,
    rotation_mode: int = 0,
    client_factory: Callable = HOSScrcpyStreamClient,
) -> HOScrcpyProbeResult:
    buffer = ProbeFrameBuffer()
    client = client_factory(buffer, save_frame=False, rotation_mode=rotation_mode)
    start_time = time.monotonic()
    frame = None
    try:
        try:
            client.start_backend()
        except Exception as exc:
            raise HOScrcpyProbeError("HOScrcpy 流启动异常: %s" % exc) from exc

        frame = buffer.wait_for_frame(timeout=timeout, client=client)
        elapsed = time.monotonic() - start_time
        width, height = _frame_size(frame)
        saved_path = _save_frame(frame, save_frame_path)
        return HOScrcpyProbeResult(
            elapsed_seconds=elapsed,
            frame_count=buffer.frame_count,
            width=width,
            height=height,
            saved_frame_path=saved_path,
        )
    finally:
        try:
            client.stop()
        except Exception as exc:
            if frame is not None:
                raise HOScrcpyProbeError("HOScrcpy 流验证成功，但停止客户端失败: %s" % exc) from exc


def _set_env_if_present(env_name: str, value) -> None:
    if value is not None and str(value).strip() != "":
        os.environ[env_name] = str(value).strip()


def apply_hoscrcpy_env_overrides(args) -> None:
    _set_env_if_present("AUTOGAME_HOSCRCPY_SN", args.sn)
    _set_env_if_present("AUTOGAME_HOSCRCPY_IP", args.ip)
    _set_env_if_present("AUTOGAME_HOSCRCPY_PORT", args.port)
    _set_env_if_present("AUTOGAME_HOSCRCPY_SCALE", args.scale)
    _set_env_if_present("AUTOGAME_HOSCRCPY_FRAME_RATE", args.frame_rate)
    _set_env_if_present("AUTOGAME_HOSCRCPY_BIT_RATE", args.bit_rate)
    _set_env_if_present("AUTOGAME_HOSCRCPY_DEVICE_PORT", args.device_port)
    _set_env_if_present("AUTOGAME_HOSCRCPY_ENCODER_TYPE", args.encoder_type)
    _set_env_if_present("AUTOGAME_HOSCRCPY_FPORT_RETRIES", args.fport_retries)
    _set_env_if_present("AUTOGAME_HOSCRCPY_FPORT_RETRY_DELAY", args.fport_delay)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检测 HOScrcpy 抓图流是否能收到首帧")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待首帧的超时时间，单位秒")
    parser.add_argument("--sn", default=None, help="HDC 设备 SN；不填则读取配置或 hdc list targets 的第一个设备")
    parser.add_argument("--ip", default=None, help="HOScrcpy SDK 连接 IP，默认走配置或 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="HDC server 端口，默认走配置或 8710")
    parser.add_argument("--scale", type=int, default=None, help="HOScrcpy 分辨率缩放倍率")
    parser.add_argument("--frame-rate", type=int, default=None, help="HOScrcpy 帧率")
    parser.add_argument("--bit-rate", type=int, default=None, help="HOScrcpy 码率")
    parser.add_argument("--device-port", type=int, default=None, help="设备端投屏服务端口")
    parser.add_argument("--encoder-type", default=None, help="编码器类型")
    parser.add_argument("--fport-retries", type=int, default=None, help="HDC fport 失败时的重试次数")
    parser.add_argument("--fport-delay", type=float, default=None, help="HDC fport 每次重试前等待的秒数")
    parser.add_argument("--rotation-mode", type=int, default=0, choices=(0, 1, 2, 3), help="0/1/2/3 对应不旋转/90/180/270")
    parser.add_argument("--save-frame", type=Path, default=None, help="保存首帧到指定 JPEG 路径")
    parser.add_argument("--no-save", action="store_true", help="不保存首帧诊断图")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    apply_hoscrcpy_env_overrides(args)

    save_frame_path = None
    if not args.no_save:
        save_frame_path = args.save_frame or _default_save_frame_path()

    result = probe_hoscrcpy_stream(
        timeout=args.timeout,
        save_frame_path=save_frame_path,
        rotation_mode=args.rotation_mode,
    )
    print(
        "[HOSProbe] OK: received %s frame(s) in %.2fs, size=%sx%s"
        % (result.frame_count, result.elapsed_seconds, result.width, result.height),
        flush=True,
    )
    if result.saved_frame_path is not None:
        print("[HOSProbe] first frame saved: %s" % result.saved_frame_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
