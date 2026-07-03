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

    def pop_latest(self):
        frame = None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                return frame

    def wait_for_frame(self, timeout: float, client=None):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            client_error = getattr(client, "_last_error", None)
            if client_error is not None:
                raise HOScrcpyProbeError(
                    "HOScrcpy 流启动失败: %s\n%s" % (
                        client_error,
                        format_client_diagnostics(client),
                    )
                ) from client_error

            main_thread = getattr(client, "main_thread", None)
            client_running = bool(getattr(client, "running", False))
            if main_thread is not None and not main_thread.is_alive() and not client_running:
                raise HOScrcpyProbeError(
                    "HOScrcpy 流线程已退出，但没有收到首帧\n%s" % format_client_diagnostics(client)
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HOScrcpyStreamTimeout(
                    "等待 HOScrcpy 首帧超时: %.1fs\n%s" % (
                        float(timeout),
                        format_client_diagnostics(client),
                    )
                )

            try:
                return self._frames.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue


def format_client_diagnostics(client) -> str:
    snapshot = {}
    if hasattr(client, "diagnostic_snapshot"):
        try:
            snapshot = client.diagnostic_snapshot() or {}
        except Exception as exc:
            snapshot = {"diagnostic_error": str(exc)}
    else:
        main_thread = getattr(client, "main_thread", None)
        snapshot = {
            "running": bool(getattr(client, "running", False)),
            "thread_alive": bool(main_thread and main_thread.is_alive()),
            "last_error": str(getattr(client, "_last_error", "") or ""),
        }
    ordered_keys = [
        "stage",
        "running",
        "thread_alive",
        "stop_requested",
        "sn",
        "ip",
        "port",
        "agent_port",
        "guest_port",
        "layout_port",
        "video_port",
        "video_server_port",
        "is_setup",
        "is_use_unix_socket_agent_so",
        "is_use_unix_socket_video_so",
        "ready_received",
        "first_frame_received",
        "callback_data_count",
        "last_data_bytes",
        "decoded_frame_count",
        "last_error",
        "elapsed_seconds",
        "diagnostic_error",
    ]
    parts = []
    for key in ordered_keys:
        if key in snapshot:
            parts.append("%s=%s" % (key, snapshot.get(key)))
    for key in sorted(snapshot):
        if key not in ordered_keys:
            parts.append("%s=%s" % (key, snapshot.get(key)))
    if not parts:
        parts.append("diagnostic_unavailable=True")
    return "[HOSProbeDiag] " + ", ".join(parts)


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


def _print_probe_result(result: HOScrcpyProbeResult) -> None:
    print(
        "[HOSProbe] OK: received %s frame(s) in %.2fs, size=%sx%s"
        % (result.frame_count, result.elapsed_seconds, result.width, result.height),
        flush=True,
    )
    if result.saved_frame_path is not None:
        print("[HOSProbe] first frame saved: %s" % result.saved_frame_path, flush=True)


def _latest_preview_frame(buffer: ProbeFrameBuffer):
    return buffer.pop_latest()


def display_realtime_frames(
    buffer: ProbeFrameBuffer,
    client,
    initial_frame,
    poll_interval_ms: int = 15,
) -> None:
    try:
        from PIL import Image, ImageTk
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise HOScrcpyProbeError("HOScrcpy 实时预览初始化失败: %s" % exc) from exc

    poll_interval_ms = max(1, int(poll_interval_ms or 15))
    root = tk.Tk()
    root.title("HOScrcpy Probe Preview")
    image_label = ttk.Label(root)
    image_label.pack()
    status_label = ttk.Label(root)
    status_label.pack(fill="x")

    state = {
        "closed_by_user": False,
        "error": None,
        "last_time": time.monotonic(),
        "displayed_in_window": 0,
        "last_photo": None,
    }

    def normalize_frame(frame):
        if frame is None:
            return None
        if hasattr(frame, "size") and hasattr(frame, "mode"):
            return frame
        try:
            import numpy as np

            array = np.asarray(frame)
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim == 3:
                return Image.fromarray(array).convert("RGB")
        except Exception:
            pass
        return frame

    def close_window():
        state["closed_by_user"] = True
        try:
            root.quit()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def stop_with_error(message):
        state["error"] = message
        try:
            root.quit()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def update_image(frame):
        frame = normalize_frame(frame)
        if frame is None:
            return
        photo = ImageTk.PhotoImage(frame)
        image_label.config(image=photo)
        image_label.image = photo
        state["last_photo"] = photo
        state["displayed_in_window"] += 1
        width, height = _frame_size(frame)
        status_label.config(text="frames=%s size=%sx%s" % (buffer.frame_count, width, height))

        now = time.monotonic()
        elapsed = now - state["last_time"]
        if elapsed >= 1.0:
            rate = state["displayed_in_window"] / elapsed
            print("[HOSProbe] preview rate %.2f fps" % rate, flush=True)
            state["displayed_in_window"] = 0
            state["last_time"] = now

    def poll():
        if state["closed_by_user"]:
            return

        client_error = getattr(client, "_last_error", None)
        if client_error is not None:
            stop_with_error(
                "HOScrcpy 实时预览中断: %s\n%s"
                % (client_error, format_client_diagnostics(client))
            )
            return

        main_thread = getattr(client, "main_thread", None)
        client_running = bool(getattr(client, "running", False))
        if main_thread is not None and not main_thread.is_alive() and not client_running:
            stop_with_error(
                "HOScrcpy 实时预览中断: 流线程已退出\n%s"
                % format_client_diagnostics(client)
            )
            return

        frame = _latest_preview_frame(buffer)
        if frame is not None:
            update_image(frame)
        root.after(poll_interval_ms, poll)

    root.protocol("WM_DELETE_WINDOW", close_window)
    update_image(initial_frame)
    print("[HOSProbe] realtime preview started. Close the preview window to stop.", flush=True)
    root.after(poll_interval_ms, poll)
    root.mainloop()

    if state["error"] is not None and not state["closed_by_user"]:
        raise HOScrcpyProbeError(state["error"])


def probe_hoscrcpy_stream(
    timeout: float = 15.0,
    save_frame_path: Optional[Path] = None,
    rotation_mode: int = 0,
    client_factory: Callable = HOSScrcpyStreamClient,
    display: bool = False,
    display_poll_ms: int = 15,
    success_callback: Optional[Callable[[HOScrcpyProbeResult], None]] = None,
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
        result = HOScrcpyProbeResult(
            elapsed_seconds=elapsed,
            frame_count=buffer.frame_count,
            width=width,
            height=height,
            saved_frame_path=saved_path,
        )
        if success_callback is not None:
            success_callback(result)
        if display:
            display_realtime_frames(buffer, client, frame, poll_interval_ms=display_poll_ms)
        return result
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
    parser.set_defaults(display=True)
    parser.add_argument("--display", dest="display", action="store_true", help="首帧成功后打开实时预览窗口，默认开启")
    parser.add_argument("--no-display", dest="display", action="store_false", help="仅验证首帧后退出，不打开实时预览")
    parser.add_argument("--display-poll-ms", type=int, default=15, help="实时预览刷新间隔，单位毫秒")
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
        display=args.display,
        display_poll_ms=args.display_poll_ms,
        success_callback=_print_probe_result if args.display else None,
    )
    if not args.display:
        _print_probe_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
