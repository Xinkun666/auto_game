"""HDC DEBUG server setup and per-run incremental log capture."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from aw.autogame.tools.ProcessUtils import (
    hdc_command_args,
    hidden_subprocess_kwargs,
)


DEFAULT_HDC_DEBUG_LEVEL = 5
HDC_DEBUG_LEVEL_ENV = "AUTOGAME_HDC_DEBUG_LEVEL"


def resolve_hdc_debug_level(default: int = DEFAULT_HDC_DEBUG_LEVEL) -> int:
    value = os.environ.get(HDC_DEBUG_LEVEL_ENV, str(default)).strip()
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HDC DEBUG 级别必须是 0-6 的整数：{value!r}") from exc
    if level < 0 or level > 6:
        raise ValueError(f"HDC DEBUG 级别必须介于 0-6：{level}")
    return level


def resolve_hdc_debug_source_path(temp_root: Optional[Path] = None) -> Path:
    if temp_root is None:
        temp_value = (
            os.environ.get("TEMP")
            or os.environ.get("TMP")
            or os.environ.get("TMPDIR")
        )
        temp_root = Path(temp_value) if temp_value else Path(tempfile.gettempdir())
    return Path(temp_root).expanduser().resolve() / "hdc.log"


def _decode_output(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _run_hdc_server_command(command_text: str, timeout: float):
    command = hdc_command_args(command_text)
    if not command:
        raise RuntimeError(f"无法构造 HDC 命令：{command_text}")
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(1.0, float(timeout)),
        **hidden_subprocess_kwargs(),
    )


def restart_hdc_debug_server(level: Optional[int] = None, timeout: float = 10.0) -> Path:
    """Restart the host HDC server once with the requested debug level."""
    debug_level = resolve_hdc_debug_level() if level is None else int(level)
    if debug_level < 0 or debug_level > 6:
        raise ValueError(f"HDC DEBUG 级别必须介于 0-6：{debug_level}")

    kill_result = None
    try:
        kill_result = _run_hdc_server_command("hdc kill", timeout)
    except Exception:
        # hdc server 未运行时 kill 可能失败；仍继续显式 start。
        pass

    start_text = f"hdc -l {debug_level} start"
    start_result = _run_hdc_server_command(start_text, timeout)
    if start_result.returncode != 0:
        kill_detail = ""
        if kill_result is not None:
            kill_detail = " kill_rc=%s kill_output=%r" % (
                kill_result.returncode,
                (_decode_output(kill_result.stdout) + _decode_output(kill_result.stderr)).strip(),
            )
        raise RuntimeError(
            "HDC DEBUG server 启动失败：command=%r returncode=%s output=%r%s"
            % (
                start_text,
                start_result.returncode,
                (_decode_output(start_result.stdout) + _decode_output(start_result.stderr)).strip(),
                kill_detail,
            )
        )
    return resolve_hdc_debug_source_path()


class HdcDebugRunCapture:
    """Continuously copy only one run's new HDC server log bytes.

    The source hdc.log is replaced whenever the HDC server restarts.  Keeping an
    open handle and checking the path identity lets the capture drain the old
    generation before following the new file.
    """

    def __init__(
        self,
        output_path: Path,
        source_path: Optional[Path] = None,
        poll_interval: float = 0.1,
    ):
        self.path = Path(output_path)
        self.source_path = Path(source_path or resolve_hdc_debug_source_path())
        self.poll_interval = max(0.02, float(poll_interval))
        self.start_error = ""
        self.capture_error = ""
        self.bytes_captured = 0
        self.rotation_count = 0
        self._target: Optional[BinaryIO] = None
        self._source: Optional[BinaryIO] = None
        self._source_identity: Optional[Tuple[int, int]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._started = False

    @staticmethod
    def _identity(stat_result) -> Tuple[int, int]:
        return (
            int(getattr(stat_result, "st_dev", 0) or 0),
            int(getattr(stat_result, "st_ino", 0) or 0),
        )

    def _write_marker(self, message: str) -> None:
        if self._target is None:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._target.write(f"\n[AutoGame][HDC] {timestamp} {message}\n".encode("utf-8"))
        self._target.flush()

    def _open_source(self, from_end: bool) -> bool:
        try:
            source = self.source_path.open("rb")
            stat_result = os.fstat(source.fileno())
            if from_end:
                source.seek(0, os.SEEK_END)
            self._source = source
            self._source_identity = self._identity(stat_result)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            self.capture_error = str(exc)
            return False

    def _close_source(self) -> None:
        source, self._source = self._source, None
        self._source_identity = None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass

    def _drain_source(self) -> None:
        if self._source is None or self._target is None:
            return
        while True:
            chunk = self._source.read(256 * 1024)
            if not chunk:
                break
            self._target.write(chunk)
            self._target.flush()
            self.bytes_captured += len(chunk)

    def _pump_once(self) -> None:
        with self._lock:
            if self._target is None:
                return
            if self._source is None:
                if self._open_source(from_end=False):
                    self._write_marker(f"hdc.log appeared source={self.source_path}")
                return

            self._drain_source()
            try:
                current_stat = self.source_path.stat()
            except FileNotFoundError:
                return

            current_identity = self._identity(current_stat)
            current_position = self._source.tell()
            if current_identity != self._source_identity:
                self._drain_source()
                self._close_source()
                self.rotation_count += 1
                self._write_marker("hdc.log rotated/replaced; following new server log")
                self._open_source(from_end=False)
            elif current_stat.st_size < current_position:
                self.rotation_count += 1
                self._write_marker("hdc.log truncated; restarting capture at offset 0")
                self._source.seek(0)

    def _worker(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._pump_once()
            except Exception as exc:
                self.capture_error = str(exc)
                with self._lock:
                    self._write_marker(f"capture warning: {exc}")

    def start(self):
        if self._started:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._target = self.path.open("ab", buffering=0)
            self._write_marker(
                f"per-run capture started source={self.source_path} initial_offset=EOF"
            )
            if not self._open_source(from_end=True):
                self._write_marker("source hdc.log not present yet; waiting for creation")
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="HdcDebugRunCapture",
                daemon=True,
            )
            self._thread.start()
            self._started = True
        except Exception as exc:
            self.start_error = str(exc)
            self._close_source()
            if self._target is not None:
                self._target.close()
                self._target = None
        return self

    def stop(self) -> None:
        if not self._started and self._target is None:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        try:
            self._pump_once()
        except Exception as exc:
            self.capture_error = str(exc)
        with self._lock:
            self._write_marker(
                "per-run capture stopped bytes=%s rotations=%s error=%r"
                % (self.bytes_captured, self.rotation_count, self.capture_error)
            )
            self._close_source()
            if self._target is not None:
                try:
                    self._target.flush()
                    self._target.close()
                finally:
                    self._target = None
        self._thread = None
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
