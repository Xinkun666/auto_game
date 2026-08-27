"""Per-run HarmonyOS hilog capture."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional

from aw.autogame.tools.ProcessUtils import (
    hdc_command_args,
    hidden_subprocess_kwargs,
)


class HilogRunCapture:
    """Stream ``hdc hilog`` directly into one run's archive."""

    def __init__(self, output_path: Path):
        self.path = Path(output_path)
        self.start_error = ""
        self.returncode = None
        self.restart_count = 0
        self._file: Optional[BinaryIO] = None
        self._process: Optional[subprocess.Popen] = None
        self._hilog_command = None
        self._supervisor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_lock = threading.RLock()
        self._atexit_callback = None
        self._started = False
        self._stopped = False

    @staticmethod
    def _decode(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")

    def _write_line(self, text: str) -> None:
        if self._file is None:
            return
        self._file.write((str(text) + "\n").encode("utf-8", errors="replace"))
        self._file.flush()

    def start(self):
        if self._started:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = self.path.open("ab", buffering=0)
            self._write_line(
                "[AutoGame][HILOG] capture started_at=%s"
                % datetime.now().astimezone().isoformat(timespec="milliseconds")
            )

            targets_command = hdc_command_args("hdc list targets")
            if not targets_command:
                raise RuntimeError("无法构造 hdc list targets 命令")
            targets = subprocess.run(
                targets_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                **hidden_subprocess_kwargs(),
            )
            self._write_line(
                "[AutoGame][HILOG] targets returncode=%s stdout=%r stderr=%r"
                % (
                    targets.returncode,
                    self._decode(targets.stdout).strip(),
                    self._decode(targets.stderr).strip(),
                )
            )

            hilog_command = hdc_command_args("hdc hilog")
            if not hilog_command:
                raise RuntimeError("无法构造 hdc hilog 命令")
            self._hilog_command = hilog_command
            self._spawn_hilog_process()
            self._atexit_callback = self.stop
            atexit.register(self._atexit_callback)
            self._stop_event.clear()
            self._supervisor_thread = threading.Thread(
                target=self._supervise,
                name="HilogRunCapture",
                daemon=True,
            )
            self._supervisor_thread.start()
            self._started = True
        except Exception as exc:
            self.start_error = str(exc)
            self._write_line("[AutoGame][HILOG] capture start failed: %s" % exc)
        except BaseException:
            self.stop()
            raise
        return self

    def _spawn_hilog_process(self) -> None:
        process_kwargs = hidden_subprocess_kwargs()
        if os.name == "nt":
            process_kwargs["creationflags"] = (
                process_kwargs.get("creationflags", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            process_kwargs["start_new_session"] = True
        self._process = subprocess.Popen(
            self._hilog_command,
            stdin=subprocess.DEVNULL,
            stdout=self._file,
            stderr=subprocess.STDOUT,
            **process_kwargs,
        )
        self._write_line(
            "[AutoGame][HILOG] pid=%s command=%r restart_count=%s"
            % (self._process.pid, self._hilog_command, self.restart_count)
        )

    def _supervise(self) -> None:
        while not self._stop_event.wait(0.5):
            with self._stop_lock:
                if self._stopped:
                    return
                process = self._process
                if process is not None and process.poll() is None:
                    continue
                previous_returncode = process.poll() if process is not None else None
                self._write_line(
                    "[AutoGame][HILOG] stream ended returncode=%s; waiting to restart"
                    % previous_returncode
                )
            if self._stop_event.wait(1.0):
                return
            with self._stop_lock:
                if self._stopped:
                    return
                try:
                    self.restart_count += 1
                    self._spawn_hilog_process()
                except Exception as exc:
                    self._write_line(
                        "[AutoGame][HILOG] restart failed count=%s error=%s"
                        % (self.restart_count, exc)
                    )

    @staticmethod
    def _terminate_process(process, force: bool) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    **hidden_subprocess_kwargs(),
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        if force:
            process.kill()
        else:
            process.terminate()

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_event.set()
            callback = self._atexit_callback
            self._atexit_callback = None
            if callback is not None:
                try:
                    atexit.unregister(callback)
                except Exception:
                    pass

            supervisor_thread = self._supervisor_thread
            self._supervisor_thread = None
        if (
            supervisor_thread is not None
            and supervisor_thread.is_alive()
            and supervisor_thread is not threading.current_thread()
        ):
            supervisor_thread.join(timeout=2.0)

        with self._stop_lock:
            process = self._process
            if process is not None and process.poll() is None:
                self._terminate_process(process, force=False)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._terminate_process(process, force=True)
                    process.wait(timeout=2)
            self.returncode = process.poll() if process is not None else None
            self._write_line(
                "[AutoGame][HILOG] capture stopped_at=%s returncode=%s restarts=%s"
                % (
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    self.returncode,
                    self.restart_count,
                )
            )
            if self._file is not None:
                self._file.close()
                self._file = None
            self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
