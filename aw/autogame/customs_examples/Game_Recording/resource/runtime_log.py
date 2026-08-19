"""Game_Recording 运行日志和 HOS 断连报告。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from aw.autogame.tools.ProcessUtils import hdc_command_args, hidden_subprocess_kwargs


class TeeTextIO:
    """将终端输出同时写入日志文件。"""

    def __init__(
        self,
        terminal: TextIO,
        log_file: TextIO,
        lock: Optional[threading.Lock] = None,
    ):
        self.terminal = terminal
        self.log_file = log_file
        self._lock = lock or threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            terminal_result = self.terminal.write(text)
            self.log_file.write(text)
            self.log_file.flush()
        return len(text) if terminal_result is None else terminal_result

    def flush(self):
        with self._lock:
            self.terminal.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.terminal, "errors", None)


class RuntimeLogCapture:
    """在 start_record 整个运行期间保存 stdout/stderr。"""

    def __init__(self, output_root: Path):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = Path(output_root) / "runtime_logs" / f"start_record_{stamp}.log"
        self._file: Optional[TextIO] = None
        self._stdout: Optional[TextIO] = None
        self._stderr: Optional[TextIO] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        shared_lock = threading.Lock()
        sys.stdout = TeeTextIO(self._stdout, self._file, shared_lock)
        sys.stderr = TeeTextIO(self._stderr, self._file, shared_lock)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()


class HilogCapture:
    """start_record 启动时持续抓取 hilog，避免断连后手机已离线无法补抓。"""

    def __init__(self, output_root: Path):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = Path(output_root) / "runtime_logs" / f"hilog_{stamp}.txt"
        self._file = None
        self._process = None
        self._stopped = False
        self.start_error = ""

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("ab", buffering=0)
        self._write_line(
            "[Game Recording] hilog capture started_at=%s"
            % datetime.now().astimezone().isoformat(timespec="milliseconds")
        )
        try:
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
                "[Game Recording] hdc targets returncode=%s stdout=%r stderr=%r"
                % (
                    targets.returncode,
                    self._decode(targets.stdout).strip(),
                    self._decode(targets.stderr).strip(),
                )
            )

            hilog_command = hdc_command_args("hdc hilog")
            if not hilog_command:
                raise RuntimeError("无法构造 hdc hilog 命令")
            self._process = subprocess.Popen(
                hilog_command,
                stdin=subprocess.DEVNULL,
                stdout=self._file,
                stderr=subprocess.STDOUT,
                **hidden_subprocess_kwargs(),
            )
            self._write_line(
                "[Game Recording] hdc hilog pid=%s command=%r"
                % (self._process.pid, hilog_command)
            )
        except Exception as exc:
            self.start_error = str(exc)
            self._write_line(
                "[Game Recording] hilog capture start failed: %s" % exc
            )
        return self

    @staticmethod
    def _decode(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value or "")

    def _write_line(self, text: str):
        if self._file is not None:
            self._file.write((str(text) + "\n").encode("utf-8", errors="replace"))

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        returncode = process.poll() if process is not None else None
        self._write_line(
            "[Game Recording] hilog capture stopped_at=%s returncode=%s"
            % (
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                returncode,
            )
        )
        if self._file is not None:
            self._file.close()
            self._file = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def save_disconnect_report(
    output_root: Path,
    diagnostic: Dict[str, Any],
    runtime_log_path: Optional[Path],
    recording_dir: Optional[Path] = None,
    recording_error: str = "",
    hilog_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """保存首次断连的独立诊断报告，并关联已收尾的录制目录。"""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_dir = Path(output_root) / "disconnect_logs" / stamp
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "hos_disconnect.json"
    paths = {"disconnect_report": report_path}
    hilog_source = Path(hilog_path) if hilog_path else None
    hilog_report_path = None
    hilog_save_error = ""
    if hilog_source is not None:
        try:
            hilog_report_path = report_dir / "hilog.txt"
            shutil.copy2(hilog_source, hilog_report_path)
            paths["hilog_log"] = hilog_report_path
        except Exception as exc:
            hilog_report_path = None
            hilog_save_error = str(exc)
            failure_path = report_dir / "hilog_capture_error.txt"
            failure_path.write_text(
                "hilog 保存失败\nsource=%s\nerror=%s\n" % (hilog_source, exc),
                encoding="utf-8",
            )
            paths["hilog_error"] = failure_path
    payload = {
        "event": "hos_disconnect",
        "occurred_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "runtime_log": str(runtime_log_path) if runtime_log_path else "",
        "hilog_source": str(hilog_source) if hilog_source else "",
        "hilog_log": str(hilog_report_path) if hilog_report_path else "",
        "hilog_save_error": hilog_save_error,
        "recording_dir": str(recording_dir) if recording_dir else "",
        "recording_save_error": str(recording_error or ""),
        "diagnostic": diagnostic,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    report_path.write_text(content, encoding="utf-8")

    if recording_dir is not None:
        session_report = Path(recording_dir) / "hos_disconnect.json"
        session_report.write_text(content, encoding="utf-8")
        paths["recording_report"] = session_report
        if hilog_report_path is not None:
            recording_hilog = Path(recording_dir) / "hilog.txt"
            try:
                shutil.copy2(hilog_report_path, recording_hilog)
                paths["recording_hilog"] = recording_hilog
            except Exception as exc:
                recording_hilog_error = Path(recording_dir) / "hilog_capture_error.txt"
                recording_hilog_error.write_text(
                    "hilog 复制到录制目录失败\nsource=%s\nerror=%s\n"
                    % (hilog_report_path, exc),
                    encoding="utf-8",
                )
                paths["recording_hilog_error"] = recording_hilog_error
    return paths
