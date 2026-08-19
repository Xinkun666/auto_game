"""Game_Recording 运行日志和 HOS 断连报告。"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


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


def save_disconnect_report(
    output_root: Path,
    diagnostic: Dict[str, Any],
    runtime_log_path: Optional[Path],
    recording_dir: Optional[Path] = None,
    recording_error: str = "",
) -> Dict[str, Path]:
    """保存首次断连的独立诊断报告，并关联已收尾的录制目录。"""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_dir = Path(output_root) / "disconnect_logs" / stamp
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "hos_disconnect.json"
    payload = {
        "event": "hos_disconnect",
        "occurred_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "runtime_log": str(runtime_log_path) if runtime_log_path else "",
        "recording_dir": str(recording_dir) if recording_dir else "",
        "recording_save_error": str(recording_error or ""),
        "diagnostic": diagnostic,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    report_path.write_text(content, encoding="utf-8")

    paths = {"disconnect_report": report_path}
    if recording_dir is not None:
        session_report = Path(recording_dir) / "hos_disconnect.json"
        session_report.write_text(content, encoding="utf-8")
        paths["recording_report"] = session_report
    return paths
