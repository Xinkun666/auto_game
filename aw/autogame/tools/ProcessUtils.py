import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("launcher")
WINDOW_SUPPRESS_ENV = "AUTOGAME_HIDE_SUBPROCESS_WINDOWS"


def hidden_subprocess_kwargs(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> Dict[str, Any]:
    if (os_name or os.name) != "nt":
        return {}

    startupinfo = subprocess_module.STARTUPINFO()
    startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": _hidden_creationflags(subprocess_module),
    }


def resolve_hdc_executable() -> str:
    return shutil.which("hdc") or shutil.which("hdc.exe") or "hdc"


def hdc_command_args(command: str, hdc_executable: Optional[str] = None) -> Optional[List[str]]:
    text = str(command or "").strip()
    if not text:
        return None

    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        return None
    if not parts:
        return None

    executable = str(parts[0]).strip("\"'")
    executable_name = os.path.basename(executable).lower()
    if executable_name not in {"hdc", "hdc.exe"}:
        return None

    hdc = hdc_executable or resolve_hdc_executable()
    args = [hdc]
    remaining = [_strip_wrapping_quotes(part) for part in parts[1:]]
    shell_index = next((index for index, part in enumerate(remaining) if part.lower() == "shell"), None)
    if shell_index is None:
        args.extend(remaining)
        return args

    args.extend(remaining[:shell_index])
    args.append("shell")
    remote_command = " ".join(remaining[shell_index + 1 :]).strip()
    if remote_command:
        args.append(remote_command)
    return args


def _strip_wrapping_quotes(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _command_text(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _hidden_creationflags(subprocess_module=subprocess) -> int:
    flags = getattr(subprocess_module, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess_module, "DETACHED_PROCESS", 0)
    return flags


def is_window_suppression_enabled(os_name: Optional[str] = None) -> bool:
    if (os_name or os.name) != "nt":
        return False
    value = os.environ.get(WINDOW_SUPPRESS_ENV, "").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


class WindowsSubprocessWindowSuppressor:
    def __init__(
        self,
        root_pid: Optional[int] = None,
        target_processes: Sequence[str] = (
            "icpm_xdc.exe",
            "hdc.exe",
            "cmd.exe",
            "conhost.exe",
            "openconsole.exe",
        ),
        direct_hide_processes: Sequence[str] = ("icpm_xdc.exe", "hdc.exe"),
        excluded_processes: Sequence[str] = (),
        hide_descendant_windows: bool = True,
        interval_seconds: float = 0.02,
        snapshot_interval_seconds: float = 0.02,
        os_name: Optional[str] = None,
    ):
        self.root_pid = int(root_pid or os.getpid())
        self.target_processes = {str(name).lower() for name in target_processes}
        self.direct_hide_processes = {str(name).lower() for name in direct_hide_processes}
        self.excluded_processes = {str(name).lower() for name in excluded_processes}
        self.hide_descendant_windows = bool(hide_descendant_windows)
        self.interval_seconds = float(interval_seconds)
        self.snapshot_interval_seconds = float(snapshot_interval_seconds)
        self.os_name = os_name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hidden_windows: Set[Tuple[int, int]] = set()
        self._process_snapshot: Dict[int, Tuple[int, str]] = {}
        self._seen_processes: Set[int] = set()
        self._last_snapshot_time = 0.0

    def start(self) -> bool:
        if not is_window_suppression_enabled(self.os_name):
            LOGGER.info(
                "subprocess window suppression disabled: os_name=%s env=%s",
                self.os_name or os.name,
                os.environ.get(WINDOW_SUPPRESS_ENV),
            )
            return False
        if self._thread is not None:
            return True

        self._thread = threading.Thread(
            target=self._run,
            name="autogame-window-suppressor",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info(
            "subprocess window suppression started: root_pid=%s targets=%s direct=%s hide_descendants=%s excluded=%s interval=%.3f",
            self.root_pid,
            sorted(self.target_processes),
            sorted(self.direct_hide_processes),
            self.hide_descendant_windows,
            sorted(self.excluded_processes),
            self.interval_seconds,
        )
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            LOGGER.debug("subprocess window suppression unavailable", exc_info=True)
            return

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetModuleFileNameExW.argtypes = [
            wintypes.HANDLE,
            wintypes.HMODULE,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD

        SW_HIDE = 0

        def enum_callback(hwnd, _lparam):
            if self._stop_event.is_set():
                return False
            if not user32.IsWindowVisible(hwnd):
                return True
            pid_value = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_value))
            pid = int(pid_value.value)
            if pid <= 0:
                return True

            process_name = self._process_name(pid)
            should_hide, reason = self._should_hide_process(pid, process_name)
            if not should_hide:
                return True

            title = self._window_title(user32, hwnd)
            class_name = self._window_class(user32, hwnd)
            user32.ShowWindow(hwnd, SW_HIDE)
            key = (int(hwnd), pid)
            if key not in self._hidden_windows:
                self._hidden_windows.add(key)
                LOGGER.info(
                    "subprocess window hidden: hwnd=%s pid=%s name=%s class=%s title=%s root_pid=%s reason=%s",
                    int(hwnd),
                    pid,
                    process_name,
                    class_name,
                    title,
                    self.root_pid,
                    reason,
                )
            return True

        callback = callback_type(enum_callback)
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - self._last_snapshot_time >= self.snapshot_interval_seconds:
                self._update_process_snapshot(self._snapshot_processes())
                self._last_snapshot_time = now
            try:
                user32.EnumWindows(callback, 0)
            except Exception:
                LOGGER.debug("subprocess window enumeration failed", exc_info=True)
            self._stop_event.wait(self.interval_seconds)

    def _window_title(self, user32, hwnd) -> str:
        try:
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return ""
            import ctypes

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value
        except Exception:
            return ""

    def _window_class(self, user32, hwnd) -> str:
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buffer, 256)
            return buffer.value
        except Exception:
            return ""

    def _process_name(self, pid: int) -> str:
        item = self._process_snapshot.get(int(pid))
        return item[1] if item else ""

    def _should_hide_process(self, pid: int, process_name: str) -> Tuple[bool, str]:
        process_name = process_name.lower()
        if int(pid) == self.root_pid:
            return False, "root"
        if process_name in self.excluded_processes:
            return False, "excluded"
        if process_name in self.direct_hide_processes:
            return True, "direct"
        if self.hide_descendant_windows and self._is_descendant_of_root(pid):
            return True, "descendant"
        if process_name in self.target_processes and self._is_descendant_of_root(pid):
            return True, "target-descendant"
        return False, "not-matched"

    def _update_process_snapshot(self, snapshot: Dict[int, Tuple[int, str]]) -> None:
        next_snapshot = {
            int(pid): (int(parent_pid), str(process_name).lower())
            for pid, (parent_pid, process_name) in snapshot.items()
        }
        previous_seen = set(self._seen_processes)
        self._process_snapshot = next_snapshot
        self._seen_processes.update(next_snapshot.keys())

        if not previous_seen:
            return

        for pid in sorted(set(next_snapshot) - previous_seen):
            parent_pid, process_name = next_snapshot[pid]
            should_trace, reason = self._should_trace_process(pid, process_name)
            if should_trace:
                LOGGER.info(
                    "subprocess process create: pid=%s ppid=%s name=%s reason=%s root_pid=%s",
                    pid,
                    parent_pid,
                    process_name,
                    reason,
                    self.root_pid,
                )

    def _should_trace_process(self, pid: int, process_name: str) -> Tuple[bool, str]:
        process_name = process_name.lower()
        if int(pid) == self.root_pid:
            return False, "root"
        if process_name in self.excluded_processes:
            return False, "excluded"
        if process_name in self.direct_hide_processes:
            return True, "direct"
        if self._is_descendant_of_root(pid):
            return True, "descendant"
        return False, "not-matched"

    def _is_descendant_of_root(self, pid: int) -> bool:
        current = int(pid)
        seen: Set[int] = set()
        while current and current not in seen:
            if current == self.root_pid:
                return True
            seen.add(current)
            parent = self._process_snapshot.get(current)
            if not parent:
                return False
            current = int(parent[0])
        return False

    def _snapshot_processes(self) -> Dict[int, Tuple[int, str]]:
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return {}

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TH32CS_SNAPPROCESS = 0x00000002
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            return self._process_snapshot
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            result: Dict[int, Tuple[int, str]] = {}
            if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                return result
            while True:
                result[int(entry.th32ProcessID)] = (
                    int(entry.th32ParentProcessID),
                    str(entry.szExeFile).lower(),
                )
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
            return result
        finally:
            kernel32.CloseHandle(snapshot)


_window_suppressor: Optional[WindowsSubprocessWindowSuppressor] = None


def start_hidden_subprocess_window_suppressor(
    root_pid: Optional[int] = None,
    os_name: Optional[str] = None,
) -> bool:
    global _window_suppressor
    if _window_suppressor is None:
        _window_suppressor = WindowsSubprocessWindowSuppressor(
            root_pid=root_pid,
            os_name=os_name,
        )
    return _window_suppressor.start()


def _should_hide_command(command: Any, target_executables: Iterable[str]) -> bool:
    text = _command_text(command).lower()
    return any(str(target).lower() in text for target in target_executables)


def _with_hidden_kwargs(
    kwargs: Dict[str, Any],
    os_name: Optional[str],
    subprocess_module=subprocess,
) -> Dict[str, Any]:
    next_kwargs = dict(kwargs)
    hidden_kwargs = hidden_subprocess_kwargs(
        os_name=os_name,
        subprocess_module=subprocess_module,
    )
    startupinfo = next_kwargs.get("startupinfo") or hidden_kwargs.get("startupinfo")
    if startupinfo is not None:
        startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        next_kwargs["startupinfo"] = startupinfo
    next_kwargs["creationflags"] = (
        (next_kwargs.get("creationflags") or 0)
        | _hidden_creationflags(subprocess_module)
    )
    return next_kwargs


@contextmanager
def hidden_subprocess_context(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
    os_module=os,
    target_executables: Iterable[str] = ("icpm_xdc.exe",),
    hide_all: bool = False,
):
    if (os_name or os.name) != "nt":
        yield False
        return

    original_popen = subprocess_module.Popen
    original_system = getattr(os_module, "system", None)
    original_popen_os = getattr(os_module, "popen", None)

    def should_hide(command: Any, executable: Any = None) -> bool:
        return (
            hide_all
            or _should_hide_command(command, target_executables)
            or _should_hide_command(executable, target_executables)
        )

    class HiddenTargetPopen(original_popen):
        def __init__(self, *args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            if should_hide(command, kwargs.get("executable")):
                kwargs = _with_hidden_kwargs(
                    kwargs,
                    os_name=os_name,
                    subprocess_module=subprocess_module,
                )
                LOGGER.info("hidden subprocess context applied: command=%s", _command_text(command))
            super().__init__(*args, **kwargs)

    def hidden_system(command):
        if should_hide(command):
            LOGGER.info("hidden os.system context applied: command=%s", _command_text(command))
            return subprocess_module.call(
                command,
                shell=True,
                **_with_hidden_kwargs(
                    {},
                    os_name=os_name,
                    subprocess_module=subprocess_module,
                ),
            )
        return original_system(command)

    try:
        subprocess_module.Popen = HiddenTargetPopen
        if original_system is not None:
            os_module.system = hidden_system
        yield True
    finally:
        subprocess_module.Popen = original_popen
        if original_system is not None:
            os_module.system = original_system
        if original_popen_os is not None:
            os_module.popen = original_popen_os


def install_hidden_subprocess_patch(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> bool:
    LOGGER.info("global subprocess patch disabled")
    return False
