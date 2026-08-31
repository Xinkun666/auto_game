"""Per-run process-tree memory and thread activity capture.

The capture deliberately lives in Launcher instead of the testcase process so
it can keep writing evidence when the testcase, HDC, or a native child crashes.
Windows heap allocations belong to a process, not a thread; thread records are
therefore CPU/activity correlation evidence rather than per-thread heap usage.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


MIB = 1024 * 1024
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_WRITE_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_PROCESSES = 40
DEFAULT_MAX_THREADS = 20
WATCHED_PROCESS_NAMES = {
    "python.exe",
    "pythonw.exe",
    "hdc.exe",
    "hdc",
    "icpm_xdc.exe",
    "conhost.exe",
    "openconsole.exe",
    "werfault.exe",
}


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _mb(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) / MIB, 3)
    except (TypeError, ValueError):
        return None


def _filetime_value(value) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


class _WindowsSnapshotProvider:
    """Collect Windows metrics without requiring psutil in packaged builds."""

    TH32CS_SNAPPROCESS = 0x00000002
    TH32CS_SNAPTHREAD = 0x00000004
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    THREAD_QUERY_LIMITED_INFORMATION = 0x0800
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def __init__(self):
        from ctypes import wintypes

        self.wintypes = wintypes
        ulong_ptr = ctypes.c_size_t
        size_t = ctypes.c_size_t

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ulong_ptr),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        class THREADENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
                ("PrivateUsage", size_t),
            ]

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        self.PROCESSENTRY32W = PROCESSENTRY32W
        self.THREADENTRY32 = THREADENTRY32
        self.PROCESS_MEMORY_COUNTERS_EX = PROCESS_MEMORY_COUNTERS_EX
        self.MEMORYSTATUSEX = MEMORYSTATUSEX

        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._configure_functions()
        self._tracked_pids: Set[int] = set()

    def _configure_functions(self) -> None:
        wt = self.wintypes
        k32 = self.kernel32
        k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
        k32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(self.PROCESSENTRY32W)]
        k32.Process32FirstW.restype = wt.BOOL
        k32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(self.PROCESSENTRY32W)]
        k32.Process32NextW.restype = wt.BOOL
        k32.Thread32First.argtypes = [wt.HANDLE, ctypes.POINTER(self.THREADENTRY32)]
        k32.Thread32First.restype = wt.BOOL
        k32.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(self.THREADENTRY32)]
        k32.Thread32Next.restype = wt.BOOL
        k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k32.OpenProcess.restype = wt.HANDLE
        k32.OpenThread.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k32.OpenThread.restype = wt.HANDLE
        k32.CloseHandle.argtypes = [wt.HANDLE]
        k32.CloseHandle.restype = wt.BOOL
        k32.GetProcessTimes.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(wt.FILETIME),
            ctypes.POINTER(wt.FILETIME),
            ctypes.POINTER(wt.FILETIME),
            ctypes.POINTER(wt.FILETIME),
        ]
        k32.GetProcessTimes.restype = wt.BOOL
        k32.GetThreadTimes.argtypes = list(k32.GetProcessTimes.argtypes)
        k32.GetThreadTimes.restype = wt.BOOL
        k32.GetProcessHandleCount.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
        k32.GetProcessHandleCount.restype = wt.BOOL
        k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(self.MEMORYSTATUSEX)]
        k32.GlobalMemoryStatusEx.restype = wt.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(self.PROCESS_MEMORY_COUNTERS_EX),
            wt.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = wt.BOOL

    def _snapshot_process_entries(self) -> Dict[int, Dict[str, Any]]:
        snapshot = self.kernel32.CreateToolhelp32Snapshot(self.TH32CS_SNAPPROCESS, 0)
        if snapshot == self.INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot(process) failed")
        entries: Dict[int, Dict[str, Any]] = {}
        try:
            entry = self.PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            ok = self.kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                pid = int(entry.th32ProcessID)
                entries[pid] = {
                    "pid": pid,
                    "ppid": int(entry.th32ParentProcessID),
                    "name": str(entry.szExeFile),
                    "thread_count": int(entry.cntThreads),
                }
                entry.dwSize = ctypes.sizeof(entry)
                ok = self.kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            self.kernel32.CloseHandle(snapshot)
        return entries

    @staticmethod
    def _descendant_pids(entries: Dict[int, Dict[str, Any]], root_pid: int) -> Set[int]:
        selected = {int(root_pid)} if int(root_pid) in entries else set()
        changed = True
        while changed:
            changed = False
            for pid, item in entries.items():
                if pid not in selected and int(item.get("ppid", 0)) in selected:
                    selected.add(pid)
                    changed = True
        return selected

    def _read_process_metrics(self, pid: int) -> Dict[str, Any]:
        wt = self.wintypes
        access = (
            self.PROCESS_QUERY_LIMITED_INFORMATION
            | self.PROCESS_QUERY_INFORMATION
            | self.PROCESS_VM_READ
        )
        handle = self.kernel32.OpenProcess(access, False, int(pid))
        if not handle:
            handle = self.kernel32.OpenProcess(
                self.PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                int(pid),
            )
        if not handle:
            return {"query_error": int(ctypes.get_last_error())}
        try:
            metrics: Dict[str, Any] = {}
            counters = self.PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            if self.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                metrics.update(
                    private_bytes=int(counters.PrivateUsage),
                    working_set_bytes=int(counters.WorkingSetSize),
                    peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                    pagefile_bytes=int(counters.PagefileUsage),
                    page_fault_count=int(counters.PageFaultCount),
                )

            handle_count = wt.DWORD(0)
            if self.kernel32.GetProcessHandleCount(handle, ctypes.byref(handle_count)):
                metrics["handle_count"] = int(handle_count.value)

            created = wt.FILETIME()
            exited = wt.FILETIME()
            kernel = wt.FILETIME()
            user = wt.FILETIME()
            if self.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                metrics["cpu_seconds"] = round(
                    (_filetime_value(kernel) + _filetime_value(user)) / 10_000_000.0,
                    3,
                )
                metrics["create_time_100ns"] = _filetime_value(created)
            return metrics
        finally:
            self.kernel32.CloseHandle(handle)

    def _snapshot_threads(self, selected_pids: Set[int]) -> List[Dict[str, Any]]:
        wt = self.wintypes
        snapshot = self.kernel32.CreateToolhelp32Snapshot(self.TH32CS_SNAPTHREAD, 0)
        if snapshot == self.INVALID_HANDLE_VALUE:
            return []
        native_names = {
            int(thread.native_id): str(thread.name)
            for thread in threading.enumerate()
            if getattr(thread, "native_id", None) is not None
        }
        result: List[Dict[str, Any]] = []
        try:
            entry = self.THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            ok = self.kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while ok:
                pid = int(entry.th32OwnerProcessID)
                tid = int(entry.th32ThreadID)
                if pid in selected_pids:
                    item: Dict[str, Any] = {"pid": pid, "tid": tid}
                    if pid == os.getpid() and tid in native_names:
                        item["name"] = native_names[tid]
                    handle = self.kernel32.OpenThread(
                        self.THREAD_QUERY_LIMITED_INFORMATION,
                        False,
                        tid,
                    )
                    if handle:
                        try:
                            created = wt.FILETIME()
                            exited = wt.FILETIME()
                            kernel = wt.FILETIME()
                            user = wt.FILETIME()
                            if self.kernel32.GetThreadTimes(
                                handle,
                                ctypes.byref(created),
                                ctypes.byref(exited),
                                ctypes.byref(kernel),
                                ctypes.byref(user),
                            ):
                                item["cpu_seconds"] = round(
                                    (
                                        _filetime_value(kernel)
                                        + _filetime_value(user)
                                    )
                                    / 10_000_000.0,
                                    3,
                                )
                        finally:
                            self.kernel32.CloseHandle(handle)
                    result.append(item)
                entry.dwSize = ctypes.sizeof(entry)
                ok = self.kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self.kernel32.CloseHandle(snapshot)
        return result

    def _system_metrics(self) -> Dict[str, Any]:
        status = self.MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not self.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"query_error": int(ctypes.get_last_error())}
        total_phys = int(status.ullTotalPhys)
        avail_phys = int(status.ullAvailPhys)
        commit_limit = int(status.ullTotalPageFile)
        commit_available = int(status.ullAvailPageFile)
        return {
            "memory_load_percent": int(status.dwMemoryLoad),
            "physical_total_bytes": total_phys,
            "physical_available_bytes": avail_phys,
            "physical_used_bytes": max(0, total_phys - avail_phys),
            "commit_limit_bytes": commit_limit,
            "commit_available_bytes": commit_available,
            "commit_used_bytes": max(0, commit_limit - commit_available),
            "virtual_total_bytes": int(status.ullTotalVirtual),
            "virtual_available_bytes": int(status.ullAvailVirtual),
        }

    def sample(self, root_pid: int) -> Dict[str, Any]:
        entries = self._snapshot_process_entries()
        descendants = self._descendant_pids(entries, root_pid)
        live_pids = set(entries)
        self._tracked_pids.intersection_update(live_pids)
        self._tracked_pids.update(descendants)

        watched = {
            pid
            for pid, item in entries.items()
            if str(item.get("name") or "").lower() in WATCHED_PROCESS_NAMES
        }
        selected_pids = self._tracked_pids | watched
        processes = []
        for pid in sorted(selected_pids):
            item = dict(entries[pid])
            if pid in descendants:
                item["scope"] = "run_tree"
            elif pid in self._tracked_pids:
                item["scope"] = "tracked_child"
            else:
                item["scope"] = "watched_name"
            item.update(self._read_process_metrics(pid))
            processes.append(item)

        return {
            "system": self._system_metrics(),
            "processes": processes,
            "threads": self._snapshot_threads(selected_pids),
        }


class _PosixSnapshotProvider:
    """Small fallback used for local development and tests on macOS/Linux."""

    def __init__(self):
        self._tracked_pids: Set[int] = set()

    @staticmethod
    def _entries() -> Dict[int, Dict[str, Any]]:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,vsz=,comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        entries: Dict[int, Dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 4)
            if len(parts) < 5:
                continue
            try:
                pid, ppid, rss_kib, vsz_kib = map(int, parts[:4])
            except ValueError:
                continue
            entries[pid] = {
                "pid": pid,
                "ppid": ppid,
                "name": Path(parts[4]).name,
                "working_set_bytes": rss_kib * 1024,
                "virtual_bytes": vsz_kib * 1024,
            }
        return entries

    def sample(self, root_pid: int) -> Dict[str, Any]:
        entries = self._entries()
        descendants = _WindowsSnapshotProvider._descendant_pids(entries, root_pid)
        self._tracked_pids.intersection_update(entries)
        self._tracked_pids.update(descendants)
        processes = []
        for pid in sorted(self._tracked_pids):
            item = dict(entries[pid])
            item["scope"] = "run_tree" if pid in descendants else "tracked_child"
            processes.append(item)
        return {"system": {}, "processes": processes, "threads": []}


def _default_provider():
    return _WindowsSnapshotProvider() if os.name == "nt" else _PosixSnapshotProvider()


class MemoryRunCapture:
    """Write one JSON record per sample to a run's ``logs/memory.log``."""

    def __init__(
        self,
        output_path: Path,
        root_pid: Optional[int] = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        write_interval_seconds: float = DEFAULT_WRITE_INTERVAL_SECONDS,
        provider=None,
        max_processes: int = DEFAULT_MAX_PROCESSES,
        max_threads: int = DEFAULT_MAX_THREADS,
    ):
        self.path = Path(output_path)
        self.root_pid = int(root_pid or os.getpid())
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.write_interval_seconds = max(0.1, float(write_interval_seconds))
        self.provider = provider
        self.max_processes = max(1, int(max_processes))
        self.max_threads = max(0, int(max_threads))
        self.start_error = ""
        self.sample_count = 0
        self.write_count = 0
        self._file = None
        self._pending_records: List[str] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._last_write_at = 0.0
        self._previous_processes: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        self._baseline_processes: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        self._previous_threads: Dict[Tuple[int, int], float] = {}
        self._previous_tree_memory = 0
        self._baseline_tree_memory: Optional[int] = None
        self._atexit_callback = None
        self._started = False
        self._stopped = False

    def _flush_pending_records(self) -> None:
        if self._file is None:
            return
        if not self._pending_records:
            return
        self._file.write("".join(self._pending_records))
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            # 部分特殊文件系统不支持 fsync；flush 仍已把 Python 缓冲交给系统。
            pass
        self._pending_records.clear()
        self._last_write_at = time.monotonic()
        self.write_count += 1

    def _write_record(self, record: Dict[str, Any], force: bool = False) -> None:
        if self._file is None:
            return
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._pending_records.append(payload + "\n")
        if (
            force
            or time.monotonic() - self._last_write_at
            >= self.write_interval_seconds
        ):
            self._flush_pending_records()

    @staticmethod
    def _process_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
        return (
            int(item.get("pid", 0)),
            int(item.get("create_time_100ns", 0) or 0),
            str(item.get("name") or "").lower(),
        )

    @staticmethod
    def _memory_bytes(item: Dict[str, Any]) -> int:
        return int(
            item.get("private_bytes")
            or item.get("working_set_bytes")
            or 0
        )

    def _decorate_processes(self, processes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decorated = []
        current: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for raw in processes:
            item = dict(raw)
            key = self._process_key(item)
            memory_bytes = self._memory_bytes(item)
            previous = self._previous_processes.get(key, {})
            baseline = self._baseline_processes.setdefault(key, dict(item))
            previous_memory = self._memory_bytes(previous)
            baseline_memory = self._memory_bytes(baseline)
            previous_cpu = float(previous.get("cpu_seconds", 0.0) or 0.0)
            current_cpu = float(item.get("cpu_seconds", 0.0) or 0.0)

            item.update(
                memory_basis=(
                    "private_bytes" if item.get("private_bytes") is not None else "working_set_bytes"
                ),
                private_mb=_mb(item.get("private_bytes")),
                working_set_mb=_mb(item.get("working_set_bytes")),
                peak_working_set_mb=_mb(item.get("peak_working_set_bytes")),
                virtual_mb=_mb(item.get("virtual_bytes")),
                delta_memory_mb=_mb(memory_bytes - previous_memory) if previous else None,
                growth_memory_mb=_mb(memory_bytes - baseline_memory),
                delta_cpu_seconds=(
                    round(max(0.0, current_cpu - previous_cpu), 3) if previous else None
                ),
            )
            for byte_key in (
                "private_bytes",
                "working_set_bytes",
                "peak_working_set_bytes",
                "pagefile_bytes",
                "virtual_bytes",
            ):
                item.pop(byte_key, None)
            current[key] = dict(raw)
            decorated.append(item)

        self._previous_processes = current
        decorated.sort(
            key=lambda item: (
                float(item.get("growth_memory_mb") or 0.0),
                float(item.get("private_mb") or item.get("working_set_mb") or 0.0),
            ),
            reverse=True,
        )
        return decorated

    def _decorate_threads(self, threads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        current: Dict[Tuple[int, int], float] = {}
        decorated = []
        for raw in threads:
            item = dict(raw)
            key = (int(item.get("pid", 0)), int(item.get("tid", 0)))
            cpu_seconds = float(item.get("cpu_seconds", 0.0) or 0.0)
            previous = self._previous_threads.get(key)
            item["delta_cpu_seconds"] = (
                round(max(0.0, cpu_seconds - previous), 3)
                if previous is not None
                else None
            )
            current[key] = cpu_seconds
            decorated.append(item)
        self._previous_threads = current
        decorated.sort(
            key=lambda item: float(item.get("delta_cpu_seconds") or 0.0),
            reverse=True,
        )
        return decorated[: self.max_threads]

    @staticmethod
    def _compact_system(system: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(system or {})
        commit_used = result.get("commit_used_bytes")
        commit_limit = result.get("commit_limit_bytes")
        if commit_used is not None and commit_limit:
            result["commit_used_percent"] = round(
                float(commit_used) * 100.0 / float(commit_limit),
                2,
            )
        for key in list(result):
            if key.endswith("_bytes"):
                result[key[:-6] + "_mb"] = _mb(result.pop(key))
        return result

    def sample_once(self) -> Dict[str, Any]:
        if self.provider is None:
            self.provider = _default_provider()
        snapshot = self.provider.sample(self.root_pid)
        processes = self._decorate_processes(snapshot.get("processes") or [])
        threads = self._decorate_threads(snapshot.get("threads") or [])

        tree_processes = [
            item for item in processes if item.get("scope") in {"run_tree", "tracked_child"}
        ]
        tree_memory = 0
        for item in tree_processes:
            memory_mb = item.get("private_mb")
            if memory_mb is None:
                memory_mb = item.get("working_set_mb")
            tree_memory += int(round(float(memory_mb or 0.0) * MIB))
        if self._baseline_tree_memory is None:
            self._baseline_tree_memory = tree_memory

        alerts = []
        for item in processes:
            delta = float(item.get("delta_memory_mb") or 0.0)
            growth = float(item.get("growth_memory_mb") or 0.0)
            if delta >= 128.0 or growth >= 512.0:
                alerts.append(
                    {
                        "pid": item.get("pid"),
                        "name": item.get("name"),
                        "delta_memory_mb": delta,
                        "growth_memory_mb": growth,
                    }
                )

        record = {
            "event": "sample",
            "timestamp": _iso_now(),
            "elapsed_seconds": round(max(0.0, time.monotonic() - self._started_at), 3),
            "sample_index": self.sample_count + 1,
            "system": self._compact_system(snapshot.get("system") or {}),
            "run_tree": {
                "process_count": len(tree_processes),
                "memory_mb": _mb(tree_memory),
                "delta_memory_mb": _mb(tree_memory - self._previous_tree_memory)
                if self.sample_count
                else None,
                "growth_memory_mb": _mb(tree_memory - int(self._baseline_tree_memory or 0)),
            },
            "processes_ranked_by_growth": processes[: self.max_processes],
            "top_threads_by_cpu": threads,
            "alerts": alerts,
            "notes": (
                "Windows heap memory is process-scoped; thread CPU deltas are correlation only, "
                "not per-thread heap ownership."
            ),
        }
        self._previous_tree_memory = tree_memory
        self.sample_count += 1
        with self._lock:
            self._write_record(record)
        return record

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception as exc:
                with self._lock:
                    self._write_record(
                        {
                            "event": "sample_error",
                            "timestamp": _iso_now(),
                            "error": repr(exc),
                        }
                    )
            if self._stop_event.wait(self.interval_seconds):
                return

    def start(self):
        if self._started:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = time.monotonic()
        self._last_write_at = self._started_at
        self._stop_event.clear()
        self._stopped = False
        try:
            self._file = self.path.open("a", encoding="utf-8", buffering=1)
            self.provider = self.provider or _default_provider()
            self._write_record(
                {
                    "event": "capture_start",
                    "timestamp": _iso_now(),
                    "root_pid": self.root_pid,
                    "interval_seconds": self.interval_seconds,
                    "write_interval_seconds": self.write_interval_seconds,
                    "format": "AutoGame memory.log JSONL v1",
                    "platform": os.name,
                    "notes": [
                        "private_mb is the preferred Windows process heap/commit indicator",
                        "working_set_mb is resident RAM and may fall without freeing committed memory",
                        "thread CPU activity cannot prove which thread owns heap allocations",
                    ],
                },
                force=True,
            )
            self._thread = threading.Thread(
                target=self._run,
                name="MemoryRunCapture",
                daemon=True,
            )
            self._thread.start()
            self._atexit_callback = self.stop
            atexit.register(self._atexit_callback)
            self._started = True
        except Exception as exc:
            self.start_error = str(exc)
            if self._file is not None:
                self._write_record(
                    {"event": "capture_start_error", "timestamp": _iso_now(), "error": repr(exc)},
                    force=True,
                )
            self.stop()
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        with self._lock:
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
            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.interval_seconds + 0.5))

        with self._lock:
            if self._file is not None:
                self._write_record(
                    {
                        "event": "capture_stop",
                        "timestamp": _iso_now(),
                        "samples": self.sample_count,
                        "writes": self.write_count + 1,
                    },
                    force=True,
                )
                self._file.close()
                self._file = None
            self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
