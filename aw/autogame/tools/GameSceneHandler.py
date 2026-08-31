import os
import importlib
import concurrent.futures
import copy
import gc
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from aw.autogame.tools.Utils import *
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
from aw.autogame.tools.MemoryCapture import (
    append_memory_log_record,
    current_process_memory_snapshot,
)
import numpy as np

DEFAULT_GROUP_NAME = "默认"
GROUPABLE_ITEM_TYPES = ("area", "special_area")
GAME_SCENE_MEMORY_INTERVAL_SECONDS = 5.0
GAME_SCENE_MEMORY_LARGE_DELTA_MB = 64.0


def _iso_now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _mb(value):
    if value is None:
        return None
    return round(float(value) / (1024 * 1024), 3)


class GameSceneMemoryMonitor:
    """Correlate GameSceneHandler phases with process-level memory changes."""

    def __init__(
        self,
        output_path=None,
        *,
        interval_seconds=None,
        snapshot_func=current_process_memory_snapshot,
        writer=append_memory_log_record,
    ):
        configured_path = (
            str(output_path)
            if output_path is not None
            else os.environ.get("AUTOGAME_MEMORY_LOG_PATH", "")
        ).strip()
        self.path = Path(configured_path) if configured_path else None
        configured_interval = interval_seconds
        if configured_interval is None:
            configured_interval = os.environ.get(
                "AUTOGAME_GAMESCENE_MEMORY_INTERVAL_SECONDS",
                str(GAME_SCENE_MEMORY_INTERVAL_SECONDS),
            )
        try:
            self.interval_seconds = max(0.5, float(configured_interval))
        except (TypeError, ValueError):
            self.interval_seconds = GAME_SCENE_MEMORY_INTERVAL_SECONDS
        self.snapshot_func = snapshot_func
        self.writer = writer
        self._last_emit_at = 0.0
        self._last_signature = None
        self._baseline_snapshot = None
        self._call_index = 0
        self._error_reported = False
        self._lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._last_detail_profile_at = 0.0
        self._last_detail_signature = None
        self._handler_totals = {}
        self._baseline_python_allocated_blocks = None

    @property
    def enabled(self):
        return self.path is not None

    def snapshot(self):
        if not self.enabled:
            return None
        try:
            with self._snapshot_lock:
                return self.snapshot_func()
        except Exception as exc:
            self._report_error(exc)
            return None

    def should_profile_task_details(self, tasks_config):
        """Reserve one detailed task-level sample about every five seconds."""
        if not self.enabled:
            return False
        now = time.monotonic()
        signature = self._task_signature(tasks_config)
        with self._lock:
            changed = signature != self._last_detail_signature
            due = now - self._last_detail_profile_at >= self.interval_seconds
            if not changed and not due:
                return False
            self._last_detail_signature = signature
            self._last_detail_profile_at = now
            return True

    @staticmethod
    def _compact_snapshot(snapshot):
        if not snapshot:
            return None
        result = {
            "pid": snapshot.get("pid"),
            "thread_count": snapshot.get("thread_count"),
            "python_thread_count": snapshot.get("python_thread_count"),
            "handle_count": snapshot.get("handle_count"),
            "cpu_seconds": snapshot.get("cpu_seconds"),
        }
        for key in (
            "private_bytes",
            "working_set_bytes",
            "peak_working_set_bytes",
            "virtual_bytes",
        ):
            result[key[:-6] + "_mb"] = _mb(snapshot.get(key))
        return {key: value for key, value in result.items() if value is not None}

    @staticmethod
    def _preferred_memory_bytes(snapshot):
        if not snapshot:
            return None
        value = snapshot.get("private_bytes")
        if value is None:
            value = snapshot.get("working_set_bytes")
        return None if value is None else int(value)

    @classmethod
    def _delta_mb(cls, before, after):
        before_value = cls._preferred_memory_bytes(before)
        after_value = cls._preferred_memory_bytes(after)
        if before_value is None or after_value is None:
            return None
        return _mb(after_value - before_value)

    @staticmethod
    def _task_signature(tasks_config):
        values = []
        for task_id, config in sorted(tasks_config.items()):
            task_type = str(config.get("type") or "unknown")
            handler = str(config.get("handler_name") or task_id)
            values.append((task_type, handler))
        return tuple(values)

    @staticmethod
    def _runtime_snapshot():
        result = {"gc_generation_counts": list(gc.get_count())}
        getallocatedblocks = getattr(sys, "getallocatedblocks", None)
        if callable(getallocatedblocks):
            try:
                result["python_allocated_blocks"] = int(getallocatedblocks())
            except Exception:
                pass

        torch_module = sys.modules.get("torch")
        cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
        try:
            if cuda is not None and cuda.is_available():
                result.update(
                    {
                        "torch_cuda_device": int(cuda.current_device()),
                        "torch_cuda_allocated_mb": _mb(cuda.memory_allocated()),
                        "torch_cuda_reserved_mb": _mb(cuda.memory_reserved()),
                        "torch_cuda_max_allocated_mb": _mb(cuda.max_memory_allocated()),
                        "torch_cuda_max_reserved_mb": _mb(cuda.max_memory_reserved()),
                    }
                )
        except Exception as exc:
            result["torch_cuda_probe_error"] = repr(exc)
        return result

    @classmethod
    def _task_summary(cls, task_metrics):
        summaries = []
        optional_fields = (
            "thread_name",
            "python_thread_id",
            "native_thread_id",
            "input_copy_kind",
            "input_shape",
            "input_owns_data",
            "input_shares_frame_memory",
            "copy_duration_ms",
            "handler_duration_ms",
            "task_start_offset_ms",
            "copy_end_offset_ms",
            "handler_end_offset_ms",
            "task_end_offset_ms",
            "copy_process_delta_mb",
            "handler_process_delta_mb",
            "task_process_delta_mb",
            "result_type",
            "result_size",
            "detailed_memory_sample",
            "process_before_copy",
            "process_after_copy",
            "process_after_handler",
            "process_after_task",
        )
        for item in task_metrics:
            summary = {
                "task_id": item["task_id"],
                "task_type": item["task_type"],
                "handler": item["handler"],
                "duration_ms": item["duration_ms"],
            }
            copy_bytes = int(item.get("input_copy_bytes") or 0)
            if copy_bytes:
                summary["input_copy_mb"] = _mb(copy_bytes)
            for field in optional_fields:
                if field in item:
                    summary[field] = item[field]
            summaries.append(summary)
        return sorted(
            summaries,
            key=lambda item: item["duration_ms"],
            reverse=True,
        )

    def _update_handler_totals(self, task_metrics):
        for item in task_metrics:
            handler = str(item.get("handler") or item.get("task_id") or "unknown")
            task_type = str(item.get("task_type") or "unknown")
            key = (task_type, handler)
            total = self._handler_totals.setdefault(
                key,
                {
                    "task_type": task_type,
                    "handler": handler,
                    "calls": 0,
                    "duration_ms": 0.0,
                    "max_duration_ms": 0.0,
                    "input_copy_bytes": 0,
                    "max_input_copy_bytes": 0,
                },
            )
            duration_ms = float(item.get("duration_ms") or 0.0)
            copy_bytes = int(item.get("input_copy_bytes") or 0)
            total["calls"] += 1
            total["duration_ms"] += duration_ms
            total["max_duration_ms"] = max(total["max_duration_ms"], duration_ms)
            total["input_copy_bytes"] += copy_bytes
            total["max_input_copy_bytes"] = max(
                total["max_input_copy_bytes"],
                copy_bytes,
            )

    def _handler_totals_summary(self):
        summaries = []
        for total in self._handler_totals.values():
            calls = int(total["calls"])
            summaries.append(
                {
                    "task_type": total["task_type"],
                    "handler": total["handler"],
                    "calls": calls,
                    "total_input_copy_mb": _mb(total["input_copy_bytes"]),
                    "max_input_copy_mb": _mb(total["max_input_copy_bytes"]),
                    "average_duration_ms": round(
                        total["duration_ms"] / calls if calls else 0.0,
                        3,
                    ),
                    "max_duration_ms": round(total["max_duration_ms"], 3),
                }
            )
        return sorted(
            summaries,
            key=lambda item: item["total_input_copy_mb"],
            reverse=True,
        )

    def _report_error(self, exc):
        if self._error_reported:
            return
        self._error_reported = True
        print(f"[GameSceneMemory] 监控失败，自动化继续运行: {exc}", flush=True)

    def _write_event(self, record):
        try:
            self.writer(self.path, record)
        except Exception as exc:
            self._report_error(exc)

    def record_phase(self, phase, before, after, duration_ms):
        if not self.enabled or (before is None and after is None):
            return
        with self._lock:
            if self._baseline_snapshot is None:
                self._baseline_snapshot = before or after
            record = {
                "event": "gamescene_memory",
                "timestamp": _iso_now(),
                "source": "aw/autogame/tools/GameSceneHandler.py",
                "reason": ["initialization_phase"],
                "phase": str(phase),
                "duration_ms": round(float(duration_ms), 3),
                "process_before": self._compact_snapshot(before),
                "process_after": self._compact_snapshot(after),
                "delta_memory_mb": self._delta_mb(before, after),
                "growth_since_gamescene_start_mb": self._delta_mb(
                    self._baseline_snapshot, after,
                ),
                "runtime": self._runtime_snapshot(),
                "notes": "Memory is process-scoped; this phase is correlation evidence.",
            }
            self._write_event(record)
            self._last_emit_at = time.monotonic()

    def record_frame(self, tasks_config, task_metrics, before, after, duration_ms):
        if not self.enabled or (before is None and after is None):
            return
        now = time.monotonic()
        signature = self._task_signature(tasks_config)
        delta_mb = self._delta_mb(before, after)
        with self._lock:
            self._call_index += 1
            self._update_handler_totals(task_metrics)
            reasons = []
            if self._last_signature is None:
                reasons.append("first_frame")
            elif signature != self._last_signature:
                reasons.append("task_set_changed")
            if now - self._last_emit_at >= self.interval_seconds:
                reasons.append("interval")
            if delta_mb is not None and delta_mb >= GAME_SCENE_MEMORY_LARGE_DELTA_MB:
                reasons.append("large_frame_delta")
            self._last_signature = signature
            if not reasons:
                return
            if self._baseline_snapshot is None:
                self._baseline_snapshot = before or after
            runtime = self._runtime_snapshot()
            allocated_blocks = runtime.get("python_allocated_blocks")
            if self._baseline_python_allocated_blocks is None:
                self._baseline_python_allocated_blocks = allocated_blocks
            if allocated_blocks is not None and self._baseline_python_allocated_blocks is not None:
                runtime["python_allocated_blocks_growth"] = (
                    allocated_blocks - self._baseline_python_allocated_blocks
                )
            frame_copy_bytes = sum(
                int(item.get("input_copy_bytes") or 0) for item in task_metrics
            )
            record = {
                "event": "gamescene_memory",
                "timestamp": _iso_now(),
                "source": "aw/autogame/tools/GameSceneHandler.py",
                "reason": reasons,
                "call_index": self._call_index,
                "duration_ms": round(float(duration_ms), 3),
                "task_count": len(tasks_config),
                "exclusive_task": len(tasks_config) == 1,
                "detailed_task_memory": any(
                    bool(item.get("detailed_memory_sample")) for item in task_metrics
                ),
                "special_input_copy_count": sum(
                    1 for item in task_metrics if item.get("input_copy_bytes")
                ),
                "frame_input_copy_mb": _mb(frame_copy_bytes),
                "tasks": self._task_summary(task_metrics),
                "handler_totals": self._handler_totals_summary(),
                "process_before": self._compact_snapshot(before),
                "process_after": self._compact_snapshot(after),
                "delta_memory_mb": delta_mb,
                "growth_since_gamescene_start_mb": self._delta_mb(
                    self._baseline_snapshot, after,
                ),
                "notes": (
                    "Per-task memory ownership is not measurable. A single-task frame "
                    "is stronger attribution evidence; concurrent task deltas are correlation only. "
                    "Copy bytes are exact NumPy buffer sizes; process deltas may overlap between tasks."
                ),
                "runtime": runtime,
            }
            self._write_event(record)
            self._last_emit_at = now


def load_stage_info(project_case, info_mod):
    """加载阶段配置，并合并项目内可选的运行时分组覆盖。"""
    stage_info = copy.deepcopy(getattr(info_mod, "STAGE_INFO"))
    override_module_path = (
        f"aw.autogame.customs_examples.{project_case}.resource.stage_group_config"
    )
    try:
        override_mod = importlib.import_module(override_module_path)
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "")
        if not (
            missing_name == override_module_path
            or override_module_path.startswith(f"{missing_name}.")
        ):
            raise
        return stage_info

    overrides = getattr(override_mod, "STAGE_GROUP_OVERRIDES", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"{override_module_path}.STAGE_GROUP_OVERRIDES 必须是字典")
    for stage_name, override in overrides.items():
        if not isinstance(override, dict):
            continue
        stage_data = stage_info.get(stage_name)
        if not isinstance(stage_data, dict):
            raise ValueError(f"分组覆盖引用了不存在的阶段: {stage_name}")
        initial_group = override.get("initial_group")
        if initial_group:
            stage_data["initial_group"] = str(initial_group)
        override_groups = override.get("groups", {})
        if isinstance(override_groups, dict):
            stage_data.setdefault("groups", {}).update(copy.deepcopy(override_groups))
    return stage_info

def load_special_handler(project_case):
    if not project_case:
        raise ValueError("TARGET_PROJECT_CASE 未设置，无法定位资源路径")

    handler_path = f"aw.autogame.customs_examples.{project_case}.resource.SpecialSceneHandler"
    try:
        special_handler_module = importlib.import_module(handler_path)
        print(f"成功从项目 [{project_case}] 加载 SpecialSceneHandler")
        return special_handler_module
    except ImportError as e:
        print(f"路径错误: 无法在 {handler_path} 找到 SpecialSceneHandler 模块")
        raise e

class GameImageProcessor:
    def __init__(self, project_name, special_handler=None, screen_resolution=None):
        self._memory_monitor = GameSceneMemoryMonitor()
        self._executor_lock = threading.Lock()
        self._executor = None
        self._executor_closed = False
        self.project_root = os.path.join(r"aw/autogame/customs_examples", project_name)

        phase_started = time.perf_counter()
        phase_before = self._memory_monitor.snapshot()
        self.template_cache = self._load_templates()
        self._memory_monitor.record_phase(
            "load_templates",
            phase_before,
            self._memory_monitor.snapshot(),
            (time.perf_counter() - phase_started) * 1000.0,
        )

        phase_started = time.perf_counter()
        phase_before = self._memory_monitor.snapshot()
        self.special_handler = special_handler or load_special_handler(project_name)
        self._memory_monitor.record_phase(
            "load_special_handler",
            phase_before,
            self._memory_monitor.snapshot(),
            (time.perf_counter() - phase_started) * 1000.0,
        )
        self.task_config = None
        self.screen_w, self.screen_h = self._resolve_screen_resolution(screen_resolution)

    def _get_memory_monitor(self):
        monitor = getattr(self, "_memory_monitor", None)
        if monitor is None:
            monitor = GameSceneMemoryMonitor()
            self._memory_monitor = monitor
        return monitor

    def _get_executor(self):
        executor_lock = getattr(self, "_executor_lock", None)
        if executor_lock is None:
            executor_lock = threading.Lock()
            self._executor_lock = executor_lock
        with executor_lock:
            if getattr(self, "_executor_closed", False):
                raise RuntimeError("GameImageProcessor 已关闭，不能继续处理画面")
            executor = getattr(self, "_executor", None)
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=8,
                    thread_name_prefix="GameSceneWorker",
                )
                self._executor = executor
            return executor

    def close(self, wait=True):
        """Stop the persistent scene worker pool exactly once."""
        executor_lock = getattr(self, "_executor_lock", None)
        if executor_lock is None:
            executor_lock = threading.Lock()
            self._executor_lock = executor_lock
        with executor_lock:
            if getattr(self, "_executor_closed", False):
                return
            self._executor_closed = True
            executor = getattr(self, "_executor", None)
            self._executor = None
        if executor is not None:
            executor.shutdown(
                wait=bool(wait),
                cancel_futures=not bool(wait),
            )

    def _resolve_screen_resolution(self, screen_resolution=None):
        if isinstance(screen_resolution, (tuple, list)) and len(screen_resolution) == 2:
            try:
                width, height = int(screen_resolution[0]), int(screen_resolution[1])
            except (TypeError, ValueError):
                width, height = 0, 0
            if width > 0 and height > 0:
                return width, height
        env_w = os.environ.get("AUTOGAME_SCREEN_WIDTH")
        env_h = os.environ.get("AUTOGAME_SCREEN_HEIGHT")
        if env_w and env_h:
            try:
                return int(env_w), int(env_h)
            except ValueError:
                pass

        screen_w, screen_h = get_resolution()
        if screen_w and screen_h:
            return int(screen_w), int(screen_h)
        return None, None

    def _load_templates(self):
        cache = {}
        if not os.path.exists(self.project_root):
            print(f"Warning: Project directory not found: {self.project_root}")
            return cache
        valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(valid_exts):
                    abs_path = os.path.join(root, file)
                    img_arr = np.fromfile(abs_path, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    cache[os.path.normpath(abs_path)] = img
        return cache

    @staticmethod
    def _offset_bbox(bbox, offset_x, offset_y):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return bbox
        return [
            int(bbox[0]) + offset_x,
            int(bbox[1]) + offset_y,
            int(bbox[2]) + offset_x,
            int(bbox[3]) + offset_y,
        ]

    @staticmethod
    def _offset_contours(contours, offset_x, offset_y):
        if not isinstance(contours, list):
            return contours
        shifted = []
        for contour in contours:
            if not isinstance(contour, list):
                continue
            points = []
            for point in contour:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                points.append([int(point[0]) + offset_x, int(point[1]) + offset_y])
            if len(points) >= 2:
                shifted.append(points)
        return shifted

    def _map_special_visualizations(self, result, crop_xyxy):
        if not isinstance(result, dict):
            return result

        visuals = result.get("__visualizations__")
        if not isinstance(visuals, list):
            return result

        x1, y1, x2, y2 = crop_xyxy
        mapped_visuals = []
        for visual in visuals:
            if not isinstance(visual, dict):
                continue
            item = dict(visual)
            if item.get("coord", "local") == "local":
                item["bbox_xyxy"] = self._offset_bbox(item.get("bbox_xyxy"), x1, y1)
                item["contours"] = self._offset_contours(item.get("contours"), x1, y1)
                item["coord"] = "frame"
                item["source_crop_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
            mapped_visuals.append(item)

        mapped_result = dict(result)
        mapped_result["__visualizations__"] = mapped_visuals
        return mapped_result

    @staticmethod
    def _split_special_timing_result(method, result):
        if getattr(method, "__special_timing_enabled__", False):
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
        return result, None

    @staticmethod
    def _format_special_result_for_info(result, timing_ms):
        if timing_ms is None:
            return result
        return [result, [timing_ms]]

    def process(self, raw_frame, tasks_config, buffer_ratio=0.3):
        memory_monitor = self._get_memory_monitor()
        should_profile_details = getattr(
            memory_monitor,
            "should_profile_task_details",
            lambda _tasks: False,
        )
        profile_task_details = bool(should_profile_details(tasks_config))
        frame_started = time.perf_counter()
        memory_before = memory_monitor.snapshot()
        self.task_config = tasks_config
        curr_h, curr_w = raw_frame.shape[:2]
        results = {}
        task_metrics = []
        task_metrics_lock = threading.Lock()

        def _execute_task(task_id, config, metric):
            try:
                task_type = config.get('type')
                origin_w = config.get('origin_width')
                origin_h = config.get('origin_height')

                if not origin_w or not origin_h:
                    return task_id, "Error: Missing origin resolution info"

                global_scale = curr_w / origin_w

                # Case 1: 特殊区域
                if task_type == 'special':
                    area_config = config.get('area_config') or config
                    copy_before = memory_monitor.snapshot() if profile_task_details else None
                    copy_started = time.perf_counter()
                    if 'anchor' in area_config or 'rect' in area_config:
                        x1, y1, x2, y2 = resolve_area_rect_for_frame(
                            curr_w,
                            curr_h,
                            area_config,
                            self.screen_w,
                            self.screen_h,
                            origin_w,
                            origin_h,
                        )
                        x1 = max(0, min(curr_w, x1))
                        y1 = max(0, min(curr_h, y1))
                        x2 = max(0, min(curr_w, x2))
                        y2 = max(0, min(curr_h, y2))
                        target_img = np.ascontiguousarray(raw_frame[y1:y2, x1:x2]).copy()
                        copy_kind = (
                            "full_frame_copy"
                            if (x1, y1, x2, y2) == (0, 0, curr_w, curr_h)
                            else "roi_copy"
                        )
                    else:
                        x1, y1, x2, y2 = 0, 0, curr_w, curr_h
                        target_img = np.ascontiguousarray(raw_frame).copy()
                        copy_kind = "full_frame_copy"

                    metric.update(
                        {
                            "input_copy_bytes": int(target_img.nbytes),
                            "input_copy_kind": copy_kind,
                            "input_shape": [int(value) for value in target_img.shape],
                            "copy_duration_ms": round(
                                (time.perf_counter() - copy_started) * 1000.0,
                                3,
                            ),
                            "copy_end_offset_ms": round(
                                (time.perf_counter() - frame_started) * 1000.0,
                                3,
                            ),
                        }
                    )
                    copy_after = None
                    if profile_task_details:
                        copy_after = memory_monitor.snapshot()
                        metric.update(
                            {
                                "input_owns_data": bool(target_img.flags["OWNDATA"]),
                                "input_shares_frame_memory": bool(
                                    np.shares_memory(target_img, raw_frame)
                                ),
                                "process_before_copy": memory_monitor._compact_snapshot(
                                    copy_before
                                ),
                                "process_after_copy": memory_monitor._compact_snapshot(
                                    copy_after
                                ),
                                "copy_process_delta_mb": memory_monitor._delta_mb(
                                    copy_before,
                                    copy_after,
                                ),
                            }
                        )

                    handler_name = config.get('handler_name', task_id)
                    method = getattr(self.special_handler, handler_name, None)
                    if not method:
                        return task_id, f"Error: {handler_name} not found"
                    handler_kwargs = {}
                    if 'seg_name' in area_config:
                        seg_name = area_config.get('seg_name')
                        if not isinstance(seg_name, str) or not seg_name.strip():
                            return task_id, "Error: seg_name must be a non-empty string"
                        handler_kwargs['seg_name'] = seg_name.strip()
                    if handler_name == 'sam3' and 'version' in area_config:
                        version = area_config.get('version')
                        if (
                            isinstance(version, bool)
                            or not isinstance(version, int)
                            or version not in (0, 1)
                        ):
                            return task_id, "Error: sam3 version must be integer 0 or 1"
                        handler_kwargs['version'] = version
                    handler_started = time.perf_counter()
                    raw_special_result = method(target_img, **handler_kwargs)
                    metric["handler_duration_ms"] = round(
                        (time.perf_counter() - handler_started) * 1000.0,
                        3,
                    )
                    metric["handler_end_offset_ms"] = round(
                        (time.perf_counter() - frame_started) * 1000.0,
                        3,
                    )
                    if profile_task_details:
                        handler_after = memory_monitor.snapshot()
                        metric["process_after_handler"] = memory_monitor._compact_snapshot(
                            handler_after
                        )
                        metric["handler_process_delta_mb"] = memory_monitor._delta_mb(
                            copy_after,
                            handler_after,
                        )
                    special_result, timing_ms = self._split_special_timing_result(method, raw_special_result)
                    mapped_result = self._map_special_visualizations(
                        special_result,
                        (x1, y1, x2, y2),
                    )
                    return task_id, self._format_special_result_for_info(mapped_result, timing_ms)

                # Case 2: 模板匹配
                elif task_type == 'template':
                    scope = config.get('scope')
                    scope_config = config.get('scope_config')
                    tpl_relative_path = config.get('template_path')

                    if scope_config or scope:
                        px_min_x, px_min_y, px_max_x, px_max_y = resolve_area_rect_for_frame(
                            curr_w,
                            curr_h,
                            scope_config or {"rect": scope},
                            self.screen_w,
                            self.screen_h,
                            origin_w,
                            origin_h,
                        )
                        px_min_x = max(0, min(curr_w, px_min_x))
                        px_min_y = max(0, min(curr_h, px_min_y))
                        px_max_x = max(0, min(curr_w, px_max_x))
                        px_max_y = max(0, min(curr_h, px_max_y))

                        w_rect = px_max_x - px_min_x
                        h_rect = px_max_y - px_min_y
                        buf_w = int(w_rect * buffer_ratio)
                        buf_h = int(h_rect * buffer_ratio)

                        crop_x1 = max(0, px_min_x - buf_w)
                        crop_y1 = max(0, px_min_y - buf_h)
                        crop_x2 = min(curr_w, px_max_x + buf_w)
                        crop_y2 = min(curr_h, px_max_y + buf_h)

                        search_img = raw_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        offset = (crop_x1, crop_y1)
                    else:
                        search_img = raw_frame
                        offset = (0, 0)

                    full_tpl_path = os.path.normpath(os.path.join(self.project_root, tpl_relative_path))
                    tpl_img_raw = self.template_cache.get(full_tpl_path)

                    if tpl_img_raw is None and os.path.exists(full_tpl_path):
                        arr = np.fromfile(full_tpl_path, dtype=np.uint8)
                        tpl_img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if tpl_img_raw is None:
                        return task_id, False

                    tpl_h, tpl_w = tpl_img_raw.shape[:2]
                    new_w = int(tpl_w * global_scale)
                    new_h = int(tpl_h * global_scale)

                    if new_w > 0 and new_h > 0:
                        tpl_img_resized = cv2.resize(tpl_img_raw, (new_w, new_h))
                    else:
                        tpl_img_resized = tpl_img_raw

                    match_res = find_template_center_multiscale(
                        search_img,
                        tpl_img_resized,
                    )

                    if match_res:
                        local_x, local_y = match_res
                        final_pixel_x = local_x + offset[0]
                        final_pixel_y = local_y + offset[1]
                        norm_x = final_pixel_x / curr_w
                        norm_y = final_pixel_y / curr_h
                        return task_id, (norm_x, norm_y)
                    else:
                        return task_id, False

            except Exception as e:
                return task_id, f"Err: {e}"

        def _execute_task_profiled(task_id, config):
            task_started = time.perf_counter()
            task_before = memory_monitor.snapshot() if profile_task_details else None
            metric = {
                "task_id": str(task_id),
                "task_type": str(config.get("type") or "unknown"),
                "handler": str(config.get("handler_name") or task_id),
                "thread_name": threading.current_thread().name,
                "python_thread_id": int(threading.get_ident()),
                "task_start_offset_ms": round(
                    (task_started - frame_started) * 1000.0,
                    3,
                ),
            }
            get_native_id = getattr(threading, "get_native_id", None)
            if callable(get_native_id):
                metric["native_thread_id"] = int(get_native_id())
            if profile_task_details:
                metric["detailed_memory_sample"] = True
            try:
                outcome = _execute_task(task_id, config, metric)
                result_value = outcome[1]
                metric["result_type"] = type(result_value).__name__
                if isinstance(result_value, (list, tuple, dict, set, str, bytes)):
                    metric["result_size"] = len(result_value)
                return outcome
            finally:
                metric["duration_ms"] = round(
                    (time.perf_counter() - task_started) * 1000.0,
                    3,
                )
                metric["task_end_offset_ms"] = round(
                    (time.perf_counter() - frame_started) * 1000.0,
                    3,
                )
                if profile_task_details:
                    task_after = memory_monitor.snapshot()
                    metric["process_after_task"] = memory_monitor._compact_snapshot(
                        task_after
                    )
                    metric["task_process_delta_mb"] = memory_monitor._delta_mb(
                        task_before,
                        task_after,
                    )
                with task_metrics_lock:
                    task_metrics.append(metric)

        executor = self._get_executor()
        futures = [
            executor.submit(_execute_task_profiled, k, v)
            for k, v in tasks_config.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            tid, res = future.result()
            results[tid] = res
        memory_monitor.record_frame(
            tasks_config,
            task_metrics,
            memory_before,
            memory_monitor.snapshot(),
            (time.perf_counter() - frame_started) * 1000.0,
        )
        return results

class StageLogicController:
    def __init__(self, screen_resolution=None):
        """
        初始化：动态加载环境变量指定的项目配置
        """
        # 从环境变量获取项目 case 名
        project_case = os.environ.get("TARGET_PROJECT_CASE")
        if not project_case:
            raise ValueError("Environment variable 'TARGET_PROJECT_CASE' is not set!")

        # 动态导包
        module_path = f"aw.autogame.customs_examples.{project_case}.info"
        info_mod = importlib.import_module(module_path)

        self.project_name = getattr(info_mod, "PROJECT_NAME")
        if screen_resolution is None:
            self.processor = GameImageProcessor(project_case)
        else:
            self.processor = GameImageProcessor(
                project_case,
                screen_resolution=screen_resolution,
            )
        self.raw_stage_info = load_stage_info(project_case, info_mod)
        self.stage_info = lock_stage_info_scene_resolutions(
            self.raw_stage_info,
            self.processor.screen_w,
            self.processor.screen_h,
        )
        print(f"[{self.project_name}] 场景分辨率已锁定: {self.processor.screen_w}x{self.processor.screen_h}")
        print(f"[{self.project_name}] 逻辑控制器已就绪。")

    def close(self, wait=True):
        processor = getattr(self, "processor", None)
        close = getattr(processor, "close", None)
        if callable(close):
            close(wait=wait)

    def refresh_resolution(self, screen_width, screen_height):
        """同步屏幕分辨率，并重新选择当前分辨率对应的场景配置。"""
        screen_width = int(screen_width)
        screen_height = int(screen_height)
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError(f"无效的屏幕分辨率: {screen_width}x{screen_height}")

        self.processor.screen_w = screen_width
        self.processor.screen_h = screen_height
        self.stage_info = lock_stage_info_scene_resolutions(
            self.raw_stage_info,
            screen_width,
            screen_height,
        )
        return self.stage_info

    def get_stage_groups(self, stage_name):
        stage_data = self.stage_info.get(stage_name, {})
        groups = stage_data.get('groups', {}) if isinstance(stage_data, dict) else {}
        if not isinstance(groups, dict) or not groups:
            return [DEFAULT_GROUP_NAME]
        names = [DEFAULT_GROUP_NAME]
        for name in groups.keys():
            if name != DEFAULT_GROUP_NAME:
                names.append(name)
        return names

    def has_group(self, stage_name, group_name):
        if not group_name:
            group_name = DEFAULT_GROUP_NAME
        return group_name in self.get_stage_groups(stage_name)

    def get_initial_group(self, stage_name):
        """返回进入阶段时应启用的运行分组。

        未配置 initial_group 的老工程继续使用内置的“默认”全量组；
        配置值无效时也安全回退，避免阶段切换后处于不存在的分组。
        """
        stage_data = self.stage_info.get(stage_name, {})
        configured = (
            stage_data.get('initial_group', DEFAULT_GROUP_NAME)
            if isinstance(stage_data, dict)
            else DEFAULT_GROUP_NAME
        )
        group_name = str(configured or DEFAULT_GROUP_NAME).strip()
        if self.has_group(stage_name, group_name):
            return group_name
        print(
            f"[WARN] 阶段 '{stage_name}' 的 initial_group '{group_name}' 不存在，"
            f"回退到 '{DEFAULT_GROUP_NAME}'。"
        )
        return DEFAULT_GROUP_NAME

    def _resolve_group_filter(self, stage_data, group_name):
        if not group_name or group_name == DEFAULT_GROUP_NAME:
            return None
        groups = stage_data.get('groups', {}) if isinstance(stage_data, dict) else {}
        if not isinstance(groups, dict):
            return None
        group_data = groups.get(group_name)
        if group_data is None:
            return set()
        if isinstance(group_data, dict) and group_data.get('all'):
            return None

        def parse_item_refs(raw_items):
            refs = set()
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                scene_name = str(item.get('scene', '')).strip()
                item_type = str(item.get('type', '')).strip()
                item_name = str(item.get('name', '')).strip()
                if scene_name and item_name and item_type in GROUPABLE_ITEM_TYPES:
                    refs.add((scene_name, item_type, item_name))
            return refs

        raw_items = group_data.get('items', []) if isinstance(group_data, dict) else []
        allowed = parse_item_refs(raw_items)
        excluded_items = (
            group_data.get('exclude_items')
            if isinstance(group_data, dict)
            else None
        )
        if not isinstance(excluded_items, list):
            return allowed

        allowed = set()
        scenes = stage_data.get('scenes', {}) if isinstance(stage_data, dict) else {}
        for scene_name, scene_info in scenes.items():
            if not isinstance(scene_info, dict):
                continue
            for area_name in scene_info.get('areas', {}):
                allowed.add((scene_name, 'area', area_name))
            for area_name in scene_info.get('special_areas', {}):
                allowed.add((scene_name, 'special_area', area_name))
        return allowed - parse_item_refs(excluded_items)

    def process_frame(self, frame_img, current_stage_name, group_name=DEFAULT_GROUP_NAME):
        """
        处理单帧逻辑。

        Args:
            frame_img: 当前视频帧
            current_stage_name (str): 由 Framework 传入的当前阶段名称 (如 '关闭弹窗')
            group_name (str): 当前阶段内要识别的分组名，默认分组识别全部区域和特殊区域

        Returns:
            dict: 检测结果
        """
        # 1. 如果 Framework 没传阶段名，或者传了 None，直接返回
        if not current_stage_name:
            return {}

        # 2. 构建任务配置 (根据传入的 stage_name 查找配置)
        # 这里不再读取全局 STAGE_DICT，而是完全依赖传入的参数
        stage_data = self.stage_info.get(current_stage_name, {})
        scenes = stage_data.get('scenes', {})
        group_filter = self._resolve_group_filter(stage_data, group_name)
        tasks_config = {}

        # 遍历该阶段下的所有场景和区域
        for scene_name, scene_info in scenes.items():
            origin_w = scene_info.get('width')
            origin_h = scene_info.get('height')

            # 普通 Areas (模板匹配)
            areas = scene_info.get('areas', {})
            required_control_anchors = {
                str(point_data.get('relative_to') or '').strip()
                for point_data in scene_info.get('points', {}).values()
                if (
                    isinstance(point_data, dict)
                    and point_data.get('positioning') == 'relative'
                    and point_data.get('relative_to')
                )
            }
            for area_name, area_data in areas.items():
                if (
                    group_filter is not None
                    and (scene_name, 'area', area_name) not in group_filter
                    and area_name not in required_control_anchors
                ):
                    continue
                task_key = f"{scene_name}__{area_name}"
                scope = area_data.get('search_scope', area_data.get('rect'))
                if area_data.get('search_scope'):
                    scope_config = {'rect': area_data.get('search_scope')}
                elif 'anchor' in area_data:
                    scope_config = area_data
                elif area_data.get('rect'):
                    scope_config = {'rect': area_data.get('rect')}
                else:
                    scope_config = None
                tasks_config[task_key] = {
                    'type': 'template',
                    'scope': scope,
                    'scope_config': scope_config,
                    'template_path': area_data.get('template'),
                    'origin_width': origin_w,
                    'origin_height': origin_h
                }

            # Special Areas (特殊函数)
            special_areas = scene_info.get('special_areas', {})
            for sa_name, sa_data in special_areas.items():
                if group_filter is not None and (scene_name, 'special_area', sa_name) not in group_filter:
                    continue
                task_key = f"{scene_name}__{sa_name}"
                tasks_config[task_key] = {
                    'type': 'special',
                    'rect': sa_data.get('rect'),
                    'area_config': sa_data,
                    'handler_name': sa_data.get('handler_name', sa_name),
                    'origin_width': origin_w,
                    'origin_height': origin_h
                }

        # 3. 调用处理器执行具体计算
        if not tasks_config:
            return {}

        final_results = self.processor.process(frame_img, tasks_config, buffer_ratio=0.3)

        return final_results

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 初始化 (仅一次)
    logic_controller = StageLogicController()

    # 模拟数据
    mock_frame = np.zeros((384, 826, 3), dtype=np.uint8)

    # 2. 循环调用 (只需传 frame)
    # 假设在其他地方 STAGE_DICT['开始游戏'] 已经被置为 True
    results = logic_controller.process_frame(mock_frame)

    print("\n运行结果:")
    for k, v in results.items():
        print(f"{k}: {v}")
