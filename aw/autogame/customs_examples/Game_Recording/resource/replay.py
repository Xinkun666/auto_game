"""Game_Recording 历史记录发现、动作解析和布局恢复。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .layout import KeyPoint, RESERVED_KEYS, load_key_layout, normalize_key_name


MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})
DRAG_KEYS = frozenset({"up", "down", "left", "right"})
HOLD_AND_DRAG_SEMANTICS_VERSION = 2


class ReplayError(ValueError):
    """录制记录无法用于回放。"""


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: float
    event: str
    key: str
    normalized_position: tuple[float, float] | None = None


@dataclass(frozen=True)
class ReplayRecord:
    directory: Path
    action_path: Path
    action_format: str
    session_path: Path | None
    initial_view_path: Path | None
    recorded_at: datetime
    duration_seconds: float
    stop_reason: str
    frame_count: int
    action_count: int

    @property
    def display_time(self) -> str:
        return self.recorded_at.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def title(self) -> str:
        return (
            f"{self.directory.name}    {self.display_time}    "
            f"{self.duration_seconds:.1f} 秒    {self.action_count} 个动作"
        )


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _recorded_at(directory: Path) -> datetime:
    for pattern in ("%Y%m%d-%H%M%S-%f", "%Y%m%d-%H%M%S"):
        try:
            return datetime.strptime(directory.name, pattern)
        except ValueError:
            continue
    return datetime.fromtimestamp(directory.stat().st_mtime)


def _safe_nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result >= 0 else default


def _safe_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(result, 0)


def _candidate_directories(records_root: Path) -> Iterable[Path]:
    seen = set()
    for filename in ("action_raw.json", "action_step.json"):
        for action_path in Path(records_root).rglob(filename):
            directory = action_path.parent.resolve()
            if directory not in seen:
                seen.add(directory)
                yield directory


def discover_replay_records(records_root: Path) -> list[ReplayRecord]:
    """扫描新版时间批次目录，同时兼容旧版直接录制目录。"""
    records_root = Path(records_root).expanduser().resolve()
    if not records_root.is_dir():
        return []
    records = []
    for directory in _candidate_directories(records_root):
        raw_path = directory / "action_raw.json"
        step_path = directory / "action_step.json"
        raw_payload = _read_json(raw_path)
        step_payload = _read_json(step_path)
        if isinstance(raw_payload, list) and raw_payload:
            action_path = raw_path
            action_format = "raw"
            action_count = len(raw_payload)
            fallback_duration = (
                _safe_nonnegative_float(raw_payload[-1].get("time"))
                if isinstance(raw_payload[-1], Mapping)
                else 0.0
            )
        elif isinstance(step_payload, list) and step_payload:
            action_path = step_path
            action_format = "step"
            action_count = len(step_payload)
            fallback_duration = (
                _safe_nonnegative_float(step_payload[-1].get("time"))
                if isinstance(step_payload[-1], Mapping)
                else 0.0
            )
        else:
            continue

        session_path = directory / "session.json"
        session = _read_json(session_path, {})
        if not isinstance(session, Mapping):
            session = {}
        records.append(
            ReplayRecord(
                directory=directory,
                action_path=action_path,
                action_format=action_format,
                session_path=session_path if session_path.is_file() else None,
                initial_view_path=(
                    directory / "initial_view.png"
                    if (directory / "initial_view.png").is_file()
                    else None
                ),
                recorded_at=_recorded_at(directory),
                duration_seconds=_safe_nonnegative_float(
                    session.get("duration_seconds"), fallback_duration
                ),
                stop_reason=str(session.get("stop_reason") or "未记录"),
                frame_count=_safe_nonnegative_int(session.get("frame_count")),
                action_count=action_count,
            )
        )
    return sorted(records, key=lambda item: item.recorded_at, reverse=True)


def _validate_timestamp(value: Any, index: int) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"第 {index + 1} 个动作的 time 无效。") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ReplayError(f"第 {index + 1} 个动作的 time 必须是非负有限数。")
    return timestamp


def _events_from_raw(
    payload: Sequence[Any],
    input_semantics_version: int,
) -> list[ReplayEvent]:
    events = []
    previous_time = -1.0
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ReplayError(f"第 {index + 1} 个原始动作不是对象。")
        timestamp = _validate_timestamp(item.get("time"), index)
        if timestamp < previous_time:
            raise ReplayError("原始动作的时间戳不是单调递增。")
        previous_time = timestamp
        event = str(item.get("event") or "").strip().lower()
        if event not in {"press", "release", "drag"}:
            raise ReplayError(f"第 {index + 1} 个原始动作类型无效：{event or '空'}。")
        key = normalize_key_name(item.get("key"))
        if not key or key in RESERVED_KEYS:
            raise ReplayError(f"第 {index + 1} 个原始动作键位无效：{key or '空'}。")
        if event == "drag":
            if key not in DRAG_KEYS:
                raise ReplayError(f"第 {index + 1} 个离散滑动方向无效：{key}。")
            normalized_position = None
            device_actions = item.get("device_actions", [])
            if isinstance(device_actions, list):
                for action in reversed(device_actions):
                    if not isinstance(action, Mapping) or action.get("method") != "touch_move":
                        continue
                    raw_position = action.get("normalized_position")
                    if isinstance(raw_position, (list, tuple)) and len(raw_position) == 2:
                        try:
                            norm_x, norm_y = float(raw_position[0]), float(raw_position[1])
                        except (TypeError, ValueError):
                            break
                        if 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0:
                            normalized_position = (norm_x, norm_y)
                    break
            events.append(ReplayEvent(timestamp, event, key, normalized_position))
            continue
        if (
            input_semantics_version < HOLD_AND_DRAG_SEMANTICS_VERSION
            and key not in MOVEMENT_KEYS
        ):
            if event == "press":
                events.append(ReplayEvent(timestamp, "tap", key))
            continue
        events.append(ReplayEvent(timestamp, event, key))
    return events


def _events_from_steps(
    payload: Sequence[Any],
    input_semantics_version: int,
) -> list[ReplayEvent]:
    events = []
    previous_time = -1.0
    pressed: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ReplayError(f"第 {index + 1} 个回放步骤不是对象。")
        timestamp = _validate_timestamp(item.get("time"), index)
        if timestamp < previous_time:
            raise ReplayError("回放步骤的时间戳不是单调递增。")
        previous_time = timestamp
        params = item.get("params", {})
        keys = params.get("keyboard_keys", []) if isinstance(params, Mapping) else []
        if not isinstance(keys, (list, tuple, set)):
            raise ReplayError(f"第 {index + 1} 个回放步骤的 keyboard_keys 无效。")
        target = {normalize_key_name(key) for key in keys}
        target.discard("")
        if target.intersection(RESERVED_KEYS):
            raise ReplayError(f"第 {index + 1} 个回放步骤占用了保留键。")
        # 先松开旧键再按下新键，与录制窗口的切向语义一致。
        for key in sorted(pressed - target):
            if (
                input_semantics_version >= HOLD_AND_DRAG_SEMANTICS_VERSION
                or key in MOVEMENT_KEYS
            ):
                events.append(ReplayEvent(timestamp, "release", key))
        for key in sorted(target - pressed):
            event = (
                "press"
                if input_semantics_version >= HOLD_AND_DRAG_SEMANTICS_VERSION
                or key in MOVEMENT_KEYS
                else "tap"
            )
            events.append(ReplayEvent(timestamp, event, key))
        pressed = target
    if pressed:
        final_time = previous_time if previous_time >= 0 else 0.0
        for key in sorted(pressed):
            if (
                input_semantics_version >= HOLD_AND_DRAG_SEMANTICS_VERSION
                or key in MOVEMENT_KEYS
            ):
                events.append(ReplayEvent(final_time, "release", key))
    return events


def load_replay_events(record: ReplayRecord) -> list[ReplayEvent]:
    payload = _read_json(record.action_path)
    if not isinstance(payload, list) or not payload:
        raise ReplayError(f"回放文件为空或格式错误：{record.action_path}")
    session = _read_json(record.session_path, {}) if record.session_path is not None else {}
    input_semantics_version = _safe_nonnegative_int(
        session.get("input_semantics_version") if isinstance(session, Mapping) else 0
    )
    if (
        record.action_format == "step"
        and input_semantics_version >= HOLD_AND_DRAG_SEMANTICS_VERSION
        and isinstance(session, Mapping)
        and bool(session.get("contains_drag_events"))
    ):
        raise ReplayError(
            "该记录包含按住后的离散滑动，但 action_raw.json 已缺失，"
            "action_step.json 无法完整恢复这段轨迹。"
        )
    events = (
        _events_from_raw(payload, input_semantics_version)
        if record.action_format == "raw"
        else _events_from_steps(payload, input_semantics_version)
    )
    if not events:
        raise ReplayError(f"回放文件中没有可执行动作：{record.action_path}")
    return events


def _recorded_layout(record: ReplayRecord) -> tuple[Mapping[str, Any], tuple[int, int]]:
    if record.session_path is None:
        return {}, (0, 0)
    session = _read_json(record.session_path, {})
    if not isinstance(session, Mapping):
        return {}, (0, 0)
    layout = session.get("layout", {})
    screen_size = session.get("screen_size", [])
    if not isinstance(layout, Mapping):
        layout = {}
    if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
        screen_size = (0, 0)
    return layout, (int(screen_size[0] or 0), int(screen_size[1] or 0))


def load_replay_layout(
    record: ReplayRecord,
    info_module: Any,
    screen_width: int,
    screen_height: int,
) -> dict[str, KeyPoint]:
    """优先恢复录制当时的布局，个别缺失键位再用当前 info.py 补齐。"""
    recorded, recorded_screen = _recorded_layout(record)
    result: dict[str, KeyPoint] = {}
    for raw_key, raw_point in recorded.items():
        key = normalize_key_name(raw_key)
        if not key or key in RESERVED_KEYS or not isinstance(raw_point, Mapping):
            continue
        normalized = raw_point.get("normalized_position")
        if isinstance(normalized, (list, tuple)) and len(normalized) == 2:
            try:
                norm_x, norm_y = float(normalized[0]), float(normalized[1])
            except (TypeError, ValueError):
                continue
        else:
            position = raw_point.get("position")
            if (
                not isinstance(position, (list, tuple))
                or len(position) != 2
                or recorded_screen[0] <= 0
                or recorded_screen[1] <= 0
            ):
                continue
            norm_x = float(position[0]) / recorded_screen[0]
            norm_y = float(position[1]) / recorded_screen[1]
        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            continue
        joystick_center = None
        raw_center = raw_point.get("joystick_center_normalized")
        if isinstance(raw_center, (list, tuple)) and len(raw_center) == 2:
            try:
                center_x, center_y = float(raw_center[0]), float(raw_center[1])
            except (TypeError, ValueError):
                center_x, center_y = -1.0, -1.0
            if 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0:
                joystick_center = (
                    min(max(int(round(center_x * screen_width)), 0), screen_width - 1),
                    min(max(int(round(center_y * screen_height)), 0), screen_height - 1),
                )
        result[key] = KeyPoint(
            key=key,
            position=(
                min(max(int(round(norm_x * screen_width)), 0), screen_width - 1),
                min(max(int(round(norm_y * screen_height)), 0), screen_height - 1),
            ),
            normalized_position=(norm_x, norm_y),
            stage=str(raw_point.get("stage") or "录制记录"),
            scene=str(raw_point.get("scene") or record.directory.name),
            is_joystick_direction=bool(raw_point.get("is_joystick_direction")),
            joystick_center=joystick_center,
        )

    required_keys = {
        event.key for event in load_replay_events(record) if event.event != "drag"
    }
    missing = required_keys.difference(result)
    if missing:
        current_layout = load_key_layout(info_module, screen_width, screen_height)
        for key in missing:
            if key in current_layout:
                result[key] = current_layout[key]
    still_missing = sorted(required_keys.difference(result))
    if still_missing:
        raise ReplayError("当前布局缺少回放所需键位：" + "、".join(still_missing))
    return result
