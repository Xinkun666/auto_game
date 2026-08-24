"""把电脑键盘事件转换为华为 HOScrcpy 单指触控。"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .layout import KeyPoint


MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})
DRAG_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class SingleTouchKeyboardController:
    """适配 HOS 官方单指接口的键盘控制器。

    w/a/s/d 被视为同一个摇杆的四个方向点。每次启动方向时，
    先在摇杆中心落指，再滑动到目标方向。切换方向会先抬起旧触点。

    所有绑定键按住期间，每次新按下一次键盘方向键，
    都可以将当前触点离散移动一次。
    """

    def __init__(
        self,
        stream_client,
        key_points: Dict[str, KeyPoint],
        tap_seconds: float = 0.05,
        movement_transition_seconds: float = 0.01,
        screen_size: Optional[Tuple[int, int]] = None,
        drag_step_ratio: float = 0.08,
        drag_duration_seconds: float = 0.12,
        drag_move_steps: int = 6,
    ):
        self.stream_client = stream_client
        self.key_points = dict(key_points)
        self.tap_seconds = max(0.01, float(tap_seconds))
        self.movement_transition_seconds = max(
            0.0,
            float(movement_transition_seconds),
        )
        self._movement_keys = frozenset(
            key
            for key, point in self.key_points.items()
            if key in MOVEMENT_KEYS or point.is_joystick_direction
        )
        self.pressed_movement: Set[str] = set()
        self._active_button_key: Optional[str] = None
        self._active_position: Optional[Tuple[int, int]] = None
        self._active_anchor_position: Optional[Tuple[int, int]] = None
        self._lock = threading.RLock()
        self._joystick_center, self._joystick_radius = self._infer_joystick_geometry()
        self.screen_size = self._resolve_screen_size(screen_size)
        self.drag_step_ratio = min(max(float(drag_step_ratio), 0.005), 0.5)
        self.drag_duration_seconds = max(0.0, float(drag_duration_seconds))
        self.drag_move_steps = max(2, int(drag_move_steps))

    def _resolve_screen_size(self, screen_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if screen_size and int(screen_size[0]) > 0 and int(screen_size[1]) > 0:
            return int(screen_size[0]), int(screen_size[1])
        max_x = max((point.position[0] for point in self.key_points.values()), default=0)
        max_y = max((point.position[1] for point in self.key_points.values()), default=0)
        return max(max_x + 1, 1), max(max_y + 1, 1)

    def _infer_joystick_geometry(self):
        generated_points = [
            point
            for point in self.key_points.values()
            if point.is_joystick_direction and point.joystick_center is not None
        ]
        if generated_points:
            center = generated_points[0].joystick_center
            if center is not None:
                radii = [
                    math.hypot(point.position[0] - center[0], point.position[1] - center[1])
                    for point in generated_points
                ]
                return center, (sum(radii) / len(radii) if radii else 0.0)
        points = [
            self.key_points[key].position
            for key in self._movement_keys
            if key in self.key_points
        ]
        if not points:
            return None, 0.0
        center = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        radii = [math.hypot(point[0] - center[0], point[1] - center[1]) for point in points]
        radius = sum(radii) / len(radii) if radii else 0.0
        return center, radius

    def _movement_target(self) -> Optional[Tuple[int, int]]:
        available = [key for key in self.pressed_movement if key in self.key_points]
        if not available:
            return None
        return self.key_points[available[-1]].position

    def _movement_origin(self, target: Tuple[int, int]) -> Tuple[int, int]:
        if self._joystick_center is not None:
            center = tuple(int(round(value)) for value in self._joystick_center)
            if center != target:
                return center
        fallback_offset = max(1, int(round(self._joystick_radius or 30.0)))
        return target[0], target[1] + fallback_offset

    def _touch_down(self, position: Tuple[int, int]) -> dict:
        self.stream_client.touch_down(*position)
        self._active_position = position
        return {"method": "touch_down", "position": list(position)}

    def _touch_move(self, position: Tuple[int, int]) -> dict:
        self.stream_client.touch_move(*position)
        self._active_position = position
        return {"method": "touch_move", "position": list(position)}

    def _touch_up(self) -> Optional[dict]:
        if self._active_position is None:
            return None
        position = self._active_position
        self.stream_client.touch_up(*position)
        self._active_position = None
        return {"method": "touch_up", "position": list(position)}

    def _sync_movement(self, force_restart: bool = False) -> List[dict]:
        if self._active_button_key is not None:
            return []
        target = self._movement_target()
        if target is None:
            released = self._touch_up()
            self._active_anchor_position = None
            return [released] if released else []
        if self._active_position is None:
            return self._start_movement(target)
        if force_restart or self._active_position != target:
            actions: List[dict] = []
            released = self._touch_up()
            if released:
                actions.append(released)
                if self.movement_transition_seconds:
                    time.sleep(self.movement_transition_seconds)
            actions.extend(self._start_movement(target))
            return actions
        return []

    def _start_movement(self, target: Tuple[int, int]) -> List[dict]:
        origin = self._movement_origin(target)
        actions = [self._touch_down(origin)]
        if self.movement_transition_seconds:
            time.sleep(self.movement_transition_seconds)
        actions.append(self._touch_move(target))
        self._active_anchor_position = target
        return actions

    def press(self, key: str) -> List[dict]:
        with self._lock:
            if key not in self.key_points:
                return []
            if self.is_movement_key(key):
                if self.pressed_movement == {key}:
                    return self._sync_movement()
                force_restart = bool(self.pressed_movement)
                self.pressed_movement.clear()
                self.pressed_movement.add(key)
                return self._sync_movement(force_restart=force_restart)

            actions: List[dict] = []
            released = self._touch_up()
            if released:
                actions.append(released)
            self._active_button_key = key
            position = self.key_points[key].position
            actions.append(self._touch_down(position))
            self._active_anchor_position = position
            return actions

    def release(self, key: str) -> List[dict]:
        with self._lock:
            if not self.is_movement_key(key):
                if key != self._active_button_key:
                    return []
                self._active_button_key = None
                actions = []
                released = self._touch_up()
                self._active_anchor_position = None
                if released:
                    actions.append(released)
                actions.extend(self._sync_movement())
                return actions
            self.pressed_movement.discard(key)
            if self._active_button_key is not None:
                return []
            return self._sync_movement()

    def tap(self, key: str) -> List[dict]:
        """兼容旧录制：非摇杆键仍执行固定时长的短按。"""
        if self.is_movement_key(key):
            return []
        actions = self.press(key)
        if actions:
            time.sleep(self.tap_seconds)
        actions.extend(self.release(key))
        return actions

    def nudge_active_control(self, direction: str) -> List[dict]:
        """将当前按住的任意控点向指定方向离散滑动一次。"""
        with self._lock:
            vector = DRAG_DIRECTIONS.get(str(direction or "").lower())
            anchor = self._active_anchor_position
            if vector is None or self.active_control_key is None or anchor is None:
                return []
            width, height = self.screen_size
            distance = max(1, int(round(min(width, height) * self.drag_step_ratio)))
            target = (
                min(max(anchor[0] + vector[0] * distance, 0), width - 1),
                min(max(anchor[1] + vector[1] * distance, 0), height - 1),
            )
            if target == anchor:
                return []
            return self._move_from_anchor(target)

    def move_active_control_to_normalized(self, norm_x: float, norm_y: float) -> List[dict]:
        """回放时按录制的归一化坐标恢复一次离散滑动。"""
        with self._lock:
            if self.active_control_key is None or self._active_anchor_position is None:
                return []
            width, height = self.screen_size
            target = (
                min(max(int(round(float(norm_x) * width)), 0), width - 1),
                min(max(int(round(float(norm_y) * height)), 0), height - 1),
            )
            if target == self._active_anchor_position:
                return []
            return self._move_from_anchor(target)

    def _move_from_anchor(self, target: Tuple[int, int]) -> List[dict]:
        anchor = self._active_anchor_position
        if anchor is None:
            return []
        actions: List[dict] = []
        if self._active_position != anchor:
            released = self._touch_up()
            if released:
                actions.append(released)
            if self.movement_transition_seconds:
                time.sleep(self.movement_transition_seconds)
            actions.append(self._touch_down(anchor))
            if self.movement_transition_seconds:
                time.sleep(self.movement_transition_seconds)

        per_step_seconds = self.drag_duration_seconds / self.drag_move_steps
        for step in range(1, self.drag_move_steps + 1):
            if per_step_seconds:
                time.sleep(per_step_seconds)
            ratio = step / self.drag_move_steps
            position = (
                int(round(anchor[0] + (target[0] - anchor[0]) * ratio)),
                int(round(anchor[1] + (target[1] - anchor[1]) * ratio)),
            )
            actions.append(self._touch_move(position))
        return actions

    def release_all(self) -> List[dict]:
        with self._lock:
            self.pressed_movement.clear()
            self._active_button_key = None
            self._active_anchor_position = None
            released = self._touch_up()
            return [released] if released else []

    @property
    def active_movement_keys(self) -> Iterable[str]:
        return tuple(sorted(self.pressed_movement))

    @property
    def movement_keys(self) -> frozenset[str]:
        return self._movement_keys

    def is_movement_key(self, key: str) -> bool:
        return str(key or "") in self._movement_keys

    @property
    def active_button_key(self) -> Optional[str]:
        return self._active_button_key

    @property
    def active_control_key(self) -> Optional[str]:
        if self._active_button_key is not None:
            return self._active_button_key
        if self._active_position is not None and self.pressed_movement:
            return sorted(self.pressed_movement)[-1]
        return None
