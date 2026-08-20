"""把电脑键盘事件转换为华为 HOScrcpy 单指触控。"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .layout import KeyPoint


MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})


class SingleTouchKeyboardController:
    """适配 HOS 官方单指接口的键盘控制器。

    w/a/s/d 被视为同一个摇杆的四个方向点。每次启动方向时，
    先在摇杆中心落指，再滑动到目标方向。切换方向会先抬起旧触点，
    新方向键会替换旧方向键。其他键位按一次会执行一次短按。
    """

    def __init__(
        self,
        stream_client,
        key_points: Dict[str, KeyPoint],
        tap_seconds: float = 0.05,
        movement_transition_seconds: float = 0.01,
    ):
        self.stream_client = stream_client
        self.key_points = dict(key_points)
        self.tap_seconds = max(0.01, float(tap_seconds))
        self.movement_transition_seconds = max(
            0.0,
            float(movement_transition_seconds),
        )
        self.pressed_movement: Set[str] = set()
        self._active_position: Optional[Tuple[int, int]] = None
        self._lock = threading.RLock()
        self._joystick_center, self._joystick_radius = self._infer_joystick_geometry()

    def _infer_joystick_geometry(self):
        points = [self.key_points[key].position for key in MOVEMENT_KEYS if key in self.key_points]
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
        target = self._movement_target()
        if target is None:
            released = self._touch_up()
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
        return actions

    def press(self, key: str) -> List[dict]:
        with self._lock:
            if key not in self.key_points:
                return []
            if key in MOVEMENT_KEYS:
                if self.pressed_movement == {key}:
                    return self._sync_movement()
                force_restart = bool(self.pressed_movement)
                self.pressed_movement.clear()
                self.pressed_movement.add(key)
                return self._sync_movement(force_restart=force_restart)

            actions: List[dict] = []
            resume_movement = bool(self.pressed_movement)
            released = self._touch_up()
            if released:
                actions.append(released)

            position = self.key_points[key].position
            actions.append(self._touch_down(position))
            time.sleep(self.tap_seconds)
            released = self._touch_up()
            if released:
                actions.append(released)
            if resume_movement:
                actions.extend(self._sync_movement())
            return actions

    def release(self, key: str) -> List[dict]:
        with self._lock:
            if key not in MOVEMENT_KEYS:
                return []
            self.pressed_movement.discard(key)
            return self._sync_movement()

    def release_all(self) -> List[dict]:
        with self._lock:
            self.pressed_movement.clear()
            released = self._touch_up()
            return [released] if released else []

    @property
    def active_movement_keys(self) -> Iterable[str]:
        return tuple(sorted(self.pressed_movement))
