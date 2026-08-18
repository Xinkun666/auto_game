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

    w/a/s/d 被视为同一个摇杆的四个方向点；组合键会合成为一条
    对角方向，仍只占用一个手机触点。其他键位按一次会执行一次短按。
    """

    def __init__(self, stream_client, key_points: Dict[str, KeyPoint], tap_seconds: float = 0.05):
        self.stream_client = stream_client
        self.key_points = dict(key_points)
        self.tap_seconds = max(0.01, float(tap_seconds))
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
        if len(available) == 1 or self._joystick_center is None or self._joystick_radius <= 0:
            return self.key_points[available[-1]].position

        center_x, center_y = self._joystick_center
        vector_x = sum(self.key_points[key].position[0] - center_x for key in available)
        vector_y = sum(self.key_points[key].position[1] - center_y for key in available)
        length = math.hypot(vector_x, vector_y)
        if length <= 1e-6:
            return None
        return (
            int(round(center_x + vector_x / length * self._joystick_radius)),
            int(round(center_y + vector_y / length * self._joystick_radius)),
        )

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

    def _sync_movement(self) -> List[dict]:
        target = self._movement_target()
        if target is None:
            released = self._touch_up()
            return [released] if released else []
        if self._active_position is None:
            return [self._touch_down(target)]
        if self._active_position != target:
            return [self._touch_move(target)]
        return []

    def press(self, key: str) -> List[dict]:
        with self._lock:
            if key not in self.key_points:
                return []
            if key in MOVEMENT_KEYS:
                self.pressed_movement.add(key)
                return self._sync_movement()

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
