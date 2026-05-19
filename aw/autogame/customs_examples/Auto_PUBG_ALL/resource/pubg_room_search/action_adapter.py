from __future__ import annotations

import math
import threading
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

from aw.autogame.tools.Utils import get_resolution


class AutoGameActionProxy:
    """Translate pubg_test action names into auto_game hdc/sendevent controls."""

    def __init__(self, worker, config: Dict[str, Any]):
        self.worker = worker
        self.config = config
        self.move_finger_id = int(config.get("move_finger_id", 0))
        self.view_finger_id = int(config.get("view_finger_id", 1))
        self.button_finger_start = int(config.get("button_finger_start", 2))
        self.move_radius_px = int(config.get("move_radius_px", 300))
        self.button_mapping = dict(config.get("button_mapping") or {})
        self._move_angle = -90.0
        self._view_thread: Optional[threading.Thread] = None
        self._view_stop_event = threading.Event()
        self._view_lock = threading.RLock()
        self._action_listeners = []
        self._button_fingers = {
            "jump": self.button_finger_start,
            "door": self.button_finger_start + 1,
            "pick_btn": self.button_finger_start + 2,
            "map": self.button_finger_start + 3,
            "attack": self.button_finger_start + 4,
        }

    def add_action_listener(self, listener):
        if listener not in self._action_listeners:
            self._action_listeners.append(listener)

    def remove_action_listener(self, listener):
        try:
            self._action_listeners.remove(listener)
        except ValueError:
            pass

    def send_actions(self, action_seq, frame_id=None, listener=None):
        action_id = str(uuid.uuid4())
        last_method = ""
        try:
            for action in action_seq or []:
                method = str(action.get("method") or "")
                args = action.get("args") or {}
                last_method = method
                self._notify_action(method, args, action_id)
                self.dispatch(method, args)
            self._notify_result(listener, last_method, action_id)
            return True
        except Exception as exc:
            print(f"[PubgRoomSearch] 执行动作失败: method={last_method}, error={exc}")
            self._notify_result(listener, last_method, action_id)
            return False

    def release_all(self):
        for finger_id in range(0, self.button_finger_start + 8):
            try:
                self.worker.move_up(finger_id)
            except Exception:
                pass
        self._view_stop_event.set()
        self._join_view_thread(timeout=0.5)

    def dispatch(self, method: str, args: Dict[str, Any]):
        if method == "release_all_controllers":
            self.release_all()
            return

        if method == "move_press":
            self._move_press(float(args.get("init_angle", -90)))
            return
        if method == "move_slide_plus":
            self._move_slide_plus(float(args.get("angle_step", 0)))
            return
        if method == "move_set_angle":
            self._move_set_angle(float(args.get("angle", self._move_angle)))
            return
        if method == "move_release":
            self.worker.move_up(self.move_finger_id)
            return

        if method == "view_slide":
            self._view_slide(
                float(args.get("delta_x", 0.0)),
                float(args.get("delta_y", 0.0)),
                args.get("target_interval"),
            )
            return
        if method == "view_keep_slide":
            self._view_keep_slide(
                float(args.get("x_speed", 0.0)),
                float(args.get("y_speed", 0.0)),
                float(args.get("slide_length", 0.0545)),
                float(args.get("target_interval", 0.3)),
            )
            return
        if method == "view_release":
            self._stop_view_keep_slide()
            return

        if method.endswith("_touch"):
            self._button_touch(method[: -len("_touch")])
            return
        if method.endswith("_release"):
            self._button_release(method[: -len("_release")])
            return

        self._button_click(method)

    def _move_press(self, angle: float):
        self._move_angle = angle
        self.worker.move_press(self.move_finger_id, self._stick_pos(angle))

    def _move_slide_plus(self, angle_step: float):
        self._move_angle = (self._move_angle + angle_step) % 360
        self.worker.move_to(
            self.move_finger_id,
            self._stick_pos(self._move_angle),
            duration_ms=160,
        )

    def _move_set_angle(self, angle: float):
        self._move_angle = angle
        self.worker.move_to(
            self.move_finger_id,
            self._stick_pos(self._move_angle),
            duration_ms=160,
        )

    def _stick_pos(self, angle: float) -> Tuple[int, int]:
        cx, cy = self._resolve_named_or_norm("摇杆", self.config.get("move_default_center"))
        rad = math.radians(angle)
        return (
            int(round(cx + self.move_radius_px * math.cos(rad))),
            int(round(cy + self.move_radius_px * math.sin(rad))),
        )

    def _view_slide(self, delta_x: float, delta_y: float, target_interval=None):
        screen_w, screen_h = self._screen_size()
        x_bias = int(round(delta_x * screen_w))
        y_bias = int(round(delta_y * screen_h))
        duration_ms = self._interval_to_ms(target_interval, default_ms=300)
        self.worker.tap_single(
            "视角",
            x_bias=x_bias,
            y_bias=y_bias,
            dura=duration_ms,
            wait=0,
            finger_id=self.view_finger_id,
        )

    def _view_keep_slide(
        self,
        x_speed: float,
        y_speed: float,
        slide_length: float,
        target_interval: float,
    ):
        if abs(x_speed) < 1e-6 and abs(y_speed) < 1e-6:
            return
        self._stop_view_keep_slide()
        self._view_stop_event.clear()
        self._run_view_slide_once(x_speed, y_speed, slide_length, target_interval)

        def _loop():
            while not self._view_stop_event.is_set():
                self._run_view_slide_once(x_speed, y_speed, slide_length, target_interval)

        self._view_thread = threading.Thread(
            target=_loop,
            name="AutoGamePubgRoomSearchViewKeepSlide",
            daemon=True,
        )
        self._view_thread.start()

    def _run_view_slide_once(
        self,
        x_speed: float,
        y_speed: float,
        slide_length: float,
        target_interval: float,
    ):
        screen_w, screen_h = self._screen_size()
        x_bias = int(round(slide_length * x_speed * screen_w))
        y_bias = int(round(slide_length * y_speed * screen_h))
        duration_ms = self._interval_to_ms(target_interval, default_ms=300)
        with self._view_lock:
            if self._view_stop_event.is_set():
                return
            self.worker.tap_single(
                "视角",
                x_bias=x_bias,
                y_bias=y_bias,
                dura=duration_ms,
                wait=0,
                finger_id=self.view_finger_id,
            )

    def _stop_view_keep_slide(self):
        self._view_stop_event.set()
        self._join_view_thread(timeout=1.0)
        try:
            self.worker.move_up(self.view_finger_id)
        except Exception:
            pass

    def _join_view_thread(self, timeout: float):
        thread = self._view_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._view_thread = None

    def _button_touch(self, logical_name: str):
        button_name = self._button_name(logical_name)
        if button_name is None:
            return
        finger_id = self._button_fingers.get(logical_name, self.button_finger_start)
        self.worker.click_down(button_name, finger_id=finger_id)

    def _button_release(self, logical_name: str):
        finger_id = self._button_fingers.get(logical_name, self.button_finger_start)
        self.worker.move_up(finger_id)

    def _button_click(self, logical_name: str):
        button_name = self._button_name(logical_name)
        if button_name is None:
            return
        self.worker.click(button_name)

    def _button_name(self, logical_name: str) -> Optional[str]:
        mapped = self.button_mapping.get(logical_name, logical_name)
        candidates = [mapped]
        if logical_name == "pick_btn":
            candidates.extend(["拾取", "拾取按钮", "开门"])
        if logical_name == "door":
            candidates.extend(["开门", "关门"])
        for name in candidates:
            if self._can_resolve_name(name):
                return name
        print(f"[PubgRoomSearch] 未找到控点映射: {logical_name} -> {mapped}")
        return None

    def _can_resolve_name(self, name: str) -> bool:
        try:
            pos, _ = self.worker.controller._resolve_pos(name)
            return bool(pos)
        except Exception:
            return False

    def _resolve_named_or_norm(self, name: str, norm_pos) -> Tuple[int, int]:
        try:
            pos, _ = self.worker.controller._resolve_pos(name)
            if pos:
                return int(pos[0]), int(pos[1])
        except Exception:
            pass

        screen_w, screen_h = self._screen_size()
        if isinstance(norm_pos, Iterable):
            values = list(norm_pos)
            if len(values) == 2:
                return int(float(values[0]) * screen_w), int(float(values[1]) * screen_h)
        return int(screen_w / 2), int(screen_h / 2)

    def _screen_size(self) -> Tuple[int, int]:
        try:
            cached = self.worker.controller._get_cached_resolution()
            if cached and cached[0] and cached[1]:
                return int(cached[0]), int(cached[1])
        except Exception:
            pass
        width, height = get_resolution()
        if width and height:
            return int(width), int(height)
        frame = getattr(self.worker, "frame", None)
        if frame is not None:
            h, w = frame.shape[:2]
            return int(w), int(h)
        return 2832, 1316

    @staticmethod
    def _interval_to_ms(value, default_ms: int) -> int:
        try:
            if value is None:
                return default_ms
            return max(1, int(round(float(value) * 1000)))
        except (TypeError, ValueError):
            return default_ms

    def _notify_action(self, method: str, args: Dict[str, Any], action_id: str):
        action = _CompatAction(action_id, method, args)
        for listener in list(self._action_listeners):
            try:
                listener.on_action(action)
            except Exception:
                pass

    @staticmethod
    def _notify_result(listener, method: str, action_id: str):
        if listener is None:
            return
        try:
            listener.on_result(None, method, action_id)
        except TypeError:
            try:
                listener.on_result(None)
            except Exception:
                pass
        except Exception:
            pass


class _CompatAction:
    def __init__(self, action_id: str, action_name: str, action_args: Dict[str, Any]):
        self.action_id = action_id
        self.action_name = action_name
        self.action_args = action_args
