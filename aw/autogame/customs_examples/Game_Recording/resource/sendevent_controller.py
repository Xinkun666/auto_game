"""Game_Recording 的 HDC sendevent 触控适配器。"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from aw.autogame.tools.Utils import (
    convert_display_point_by_rotation,
    get_natural_resolution_by_rotation,
    infer_landscape_rotation,
    normalize_rotation,
)


EVENT_NAME_PATTERN = re.compile(r"event\d+")
GETEVENT_COMMANDS = (
    "getevent -lp",
    "getevent -p",
    "/data/test/getevent -p",
)


def _normalize_input_device(value: str) -> str:
    name = os.path.basename(str(value or "").strip())
    if not EVENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "sendevent 触摸设备必须是 eventX 或 /dev/input/eventX：%s" % value
        )
    return name


def discover_touch_panel(dut_handle, parser=None) -> Tuple[int, int, str, str]:
    """通过设备端 getevent 探测触摸面板范围。"""
    if parser is None:
        from aw.autogame.tools.GameFrameWorker import get_panel_abs_xy

        parser = get_panel_abs_xy

    failures = []
    for command in GETEVENT_COMMANDS:
        try:
            output = dut_handle.run_cmd_with_ret(command)
            abs_max_x, abs_max_y, input_device = parser(output)
            return (
                int(abs_max_x),
                int(abs_max_y),
                _normalize_input_device(input_device),
                command,
            )
        except Exception as exc:
            failures.append("%s: %s" % (command, exc))
    raise RuntimeError(
        "无法自动探测 sendevent 触摸设备。"
        "可使用 --sendevent-device/--sendevent-max-x/--sendevent-max-y 手动指定。"
        " attempts=%s" % " | ".join(failures)
    )


class SendeventTouchAdapter:
    """把 Game_Recording 的 touch_down/move/up 转换为 sendevent。"""

    def __init__(
        self,
        screen_size: Tuple[int, int],
        device_id: str = "",
        input_device: str = "",
        abs_max_x: Optional[int] = None,
        abs_max_y: Optional[int] = None,
        controller=None,
    ):
        screen_width, screen_height = (int(screen_size[0]), int(screen_size[1]))
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("sendevent 画面分辨率无效：%s" % (screen_size,))

        manual_mapping = bool(input_device or abs_max_x is not None or abs_max_y is not None)
        if manual_mapping:
            if not input_device or abs_max_x is None or abs_max_y is None:
                raise ValueError(
                    "手动配置 sendevent 时必须同时提供设备、max-x 和 max-y"
                )
            selected_device = _normalize_input_device(input_device)
            selected_max_x = int(abs_max_x)
            selected_max_y = int(abs_max_y)
            discovery_command = "manual"

        owns_controller = controller is None
        if owns_controller:
            from aw.autogame.tools.GameFrameWorker import SendEventController

            controller = SendEventController(
                device_id=str(device_id or ""),
                auto_prepare=False,
            )
        self._controller = controller
        self._closed = False
        self.screen_size = (screen_width, screen_height)

        if not manual_mapping:
            try:
                (
                    selected_max_x,
                    selected_max_y,
                    selected_device,
                    discovery_command,
                ) = discover_touch_panel(controller.dut_handle)
            except Exception:
                if owns_controller:
                    controller.close()
                self._closed = True
                raise

        if selected_max_x <= 0 or selected_max_y <= 0:
            if owns_controller:
                controller.close()
            self._closed = True
            raise ValueError(
                "sendevent 触摸面板范围无效：max_x=%s max_y=%s"
                % (selected_max_x, selected_max_y)
            )

        try:
            rotation = normalize_rotation(controller.dut_handle.get_screen_rotation())
        except Exception:
            rotation = None
        if rotation is None:
            rotation = infer_landscape_rotation(screen_width, screen_height)

        pixel_width, pixel_height = get_natural_resolution_by_rotation(
            screen_width,
            screen_height,
            rotation,
        )
        mt = controller.mt
        mt.abs_w0 = 0
        mt.abs_h0 = 0
        mt.abs_w = selected_max_x
        mt.abs_h = selected_max_y
        mt.input_device = selected_device
        mt.rotation = rotation
        mt.display_pixel_w = screen_width
        mt.display_pixel_h = screen_height
        mt.pixel_w = int(pixel_width)
        mt.pixel_h = int(pixel_height)
        mt.screen_mapping_ready = True

        self.input_device = selected_device
        self.abs_max_x = selected_max_x
        self.abs_max_y = selected_max_y
        self.rotation = int(rotation)
        self.discovery_command = discovery_command
        self.permission_check = "not_checked"
        if owns_controller:
            event_path = "/dev/input/%s" % self.input_device
            try:
                self.permission_check = controller.dut_handle.run_cmd_with_ret(
                    "if [ -w %s ]; then echo writable; else echo not_writable; fi"
                    % event_path
                ).strip()
            except Exception as exc:
                self.permission_check = "check_failed: %s" % exc
            if "not_writable" in self.permission_check:
                controller.close()
                self._closed = True
                raise RuntimeError(
                    "HDC Shell 没有写入 %s 的权限，sendevent 无法使用"
                    % event_path
                )

    def _transform(self, x: int, y: int) -> Tuple[int, int]:
        transformed_x, transformed_y = convert_display_point_by_rotation(
            int(x),
            int(y),
            self.screen_size[0],
            self.screen_size[1],
            self.rotation,
        )
        mt = self._controller.mt
        return (
            min(max(int(transformed_x), 0), max(int(mt.pixel_w) - 1, 0)),
            min(max(int(transformed_y), 0), max(int(mt.pixel_h) - 1, 0)),
        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("sendevent 触控后端已关闭")

    def touch_down(self, x: int, y: int):
        self._ensure_open()
        self._controller.move_press(0, self._transform(x, y))

    def touch_move(self, x: int, y: int):
        self._ensure_open()
        self._controller.move_to(0, self._transform(x, y), duration_ms=0)

    def touch_up(self, x: int, y: int):
        self._ensure_open()
        self._controller.move_up(0)

    def diagnostic_snapshot(self):
        return {
            "backend": "sendevent",
            "input_device": "/dev/input/%s" % self.input_device,
            "abs_max_x": self.abs_max_x,
            "abs_max_y": self.abs_max_y,
            "rotation": self.rotation,
            "screen_size": list(self.screen_size),
            "discovery_command": self.discovery_command,
            "permission_check": self.permission_check,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._controller.close()
