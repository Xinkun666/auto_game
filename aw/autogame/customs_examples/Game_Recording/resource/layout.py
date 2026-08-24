"""从标注工具导出的 info.py 中读取键盘控点。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from aw.autogame.tools.Utils import select_scene_resolution


# 录制已由界面按钮控制，q/e 不再是保留键。
RESERVED_KEYS = frozenset()
REQUIRED_MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})
JOYSTICK_CENTER_POINT_NAME = "center"
JOYSTICK_BOUNDARY_POINT_NAME = "boundary"
JOYSTICK_DIRECTION_SPECS = (
    ("__joystick_up__", "↑ 上（摇杆）", (0.0, -1.0)),
    ("__joystick_down__", "↓ 下（摇杆）", (0.0, 1.0)),
    ("__joystick_left__", "← 左（摇杆）", (-1.0, 0.0)),
    ("__joystick_right__", "→ 右（摇杆）", (1.0, 0.0)),
    ("__joystick_up_left__", "↖ 左上（摇杆）", (-math.sqrt(0.5), -math.sqrt(0.5))),
    ("__joystick_up_right__", "↗ 右上（摇杆）", (math.sqrt(0.5), -math.sqrt(0.5))),
    ("__joystick_down_left__", "↙ 左下（摇杆）", (-math.sqrt(0.5), math.sqrt(0.5))),
    ("__joystick_down_right__", "↘ 右下（摇杆）", (math.sqrt(0.5), math.sqrt(0.5))),
)
KEY_ALIASES = {
    "空格": "space",
    "空格键": "space",
    "spacebar": "space",
    "上": "up",
    "下": "down",
    "左": "left",
    "右": "right",
    "上箭头": "up",
    "下箭头": "down",
    "左箭头": "left",
    "右箭头": "right",
    "回车": "enter",
    "回车键": "enter",
    "换挡": "shift",
    "控制": "ctrl",
}


class LayoutError(ValueError):
    """info.py 中的键位布局无法用于录制。"""


@dataclass(frozen=True)
class KeyPoint:
    key: str
    position: Tuple[int, int]
    normalized_position: Tuple[float, float]
    stage: str
    scene: str
    is_joystick_direction: bool = False
    joystick_center: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class JoystickDirectionPoint:
    binding_name: str
    display_name: str
    normalized_position: Tuple[float, float]
    center_position: Tuple[int, int]


def normalize_key_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if text.startswith("key:"):
        text = text[4:].strip()
    return KEY_ALIASES.get(text, text)


def normalize_control_point_name(name: Any) -> str:
    return str(name or "").strip().casefold()


def is_joystick_geometry_point(name: Any) -> bool:
    return normalize_control_point_name(name) in {
        JOYSTICK_CENTER_POINT_NAME,
        JOYSTICK_BOUNDARY_POINT_NAME,
    }


def joystick_marker_names(points: Mapping[str, Any]) -> Dict[str, str]:
    markers: Dict[str, str] = {}
    for raw_name in points:
        name = normalize_control_point_name(raw_name)
        if name in {JOYSTICK_CENTER_POINT_NAME, JOYSTICK_BOUNDARY_POINT_NAME}:
            markers.setdefault(name, str(raw_name))
    return markers


def _normalized_point_center(point_data: Any, point_name: str) -> Tuple[float, float]:
    rect = point_data.get("rect") if isinstance(point_data, Mapping) else None
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        raise ValueError(f"控点“{point_name}”没有有效的 rect。")
    try:
        norm_x = (float(rect[0]) + float(rect[2])) / 2.0
        norm_y = (float(rect[1]) + float(rect[3])) / 2.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"控点“{point_name}”的 rect 不是数字。") from exc
    if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
        raise ValueError(f"控点“{point_name}”超出画面范围。")
    return norm_x, norm_y


def joystick_direction_points(
    points: Mapping[str, Any],
    screen_width: int,
    screen_height: int,
) -> Tuple[JoystickDirectionPoint, ...]:
    """由 center 与 boundary 控点生成八个摇杆边界方向。"""
    markers = joystick_marker_names(points)
    has_center = JOYSTICK_CENTER_POINT_NAME in markers
    has_boundary = JOYSTICK_BOUNDARY_POINT_NAME in markers
    if not has_center and not has_boundary:
        return ()
    if not (has_center and has_boundary):
        missing = JOYSTICK_BOUNDARY_POINT_NAME if has_center else JOYSTICK_CENTER_POINT_NAME
        raise ValueError(
            "摇杆标注必须同时包含 center 和 boundary；缺少：%s。" % missing
        )

    center_norm = _normalized_point_center(
        points[markers[JOYSTICK_CENTER_POINT_NAME]],
        markers[JOYSTICK_CENTER_POINT_NAME],
    )
    boundary_norm = _normalized_point_center(
        points[markers[JOYSTICK_BOUNDARY_POINT_NAME]],
        markers[JOYSTICK_BOUNDARY_POINT_NAME],
    )
    center_x = min(max(int(round(center_norm[0] * screen_width)), 0), screen_width - 1)
    center_y = min(max(int(round(center_norm[1] * screen_height)), 0), screen_height - 1)
    boundary_x = min(max(int(round(boundary_norm[0] * screen_width)), 0), screen_width - 1)
    boundary_y = min(max(int(round(boundary_norm[1] * screen_height)), 0), screen_height - 1)
    radius = math.hypot(boundary_x - center_x, boundary_y - center_y)
    if radius <= 0:
        raise ValueError("center 与 boundary 不能重合；boundary 必须标在摇杆边界上。")

    result = []
    for binding_name, display_name, (vector_x, vector_y) in JOYSTICK_DIRECTION_SPECS:
        target_x = min(max(int(round(center_x + radius * vector_x)), 0), screen_width - 1)
        target_y = min(max(int(round(center_y + radius * vector_y)), 0), screen_height - 1)
        result.append(
            JoystickDirectionPoint(
                binding_name=binding_name,
                display_name=display_name,
                normalized_position=(target_x / screen_width, target_y / screen_height),
                center_position=(center_x, center_y),
            )
        )
    return tuple(result)


def _active_stage_name(stage_dict: Mapping[str, Any], stage_info: Mapping[str, Any]) -> str:
    for name, active in stage_dict.items():
        if active and name in stage_info:
            return str(name)
    if stage_info:
        return str(next(iter(stage_info)))
    raise LayoutError(
        "info.py 里还没有场景。请先用标注工具新建阶段和场景，"
        "再把 w、a、s、d 等键位标成控点并导出。"
    )


def _iter_selected_scenes(
    stage_name: str,
    stage_data: Mapping[str, Any],
    screen_width: int,
    screen_height: int,
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    scenes = stage_data.get("scenes", {})
    if not isinstance(scenes, Mapping) or not scenes:
        raise LayoutError(f"阶段“{stage_name}”里还没有场景。")
    for scene_name, scene_data in scenes.items():
        selected = select_scene_resolution(scene_data, screen_width, screen_height)
        if isinstance(selected, Mapping):
            yield str(scene_name), selected


def load_key_layout(
    info_module: Any,
    screen_width: int,
    screen_height: int,
) -> Dict[str, KeyPoint]:
    """读取当前活动阶段的全部控点，并换算为手机显示坐标。"""
    if screen_width <= 0 or screen_height <= 0:
        raise LayoutError(f"手机分辨率无效：{screen_width}x{screen_height}")

    stage_info = getattr(info_module, "STAGE_INFO", {})
    stage_dict = getattr(info_module, "STAGE_DICT", {})
    if not isinstance(stage_info, Mapping):
        raise LayoutError("info.py 的 STAGE_INFO 格式不正确。")
    if not isinstance(stage_dict, Mapping):
        stage_dict = {}

    stage_name = _active_stage_name(stage_dict, stage_info)
    stage_data = stage_info.get(stage_name, {})
    if not isinstance(stage_data, Mapping):
        raise LayoutError(f"阶段“{stage_name}”的数据格式不正确。")

    all_bindings = getattr(info_module, "KEY_BINDINGS", {})
    stage_bindings = (
        all_bindings.get(stage_name, {}) if isinstance(all_bindings, Mapping) else {}
    )
    if not isinstance(stage_bindings, Mapping):
        stage_bindings = {}

    result: Dict[str, KeyPoint] = {}
    uses_generated_joystick = False

    def add_key_point(
        key: str,
        norm_x: float,
        norm_y: float,
        scene_name: str,
        is_joystick_direction: bool = False,
        joystick_center: Optional[Tuple[int, int]] = None,
    ):
        point = KeyPoint(
            key=key,
            position=(
                min(max(int(round(norm_x * screen_width)), 0), screen_width - 1),
                min(max(int(round(norm_y * screen_height)), 0), screen_height - 1),
            ),
            normalized_position=(norm_x, norm_y),
            stage=stage_name,
            scene=scene_name,
            is_joystick_direction=is_joystick_direction,
            joystick_center=joystick_center,
        )
        previous = result.get(key)
        if previous and previous.normalized_position != point.normalized_position:
            raise LayoutError(
                f"键位“{key}”在当前阶段出现了多个不同控点："
                f"{previous.scene} 和 {scene_name}。请只保留一个。"
            )
        result[key] = point

    for scene_name, scene_data in _iter_selected_scenes(
        stage_name,
        stage_data,
        screen_width,
        screen_height,
    ):
        points = scene_data.get("points", {})
        if not isinstance(points, Mapping):
            continue
        scene_bindings = stage_bindings.get(scene_name, {})
        if not isinstance(scene_bindings, Mapping):
            scene_bindings = {}
        try:
            direction_points = joystick_direction_points(
                points,
                screen_width,
                screen_height,
            )
        except ValueError as exc:
            raise LayoutError(f"场景“{scene_name}”的{exc}") from exc
        uses_generated_joystick = uses_generated_joystick or bool(direction_points)
        for raw_name, point_data in points.items():
            if is_joystick_geometry_point(raw_name):
                continue
            # 控点名不等于键盘绑定；缺少明确记录时必须先绑定。
            bound_name = scene_bindings.get(raw_name, "")
            key = normalize_key_name(bound_name)
            if not key:
                raise LayoutError(f"控点“{raw_name}”还没有绑定键盘按键。")
            if key in RESERVED_KEYS:
                raise LayoutError(f"控点“{raw_name}”占用了录制键 {key}，请改用其他键名。")
            try:
                norm_x, norm_y = _normalized_point_center(point_data, str(raw_name))
            except ValueError as exc:
                raise LayoutError(str(exc)) from exc
            add_key_point(key, norm_x, norm_y, scene_name)

        for direction in direction_points:
            key = normalize_key_name(scene_bindings.get(direction.binding_name, ""))
            if not key:
                continue
            if key in RESERVED_KEYS:
                raise LayoutError(
                    f"摇杆方向“{direction.display_name}”占用了录制键 {key}，请改用其他键名。"
                )
            add_key_point(
                key,
                *direction.normalized_position,
                scene_name,
                is_joystick_direction=True,
                joystick_center=direction.center_position,
            )

    if not result and not uses_generated_joystick:
        raise LayoutError(
            f"阶段“{stage_name}”里还没有控点。请至少标注 w、a、s、d。"
        )
    missing = sorted(REQUIRED_MOVEMENT_KEYS.difference(result))
    if missing and not uses_generated_joystick:
        raise LayoutError(
            "移动摇杆控点不完整，还缺少：" + "、".join(missing) + "。"
        )
    return result
