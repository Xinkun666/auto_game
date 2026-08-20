"""从标注工具导出的 info.py 中读取键盘控点。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from aw.autogame.tools.Utils import select_scene_resolution


RESERVED_KEYS = frozenset({"q", "e"})
REQUIRED_MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})
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


def normalize_key_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if text.startswith("key:"):
        text = text[4:].strip()
    return KEY_ALIASES.get(text, text)


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
        for raw_name, point_data in points.items():
            # 旧工程没有 KEY_BINDINGS 时继续使用控点名作为键位。
            bound_name = scene_bindings.get(raw_name, raw_name)
            key = normalize_key_name(bound_name)
            if not key:
                raise LayoutError(f"控点“{raw_name}”还没有绑定键盘按键。")
            if key in RESERVED_KEYS:
                raise LayoutError(f"控点“{raw_name}”占用了录制键 {key}，请改用其他键名。")
            rect = point_data.get("rect") if isinstance(point_data, Mapping) else None
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise LayoutError(f"控点“{raw_name}”没有有效的 rect。")
            try:
                norm_x = (float(rect[0]) + float(rect[2])) / 2.0
                norm_y = (float(rect[1]) + float(rect[3])) / 2.0
            except (TypeError, ValueError) as exc:
                raise LayoutError(f"控点“{raw_name}”的 rect 不是数字。") from exc
            if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
                raise LayoutError(f"控点“{raw_name}”超出画面范围。")

            point = KeyPoint(
                key=key,
                position=(
                    min(max(int(round(norm_x * screen_width)), 0), screen_width - 1),
                    min(max(int(round(norm_y * screen_height)), 0), screen_height - 1),
                ),
                normalized_position=(norm_x, norm_y),
                stage=stage_name,
                scene=scene_name,
            )
            previous = result.get(key)
            if previous and previous.normalized_position != point.normalized_position:
                raise LayoutError(
                    f"键位“{key}”在当前阶段出现了多个不同控点："
                    f"{previous.scene} 和 {scene_name}。请只保留一个。"
                )
            result[key] = point

    if not result:
        raise LayoutError(
            f"阶段“{stage_name}”里还没有控点。请至少标注 w、a、s、d。"
        )
    missing = sorted(REQUIRED_MOVEMENT_KEYS.difference(result))
    if missing:
        raise LayoutError(
            "移动摇杆控点不完整，还缺少：" + "、".join(missing) + "。"
        )
    return result
