"""Game_Recording 按键绑定的读取、编辑和 info.py 回写。"""

from __future__ import annotations

import ast
import copy
import os
import pprint
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

from .layout import (
    RESERVED_KEYS,
    is_joystick_geometry_point,
    joystick_direction_points,
    normalize_key_name,
)


BINDINGS_NAME = "KEY_BINDINGS"


class BindingConfigError(ValueError):
    """按键绑定配置无法读取或保存。"""


def active_stage_name(stage_dict: Mapping[str, Any], stage_info: Mapping[str, Any]) -> str:
    for name, active in stage_dict.items():
        if active and name in stage_info:
            return str(name)
    if stage_info:
        return str(next(iter(stage_info)))
    raise BindingConfigError(
        "info.py 里还没有阶段和场景，请先用标注工具导出控点。"
    )


def _iter_scene_versions(scene_data: Any) -> Iterable[tuple[str, MutableMapping[str, Any]]]:
    if not isinstance(scene_data, MutableMapping):
        return
    resolutions = scene_data.get("resolutions")
    if isinstance(resolutions, MutableMapping) and resolutions:
        for resolution_key, version in resolutions.items():
            if isinstance(version, MutableMapping):
                yield str(resolution_key), version
        return
    yield "", scene_data


@dataclass
class BindingScene:
    stage: str
    name: str
    resolution_key: str
    data: MutableMapping[str, Any]

    @property
    def width(self) -> int:
        return max(int(self.data.get("width") or 0), 1)

    @property
    def height(self) -> int:
        return max(int(self.data.get("height") or 0), 1)

    @property
    def image(self) -> str:
        return str(self.data.get("image") or "")

    @property
    def points(self) -> MutableMapping[str, Any]:
        points = self.data.setdefault("points", {})
        return points if isinstance(points, MutableMapping) else {}

    @property
    def display_name(self) -> str:
        suffix = f"  {self.width}x{self.height}" if self.resolution_key else ""
        return f"{self.name}{suffix}"


@dataclass(frozen=True)
class BindingControl:
    name: str
    display_name: str
    normalized_position: tuple[float, float]
    is_virtual_joystick_direction: bool = False


class BindingConfiguration:
    """info.py 的可取消编辑副本。"""

    def __init__(self, info_module: Any):
        self.info_module = info_module
        self.info_path = Path(getattr(info_module, "__file__", "")).resolve()
        if not self.info_path.is_file():
            raise BindingConfigError(f"找不到可回写的 info.py：{self.info_path}")

        stage_info = getattr(info_module, "STAGE_INFO", {})
        stage_dict = getattr(info_module, "STAGE_DICT", {})
        scene_pool = getattr(info_module, "SCENE_POOL", {})
        bindings = getattr(info_module, BINDINGS_NAME, {})
        if not isinstance(stage_info, Mapping):
            raise BindingConfigError("info.py 的 STAGE_INFO 格式不正确。")

        self.stage_dict = copy.deepcopy(dict(stage_dict)) if isinstance(stage_dict, Mapping) else {}
        self.stage_info: Dict[str, Any] = copy.deepcopy(dict(stage_info))
        self.scene_pool: Dict[str, Any] = (
            copy.deepcopy(dict(scene_pool)) if isinstance(scene_pool, Mapping) else {}
        )
        self.bindings: Dict[str, Any] = (
            copy.deepcopy(dict(bindings)) if isinstance(bindings, Mapping) else {}
        )
        self.stage_name = active_stage_name(self.stage_dict, self.stage_info)
        self.scenes = self._build_scenes()
        self._ensure_default_bindings()

    @property
    def project_dir(self) -> Path:
        return self.info_path.parent

    def _build_scenes(self) -> list[BindingScene]:
        stage_data = self.stage_info.get(self.stage_name, {})
        scenes_data = stage_data.get("scenes", {}) if isinstance(stage_data, Mapping) else {}
        scenes: list[BindingScene] = []
        if not isinstance(scenes_data, MutableMapping):
            return scenes
        for scene_name, scene_data in scenes_data.items():
            for resolution_key, version in _iter_scene_versions(scene_data):
                scenes.append(
                    BindingScene(
                        stage=self.stage_name,
                        name=str(scene_name),
                        resolution_key=resolution_key,
                        data=version,
                    )
                )
        return scenes

    def _scene_bindings(self, scene_name: str) -> MutableMapping[str, str]:
        stage_bindings = self.bindings.setdefault(self.stage_name, {})
        if not isinstance(stage_bindings, MutableMapping):
            stage_bindings = {}
            self.bindings[self.stage_name] = stage_bindings
        scene_bindings = stage_bindings.setdefault(scene_name, {})
        if not isinstance(scene_bindings, MutableMapping):
            scene_bindings = {}
            stage_bindings[scene_name] = scene_bindings
        return scene_bindings

    def _ensure_default_bindings(self):
        for scene in self.scenes:
            mapping = self._scene_bindings(scene.name)
            for control in self.bindable_controls(scene):
                if control.name not in mapping:
                    # 控点名只是标注名称；用户在绑定窗口按过键盘后才算已绑定。
                    mapping[control.name] = ""

    def joystick_geometry_error(self, scene: BindingScene) -> str:
        try:
            joystick_direction_points(scene.points, scene.width, scene.height)
        except ValueError as exc:
            return str(exc)
        return ""

    def bindable_controls(self, scene: BindingScene) -> list[BindingControl]:
        controls = []
        for raw_name, point_data in scene.points.items():
            if is_joystick_geometry_point(raw_name):
                continue
            rect = point_data.get("rect") if isinstance(point_data, Mapping) else None
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                continue
            try:
                norm_x = (float(rect[0]) + float(rect[2])) / 2.0
                norm_y = (float(rect[1]) + float(rect[3])) / 2.0
            except (TypeError, ValueError):
                continue
            controls.append(
                BindingControl(
                    name=str(raw_name),
                    display_name=str(raw_name),
                    normalized_position=(norm_x, norm_y),
                )
            )
        try:
            directions = joystick_direction_points(scene.points, scene.width, scene.height)
        except ValueError:
            directions = ()
        controls.extend(
            BindingControl(
                name=direction.binding_name,
                display_name=direction.display_name,
                normalized_position=direction.normalized_position,
                is_virtual_joystick_direction=True,
            )
            for direction in directions
        )
        return controls

    def is_bindable_control(self, scene: BindingScene, point_name: str) -> bool:
        return self.control_for(scene, point_name) is not None

    def control_for(self, scene: BindingScene, point_name: str) -> BindingControl | None:
        return next(
            (control for control in self.bindable_controls(scene) if control.name == point_name),
            None,
        )

    def binding_for(self, scene: BindingScene, point_name: str) -> str:
        return str(self._scene_bindings(scene.name).get(point_name) or "")

    def bind_key(self, scene: BindingScene, point_name: str, key: str):
        normalized = normalize_key_name(key)
        if not normalized:
            raise BindingConfigError("没有识别到可绑定的键。")
        if normalized in RESERVED_KEYS:
            raise BindingConfigError(f"{normalized} 是开始/结束录制键，不能绑定为游戏控点。")
        if not self.is_bindable_control(scene, point_name):
            raise BindingConfigError(f"场景“{scene.name}”中不存在控点“{point_name}”。")
        mapping = self._scene_bindings(scene.name)
        for other_name, other_key in list(mapping.items()):
            if other_name != point_name and normalize_key_name(other_key) == normalized:
                mapping[other_name] = ""
        mapping[point_name] = normalized

    def set_point_center(self, scene: BindingScene, point_name: str, norm_x: float, norm_y: float):
        point_data = scene.points.get(point_name)
        rect = point_data.get("rect") if isinstance(point_data, MutableMapping) else None
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise BindingConfigError(f"控点“{point_name}”没有有效的 rect。")
        try:
            left, top, right, bottom = (float(value) for value in rect)
        except (TypeError, ValueError) as exc:
            raise BindingConfigError(f"控点“{point_name}”的 rect 不是数字。") from exc
        rect_width = min(max(right - left, 0.0), 1.0)
        rect_height = min(max(bottom - top, 0.0), 1.0)
        new_left = min(max(float(norm_x) - rect_width / 2.0, 0.0), 1.0 - rect_width)
        new_top = min(max(float(norm_y) - rect_height / 2.0, 0.0), 1.0 - rect_height)
        point_data["rect"] = [
            new_left,
            new_top,
            new_left + rect_width,
            new_top + rect_height,
        ]

    def validate(self):
        if not self.scenes:
            raise BindingConfigError(
                "当前阶段还没有场景，请先用标注工具新建场景并导出。"
            )
        all_keys = set()
        positions_by_key: Dict[str, Dict[str, set[tuple[float, float]]]] = {}
        uses_generated_joystick = False
        for scene in self.scenes:
            seen = set()
            geometry_error = self.joystick_geometry_error(scene)
            if geometry_error:
                raise BindingConfigError(f"场景“{scene.display_name}”的{geometry_error}")
            controls = self.bindable_controls(scene)
            uses_generated_joystick = uses_generated_joystick or any(
                control.is_virtual_joystick_direction for control in controls
            )
            for control in controls:
                point_name = control.name
                key = normalize_key_name(self.binding_for(scene, point_name))
                if not key:
                    if control.is_virtual_joystick_direction:
                        continue
                    raise BindingConfigError(
                        f"场景“{scene.display_name}”的控点“{control.display_name}”还没有绑定键盘按键。"
                    )
                if key in RESERVED_KEYS:
                    raise BindingConfigError(f"控点“{point_name}”占用了录制键 {key}。")
                if key in seen:
                    raise BindingConfigError(
                        f"场景“{scene.display_name}”里键位 {key} 被重复绑定。"
                    )
                seen.add(key)
                all_keys.add(key)
                positions_by_key.setdefault(key, {}).setdefault(scene.name, set()).add(
                    control.normalized_position
                )
        missing = sorted({"w", "a", "s", "d"}.difference(all_keys))
        if missing and not uses_generated_joystick:
            raise BindingConfigError("移动键还缺少：" + "、".join(missing))
        for key, scene_positions in positions_by_key.items():
            if len(scene_positions) <= 1:
                continue
            distinct_positions = set().union(*scene_positions.values())
            if len(distinct_positions) > 1:
                scene_names = "、".join(scene_positions)
                raise BindingConfigError(
                    f"键位 {key} 在多个场景中对应了不同位置（{scene_names}）。"
                    "当前录制模块会合并当前阶段的控点，请为这些控点使用不同键位，"
                    "或将它们放在相同位置。"
                )

    @staticmethod
    def _matching_versions(container: Mapping[str, Any]) -> Iterable[MutableMapping[str, Any]]:
        for stage_or_scene in container.values():
            if not isinstance(stage_or_scene, MutableMapping):
                continue
            scenes = stage_or_scene.get("scenes")
            if isinstance(scenes, MutableMapping):
                for scene_data in scenes.values():
                    yield from (version for _, version in _iter_scene_versions(scene_data))
                continue
            resolutions = stage_or_scene.get("resolutions")
            if isinstance(resolutions, MutableMapping):
                for version in resolutions.values():
                    if isinstance(version, MutableMapping):
                        yield version

    @staticmethod
    def _same_scene_version(source: BindingScene, candidate: Mapping[str, Any]) -> bool:
        if source.image and str(candidate.get("image") or "") == source.image:
            return True
        return (
            not source.image
            and int(candidate.get("width") or 0) == source.width
            and int(candidate.get("height") or 0) == source.height
        )

    def _synchronize_point_rects(self):
        stage_versions = list(self._matching_versions(self.stage_info))
        pool_scenes = self.scene_pool.get("scenes", {})
        pool_versions = (
            list(self._matching_versions(pool_scenes))
            if isinstance(pool_scenes, Mapping)
            else []
        )
        for scene in self.scenes:
            for point_name, point_data in scene.points.items():
                rect = point_data.get("rect") if isinstance(point_data, Mapping) else None
                if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                    continue
                for candidate in stage_versions + pool_versions:
                    if candidate is scene.data or not self._same_scene_version(scene, candidate):
                        continue
                    candidate_points = candidate.get("points", {})
                    if isinstance(candidate_points, MutableMapping):
                        candidate_point = candidate_points.get(point_name)
                        if isinstance(candidate_point, MutableMapping):
                            candidate_point["rect"] = list(rect)

    def save(self):
        self.validate()
        self._synchronize_point_rects()
        replace_info_assignments(
            self.info_path,
            {
                "STAGE_INFO": self.stage_info,
                "SCENE_POOL": self.scene_pool,
                BINDINGS_NAME: self.bindings,
            },
        )
        self.info_module.STAGE_INFO = copy.deepcopy(self.stage_info)
        self.info_module.SCENE_POOL = copy.deepcopy(self.scene_pool)
        setattr(self.info_module, BINDINGS_NAME, copy.deepcopy(self.bindings))


def replace_info_assignments(path: Path, values: Mapping[str, Any]):
    """原子替换 info.py 中的顶层常量，保留其他注释和字段。"""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise BindingConfigError(f"info.py 无法解析：{exc}") from exc

    lines = source.splitlines(keepends=True)
    assignments: Dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in values:
                assignments[target.id] = (node.lineno - 1, node.end_lineno or node.lineno)

    replacements = []
    for name, value in values.items():
        rendered = f"{name} = {pprint.pformat(value, width=120, sort_dicts=False)}\n"
        if name in assignments:
            start, end = assignments[name]
            replacements.append((start, end, rendered))
        else:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += "\n"
            lines.append("\n" + rendered)

    for start, end, rendered in sorted(replacements, reverse=True):
        lines[start:end] = [rendered]
    updated = "".join(lines)

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
