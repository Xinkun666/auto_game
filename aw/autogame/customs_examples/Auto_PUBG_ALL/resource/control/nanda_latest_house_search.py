"""南大最新版房型匹配与单摇杆回放的 auto_game 适配实现。

房型匹配直接在当前搜房进程内执行；门前位姿只使用已有入门点方向和
YOLO 门框，不调用 SAM3。房型配准直接消费搜房阶段
``get_info("sam3")`` 产生的 special_area 分割结果，不再重复推理。
真正的房屋回放使用当前 HOScrcpy 流的单指触控通道，只复现
最新版房型库中唯一实际使用的 ``do_move`` 动作。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_house_search_strategy import (
    NandaEntryPosePreparer,
    NandaHouseSearchStrategy,
    NandaReplayExecutor,
    NandaRoomMatch,
    NandaRoomMatcher,
    NandaSearchContext,
    NandaSearchResult,
    NandaSearchStatus,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_room_matcher_runtime import (
    IntegratedNandaRoomMatcher,
    NandaMatcherAssetPaths,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    plan_view_turn_motion,
)


LOGGER = logging.getLogger("NandaLatestHouseSearch")


_UNSUPPORTED_REPLAY_ACTIONS = {
    "view_left",
    "view_right",
    "view_up",
    "view_down",
    "jump",
    "pick_btn",
    "map",
    "door",
    "attack",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class NandaLatestSettings:
    enabled: bool = False
    door_pose_backend: str = "entry_direction_yolo"
    room_segmenter_backend: str = "sam3_special_area"
    dino_model_dir: str = ""
    mlp_model_path: str = ""
    room_library_path: str = ""
    matcher_device: str = ""

    max_entry_distance: float = 2.5
    direction_tolerance_degrees: float = 3.0
    area_min_ratio: float = 0.02
    area_max_ratio: float = 0.04
    area_acceptable_min_ratio: float = 0.015
    area_acceptable_max_ratio: float = 0.055
    acceptable_center_ratio: float = 0.03
    lateral_band_ratio: float = 0.05
    lateral_band_duration_ms: int = 30
    stable_required_count: int = 2
    max_pose_actions: int = 18
    move_axis_bias: int = 240
    pose_min_duration_ms: int = 60
    pose_max_duration_ms: int = 600
    pose_wait_ms: int = 500

    match_backoff_duration_ms: int = 500
    match_backoff_wait_ms: int = 500
    mask_top_edge_ratio: float = 0.01
    pitch_compensation: bool = True
    pitch_height_ratio: float = 0.08
    pitch_duration_ms: int = 320
    pitch_wait_ms: int = 450

    joystick_center_x_ratio: float = 0.1965
    joystick_center_y_ratio: float = 0.7563
    joystick_radius_px: int = 0
    joystick_radius_height_ratio: float = 0.0974
    joystick_slide_duration_ms: int = 100
    replay_skip_idle: bool = True
    replay_touch_recovery_timeout_seconds: float = 60.0

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "NandaLatestSettings":
        raw = dict(config or {})
        return cls(
            enabled=_as_bool(raw.get("enabled"), False),
            door_pose_backend=str(
                raw.get("door_pose_backend") or cls.door_pose_backend
            ).strip(),
            room_segmenter_backend=str(
                raw.get("room_segmenter_backend") or cls.room_segmenter_backend
            ).strip(),
            dino_model_dir=str(
                os.environ.get("AUTOGAME_NANDA_DINO_MODEL_DIR", "").strip()
                or raw.get("dino_model_dir")
                or ""
            ).strip(),
            mlp_model_path=str(
                os.environ.get("AUTOGAME_NANDA_MLP_MODEL_PATH", "").strip()
                or raw.get("mlp_model_path")
                or ""
            ).strip(),
            room_library_path=str(
                os.environ.get("AUTOGAME_NANDA_ROOM_LIBRARY", "").strip()
                or raw.get("room_library_path")
                or ""
            ).strip(),
            matcher_device=str(
                os.environ.get("AUTOGAME_NANDA_DEVICE", "").strip()
                or raw.get("matcher_device")
                or ""
            ).strip(),
            max_entry_distance=max(
                0.1, _as_float(raw.get("max_entry_distance"), cls.max_entry_distance)
            ),
            direction_tolerance_degrees=max(
                0.1,
                _as_float(
                    raw.get("direction_tolerance_degrees"),
                    cls.direction_tolerance_degrees,
                ),
            ),
            area_min_ratio=max(0.0, _as_float(raw.get("area_min_ratio"), cls.area_min_ratio)),
            area_max_ratio=max(0.0, _as_float(raw.get("area_max_ratio"), cls.area_max_ratio)),
            area_acceptable_min_ratio=max(
                0.0,
                _as_float(
                    raw.get("area_acceptable_min_ratio"), cls.area_acceptable_min_ratio
                ),
            ),
            area_acceptable_max_ratio=max(
                0.0,
                _as_float(
                    raw.get("area_acceptable_max_ratio"), cls.area_acceptable_max_ratio
                ),
            ),
            acceptable_center_ratio=max(
                0.001,
                _as_float(raw.get("acceptable_center_ratio"), cls.acceptable_center_ratio),
            ),
            lateral_band_ratio=max(
                0.001,
                _as_float(raw.get("lateral_band_ratio"), cls.lateral_band_ratio),
            ),
            lateral_band_duration_ms=max(
                1,
                _as_int(
                    raw.get("lateral_band_duration_ms"),
                    cls.lateral_band_duration_ms,
                ),
            ),
            stable_required_count=max(
                1, _as_int(raw.get("stable_required_count"), cls.stable_required_count)
            ),
            max_pose_actions=max(1, _as_int(raw.get("max_pose_actions"), cls.max_pose_actions)),
            move_axis_bias=max(1, _as_int(raw.get("move_axis_bias"), cls.move_axis_bias)),
            pose_min_duration_ms=max(
                1, _as_int(raw.get("pose_min_duration_ms"), cls.pose_min_duration_ms)
            ),
            pose_max_duration_ms=max(
                1, _as_int(raw.get("pose_max_duration_ms"), cls.pose_max_duration_ms)
            ),
            pose_wait_ms=max(0, _as_int(raw.get("pose_wait_ms"), cls.pose_wait_ms)),
            match_backoff_duration_ms=max(
                1,
                _as_int(
                    raw.get("match_backoff_duration_ms"),
                    cls.match_backoff_duration_ms,
                ),
            ),
            match_backoff_wait_ms=max(
                0,
                _as_int(
                    raw.get("match_backoff_wait_ms"),
                    cls.match_backoff_wait_ms,
                ),
            ),
            mask_top_edge_ratio=min(
                0.2,
                max(
                    0.0,
                    _as_float(
                        raw.get("mask_top_edge_ratio"), cls.mask_top_edge_ratio
                    ),
                ),
            ),
            pitch_compensation=_as_bool(
                raw.get("pitch_compensation"), cls.pitch_compensation
            ),
            pitch_height_ratio=min(
                0.2,
                max(
                    0.001,
                    _as_float(raw.get("pitch_height_ratio"), cls.pitch_height_ratio),
                ),
            ),
            pitch_duration_ms=max(
                1,
                _as_int(raw.get("pitch_duration_ms"), cls.pitch_duration_ms),
            ),
            pitch_wait_ms=max(0, _as_int(raw.get("pitch_wait_ms"), cls.pitch_wait_ms)),
            joystick_center_x_ratio=_as_float(
                raw.get("joystick_center_x_ratio"), cls.joystick_center_x_ratio
            ),
            joystick_center_y_ratio=_as_float(
                raw.get("joystick_center_y_ratio"), cls.joystick_center_y_ratio
            ),
            joystick_radius_px=max(
                0,
                _as_int(raw.get("joystick_radius_px"), cls.joystick_radius_px),
            ),
            joystick_radius_height_ratio=max(
                0.01,
                _as_float(
                    raw.get("joystick_radius_height_ratio"),
                    cls.joystick_radius_height_ratio,
                ),
            ),
            joystick_slide_duration_ms=max(
                0,
                _as_int(
                    raw.get("joystick_slide_duration_ms"),
                    cls.joystick_slide_duration_ms,
                ),
            ),
            replay_skip_idle=_as_bool(raw.get("replay_skip_idle"), cls.replay_skip_idle),
            replay_touch_recovery_timeout_seconds=max(
                1.0,
                _as_float(
                    raw.get("replay_touch_recovery_timeout_seconds"),
                    cls.replay_touch_recovery_timeout_seconds,
                ),
            ),
        )


class NandaYoloDoorPosePreparer(NandaEntryPosePreparer):
    """只用入门点方向和现有 YOLO 门框收敛南大回放门前位姿。"""

    def __init__(self, settings: NandaLatestSettings):
        self.settings = settings
        self._pose_key = None
        self._action_count = 0
        self._stable_count = 0

    def reset(self) -> None:
        self._pose_key = None
        self._action_count = 0
        self._stable_count = 0

    @staticmethod
    def _angular_error(current: Optional[float], target: Optional[float]) -> Optional[float]:
        if current is None or target is None:
            return None
        return abs((float(target) - float(current) + 540.0) % 360.0 - 180.0)

    @staticmethod
    def _screen_width(context: NandaSearchContext) -> Optional[float]:
        controller = getattr(context.worker, "controller", None)
        get_resolution = getattr(controller, "_get_cached_resolution", None)
        if callable(get_resolution):
            resolution = get_resolution()
            if resolution and resolution[0]:
                return float(resolution[0])
        frame = context.frame
        if frame is not None and getattr(frame, "shape", None) is not None:
            return float(frame.shape[1])
        return None

    def _duration_for_error(self, error: float, threshold: float, reference: float) -> int:
        active_error = max(0.0, abs(error) - abs(threshold))
        ratio = min(1.0, active_error / max(1e-6, reference - abs(threshold)))
        low = min(self.settings.pose_min_duration_ms, self.settings.pose_max_duration_ms)
        high = max(self.settings.pose_min_duration_ms, self.settings.pose_max_duration_ms)
        return int(round(low + (high - low) * ratio))

    def _lateral_duration_for_error(self, center_error: float) -> Tuple[int, int]:
        segment = self.settings.lateral_band_ratio
        active_error = max(
            0.0,
            center_error - self.settings.acceptable_center_ratio,
        )
        band = max(1, math.ceil(active_error / segment - 1e-9))
        return band, min(500, band * self.settings.lateral_band_duration_ms)

    def _retry_after_action(
        self,
        context: NandaSearchContext,
        message: str,
        *,
        x_bias: int = 0,
        y_bias: int = 0,
        duration_ms: int,
        control: str = "摇杆",
    ) -> NandaSearchResult:
        self._action_count += 1
        self._stable_count = 0
        context.worker.frame_log(f"[NandaPose] {message}")
        context.worker.tap_single(
            control,
            x_bias=int(x_bias),
            y_bias=int(y_bias),
            dura=int(duration_ms),
            wait=self.settings.pose_wait_ms,
        )
        context.refresh_frame(f"NandaPose {message}")
        return NandaSearchResult(
            NandaSearchStatus.RETRY,
            message,
            metadata={"phase": "pose", "action_count": self._action_count},
        )

    def _pose_timeout_result(
        self,
        center_error: float,
        area_ratio: float,
    ) -> Optional[NandaSearchResult]:
        if self._action_count < self.settings.max_pose_actions:
            return None
        acceptable = (
            center_error <= self.settings.acceptable_center_ratio
            and self.settings.area_acceptable_min_ratio
            <= area_ratio
            <= self.settings.area_acceptable_max_ratio
        )
        if acceptable:
            return None
        return NandaSearchResult(
            NandaSearchStatus.NO_MATCH,
            "门前位姿多次调整仍未进入南大方案可接受范围",
            metadata={
                "phase": "pose",
                "action_count": self._action_count,
                "center_error": center_error,
                "door_area_ratio": area_ratio,
            },
        )

    def prepare(self, context: NandaSearchContext) -> Optional[NandaSearchResult]:
        pose_key = (context.house_id, context.entry_location)
        if pose_key != self._pose_key:
            self._pose_key = pose_key
            self._action_count = 0
            self._stable_count = 0

        if context.distance_to_entry is None or (
            context.distance_to_entry > self.settings.max_entry_distance
        ):
            return NandaSearchResult(
                NandaSearchStatus.NO_MATCH,
                f"距离入门点尚未进入 {self.settings.max_entry_distance:g} 范围",
                metadata={"phase": "pose"},
            )
        if context.door_box is None or context.door_area_ratio is None:
            return NandaSearchResult(
                NandaSearchStatus.NO_MATCH,
                "门检测框无效，退回原搜房策略",
                metadata={"phase": "pose"},
            )

        direction_error = self._angular_error(
            context.current_direction,
            context.entry_direction,
        )
        if direction_error is not None and (
            direction_error > self.settings.direction_tolerance_degrees
        ):
            motion = plan_view_turn_motion(
                context.current_direction,
                context.entry_direction,
                min_dura=self.settings.pose_min_duration_ms,
                max_dura=self.settings.pose_max_duration_ms,
                max_px=400,
            )
            if motion is None:
                return NandaSearchResult(
                    NandaSearchStatus.NO_MATCH,
                    "无法计算入门方向校准动作，退回原搜房策略",
                    metadata={"phase": "pose"},
                )
            return self._retry_after_action(
                context,
                f"入门方向误差 {direction_error:.1f}°，先恢复门的垂直观察方向",
                x_bias=int(motion["x_bias"]),
                duration_ms=int(motion["dura"]),
                control="视角",
            )

        screen_width = self._screen_width(context)
        if not screen_width or context.door_center_offset_px is None:
            return NandaSearchResult(
                NandaSearchStatus.NO_MATCH,
                "无法计算门中心归一化偏差，退回原搜房策略",
                metadata={"phase": "pose"},
            )
        center_delta = float(context.door_center_offset_px) / screen_width
        center_error = abs(center_delta)
        area_ratio = float(context.door_area_ratio)

        timeout_result = self._pose_timeout_result(center_error, area_ratio)
        if timeout_result is not None:
            return timeout_result
        relaxed_accept = self._action_count >= self.settings.max_pose_actions

        # 3% 内视为精准对准；之后每增加 5%，横移时间增加 30ms。
        if not relaxed_accept and center_error > self.settings.acceptable_center_ratio:
            band, duration = self._lateral_duration_for_error(center_error)
            side = 1 if center_delta > 0 else -1
            return self._retry_after_action(
                context,
                f"门中心偏差 {center_delta:+.3f}({center_error:.1%})，"
                f"第 {band} 档横移 {duration}ms",
                x_bias=side * self.settings.move_axis_bias,
                duration_ms=duration,
            )

        if not relaxed_accept and area_ratio < self.settings.area_min_ratio:
            duration = self._duration_for_error(
                self.settings.area_min_ratio - area_ratio,
                0.0,
                0.03,
            )
            return self._retry_after_action(
                context,
                f"门框面积 {area_ratio:.3f} 偏小，向前靠近标准回放距离",
                y_bias=-self.settings.move_axis_bias,
                duration_ms=duration,
            )
        if not relaxed_accept and area_ratio > self.settings.area_max_ratio:
            duration = self._duration_for_error(
                area_ratio - self.settings.area_max_ratio,
                0.0,
                0.04,
            )
            return self._retry_after_action(
                context,
                f"门框面积 {area_ratio:.3f} 偏大，向后退到标准回放距离",
                y_bias=self.settings.move_axis_bias,
                duration_ms=duration,
            )

        self._stable_count += 1
        if self._stable_count < self.settings.stable_required_count:
            return NandaSearchResult(
                NandaSearchStatus.RETRY,
                f"门前位姿已达标，等待稳定帧 {self._stable_count}/"
                f"{self.settings.stable_required_count}",
                metadata={"phase": "pose", "stable_count": self._stable_count},
            )
        context.worker.frame_log(
            f"[NandaPose] 入门方向+YOLO门框位姿完成：center={center_delta:+.3f}，"
            f"area={area_ratio:.3f}，stable={self._stable_count}"
        )
        return None


# 兼容已经引用过旧名称的本地测试/扩展；实际实现明确是 YOLO 门框校准。
NandaDoorPosePreparer = NandaYoloDoorPosePreparer


class _NandaSpecialAreaRoomMatcher(NandaRoomMatcher):
    """进程内匹配器共用的 SAM3 分组切换与立面重建。"""

    @staticmethod
    def _read_sam3_info(context: NandaSearchContext) -> Mapping[str, Any]:
        get_info = getattr(context.worker, "get_info", None)
        if not callable(get_info):
            raise ValueError("FrameWorker 不支持 get_info('sam3')")
        sam3_info = get_info("sam3")
        if not isinstance(sam3_info, dict):
            raise ValueError("搜房阶段未取得 sam3 special_area 结果")
        return sam3_info

    @staticmethod
    def _activate_sam3_perception(
        context: NandaSearchContext,
    ) -> str:
        """按实际导出配置启用 SAM3，返回恢复所需的原状态。

        SAM3 必须是搜房阶段内的按需分组，不能作为常态感知或独立阶段运行。
        """
        worker = context.worker
        original_stage = str(
            getattr(worker, "current_stage", None)
            or (worker.get_stage() if callable(getattr(worker, "get_stage", None)) else "")
            or ""
        )
        original_group = str(getattr(worker, "current_group", None) or "默认")
        if not original_stage:
            raise ValueError("启用 sam3 前无法确定当前阶段")

        resolver = getattr(worker, "stage_resolver", None)
        has_group = getattr(resolver, "has_group", None)
        if not callable(has_group) or not has_group(original_stage, "sam3"):
            raise ValueError(f"阶段 {original_stage!r} 中没有 sam3 分组")
        change_group = getattr(worker, "change_group", None)
        if not callable(change_group) or change_group("sam3") is not True:
            raise ValueError("切换到 sam3 分组失败")
        worker.frame_log(
            f"[NandaMatch] 房型匹配按需启用分组: "
            f"{original_stage}/{original_group} -> sam3"
        )
        return original_group

    @staticmethod
    def _restore_perception(
        context: NandaSearchContext,
        original_group: Optional[str],
    ) -> None:
        if original_group is not None:
            if context.worker.change_group(original_group) is not True:
                raise RuntimeError(f"恢复搜房分组失败: {original_group}")

    def _capture_match_frame(
        self,
        context: NandaSearchContext,
        *,
        apply_pitch: bool = False,
    ) -> Tuple[np.ndarray, Mapping[str, Any]]:
        frame = context.frame
        if frame is None:
            raise ValueError("当前没有可用于房型匹配的画面")
        pitch_bias = max(
            1,
            int(round(float(frame.shape[0]) * self.settings.pitch_height_ratio)),
        )
        pitch_applied = False
        original_group = None
        try:
            if apply_pitch:
                context.worker.frame_log(
                    f"[NandaMatch] 第三阶段轻微抬头复核："
                    f"y_bias=-{pitch_bias}，dura={self.settings.pitch_duration_ms}ms"
                )
                context.worker.tap_single(
                    "视角",
                    y_bias=-pitch_bias,
                    dura=self.settings.pitch_duration_ms,
                    wait=self.settings.pitch_wait_ms,
                )
                pitch_applied = True
                if not context.refresh_frame("NandaMatch 轻微抬头采集房屋正面"):
                    raise ValueError("轻微抬头后刷新画面失败")

            original_group = self._activate_sam3_perception(context)

            captured = getattr(context.worker, "frame", None)
            if captured is None:
                raise ValueError("sam3 分割后未取得匹配画面")
            return captured.copy(), self._read_sam3_info(context)
        finally:
            restore_error = None
            try:
                self._restore_perception(
                    context,
                    original_group,
                )
            except Exception as exc:
                restore_error = exc
            if pitch_applied:
                context.worker.tap_single(
                    "视角",
                    y_bias=pitch_bias,
                    dura=self.settings.pitch_duration_ms,
                    wait=self.settings.pitch_wait_ms,
                )
                if not context.refresh_frame("NandaMatch 恢复门前回放视角"):
                    restore_error = restore_error or RuntimeError(
                        "恢复门前视角后刷新画面失败"
                    )
            if restore_error is not None:
                raise restore_error

    @staticmethod
    def _special_area_facade(
        frame: np.ndarray,
        sam3_info: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        if sam3_info.get("found") is False:
            raise ValueError("sam3 special_area 未分割到房屋")
        visuals = sam3_info.get("__visualizations__")
        if not isinstance(visuals, list):
            raise ValueError("sam3 special_area 结果缺少可还原的 mask 轮廓")

        frame_h, frame_w = frame.shape[:2]
        best_mask = None
        best_area = 0
        for visual in visuals:
            if not isinstance(visual, dict) or visual.get("type") != "sam3_mask":
                continue
            contours = visual.get("contours")
            if not isinstance(contours, list):
                continue
            coord = str(visual.get("coord") or "local")
            offset_x = 0
            offset_y = 0
            if coord != "frame":
                source_crop = visual.get("source_crop_xyxy")
                if isinstance(source_crop, (list, tuple)) and len(source_crop) >= 2:
                    offset_x = int(round(float(source_crop[0])))
                    offset_y = int(round(float(source_crop[1])))
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            polygons = []
            for contour in contours:
                if not isinstance(contour, list):
                    continue
                points = []
                for point in contour:
                    if isinstance(point, dict):
                        raw_x, raw_y = point.get("x"), point.get("y")
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        raw_x, raw_y = point[0], point[1]
                    else:
                        continue
                    try:
                        point_x = int(round(float(raw_x))) + offset_x
                        point_y = int(round(float(raw_y))) + offset_y
                    except (TypeError, ValueError):
                        continue
                    points.append(
                        [
                            min(frame_w - 1, max(0, point_x)),
                            min(frame_h - 1, max(0, point_y)),
                        ]
                    )
                if len(points) >= 3:
                    polygons.append(np.asarray(points, dtype=np.int32))
            if polygons:
                cv2.fillPoly(mask, polygons, 255)
            mask_area = int(np.count_nonzero(mask))
            if mask_area > best_area:
                best_mask = mask
                best_area = mask_area

        if best_mask is None or best_area <= 0:
            raise ValueError("sam3 special_area mask 轮廓为空")
        nonzero_y, nonzero_x = np.nonzero(best_mask)
        x1 = int(nonzero_x.min())
        y1 = int(nonzero_y.min())
        x2 = int(nonzero_x.max()) + 1
        y2 = int(nonzero_y.max()) + 1
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("sam3 special_area mask 裁剪范围过小")

        cropped_bgr = np.ascontiguousarray(frame[y1:y2, x1:x2]).copy()
        cropped_mask = np.ascontiguousarray(best_mask[y1:y2, x1:x2]).copy()
        segmented_bgr = np.zeros_like(cropped_bgr)
        segmented_bgr[cropped_mask > 0] = cropped_bgr[cropped_mask > 0]
        return segmented_bgr, cropped_mask, cropped_bgr, (x1, y1, x2, y2)


def _version_tuple(value: str) -> Tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(number) for number in numbers[:3])


@dataclass(frozen=True)
class _NandaRoomMatchAttempt:
    index: int
    label: str
    room_id: Optional[str]
    replay_path: Optional[str]
    score: Optional[float]
    mask_found: bool
    touches_top: bool
    crop_xyxy: Optional[Tuple[int, int, int, int]]
    sam3_score: Any
    replay_allowed: bool
    no_match_reason: str
    decision: Mapping[str, Any]
    debug_payload: Mapping[str, Any]

    @property
    def valid(self) -> bool:
        return bool(
            self.room_id
            and self.replay_path
            and self.replay_allowed
        )

    def summary(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "room_id": self.room_id,
            "confidence": self.score,
            "mask_found": self.mask_found,
            "touches_top": self.touches_top,
            "crop_xyxy": self.crop_xyxy,
            "sam3_score": self.sam3_score,
            "replay_allowed": self.replay_allowed,
            "no_match_reason": self.no_match_reason,
        }


class NandaLocalRoomMatcher(_NandaSpecialAreaRoomMatcher):
    """在当前搜房进程内直接执行 DINOv3 + MLP 房型配准。"""

    _shared_runtime_lock = threading.Lock()
    _shared_runtimes: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        settings: NandaLatestSettings,
        runtime_factory: Callable[..., Any] = IntegratedNandaRoomMatcher,
    ):
        self.settings = settings
        self._runtime_factory = runtime_factory
        self._runtime = None
        self.unavailable_reason = "进程内 DINOv3/MLP 尚未检查"

    def _asset_paths(self) -> NandaMatcherAssetPaths:
        defaults = NandaMatcherAssetPaths.auto_game_defaults()
        return NandaMatcherAssetPaths(
            dino_model_dir=(
                Path(self.settings.dino_model_dir)
                if self.settings.dino_model_dir
                else defaults.dino_model_dir
            ),
            mlp_model_path=(
                Path(self.settings.mlp_model_path)
                if self.settings.mlp_model_path
                else defaults.mlp_model_path
            ),
            room_library_path=(
                Path(self.settings.room_library_path)
                if self.settings.room_library_path
                else defaults.room_library_path
            ),
        ).resolved()

    @staticmethod
    def _dependency_problem() -> Optional[str]:
        missing = [
            name
            for name in ("torch", "transformers", "safetensors", "sklearn")
            if find_spec(name) is None
        ]
        if missing:
            return "当前 auto_game 进程缺少依赖: " + ", ".join(missing)
        try:
            numpy_version = importlib_metadata.version("numpy")
            sklearn_version = importlib_metadata.version("scikit-learn")
        except importlib_metadata.PackageNotFoundError as exc:
            return f"当前 auto_game 进程缺少依赖: {exc.name}"
        if _version_tuple(numpy_version) < (2, 0):
            return f"南大 MLP 要求 numpy>=2.0，当前为 {numpy_version}"
        if _version_tuple(sklearn_version) < (1, 7, 2):
            return f"南大 MLP 要求 scikit-learn>=1.7.2，当前为 {sklearn_version}"
        return None

    def is_available(self) -> bool:
        dependency_problem = self._dependency_problem()
        if dependency_problem:
            self.unavailable_reason = dependency_problem
            return False
        try:
            self._asset_paths().validate()
        except (OSError, RuntimeError, ValueError) as exc:
            self.unavailable_reason = f"进程内房型匹配资产不可用: {exc}"
            return False
        self.unavailable_reason = "进程内 DINOv3/MLP 资产与依赖已就绪"
        return True

    def _ensure_runtime(self, emit_log: Callable[[str], None]):
        if self._runtime is not None:
            return self._runtime

        paths = self._asset_paths()
        cache_key = (
            str(paths.dino_model_dir),
            str(paths.mlp_model_path),
            str(paths.room_library_path),
            self.settings.matcher_device or "auto",
            self._runtime_factory,
        )
        with self._shared_runtime_lock:
            shared_runtime = self._shared_runtimes.get(cache_key)
            if shared_runtime is not None:
                self._runtime = shared_runtime
                emit_log(
                    "[NandaMatch] 复用当前进程已加载的 "
                    "DINOv3、MLP 和房型索引，不重复加载权重"
                )
            else:
                emit_log(
                    "[NandaMatch] 当前进程首次初始化，正在加载 "
                    f"DINOv3、MLP 和房型索引：{paths.to_jsonable()}；"
                    "本进程后续房屋将直接复用"
                )
                self._runtime = self._runtime_factory(
                    paths,
                    device=self.settings.matcher_device or None,
                )
                self._shared_runtimes[cache_key] = self._runtime
        return self._runtime

    def warmup(self) -> None:
        self._ensure_runtime(LOGGER.info)

    def _get_runtime(self, context: NandaSearchContext):
        return self._ensure_runtime(context.worker.frame_log)

    @staticmethod
    def _optional_score(value: Any) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    @classmethod
    def _decision_score(cls, decision: Mapping[str, Any]) -> Optional[float]:
        for key in ("total_score", "mlp_score", "room_best_dino_score"):
            score = cls._optional_score(decision.get(key))
            if score is not None:
                return score
        return None

    @staticmethod
    def _invalid_attempt(
        index: int,
        label: str,
        reason: str,
    ) -> _NandaRoomMatchAttempt:
        return _NandaRoomMatchAttempt(
            index=index,
            label=label,
            room_id=None,
            replay_path=None,
            score=None,
            mask_found=False,
            touches_top=False,
            crop_xyxy=None,
            sam3_score=None,
            replay_allowed=False,
            no_match_reason=str(reason),
            decision={},
            debug_payload={},
        )

    def _run_match_attempt(
        self,
        context: NandaSearchContext,
        *,
        index: int,
        label: str,
        apply_pitch: bool = False,
    ) -> _NandaRoomMatchAttempt:
        if context.should_abort():
            return self._invalid_attempt(index, label, "搜房阶段已中止")
        frame, sam3_info = self._capture_match_frame(
            context,
            apply_pitch=apply_pitch,
        )
        try:
            segmented_bgr, cropped_mask, cropped_bgr, crop_xyxy = (
                self._special_area_facade(frame, sam3_info)
            )
        except ValueError as exc:
            context.worker.frame_log(
                f"[NandaMatch] 第{index}阶段({label}) SAM3 房屋分割无效：{exc}"
            )
            return self._invalid_attempt(index, label, str(exc))

        top_edge_limit = max(
            0,
            int(round(float(frame.shape[0]) * self.settings.mask_top_edge_ratio)),
        )
        touches_top = int(crop_xyxy[1]) <= top_edge_limit

        context.worker.frame_log(
            f"[NandaMatch] 第{index}阶段({label}) sam3_tiny 分割完成："
            f"crop={crop_xyxy}，top_limit={top_edge_limit}，"
            f"touches_top={touches_top}，"
            f"score={sam3_info.get('score')}；直接在当前进程执行 DINOv3/MLP"
        )
        runtime = self._get_runtime(context)
        room_id, replay_path, debug_payload = runtime.match(
            segmented_bgr,
            cropped_mask,
            cropped_bgr,
        )
        debug_payload = debug_payload if isinstance(debug_payload, dict) else {}
        decision = debug_payload.get("decision")
        decision = decision if isinstance(decision, dict) else {}
        dino_score = decision.get("room_best_dino_score")
        mlp_score = decision.get("mlp_score")
        total_score = decision.get("total_score")
        score = self._decision_score(decision)
        margin = debug_payload.get("top2_margin")
        elapsed_ms = debug_payload.get("elapsed_ms")
        replay_allowed = decision.get("replay_allow_actions") is not False
        no_match_reason = str(debug_payload.get("no_match_reason") or "")
        if room_id is None or replay_path is None or not replay_allowed:
            if not replay_allowed:
                no_match_reason = (
                    str(decision.get("replay_disabled_reason") or "")
                    or "replay_actions_rejected"
                )
            context.worker.frame_log(
                f"[NandaMatch] 第{index}阶段({label}) 房型配准未通过："
                f"reason={no_match_reason or 'unknown'}，"
                f"dino={dino_score}，mlp={mlp_score}，total={total_score}，"
                f"margin={margin}，elapsed_ms={elapsed_ms}"
            )
        else:
            context.worker.frame_log(
                f"[NandaMatch] 第{index}阶段({label}) 匹配候选："
                f"room={room_id}，confidence={score}，dino={dino_score}，"
                f"mlp={mlp_score}，total={total_score}，margin={margin}，"
                f"touches_top={touches_top}，elapsed_ms={elapsed_ms}"
            )

        return _NandaRoomMatchAttempt(
            index=index,
            label=label,
            room_id=None if room_id is None else str(room_id),
            replay_path=None if replay_path is None else str(replay_path),
            score=score,
            mask_found=True,
            touches_top=touches_top,
            crop_xyxy=crop_xyxy,
            sam3_score=sam3_info.get("score"),
            replay_allowed=replay_allowed,
            no_match_reason=no_match_reason,
            decision=decision,
            debug_payload=debug_payload,
        )

    def _move_backward_for_rematch(self, context: NandaSearchContext) -> None:
        context.worker.frame_log(
            f"[NandaMatch] 首次结果不足以直接采信，后拉 "
            f"{self.settings.match_backoff_duration_ms}ms 扩大房屋正面取景"
        )
        context.worker.tap_single(
            "摇杆",
            y_bias=self.settings.move_axis_bias,
            dura=self.settings.match_backoff_duration_ms,
            wait=self.settings.match_backoff_wait_ms,
        )
        if not context.refresh_frame("NandaMatch 后拉后重新采集房屋正面"):
            raise RuntimeError("房型复核后拉后刷新画面失败")

    @staticmethod
    def _best_valid_attempt(
        attempts: Sequence[_NandaRoomMatchAttempt],
    ) -> Optional[_NandaRoomMatchAttempt]:
        valid = [attempt for attempt in attempts if attempt.valid]
        if not valid:
            return None

        def sort_key(attempt: _NandaRoomMatchAttempt) -> Tuple[float, int]:
            score = attempt.score
            return (
                score if score is not None and math.isfinite(score) else float("-inf"),
                attempt.index,
            )

        return max(valid, key=sort_key)

    @classmethod
    def _consensus_attempt(
        cls,
        attempts: Sequence[_NandaRoomMatchAttempt],
    ) -> Optional[_NandaRoomMatchAttempt]:
        by_room: Dict[str, List[_NandaRoomMatchAttempt]] = {}
        for attempt in attempts:
            if attempt.valid and attempt.room_id is not None:
                by_room.setdefault(attempt.room_id, []).append(attempt)
        repeated = [room_attempts for room_attempts in by_room.values() if len(room_attempts) >= 2]
        if not repeated:
            return None
        repeated.sort(
            key=lambda room_attempts: (
                len(room_attempts),
                max(
                    attempt.score
                    if attempt.score is not None and math.isfinite(attempt.score)
                    else float("-inf")
                    for attempt in room_attempts
                ),
            ),
            reverse=True,
        )
        return cls._best_valid_attempt(repeated[0])

    def _build_selected_match(
        self,
        context: NandaSearchContext,
        selected: _NandaRoomMatchAttempt,
        attempts: Sequence[_NandaRoomMatchAttempt],
        *,
        selection_reason: str,
        requires_pose_realign: bool,
    ) -> NandaRoomMatch:
        if not selected.valid or selected.replay_path is None or selected.room_id is None:
            raise ValueError("无法从无效候选构建南大房型匹配结果")

        replay_file = Path(selected.replay_path).resolve()
        try:
            replay_steps = json.loads(replay_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取南大回放 DSL: {replay_file}: {exc}") from exc
        if not isinstance(replay_steps, list):
            raise ValueError(f"南大回放 DSL 不是列表: {replay_file}")

        attempt_summaries = [dict(attempt.summary()) for attempt in attempts]
        context.worker.frame_log(
            f"[NandaMatch] 三阶段决策完成：reason={selection_reason}，"
            f"selected_room={selected.room_id}，confidence={selected.score}，"
            f"selected_stage={selected.index}({selected.label})，"
            f"requires_pose_realign={requires_pose_realign}，attempts={attempt_summaries}；"
            "恢复标准门前位姿后再开始 HOS 摇杆回放"
        )
        return NandaRoomMatch(
            room_id=selected.room_id,
            replay_path=str(replay_file),
            score=selected.score,
            metadata={
                "decision": dict(selected.decision),
                "thresholds": selected.debug_payload.get("thresholds"),
                "top2_margin": selected.debug_payload.get("top2_margin"),
                "top_candidates": selected.debug_payload.get("top_candidates"),
                "matcher_elapsed_ms": selected.debug_payload.get("elapsed_ms"),
                "input_contract": self.settings.room_segmenter_backend,
                "execution_mode": "inprocess",
                "structure_mode": "disabled_zero_vector_no_extra_sam3",
                "matching_attempts": attempt_summaries,
                "selected_attempt": selected.index,
                "selection_reason": selection_reason,
                "requires_pose_realign": requires_pose_realign,
            },
            replay_steps=replay_steps,
        )

    def _finish_attempts(
        self,
        context: NandaSearchContext,
        attempts: Sequence[_NandaRoomMatchAttempt],
        *,
        selection_reason: str,
        requires_pose_realign: bool,
        prefer_consensus: bool = False,
    ) -> Optional[NandaRoomMatch]:
        selected = self._consensus_attempt(attempts) if prefer_consensus else None
        resolved_reason = selection_reason
        if selected is not None:
            resolved_reason = "room_consensus"
        else:
            selected = self._best_valid_attempt(attempts)
        if selected is None:
            context.worker.frame_log(
                f"[NandaMatch] 多阶段房型匹配结束仍无可用候选："
                f"reason={selection_reason}，"
                f"attempts={[dict(attempt.summary()) for attempt in attempts]}"
            )
            return None
        return self._build_selected_match(
            context,
            selected,
            attempts,
            selection_reason=resolved_reason,
            requires_pose_realign=requires_pose_realign,
        )

    def match(self, context: NandaSearchContext) -> Optional[NandaRoomMatch]:
        if context.should_abort():
            return None

        attempts: List[_NandaRoomMatchAttempt] = []
        first = self._run_match_attempt(
            context,
            index=1,
            label="normal",
        )
        attempts.append(first)
        if first.valid and not first.touches_top:
            return self._finish_attempts(
                context,
                attempts,
                selection_reason="first_match_complete_mask",
                requires_pose_realign=False,
            )

        if context.should_abort():
            return None
        self._move_backward_for_rematch(context)
        second = self._run_match_attempt(
            context,
            index=2,
            label="backoff_500ms",
        )
        attempts.append(second)

        if first.valid and second.valid and first.room_id == second.room_id:
            return self._finish_attempts(
                context,
                attempts,
                selection_reason="room_consensus",
                requires_pose_realign=True,
                prefer_consensus=True,
            )

        if second.mask_found and not second.touches_top:
            return self._finish_attempts(
                context,
                attempts,
                selection_reason="backoff_mask_complete_best_confidence",
                requires_pose_realign=True,
            )

        if not self.settings.pitch_compensation or context.should_abort():
            return self._finish_attempts(
                context,
                attempts,
                selection_reason="pitch_disabled_best_confidence",
                requires_pose_realign=True,
            )

        context.worker.frame_log(
            f"[NandaMatch] 第二阶段 mask_found={second.mask_found}，"
            f"touches_top={second.touches_top}，启动一次轻微抬头终检"
        )
        third = self._run_match_attempt(
            context,
            index=3,
            label="slight_pitch",
            apply_pitch=True,
        )
        attempts.append(third)
        return self._finish_attempts(
            context,
            attempts,
            selection_reason="highest_confidence_after_three_attempts",
            requires_pose_realign=True,
            prefer_consensus=True,
        )


@dataclass(frozen=True)
class NandaJoystickReplayStep:
    timestamp: float
    move_direction: int
    moving: bool

    @property
    def is_idle(self) -> bool:
        return not self.moving


def parse_nanda_joystick_replay(
    raw: Sequence[Mapping[str, Any]],
    source: str = "南大 replay_steps",
) -> List[NandaJoystickReplayStep]:
    """验证新版 DSL；任何非摇杆有效动作都会拒绝执行。"""
    if not isinstance(raw, list):
        raise ValueError(f"{source} 必须是列表")

    by_timestamp = OrderedDict()
    previous_timestamp = None
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"回放第 {index} 项不是对象")
        try:
            timestamp = float(item["time"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"回放第 {index} 项 time 无效")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"回放第 {index} 项 time 必须是非负有限数")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError(f"回放第 {index} 项时间戳不是单调递增")
        previous_timestamp = timestamp

        actions = item.get("actions")
        if not isinstance(actions, dict):
            raise ValueError(f"回放第 {index} 项 actions 无效")
        try:
            do_move = int(actions.get("do_move", 0))
        except (TypeError, ValueError):
            raise ValueError(f"回放第 {index} 项 do_move 无效")
        if do_move not in (0, 1):
            raise ValueError(f"回放第 {index} 项 do_move 只能是 0/1")
        active_unsupported = []
        for action_name in _UNSUPPORTED_REPLAY_ACTIONS:
            try:
                active = int(actions.get(action_name, 0)) != 0
            except (TypeError, ValueError):
                raise ValueError(f"回放第 {index} 项 {action_name} 无效")
            if active:
                active_unsupported.append(action_name)
        if active_unsupported:
            raise ValueError(
                "HOS 单指南大回放只允许摇杆动作，发现: "
                + ", ".join(sorted(active_unsupported))
            )

        try:
            move_direction = int(item.get("move_direction", 0))
        except (TypeError, ValueError):
            raise ValueError(f"回放第 {index} 项 move_direction 无效")
        if move_direction < 0 or move_direction > 360:
            raise ValueError(f"回放第 {index} 项 move_direction 超出 [0, 360]")
        params = item.get("params", {})
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"回放第 {index} 项 params 无效")

        # 与南大 PubgDslRecord 一致：相同时间戳保留最后一项。
        by_timestamp[timestamp] = NandaJoystickReplayStep(
            timestamp=timestamp,
            move_direction=move_direction,
            moving=do_move == 1,
        )
    if not by_timestamp:
        raise ValueError(f"{source} 没有动作")
    return list(by_timestamp.values())


def load_nanda_joystick_replay(path: str) -> List[NandaJoystickReplayStep]:
    """读取并验证本地南大回放文件（兼容旧的进程内匹配器）。"""
    with open(path, "r", encoding="utf-8") as replay_file:
        raw = json.load(replay_file)
    return parse_nanda_joystick_replay(raw, source=path)


class NandaHosJoystickReplayExecutor(NandaReplayExecutor):
    """把南大 scrcpy 摇杆 DSL 复刻到 HOScrcpy 单指触控通道。"""

    _ABORT_POLL_SECONDS = 0.05
    _TOUCH_RECOVERY_POLL_SECONDS = 0.1
    _STREAM_DIAG_INTERVAL_SECONDS = 1.0
    _TRANSIENT_TOUCH_ERRORS = (
        "HOS touch is unavailable: stream device is not ready",
        "HOS touch is unavailable: capture is paused",
        "HOS touch is unavailable: device setup is incomplete",
    )

    def __init__(
        self,
        settings: NandaLatestSettings,
        touch_controller_factory: Optional[Callable[[Any], Any]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self._touch_controller_factory = touch_controller_factory
        self._monotonic = monotonic
        self._sleep = sleeper

    def _make_touch_controller(self, stream_client: Any) -> Any:
        if self._touch_controller_factory is not None:
            return self._touch_controller_factory(stream_client)
        from aw.autogame.tools.GameFrameWorker import HOSTouchController

        return HOSTouchController(stream_client)

    def _resolve_joystick_geometry(
        self,
        context: NandaSearchContext,
    ) -> Tuple[Tuple[int, int], int]:
        controller = getattr(context.worker, "controller", None)
        resolution = None
        get_resolution = getattr(controller, "_get_cached_resolution", None)
        if callable(get_resolution):
            resolution = get_resolution()

        center = None
        resolve_pos = getattr(controller, "_resolve_pos", None)
        if callable(resolve_pos):
            resolved = resolve_pos("摇杆")
            if resolved:
                center = resolved[0]

        if resolution and resolution[0] and resolution[1]:
            screen_width, screen_height = int(resolution[0]), int(resolution[1])
        else:
            frame = getattr(context.worker, "frame", None)
            if frame is None:
                frame = context.frame
            if frame is None:
                raise RuntimeError("无法取得 HOS 回放屏幕尺寸")
            screen_height, screen_width = (int(value) for value in frame.shape[:2])

        if center is None:
            center = (
                int(round(screen_width * self.settings.joystick_center_x_ratio)),
                int(round(screen_height * self.settings.joystick_center_y_ratio)),
            )
        if self.settings.joystick_radius_px > 0:
            radius = self.settings.joystick_radius_px
        else:
            radius_reference = min(screen_width, screen_height)
            radius = max(
                1,
                int(round(radius_reference * self.settings.joystick_radius_height_ratio)),
            )
        return (int(center[0]), int(center[1])), radius

    @staticmethod
    def _target_for_direction(
        center: Tuple[int, int],
        radius: int,
        direction: int,
    ) -> Tuple[int, int]:
        angle_radians = math.radians(float(direction) - 90.0)
        return (
            int(round(center[0] + radius * math.cos(angle_radians))),
            int(round(center[1] + radius * math.sin(angle_radians))),
        )

    def _wait_until(
        self,
        deadline: float,
        context: NandaSearchContext,
        periodic_callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        while True:
            if context.should_abort():
                return False
            if periodic_callback is not None:
                periodic_callback()
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return True
            self._sleep(min(self._ABORT_POLL_SECONDS, remaining))

    @staticmethod
    def _stream_touch_ready(stream_client: Any) -> bool:
        """仅对真实 HOS 流客户端检查就绪状态；测试替身默认可用。"""
        has_runtime_state = any(
            hasattr(stream_client, name)
            for name in ("device", "running", "is_capture_paused")
        )
        if not has_runtime_state:
            return True
        if hasattr(stream_client, "running") and not bool(stream_client.running):
            return False
        if getattr(stream_client, "device", None) is None:
            return False
        if getattr(stream_client, "_last_error", None) is not None:
            return False
        stream_error_event = getattr(stream_client, "_stream_error_event", None)
        if stream_error_event is not None and stream_error_event.is_set():
            return False
        is_capture_paused = getattr(stream_client, "is_capture_paused", None)
        if callable(is_capture_paused) and is_capture_paused():
            return False
        inner_device = getattr(getattr(stream_client, "device", None), "device", None)
        if inner_device is not None and not bool(getattr(inner_device, "is_setup", False)):
            return False
        return True

    def _wait_for_touch_recovery(
        self,
        context: NandaSearchContext,
        stream_client: Any,
        reason: str,
    ) -> Optional[float]:
        started_at = self._monotonic()
        timeout = self.settings.replay_touch_recovery_timeout_seconds
        context.worker.frame_log(
            f"[NandaReplay] HOS触控流暂时不可用：{reason}；"
            f"暂停回放并等待流恢复，超时={timeout:g}s"
        )
        while self._monotonic() - started_at < timeout:
            if context.should_abort():
                return None
            if self._stream_touch_ready(stream_client):
                elapsed = self._monotonic() - started_at
                context.worker.frame_log(
                    f"[NandaReplay] HOS触控流已恢复，等待={elapsed:.2f}s，"
                    "从当前摇杆方向续播"
                )
                return elapsed
            self._sleep(self._TOUCH_RECOVERY_POLL_SECONDS)
        raise RuntimeError(
            f"HOS触控流在 {timeout:g}s 内未恢复: {reason}"
        )

    def _is_transient_touch_error(self, exc: Exception) -> bool:
        message = str(exc)
        return any(marker in message for marker in self._TRANSIENT_TOUCH_ERRORS)

    @staticmethod
    def _sanitize_diagnostic_text(value: Any, limit: int = 240) -> str:
        text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
        return text[:limit] if text else "-"

    def _log_stream_diagnostic(
        self,
        context: NandaSearchContext,
        stream_client: Any,
        checkpoint: str,
        replay_started_at: float,
        *,
        step_index: int = 0,
        step_count: int = 0,
        step: Optional[NandaJoystickReplayStep] = None,
        action: str = "-",
        previous_counts: Optional[Tuple[int, int, int]] = None,
        touch_error: str = "",
    ) -> Tuple[int, int, int]:
        snapshot = {}
        snapshot_error = ""
        diagnostic_snapshot = getattr(stream_client, "diagnostic_snapshot", None)
        if callable(diagnostic_snapshot):
            try:
                raw_snapshot = diagnostic_snapshot()
                if isinstance(raw_snapshot, dict):
                    snapshot = raw_snapshot
            except Exception as exc:
                snapshot_error = f"diagnostic_snapshot failed: {exc}"

        callback_count = int(
            snapshot.get(
                "callback_data_count",
                getattr(stream_client, "_callback_data_count", 0),
            )
            or 0
        )
        decoded_count = int(
            snapshot.get(
                "decoded_frame_count",
                getattr(stream_client, "_decoded_frame_count", 0),
            )
            or 0
        )
        frame_buffer = getattr(stream_client, "buffer", None)
        buffer_count = int(getattr(frame_buffer, "count", 0) or 0)
        counts = (callback_count, decoded_count, buffer_count)

        if previous_counts is None:
            count_text = (
                f"callback={callback_count}(delta=-) "
                f"decoded={decoded_count}(delta=-) "
                f"buffer={buffer_count}(delta=-)"
            )
        else:
            count_text = (
                f"callback={callback_count}(delta={callback_count - previous_counts[0]:+d}) "
                f"decoded={decoded_count}(delta={decoded_count - previous_counts[1]:+d}) "
                f"buffer={buffer_count}(delta={buffer_count - previous_counts[2]:+d})"
            )

        now = self._monotonic()

        def age_text(timestamp: Any) -> str:
            try:
                if timestamp is None:
                    return "-"
                return f"{max(0.0, now - float(timestamp)):.2f}s"
            except (TypeError, ValueError):
                return "-"

        last_error = snapshot_error or snapshot.get("last_error", "")
        direction = step.move_direction if step is not None else "-"
        record_time = f"{step.timestamp:.3f}s" if step is not None else "-"
        context.worker.frame_log(
            f"[NandaStreamDiag] checkpoint={checkpoint} "
            f"elapsed={max(0.0, now - replay_started_at):.2f}s "
            f"step={step_index}/{step_count} record_time={record_time} "
            f"action={action} direction={direction} "
            f"stream_stage={snapshot.get('stage', '-')} "
            f"running={snapshot.get('running', getattr(stream_client, 'running', '-'))} "
            f"device_ready={self._stream_touch_ready(stream_client)} "
            f"capture_paused={snapshot.get('capture_paused', '-')} "
            f"touch_replay={snapshot.get('touch_replay_active', '-')} "
            f"touch_replay_waiting={snapshot.get('touch_replay_waiting_fresh_frame', '-')} "
            f"touch_replay_deferred={self._sanitize_diagnostic_text(snapshot.get('touch_replay_deferred_error', ''))} "
            f"{count_text} "
            f"last_data_age={age_text(getattr(stream_client, '_last_data_at', None))} "
            f"last_decoded_age={age_text(getattr(stream_client, '_last_decoded_at', None))} "
            f"reconnect={snapshot.get('reconnect_attempt', '-')}/"
            f"{snapshot.get('max_reconnect_attempts', '-')} "
            f"last_error={self._sanitize_diagnostic_text(last_error)} "
            f"touch_error={self._sanitize_diagnostic_text(touch_error)}"
        )
        return counts

    @staticmethod
    def _close_touch(touch: Any) -> None:
        close = getattr(touch, "close", None)
        if callable(close):
            close()

    def replay(
        self,
        context: NandaSearchContext,
        match: NandaRoomMatch,
    ) -> NandaSearchResult:
        if match.replay_steps is not None:
            steps = parse_nanda_joystick_replay(
                match.replay_steps,
                source=f"room={match.room_id} replay_steps",
            )
        else:
            steps = load_nanda_joystick_replay(match.replay_path)
        stream_client = getattr(context.worker, "stream_client", None)
        if stream_client is None:
            raise RuntimeError("当前 FrameWorker 没有可用的 HOScrcpy 触控流")
        center, radius = self._resolve_joystick_geometry(context)
        started_at = self._monotonic()
        diagnostic_counts = self._log_stream_diagnostic(
            context,
            stream_client,
            "start",
            started_at,
            step_count=len(steps),
            action="prepare",
        )
        last_diagnostic_at = started_at
        if not self._stream_touch_ready(stream_client):
            recovered = self._wait_for_touch_recovery(
                context,
                stream_client,
                "回放开始前 stream device is not ready",
            )
            if recovered is None:
                self._log_stream_diagnostic(
                    context,
                    stream_client,
                    "aborted",
                    started_at,
                    step_count=len(steps),
                    action="wait_stream_before_start",
                    previous_counts=diagnostic_counts,
                )
                return NandaSearchResult(
                    NandaSearchStatus.ABORTED,
                    "等待 HOS 触控流恢复时搜房阶段已中止",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={"phase": "replay"},
                )
            diagnostic_counts = self._log_stream_diagnostic(
                context,
                stream_client,
                "recovered",
                started_at,
                step_count=len(steps),
                action="stream_before_start",
                previous_counts=diagnostic_counts,
            )
            last_diagnostic_at = self._monotonic()
        touch = self._make_touch_controller(stream_client)
        begin_touch_replay = getattr(stream_client, "begin_touch_replay", None)
        end_touch_replay = getattr(stream_client, "end_touch_replay", None)
        replay_guard_started = False
        replay_guard_stream_healthy = None
        if callable(begin_touch_replay):
            replay_guard_started = bool(
                begin_touch_replay(f"Nanda room={match.room_id}")
            )
            if not replay_guard_started:
                raise RuntimeError("无法启用 HOS 南大回放抓流保护期")
            context.worker.frame_log(
                "[NandaReplay] 已启用 HOS 回放抓流保护："
                "回放中普通视频异常延迟确认，回放结束以新解码帧决定是否恢复流"
            )
        current_moving = False
        current_direction = 0
        previous_record_time = 0.0
        previous_was_idle = True
        scheduled_at = self._monotonic()
        context.worker.frame_log(
            f"[NandaReplay] 开始 HOS 单摇杆回放：room={match.room_id}，"
            f"steps={len(steps)}，center={center}，radius={radius}"
        )
        try:
            for step_index, step in enumerate(steps, start=1):
                if step.moving:
                    if not current_moving:
                        action = "press"
                    elif step.move_direction != current_direction:
                        action = "move"
                    else:
                        action = "hold"
                else:
                    action = "up" if current_moving else "idle"

                def log_periodic() -> None:
                    nonlocal diagnostic_counts, last_diagnostic_at
                    now = self._monotonic()
                    if now - last_diagnostic_at < self._STREAM_DIAG_INTERVAL_SECONDS:
                        return
                    diagnostic_counts = self._log_stream_diagnostic(
                        context,
                        stream_client,
                        "periodic",
                        started_at,
                        step_index=step_index,
                        step_count=len(steps),
                        step=step,
                        action=f"waiting->{action}",
                        previous_counts=diagnostic_counts,
                    )
                    last_diagnostic_at = now

                wait_seconds = step.timestamp - previous_record_time
                previous_record_time = step.timestamp
                skip_wait = self.settings.replay_skip_idle and previous_was_idle
                previous_was_idle = step.is_idle
                if skip_wait or wait_seconds < 0.001:
                    scheduled_at = self._monotonic()
                else:
                    scheduled_at += wait_seconds
                    if not self._wait_until(
                        scheduled_at,
                        context,
                        periodic_callback=log_periodic,
                    ):
                        self._log_stream_diagnostic(
                            context,
                            stream_client,
                            "aborted",
                            started_at,
                            step_index=step_index,
                            step_count=len(steps),
                            step=step,
                            action=f"waiting->{action}",
                            previous_counts=diagnostic_counts,
                        )
                        return NandaSearchResult(
                            NandaSearchStatus.ABORTED,
                            "南大摇杆回放被搜房阶段中止",
                            room_id=match.room_id,
                            replay_path=match.replay_path,
                            metadata={"phase": "replay"},
                        )

                if context.should_abort():
                    self._log_stream_diagnostic(
                        context,
                        stream_client,
                        "aborted",
                        started_at,
                        step_index=step_index,
                        step_count=len(steps),
                        step=step,
                        action=action,
                        previous_counts=diagnostic_counts,
                    )
                    return NandaSearchResult(
                        NandaSearchStatus.ABORTED,
                        "南大摇杆回放被搜房阶段中止",
                        room_id=match.room_id,
                        replay_path=match.replay_path,
                        metadata={"phase": "replay"},
                    )
                try:
                    if step.moving:
                        target = self._target_for_direction(center, radius, step.move_direction)
                        if not current_moving:
                            touch.move_press(0, target)
                        elif step.move_direction != current_direction:
                            touch.move_to(
                                0,
                                target,
                                duration_ms=self.settings.joystick_slide_duration_ms,
                            )
                        current_moving = True
                        current_direction = step.move_direction
                    elif current_moving:
                        touch.move_up(0)
                        current_moving = False
                except RuntimeError as exc:
                    diagnostic_counts = self._log_stream_diagnostic(
                        context,
                        stream_client,
                        "touch_error",
                        started_at,
                        step_index=step_index,
                        step_count=len(steps),
                        step=step,
                        action=action,
                        previous_counts=diagnostic_counts,
                        touch_error=str(exc),
                    )
                    last_diagnostic_at = self._monotonic()
                    if not self._is_transient_touch_error(exc):
                        raise
                    self._close_touch(touch)
                    if replay_guard_started and callable(end_touch_replay):
                        replay_guard_stream_healthy = bool(end_touch_replay())
                        replay_guard_started = False
                    recovered = self._wait_for_touch_recovery(
                        context,
                        stream_client,
                        str(exc),
                    )
                    if recovered is None:
                        self._log_stream_diagnostic(
                            context,
                            stream_client,
                            "aborted",
                            started_at,
                            step_index=step_index,
                            step_count=len(steps),
                            step=step,
                            action="wait_touch_recovery",
                            previous_counts=diagnostic_counts,
                            touch_error=str(exc),
                        )
                        return NandaSearchResult(
                            NandaSearchStatus.ABORTED,
                            "等待 HOS 触控流恢复时搜房阶段已中止",
                            room_id=match.room_id,
                            replay_path=match.replay_path,
                            metadata={"phase": "replay"},
                        )
                    diagnostic_counts = self._log_stream_diagnostic(
                        context,
                        stream_client,
                        "recovered",
                        started_at,
                        step_index=step_index,
                        step_count=len(steps),
                        step=step,
                        action="retry_current_direction",
                        previous_counts=diagnostic_counts,
                    )
                    last_diagnostic_at = self._monotonic()
                    scheduled_at += recovered
                    touch = self._make_touch_controller(stream_client)
                    if callable(begin_touch_replay):
                        replay_guard_started = bool(
                            begin_touch_replay(f"Nanda room={match.room_id} resumed")
                        )
                        if not replay_guard_started:
                            raise RuntimeError("HOS 触控流恢复后无法重新启用抓流保护期")
                    current_moving = False
                    if step.moving:
                        target = self._target_for_direction(center, radius, step.move_direction)
                        touch.move_press(0, target)
                        current_moving = True
                        current_direction = step.move_direction
        finally:
            try:
                touch.move_up(0)
            except RuntimeError as exc:
                if not self._is_transient_touch_error(exc):
                    raise
                diagnostic_counts = self._log_stream_diagnostic(
                    context,
                    stream_client,
                    "cleanup_error",
                    started_at,
                    step_index=len(steps),
                    step_count=len(steps),
                    step=steps[-1] if steps else None,
                    action="up",
                    previous_counts=diagnostic_counts,
                    touch_error=str(exc),
                )
                context.worker.frame_log(
                    f"[NandaReplay] 回放收尾时 HOS 流已断开，无法发送抬指：{exc}；"
                    "设备触控会话已随流断开，继续关闭本地回放控制器"
                )
            finally:
                self._close_touch(touch)
                if replay_guard_started and callable(end_touch_replay):
                    replay_guard_stream_healthy = bool(end_touch_replay())
                    if replay_guard_stream_healthy:
                        context.worker.frame_log(
                            "[NandaReplay] HOS 回放结束后已收到新帧，抓流保持正常"
                        )
                    else:
                        context.worker.frame_log(
                            "[NandaReplay] HOS 回放结束后未收到新帧，"
                            "已触发流内重连；用例将在刷新帧处等待恢复"
                        )

        elapsed = self._monotonic() - started_at
        self._log_stream_diagnostic(
            context,
            stream_client,
            "end",
            started_at,
            step_index=len(steps),
            step_count=len(steps),
            step=steps[-1] if steps else None,
            action="completed",
            previous_counts=diagnostic_counts,
        )
        context.worker.frame_log(
            f"[NandaReplay] HOS 单摇杆回放结束：room={match.room_id}，elapsed={elapsed:.2f}s"
        )
        return NandaSearchResult.completed(
            match,
            message="南大 HOS 单摇杆回放执行完成",
            metadata={
                "phase": "replay",
                "backend": "hos_single_touch",
                "step_count": len(steps),
                "elapsed_seconds": elapsed,
                "source_score": match.score,
                "stream_guard_healthy": replay_guard_stream_healthy,
            },
        )


def build_nanda_house_search_strategy(
    autogame_config: Optional[Mapping[str, Any]],
) -> NandaHouseSearchStrategy:
    raw_config = dict(autogame_config or {})
    section = raw_config.get("nanda_house_search", raw_config)
    if not isinstance(section, dict):
        section = {}
    settings = NandaLatestSettings.from_mapping(section)
    if not settings.enabled:
        return NandaHouseSearchStrategy()
    if settings.door_pose_backend != "entry_direction_yolo":
        raise ValueError(
            "南大门前位姿当前只允许 entry_direction_yolo，禁止接入 SAM3 门校准"
        )
    if settings.room_segmenter_backend != "sam3_special_area":
        raise ValueError(
            "南大房型配准当前只允许 sam3_special_area 输入"
        )
    matcher = NandaLocalRoomMatcher(settings)
    return NandaHouseSearchStrategy(
        matcher=matcher,
        replay_executor=NandaHosJoystickReplayExecutor(settings),
        pose_preparer=NandaYoloDoorPosePreparer(settings),
        exclusive=True,
    )


__all__ = [
    "NandaDoorPosePreparer",
    "NandaHosJoystickReplayExecutor",
    "NandaLocalRoomMatcher",
    "NandaJoystickReplayStep",
    "NandaLatestSettings",
    "NandaYoloDoorPosePreparer",
    "build_nanda_house_search_strategy",
    "load_nanda_joystick_replay",
    "parse_nanda_joystick_replay",
]
