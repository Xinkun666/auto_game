"""南大最新版房型匹配与单摇杆回放的 auto_game 适配实现。

匹配器通过独立 HTTP 服务调用南大 Python 3.11 环境；门前位姿只使用已有
入门点方向和 YOLO 门框，不调用 SAM3。房型配准直接消费搜房阶段
``get_info("sam3")`` 产生的 special_area 分割结果，不再重复推理。
真正的房屋回放使用当前 HOScrcpy 流的单指触控通道，只复现
最新版房型库中唯一实际使用的 ``do_move`` 动作。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import json
import math
import os
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

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
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    plan_view_turn_motion,
)


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
    matcher_url: str = "http://127.0.0.1:7789/match"
    matcher_health_timeout_seconds: float = 0.35
    matcher_health_cache_seconds: float = 2.0
    matcher_timeout_seconds: float = 180.0
    jpeg_quality: int = 92

    max_entry_distance: float = 2.5
    direction_tolerance_degrees: float = 3.0
    area_min_ratio: float = 0.02
    area_max_ratio: float = 0.04
    area_acceptable_min_ratio: float = 0.015
    area_acceptable_max_ratio: float = 0.055
    coarse_center_ratio: float = 0.04
    final_center_ratio: float = 0.01
    acceptable_center_ratio: float = 0.03
    stable_required_count: int = 2
    max_pose_actions: int = 18
    move_axis_bias: int = 240
    pose_min_duration_ms: int = 60
    pose_max_duration_ms: int = 600
    pose_wait_ms: int = 500

    pitch_compensation: bool = True
    pitch_pixels_per_second: float = 1000.0
    pitch_max_seconds: float = 0.8
    pitch_wait_ms: int = 450

    joystick_center_x_ratio: float = 0.1965
    joystick_center_y_ratio: float = 0.7563
    joystick_radius_height_ratio: float = 0.0974
    joystick_slide_duration_ms: int = 100
    replay_skip_idle: bool = True

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
            matcher_url=str(
                os.environ.get("AUTOGAME_NANDA_MATCHER_URL", "").strip()
                or raw.get("matcher_url")
                or cls.matcher_url
            ),
            matcher_health_timeout_seconds=max(
                0.05,
                _as_float(
                    raw.get("matcher_health_timeout_seconds"),
                    cls.matcher_health_timeout_seconds,
                ),
            ),
            matcher_health_cache_seconds=max(
                0.0,
                _as_float(
                    raw.get("matcher_health_cache_seconds"),
                    cls.matcher_health_cache_seconds,
                ),
            ),
            matcher_timeout_seconds=max(
                1.0,
                _as_float(raw.get("matcher_timeout_seconds"), cls.matcher_timeout_seconds),
            ),
            jpeg_quality=max(50, min(100, _as_int(raw.get("jpeg_quality"), cls.jpeg_quality))),
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
            coarse_center_ratio=max(
                0.001,
                _as_float(raw.get("coarse_center_ratio"), cls.coarse_center_ratio),
            ),
            final_center_ratio=max(
                0.001,
                _as_float(raw.get("final_center_ratio"), cls.final_center_ratio),
            ),
            acceptable_center_ratio=max(
                0.001,
                _as_float(raw.get("acceptable_center_ratio"), cls.acceptable_center_ratio),
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
            pitch_compensation=_as_bool(
                raw.get("pitch_compensation"), cls.pitch_compensation
            ),
            pitch_pixels_per_second=max(
                1.0,
                _as_float(
                    raw.get("pitch_pixels_per_second"), cls.pitch_pixels_per_second
                ),
            ),
            pitch_max_seconds=max(
                0.0, _as_float(raw.get("pitch_max_seconds"), cls.pitch_max_seconds)
            ),
            pitch_wait_ms=max(0, _as_int(raw.get("pitch_wait_ms"), cls.pitch_wait_ms)),
            joystick_center_x_ratio=_as_float(
                raw.get("joystick_center_x_ratio"), cls.joystick_center_x_ratio
            ),
            joystick_center_y_ratio=_as_float(
                raw.get("joystick_center_y_ratio"), cls.joystick_center_y_ratio
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
            "门前位姿多次调整仍未进入南大方案可接受范围，退回原搜房策略",
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

        # 新版南大校准先用 4% 粗阈值横移人物，再用门框面积调整距离。
        if not relaxed_accept and center_error > self.settings.coarse_center_ratio:
            duration = self._duration_for_error(
                center_error,
                self.settings.coarse_center_ratio,
                0.10,
            )
            side = 1 if center_delta > 0 else -1
            return self._retry_after_action(
                context,
                f"门中心偏差 {center_delta:+.3f}，横移人物做粗对中",
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

        # 距离收敛后，以 1% 的最新版最终阈值再次横移人物对中。
        if not relaxed_accept and center_error > self.settings.final_center_ratio:
            duration = self._duration_for_error(
                center_error,
                self.settings.final_center_ratio,
                0.03,
            )
            side = 1 if center_delta > 0 else -1
            return self._retry_after_action(
                context,
                f"距离已稳定，门中心偏差 {center_delta:+.3f}，做最终横移对中",
                x_bias=side * self.settings.move_axis_bias,
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


class NandaHttpRoomMatcher(NandaRoomMatcher):
    """把已有 SAM3 special_area 分割结果交给南大房型配准服务。"""

    def __init__(
        self,
        settings: NandaLatestSettings,
        opener: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self._opener = opener or urlopen
        self._last_health_at = 0.0
        self._last_health_ok = False
        self.unavailable_reason = "南大房型匹配服务尚未启动"

    def reset(self) -> None:
        self._last_health_at = 0.0
        self._last_health_ok = False

    def _health_url(self) -> str:
        parts = urlsplit(self.settings.matcher_url)
        return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    def _open_json(self, request: Request, timeout: float) -> Mapping[str, Any]:
        response = self._opener(request, timeout=timeout)
        try:
            payload = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("南大匹配服务返回的 JSON 不是对象")
        return decoded

    def is_available(self) -> bool:
        now = time.monotonic()
        if self._last_health_at > 0.0 and (
            now - self._last_health_at <= self.settings.matcher_health_cache_seconds
        ):
            return self._last_health_ok
        self._last_health_at = now
        try:
            payload = self._open_json(
                Request(self._health_url(), method="GET"),
                timeout=self.settings.matcher_health_timeout_seconds,
            )
            self._last_health_ok = payload.get("status") == "ok" and bool(
                payload.get("ready", True)
            )
            actual_input_contract = str(
                payload.get("input_contract") or ""
            ).strip()
            if actual_input_contract != self.settings.room_segmenter_backend:
                self._last_health_ok = False
                self.unavailable_reason = (
                    "南大房型匹配服务输入协议不一致: "
                    f"expected={self.settings.room_segmenter_backend}, "
                    f"actual={actual_input_contract or 'unknown'}"
                )
                return False
            self.unavailable_reason = str(
                payload.get("message") or "南大房型匹配服务尚未就绪"
            )
        except (OSError, ValueError, HTTPError, URLError) as exc:
            self._last_health_ok = False
            self.unavailable_reason = f"南大房型匹配服务不可用: {exc}"
        return self._last_health_ok

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
    ) -> Tuple[np.ndarray, Mapping[str, Any]]:
        frame = context.frame
        if frame is None:
            raise ValueError("当前没有可用于房型匹配的画面")
        pitch_seconds = 0.0
        if self.settings.pitch_compensation and context.door_box is not None:
            frame_h = max(1, int(frame.shape[0]))
            door_top_ratio = float(context.door_box[1]) / float(frame_h)
            pitch_seconds = min(
                self.settings.pitch_max_seconds,
                max(0.0, (0.5 - door_top_ratio) * 2.0),
            )
        pitch_bias = 0
        duration_ms = 0
        pitch_applied = False
        original_group = None
        try:
            if pitch_seconds > 0.0:
                pitch_bias = int(
                    round(pitch_seconds * self.settings.pitch_pixels_per_second)
                )
                duration_ms = max(100, int(round(pitch_seconds * 1000.0)))
                context.worker.frame_log(
                    f"[NandaMatch] 匹配前抬高视角 {pitch_seconds:.2f}s，完整采集门面"
                )
                context.worker.tap_single(
                    "视角",
                    y_bias=-pitch_bias,
                    dura=duration_ms,
                    wait=self.settings.pitch_wait_ms,
                )
                pitch_applied = True
                if not context.refresh_frame("NandaMatch 抬高视角采集房屋正面"):
                    raise ValueError("抬高视角后刷新画面失败")

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
                    dura=duration_ms,
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

    @classmethod
    def _build_match_payload(
        cls,
        frame: np.ndarray,
        sam3_info: Mapping[str, Any],
    ) -> bytes:
        segmented_bgr, cropped_mask, cropped_bgr, crop_xyxy = (
            cls._special_area_facade(frame, sam3_info)
        )
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            segmented_bgr=segmented_bgr,
            cropped_mask=cropped_mask,
            cropped_bgr=cropped_bgr,
            crop_xyxy=np.asarray(crop_xyxy, dtype=np.int32),
            sam3_score=np.asarray(
                [float(sam3_info.get("score") or 0.0)], dtype=np.float32
            ),
            sam3_inference_ms=np.asarray(
                [float(sam3_info.get("sam3_inference_ms") or 0.0)],
                dtype=np.float32,
            ),
        )
        return payload.getvalue()

    def match(self, context: NandaSearchContext) -> Optional[NandaRoomMatch]:
        if context.should_abort():
            return None
        try:
            frame, sam3_info = self._capture_match_frame(context)
            request_body = self._build_match_payload(frame, sam3_info)
        except ValueError as exc:
            context.worker.frame_log(f"[NandaMatch] {exc}，本轮回退原搜房流程")
            return None
        request = Request(
            self.settings.matcher_url,
            data=request_body,
            headers={
                "Content-Type": "application/x-npz",
                "X-Nanda-House-Id": str(context.house_id or ""),
                "X-Nanda-Door-Pose-Backend": self.settings.door_pose_backend,
                "X-Nanda-Input-Contract": self.settings.room_segmenter_backend,
            },
            method="POST",
        )
        payload = self._open_json(request, timeout=self.settings.matcher_timeout_seconds)
        status = str(payload.get("status") or "")
        if status == "no_match":
            context.worker.frame_log(
                f"[NandaMatch] 未通过最新版房型阈值: {payload.get('reason') or 'unknown'}"
            )
            return None
        if status != "matched":
            raise RuntimeError(str(payload.get("message") or f"匹配服务状态异常: {status}"))

        room_id = str(payload.get("room_id") or "").strip()
        replay_path = str(payload.get("replay_path") or "").strip()
        replay_steps = payload.get("replay_steps")
        if not room_id or not replay_path:
            raise ValueError("匹配服务没有返回 room_id/replay_path")
        if not isinstance(replay_steps, list):
            raise ValueError("匹配服务没有返回 replay_steps")
        score_raw = payload.get("score")
        score = None if score_raw is None else float(score_raw)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        context.worker.frame_log(
            f"[NandaMatch] 房型匹配成功：room={room_id}，score={score}"
        )
        return NandaRoomMatch(
            room_id=room_id,
            replay_path=replay_path,
            score=score,
            metadata=metadata,
            replay_steps=replay_steps,
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

    def _wait_until(self, deadline: float, context: NandaSearchContext) -> bool:
        while True:
            if context.should_abort():
                return False
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return True
            self._sleep(min(self._ABORT_POLL_SECONDS, remaining))

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
        touch = self._make_touch_controller(stream_client)
        current_moving = False
        current_direction = 0
        previous_record_time = 0.0
        previous_was_idle = True
        scheduled_at = self._monotonic()
        started_at = scheduled_at
        context.worker.frame_log(
            f"[NandaReplay] 开始 HOS 单摇杆回放：room={match.room_id}，"
            f"steps={len(steps)}，center={center}，radius={radius}"
        )
        try:
            for step in steps:
                wait_seconds = step.timestamp - previous_record_time
                previous_record_time = step.timestamp
                skip_wait = self.settings.replay_skip_idle and previous_was_idle
                previous_was_idle = step.is_idle
                if skip_wait or wait_seconds < 0.001:
                    scheduled_at = self._monotonic()
                else:
                    scheduled_at += wait_seconds
                    if not self._wait_until(scheduled_at, context):
                        return NandaSearchResult(
                            NandaSearchStatus.ABORTED,
                            "南大摇杆回放被搜房阶段中止",
                            room_id=match.room_id,
                            replay_path=match.replay_path,
                            metadata={"phase": "replay"},
                        )

                if context.should_abort():
                    return NandaSearchResult(
                        NandaSearchStatus.ABORTED,
                        "南大摇杆回放被搜房阶段中止",
                        room_id=match.room_id,
                        replay_path=match.replay_path,
                        metadata={"phase": "replay"},
                    )
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
        finally:
            try:
                touch.move_up(0)
            finally:
                close = getattr(touch, "close", None)
                if callable(close):
                    close()

        elapsed = self._monotonic() - started_at
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
    matcher = NandaHttpRoomMatcher(settings)
    return NandaHouseSearchStrategy(
        matcher=matcher,
        replay_executor=NandaHosJoystickReplayExecutor(settings),
        pose_preparer=NandaYoloDoorPosePreparer(settings),
    )


__all__ = [
    "NandaDoorPosePreparer",
    "NandaHosJoystickReplayExecutor",
    "NandaHttpRoomMatcher",
    "NandaJoystickReplayStep",
    "NandaLatestSettings",
    "NandaYoloDoorPosePreparer",
    "build_nanda_house_search_strategy",
    "load_nanda_joystick_replay",
    "parse_nanda_joystick_replay",
]
