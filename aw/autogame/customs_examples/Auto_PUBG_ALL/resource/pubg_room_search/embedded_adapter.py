from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.config import (
    get_pubg_room_search_config,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.frame_adapter import (
    AutoGameRoomPicCapture,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3_embedded import (
    get_sam3_perception,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.yolo_embedded import (
    get_yolo_perception,
)


@dataclass
class EmbeddedHouseSearchRunResult:
    ok: bool
    result_name: str
    fallback_to_legacy: bool
    reason: str = ""


class EmbeddedHouseSearchAdapter:
    """Run the room-search handoff inside auto_game instead of pubg_test services."""

    def __init__(self, worker, config: Optional[Dict[str, Any]] = None):
        self.worker = worker
        self.config = config or get_pubg_room_search_config()
        self.capture = AutoGameRoomPicCapture(
            worker,
            control_proxy=None,
            frame_color=self.config.get("frame_color", "rgb"),
            refresh_interval_sec=float(
                self.config.get("frame_refresh_interval_sec", 0.12)
            ),
            refresh_mode=self.config.get("frame_refresh_mode", "worker_refresh"),
        )

    @classmethod
    def from_config(cls, worker) -> Optional["EmbeddedHouseSearchAdapter"]:
        config = get_pubg_room_search_config()
        if not config.get("enabled", False):
            return None
        if not config.get("embedded_enabled", True):
            return None
        return cls(worker, config)

    def search_from_door_front(
        self,
        *,
        source: str,
        enter_after_refine: Optional[bool] = None,
    ) -> EmbeddedHouseSearchRunResult:
        try:
            refined = self._refine_door_alignment()
            if not refined:
                return self._fallback("DOOR_REFINE_FAILED", "门定位/微调失败")

            replay_searcher_class = self._load_replay_searcher_class()
            if replay_searcher_class is None:
                return self._fallback(
                    "REPLAY_CORE_NOT_READY",
                    "内嵌房型匹配/回放核心尚未迁入，已回退旧搜房逻辑",
                )

            should_enter = (
                bool(self.config.get("embedded_enter_after_refine", True))
                if enter_after_refine is None
                else bool(enter_after_refine)
            )
            if should_enter:
                self._interact_and_enter()

            return self._run_embedded_replay_core(
                replay_searcher_class,
                source=source,
            )
        except Exception as exc:
            return self._fallback("EXCEPTION", str(exc))

    def _fallback(self, result_name: str, reason: str) -> EmbeddedHouseSearchRunResult:
        return EmbeddedHouseSearchRunResult(
            ok=False,
            result_name=result_name,
            fallback_to_legacy=bool(self.config.get("embedded_allow_legacy_fallback", True)),
            reason=reason,
        )

    def _refine_door_alignment(self) -> bool:
        attempts = max(1, int(self.config.get("embedded_door_refine_attempts", 3)))
        tolerance = float(self.config.get("embedded_door_center_tolerance_px", 80))
        view_scale = float(self.config.get("embedded_door_view_scale", 0.33))

        for attempt in range(1, attempts + 1):
            frame = self.capture.get_current_frame(force_refresh=True)
            if frame is None:
                print("[EmbeddedRoomSearch] 无法获取画面，门定位失败")
                return False

            bbox = self._locate_door_bbox(frame)
            if bbox is None:
                print(f"[EmbeddedRoomSearch] 未定位到门 attempt={attempt}/{attempts}")
                continue

            frame_h, frame_w = frame.shape[:2]
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            offset = center_x - (frame_w / 2.0)
            print(
                f"[EmbeddedRoomSearch] 门定位成功 offset={offset:.1f}, "
                f"bbox={tuple(int(v) for v in bbox)}"
            )
            if abs(offset) <= tolerance:
                return True

            bias = int(max(-450, min(450, offset * view_scale)))
            self.worker.tap_single("视角", x_bias=bias, dura=500, wait=450)
            self.worker.refresh_frame()

        return False

    def _locate_door_bbox(self, frame) -> Optional[Sequence[float]]:
        bbox = self._locate_door_by_sam3(frame)
        if bbox is not None:
            return bbox
        return self._locate_door_by_yolo(frame)

    def _locate_door_by_sam3(self, frame) -> Optional[Sequence[float]]:
        try:
            result = get_sam3_perception().segment_door(frame)
        except Exception as exc:
            print(f"[EmbeddedRoomSearch] SAM3 门定位不可用: {exc}")
            return None
        if result is None:
            return None
        return result.bbox_xyxy

    def _locate_door_by_yolo(self, frame) -> Optional[Sequence[float]]:
        try:
            result = get_yolo_perception().detect(frame)
        except Exception as exc:
            print(f"[EmbeddedRoomSearch] YOLO 门定位不可用: {exc}")
            return None
        candidates = [
            item
            for item in result.detections
            if "door" in str(item.class_name).lower()
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item.bbox_area())
        h, w = frame.shape[:2]
        return [
            best.bbox.x1_norm * w,
            best.bbox.y1_norm * h,
            best.bbox.x2_norm * w,
            best.bbox.y2_norm * h,
        ]

    def _interact_and_enter(self) -> None:
        self.worker.refresh_frame()
        if self.worker.get_info("开门"):
            print("[EmbeddedRoomSearch] 检测到开门按钮，开门后进入")
            self.worker.click("开门")
            time.sleep(0.7)
        elif self.worker.get_info("关门"):
            print("[EmbeddedRoomSearch] 门处于打开状态，直接进入")

        duration = int(self.config.get("embedded_enter_move_duration_ms", 360))
        wait = int(self.config.get("embedded_enter_wait_ms", 900))
        self.worker.tap_single("摇杆", y_bias=-300, dura=duration, wait=wait)
        self.worker.refresh_frame()

    def _load_replay_searcher_class(self):
        try:
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.core.replay_searcher import (
                AutoGameReplayRoomSearcher,
            )
        except ImportError:
            return None
        return AutoGameReplayRoomSearcher

    def _run_embedded_replay_core(
        self,
        replay_searcher_class,
        *,
        source: str,
    ) -> EmbeddedHouseSearchRunResult:
        searcher = replay_searcher_class(self.worker, self.config)
        result = searcher.search_current_house(source=source)
        if isinstance(result, EmbeddedHouseSearchRunResult):
            return result
        result_name = getattr(result, "result_name", str(result))
        ok = bool(getattr(result, "ok", False))
        fallback = bool(getattr(result, "fallback_to_legacy", not ok))
        reason = str(getattr(result, "reason", ""))
        return EmbeddedHouseSearchRunResult(
            ok=ok,
            result_name=result_name,
            fallback_to_legacy=fallback,
            reason=reason,
        )
