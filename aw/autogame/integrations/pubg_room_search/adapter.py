from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aw.autogame.integrations.pubg_room_search.action_adapter import AutoGameActionProxy
from aw.autogame.integrations.pubg_room_search.config import (
    ensure_pubg_test_import_path,
    get_pubg_room_search_config,
)
from aw.autogame.integrations.pubg_room_search.frame_adapter import AutoGameRoomPicCapture


@dataclass
class HouseSearchRunResult:
    ok: bool
    result_name: str
    fallback_to_legacy: bool
    reason: str = ""


class HouseSearchAdapter:
    """Bridge auto_game's frame/control APIs to pubg_test room search."""

    def __init__(self, worker, config: Optional[Dict[str, Any]] = None):
        self.worker = worker
        self.config = config or get_pubg_room_search_config()
        self.action_proxy = AutoGameActionProxy(worker, self.config)
        self.room_pic_capture = AutoGameRoomPicCapture(
            worker,
            self.action_proxy,
            frame_color=self.config.get("frame_color", "rgb"),
            refresh_interval_sec=float(
                self.config.get("frame_refresh_interval_sec", 0.12)
            ),
            refresh_mode=self.config.get("frame_refresh_mode", "worker_refresh"),
        )
        self._explorer = None
        self._frame_pump_stop = threading.Event()
        self._frame_pump_thread: Optional[threading.Thread] = None

    @classmethod
    def from_config(cls, worker) -> Optional["HouseSearchAdapter"]:
        config = get_pubg_room_search_config()
        if not config.get("enabled", False):
            return None
        return cls(worker, config)

    def search_current_house(self) -> HouseSearchRunResult:
        if not self.config.get("enabled", False):
            return HouseSearchRunResult(
                ok=False,
                result_name="DISABLED",
                fallback_to_legacy=True,
                reason="pubg_room_search disabled",
            )

        try:
            self._ensure_imports()
            self._release_existing_controls()
            explorer = self._get_or_create_explorer()
            explorer.on_video_ready()
            self._prime_frame(explorer)
            self._start_frame_pump(explorer)
            future = explorer.start()
            if future is None:
                return HouseSearchRunResult(
                    ok=False,
                    result_name="START_FAILED",
                    fallback_to_legacy=self._fallback_enabled(),
                    reason="pubg_test explorer did not start",
                )
            result = future.result()
            result_name = getattr(result, "name", str(result))
            return HouseSearchRunResult(
                ok=result_name == "SUCCESS",
                result_name=result_name,
                fallback_to_legacy=self._should_fallback(result_name),
            )
        except Exception as exc:
            return HouseSearchRunResult(
                ok=False,
                result_name="EXCEPTION",
                fallback_to_legacy=self._fallback_enabled(),
                reason=str(exc),
            )
        finally:
            self._stop_frame_pump()
            self._release_existing_controls()

    def _ensure_imports(self):
        ensure_pubg_test_import_path(self.config)

    def _get_or_create_explorer(self):
        if self._explorer is not None:
            return self._explorer

        ensure_pubg_test_import_path(self.config)
        from gametest_proxy.common.vision import VisionResultStore
        from gametest_proxy.pubg_common.pubg_action_runner import PubgActionRunner
        from gametest_proxy.pubg_room_explore.explore_replay_mix_yolo import (
            PubgReplayMixedYolo,
        )
        from gametest_proxy.pubg_room_explore.explore_with_dsl_replay import (
            PubgExploreWithReplay,
        )
        from gametest_proxy.pubg_with_yolo.yolo.client import YoloClient

        vision_store = VisionResultStore()
        action_runner = PubgActionRunner(proxy=self.action_proxy)
        yolo_client = YoloClient(
            host=str(self.config.get("yolo_host") or "localhost"),
            port=int(self.config.get("yolo_port") or 6666),
        )
        common_kwargs = {
            "action_runner": action_runner,
            "room_pic_capture": self.room_pic_capture,
            "yolo_client": yolo_client,
            "vision_result_store": vision_store,
            "room_match_dump_dir": self.config.get("room_match_dump_dir") or None,
            "sam3_host": self.config.get("sam3_host") or "localhost",
            "sam3_port": int(self.config.get("sam3_port") or 12345),
        }

        mode = str(self.config.get("mode") or "mixed_yolo").lower()
        if mode in {"replay", "dsl_replay", "no_fallback"}:
            self._explorer = PubgExploreWithReplay(
                **common_kwargs,
                door_calibration_backend=self.config.get(
                    "door_calibration_backend", "sam3"
                ),
            )
        else:
            self._explorer = PubgReplayMixedYolo(**common_kwargs)
        return self._explorer

    def _prime_frame(self, explorer):
        frame = self.room_pic_capture.get_current_frame(force_refresh=True)
        if frame is not None:
            explorer.on_frame(frame, str(int(time.time() * 1000)))

    def _start_frame_pump(self, explorer):
        self._frame_pump_stop.clear()

        def _pump():
            interval = max(
                0.03,
                float(self.config.get("frame_refresh_interval_sec", 0.12) or 0.12),
            )
            while not self._frame_pump_stop.wait(interval):
                frame = self.room_pic_capture.get_current_frame()
                if frame is not None:
                    explorer.on_frame(frame, str(int(time.time() * 1000)))

        self._frame_pump_thread = threading.Thread(
            target=_pump,
            name="AutoGamePubgRoomSearchFramePump",
            daemon=True,
        )
        self._frame_pump_thread.start()

    def _stop_frame_pump(self):
        self._frame_pump_stop.set()
        if self._frame_pump_thread is not None:
            self._frame_pump_thread.join(timeout=1.0)
        self._frame_pump_thread = None

    def _release_existing_controls(self):
        try:
            self.action_proxy.release_all()
        except Exception:
            pass

    def _fallback_enabled(self) -> bool:
        return bool(self.config.get("fallback_to_legacy", True))

    def _should_fallback(self, result_name: str) -> bool:
        if not self._fallback_enabled():
            return False
        return result_name in {
            "NO_MATCH",
            "REPLAY_REJECTED",
            "NEED_LEAVE_FALLBACK",
            "FAILED",
            "STOPPED",
            "START_FAILED",
            "EXCEPTION",
        }
