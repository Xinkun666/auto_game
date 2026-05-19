from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class AutoGameRoomPicCapture:
    """Frame provider compatible with pubg_test room-search explorers."""

    def __init__(
        self,
        worker,
        control_proxy,
        frame_color: str = "rgb",
        refresh_interval_sec: float = 0.12,
        refresh_mode: str = "worker_refresh",
    ):
        self.worker = worker
        self.control_proxy = control_proxy
        self.frame_color = str(frame_color or "rgb").lower()
        self.refresh_interval_sec = max(0.01, float(refresh_interval_sec or 0.12))
        self.refresh_mode = str(refresh_mode or "worker_refresh").lower()
        self.lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_id: Optional[str] = None
        self.last_refresh_time = 0.0

    def on_frame(self, frame, frame_id: str = ""):
        if frame is None:
            return
        with self.lock:
            self.current_frame = self._to_bgr(frame)
            self.current_frame_id = frame_id or str(int(time.time() * 1000))

    def get_current_frame(self, *, force_refresh: bool = False):
        self.refresh_from_autogame(force_refresh=force_refresh)

        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def refresh_from_autogame(self, *, force_refresh: bool = False) -> bool:
        now = time.time()
        if not force_refresh and now - self.last_refresh_time < self.refresh_interval_sec:
            return True

        with self.refresh_lock:
            now = time.time()
            if not force_refresh and now - self.last_refresh_time < self.refresh_interval_sec:
                return True

            frame = self._pull_frame()
            if frame is None:
                return False
            self.on_frame(frame, str(int(time.time() * 1000)))
            self.last_refresh_time = time.time()
            return True

    def _pull_frame(self):
        if self.refresh_mode == "buffer":
            buffer = getattr(self.worker, "buffer", None)
            if buffer is not None:
                try:
                    frame = buffer.get_latest(timeout=1.0, must_new=True)
                    if frame is not None:
                        self.worker.frame = np.array(frame, copy=True)
                        return self.worker.frame
                except Exception as exc:
                    print(f"[PubgRoomSearch] 从 auto_game buffer 取帧失败: {exc}")

        try:
            self.worker.refresh_frame()
        except Exception as exc:
            print(f"[PubgRoomSearch] 刷新 auto_game 画面失败: {exc}")
        return getattr(self.worker, "frame", None)

    def _to_bgr(self, frame):
        arr = np.array(frame, copy=True)
        if arr.ndim == 3 and arr.shape[2] == 3 and self.frame_color == "rgb":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def capture(self, need_map: bool = True):
        return {"center": self.get_current_frame(force_refresh=True)}

    def capture_and_save(self, room_id: Optional[str] = None):
        frame = self.get_current_frame()
        if frame is None:
            return
        print("[PubgRoomSearch] 当前适配模式不自动写入房屋库截图")
