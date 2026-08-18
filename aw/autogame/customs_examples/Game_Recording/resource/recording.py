"""本地视频和键盘动作录制。"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


BUTTON_FLAG_BY_KEY = {
    "space": "jump",
    "j": "pick_btn",
    "m": "map",
    "k": "door",
    "f": "attack",
}
ACTION_FLAGS = (
    "do_move",
    "view_left",
    "view_right",
    "view_up",
    "view_down",
    "jump",
    "pick_btn",
    "map",
    "door",
    "attack",
)
MOVEMENT_VECTORS = {
    "w": (0.0, -1.0),
    "a": (-1.0, 0.0),
    "s": (0.0, 1.0),
    "d": (1.0, 0.0),
}


def _movement_direction(keys: Iterable[str]) -> Tuple[int, int]:
    vectors = [MOVEMENT_VECTORS[key] for key in keys if key in MOVEMENT_VECTORS]
    x = sum(vector[0] for vector in vectors)
    y = sum(vector[1] for vector in vectors)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return 0, 0
    direction = int(round(math.degrees(math.atan2(x, -y)))) % 360
    return 1, direction


class RecordingSession:
    def __init__(
        self,
        output_root: Path,
        fps: float = 15.0,
        writer_factory: Optional[Callable[..., Any]] = None,
    ):
        self.output_root = Path(output_root)
        self.fps = max(1.0, float(fps))
        self.writer_factory = writer_factory or cv2.VideoWriter
        self._lock = threading.RLock()
        self._writer = None
        self._frame_size = None
        self._started_at = None
        self._session_dir: Optional[Path] = None
        self._screen_size: Optional[Tuple[int, int]] = None
        self._layout: Dict[str, Any] = {}
        self._raw_events = []
        self._steps = []
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._started_at is not None

    @property
    def session_dir(self) -> Optional[Path]:
        with self._lock:
            return self._session_dir

    @staticmethod
    def _as_rgb(frame: Any) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("录制画面必须是 HxWx3 RGB 图像")
        return np.ascontiguousarray(image, dtype=np.uint8)

    def start(
        self,
        initial_frame: Any,
        screen_size: Tuple[int, int],
        layout: Dict[str, Any],
        pressed_keys: Iterable[str] = (),
    ) -> Path:
        with self._lock:
            if self._started_at is not None:
                raise RuntimeError("当前已经在录制")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self._session_dir = self.output_root / stamp
            self._session_dir.mkdir(parents=True, exist_ok=False)
            self._started_at = time.monotonic()
            self._screen_size = (int(screen_size[0]), int(screen_size[1]))
            self._layout = dict(layout)
            self._raw_events = []
            self._steps = []
            self._frame_count = 0
            self._writer = None
            self._frame_size = None

            if initial_frame is not None:
                rgb = self._as_rgb(initial_frame)
                initial_path = self._session_dir / "initial_view.png"
                if not cv2.imwrite(str(initial_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"初始画面保存失败：{initial_path}")
            self._append_step(pressed_keys, timestamp=0.0)
            return self._session_dir

    def _ensure_writer(self, rgb: np.ndarray):
        if self._writer is not None:
            return
        if self._session_dir is None:
            raise RuntimeError("录制目录尚未创建")
        height, width = rgb.shape[:2]
        self._frame_size = (width, height)
        video_path = self._session_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = self.writer_factory(str(video_path), fourcc, self.fps, self._frame_size)
        if hasattr(writer, "isOpened") and not writer.isOpened():
            writer.release()
            raise OSError(f"视频编码器启动失败：{video_path}")
        self._writer = writer

    def accept_frame(self, frame: Any) -> bool:
        with self._lock:
            if self._started_at is None:
                return False
            rgb = self._as_rgb(frame)
            self._ensure_writer(rgb)
            if (rgb.shape[1], rgb.shape[0]) != self._frame_size:
                rgb = cv2.resize(rgb, self._frame_size)
            self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self._frame_count += 1
            return True

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return round(time.monotonic() - self._started_at, 4)

    def _append_step(self, pressed_keys: Iterable[str], timestamp: Optional[float] = None):
        keys = set(pressed_keys)
        moving, direction = _movement_direction(keys)
        flags = {name: 0 for name in ACTION_FLAGS}
        flags["do_move"] = moving
        for key, flag in BUTTON_FLAG_BY_KEY.items():
            if key in keys:
                flags[flag] = 1
        step = {
            "time": self._elapsed() if timestamp is None else round(float(timestamp), 4),
            "move_direction": direction,
            "actions": flags,
            "params": {"keyboard_keys": sorted(keys)},
        }
        if self._steps and self._steps[-1]["move_direction"] == step["move_direction"]:
            previous = dict(self._steps[-1])
            previous.pop("time", None)
            current = dict(step)
            current.pop("time", None)
            if previous == current:
                return
        self._steps.append(step)

    def record_key_event(
        self,
        event_type: str,
        key: str,
        pressed_keys: Iterable[str],
        device_actions=None,
    ) -> bool:
        with self._lock:
            if self._started_at is None:
                return False
            self._raw_events.append(
                {
                    "time": self._elapsed(),
                    "event": str(event_type),
                    "key": str(key),
                    "pressed_keys": sorted(set(pressed_keys)),
                    "device_actions": list(device_actions or []),
                }
            )
            self._append_step(pressed_keys)
            return True

    def stop(self, reason: str = "e") -> Optional[Path]:
        with self._lock:
            if self._started_at is None or self._session_dir is None:
                return None
            duration = self._elapsed()
            session_dir = self._session_dir
            # 回放文件必须以松开状态结束，避免回放结束后摇杆仍保持按下。
            self._append_step(set(), timestamp=duration)
            writer = self._writer
            self._writer = None
            if writer is not None:
                writer.release()

            (session_dir / "action_raw.json").write_text(
                json.dumps(self._raw_events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (session_dir / "action_step.json").write_text(
                json.dumps(self._steps, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metadata = {
                "duration_seconds": duration,
                "frame_count": self._frame_count,
                "fps": self.fps,
                "screen_size": list(self._screen_size or ()),
                "layout": self._layout,
                "stop_reason": reason,
            }
            (session_dir / "session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self._started_at = None
            self._session_dir = None
            self._screen_size = None
            return session_dir
