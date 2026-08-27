"""本地视频和键盘动作录制。"""

from __future__ import annotations

import copy
import json
import math
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
INPUT_SEMANTICS_VERSION = 2
INVALID_RECORDING_NAME_CHARS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


def normalize_recording_name(value: Any) -> str:
    """验证用户输入的录制目录名，空字符串表示使用默认时间戳。"""
    name = str(value or "").strip()
    if not name:
        return ""
    if len(name) > 120:
        raise ValueError("录制名称不能超过 120 个字符。")
    if name in {".", ".."}:
        raise ValueError("录制名称不能是 . 或 ..。")
    invalid = sorted({char for char in name if char in INVALID_RECORDING_NAME_CHARS or ord(char) < 32})
    if invalid:
        raise ValueError("录制名称不能包含这些字符：" + " ".join(invalid))
    if name.endswith("."):
        raise ValueError("录制名称不能以句点结尾。")
    if name.split(".", 1)[0].lower() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"录制名称不能使用系统保留名：{name}。")
    return name


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
        self._latest_frame: Optional[np.ndarray] = None
        self._reference_scenes: list[dict[str, Any]] = []
        self._raw_events = []
        self._steps = []
        self._frame_count = 0
        self._custom_recording_name = False

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
        session_name: str = "",
        reference_scenes: Sequence[Mapping[str, Any]] = (),
    ) -> Path:
        with self._lock:
            if self._started_at is not None:
                raise RuntimeError("当前已经在录制")
            custom_name = normalize_recording_name(session_name)
            directory_name = custom_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            session_dir = self.output_root / directory_name
            if session_dir.exists():
                raise FileExistsError(f"录制名称已存在：{directory_name}，请换一个名称。")
            session_dir.mkdir(parents=True, exist_ok=False)
            self._session_dir = session_dir
            self._custom_recording_name = bool(custom_name)
            self._started_at = time.monotonic()
            self._screen_size = (int(screen_size[0]), int(screen_size[1]))
            self._layout = dict(layout)
            self._raw_events = []
            self._steps = []
            self._frame_count = 0
            self._writer = None
            self._frame_size = None
            self._latest_frame = None
            self._reference_scenes = self._export_reference_scenes(reference_scenes)

            if initial_frame is not None:
                rgb = self._as_rgb(initial_frame)
                self._latest_frame = rgb.copy()
                initial_path = self._session_dir / "initial_view.png"
                if not cv2.imwrite(str(initial_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"初始画面保存失败：{initial_path}")
            self._append_step(pressed_keys, timestamp=0.0)
            return self._session_dir

    def _export_reference_scenes(
        self,
        reference_scenes: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._session_dir is None:
            raise RuntimeError("录制目录尚未创建")
        exported: list[dict[str, Any]] = []
        target_root = self._session_dir / "reference_scenes"
        for index, raw_scene in enumerate(reference_scenes):
            if not isinstance(raw_scene, Mapping):
                continue
            scene_id = f"scene-{index + 1:03d}"
            source_text = str(raw_scene.get("source_image_path") or "")
            source_path = Path(source_text) if source_text else None
            relative_image = ""
            if source_path is not None and source_path.is_file():
                suffix = source_path.suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    suffix = ".png"
                target_root.mkdir(parents=True, exist_ok=True)
                target_path = target_root / f"{scene_id}{suffix}"
                shutil.copy2(source_path, target_path)
                relative_image = target_path.relative_to(self._session_dir).as_posix()
            screen_size = raw_scene.get("screen_size")
            if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
                screen_size = [0, 0]
            keys = raw_scene.get("keys")
            if not isinstance(keys, (list, tuple)):
                keys = []
            points = raw_scene.get("points")
            if not isinstance(points, Mapping):
                points = {}
            exported.append(
                {
                    "id": scene_id,
                    "stage": str(raw_scene.get("stage") or ""),
                    "scene": str(raw_scene.get("scene") or f"场景 {index + 1}"),
                    "resolution_key": str(raw_scene.get("resolution_key") or ""),
                    "screen_size": [int(screen_size[0] or 0), int(screen_size[1] or 0)],
                    "image": relative_image,
                    "calibration_image": "",
                    "source_image": str(raw_scene.get("source_image") or ""),
                    "keys": [str(key) for key in keys if str(key)],
                    "points": copy.deepcopy(dict(points)),
                }
            )
        return exported

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
            self._latest_frame = rgb.copy()
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
            normalized_actions = []
            for action in device_actions or []:
                item = dict(action) if isinstance(action, dict) else action
                if isinstance(item, dict):
                    position = item.get("position")
                    if (
                        isinstance(position, (list, tuple))
                        and len(position) == 2
                        and self._screen_size
                        and self._screen_size[0] > 0
                        and self._screen_size[1] > 0
                    ):
                        item["normalized_position"] = [
                            float(position[0]) / self._screen_size[0],
                            float(position[1]) / self._screen_size[1],
                        ]
                normalized_actions.append(item)
            self._raw_events.append(
                {
                    "time": self._elapsed(),
                    "event": str(event_type),
                    "key": str(key),
                    "pressed_keys": sorted(set(pressed_keys)),
                    "device_actions": normalized_actions,
                }
            )
            self._append_step(pressed_keys)
            return True

    def stop(self, reason: str = "button") -> Optional[Path]:
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
            scene_view_path = session_dir / "scene_view.png"
            if self._latest_frame is not None and not cv2.imwrite(
                str(scene_view_path),
                cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR),
            ):
                raise OSError(f"结束场景图保存失败：{scene_view_path}")
            control_points_path = session_dir / "control_points.json"
            control_points_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "screen_size": list(self._screen_size or ()),
                        "points": self._layout,
                        "reference_scenes": self._reference_scenes,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata = {
                "input_semantics_version": INPUT_SEMANTICS_VERSION,
                "contains_drag_events": any(
                    event.get("event") == "drag" for event in self._raw_events
                ),
                "duration_seconds": duration,
                "frame_count": self._frame_count,
                "fps": self.fps,
                "screen_size": list(self._screen_size or ()),
                "layout": self._layout,
                "scene_view": scene_view_path.name,
                "control_points_path": control_points_path.name,
                "control_point_count": len(self._layout),
                "reference_scenes": self._reference_scenes,
                "reference_scene_count": len(self._reference_scenes),
                "stop_reason": reason,
                "recording_name": session_dir.name,
                "custom_recording_name": self._custom_recording_name,
            }
            (session_dir / "session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self._started_at = None
            self._session_dir = None
            self._screen_size = None
            self._layout = {}
            self._latest_frame = None
            self._reference_scenes = []
            self._custom_recording_name = False
            return session_dir
