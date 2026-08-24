"""完整回放的视频归档，以及其与源录制记录的关联。"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .replay import ReplayRecord


@dataclass(frozen=True)
class ReplayVideoRecord:
    directory: Path
    video_path: Path
    source_directory: Path
    recorded_at: datetime
    duration_seconds: float
    source_duration_seconds: float
    frame_count: int

    @property
    def title(self) -> str:
        return (
            f"{self.recorded_at:%Y-%m-%d %H:%M:%S}    "
            f"{self.duration_seconds:.1f} 秒    "
            f"源录制：{self.source_directory.name}"
        )


class ReplayVideoRecorder:
    """按固定帧率保存回放期的实时画面，并保存来源关系。"""

    def __init__(self, history_root: Path, source_record: ReplayRecord, fps: float = 15.0):
        self.history_root = Path(history_root)
        self.source_record = source_record
        self.fps = max(1.0, float(fps))
        self.directory: Path | None = None
        self._started_at: float | None = None
        self._writer = None
        self._last_frame = None
        self._frame_size = None
        self._frame_count = 0

    @staticmethod
    def _as_rgb(frame) -> np.ndarray:
        image = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("回放画面必须是 HxWx3 RGB 图像")
        return image

    def start(self, initial_frame, timeline_started_at: float) -> Path:
        if self._started_at is not None:
            raise RuntimeError("回放视频已经开始保存")
        self.history_root.mkdir(parents=True, exist_ok=True)
        directory = self.history_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory.mkdir(parents=False, exist_ok=False)
        self.directory = directory
        self._started_at = float(timeline_started_at)
        if initial_frame is not None:
            self.accept_frame(initial_frame)
        return directory

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        if self.directory is None:
            raise RuntimeError("回放视频目录尚未创建")
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        writer = cv2.VideoWriter(
            str(self.directory / "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self._frame_size,
        )
        if not writer.isOpened():
            writer.release()
            raise OSError("无法启动回放视频编码器")
        self._writer = writer

    def accept_frame(self, frame):
        """按时间轴补齐固定帧率，避免界面刷新率改变视频总时长。"""
        if self._started_at is None:
            return
        rgb = self._as_rgb(frame)
        self._last_frame = rgb
        self._ensure_writer(rgb)
        elapsed = max(0.0, time.monotonic() - self._started_at)
        target_count = max(1, int(elapsed * self.fps) + 1)
        while self._frame_count < target_count:
            image = rgb
            if (rgb.shape[1], rgb.shape[0]) != self._frame_size:
                image = cv2.resize(rgb, self._frame_size)
            self._writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self._frame_count += 1

    def stop(self, success: bool, timeline_ended_at: float | None = None) -> Path | None:
        if self._started_at is None or self.directory is None:
            return None
        directory = self.directory
        try:
            if not success or self._writer is None or self._last_frame is None:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                shutil.rmtree(directory, ignore_errors=True)
                return None
            ended_at = time.monotonic() if timeline_ended_at is None else float(timeline_ended_at)
            duration = max(0.0, ended_at - self._started_at)
            target_count = max(1, int(round(duration * self.fps)))
            while self._frame_count < target_count:
                image = self._last_frame
                if (image.shape[1], image.shape[0]) != self._frame_size:
                    image = cv2.resize(image, self._frame_size)
                self._writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self._frame_count += 1
            self._writer.release()
            self._writer = None
            source_directory = self.source_record.directory.resolve()
            try:
                source_value = str(source_directory.relative_to(self.history_root.parent.resolve()))
            except ValueError:
                source_value = str(source_directory)
            metadata = {
                "source_record_directory": source_value,
                "source_duration_seconds": self.source_record.duration_seconds,
                "duration_seconds": duration,
                "frame_count": self._frame_count,
                "fps": self.fps,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            (directory / "replay_session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return directory
        finally:
            if self._writer is not None:
                self._writer.release()
            self._writer = None
            self._started_at = None


def discover_replay_video_records(records_root: Path) -> list[ReplayVideoRecord]:
    """读取成功完成的回放视频；缺失源录制时也保留历史条目供提示。"""
    root = Path(records_root).expanduser().resolve()
    history_root = root / "replays"
    if not history_root.is_dir():
        return []
    records = []
    for metadata_path in history_root.glob("*/replay_session.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                continue
            video_path = metadata_path.parent / "video.mp4"
            source_value = str(metadata.get("source_record_directory") or "")
            source_path = Path(source_value)
            if not source_path.is_absolute():
                source_path = root / source_path
            records.append(
                ReplayVideoRecord(
                    directory=metadata_path.parent,
                    video_path=video_path,
                    source_directory=source_path,
                    recorded_at=datetime.fromtimestamp(metadata_path.parent.stat().st_mtime),
                    duration_seconds=max(0.0, float(metadata.get("duration_seconds") or 0.0)),
                    source_duration_seconds=max(
                        0.0, float(metadata.get("source_duration_seconds") or 0.0)
                    ),
                    frame_count=max(0, int(metadata.get("frame_count") or 0)),
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda record: record.recorded_at, reverse=True)
