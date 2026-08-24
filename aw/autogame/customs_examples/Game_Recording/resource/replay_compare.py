"""历史回放视频与源录制视频的离线同步对比界面。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .replay_video import ReplayVideoRecord, discover_replay_video_records


class ReplayComparisonPanel(QWidget):
    """按共同时间轴播放：左回放视频，右源录制视频。"""

    def __init__(self, records_root: Path, parent=None):
        super().__init__(parent)
        self.records_root = Path(records_root)
        self.records: list[ReplayVideoRecord] = []
        self.replay_capture = None
        self.recording_capture = None
        self.replay_frame_count = 0
        self.recording_frame_count = 0
        self.replay_frame_index = -1
        self.recording_frame_index = -1
        self.duration_seconds = 0.0
        self.position_seconds = 0.0
        self._play_started_at = None
        self._play_started_position = 0.0
        self._playing = False

        self.record_list = QListWidget()
        self.record_list.currentItemChanged.connect(self._selection_changed)
        self.refresh_button = QPushButton("刷新历史回放")
        self.refresh_button.clicked.connect(self.refresh_records)
        self.start_button = QPushButton("加载并开始对比")
        self.start_button.clicked.connect(self.start_comparison)
        self.start_button.setEnabled(False)
        self.detail_label = QLabel("选择一条已完整回放的历史记录。")
        self.detail_label.setWordWrap(True)

        selection = QWidget()
        selection_layout = QHBoxLayout(selection)
        list_column = QVBoxLayout()
        list_column.addWidget(QLabel("历史回放记录（新的在前）"))
        list_column.addWidget(self.record_list, 1)
        list_column.addWidget(self.refresh_button)
        list_column.addWidget(self.start_button)
        detail_column = QVBoxLayout()
        detail_column.addWidget(QLabel("关联信息"))
        detail_column.addWidget(self.detail_label, 1)
        selection_layout.addLayout(list_column, 1)
        selection_layout.addLayout(detail_column, 1)

        self.replay_label = self._video_label("请先加载历史回放视频")
        self.recording_label = self._video_label("请先加载源录制视频")
        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        videos = QHBoxLayout()
        replay_column = QVBoxLayout()
        replay_column.addWidget(QLabel("回放视频"))
        replay_column.addWidget(self.replay_label, 1)
        recording_column = QVBoxLayout()
        recording_column.addWidget(QLabel("原录制视频"))
        recording_column.addWidget(self.recording_label, 1)
        videos.addLayout(replay_column, 1)
        videos.addLayout(recording_column, 1)
        viewer_layout.addLayout(videos, 1)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.sliderMoved.connect(self._seek_slider)
        self.time_label = QLabel("0.00 / 0.00 秒")
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self._toggle_play)
        self.speed_combo = QComboBox()
        for value in ("0.25x", "0.5x", "1x", "1.5x", "2x", "3x"):
            self.speed_combo.addItem(value, float(value[:-1]))
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.back_button = QPushButton("返回历史记录")
        self.back_button.clicked.connect(self._back_to_selection)
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(QLabel("倍速"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.time_label, 1)
        controls.addWidget(self.back_button)
        viewer_layout.addWidget(self.timeline)
        viewer_layout.addLayout(controls)

        self.pages = QStackedWidget()
        self.pages.addWidget(selection)
        self.pages.addWidget(viewer)
        root = QVBoxLayout(self)
        root.addWidget(self.pages)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.refresh_records()

    @staticmethod
    def _video_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(420, 236)
        label.setStyleSheet("background: #15191e; color: #ddd;")
        return label

    @property
    def selected_record(self) -> ReplayVideoRecord | None:
        item = self.record_list.currentItem()
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        return self.records[index] if isinstance(index, int) else None

    def refresh_records(self):
        selected = self.selected_record
        selected_path = selected.directory if selected is not None else None
        self.records = discover_replay_video_records(self.records_root)
        self.record_list.blockSignals(True)
        self.record_list.clear()
        selected_index = 0
        for index, record in enumerate(self.records):
            item = QListWidgetItem(record.title)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.record_list.addItem(item)
            if record.directory == selected_path:
                selected_index = index
        if self.records:
            self.record_list.setCurrentRow(selected_index)
        self.record_list.blockSignals(False)
        self._selection_changed(self.record_list.currentItem(), None)

    def _selection_changed(self, current, previous):
        del previous
        record = self.selected_record if current is not None else None
        if record is None:
            self.start_button.setEnabled(False)
            self.detail_label.setText("没有找到已完整回放的视频。")
            return
        source_video = record.source_directory / "video.mp4"
        source_state = "可用" if source_video.is_file() else "源录制视频已被删除或缺失"
        replay_state = "可用" if record.video_path.is_file() else "回放视频已缺失"
        self.start_button.setEnabled(source_video.is_file() and record.video_path.is_file())
        self.detail_label.setText(
            f"回放时间：{record.recorded_at:%Y-%m-%d %H:%M:%S}\n"
            f"回放时长：{record.duration_seconds:.2f} 秒\n"
            f"关联源录制：{record.source_directory.name}\n"
            f"源录制路径：{record.source_directory}\n"
            f"回放视频：{replay_state}\n"
            f"源录制视频：{source_state}"
        )

    def start_comparison(self):
        record = self.selected_record
        if record is None:
            return
        self._close_videos()
        replay = cv2.VideoCapture(str(record.video_path))
        recording = cv2.VideoCapture(str(record.source_directory / "video.mp4"))
        if not replay.isOpened() or not recording.isOpened():
            replay.release()
            recording.release()
            self.detail_label.setText("无法打开回放视频或其关联的源录制视频。")
            return
        self.replay_capture = replay
        self.recording_capture = recording
        self.replay_frame_count = max(1, int(replay.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.recording_frame_count = max(1, int(recording.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.replay_frame_index = -1
        self.recording_frame_index = -1
        # 以源录制的动作时间轴为准；回放收尾的连接清理时间不应拉长对比画面。
        self.duration_seconds = max(
            0.001,
            record.source_duration_seconds or record.duration_seconds,
        )
        self.position_seconds = 0.0
        self.pages.setCurrentIndex(1)
        self._render_position()
        self._set_playing(True)

    def _set_playing(self, playing: bool):
        self._playing = bool(playing)
        if self._playing:
            self._play_started_at = time.monotonic()
            self._play_started_position = self.position_seconds
            self.timer.start(30)
            self.play_button.setText("暂停")
        else:
            self.timer.stop()
            self.play_button.setText("播放")

    def _toggle_play(self):
        self._set_playing(not self._playing)

    def _speed_changed(self):
        if self._playing:
            self._tick()
            self._play_started_at = time.monotonic()
            self._play_started_position = self.position_seconds

    def _tick(self):
        if not self._playing:
            return
        speed = float(self.speed_combo.currentData() or 1.0)
        self.position_seconds = min(
            self.duration_seconds,
            self._play_started_position + (time.monotonic() - self._play_started_at) * speed,
        )
        self._render_position()
        if self.position_seconds >= self.duration_seconds:
            self._set_playing(False)

    def _seek_slider(self, value: int):
        self.position_seconds = self.duration_seconds * int(value) / 10000
        if self._playing:
            self._play_started_at = time.monotonic()
            self._play_started_position = self.position_seconds
        self._render_position()

    def _render_position(self):
        if self.replay_capture is None or self.recording_capture is None:
            return
        fraction = min(1.0, max(0.0, self.position_seconds / self.duration_seconds))
        self._render_video(
            self.replay_capture,
            self.replay_label,
            min(self.replay_frame_count - 1, int(fraction * self.replay_frame_count)),
            "replay",
        )
        self._render_video(
            self.recording_capture,
            self.recording_label,
            min(self.recording_frame_count - 1, int(fraction * self.recording_frame_count)),
            "recording",
        )
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(round(fraction * 10000)))
        self.timeline.blockSignals(False)
        self.time_label.setText(f"{self.position_seconds:.2f} / {self.duration_seconds:.2f} 秒")

    def _render_video(self, capture, label: QLabel, frame_index: int, side: str):
        previous = self.replay_frame_index if side == "replay" else self.recording_frame_index
        if frame_index == previous:
            return
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            label.setText("读取视频帧失败")
            return
        if side == "replay":
            self.replay_frame_index = frame_index
        else:
            self.recording_frame_index = frame_index
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        image = QImage(
            frame_rgb.data, width, height, int(frame_rgb.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        label.setText("")
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def _back_to_selection(self):
        self._set_playing(False)
        self._close_videos()
        self.pages.setCurrentIndex(0)
        self.refresh_records()

    def _close_videos(self):
        for capture in (self.replay_capture, self.recording_capture):
            if capture is not None:
                capture.release()
        self.replay_capture = None
        self.recording_capture = None

    def stop(self):
        self._set_playing(False)
        self._close_videos()
