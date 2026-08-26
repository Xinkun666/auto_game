"""Launcher 内嵌的独立回放记录浏览与视频预览页面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .replay import (
    ReplayRecord,
    discover_replay_records,
    load_recorded_control_points,
)


def format_replay_time(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


class LauncherReplayPanel(QWidget):
    """先看录制视频、后确认控点绑定并启动真实回放。"""

    replayRequested = pyqtSignal(object)

    def __init__(self, records_root: Path, parent=None):
        super().__init__(parent)
        self.records_root = Path(records_root)
        self.records: list[ReplayRecord] = []
        self._capture = None
        self._video_pixmap = QPixmap()
        self._video_fps = 15.0
        self._video_frame_count = 0
        self._video_frame_index = -1
        self._video_duration_seconds = 0.0
        self._binding_pixmap = QPixmap()
        self._playing = False
        self._replay_active = False

        self.record_tree = QTreeWidget()
        self.record_tree.setObjectName("previewInfo")
        self.record_tree.setHeaderLabels(["记录", "录制时间", "时长", "动作数"])
        self.record_tree.setRootIsDecorated(False)
        self.record_tree.setAlternatingRowColors(True)
        self.record_tree.setUniformRowHeights(True)
        self.record_tree.setMinimumWidth(430)
        self.record_tree.setColumnWidth(0, 140)
        self.record_tree.setColumnWidth(1, 135)
        self.record_tree.setColumnWidth(2, 55)
        self.record_tree.setColumnWidth(3, 55)
        self.record_tree.itemSelectionChanged.connect(self._on_record_selected)

        self.refresh_button = QPushButton("刷新记录")
        self.refresh_button.clicked.connect(self.refresh_records)
        left_group = QGroupBox("可回放记录（新的在前）")
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(12, 10, 12, 12)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.record_tree, 1)
        left_layout.addWidget(self.refresh_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.record_detail_label = QLabel("请先选择一条录制记录")
        self.record_detail_label.setWordWrap(True)
        self.record_detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.video_label = QLabel("选择记录后显示录制视频")
        self.video_label.setObjectName("previewSurface")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(520, 280)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.play_pause_button = QPushButton("播放")
        self.play_pause_button.setMinimumWidth(96)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.clicked.connect(self.toggle_video_playback)
        self.video_progress_label = QLabel("00:00 / 00:00")
        self.video_progress_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        video_controls = QHBoxLayout()
        video_controls.setContentsMargins(0, 0, 0, 0)
        video_controls.addWidget(self.play_pause_button)
        video_controls.addStretch(1)
        video_controls.addWidget(self.video_progress_label)

        video_group = QGroupBox("录制视频预览")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(12, 10, 12, 12)
        video_layout.setSpacing(8)
        video_layout.addWidget(self.record_detail_label, 0)
        video_layout.addWidget(self.video_label, 1)
        video_layout.addLayout(video_controls)

        self.binding_empty_label = QLabel("该录制记录没有保存场景图和控点数据。")
        self.binding_empty_label.setObjectName("previewTemplate")
        self.binding_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binding_empty_label.setWordWrap(True)
        self.binding_empty_label.setMinimumHeight(110)

        self.binding_preview_label = QLabel("暂无录制场景图")
        self.binding_preview_label.setObjectName("previewTemplate")
        self.binding_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binding_preview_label.setMinimumSize(260, 150)
        self.binding_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.binding_tree = QTreeWidget()
        self.binding_tree.setObjectName("previewInfo")
        self.binding_tree.setHeaderLabels(["按键", "场景", "录制时位置"])
        self.binding_tree.setRootIsDecorated(False)
        self.binding_tree.setUniformRowHeights(True)
        self.binding_tree.setMinimumWidth(250)
        self.binding_tree.setColumnWidth(0, 72)
        self.binding_tree.setColumnWidth(1, 90)
        self.binding_content = QWidget()
        binding_content_layout = QHBoxLayout(self.binding_content)
        binding_content_layout.setContentsMargins(0, 0, 0, 0)
        binding_content_layout.setSpacing(10)
        binding_content_layout.addWidget(self.binding_preview_label, 3)
        binding_content_layout.addWidget(self.binding_tree, 2)
        self.binding_content.hide()

        binding_group = QGroupBox("控点绑定")
        binding_layout = QVBoxLayout(binding_group)
        binding_layout.setContentsMargins(12, 10, 12, 12)
        binding_layout.addWidget(self.binding_empty_label)
        binding_layout.addWidget(self.binding_content, 1)

        self.status_label = QLabel("选择一条记录，查看视频后即可开始回放。")
        self.status_label.setWordWrap(True)
        self.start_replay_button = QPushButton("开始回放")
        self.start_replay_button.setProperty("primaryButton", True)
        self.start_replay_button.setMinimumWidth(128)
        self.start_replay_button.setEnabled(False)
        self.start_replay_button.clicked.connect(self._request_replay)
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self.status_label, 1)
        action_layout.addWidget(self.start_replay_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(video_group, 3)
        right_layout.addWidget(binding_group, 1)
        right_layout.addLayout(action_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.addWidget(left_group)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([460, 720])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter, 1)

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._advance_video)
        self.refresh_records()

    @property
    def selected_record(self) -> ReplayRecord | None:
        items = self.record_tree.selectedItems()
        if not items:
            return None
        index = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not 0 <= index < len(self.records):
            return None
        return self.records[index]

    def refresh_records(self):
        selected = self.selected_record
        selected_path = selected.directory if selected is not None else None
        self.records = discover_replay_records(self.records_root)
        self.record_tree.blockSignals(True)
        self.record_tree.clear()
        selected_item = None
        for index, record in enumerate(self.records):
            item = QTreeWidgetItem(
                [
                    record.directory.name,
                    record.display_time,
                    format_replay_time(record.duration_seconds),
                    str(record.action_count),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            item.setToolTip(0, str(record.directory))
            self.record_tree.addTopLevelItem(item)
            if selected_path is not None and record.directory == selected_path:
                selected_item = item
        if selected_item is None and self.record_tree.topLevelItemCount():
            selected_item = self.record_tree.topLevelItem(0)
        if selected_item is not None:
            self.record_tree.setCurrentItem(selected_item)
        self.record_tree.blockSignals(False)
        self._on_record_selected()

    def _on_record_selected(self):
        record = self.selected_record
        self._release_video()
        self.video_label.setToolTip("")
        if record is None:
            self.record_detail_label.setText("没有找到可回放记录，请先在“录制”页完成录制。")
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("暂无录制视频")
            self.video_progress_label.setText("00:00 / 00:00")
            self.start_replay_button.setEnabled(False)
            self._clear_binding_snapshot()
            return

        self.record_detail_label.setText(
            f"{record.directory.name}  ·  {record.display_time}  ·  "
            f"{record.duration_seconds:.1f} 秒  ·  {record.action_count} 个动作"
        )
        self.start_replay_button.setEnabled(not self._replay_active)
        self._load_binding_snapshot(record)
        self._load_video(record)

    @staticmethod
    def _normalized_position(
        point: Mapping[str, Any], screen_size: tuple[int, int]
    ) -> tuple[float, float] | None:
        normalized = point.get("normalized_position")
        if isinstance(normalized, (list, tuple)) and len(normalized) == 2:
            try:
                norm_x, norm_y = float(normalized[0]), float(normalized[1])
            except (TypeError, ValueError):
                return None
        else:
            position = point.get("position")
            if (
                not isinstance(position, (list, tuple))
                or len(position) != 2
                or screen_size[0] <= 0
                or screen_size[1] <= 0
            ):
                return None
            try:
                norm_x = float(position[0]) / screen_size[0]
                norm_y = float(position[1]) / screen_size[1]
            except (TypeError, ValueError):
                return None
        if 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0:
            return norm_x, norm_y
        return None

    def _clear_binding_snapshot(self):
        self._binding_pixmap = QPixmap()
        self.binding_tree.clear()
        self.binding_content.hide()
        self.binding_empty_label.show()

    def _load_binding_snapshot(self, record: ReplayRecord):
        self._clear_binding_snapshot()
        raw_points, screen_size = load_recorded_control_points(record)
        scene_path = record.scene_view_path
        scene_pixmap = QPixmap(str(scene_path)) if scene_path is not None else QPixmap()
        valid_points = []
        for raw_key, raw_point in raw_points.items():
            if not isinstance(raw_point, Mapping):
                continue
            normalized = self._normalized_position(raw_point, screen_size)
            if normalized is None:
                continue
            valid_points.append((str(raw_key), raw_point, normalized))

        if scene_pixmap.isNull() and not valid_points:
            self.status_label.setText("该记录没有可用的场景图或控点快照。")
            return

        if not scene_pixmap.isNull():
            annotated = scene_pixmap.copy()
            painter = QPainter(annotated)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            marker_radius = max(
                10, int(round(min(annotated.width(), annotated.height()) * 0.018))
            )
            painter.setPen(QPen(QColor("#ffffff"), max(2, marker_radius // 5)))
            painter.setBrush(QColor("#ff4d6d"))
            label_font = QFont(painter.font())
            label_font.setPixelSize(
                max(16, int(round(min(annotated.width(), annotated.height()) * 0.025)))
            )
            label_font.setBold(True)
            painter.setFont(label_font)
            for key, _point, (norm_x, norm_y) in valid_points:
                x = int(round(norm_x * max(annotated.width() - 1, 0)))
                y = int(round(norm_y * max(annotated.height() - 1, 0)))
                painter.drawEllipse(
                    x - marker_radius,
                    y - marker_radius,
                    marker_radius * 2,
                    marker_radius * 2,
                )
                painter.drawText(
                    x + marker_radius + 5,
                    y - marker_radius // 2,
                    key,
                )
            painter.end()
            self._binding_pixmap = annotated
            self._render_binding_pixmap()
        else:
            self.binding_preview_label.setPixmap(QPixmap())
            self.binding_preview_label.setText("该记录没有保存场景图")

        for key, point, normalized in valid_points:
            position = point.get("position")
            if isinstance(position, (list, tuple)) and len(position) == 2:
                try:
                    position_text = f"{int(position[0])}, {int(position[1])}"
                except (TypeError, ValueError):
                    position_text = f"{normalized[0]:.3f}, {normalized[1]:.3f}"
            else:
                position_text = f"{normalized[0]:.3f}, {normalized[1]:.3f}"
            self.binding_tree.addTopLevelItem(
                QTreeWidgetItem(
                    [key, str(point.get("scene") or "未命名"), position_text]
                )
            )
        self.binding_empty_label.hide()
        self.binding_content.show()
        if valid_points:
            self.status_label.setText(
                f"已加载录制时保存的场景图和 {len(valid_points)} 个控点；"
                "开始回放时优先使用这份记录级布局。"
            )
        else:
            self.status_label.setText("已加载录制场景图，但该记录没有有效控点。")

    def _render_binding_pixmap(self):
        if self._binding_pixmap.isNull():
            return
        self.binding_preview_label.setText("")
        self.binding_preview_label.setPixmap(
            self._binding_pixmap.scaled(
                self.binding_preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _load_video(self, record: ReplayRecord):
        video_path = record.directory / "video.mp4"
        if not video_path.is_file():
            self._show_initial_view_fallback(record, "该记录没有录制视频，无法预览。")
            return
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            self._show_initial_view_fallback(record, "录制视频无法打开。")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            self._show_initial_view_fallback(record, "录制视频没有可读取的画面。")
            return
        self._capture = capture
        self._video_fps = fps if fps > 0 else 15.0
        self._video_frame_count = frame_count
        self._video_frame_index = 0
        self._video_duration_seconds = (
            frame_count / self._video_fps
            if frame_count > 0
            else max(0.0, record.duration_seconds)
        )
        self.play_pause_button.setEnabled(True)
        self.play_pause_button.setText("播放")
        self._render_video_frame(frame)
        self._update_video_progress()

    def _show_initial_view_fallback(self, record: ReplayRecord, message: str):
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setText("播放")
        self.video_progress_label.setText("00:00 / 00:00")
        fallback = (
            QPixmap(str(record.initial_view_path))
            if record.initial_view_path is not None
            else QPixmap()
        )
        if fallback.isNull():
            self._video_pixmap = QPixmap()
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText(message)
            return
        self._video_pixmap = fallback
        self.video_label.setToolTip(message)
        self._render_video_pixmap()

    def toggle_video_playback(self):
        if self._capture is None:
            return
        if self._playing:
            self._pause_video("播放")
            return
        if (
            self._video_frame_count > 0
            and self._video_frame_index >= self._video_frame_count - 1
        ):
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_frame_index = -1
        self._playing = True
        self.play_pause_button.setText("暂停")
        interval_ms = max(15, min(200, int(round(1000.0 / self._video_fps))))
        self.video_timer.start(interval_ms)

    def _advance_video(self):
        capture = self._capture
        if capture is None:
            self._pause_video("播放")
            return
        ok, frame = capture.read()
        if not ok:
            self._pause_video("重新播放")
            return
        self._video_frame_index += 1
        self._render_video_frame(frame)
        self._update_video_progress()

    def _render_video_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        image = QImage(
            frame_rgb.data,
            width,
            height,
            int(frame_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._video_pixmap = QPixmap.fromImage(image)
        self._render_video_pixmap()

    def _render_video_pixmap(self):
        if self._video_pixmap.isNull():
            return
        self.video_label.setText("")
        self.video_label.setPixmap(
            self._video_pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_video_progress(self):
        elapsed = max(0, self._video_frame_index) / max(self._video_fps, 1.0)
        self.video_progress_label.setText(
            f"{format_replay_time(elapsed)} / "
            f"{format_replay_time(self._video_duration_seconds)}"
        )

    def _pause_video(self, button_text: str = "播放"):
        self.video_timer.stop()
        self._playing = False
        self.play_pause_button.setText(button_text)

    def _release_video(self):
        self._pause_video()
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        self._video_pixmap = QPixmap()
        self._video_fps = 15.0
        self._video_frame_count = 0
        self._video_frame_index = -1
        self._video_duration_seconds = 0.0
        self.play_pause_button.setEnabled(False)

    def _request_replay(self):
        record = self.selected_record
        if record is None or self._replay_active:
            return
        self._pause_video()
        self.replayRequested.emit(record)

    def set_replay_active(self, active: bool):
        self._replay_active = bool(active)
        self.start_replay_button.setEnabled(
            self.selected_record is not None and not self._replay_active
        )
        self.status_label.setText(
            "回放窗口已启动，请在回放结束后返回这里选择其他记录。"
            if self._replay_active
            else "选择一条记录，查看视频后即可开始回放。"
        )

    def stop(self):
        self._release_video()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_video_pixmap()
        self._render_binding_pixmap()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
