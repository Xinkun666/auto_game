"""Launcher 内嵌的独立回放记录浏览与视频预览页面。"""

from __future__ import annotations

import copy
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import cv2
from PyQt6.QtCore import QPointF, QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QImage, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aw.autogame.tools.ProcessUtils import resolve_hdc_executable

from .replay import (
    ReplayRecord,
    discover_replay_records,
    load_recorded_control_points,
    save_recorded_control_points,
)


def format_replay_time(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


class ReplayBindingView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None and not self.scene().sceneRect().isEmpty():
            self.fitInView(
                self.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )


class ReplayBindingPointItem(QGraphicsEllipseItem):
    """只移动记录级控点，不接触全局标注配置。"""

    def __init__(
        self,
        key: str,
        position: QPointF,
        scene_width: float,
        scene_height: float,
        moved,
        selected,
    ):
        super().__init__(-7, -7, 14, 14)
        self.key = key
        self.scene_width = scene_width
        self.scene_height = scene_height
        self.moved = moved
        self.selected = selected
        self._initialized = False
        self.setBrush(QBrush(QColor(255, 77, 109, 220)))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(10)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        label = QGraphicsSimpleTextItem(key, self)
        label.setBrush(QBrush(QColor("white")))
        label.setPos(10, -7)
        self.setPos(position)
        self._initialized = True

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = value
            return QPointF(
                min(max(float(point.x()), 0.0), self.scene_width),
                min(max(float(point.y()), 0.0), self.scene_height),
            )
        result = super().itemChange(change, value)
        if (
            self._initialized
            and change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
        ):
            self.moved(self.key, self.pos())
        elif (
            self._initialized
            and change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and bool(value)
        ):
            self.selected(self.key)
        return result


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
        self._binding_source_pixmap = QPixmap()
        self._binding_points: dict[str, Any] = {}
        self._binding_screen_size = (0, 0)
        self._binding_record: ReplayRecord | None = None
        self._binding_items: dict[str, ReplayBindingPointItem] = {}
        self._binding_dirty = False
        self._syncing_binding_selection = False
        self._capture_process = QProcess(self)
        self._capture_process.finished.connect(self._capture_process_finished)
        self._capture_process.errorOccurred.connect(self._capture_process_error)
        self._capture_stage = ""
        self._capture_remote_path = ""
        self._capture_local_path: Path | None = None
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

        self.binding_scene = QGraphicsScene(self)
        self.binding_view = ReplayBindingView(self.binding_scene)
        self.binding_view.setObjectName("previewTemplate")
        self.binding_view.setMinimumSize(300, 210)
        self.binding_view.setBackgroundBrush(QBrush(QColor("#20242a")))
        self.binding_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.binding_tree = QTreeWidget()
        self.binding_tree.setObjectName("previewInfo")
        self.binding_tree.setHeaderLabels(["按键", "场景", "当前绑定位置"])
        self.binding_tree.setRootIsDecorated(False)
        self.binding_tree.setUniformRowHeights(True)
        self.binding_tree.setMinimumWidth(250)
        self.binding_tree.setColumnWidth(0, 72)
        self.binding_tree.setColumnWidth(1, 90)
        self.binding_tree.itemSelectionChanged.connect(
            self._binding_tree_selection_changed
        )
        self.binding_content = QWidget()
        binding_content_layout = QHBoxLayout(self.binding_content)
        binding_content_layout.setContentsMargins(0, 0, 0, 0)
        binding_content_layout.setSpacing(10)
        binding_content_layout.addWidget(self.binding_view, 3)
        binding_content_layout.addWidget(self.binding_tree, 2)
        self.binding_content.hide()

        self.binding_help_label = QLabel(
            "拖动画面中的控点校准位置；这里只修改当前回放记录，不会写入全局标注工具。"
        )
        self.binding_help_label.setWordWrap(True)
        self.capture_scene_button = QPushButton("抓取当前设备画面")
        self.capture_scene_button.setMinimumWidth(144)
        self.capture_scene_button.clicked.connect(self.capture_device_scene)
        self.reset_binding_button = QPushButton("撤销修改")
        self.reset_binding_button.clicked.connect(self.reset_binding)
        self.reset_binding_button.setEnabled(False)
        self.save_binding_button = QPushButton("保存控点绑定")
        self.save_binding_button.setMinimumWidth(120)
        self.save_binding_button.clicked.connect(self.save_binding)
        self.save_binding_button.setEnabled(False)
        binding_actions = QHBoxLayout()
        binding_actions.setContentsMargins(0, 0, 0, 0)
        binding_actions.addWidget(self.binding_help_label, 1)
        binding_actions.addWidget(self.capture_scene_button)
        binding_actions.addWidget(self.reset_binding_button)
        binding_actions.addWidget(self.save_binding_button)

        binding_group = QGroupBox("控点绑定")
        binding_layout = QVBoxLayout(binding_group)
        binding_layout.setContentsMargins(12, 10, 12, 12)
        binding_layout.addWidget(self.binding_empty_label)
        binding_layout.addWidget(self.binding_content, 1)
        binding_layout.addLayout(binding_actions)

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
        right_layout.addWidget(binding_group, 2)
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
        if self._binding_dirty:
            self.status_label.setText("当前控点有未保存修改；请先保存或撤销后再刷新记录。")
            return
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
        self._binding_source_pixmap = QPixmap()
        self._binding_points = {}
        self._binding_screen_size = (0, 0)
        self._binding_record = None
        self._binding_items.clear()
        self.binding_scene.clear()
        self.binding_tree.clear()
        self.binding_content.hide()
        self.binding_empty_label.show()
        self._set_binding_dirty(False)

    def _load_binding_snapshot(self, record: ReplayRecord):
        self._clear_binding_snapshot()
        raw_points, screen_size = load_recorded_control_points(record)
        scene_path = record.scene_view_path
        scene_pixmap = QPixmap(str(scene_path)) if scene_path is not None else QPixmap()
        self._binding_record = record
        self._binding_points = copy.deepcopy(dict(raw_points))
        self._binding_screen_size = (
            screen_size
            if screen_size[0] > 0 and screen_size[1] > 0
            else (
                (scene_pixmap.width(), scene_pixmap.height())
                if not scene_pixmap.isNull()
                else (0, 0)
            )
        )
        self._binding_source_pixmap = scene_pixmap
        point_count = self._rebuild_binding_editor()
        if scene_pixmap.isNull() and point_count == 0:
            self.status_label.setText("该记录没有可用的场景图或控点快照。")
            self.capture_scene_button.setEnabled(not self._replay_active)
            return
        if point_count:
            self.status_label.setText(
                f"已加载记录级场景图和 {point_count} 个控点；"
                "可直接拖动控点，或抓取当前设备画面后重新校准。"
            )
        else:
            self.status_label.setText(
                "已加载记录级场景图，但没有可校准的控点；抓图不会创建新控点。"
            )

    def _rebuild_binding_editor(self) -> int:
        self.binding_scene.clear()
        self.binding_tree.clear()
        self._binding_items.clear()
        pixmap = self._binding_source_pixmap
        scene_width = pixmap.width() if not pixmap.isNull() else self._binding_screen_size[0]
        scene_height = pixmap.height() if not pixmap.isNull() else self._binding_screen_size[1]
        if scene_width <= 0 or scene_height <= 0:
            self.binding_content.hide()
            self.binding_empty_label.show()
            self._update_binding_controls()
            return 0

        self.binding_scene.setSceneRect(0, 0, scene_width, scene_height)
        if not pixmap.isNull():
            background = self.binding_scene.addPixmap(pixmap)
            background.setZValue(-10)
        else:
            background = self.binding_scene.addRect(
                0,
                0,
                scene_width,
                scene_height,
                QPen(QColor("#555")),
                QBrush(QColor("#282d34")),
            )
            background.setZValue(-10)

        valid_count = 0
        for raw_key, raw_point in self._binding_points.items():
            if not isinstance(raw_point, Mapping):
                continue
            normalized = self._normalized_position(
                raw_point,
                self._binding_screen_size,
            )
            if normalized is None:
                continue
            key = str(raw_key)
            item = ReplayBindingPointItem(
                key,
                QPointF(normalized[0] * scene_width, normalized[1] * scene_height),
                scene_width,
                scene_height,
                moved=self._binding_point_moved,
                selected=self._binding_point_selected,
            )
            self.binding_scene.addItem(item)
            self._binding_items[key] = item
            tree_item = QTreeWidgetItem(
                [
                    key,
                    str(raw_point.get("scene") or "未命名"),
                    self._binding_position_text(raw_point, normalized),
                ]
            )
            tree_item.setData(0, Qt.ItemDataRole.UserRole, key)
            self.binding_tree.addTopLevelItem(tree_item)
            valid_count += 1

        self.binding_empty_label.hide()
        self.binding_content.show()
        self.binding_view.fitInView(
            self.binding_scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        if self.binding_tree.topLevelItemCount():
            self.binding_tree.setCurrentItem(self.binding_tree.topLevelItem(0))
        self._update_binding_controls()
        return valid_count

    @staticmethod
    def _binding_position_text(
        point: Mapping[str, Any], normalized: tuple[float, float]
    ) -> str:
        position = point.get("position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            try:
                return f"{int(position[0])}, {int(position[1])}"
            except (TypeError, ValueError):
                pass
        return f"{normalized[0]:.3f}, {normalized[1]:.3f}"

    def _binding_point_moved(self, key: str, position: QPointF):
        point = self._binding_points.get(key)
        scene_rect = self.binding_scene.sceneRect()
        if not isinstance(point, dict) or scene_rect.width() <= 0 or scene_rect.height() <= 0:
            return
        norm_x = min(max(position.x() / scene_rect.width(), 0.0), 1.0)
        norm_y = min(max(position.y() / scene_rect.height(), 0.0), 1.0)
        screen_width = self._binding_screen_size[0] or int(round(scene_rect.width()))
        screen_height = self._binding_screen_size[1] or int(round(scene_rect.height()))
        point["normalized_position"] = [norm_x, norm_y]
        point["position"] = [
            min(max(int(round(norm_x * screen_width)), 0), max(screen_width - 1, 0)),
            min(max(int(round(norm_y * screen_height)), 0), max(screen_height - 1, 0)),
        ]
        self._refresh_binding_tree_item(key)
        self._set_binding_dirty(True)
        self.status_label.setText(
            f"已调整控点 {key}；请保存绑定后再开始回放。"
        )

    def _refresh_binding_tree_item(self, key: str):
        point = self._binding_points.get(key)
        if not isinstance(point, Mapping):
            return
        normalized = self._normalized_position(point, self._binding_screen_size)
        if normalized is None:
            return
        for index in range(self.binding_tree.topLevelItemCount()):
            item = self.binding_tree.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == key:
                item.setText(2, self._binding_position_text(point, normalized))
                return

    def _binding_tree_selection_changed(self):
        if self._syncing_binding_selection:
            return
        selected = self.binding_tree.selectedItems()
        if not selected:
            return
        key = selected[0].data(0, Qt.ItemDataRole.UserRole)
        point_item = self._binding_items.get(str(key))
        if point_item is None:
            return
        self._syncing_binding_selection = True
        self.binding_scene.clearSelection()
        point_item.setSelected(True)
        self._syncing_binding_selection = False

    def _binding_point_selected(self, key: str):
        if self._syncing_binding_selection:
            return
        for index in range(self.binding_tree.topLevelItemCount()):
            item = self.binding_tree.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == key:
                self._syncing_binding_selection = True
                self.binding_tree.setCurrentItem(item)
                self._syncing_binding_selection = False
                return

    def _set_binding_dirty(self, dirty: bool):
        self._binding_dirty = bool(dirty)
        self._update_binding_controls()

    def _update_binding_controls(self):
        capture_busy = self._capture_process.state() != QProcess.ProcessState.NotRunning
        has_record = self._binding_record is not None
        editable = has_record and not self._replay_active and not capture_busy
        self.capture_scene_button.setEnabled(editable)
        self.save_binding_button.setEnabled(editable and self._binding_dirty)
        self.reset_binding_button.setEnabled(editable and self._binding_dirty)
        self.record_tree.setEnabled(not self._binding_dirty and not capture_busy)
        self.refresh_button.setEnabled(not self._binding_dirty and not capture_busy)
        self.start_replay_button.setEnabled(
            self.selected_record is not None
            and not self._replay_active
            and not self._binding_dirty
            and not capture_busy
        )

    def reset_binding(self):
        record = self._binding_record
        if record is not None:
            self._remove_capture_temp_file()
            self._load_binding_snapshot(record)

    def capture_device_scene(self):
        if self._binding_record is None:
            return
        if self._capture_process.state() != QProcess.ProcessState.NotRunning:
            return
        token = uuid.uuid4().hex
        self._capture_remote_path = f"/data/local/tmp/autogame_replay_binding_{token}.jpeg"
        self._capture_local_path = Path(tempfile.gettempdir()) / (
            f"autogame_replay_binding_{token}.jpeg"
        )
        self._capture_stage = "snapshot"
        self.capture_scene_button.setText("正在抓图…")
        self.status_label.setText("正在从当前连接设备抓取画面……")
        self._update_binding_controls()
        self._capture_process.start(
            resolve_hdc_executable(),
            ["shell", "snapshot_display", "-f", self._capture_remote_path],
        )

    def _capture_process_finished(self, exit_code: int, _exit_status):
        stage = self._capture_stage
        if not stage:
            return
        output = (
            bytes(self._capture_process.readAllStandardError())
            + bytes(self._capture_process.readAllStandardOutput())
        ).decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            self._capture_failed(output or f"HDC {stage} 命令执行失败。")
            return
        if stage == "snapshot":
            self._capture_stage = "receive"
            QTimer.singleShot(
                0,
                lambda: self._capture_process.start(
                    resolve_hdc_executable(),
                    [
                        "file",
                        "recv",
                        self._capture_remote_path,
                        str(self._capture_local_path),
                    ],
                ),
            )
            return
        if stage != "receive" or self._capture_local_path is None:
            self._capture_failed("抓图流程状态异常。")
            return
        pixmap = QPixmap(str(self._capture_local_path))
        if pixmap.isNull():
            self._capture_failed("HDC 已返回截图，但图片无法读取。")
            return
        self._cleanup_remote_capture()
        self._capture_stage = ""
        self.capture_scene_button.setText("抓取当前设备画面")
        self._apply_captured_pixmap(pixmap)

    def _apply_captured_pixmap(self, pixmap: QPixmap):
        previous_screen_size = self._binding_screen_size
        next_screen_size = (pixmap.width(), pixmap.height())
        for point in self._binding_points.values():
            if not isinstance(point, dict):
                continue
            normalized = self._normalized_position(point, previous_screen_size)
            if normalized is None:
                continue
            point["normalized_position"] = [normalized[0], normalized[1]]
            point["position"] = [
                min(
                    max(int(round(normalized[0] * next_screen_size[0])), 0),
                    max(next_screen_size[0] - 1, 0),
                ),
                min(
                    max(int(round(normalized[1] * next_screen_size[1])), 0),
                    max(next_screen_size[1] - 1, 0),
                ),
            ]
        self._binding_source_pixmap = pixmap
        self._binding_screen_size = next_screen_size
        count = self._rebuild_binding_editor()
        self._set_binding_dirty(True)
        self.status_label.setText(
            f"已抓取 {pixmap.width()}×{pixmap.height()} 当前画面并加载 {count} 个控点；"
            "请拖动校准后保存。"
        )

    def _capture_process_error(self, _error):
        if not self._capture_stage:
            return
        self._capture_failed(self._capture_process.errorString() or "HDC 进程启动失败。")

    def _capture_failed(self, message: str):
        self._cleanup_remote_capture()
        self._remove_capture_temp_file()
        self._capture_stage = ""
        self.capture_scene_button.setText("抓取当前设备画面")
        self._update_binding_controls()
        self.status_label.setText(f"抓图失败：{message}")
        QMessageBox.warning(self, "抓图失败", message)

    def _cleanup_remote_capture(self):
        if not self._capture_remote_path:
            return
        QProcess.startDetached(
            resolve_hdc_executable(),
            ["shell", "rm", self._capture_remote_path],
        )
        self._capture_remote_path = ""

    def _remove_capture_temp_file(self):
        path, self._capture_local_path = self._capture_local_path, None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def save_binding(self) -> bool:
        record = self._binding_record
        if record is None or not self._binding_dirty:
            return False
        if self._binding_source_pixmap.isNull():
            QMessageBox.warning(self, "无法保存", "请先加载或抓取一张有效场景图。")
            return False
        scene_temp_path = record.directory / ".scene_view.tmp.png"
        scene_view_path = record.directory / "scene_view.png"
        try:
            if not self._binding_source_pixmap.save(
                str(scene_temp_path),
                "PNG",
            ):
                raise OSError(f"场景图写入失败：{scene_temp_path}")
            save_recorded_control_points(
                record,
                self._binding_points,
                self._binding_screen_size,
            )
            scene_temp_path.replace(scene_view_path)
        except (OSError, ValueError) as exc:
            try:
                scene_temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._remove_capture_temp_file()
        self._set_binding_dirty(False)
        self.status_label.setText(
            f"已保存 {len(self._binding_items)} 个记录级控点和当前场景图；"
            "现在可以开始回放。"
        )
        return True

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
        if (
            record is None
            or self._replay_active
            or self._binding_dirty
            or self._capture_process.state() != QProcess.ProcessState.NotRunning
        ):
            return
        self._pause_video()
        self.replayRequested.emit(record)

    def set_replay_active(self, active: bool):
        self._replay_active = bool(active)
        self._update_binding_controls()
        self.status_label.setText(
            "回放窗口已启动，请在回放结束后返回这里选择其他记录。"
            if self._replay_active
            else "选择一条记录，查看视频后即可开始回放。"
        )

    def stop(self):
        self._release_video()
        if self._capture_process.state() != QProcess.ProcessState.NotRunning:
            self._capture_stage = ""
            self._capture_process.kill()
            self._capture_process.waitForFinished(1000)
        self._cleanup_remote_capture()
        if not self._binding_dirty:
            self._remove_capture_temp_file()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_video_pixmap()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
