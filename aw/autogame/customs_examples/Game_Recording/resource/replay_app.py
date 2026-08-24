"""Game_Recording 记录选择对话框和 HOS 回放窗口。"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aw.autogame.customs_examples.Game_Recording import info
from aw.autogame.tools.Utils import get_resolution

from .app import _prefer_bundled_pyqt_plugins, configure_fail_fast_stream
from .replay import (
    ReplayError,
    ReplayEvent,
    ReplayRecord,
    discover_replay_records,
    load_replay_events,
    load_replay_layout,
)
from .touch_controller import SingleTouchKeyboardController


LOGGER = logging.getLogger("GameReplay")


class ReplaySelectionWidget(QWidget):
    """可嵌入页面或对话框的历史回放记录选择器。"""

    selectionChanged = pyqtSignal()
    recordActivated = pyqtSignal()

    def __init__(self, records_root: Path, parent=None):
        super().__init__(parent)
        self.records_root = Path(records_root)
        self.records = discover_replay_records(self.records_root)
        self._preview_pixmap = QPixmap()

        self.setWindowTitle("Game Recording - 选择回放记录")
        self.resize(980, 620)
        self.setMinimumSize(760, 480)

        title = QLabel("请选择一条之前的录制记录")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root_hint = QLabel(f"记录目录：{self.records_root}")
        root_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root_hint.setStyleSheet("color: #666;")

        self.record_list = QListWidget()
        self.record_list.currentItemChanged.connect(self._selection_changed)
        self.record_list.itemDoubleClicked.connect(lambda _item: self.recordActivated.emit())

        self.preview_label = QLabel("无初始画面")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 240)
        self.preview_label.setStyleSheet("background: #1d2228; color: #ddd;")
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        left = QVBoxLayout()
        left.addWidget(QLabel("可回放记录（新的在前）"))
        left.addWidget(self.record_list, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(self.preview_label, 1)
        right.addWidget(self.detail_label)
        right_widget = QWidget()
        right_widget.setLayout(right)

        content = QHBoxLayout()
        content.addWidget(left_widget, 1)
        content.addWidget(right_widget, 1)

        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet("color: #b00020;")
        if not self.records:
            self.empty_label.setText(
                "没有找到可回放的记录。请先运行 start_record.py，"
                "在录制窗口中点击“开启录制”和“关闭录制”。"
            )

        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(root_hint)
        root.addLayout(content, 1)
        root.addWidget(self.empty_label)
        self.refresh_records()

    @property
    def selected_record(self) -> ReplayRecord | None:
        item = self.record_list.currentItem()
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        return self.records[int(index)] if isinstance(index, int) else None

    def _selection_changed(self, current, previous):
        del previous
        record = self.selected_record if current is not None else None
        if record is None:
            self.detail_label.clear()
            self.preview_label.setText("无初始画面")
            self.preview_label.setPixmap(QPixmap())
            self.selectionChanged.emit()
            return
        source_label = "精确键盘事件" if record.action_format == "raw" else "状态化回放步骤"
        self.detail_label.setText(
            f"录制时间：{record.display_time}\n"
            f"时长：{record.duration_seconds:.2f} 秒\n"
            f"动作数：{record.action_count}\n"
            f"视频帧数：{record.frame_count}\n"
            f"结束原因：{record.stop_reason}\n"
            f"动作来源：{source_label}\n"
            f"路径：{record.directory}"
        )
        self._preview_pixmap = (
            QPixmap(str(record.initial_view_path))
            if record.initial_view_path is not None
            else QPixmap()
        )
        self._refresh_preview()
        self.selectionChanged.emit()

    def refresh_records(self):
        """重新扫描，以便同一主窗口内刚完成的录制立即可回放。"""
        selected = self.selected_record
        selected_path = selected.directory if selected is not None else None
        self.records = discover_replay_records(self.records_root)
        self.record_list.blockSignals(True)
        self.record_list.clear()
        selected_index = 0
        for index, record in enumerate(self.records):
            item = QListWidgetItem(record.title)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(str(record.directory))
            self.record_list.addItem(item)
            if selected_path is not None and record.directory == selected_path:
                selected_index = index
        if self.records:
            self.record_list.setCurrentRow(selected_index)
        self.record_list.blockSignals(False)
        self.empty_label.setText(
            ""
            if self.records
            else "没有找到可回放的记录。请先在“录制”页完成一次录制。"
        )
        self._selection_changed(self.record_list.currentItem(), None)

    def _refresh_preview(self):
        if self._preview_pixmap.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("该记录没有 initial_view.png")
            return
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            self._preview_pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

class ReplaySelectionDialog(QDialog):
    """兼容旧 start_replay.py 的独立选择对话框。"""

    def __init__(self, records_root: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Recording - 选择回放记录")
        self.resize(980, 620)
        self.setMinimumSize(760, 480)
        self.selector = ReplaySelectionWidget(records_root, self)
        self.selector.selectionChanged.connect(self._update_start_button)
        self.selector.recordActivated.connect(self._accept_selected)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.start_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.start_button.setText("开始回放")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._accept_selected)
        self.buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(self.selector, 1)
        root.addWidget(self.buttons)
        self._update_start_button()

    @property
    def selected_record(self) -> ReplayRecord | None:
        return self.selector.selected_record

    def _update_start_button(self):
        self.start_button.setEnabled(self.selected_record is not None)

    def _accept_selected(self):
        if self.selected_record is not None:
            self.accept()


class PreviewFramePump(threading.Thread):
    def __init__(self, buffer):
        super().__init__(name="GameReplayFramePump", daemon=True)
        self.buffer = buffer
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None

    def run(self):
        while not self._stop_event.is_set():
            frame = self.buffer.get_latest(timeout=1.0)
            if frame is None:
                continue
            rgb = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                continue
            with self._frame_lock:
                self._latest_frame = rgb.copy()

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._stop_event.set()


class ReplayThread(QThread):
    progressChanged = pyqtSignal(float, str)
    replayEnded = pyqtSignal(bool, str)

    def __init__(
        self,
        events: list[ReplayEvent],
        duration_seconds: float,
        controller: SingleTouchKeyboardController,
        stream_client,
        parent=None,
    ):
        super().__init__(parent)
        self.events = list(events)
        self.duration_seconds = max(
            float(duration_seconds),
            self.events[-1].timestamp if self.events else 0.0,
        )
        self.controller = controller
        self.stream_client = stream_client

    def _wait_until(self, deadline: float) -> bool:
        while not self.isInterruptionRequested():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.02, remaining))
        return False

    def run(self):
        guard_started = False
        success = False
        message = "回放已取消"
        try:
            begin_guard = getattr(self.stream_client, "begin_touch_replay", None)
            if callable(begin_guard):
                guard_started = bool(begin_guard("Game Recording replay"))
            started_at = time.monotonic()
            for index, event in enumerate(self.events):
                if not self._wait_until(started_at + event.timestamp):
                    return
                if event.event == "press":
                    self.controller.press(event.key)
                elif event.event == "release":
                    self.controller.release(event.key)
                elif event.event == "tap":
                    self.controller.tap(event.key)
                else:
                    if event.normalized_position is not None:
                        self.controller.move_active_control_to_normalized(
                            *event.normalized_position
                        )
                    else:
                        self.controller.nudge_active_control(event.key)
                progress = (
                    event.timestamp / self.duration_seconds
                    if self.duration_seconds > 0
                    else (index + 1) / len(self.events)
                )
                self.progressChanged.emit(
                    min(max(progress, 0.0), 1.0),
                    f"{event.timestamp:.2f}s  {event.event} {event.key}",
                )
            if not self._wait_until(started_at + self.duration_seconds):
                return
            success = True
            message = f"回放完成，总时长 {self.duration_seconds:.2f} 秒"
        except Exception as exc:
            LOGGER.exception("执行回放失败")
            message = f"回放失败：{exc}"
        finally:
            try:
                self.controller.release_all()
            except Exception as exc:
                LOGGER.exception("回放结束时释放触控失败")
                if success:
                    success = False
                    message = f"回放结束时释放触控失败：{exc}"
            if guard_started and not self.isInterruptionRequested():
                try:
                    end_guard = getattr(self.stream_client, "end_touch_replay", None)
                    if callable(end_guard) and not bool(end_guard()):
                        success = False
                        message = "回放已执行，但结束后 HOS 画面没有恢复。"
                except Exception as exc:
                    LOGGER.exception("结束 HOS 回放保护失败")
                    success = False
                    message = f"回放后 HOS 连接异常：{exc}"
            self.replayEnded.emit(success, message)


class ReplayWindow(QMainWindow):
    def __init__(
        self,
        record: ReplayRecord,
        video_so: str = "auto",
        touch_backend: str = "hos",
        sendevent_device: str = "",
        sendevent_max_x: int | None = None,
        sendevent_max_y: int | None = None,
    ):
        super().__init__()
        if shutil.which("hdc") is None:
            raise RuntimeError(
                "当前终端找不到 hdc。请先配置 HDC，并确保手机已连接。"
            )
        from aw.autogame.stream_client.stream_client import FrameBuffer, HOSScrcpyStreamClient

        self.record = record
        self.events = load_replay_events(record)
        width, height = get_resolution()
        self.screen_size = (int(width), int(height))
        self.key_points = load_replay_layout(record, info, *self.screen_size)
        self.touch_backend_name = str(touch_backend or "hos").strip().lower()
        self.sendevent_touch = None
        self._closed = False
        self._replay_started = False
        self._stream_stopped = False
        self._stream_failure_message = ""

        self.buffer = FrameBuffer(size=5)
        requested_video_so = str(video_so or "auto").strip()
        force_video_so = "" if requested_video_so == "reuse" else requested_video_so
        self.stream_client = configure_fail_fast_stream(
            HOSScrcpyStreamClient(self.buffer, force_video_so=force_video_so)
        )
        touch_client = self.stream_client
        if self.touch_backend_name == "sendevent":
            from .sendevent_controller import SendeventTouchAdapter

            self.sendevent_touch = SendeventTouchAdapter(
                screen_size=self.screen_size,
                device_id=getattr(self.stream_client, "sn", ""),
                input_device=sendevent_device,
                abs_max_x=sendevent_max_x,
                abs_max_y=sendevent_max_y,
            )
            touch_client = self.sendevent_touch
        elif self.touch_backend_name != "hos":
            raise ValueError(f"不支持的触控后端：{self.touch_backend_name}")
        self.controller = SingleTouchKeyboardController(
            touch_client,
            self.key_points,
            screen_size=self.screen_size,
        )
        self.frame_pump = PreviewFramePump(self.buffer)
        self.replay_thread = ReplayThread(
            events=self.events,
            duration_seconds=record.duration_seconds,
            controller=self.controller,
            stream_client=self.stream_client,
            parent=self,
        )
        self.replay_thread.progressChanged.connect(self._update_progress)
        self.replay_thread.replayEnded.connect(self._replay_ended)

        self.setWindowTitle(f"Game Recording - 回放 {record.display_time}")
        self.resize(980, 680)
        self.status_label = QLabel("正在连接华为手机画面，首帧到达后自动开始回放……")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: 600;")
        self.record_label = QLabel(f"回放记录：{record.directory}")
        self.record_label.setWordWrap(True)
        self.record_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label = QLabel("等待 HOScrcpy 首帧")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background: #15191e; color: #ddd;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.action_label = QLabel("尚未开始执行动作")
        self.close_button = QPushButton("取消回放")
        self.close_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.record_label)
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.progress_bar)
        bottom = QHBoxLayout()
        bottom.addWidget(self.action_label, 1)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(50)
        self.frame_pump.start()
        print(
            f"[Game Replay] 选中记录：{record.directory}\n"
            f"[Game Replay] 触控后端：{self.touch_backend_name}\n"
            f"[Game Replay] HOS 投屏 SO 策略：{force_video_so or 'reuse-device-existing'}",
            flush=True,
        )
        self.stream_client.start_backend()

    def _refresh(self):
        if self._closed:
            return
        frame = self.frame_pump.latest_frame()
        if frame is not None:
            height, width = frame.shape[:2]
            image = QImage(
                frame.data,
                width,
                height,
                int(frame.strides[0]),
                QImage.Format.Format_RGB888,
            ).copy()
            self.preview_label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            if not self._replay_started:
                self._replay_started = True
                self.status_label.setText("手机画面已就绪，正在回放……")
                self.status_label.setStyleSheet("color: #167c36; font-weight: 600;")
                self.replay_thread.start()
        stream_error = getattr(self.stream_client, "_last_error", None)
        if stream_error is not None and not self._stream_failure_message:
            self._stream_failure_message = f"HOS 视频连接失败：{stream_error}"
            if self.replay_thread.isRunning():
                self.replay_thread.requestInterruption()
            else:
                self._replay_ended(False, self._stream_failure_message)

    def _update_progress(self, progress: float, action: str):
        self.progress_bar.setValue(int(round(progress * 1000)))
        self.action_label.setText(action)

    def _replay_ended(self, success: bool, message: str):
        if self._closed:
            return
        if self._stream_failure_message:
            success = False
            message = self._stream_failure_message
        self.refresh_timer.stop()
        if success:
            self.progress_bar.setValue(1000)
            self.status_label.setStyleSheet("color: #167c36; font-weight: 600;")
            self.close_button.setText("关闭")
        else:
            self.status_label.setStyleSheet("color: #b00020; font-weight: 600;")
            self.close_button.setText("关闭")
        self.status_label.setText(message)
        self._stop_stream()

    def _stop_stream(self):
        if self._stream_stopped:
            return
        self._stream_stopped = True
        self.frame_pump.stop()
        try:
            self.stream_client.stop()
        finally:
            self.frame_pump.join(timeout=2.0)
            if self.sendevent_touch is not None:
                backend = self.sendevent_touch
                self.sendevent_touch = None
                backend.close()

    def closeEvent(self, event: QCloseEvent):
        if not self._closed:
            self._closed = True
            self.refresh_timer.stop()
            if self.replay_thread.isRunning():
                self.replay_thread.requestInterruption()
                self.replay_thread.wait(2000)
            try:
                self.controller.release_all()
            except Exception:
                LOGGER.exception("关闭回放窗口时释放触控失败")
            self._stop_stream()
        event.accept()


def run(
    records_root: Path,
    video_so: str = "auto",
    touch_backend: str = "hos",
    sendevent_device: str = "",
    sendevent_max_x: int | None = None,
    sendevent_max_y: int | None = None,
) -> int:
    _prefer_bundled_pyqt_plugins()
    app = QApplication.instance() or QApplication([])
    try:
        dialog = ReplaySelectionDialog(records_root)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            print("[Game Replay] 已取消，未连接手机。", flush=True)
            return 0
        record = dialog.selected_record
        if record is None:
            return 0
        window = ReplayWindow(
            record=record,
            video_so=video_so,
            touch_backend=touch_backend,
            sendevent_device=sendevent_device,
            sendevent_max_x=sendevent_max_x,
            sendevent_max_y=sendevent_max_y,
        )
    except (ReplayError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"回放启动失败：{exc}") from exc
    window.show()
    window.activateWindow()
    return int(app.exec())
