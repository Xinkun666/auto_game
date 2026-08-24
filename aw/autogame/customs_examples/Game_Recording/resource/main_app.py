"""Game_Recording 统一录制/回放界面，共用一条 HOS 投屏连接。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aw.autogame.customs_examples.Game_Recording import info

from .app import RecorderWindow, _prefer_bundled_pyqt_plugins
from .binding_config import BindingConfigError
from .binding_dialog import BindingDialog
from .layout import LayoutError
from .replay import ReplayError, load_replay_events, load_replay_layout
from .replay_app import ReplaySelectionWidget, ReplayThread
from .touch_controller import SingleTouchKeyboardController


LOGGER = logging.getLogger("GameRecordingMain")


class SharedReplayPanel(QWidget):
    """回放页：选择历史记录后复用录制页的 HOS 流与触控后端。"""

    def __init__(self, recorder_window: RecorderWindow, records_root: Path, parent=None):
        super().__init__(parent)
        self.recorder_window = recorder_window
        self.records_root = Path(records_root)
        self.replay_thread = None
        self._replay_controller = None
        self._recorded_capture = None
        self._closing = False

        self.selector = ReplaySelectionWidget(self.records_root, self)
        self.selector.selectionChanged.connect(self._update_start_state)
        self.selector.recordActivated.connect(self.start_replay)

        self.start_button = QPushButton("开始回放所选记录")
        self.start_button.clicked.connect(self.start_replay)
        self.start_button.setEnabled(False)
        self.status_label = QLabel("选择一条录制记录后开始回放。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555;")
        self.preview_label = QLabel("复用录制页的手机画面")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 236)
        self.preview_label.setStyleSheet("background: #15191e; color: #ddd;")
        self.recorded_video_label = QLabel("尚未开始原录制视频播放")
        self.recorded_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recorded_video_label.setMinimumSize(420, 236)
        self.recorded_video_label.setStyleSheet("background: #15191e; color: #ddd;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.action_label = QLabel("尚未开始回放")

        self.selection_page = QWidget()
        selection_layout = QVBoxLayout(self.selection_page)
        selection_layout.addWidget(self.selector)

        self.comparison_page = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_page)
        videos = QHBoxLayout()
        recorded_column = QVBoxLayout()
        recorded_column.addWidget(QLabel("原录制视频"))
        recorded_column.addWidget(self.recorded_video_label, 1)
        current_column = QVBoxLayout()
        current_column.addWidget(QLabel("当前回放画面"))
        current_column.addWidget(self.preview_label, 1)
        videos.addLayout(recorded_column, 1)
        videos.addLayout(current_column, 1)
        comparison_layout.addLayout(videos, 1)
        comparison_layout.addWidget(self.progress_bar)
        comparison_layout.addWidget(self.action_label)
        self.return_button = QPushButton("返回记录选择")
        self.return_button.clicked.connect(self._return_to_selection)
        comparison_layout.addWidget(self.return_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.selection_page)
        self.pages.addWidget(self.comparison_page)
        controls = QHBoxLayout()
        controls.addWidget(self.status_label, 1)
        controls.addWidget(self.start_button)

        root = QVBoxLayout(self)
        root.addWidget(self.pages, 1)
        root.addLayout(controls)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self.preview_timer.start(50)
        self.recorded_video_timer = QTimer(self)
        self.recorded_video_timer.timeout.connect(self._refresh_recorded_video)
        self._update_start_state()

    def _set_status(self, text: str, error: bool = False):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "color: #b00020; font-weight: 600;" if error else "color: #167c36; font-weight: 600;"
        )

    def _update_start_state(self):
        can_start = (
            not self._closing
            and self.selector.selected_record is not None
            and not (
                self.replay_thread is not None and self.replay_thread.isRunning()
            )
        )
        self.start_button.setEnabled(can_start)

    def _refresh_preview(self):
        if self._closing:
            return
        frame = self.recorder_window.frame_pump.latest_frame()
        if frame is None:
            stream_error = getattr(self.recorder_window.stream_client, "_last_error", None)
            if stream_error is not None:
                self.preview_label.setText(f"HOS 画面不可用：{stream_error}")
            return
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _start_recorded_video(self, video_path: Path):
        """从回放动作开始时播放同一条录制视频，不影响实时回放。"""
        self._stop_recorded_video()
        self.recorded_video_label.setPixmap(QPixmap())
        if not video_path.is_file():
            self.recorded_video_label.setText("该记录没有 video.mp4，仍会继续实时回放。")
            return
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            self.recorded_video_label.setText("无法打开该记录的 video.mp4，仍会继续实时回放。")
            return
        self._recorded_capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        interval_ms = max(15, min(1000, int(round(1000 / max(1.0, fps)))))
        self.recorded_video_timer.start(interval_ms)
        self._refresh_recorded_video()

    def _refresh_recorded_video(self):
        capture = self._recorded_capture
        if capture is None:
            return
        ok, frame = capture.read()
        if not ok:
            self._stop_recorded_video()
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        image = QImage(
            frame_rgb.data,
            width,
            height,
            int(frame_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.recorded_video_label.setText("")
        self.recorded_video_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.recorded_video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _stop_recorded_video(self):
        self.recorded_video_timer.stop()
        capture, self._recorded_capture = self._recorded_capture, None
        if capture is not None:
            capture.release()

    def _return_to_selection(self):
        if self.replay_thread is not None and self.replay_thread.isRunning():
            self._set_status("回放进行中，请等待结束后再返回选择。", error=True)
            return
        self._stop_recorded_video()
        self.pages.setCurrentWidget(self.selection_page)
        self.selector.refresh_records()
        self._update_start_state()

    def start_replay(self):
        record = self.selector.selected_record
        if record is None:
            self._set_status("请先选择一条回放记录。", error=True)
            return
        if self.recorder_window.recorder.is_recording:
            self._set_status("请先在“录制”页关闭当前录制，再开始回放。", error=True)
            return
        if self.recorder_window.frame_pump.latest_frame() is None:
            self._set_status("手机画面尚未就绪，请稍候再开始回放。", error=True)
            return
        try:
            events = load_replay_events(record)
            key_points = load_replay_layout(
                record,
                info,
                *self.recorder_window.screen_size,
            )
            self.recorder_window._release_controls()
            self._replay_controller = SingleTouchKeyboardController(
                self.recorder_window.touch_client,
                key_points,
                screen_size=self.recorder_window.screen_size,
            )
            self.replay_thread = ReplayThread(
                events=events,
                duration_seconds=record.duration_seconds,
                controller=self._replay_controller,
                stream_client=self.recorder_window.stream_client,
                parent=self,
            )
            self.replay_thread.progressChanged.connect(self._update_progress)
            self.replay_thread.replayEnded.connect(self._replay_ended)
            self.progress_bar.setValue(0)
            self.action_label.setText("正在等待第一条动作")
            self.pages.setCurrentWidget(self.comparison_page)
            self._start_recorded_video(record.directory / "video.mp4")
            self._set_status(f"正在回放：{record.directory.name}")
            self._update_start_state()
            self.replay_thread.start()
        except (ReplayError, RuntimeError, ValueError, OSError) as exc:
            self._set_status(f"无法开始回放：{exc}", error=True)

    def _update_progress(self, progress: float, action: str):
        self.progress_bar.setValue(int(round(progress * 1000)))
        self.action_label.setText(action)

    def _replay_ended(self, success: bool, message: str):
        if self._closing:
            return
        self.progress_bar.setValue(1000 if success else self.progress_bar.value())
        self._stop_recorded_video()
        self._set_status(message, error=not success)
        self._update_start_state()

    def stop(self):
        self._closing = True
        self.preview_timer.stop()
        self._stop_recorded_video()
        thread = self.replay_thread
        if thread is not None and thread.isRunning():
            thread.requestInterruption()
            thread.wait(2000)
        if self._replay_controller is not None:
            try:
                self._replay_controller.release_all()
            except Exception:
                LOGGER.exception("关闭统一回放页时释放触控失败")


class GameRecordingMainWindow(QMainWindow):
    """绑定完成后的统一主窗口。"""

    def __init__(
        self,
        output_root: Path,
        fps: float = 15.0,
        runtime_log_path: Path | None = None,
        hilog_capture=None,
        video_so: str = "auto",
        touch_backend: str = "hos",
        sendevent_device: str = "",
        sendevent_max_x: int | None = None,
        sendevent_max_y: int | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Game Recording - 录制与回放")
        self.resize(1280, 820)
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self.recorder_window = RecorderWindow(
            output_root=output_root,
            fps=fps,
            runtime_log_path=runtime_log_path,
            hilog_capture=hilog_capture,
            video_so=video_so,
            touch_backend=touch_backend,
            sendevent_device=sendevent_device,
            sendevent_max_x=sendevent_max_x,
            sendevent_max_y=sendevent_max_y,
            parent=self.tabs,
        )
        self.replay_panel = SharedReplayPanel(
            self.recorder_window,
            records_root=Path(output_root).parent,
            parent=self.tabs,
        )
        self.tabs.addTab(self.recorder_window, "录制")
        self.tabs.addTab(self.replay_panel, "回放")
        self.tabs.currentChanged.connect(self._tab_changed)

    def _tab_changed(self, index: int):
        if index == 0:
            self.recorder_window.setFocus()
        else:
            self.replay_panel.selector.refresh_records()

    def closeEvent(self, event: QCloseEvent):
        self.replay_panel.stop()
        if not self.recorder_window._closed:
            self.recorder_window.close()
        event.accept()


def run(
    output_root: Path,
    fps: float = 15.0,
    runtime_log_path: Path | None = None,
    hilog_capture=None,
    video_so: str = "auto",
    touch_backend: str = "hos",
    sendevent_device: str = "",
    sendevent_max_x: int | None = None,
    sendevent_max_y: int | None = None,
) -> int:
    """统一入口：先绑定，再进入录制/回放标签页。"""
    _prefer_bundled_pyqt_plugins()
    app = QApplication.instance() or QApplication([])
    try:
        binding_dialog = BindingDialog(info)
        binding_dialog.show()
        binding_dialog.raise_()
        binding_dialog.activateWindow()
        if binding_dialog.exec() != QDialog.DialogCode.Accepted:
            print("[Game Recording] 已取消按键绑定，未启动抓流。", flush=True)
            return 0
        window = GameRecordingMainWindow(
            output_root=Path(output_root),
            fps=fps,
            runtime_log_path=runtime_log_path,
            hilog_capture=hilog_capture,
            video_so=video_so,
            touch_backend=touch_backend,
            sendevent_device=sendevent_device,
            sendevent_max_x=sendevent_max_x,
            sendevent_max_y=sendevent_max_y,
        )
    except (BindingConfigError, LayoutError) as exc:
        raise SystemExit(f"键位布局不可用：{exc}") from exc
    except (ReplayError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"统一界面启动失败：{exc}") from exc
    window.show()
    window.activateWindow()
    window.recorder_window.setFocus()
    return int(app.exec())
