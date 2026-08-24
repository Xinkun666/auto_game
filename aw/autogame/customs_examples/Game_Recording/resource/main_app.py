"""Game_Recording 统一录制/回放界面，共用一条 HOS 投屏连接。"""

from __future__ import annotations

import logging
import time
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
    QMessageBox,
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
from .replay_compare import ReplayComparisonPanel
from .replay_video import ReplayVideoRecorder
from .touch_controller import SingleTouchKeyboardController


LOGGER = logging.getLogger("GameRecordingMain")
GAME_RECORDING_PROJECT_DIR = Path(__file__).resolve().parents[1]


class SharedReplayPanel(QWidget):
    """回放页：选择历史记录后复用录制页的 HOS 流与触控后端。"""

    def __init__(self, recorder_window: RecorderWindow, records_root: Path, parent=None):
        super().__init__(parent)
        self.recorder_window = recorder_window
        self.records_root = Path(records_root)
        self.replay_thread = None
        self._replay_controller = None
        self._recorded_capture = None
        self._recorded_frame_count = 0
        self._recorded_frame_index = -1
        self._recorded_duration_seconds = 0.0
        self._comparison_started_at = None
        self._pending_recorded_video = None
        self._replay_video_recorder = None
        self._active_source_record = None
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
        if self._replay_video_recorder is not None:
            try:
                self._replay_video_recorder.accept_frame(frame)
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.exception("保存回放视频失败")
                self._set_status(f"回放仍在继续，但视频保存失败：{exc}", error=True)
                self._replay_video_recorder.stop(success=False)
                self._replay_video_recorder = None
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

    def _start_recorded_video(
        self,
        video_path: Path,
        duration_seconds: float,
        timeline_started_at: float,
    ):
        """按回放时间轴播放视频，而非按视频文件标称 FPS 播放。"""
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
        self._recorded_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if self._recorded_frame_count <= 0:
            self._stop_recorded_video()
            self.recorded_video_label.setText("原录制视频没有可播放帧，仍会继续实时回放。")
            return
        self._recorded_duration_seconds = max(0.001, float(duration_seconds))
        self._recorded_frame_index = -1
        self._comparison_started_at = float(timeline_started_at)
        # 每次按相同的已过回放时间定位视频帧，避免实际取帧率低于标称 FPS 时越播越慢。
        self.recorded_video_timer.start(30)
        self._refresh_recorded_video()

    def _start_pending_recorded_video(self, timeline_started_at: float):
        pending = self._pending_recorded_video
        if pending is None or self._closing:
            return
        self._start_recorded_video(*pending, timeline_started_at)
        source_record = self._active_source_record
        if source_record is None:
            return
        try:
            recorder = ReplayVideoRecorder(
                self.records_root / "replays",
                source_record,
                fps=self.recorder_window.recorder.fps,
            )
            recorder.start(
                self.recorder_window.frame_pump.latest_frame(),
                timeline_started_at,
            )
            self._replay_video_recorder = recorder
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("初始化回放视频保存失败")
            self._set_status(f"回放继续执行，但无法保存回放视频：{exc}", error=True)

    def _refresh_recorded_video(self):
        capture = self._recorded_capture
        if capture is None:
            return
        started_at = self._comparison_started_at
        if started_at is None:
            return
        progress = min(
            1.0,
            max(0.0, (time.monotonic() - started_at) / self._recorded_duration_seconds),
        )
        target_index = min(
            self._recorded_frame_count - 1,
            int(progress * self._recorded_frame_count),
        )
        if target_index == self._recorded_frame_index:
            return
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, frame = capture.read()
        if not ok:
            self._stop_recorded_video()
            return
        self._recorded_frame_index = target_index
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
        self._recorded_frame_count = 0
        self._recorded_frame_index = -1
        self._recorded_duration_seconds = 0.0
        self._comparison_started_at = None
        if capture is not None:
            capture.release()

    def _return_to_selection(self):
        if self.replay_thread is not None and self.replay_thread.isRunning():
            self._set_status("回放进行中，请等待结束后再返回选择。", error=True)
            return
        self._stop_recorded_video()
        self._pending_recorded_video = None
        self._active_source_record = None
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
            self._pending_recorded_video = (
                record.directory / "video.mp4",
                record.duration_seconds,
            )
            self._active_source_record = record
            self.replay_thread.timelineStarted.connect(self._start_pending_recorded_video)
            self.replay_thread.progressChanged.connect(self._update_progress)
            self.replay_thread.replayEnded.connect(self._replay_ended)
            self.progress_bar.setValue(0)
            self.action_label.setText("正在等待第一条动作")
            self.pages.setCurrentWidget(self.comparison_page)
            self.recorded_video_label.setText("正在与回放时间轴同步…")
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
        self._pending_recorded_video = None
        replay_video = self._replay_video_recorder
        self._replay_video_recorder = None
        if replay_video is not None:
            try:
                saved_directory = replay_video.stop(success=success)
                if saved_directory is not None:
                    message = f"{message} 已保存回放视频：{saved_directory.name}"
            except OSError as exc:
                LOGGER.exception("完成回放视频保存失败")
                message = f"{message} 回放视频保存失败：{exc}"
        self._active_source_record = None
        self._set_status(message, error=not success)
        self._update_start_state()

    def stop(self):
        self._closing = True
        self.preview_timer.stop()
        self._stop_recorded_video()
        self._pending_recorded_video = None
        self._active_source_record = None
        replay_video = self._replay_video_recorder
        self._replay_video_recorder = None
        if replay_video is not None:
            replay_video.stop(success=False)
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
        self._shutdown_complete = False
        self.label_tool_window = None

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)
        toolbar = QHBoxLayout()
        project_hint = QLabel("当前标注工程：Game_Recording")
        project_hint.setStyleSheet("color: #555;")
        self.open_label_tool_button = QPushButton("打开标注工具")
        self.open_label_tool_button.setToolTip(
            "打开并自动加载 Game_Recording 的控点标注工程"
        )
        self.open_label_tool_button.clicked.connect(self._open_label_tool)
        toolbar.addWidget(project_hint)
        toolbar.addStretch(1)
        toolbar.addWidget(self.open_label_tool_button)
        root_layout.addLayout(toolbar)

        self.tabs = QTabWidget(self)
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
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
        self.comparison_panel = ReplayComparisonPanel(
            records_root=Path(output_root).parent,
            parent=self.tabs,
        )
        self.tabs.addTab(self.recorder_window, "录制")
        self.tabs.addTab(self.replay_panel, "回放")
        self.tabs.addTab(self.comparison_panel, "对比")
        self.tabs.currentChanged.connect(self._tab_changed)

    def _label_tool_destroyed(self, *_args):
        self.label_tool_window = None

    def _open_label_tool(self):
        """在当前 Qt 进程中打开标注工具，并固定加载 Game_Recording。"""
        if not (GAME_RECORDING_PROJECT_DIR / "info.py").is_file():
            QMessageBox.warning(
                self,
                "无法打开标注工具",
                "未找到 Game_Recording/info.py，请检查标注工程。",
            )
            return

        label_window = self.label_tool_window
        if label_window is None:
            try:
                from aw.autogame.tools.Label import AutoStudioWindow

                label_window = AutoStudioWindow()
                label_window.load_project_from_dir(str(GAME_RECORDING_PROJECT_DIR))
                label_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
                label_window.destroyed.connect(self._label_tool_destroyed)
                self.label_tool_window = label_window
            except Exception as exc:
                LOGGER.exception("打开 Game_Recording 标注工具失败")
                QMessageBox.critical(
                    self,
                    "无法打开标注工具",
                    f"无法加载 Game_Recording 标注工程：{exc}",
                )
                return

        label_window.show()
        label_window.raise_()
        label_window.activateWindow()

    def _tab_changed(self, index: int):
        if index == 0:
            self.recorder_window.setFocus()
        elif index == 1:
            self.replay_panel.selector.refresh_records()
        else:
            self.comparison_panel.refresh_records()

    def shutdown(self):
        """统一窗口关闭和 Qt 应用退出共用同一套资源收尾。"""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self.replay_panel.stop()
        self.comparison_panel.stop()
        self.recorder_window.shutdown()
        if self.label_tool_window is not None:
            self.label_tool_window.close()

    def closeEvent(self, event: QCloseEvent):
        self.shutdown()
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
    # 用户点右上角关闭、系统请求退出、或其他代码调用 app.quit() 都必须停止 hilog。
    app.aboutToQuit.connect(window.shutdown)
    return int(app.exec())
