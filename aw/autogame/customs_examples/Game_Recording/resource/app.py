"""Game_Recording 的独立键盘录制窗口。"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import PyQt6
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from aw.autogame.customs_examples.Game_Recording import info
from aw.autogame.tools.Utils import get_resolution

from .layout import LayoutError, RESERVED_KEYS, load_key_layout
from .recording import RecordingSession
from .runtime_log import save_disconnect_report
from .touch_controller import MOVEMENT_KEYS, SingleTouchKeyboardController


LOGGER = logging.getLogger("GameRecording")
SPECIAL_KEYS = {
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_Up: "up",
    Qt.Key.Key_Down: "down",
    Qt.Key.Key_Left: "left",
    Qt.Key.Key_Right: "right",
    Qt.Key.Key_Return: "enter",
    Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Control: "ctrl",
    Qt.Key.Key_Alt: "alt",
    Qt.Key.Key_Tab: "tab",
}


def _prefer_bundled_pyqt_plugins():
    """确保使用当前 PyQt6 自带插件，而不是 Conda/OpenCV 的 Qt 插件。"""
    plugin_root = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
    if not (plugin_root / "platforms").is_dir():
        return
    QCoreApplication.addLibraryPath(str(plugin_root))
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)


class FramePump(threading.Thread):
    def __init__(self, buffer, recorder: RecordingSession):
        super().__init__(name="GameRecordingFramePump", daemon=True)
        self.buffer = buffer
        self.recorder = recorder
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self.error = None

    def run(self):
        while not self._stop_event.is_set():
            frame = self.buffer.get_latest(timeout=1.0)
            if frame is None:
                continue
            try:
                rgb = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(f"收到不支持的画面尺寸：{rgb.shape}")
                with self._frame_lock:
                    self._latest_frame = rgb.copy()
                self.recorder.accept_frame(rgb)
            except Exception as exc:
                self.error = exc
                LOGGER.exception("处理 HOS 画面失败")

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._stop_event.set()


def key_name_from_event(event: QKeyEvent):
    special = SPECIAL_KEYS.get(event.key())
    if special:
        return special
    text = event.text().strip().lower()
    return text if len(text) == 1 else None


def configure_fail_fast_stream(client):
    """Game_Recording 专用：不重试同一 SO，auto 模式可继续换下一个候选。"""
    client.max_reconnect_attempts = 0
    # HOS 对 USB offline 和 cleanup 错误还有独立恢复分支，也在此模块禁用。
    client._hdc_recovery_attempted = True
    client._cleanup_hdc_recovery_attempted = True
    return client


class RecorderWindow(QMainWindow):
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
        # 先检查空工程，避免 info.py 尚未标注时被设备连接错误遮住。
        load_key_layout(info, 1, 1)
        if shutil.which("hdc") is None:
            raise RuntimeError(
                "当前终端找不到 hdc。请先安装/配置华为 HDC，并确保执行 `hdc list targets` 能看到手机。"
            )
        from aw.autogame.stream_client.stream_client import FrameBuffer, HOSScrcpyStreamClient

        screen_width, screen_height = get_resolution()
        self.screen_size = (int(screen_width), int(screen_height))
        self.key_points = load_key_layout(info, *self.screen_size)

        self.output_root = Path(output_root)
        self.runtime_log_path = Path(runtime_log_path) if runtime_log_path else None
        self.hilog_capture = hilog_capture
        self.buffer = FrameBuffer(size=5)
        requested_video_so = str(video_so or "auto").strip()
        force_video_so = "" if requested_video_so == "reuse" else requested_video_so
        self.stream_client = configure_fail_fast_stream(
            HOSScrcpyStreamClient(self.buffer, force_video_so=force_video_so)
        )
        self.touch_backend_name = str(touch_backend or "hos").strip().lower()
        self.sendevent_touch = None
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
            raise ValueError("不支持的触控后端：%s" % self.touch_backend_name)
        self.recorder = RecordingSession(
            output_root=self.output_root / "recordings",
            fps=fps,
        )
        self.frame_pump = FramePump(self.buffer, self.recorder)
        self.controller = SingleTouchKeyboardController(touch_client, self.key_points)
        self.pressed_keys = set()
        self._closed = False
        self._disconnect_handled = False

        self.setWindowTitle("Game Recording - q 开始 / e 结束")
        self.resize(960, 620)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.status_label = QLabel("正在连接华为手机画面……")
        self.status_label.setWordWrap(True)
        self.keys_label = QLabel("已加载键位：" + "、".join(sorted(self.key_points)))
        self.keys_label.setWordWrap(True)
        self.preview_label = QLabel("等待 HOScrcpy 首帧")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background: #151515; color: #dddddd;")

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.keys_label)
        layout.addWidget(self.preview_label, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self.preview_timer.start(50)

        self.stream_monitor_timer = QTimer(self)
        self.stream_monitor_timer.timeout.connect(self._monitor_stream_health)
        self.stream_monitor_timer.start(50)

        self.frame_pump.start()
        print(
            "[Game Recording] HOS 不重试同一 SO；auto 模式会换包，"
            "所有候选失败后才停止。",
            flush=True,
        )
        print(
            "[Game Recording] HOS 投屏 SO 策略：%s"
            % (force_video_so or "reuse-device-existing"),
            flush=True,
        )
        print(
            "[Game Recording] 触控后端：%s" % self.touch_backend_name,
            flush=True,
        )
        if self.sendevent_touch is not None:
            print(
                "[Game Recording] sendevent 配置：%s"
                % self.sendevent_touch.diagnostic_snapshot(),
                flush=True,
            )
        self.stream_client.start_backend()

    def _set_status(self, text: str, error: bool = False):
        color = "#b00020" if error else "#167c36"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.status_label.setText(text)

    def _refresh_preview(self):
        if self.frame_pump.error is not None:
            self._set_status(f"画面处理失败：{self.frame_pump.error}", error=True)
        frame = self.frame_pump.latest_frame()
        if frame is None:
            stream_error = getattr(self.stream_client, "_last_error", None)
            if stream_error is not None:
                self._set_status(f"HOScrcpy 暂未取得画面：{stream_error}", error=True)
            return
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)
        if not self.recorder.is_recording and "等待" in self.status_label.text():
            self._set_status("控制已就绪：q 开始录制，e 停止并保存；请保持本窗口在前台。")

    def _monitor_stream_health(self):
        if self._closed or self._disconnect_handled:
            return
        stream_error = getattr(self.stream_client, "_last_error", None)
        if stream_error is not None:
            self._handle_hos_disconnect(stream_error)

    def _diagnostic_snapshot(self, stream_error: Exception):
        diagnostic = {
            "last_error": str(stream_error),
            "touch_backend": self.touch_backend_name,
        }
        if self.sendevent_touch is not None:
            diagnostic["sendevent"] = self.sendevent_touch.diagnostic_snapshot()
        snapshot = getattr(self.stream_client, "diagnostic_snapshot", None)
        if callable(snapshot):
            try:
                diagnostic.update(snapshot() or {})
            except Exception as exc:
                diagnostic["snapshot_error"] = str(exc)
                LOGGER.exception("获取 HOS 断连诊断失败")
        return diagnostic

    def _release_controls(self):
        self.pressed_keys.clear()
        try:
            self.controller.release_all()
        except Exception:
            LOGGER.exception("停止时释放触控失败")

    def _stop_stream(self):
        self.frame_pump.stop()
        self.stream_client.stop()
        self.frame_pump.join(timeout=2.0)

    def _close_touch_backend(self):
        if self.sendevent_touch is None:
            return
        backend = self.sendevent_touch
        self.sendevent_touch = None
        backend.close()

    def _handle_hos_disconnect(self, stream_error: Exception):
        self._disconnect_handled = True
        self._closed = True
        self.preview_timer.stop()
        self.stream_monitor_timer.stop()
        diagnostic = self._diagnostic_snapshot(stream_error)
        self._set_status("检测到 HOS 断连，正在停止并保存日志……", error=True)
        print(f"[Game Recording] 检测到 HOS 断连：{stream_error}", flush=True)

        self._release_controls()
        try:
            self._close_touch_backend()
        except Exception as exc:
            diagnostic["touch_backend_close_error"] = str(exc)
            LOGGER.exception("HOS 断连后关闭触控后端失败")
        recording_dir = self.recorder.session_dir
        recording_error = ""
        try:
            saved_dir = self.recorder.stop(reason="hos_disconnect")
            if saved_dir is not None:
                recording_dir = saved_dir
                print(f"[Game Recording] 断连前的录制已保存：{saved_dir}", flush=True)
        except Exception as exc:
            recording_error = str(exc)
            LOGGER.exception("HOS 断连后保存录制失败")

        try:
            self._stop_stream()
        except Exception as exc:
            diagnostic["stream_shutdown_error"] = str(exc)
            LOGGER.exception("HOS 断连后停止抓流失败")
        hilog_path = None
        if self.hilog_capture is not None:
            try:
                self.hilog_capture.stop()
                hilog_path = self.hilog_capture.path
                print(f"[Game Recording] hilog 抓取已停止：{hilog_path}", flush=True)
            except Exception as exc:
                diagnostic["hilog_capture_stop_error"] = str(exc)
                LOGGER.exception("HOS 断连后停止 hilog 抓取失败")
        try:
            report_paths = save_disconnect_report(
                output_root=self.output_root,
                diagnostic=diagnostic,
                runtime_log_path=self.runtime_log_path,
                recording_dir=recording_dir,
                recording_error=recording_error,
                hilog_path=hilog_path,
            )
            print(
                f"[Game Recording] 断连报告已保存：{report_paths['disconnect_report']}",
                flush=True,
            )
            if "hilog_log" in report_paths:
                print(
                    f"[Game Recording] hilog 已保存：{report_paths['hilog_log']}",
                    flush=True,
                )
            elif "hilog_error" in report_paths:
                print(
                    f"[Game Recording] hilog 抓取失败说明：{report_paths['hilog_error']}",
                    flush=True,
                )
        except Exception:
            LOGGER.exception("保存 HOS 断连报告失败")
        if self.runtime_log_path is not None:
            print(f"[Game Recording] 完整运行日志：{self.runtime_log_path}", flush=True)
        QTimer.singleShot(0, self.close)

    def _layout_for_metadata(self):
        return {
            key: {
                "position": list(point.position),
                "normalized_position": list(point.normalized_position),
                "stage": point.stage,
                "scene": point.scene,
            }
            for key, point in self.key_points.items()
        }

    def _start_recording(self):
        if self.recorder.is_recording:
            self._set_status("已经在录制中；按 e 结束。")
            return
        frame = self.frame_pump.latest_frame()
        if frame is None:
            self._set_status("还没有收到手机画面，请等待首帧后再按 q。", error=True)
            return
        session_dir = self.recorder.start(
            frame,
            self.screen_size,
            self._layout_for_metadata(),
            pressed_keys=self.pressed_keys,
        )
        self._set_status(f"● 正在录制：{session_dir.name}（按 e 结束）")

    def _stop_recording(self, reason: str = "e"):
        session_dir = self.recorder.stop(reason=reason)
        if session_dir is None:
            self._set_status("当前没有正在进行的录制；按 q 开始。")
            return
        self._set_status(f"录制已保存：{session_dir}")

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = key_name_from_event(event)
        if not key or key in RESERVED_KEYS:
            return
        if key in self.pressed_keys:
            return
        if key not in self.key_points:
            self._set_status(f"键位 {key} 未在 info.py 中标注，已忽略。", error=True)
            return

        previous_pressed_keys = set(self.pressed_keys)
        if key in MOVEMENT_KEYS:
            self.pressed_keys.difference_update(MOVEMENT_KEYS)
        self.pressed_keys.add(key)
        try:
            actions = self.controller.press(key)
            self.recorder.record_key_event("press", key, self.pressed_keys, actions)
        except Exception as exc:
            self.pressed_keys.clear()
            self.pressed_keys.update(previous_pressed_keys)
            LOGGER.exception("发送键位 %s 失败", key)
            self._set_status(f"发送键位 {key} 失败：{exc}", error=True)
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = key_name_from_event(event)
        if key == "q":
            self._start_recording()
            event.accept()
            return
        if key == "e":
            self._stop_recording()
            event.accept()
            return
        if not key or key not in self.pressed_keys:
            return

        self.pressed_keys.discard(key)
        try:
            actions = self.controller.release(key)
            self.recorder.record_key_event("release", key, self.pressed_keys, actions)
        except Exception as exc:
            LOGGER.exception("释放键位 %s 失败", key)
            self._set_status(f"释放键位 {key} 失败：{exc}", error=True)
        event.accept()

    def closeEvent(self, event: QCloseEvent):
        if not self._closed:
            self._closed = True
            self.preview_timer.stop()
            self.stream_monitor_timer.stop()
            self._release_controls()
            self._stop_recording(reason="window_closed")
            try:
                self._close_touch_backend()
            except Exception:
                LOGGER.exception("关闭 sendevent 触控后端失败")
            self._stop_stream()
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
    _prefer_bundled_pyqt_plugins()
    app = QApplication.instance() or QApplication([])
    try:
        window = RecorderWindow(
            output_root=output_root,
            fps=fps,
            runtime_log_path=runtime_log_path,
            hilog_capture=hilog_capture,
            video_so=video_so,
            touch_backend=touch_backend,
            sendevent_device=sendevent_device,
            sendevent_max_x=sendevent_max_x,
            sendevent_max_y=sendevent_max_y,
        )
    except LayoutError as exc:
        raise SystemExit(f"键位布局不可用：{exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"录制启动失败：{exc}") from exc
    window.show()
    window.activateWindow()
    window.setFocus()
    return int(app.exec())
