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
from .touch_controller import SingleTouchKeyboardController


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


class RecorderWindow(QMainWindow):
    def __init__(self, output_root: Path, fps: float = 15.0):
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

        self.buffer = FrameBuffer(size=5)
        self.stream_client = HOSScrcpyStreamClient(self.buffer)
        self.recorder = RecordingSession(output_root=output_root, fps=fps)
        self.frame_pump = FramePump(self.buffer, self.recorder)
        self.controller = SingleTouchKeyboardController(self.stream_client, self.key_points)
        self.pressed_keys = set()
        self._closed = False

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

        self.frame_pump.start()
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

        self.pressed_keys.add(key)
        try:
            actions = self.controller.press(key)
            self.recorder.record_key_event("press", key, self.pressed_keys, actions)
        except Exception as exc:
            self.pressed_keys.discard(key)
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
            self._set_status(f"释放键位 {key} 失败：{exc}", error=True)
        event.accept()

    def closeEvent(self, event: QCloseEvent):
        if not self._closed:
            self._closed = True
            try:
                self.controller.release_all()
            except Exception:
                LOGGER.exception("关闭窗口时释放触控失败")
            self._stop_recording(reason="window_closed")
            self.frame_pump.stop()
            self.stream_client.stop()
            self.frame_pump.join(timeout=2.0)
        event.accept()


def run(output_root: Path, fps: float = 15.0) -> int:
    _prefer_bundled_pyqt_plugins()
    app = QApplication.instance() or QApplication([])
    try:
        window = RecorderWindow(output_root=output_root, fps=fps)
    except LayoutError as exc:
        raise SystemExit(f"键位布局不可用：{exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"录制启动失败：{exc}") from exc
    window.show()
    window.activateWindow()
    window.setFocus()
    return int(app.exec())
