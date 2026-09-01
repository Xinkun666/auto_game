#!/usr/bin/env python3
"""从当前设备画面保存数据集图片。

默认保存整张画面；设置归一化 ROI 后，会先按当前画面宽高转换为像素坐标再裁剪。
坐标原点为画面左上角，ROI 格式为 (x1, y1, x2, y2)，每个值的范围是 0～1。

支持两种采集模式：interval 按固定间隔自动保存，single 由用户点击预览按钮保存。
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from aw.autogame.tools.Utils import write_image_unicode


# ============================================================================
# 用户配置：平时只需要修改这一块
# ============================================================================
SAVE_DIRECTORY = "./dataset_images"       # 图片保存目录，支持中文路径
NORMALIZED_ROI = None                      # 整图：None
# NORMALIZED_ROI = (0.1, 0.3, 0.8, 0.9)   # ROI：(左, 上, 右, 下)，范围 0～1
CAPTURE_INTERVAL_SECONDS = 1.0             # 每隔多少秒保存一张
CAPTURE_COUNT = 0                          # 0=持续采集；大于 0=保存指定张数后停止
CAPTURE_MODE = "interval"                  # interval=定时保存；single=点击按钮保存
SHOW_PREVIEW = True                        # 启动预览窗口，显示实际保存的画面
# ============================================================================


NormalizedRoi = Tuple[float, float, float, float]
PixelRoi = Tuple[int, int, int, int]
FIRST_FRAME_TIMEOUT_SECONDS = 20.0
SINGLE_PREVIEW_REFRESH_SECONDS = 0.03
PREVIEW_WINDOW_NAME = "Dataset Image Preview"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从设备画面保存 PNG 图片，支持定时采集和手动单张采集。"
            "不传参数时使用脚本顶部的用户配置。"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=SAVE_DIRECTORY,
        help=f"临时覆盖保存目录；默认使用顶部配置：{SAVE_DIRECTORY}",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        default=NORMALIZED_ROI,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="临时覆盖归一化 ROI，例如 --roi 0.1 0.3 0.8 0.9",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help=f"临时覆盖休眠秒数；默认使用顶部配置：{CAPTURE_INTERVAL_SECONDS:g}",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=CAPTURE_COUNT,
        help=f"临时覆盖采集张数；默认使用顶部配置：{CAPTURE_COUNT}",
    )
    parser.add_argument(
        "--mode",
        choices=("interval", "single"),
        default=CAPTURE_MODE,
        help=(
            "采集模式：interval=按间隔自动保存；"
            "single=点击预览窗口中的保存按钮才保存"
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_false",
        dest="show_preview",
        default=SHOW_PREVIEW,
        help="不显示采集预览窗口，适用于没有图形界面的运行环境",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="不打开配置窗口，直接按命令行参数开始采集",
    )
    args = parser.parse_args(argv)

    if args.interval < 0:
        parser.error("--interval 不能小于 0")
    if args.count < 0:
        parser.error("--count 不能小于 0")
    if args.cli and args.mode == "single" and not args.show_preview:
        parser.error("single 模式需要预览窗口，不能与 --no-preview 同时使用")
    if args.roi is not None:
        args.roi = tuple(args.roi)
        try:
            validate_normalized_roi(args.roi)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def validate_normalized_roi(roi: NormalizedRoi) -> None:
    """校验归一化 ROI 是有效的 (左, 上, 右, 下) 矩形。"""
    if len(roi) != 4:
        raise ValueError("ROI 必须有 4 个值：(x1, y1, x2, y2)")
    x1, y1, x2, y2 = roi
    if not all(math.isfinite(value) for value in roi):
        raise ValueError("ROI 不能包含 NaN 或无穷大")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            f"ROI {roi} 无效：必须满足 0≤x1<x2≤1 且 0≤y1<y2≤1"
        )


def normalized_roi_to_pixels(
    roi: NormalizedRoi,
    width: int,
    height: int,
) -> PixelRoi:
    """将归一化 ROI 转换为当前画面的像素坐标。"""
    validate_normalized_roi(roi)
    if width <= 0 or height <= 0:
        raise ValueError(f"画面宽高无效：{width}×{height}")

    x1, y1, x2, y2 = roi
    return (
        int(math.floor(x1 * width)),
        int(math.floor(y1 * height)),
        int(math.ceil(x2 * width)),
        int(math.ceil(y2 * height)),
    )


def crop_frame(frame: np.ndarray, roi: Optional[NormalizedRoi]) -> np.ndarray:
    """将归一化 ROI 转成像素坐标后裁剪 RGB 画面。"""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("只支持 H×W×3 的 RGB 画面")
    if roi is None:
        return np.ascontiguousarray(image)

    height, width = image.shape[:2]
    x1, y1, x2, y2 = normalized_roi_to_pixels(roi, width, height)
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def parse_roi_text(value: str) -> Optional[NormalizedRoi]:
    """解析配置窗口中的 ROI 文本；留空表示保存整张画面。"""
    text = value.strip()
    if not text:
        return None
    try:
        roi = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError("ROI 必须是 4 个逗号分隔的数字") from exc
    validate_normalized_roi(roi)
    return roi


def save_rgb_frame(frame: np.ndarray, output_dir: Path, index: int) -> Path:
    """以时间戳和序号命名，安全保存 RGB PNG。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_dir / f"{timestamp}_{index:06d}.png"
    image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not write_image_unicode(output_path, image_bgr):
        raise OSError(f"保存图片失败：{output_path}")
    return output_path


class CapturePreview:
    """管理预览窗口，以及 single 模式的鼠标保存按钮。"""

    def __init__(self, capture_mode: str) -> None:
        self.single_mode = capture_mode == "single"
        self._button_rect: Optional[PixelRoi] = None
        self._save_requested = False

    def open(self) -> None:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(PREVIEW_WINDOW_NAME, self._handle_mouse)

    def _handle_mouse(self, event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP or self._button_rect is None:
            return
        x1, y1, x2, y2 = self._button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._save_requested = True

    def show(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """刷新窗口，返回 (继续采集, 是否保存当前帧)。"""
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.single_mode:
            height, width = display.shape[:2]
            x1, y1 = 16, 16
            x2 = min(width - 16, x1 + 220)
            y2 = min(height - 16, y1 + 52)
            self._button_rect = (x1, y1, x2, y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (55, 160, 55), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                display,
                "SAVE CURRENT IMAGE",
                (x1 + 10, y1 + 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(PREVIEW_WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        keep_running = key not in (ord("q"), 27)
        save_current_frame = self._save_requested
        self._save_requested = False
        return keep_running, save_current_frame


def create_capture_client():
    """按项目 config.json 中的 screen_mode 创建取帧客户端。"""
    from aw.autogame.stream_client.stream_client import FrameBuffer
    from aw.autogame.tools.GameAutomator import create_stream_client_for_mode
    from aw.autogame.tools.Utils import (
        get_display_rotation,
        get_resolution,
        get_screen_mode,
        get_wh,
    )

    screen_mode = get_screen_mode()
    buffer = FrameBuffer(size=5)

    if screen_mode == "0":
        screen_w, screen_h = get_resolution()
        width, height = get_wh()
        display_rotation = get_display_rotation()
    else:
        # HDC 截图和 HOScrcpy 模式会直接从帧中获取宽高。
        screen_w = screen_h = width = height = 0
        display_rotation = None

    client = create_stream_client_for_mode(
        screen_mode,
        buffer,
        screen_w,
        screen_h,
        width,
        height,
        display_rotation,
    )
    if screen_mode == "0":
        client.start_backend(
            lowh=0,
            highh=10000,
            skip=20,
            width=width,
            height=height,
        )
    elif screen_mode == "1":
        client.start_backend()
    elif screen_mode == "2":
        client.start_backend()
    else:
        raise ValueError(f"不支持的 screen_mode：{screen_mode}")

    return client, buffer, screen_mode


class DatasetCaptureApp:
    """配置、预览和采集控制共用的桌面窗口。"""

    PREVIEW_MAX_SIZE = (960, 640)
    POLL_INTERVAL_MS = 20

    def __init__(self, initial_args: argparse.Namespace) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
            from PIL import Image, ImageTk
        except Exception as exc:
            raise RuntimeError(
                "无法初始化采集窗口，请确认已安装 tkinter 和 Pillow"
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Image = Image
        self.ImageTk = ImageTk

        self.root = tk.Tk()
        self.root.title("数据集图片采集")
        self.root.minsize(720, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.mode_var = tk.StringVar(value=initial_args.mode)
        self.output_var = tk.StringVar(value=str(initial_args.output))
        self.interval_var = tk.StringVar(value=f"{initial_args.interval:g}")
        self.count_var = tk.StringVar(value=str(initial_args.count))
        roi_text = "" if initial_args.roi is None else ", ".join(
            f"{value:g}" for value in initial_args.roi
        )
        self.roi_var = tk.StringVar(value=roi_text)
        self.status_var = tk.StringVar(value="请先设置参数，然后点击“开始抓流”。")

        self.capture_active = False
        self.client = None
        self.buffer = None
        self.output_dir: Optional[Path] = None
        self.roi: Optional[NormalizedRoi] = None
        self.interval = 0.0
        self.count_limit = 0
        self.saved_count = 0
        self.latest_frame: Optional[np.ndarray] = None
        self.last_save_time: Optional[float] = None
        self.first_frame_deadline: Optional[float] = None
        self._preview_photo = None
        self._settings_widgets = []

        self._build_ui()
        self._update_mode_controls()

    def _build_ui(self) -> None:
        container = self.ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)

        settings = self.ttk.LabelFrame(container, text="采集设置", padding=12)
        settings.grid(row=0, column=0, columnspan=3, sticky="ew")
        settings.columnconfigure(1, weight=1)

        self.ttk.Label(settings, text="采集模式").grid(
            row=0, column=0, sticky="w", pady=4
        )
        interval_radio = self.ttk.Radiobutton(
            settings,
            text="定时模式（按间隔自动保存）",
            variable=self.mode_var,
            value="interval",
            command=self._update_mode_controls,
        )
        interval_radio.grid(row=0, column=1, sticky="w", pady=4)
        single_radio = self.ttk.Radiobutton(
            settings,
            text="Single 模式（手动点击保存）",
            variable=self.mode_var,
            value="single",
            command=self._update_mode_controls,
        )
        single_radio.grid(row=0, column=2, sticky="w", padx=(12, 0), pady=4)

        self.ttk.Label(settings, text="保存目录").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.output_entry = self.ttk.Entry(settings, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.choose_directory_button = self.ttk.Button(
            settings, text="选择目录", command=self._choose_directory
        )
        self.choose_directory_button.grid(row=1, column=2, sticky="ew", padx=(12, 0), pady=4)

        self.ttk.Label(settings, text="最大保存张数").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.count_entry = self.ttk.Entry(settings, textvariable=self.count_var, width=16)
        self.count_entry.grid(row=2, column=1, sticky="w", pady=4)
        self.ttk.Label(settings, text="0 表示不限张数").grid(
            row=2, column=2, sticky="w", padx=(12, 0), pady=4
        )

        self.ttk.Label(settings, text="保存间隔（秒）").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.interval_entry = self.ttk.Entry(
            settings, textvariable=self.interval_var, width=16
        )
        self.interval_entry.grid(row=3, column=1, sticky="w", pady=4)
        self.interval_hint = self.ttk.Label(
            settings, text="仅定时模式生效"
        )
        self.interval_hint.grid(row=3, column=2, sticky="w", padx=(12, 0), pady=4)

        self.ttk.Label(settings, text="ROI（可选）").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.roi_entry = self.ttk.Entry(settings, textvariable=self.roi_var)
        self.roi_entry.grid(row=4, column=1, sticky="ew", pady=4)
        self.ttk.Label(settings, text="留空为整图；例：0.1, 0.3, 0.8, 0.9").grid(
            row=4, column=2, sticky="w", padx=(12, 0), pady=4
        )

        self._settings_widgets = [
            interval_radio,
            single_radio,
            self.output_entry,
            self.choose_directory_button,
            self.count_entry,
            self.interval_entry,
            self.roi_entry,
        ]

        preview_box = self.ttk.LabelFrame(container, text="保存画面预览", padding=8)
        preview_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(14, 0))
        container.rowconfigure(1, weight=1)
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self.preview_label = self.ttk.Label(
            preview_box,
            text="尚未建流。点击下方“开始抓流”后会在此显示保存画面。",
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        action_bar = self.ttk.Frame(container)
        action_bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        action_bar.columnconfigure(0, weight=1)
        self.status_label = self.ttk.Label(action_bar, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky="w")
        self.start_button = self.ttk.Button(
            action_bar, text="开始抓流", command=self.start_capture
        )
        self.start_button.grid(row=0, column=1, padx=(12, 0))
        self.save_button = self.ttk.Button(
            action_bar, text="保存当前图像", command=self.save_current_frame
        )
        self.stop_button = self.ttk.Button(
            action_bar, text="停止抓流", command=self.stop_capture
        )

    def _update_mode_controls(self) -> None:
        if self.capture_active:
            return
        if self.mode_var.get() == "interval":
            self.interval_entry.configure(state="normal")
            self.interval_hint.configure(text="仅定时模式生效")
        else:
            self.interval_entry.configure(state="disabled")
            self.interval_hint.configure(text="Single 模式无需设置间隔")

    def _choose_directory(self) -> None:
        initial_directory = self.output_var.get().strip() or str(Path.cwd())
        selected = self.filedialog.askdirectory(initialdir=initial_directory)
        if selected:
            self.output_var.set(selected)

    def _read_settings(self) -> Tuple[Path, Optional[NormalizedRoi], float, int, str]:
        output_text = self.output_var.get().strip()
        if not output_text:
            raise ValueError("请选择保存目录")
        try:
            count = int(self.count_var.get().strip())
        except ValueError as exc:
            raise ValueError("最大保存张数必须是整数") from exc
        if count < 0:
            raise ValueError("最大保存张数不能小于 0")

        mode = self.mode_var.get()
        interval = 0.0
        if mode == "interval":
            try:
                interval = float(self.interval_var.get().strip())
            except ValueError as exc:
                raise ValueError("保存间隔必须是数字") from exc
            if not math.isfinite(interval) or interval < 0:
                raise ValueError("保存间隔必须是大于等于 0 的有限数字")
        return Path(output_text).expanduser(), parse_roi_text(self.roi_var.get()), interval, count, mode

    def _set_settings_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._settings_widgets:
            widget.configure(state=state)
        if enabled:
            self._update_mode_controls()

    def start_capture(self) -> None:
        if self.capture_active:
            return
        try:
            output_dir, roi, interval, count, mode = self._read_settings()
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("设置有误", str(exc), parent=self.root)
            return

        self.status_var.set("正在建立画面流，请稍候…")
        self.root.update_idletasks()
        try:
            client, buffer, screen_mode = create_capture_client()
        except Exception as exc:
            self.status_var.set("建流失败。")
            self.messagebox.showerror("建流失败", str(exc), parent=self.root)
            return

        self.client = client
        self.buffer = buffer
        self.output_dir = output_dir
        self.roi = roi
        self.interval = interval
        self.count_limit = count
        self.saved_count = 0
        self.latest_frame = None
        self.last_save_time = None
        self.first_frame_deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SECONDS
        self.capture_active = True
        self._set_settings_enabled(False)
        self.start_button.configure(state="disabled")
        self.stop_button.grid(row=0, column=3, padx=(8, 0))
        if mode == "single":
            self.save_button.grid(row=0, column=2, padx=(8, 0))
            self.status_var.set(
                f"已建立画面流（screen_mode={screen_mode}）；等待画面后可手动保存。"
            )
        else:
            self.save_button.grid_remove()
            self.status_var.set(
                f"已建立画面流（screen_mode={screen_mode}）；将按间隔自动保存。"
            )
        self._poll_frame()

    def _poll_frame(self) -> None:
        if not self.capture_active or self.buffer is None:
            return
        try:
            frame = self.buffer.get_latest(timeout=0)
        except Exception as exc:
            self.stop_capture(f"读取画面失败：{exc}")
            return

        if frame is not None:
            try:
                self.latest_frame = crop_frame(frame, self.roi)
                self._display_frame(self.latest_frame)
            except Exception as exc:
                self.stop_capture(f"处理画面失败：{exc}")
                return

            if self.mode_var.get() == "interval":
                now = time.monotonic()
                if self.last_save_time is None or now - self.last_save_time >= self.interval:
                    self.save_current_frame()
        elif (
            self.latest_frame is None
            and self.first_frame_deadline is not None
            and time.monotonic() >= self.first_frame_deadline
        ):
            self.stop_capture(
                f"{FIRST_FRAME_TIMEOUT_SECONDS:g} 秒内未获取到新画面，请检查设备连接和 screen_mode。"
            )
            return

        if self.capture_active:
            self.root.after(self.POLL_INTERVAL_MS, self._poll_frame)

    def _display_frame(self, frame: np.ndarray) -> None:
        image = self.Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
        image.thumbnail(self.PREVIEW_MAX_SIZE, self.Image.Resampling.LANCZOS)
        self._preview_photo = self.ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self._preview_photo, text="")

    def save_current_frame(self) -> None:
        if not self.capture_active:
            return
        if self.latest_frame is None:
            self.status_var.set("尚未收到可保存的画面，请稍候。")
            return
        if self.count_limit and self.saved_count >= self.count_limit:
            self.stop_capture(f"已达到最大保存张数：{self.count_limit}。")
            return
        try:
            output_path = save_rgb_frame(
                self.latest_frame,
                self.output_dir,
                self.saved_count + 1,
            )
        except (OSError, cv2.error) as exc:
            self.status_var.set(f"保存失败：{exc}")
            return

        self.saved_count += 1
        self.last_save_time = time.monotonic()
        self.status_var.set(f"已保存 {self.saved_count} 张：{output_path.name}")
        if self.count_limit and self.saved_count >= self.count_limit:
            self.stop_capture(f"已抓满 {self.count_limit} 张，画面流已停止。")

    def stop_capture(self, message: str = "画面流已停止。") -> None:
        if not self.capture_active:
            return
        self.capture_active = False
        client, self.client = self.client, None
        self.buffer = None
        if client is not None:
            try:
                client.stop()
            except Exception as exc:
                message = f"{message} 停止画面流时出现提示：{exc}"
        self._set_settings_enabled(True)
        self.start_button.configure(state="normal")
        self.save_button.grid_remove()
        self.stop_button.grid_remove()
        self.status_var.set(message)

    def close(self) -> None:
        self.stop_capture("已关闭采集窗口。")
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_capture_window(initial_args: argparse.Namespace) -> None:
    """启动配置窗口；在用户点击开始前不创建任何设备画面流。"""
    DatasetCaptureApp(initial_args).run()


def run_capture(
    output_dir: Path,
    roi: Optional[NormalizedRoi],
    interval: float,
    count: int,
    capture_mode: str,
    show_preview_window: bool,
) -> int:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    preview = CapturePreview(capture_mode) if show_preview_window else None
    saved_count = 0
    try:
        client, buffer, screen_mode = create_capture_client()
        roi_text = "整张画面" if roi is None else f"ROI={roi}"
        count_text = "持续采集" if count == 0 else f"共 {count} 张"
        mode_text = (
            f"定时采集（每 {interval:g} 秒保存一张）"
            if capture_mode == "interval"
            else "单张手动采集（点击预览窗口的保存按钮）"
        )
        print(f"采集模式：screen_mode={screen_mode}")
        print(f"保存目录：{output_dir}")
        print(f"保存方式：{mode_text}")
        print(f"采集范围：{roi_text}；{count_text}")
        if preview is not None:
            preview.open()
            if capture_mode == "single":
                print("预览窗口已启动；点击 SAVE CURRENT IMAGE 保存当前画面。")
            else:
                print("预览窗口已启动；按 q 或 Esc，或在终端按 Ctrl+C 停止。")
        else:
            print("按 Ctrl+C 停止。")

        while count == 0 or saved_count < count:
            frame = buffer.get_latest(timeout=FIRST_FRAME_TIMEOUT_SECONDS)
            if frame is None:
                raise RuntimeError(
                    f"{FIRST_FRAME_TIMEOUT_SECONDS:g} 秒内未获取到新画面，"
                    "请检查设备连接和 aw/autogame/config/config.json 中的 screen_mode"
                )

            if saved_count == 0 and roi is not None:
                frame_height, frame_width = np.asarray(frame).shape[:2]
                pixel_roi = normalized_roi_to_pixels(roi, frame_width, frame_height)
                print(
                    f"ROI 换算：{roi} -> 像素坐标 {pixel_roi} "
                    f"(画面 {frame_width}×{frame_height})"
                )

            cropped = crop_frame(frame, roi)
            keep_running = True
            save_requested = False
            if preview is not None:
                keep_running, save_requested = preview.show(cropped)

            should_save = capture_mode == "interval" or save_requested
            if should_save:
                saved_count += 1
                output_path = save_rgb_frame(cropped, output_dir, saved_count)
                print(f"[{saved_count}] 已保存：{output_path.name}")

            if not keep_running:
                print("已通过预览窗口停止采集。")
                break

            if count == 0 or saved_count < count:
                if capture_mode == "interval":
                    time.sleep(interval)
                else:
                    time.sleep(SINGLE_PREVIEW_REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\n已收到停止指令。")
    finally:
        if client is not None:
            client.stop()
        if show_preview_window:
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)

    print(f"采集结束，共保存 {saved_count} 张：{output_dir}")
    return saved_count


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.cli:
        run_capture_window(args)
        return 0

    run_capture(
        output_dir=Path(args.output),
        roi=args.roi,
        interval=args.interval,
        count=args.count,
        capture_mode=args.mode,
        show_preview_window=args.show_preview,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
