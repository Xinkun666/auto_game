#!/usr/bin/env python3
"""批量提取截图的方向与位置，并在人工确认后导出结果。

启动后先选择截图目录并确认两个归一化 ROI；点击“获取信息”才会加载
当前工程的方向 CTC 模型与位置匹配工具。自动结果只是候选值，只有点击
“保存已确认结果”才会在图片目录写出确认后的 CSV 和 JSON。
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = (
    PROJECT_ROOT / "aw/autogame/customs_examples/Auto_PUBG_ALL/resource"
)
DIRECTION_MODEL_PATH = RESOURCE_ROOT / "weights/direction_ctc.pt"
MAP_IMAGE_PATH = RESOURCE_ROOT / "map/hpjy.png"

# 来源：Auto_PUBG_ALL/info.py 的“游戏场景/2832_1280/special_areas”。
# 这是归一化坐标，截图分辨率变化时会自动等比例换算。
DEFAULT_DIRECTION_ROI = (0.48312853107344633, 0.0275859375, 0.5173799435028249, 0.0650859375)
DEFAULT_LOCATION_ROI = (0.8534757203321719, 0.0005393644762167754, 0.9646396530575277, 0.24217464982133213)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NormalizedRoi = Tuple[float, float, float, float]


@dataclass
class ImageResult:
    filename: str
    direction: Optional[int]
    location_x: Optional[int]
    location_y: Optional[int]
    direction_confidence: Optional[float]
    location_mode: str
    status: str
    manually_edited: bool = False
    reviewed: bool = False


def roi_to_text(roi: NormalizedRoi) -> str:
    return ", ".join(f"{value:.17g}" for value in roi)


def parse_roi_text(value: str) -> NormalizedRoi:
    try:
        roi = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("ROI 必须是 4 个逗号分隔的小数") from exc
    if len(roi) != 4 or not all(math.isfinite(item) for item in roi):
        raise ValueError("ROI 必须是 4 个有限数字")
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("ROI 必须满足 0≤x1<x2≤1 且 0≤y1<y2≤1")
    return roi  # type: ignore[return-value]


def crop_roi(image_rgb: np.ndarray, roi: NormalizedRoi) -> np.ndarray:
    """按归一化 ROI 裁出 RGB 图像，坐标规则与项目现有 special area 一致。"""
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = roi
    left = int(math.floor(x1 * width))
    top = int(math.floor(y1 * height))
    right = int(math.ceil(x2 * width))
    bottom = int(math.ceil(y2 * height))
    return np.ascontiguousarray(image_rgb[top:bottom, left:right])


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def display_value(value: Optional[object]) -> str:
    return "" if value is None else str(value)


class DirLocApp:
    """人工复核工具：自动提取候选值，人工修改后统一确认保存。"""

    PREVIEW_FALLBACK_SIZE = (480, 480)
    PREVIEW_PADDING = 20
    POLL_QUEUE_INTERVAL_MS = 80

    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
            from PIL import Image, ImageDraw, ImageTk
        except Exception as exc:
            raise RuntimeError("无法启动图形界面，请确认已安装 tkinter 和 Pillow") from exc

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageTk = ImageTk
        self.resample_filter = getattr(Image, "Resampling", Image).LANCZOS

        self.root = tk.Tk()
        self.root.title("方向与位置批量提取（人工确认）")
        self.root.geometry("1420x900")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.directory_var = tk.StringVar()
        self.model_path_var = tk.StringVar(value=str(DIRECTION_MODEL_PATH))
        self.direction_roi_var = tk.StringVar(value=roi_to_text(DEFAULT_DIRECTION_ROI))
        self.location_roi_var = tk.StringVar(value=roi_to_text(DEFAULT_LOCATION_ROI))
        self.direction_var = tk.StringVar()
        self.location_x_var = tk.StringVar()
        self.location_y_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择截图目录，并检查两个 ROI。")

        self.image_paths: list[Path] = []
        self.results: list[ImageResult] = []
        self.selected_index: Optional[int] = None
        self.worker: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.result_queue: queue.Queue = queue.Queue()
        self._preview_photo = None
        self._preview_image_path: Optional[Path] = None
        self._preview_resize_after_id = None
        self._last_preview_bounds: Optional[Tuple[int, int]] = None
        self._config_widgets = []
        self._build_ui()

    def _build_ui(self) -> None:
        container = self.ttk.Frame(self.root, padding=14)
        container.pack(fill="both", expand=True)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)

        settings = self.ttk.LabelFrame(container, text="图片目录与识别区域", padding=10)
        settings.grid(row=0, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)

        self.ttk.Label(settings, text="图片目录").grid(row=0, column=0, sticky="w", pady=3)
        self.directory_entry = self.ttk.Entry(settings, textvariable=self.directory_var)
        self.directory_entry.grid(row=0, column=1, sticky="ew", pady=3)
        self.choose_directory_button = self.ttk.Button(
            settings, text="选择目录", command=self.choose_directory
        )
        self.choose_directory_button.grid(row=0, column=2, padx=(10, 0), pady=3)

        self.ttk.Label(settings, text="Direction 模型").grid(row=1, column=0, sticky="w", pady=3)
        self.model_path_entry = self.ttk.Entry(settings, textvariable=self.model_path_var)
        self.model_path_entry.grid(row=1, column=1, sticky="ew", pady=3)
        self.choose_model_button = self.ttk.Button(
            settings, text="选择模型", command=self.choose_direction_model
        )
        self.choose_model_button.grid(row=1, column=2, padx=(10, 0), pady=3)

        self.ttk.Label(settings, text="Direction ROI").grid(row=2, column=0, sticky="w", pady=3)
        self.direction_roi_entry = self.ttk.Entry(settings, textvariable=self.direction_roi_var)
        self.direction_roi_entry.grid(row=2, column=1, sticky="ew", pady=3)
        self.ttk.Label(settings, text="x1, y1, x2, y2（归一化）").grid(
            row=2, column=2, sticky="w", padx=(10, 0), pady=3
        )

        self.ttk.Label(settings, text="Location ROI").grid(row=3, column=0, sticky="w", pady=3)
        self.location_roi_entry = self.ttk.Entry(settings, textvariable=self.location_roi_var)
        self.location_roi_entry.grid(row=3, column=1, sticky="ew", pady=3)
        self.ttk.Label(settings, text="x1, y1, x2, y2（归一化）").grid(
            row=3, column=2, sticky="w", padx=(10, 0), pady=3
        )

        settings_actions = self.ttk.Frame(settings)
        settings_actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.preview_roi_button = self.ttk.Button(
            settings_actions, text="预览当前 ROI", command=self.preview_current_roi
        )
        self.preview_roi_button.pack(side="left")
        self.extract_button = self.ttk.Button(
            settings_actions, text="获取信息", command=self.start_extraction
        )
        self.extract_button.pack(side="left", padx=(8, 0))
        self.ttk.Label(
            settings_actions,
            text="自动结果仅供检查；点击下方保存才算人工确认。",
        ).pack(side="left", padx=(14, 0))
        self._config_widgets = [
            self.directory_entry,
            self.choose_directory_button,
            self.model_path_entry,
            self.choose_model_button,
            self.direction_roi_entry,
            self.location_roi_entry,
            self.preview_roi_button,
        ]

        pane = self.ttk.PanedWindow(container, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        list_frame = self.ttk.LabelFrame(pane, text="识别结果（选中一行后可修改）", padding=8)
        preview_frame = self.ttk.LabelFrame(pane, text="当前图片与 ROI", padding=8)
        self.preview_frame = preview_frame
        pane.add(list_frame, weight=3)
        pane.add(preview_frame, weight=4)

        columns = ("index", "filename", "direction", "location", "confidence", "status")
        self.tree = self.ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "index": ("序号", 55),
            "filename": ("文件名", 260),
            "direction": ("Direction", 85),
            "location": ("Location", 125),
            "confidence": ("置信度", 75),
            "status": ("状态", 150),
        }
        for name, (title, width) in headings.items():
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, minwidth=45, stretch=name in {"filename", "status"})
        tree_scroll = self.ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_result)

        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = self.ttk.Label(
            preview_frame,
            text="选择目录后可预览第一张图片；识别完成后点击列表项查看。",
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.bind("<Configure>", self._schedule_preview_resize)

        editor = self.ttk.LabelFrame(container, text="人工修正与确认", padding=10)
        editor.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.ttk.Label(editor, text="Direction（0-359；-1 表示无效）").grid(
            row=0, column=0, sticky="w"
        )
        self.direction_entry = self.ttk.Entry(editor, textvariable=self.direction_var, width=16)
        self.direction_entry.grid(row=0, column=1, sticky="w", padx=(8, 18))
        self.ttk.Label(editor, text="Location X").grid(row=0, column=2, sticky="w")
        self.location_x_entry = self.ttk.Entry(editor, textvariable=self.location_x_var, width=14)
        self.location_x_entry.grid(row=0, column=3, sticky="w", padx=(8, 18))
        self.ttk.Label(editor, text="Location Y").grid(row=0, column=4, sticky="w")
        self.location_y_entry = self.ttk.Entry(editor, textvariable=self.location_y_var, width=14)
        self.location_y_entry.grid(row=0, column=5, sticky="w", padx=(8, 18))
        self.apply_button = self.ttk.Button(editor, text="更新当前项", command=self.apply_current_edit)
        self.apply_button.grid(row=0, column=6, sticky="e")
        self.save_button = self.ttk.Button(
            editor, text="保存已确认结果", command=self.save_confirmed_results
        )
        self.save_button.grid(row=0, column=7, sticky="e", padx=(8, 0))
        self.save_button.configure(state="disabled")

        self.ttk.Label(container, textvariable=self.status_var).grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )

    def choose_directory(self) -> None:
        selected = self.filedialog.askdirectory(
            initialdir=self.directory_var.get().strip() or str(PROJECT_ROOT)
        )
        if not selected:
            return
        self.directory_var.set(selected)
        self.load_directory(Path(selected))

    def load_directory(self, directory: Path) -> None:
        try:
            image_paths = iter_images(directory)
        except OSError as exc:
            self.messagebox.showerror("读取目录失败", str(exc), parent=self.root)
            return
        if not image_paths:
            self.messagebox.showwarning("没有图片", "目录中没有 PNG/JPG/JPEG/BMP/WEBP 图片。", parent=self.root)
            return
        self.image_paths = image_paths
        self.results = []
        self.selected_index = None
        self.save_button.configure(state="disabled")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.status_var.set(f"已找到 {len(image_paths)} 张图片；请检查 ROI 后点击“获取信息”。")
        self.show_image(image_paths[0])

    def choose_direction_model(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="选择 direction_ctc 模型权重",
            initialdir=str(Path(self.model_path_var.get()).expanduser().parent),
            filetypes=[("PyTorch 权重", "*.pt *.pth *.ckpt"), ("所有文件", "*.*")],
        )
        if selected:
            self.model_path_var.set(selected)

    def _read_rois(self) -> Tuple[NormalizedRoi, NormalizedRoi]:
        return parse_roi_text(self.direction_roi_var.get()), parse_roi_text(self.location_roi_var.get())

    def preview_current_roi(self) -> None:
        if not self.image_paths:
            self.messagebox.showwarning("请先选目录", "请先选择包含图片的目录。", parent=self.root)
            return
        try:
            self._read_rois()
        except ValueError as exc:
            self.messagebox.showerror("ROI 有误", str(exc), parent=self.root)
            return
        image_path = self.image_paths[self.selected_index or 0]
        self.show_image(image_path)
        self.status_var.set("已按当前输入的 ROI 刷新预览。")

    def start_extraction(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        directory = Path(self.directory_var.get().strip()).expanduser()
        if not directory.is_dir():
            self.messagebox.showerror("目录无效", "请选择一个存在的图片目录。", parent=self.root)
            return
        if not self.image_paths:
            self.load_directory(directory)
        if not self.image_paths:
            return
        try:
            direction_roi, location_roi = self._read_rois()
        except ValueError as exc:
            self.messagebox.showerror("ROI 有误", str(exc), parent=self.root)
            return
        direction_model_path = Path(self.model_path_var.get().strip()).expanduser()
        if not direction_model_path.is_file():
            self.messagebox.showerror(
                "缺少 Direction 模型",
                "当前工程默认的 direction_ctc.pt 不在此电脑上。请选择实际的方向模型权重文件。",
                parent=self.root,
            )
            return

        self.results = []
        self.selected_index = None
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.cancel_event.clear()
        self.extract_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self._set_config_enabled(False)
        self.status_var.set("正在加载当前工程的 direction/location 工具…")
        self.worker = threading.Thread(
            target=self._extract_worker,
            args=(list(self.image_paths), direction_roi, location_roi, direction_model_path),
            daemon=True,
        )
        self.worker.start()
        self.root.after(self.POLL_QUEUE_INTERVAL_MS, self.poll_worker_queue)

    def _extract_worker(
        self,
        image_paths: list[Path],
        direction_roi: NormalizedRoi,
        location_roi: NormalizedRoi,
        direction_model_path: Path,
    ) -> None:
        try:
            from PIL import Image
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.direction_ctc_service import Get_Direction
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.location_service import LocatePoints

            direction_tool = Get_Direction(model_weight=direction_model_path)
            location_tool = LocatePoints(big_map_path=str(MAP_IMAGE_PATH))
            self.result_queue.put(("ready", None))
        except Exception as exc:
            self.result_queue.put(("error", f"初始化识别工具失败：{exc}"))
            return

        for index, image_path in enumerate(image_paths):
            if self.cancel_event.is_set():
                self.result_queue.put(("done", "已取消。"))
                return
            try:
                with Image.open(image_path) as image:
                    frame_rgb = np.array(image.convert("RGB"), copy=True)
                direction_crop = crop_roi(frame_rgb, direction_roi)
                location_crop = crop_roi(frame_rgb, location_roi)

                direction_detail = direction_tool.get_direction_detail(direction_crop)
                direction = int(direction_detail.get("angle", -1))
                confidence = float(direction_detail.get("confidence", 0.0))
                # 离线截图不是连续视频帧，不能使用带卡尔曼历史状态的 get_location。
                location, location_mode = location_tool.get_global_location(
                    cv2.cvtColor(location_crop, cv2.COLOR_RGB2BGR)
                )
                location_x, location_y = location
                result = ImageResult(
                    filename=image_path.name,
                    direction=direction,
                    location_x=location_x,
                    location_y=location_y,
                    direction_confidence=confidence,
                    location_mode=str(location_mode),
                    status="待人工确认",
                )
            except Exception as exc:
                result = ImageResult(
                    filename=image_path.name,
                    direction=None,
                    location_x=None,
                    location_y=None,
                    direction_confidence=None,
                    location_mode="error",
                    status=f"识别失败：{exc}",
                )
            self.result_queue.put(("result", (index, result)))
        self.result_queue.put(("done", f"自动提取完成，共 {len(image_paths)} 张；请逐项人工检查并保存。"))

    def poll_worker_queue(self) -> None:
        active = self.worker is not None and self.worker.is_alive()
        while True:
            try:
                kind, payload = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                self.status_var.set("识别工具已加载，正在获取图片信息…")
            elif kind == "result":
                index, result = payload
                self.results.append(result)
                self.insert_or_update_result(index, result)
                self.status_var.set(f"已获取 {len(self.results)}/{len(self.image_paths)} 张图片的信息。")
            elif kind == "error":
                self.status_var.set(payload)
                self.messagebox.showerror("获取信息失败", payload, parent=self.root)
            elif kind == "done":
                self.status_var.set(payload)

        if active:
            self.root.after(self.POLL_QUEUE_INTERVAL_MS, self.poll_worker_queue)
        else:
            self.extract_button.configure(state="normal")
            self.save_button.configure(state="normal" if self.results else "disabled")
            self._set_config_enabled(True)

    def _set_config_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._config_widgets:
            widget.configure(state=state)

    def insert_or_update_result(self, index: int, result: ImageResult) -> None:
        values = self.result_values(index, result)
        item_id = str(index)
        if self.tree.exists(item_id):
            self.tree.item(item_id, values=values)
        else:
            self.tree.insert("", "end", iid=item_id, values=values)

    @staticmethod
    def result_values(index: int, result: ImageResult) -> Tuple[object, ...]:
        location = "" if result.location_x is None or result.location_y is None else f"{result.location_x}, {result.location_y}"
        confidence = "" if result.direction_confidence is None else f"{result.direction_confidence:.3f}"
        return (index + 1, result.filename, display_value(result.direction), location, confidence, result.status)

    def on_select_result(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        index = int(selected[0])
        if index >= len(self.results):
            return
        self.selected_index = index
        result = self.results[index]
        self.direction_var.set(display_value(result.direction))
        self.location_x_var.set(display_value(result.location_x))
        self.location_y_var.set(display_value(result.location_y))
        self.show_image(self.image_paths[index])

    def show_image(self, image_path: Path) -> None:
        self._preview_image_path = image_path
        self._last_preview_bounds = None
        self._render_preview_image()

    def _schedule_preview_resize(self, _event=None) -> None:
        if self._preview_image_path is None:
            return
        if self._preview_resize_after_id is not None:
            self.root.after_cancel(self._preview_resize_after_id)
        self._preview_resize_after_id = self.root.after(80, self._render_preview_image)

    def _preview_bounds(self) -> Tuple[int, int]:
        width = self.preview_frame.winfo_width() - self.PREVIEW_PADDING
        height = self.preview_frame.winfo_height() - self.PREVIEW_PADDING
        if width <= 80 or height <= 80:
            return self.PREVIEW_FALLBACK_SIZE
        return max(80, width), max(80, height)

    def _render_preview_image(self) -> None:
        self._preview_resize_after_id = None
        image_path = self._preview_image_path
        if image_path is None:
            return
        bounds = self._preview_bounds()
        if bounds == self._last_preview_bounds:
            return
        try:
            direction_roi, location_roi = self._read_rois()
            with self.Image.open(image_path) as image:
                preview = image.convert("RGB").copy()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"预览失败：{exc}")
            return

        draw = self.ImageDraw.Draw(preview)
        width, height = preview.size
        for roi, color in ((direction_roi, "#21d4fd"), (location_roi, "#ff9f1c")):
            x1, y1, x2, y2 = roi
            draw.rectangle(
                (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)),
                outline=color,
                width=max(2, min(width, height) // 500),
            )
        preview.thumbnail(bounds, self.resample_filter)
        self._preview_photo = self.ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self._preview_photo, text="")
        self._last_preview_bounds = bounds

    def apply_current_edit(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.results):
            self.messagebox.showwarning("请先选择图片", "请在结果列表中选择一张图片。", parent=self.root)
            return
        try:
            direction = int(self.direction_var.get().strip())
            if direction != -1 and not 0 <= direction <= 359:
                raise ValueError("Direction 必须是 0 到 359，或 -1")
            location_x = int(self.location_x_var.get().strip())
            location_y = int(self.location_y_var.get().strip())
        except ValueError as exc:
            self.messagebox.showerror("修改有误", str(exc), parent=self.root)
            return

        result = self.results[self.selected_index]
        result.direction = direction
        result.location_x = location_x
        result.location_y = location_y
        result.manually_edited = True
        result.status = "已人工修改"
        self.insert_or_update_result(self.selected_index, result)
        self.status_var.set(f"已更新：{result.filename}。确认无误后点击“保存已确认结果”。")

    def save_confirmed_results(self) -> None:
        if not self.results:
            self.messagebox.showwarning("暂无结果", "请先点击“获取信息”并完成人工检查。", parent=self.root)
            return
        output_directory = Path(self.directory_var.get().strip()).expanduser()
        if not output_directory.is_dir():
            self.messagebox.showerror("目录无效", "图片目录不存在，无法保存结果。", parent=self.root)
            return

        confirmed_at = datetime.now().isoformat(timespec="seconds")
        for result in self.results:
            result.reviewed = True
            if result.status == "待人工确认":
                result.status = "已人工确认"
        csv_path = output_directory / "dir_loc_confirmed.csv"
        json_path = output_directory / "dir_loc_confirmed.json"
        if (csv_path.exists() or json_path.exists()) and not self.messagebox.askyesno(
            "覆盖已有结果？",
            "目录中已存在 dir_loc_confirmed.csv 或 dir_loc_confirmed.json，是否覆盖？",
            parent=self.root,
        ):
            return
        try:
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filename", "direction", "location_x", "location_y",
                        "direction_confidence", "location_mode", "status",
                        "manually_edited", "reviewed",
                    ],
                )
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
            payload = {
                "confirmed_at": confirmed_at,
                "image_directory": str(output_directory.resolve()),
                "direction_model_path": str(Path(self.model_path_var.get()).expanduser()),
                "direction_roi": parse_roi_text(self.direction_roi_var.get()),
                "location_roi": parse_roi_text(self.location_roi_var.get()),
                "results": [asdict(result) for result in self.results],
            }
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.messagebox.showerror("保存失败", str(exc), parent=self.root)
            return

        for index, result in enumerate(self.results):
            self.insert_or_update_result(index, result)
        self.status_var.set(f"已保存人工确认结果：{csv_path.name}、{json_path.name}")
        self.messagebox.showinfo(
            "保存完成",
            f"已保存 {len(self.results)} 条人工确认结果：\n{csv_path.name}\n{json_path.name}",
            parent=self.root,
        )

    def close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    DirLocApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
