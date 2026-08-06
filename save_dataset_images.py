#!/usr/bin/env python3
"""按固定间隔从当前设备画面保存数据集图片。

默认保存整张画面；设置归一化 ROI 后，会先按当前画面宽高转换为像素坐标再裁剪。
坐标原点为画面左上角，ROI 格式为 (x1, y1, x2, y2)，每个值的范围是 0～1。
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
# ============================================================================


NormalizedRoi = Tuple[float, float, float, float]
PixelRoi = Tuple[int, int, int, int]
FIRST_FRAME_TIMEOUT_SECONDS = 20.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按固定间隔从设备画面保存 PNG 图片。"
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
    args = parser.parse_args(argv)

    if args.interval < 0:
        parser.error("--interval 不能小于 0")
    if args.count < 0:
        parser.error("--count 不能小于 0")
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


def save_rgb_frame(frame: np.ndarray, output_dir: Path, index: int) -> Path:
    """以时间戳和序号命名，安全保存 RGB PNG。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_dir / f"{timestamp}_{index:06d}.png"
    image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not write_image_unicode(output_path, image_bgr):
        raise OSError(f"保存图片失败：{output_path}")
    return output_path


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


def run_capture(
    output_dir: Path,
    roi: Optional[NormalizedRoi],
    interval: float,
    count: int,
) -> int:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    saved_count = 0
    try:
        client, buffer, screen_mode = create_capture_client()
        roi_text = "整张画面" if roi is None else f"ROI={roi}"
        count_text = "持续采集" if count == 0 else f"共 {count} 张"
        print(f"采集模式：screen_mode={screen_mode}")
        print(f"保存目录：{output_dir}")
        print(f"采集范围：{roi_text}；间隔：{interval:g} 秒；{count_text}")
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
            saved_count += 1
            output_path = save_rgb_frame(cropped, output_dir, saved_count)
            print(f"[{saved_count}] 已保存：{output_path.name}")

            if count == 0 or saved_count < count:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n已收到停止指令。")
    finally:
        if client is not None:
            client.stop()

    print(f"采集结束，共保存 {saved_count} 张：{output_dir}")
    return saved_count


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_capture(
        output_dir=Path(args.output),
        roi=args.roi,
        interval=args.interval,
        count=args.count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
