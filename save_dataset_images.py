#!/usr/bin/env python3
"""按固定间隔从当前设备画面保存数据集图片。

默认保存整张画面；传入 --roi X1 Y1 X2 Y2 后只保存该矩形区域。
坐标原点为画面左上角，(X2, Y2) 是裁剪区域右下角的开区间边界。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from aw.autogame.tools.Utils import write_image_unicode


Roi = Tuple[int, int, int, int]
DEFAULT_INTERVAL_SECONDS = 1.0
FIRST_FRAME_TIMEOUT_SECONDS = 20.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按固定间隔从设备画面保存 PNG 图片，默认保存整张画面。",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="图片保存目录，支持中文路径；目录不存在时会自动创建",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="只保存 ROI，例如 --roi 100 200 800 600；不传则保存整张画面",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="两次保存之间的休眠秒数，默认 1 秒",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=0,
        help="保存多少张后自动停止；默认 0 表示持续保存，直到按 Ctrl+C",
    )
    args = parser.parse_args(argv)

    if args.interval < 0:
        parser.error("--interval 不能小于 0")
    if args.count < 0:
        parser.error("--count 不能小于 0")
    if args.roi is not None:
        args.roi = tuple(args.roi)
    return args


def crop_frame(frame: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    """根据 ROI 裁剪 RGB 画面，并对越界或空区域报错。"""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("只支持 H×W×3 的 RGB 画面")
    if roi is None:
        return np.ascontiguousarray(image)

    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"ROI {roi} 超出画面范围或为空：当前画面宽高为 {width}×{height}"
        )
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
    roi: Optional[Roi],
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
