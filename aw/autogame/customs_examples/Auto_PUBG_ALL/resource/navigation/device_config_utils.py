"""Legacy device/config calibration helpers.

Resolution and shell execution intentionally come from the shared
``aw.autogame.tools.Utils`` implementation.
"""

import json
import re
import subprocess

from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs
from aw.autogame.tools.Utils import get_resolution, get_wh, run_shell


def get_dms_rotation_mode():
    try:
        result = subprocess.run(
            ["hdc", "shell", "hidumper", "-s", "DisplayManagerService", "-a", "-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.SubprocessError as exc:
        print("Error running hdc:", exc)
        return None

    match = re.search(r"^\s*Rotation:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_tuple_str(value):
    if not isinstance(value, str):
        return value
    normalized = value.strip().strip('"').strip("'")
    if not (normalized.startswith("(") and normalized.endswith(")")):
        raise ValueError(f"字符串格式错误：{normalized}")
    normalized = normalized[1:-1].strip()
    if not normalized:
        return ()
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    try:
        return tuple(float(item) for item in parts)
    except ValueError as exc:
        raise ValueError(f"无法解析为数字：{normalized}") from exc


def get_buttons():
    width, height = get_resolution()
    button_dict = {}
    with open(r"config\config.json", "r", encoding="utf-8") as file:
        buttons = json.load(file)["button"]
    for key, value in buttons.items():
        ratios = parse_tuple_str(value)
        button_dict[key] = (
            int(ratios[0] * width),
            int(ratios[1] * height),
        )
    return button_dict


def get_rois():
    width, height = get_wh()
    resize_w, resize_h = (
        (width, height)
        if width > height
        else (height, width)
    )
    roi_dict = {}
    with open(r"config\config.json", "r", encoding="utf-8") as file:
        rois = json.load(file)["roi"]
    for key, value in rois.items():
        ratios = parse_tuple_str(value)
        roi_dict[key] = (
            int(ratios[0] * resize_w),
            int(ratios[1] * resize_h),
            int(ratios[2] * resize_w),
            int(ratios[3] * resize_h),
        )
    return roi_dict


def get_brightness():
    text = run_shell("hdc shell hidumper -s 3308", r=True)
    return int(re.search(r"DeviceBrightness=(\d+)", text).group(1))


def get_auto_brightness():
    text = run_shell("hdc shell hidumper -s 3308", r=True)
    return re.search(
        r"Auto Adjust Brightness:\s*(ON|OFF)",
        text,
    ).group(1) == "ON"
