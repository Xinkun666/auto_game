import os
import gc
import math
import re
import time
import tempfile
import subprocess
import threading
import importlib
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from aw.autogame.tools.Utils import *
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
from aw.autogame.tools.GameSceneHandler import StageLogicController

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class HdcDut:
    """基于 hdc 的设备控制封装。"""

    def __init__(self, device_id=None):
        self.device_id = device_id

    def _build_cmd(self, *args):
        cmd = ["hdc"]
        if self.device_id not in (None, "", 0):
            cmd += ["-t", str(self.device_id)]
        cmd += [str(arg) for arg in args]
        return cmd

    def run_cmd_with_ret(self, shell_cmd):
        cmd = self._build_cmd("shell", shell_cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"[run_cmd_with_ret] 执行失败\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return result.stdout.strip()

    def push_file(self, source_path, dest_path):
        cmd = self._build_cmd("file", "send", source_path, dest_path)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"[push_file] 推送失败\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

    def run_cmd_by_file_with_ret(self, cmd_str):
        remote_dir = "/data/test"
        filename = f"temp_cmd_{int(time.time() * 1000)}.txt"
        remote_path = f"{remote_dir}/{filename}"

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tmp_file:
            local_path = tmp_file.name
            tmp_file.write(cmd_str)

        try:
            self.push_file(local_path, remote_path)
            self.run_cmd_with_ret(f"chmod 777 {remote_dir}/EventRecordV3_ohos")
            return self.run_cmd_with_ret(f"cd {remote_dir} && ./EventRecordV3_ohos {filename}")
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    def get_resolution(self):
        rotation = normalize_rotation(self.get_screen_rotation())

        try:
            ret = self.run_cmd_with_ret("wm size")
            match = re.search(r"(\d+)\s*x\s*(\d+)", ret)
            if match:
                return normalize_resolution_by_rotation(
                    int(match.group(1)),
                    int(match.group(2)),
                    rotation,
                )
        except Exception:
            pass

        try:
            ret = self.run_cmd_with_ret("hidumper -s RenderService -a screen")
            match = re.search(r"activeMode:\s*(\d+)x(\d+)", ret)
            if match:
                return normalize_resolution_by_rotation(
                    int(match.group(1)),
                    int(match.group(2)),
                    rotation,
                )
        except Exception:
            pass

        raise RuntimeError("获取分辨率失败，请检查 hdc shell 输出")

    def get_screen_rotation(self):
        candidates = [
            "hidumper -s RenderService -a screen",
            "hidumper -s WindowManagerService -a '-a'",
            "snapshot_display",
        ]

        for cmd in candidates:
            try:
                ret = self.run_cmd_with_ret(cmd)
                match = re.search(r"rotation[^0-9]*(\d+)", ret, re.IGNORECASE)
                if not match:
                    continue

                value = int(match.group(1))
                if value in (0, 90, 180, 270):
                    return value
                if value == 1:
                    return 90
                if value == 2:
                    return 180
                if value == 3:
                    return 270
                return 0
            except Exception:
                continue

        return 0


def get_dut_controller_handle(device_id=""):
    if device_id:
        return HdcDut(device_id)
    return HdcDut()


class BezierEasing:
    """三次贝塞尔缓动函数。"""

    def __init__(self, control1: Tuple[float, float], control2: Tuple[float, float]):
        self.c1 = np.array(control1)
        self.c2 = np.array(control2)

    def __call__(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0

        x = t
        for _ in range(8):
            fx = self._bezier_x(x) - t
            dfx = self._bezier_dx(x)
            if abs(dfx) < 1e-6:
                break

            x_next = np.clip(x - fx / dfx, 0, 1)
            if abs(x_next - x) < 1e-6:
                x = x_next
                break
            x = x_next

        return self._bezier_y(x)

    def _bezier_x(self, t: float) -> float:
        return 3 * (1 - t) ** 2 * t * self.c1[0] + 3 * (1 - t) * t ** 2 * self.c2[0] + t ** 3

    def _bezier_y(self, t: float) -> float:
        return 3 * (1 - t) ** 2 * t * self.c1[1] + 3 * (1 - t) * t ** 2 * self.c2[1] + t ** 3

    def _bezier_dx(self, t: float) -> float:
        return 3 * self.c1[0] * (1 - 4 * t + 3 * t ** 2) + 3 * self.c2[0] * (2 * t - 3 * t ** 2) + 3 * t ** 2


def parse_name_re(src_string, re_string):
    match_info = re.findall(re_string, src_string)
    if match_info:
        return match_info[0]
    return ""


def get_panel_abs_xy(dev_info_str):
    dev_info_lines = dev_info_str.splitlines()
    event_map = {}
    need_read_name_flag = False
    now_driver_name = ""
    device_name = ""

    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1

    for line in dev_info_lines:
        if (not need_read_name_flag) or ("add device" in line):
            device_driver_name = parse_name_re(line, re_string=r"/dev/input/(\w+)")
            if device_driver_name:
                now_driver_name = device_driver_name
            x_min = -1
            x_max = -1
            y_min = -1
            y_max = -1
            need_read_name_flag = True
            continue

        if "name" in line:
            device_name = parse_name_re(line, re_string=r'"(.*)"')
            if not device_name:
                continue

        if "input_mt" not in device_name:
            continue

        if "0035" not in line and "0036" not in line:
            continue

        if "0035" in line:
            x_min = int(parse_name_re(line, re_string=r"min\s*(\d+)"))
            x_max = int(parse_name_re(line, re_string=r"max\s*(\d+)"))

        if "0036" in line:
            y_min = int(parse_name_re(line, re_string=r"min\s*(\d+)"))
            y_max = int(parse_name_re(line, re_string=r"max\s*(\d+)"))

        if x_min >= 0 and x_max >= 0 and y_min >= 0 and y_max >= 0 and now_driver_name:
            event_map[now_driver_name] = [x_min, y_min, x_max, y_max]
            need_read_name_flag = False

    print(f"all panel abs xy is {event_map}")

    if not event_map:
        raise RuntimeError("未解析到触摸设备，请检查 /data/test/getevent -p 输出")

    choose_panel = list(event_map.values())[0]
    device_name = list(event_map.keys())[0]
    return choose_panel[2], choose_panel[3], device_name


def pixel_2_panel_abs_pt(x0, y0, pixel_w, pixel_h, abs_w0, abs_h0, abs_w, abs_h):
    abs_x = round(x0 * (abs_w - abs_w0) / pixel_w) + abs_w0
    abs_y = round(y0 * (abs_h - abs_h0) / pixel_h) + abs_h0
    return abs_x, abs_y


def check_tool_exist(dut_handle):
    try:
        ret = dut_handle.run_cmd_with_ret("ls /data/test")
    except Exception:
        dut_handle.run_cmd_with_ret("mkdir -p /data/test")
        ret = ""

    if "EventRecordV3_ohos" not in ret or "getevent" not in ret:
        print("录制回放工具不存在，开始推送")
        dut_handle.run_cmd_with_ret("mkdir -p /data/test")
        dut_handle.push_file(os.path.join(BASE_DIR, "getevent"), "/data/test/getevent")
        dut_handle.push_file(os.path.join(BASE_DIR, "EventRecordV3_ohos"), "/data/test/EventRecordV3_ohos")
        dut_handle.run_cmd_with_ret("chmod 777 /data/test/getevent")
        dut_handle.run_cmd_with_ret("chmod 777 /data/test/EventRecordV3_ohos")

    ret = dut_handle.run_cmd_with_ret("ls /data/test")
    if "EventRecordV3_ohos" not in ret or "getevent" not in ret:
        raise RuntimeError("录制回放工具推送失败，请检查 root 权限和 usb 调试权限")


def get_panel_abs_xy_check_rotation(dut_handle):
    device_info = dut_handle.run_cmd_with_ret("/data/test/getevent -p")
    abs_w, abs_h, input_device = get_panel_abs_xy(device_info)
    rotation = int(dut_handle.get_screen_rotation())
    return 0, 0, abs_w, abs_h, input_device, rotation


def gen_cmd_str_by_list(cmd_list):
    return " && ".join(cmd_list)


def gen_cmd_str_by_list2(cmd_list):
    return "\n".join(cmd_list)


def gen_x_cmd2(input_device, x_abs):
    return f"{input_device}: 0003 0035 {x_abs:08x}"


def gen_y_cmd2(input_device, y_abs):
    return f"{input_device}: 0003 0036 {y_abs:08x}"


def gen_touch_shape_cmd2(input_device):
    pressure_cmd = f"{input_device}: 0003 003a 00000200"
    tracking_id = f"{input_device}: 0003 0039 00000000"
    major_cmd = f"{input_device}: 0003 0030 0000007e"
    minor_cmd = f"{input_device}: 0003 0031 00000073"
    orientation_cmd = f"{input_device}: 0003 0034 fffffff3"
    blob_cmd = f"{input_device}: 0003 0038 00000000"
    sync_mt_cmd = f"{input_device}: 0000 0002 00000000"
    sync_cmd = f"{input_device}: 0000 0000 00000000"
    return gen_cmd_str_by_list2([
        pressure_cmd,
        tracking_id,
        major_cmd,
        minor_cmd,
        orientation_cmd,
        blob_cmd,
        sync_mt_cmd,
        sync_cmd,
    ])


def gen_touch_shape_cmd_with_down2(input_device):
    pressure_cmd = f"{input_device}: 0003 003a 00000200"
    tracking_id = f"{input_device}: 0003 0039 00000000"
    major_cmd = f"{input_device}: 0003 0030 0000007e"
    minor_cmd = f"{input_device}: 0003 0031 00000073"
    orientation_cmd = f"{input_device}: 0003 0034 fffffff3"
    blob_cmd = f"{input_device}: 0003 0038 00000000"
    sync_mt_cmd = f"{input_device}: 0000 0002 00000000"
    down_cmd = f"{input_device}: 0001 014a 00000001"
    sync_cmd = f"{input_device}: 0000 0000 00000000"
    return gen_cmd_str_by_list2([
        pressure_cmd,
        tracking_id,
        major_cmd,
        minor_cmd,
        orientation_cmd,
        blob_cmd,
        sync_mt_cmd,
        down_cmd,
        sync_cmd,
    ])


def gen_up_cmd2(input_device):
    sync_mt_cmd = f"{input_device}: 0000 0002 00000000"
    up_cmd = f"{input_device}: 0001 014a 00000000"
    sync_cmd = f"{input_device}: 0000 0000 00000000"
    return gen_cmd_str_by_list2([sync_mt_cmd, up_cmd, sync_cmd])


def gen_single_move_cmd_str2(point_idx, input_device, x_abs, y_abs, button_status="no"):
    org_input_device = input_device
    now_time = point_idx * 5 / 1000
    int_str = f"{int(now_time):>8}"
    frac_str = f"{now_time:.6f}".split(".")[1]
    input_device = f"[{int_str}.{frac_str}] /dev/input/{input_device}"

    x_cmd = gen_x_cmd2(input_device, x_abs)
    y_cmd = gen_y_cmd2(input_device, y_abs)

    if button_status == "no":
        shape_cmd = gen_touch_shape_cmd2(input_device)
    elif button_status == "down":
        shape_cmd = gen_touch_shape_cmd_with_down2(input_device)
    else:
        shape_cmd = gen_touch_shape_cmd2(input_device)
        now_time = (point_idx + 2) * 5 / 1000
        int_str = f"{int(now_time):>8}"
        frac_str = f"{now_time:.6f}".split(".")[1]
        input_device = f"[{int_str}.{frac_str}] /dev/input/{org_input_device}"
        shape_cmd = shape_cmd + "\n" + gen_up_cmd2(input_device) + "\n"

    return gen_cmd_str_by_list2([x_cmd, y_cmd, shape_cmd])


def gen_slider_points_by_bezier(start, end, n):
    x1, y1 = start
    x2, y2 = end
    easing = BezierEasing((0.42, 0.0), (0.58, 1.0))
    t_linear = np.linspace(0, 1, n)
    t_eased = np.array([easing(t) for t in t_linear])

    x_array = x1 + (x2 - x1) * t_eased
    y_array = y1 + (y2 - y1) * t_eased
    x_list = [int(x) for x in x_array.tolist()]
    y_list = [int(y) for y in y_array.tolist()]
    return list(zip(x_list, y_list))


def gen_slider_points_by_linear(start, end, n):
    x1, y1 = start
    x2, y2 = end
    t_linear = np.linspace(0, 1, n)
    x_array = x1 + (x2 - x1) * t_linear
    y_array = y1 + (y2 - y1) * t_linear
    x_list = [int(round(x)) for x in x_array.tolist()]
    y_list = [int(round(y)) for y in y_array.tolist()]
    return list(zip(x_list, y_list))


def gen_slider_points(start, end, n, trajectory="linear"):
    if trajectory == "linear":
        return gen_slider_points_by_linear(start, end, n)
    if trajectory == "bezier":
        return gen_slider_points_by_bezier(start, end, n)
    raise ValueError(f"不支持的轨迹类型: {trajectory}")


@dataclass
class FingerState:
    finger_id: int
    tracking_id: int
    x_abs: int
    y_abs: int
    x_px: int
    y_px: int
    pressure: int = 0x024C
    major: int = 0x00CE
    minor: int = 0x00CE
    orientation: int = -44
    blob: int = 0
    contact_phase: int = 0


class MultiTouchController:
    """缓存整段多指轨迹，最后一次性回放。"""
    RECORD_CONTACT_PROFILES = (
        {
            "vendor_2a": 0x00001B00,
            "vendor_2b": 0x000D3B2C,
            "pressure": 0x000001B8,
            "major": 0x000000CE,
            "minor": 0x000000A9,
            "orientation": 0x00000054,
            "blob": 0x00000000,
        },
        {
            "vendor_2a": None,
            "vendor_2b": 0x000D72E6,
            "pressure": 0x000001E6,
            "major": 0x000000CE,
            "minor": 0x000000A9,
            "orientation": 0x00000055,
            "blob": 0x00000002,
        },
        {
            "vendor_2a": None,
            "vendor_2b": 0x000DAA57,
            "pressure": 0x00000197,
            "major": 0x000000BC,
            "minor": 0x000000A9,
            "orientation": 0x00000055,
            "blob": 0x00000002,
        },
        {
            "vendor_2a": None,
            "vendor_2b": 0x000DCACB,
            "pressure": 0x000000D3,
            "major": 0x000000A9,
            "minor": 0x000000A9,
            "orientation": 0x00000038,
            "blob": 0x00000002,
        },
    )

    def __init__(self, dut_handle, frame_interval_ms: int = 80):
        self.dut_handle = dut_handle
        self.abs_w0 = 0
        self.abs_h0 = 0
        self.abs_w = 0
        self.abs_h = 0
        self.input_device = None
        self.rotation = 0
        self.display_pixel_w = 0
        self.display_pixel_h = 0
        self.pixel_w = 0
        self.pixel_h = 0
        self.screen_mapping_ready = False
        self.active_fingers: Dict[int, FingerState] = {}
        self.finger_tracking_map: Dict[int, int] = {}
        self.last_changed_finger_id = None
        self.frames = []
        self.current_time_ms = 0
        self.frame_interval_ms = frame_interval_ms
        self.frame_seq = 0

    def reset_frames(self):
        self.frames = []
        self.current_time_ms = 0
        self.frame_seq = 0

    def _get_report_fingers(self):
        if not self.active_fingers:
            return []

        finger_ids = list(self.active_fingers.keys())
        if self.last_changed_finger_id in self.active_fingers:
            finger_ids.remove(self.last_changed_finger_id)
            finger_ids.insert(0, self.last_changed_finger_id)
        return [self.active_fingers[finger_id] for finger_id in finger_ids]

    def refresh_screen_mapping(self, force: bool = False):
        if self.screen_mapping_ready and not force:
            return
        self.abs_w0, self.abs_h0, self.abs_w, self.abs_h, self.input_device, self.rotation = \
            get_panel_abs_xy_check_rotation(self.dut_handle)
        current_pixel_w, current_pixel_h = self.dut_handle.get_resolution()
        self.display_pixel_w = current_pixel_w
        self.display_pixel_h = current_pixel_h
        self.pixel_w, self.pixel_h = get_natural_resolution_by_rotation(
            current_pixel_w,
            current_pixel_h,
            self.rotation,
        )
        self.screen_mapping_ready = True

    def _ensure_tracking_id(self, finger_id: int) -> int:
        if finger_id in self.finger_tracking_map:
            return self.finger_tracking_map[finger_id]

        if finger_id < 0 or finger_id > 9:
            raise ValueError("当前设备 tracking_id 范围疑似为 0~9，finger_id 请控制在 0~9")

        used = set(self.finger_tracking_map.values())
        tracking_id = finger_id
        if tracking_id in used:
            tracking_id = 0
            while tracking_id in used:
                tracking_id += 1
            if tracking_id > 9:
                raise RuntimeError("当前活动手指数超过设备支持范围")

        self.finger_tracking_map[finger_id] = tracking_id
        return tracking_id

    def _pixel_to_abs(self, x0: int, y0: int, return_trace: bool = False):
        self.refresh_screen_mapping()
        raw_x_abs, raw_y_abs = pixel_2_panel_abs_pt(
            x0,
            y0,
            self.pixel_w,
            self.pixel_h,
            self.abs_w0,
            self.abs_h0,
            self.abs_w,
            self.abs_h,
        )

        x_abs, y_abs = raw_x_abs, raw_y_abs

        if return_trace:
            trace = {
                "input_display_xy": (int(x0), int(y0)),
                "display_resolution": (int(self.display_pixel_w), int(self.display_pixel_h)),
                "pixel_resolution": (int(self.pixel_w), int(self.pixel_h)),
                "panel_origin_abs": (int(self.abs_w0), int(self.abs_h0)),
                "panel_max_abs": (int(self.abs_w), int(self.abs_h)),
                "panel_rotation": int(self.rotation),
                "rotation_applied_in_panel_transform": False,
                "input_device": self.input_device,
                "raw_panel_abs": (int(raw_x_abs), int(raw_y_abs)),
                "final_panel_abs": (int(x_abs), int(y_abs)),
            }
            return (x_abs, y_abs), trace

        return x_abs, y_abs

    @staticmethod
    def _signed_to_hex8(value: int) -> str:
        return f"{value & 0xffffffff:08x}"

    def _get_contact_profile(self, finger: FingerState):
        idx = min(max(int(finger.contact_phase), 0), len(self.RECORD_CONTACT_PROFILES) - 1)
        return self.RECORD_CONTACT_PROFILES[idx]

    def _build_contact_block(self, prefix: str, finger: FingerState, include_vendor_prefix: bool = False):
        profile = self._get_contact_profile(finger)
        block = []
        if include_vendor_prefix:
            vendor_2a = profile.get("vendor_2a")
            if vendor_2a is not None and self.frame_seq == 0:
                block.append(f"{prefix}: 0003 002a {vendor_2a:08x}")

            # 录制样本里 002b 只出现在每帧第一组 contact 前，且会持续变化；
            # 这里保留 profile 基值并叠加帧序号，避免多指时为每根手指重复塞 vendor 头字段。
            vendor_2b = int(profile["vendor_2b"]) + int(self.frame_seq)
            block.append(f"{prefix}: 0003 002b {vendor_2b:08x}")

        block.extend([
            f"{prefix}: 0003 0035 {finger.x_abs:08x}",
            f"{prefix}: 0003 0036 {finger.y_abs:08x}",
            f"{prefix}: 0003 003a {int(profile['pressure']):08x}",
            f"{prefix}: 0003 0039 {finger.tracking_id:08x}",
            f"{prefix}: 0003 0030 {int(profile['major']):08x}",
            f"{prefix}: 0003 0031 {int(profile['minor']):08x}",
            f"{prefix}: 0003 0034 {self._signed_to_hex8(int(profile['orientation']))}",
            f"{prefix}: 0003 0038 {int(profile['blob']):08x}",
            f"{prefix}: 0000 0002 00000000",
        ])
        return block

    def _build_frame_text(self, timestamp_ms: int, include_btn_touch_down: bool = False, include_btn_touch_up: bool = False):
        now_time = timestamp_ms / 1000.0
        int_str = f"{int(now_time):>8}"
        frac_str = f"{now_time:.6f}".split(".")[1]
        prefix = f"[{int_str}.{frac_str}] /dev/input/{self.input_device}"

        cmd_list = []
        for idx, finger in enumerate(self._get_report_fingers()):
            cmd_list.extend(self._build_contact_block(prefix, finger, include_vendor_prefix=(idx == 0)))

        if include_btn_touch_down:
            cmd_list.append(f"{prefix}: 0001 014a 00000001")

        if include_btn_touch_up:
            cmd_list.append(f"{prefix}: 0000 0002 00000000")
            cmd_list.append(f"{prefix}: 0001 014a 00000000")

        cmd_list.append(f"{prefix}: 0000 0000 00000000")
        return gen_cmd_str_by_list2(cmd_list)

    def commit_frame(self, include_btn_touch_down: bool = False, include_btn_touch_up: bool = False, duration_ms: int = None):
        if duration_ms is None:
            duration_ms = self.frame_interval_ms

        frame_text = self._build_frame_text(
            timestamp_ms=self.current_time_ms,
            include_btn_touch_down=include_btn_touch_down,
            include_btn_touch_up=include_btn_touch_up,
        )
        self.frames.append(frame_text)
        self.current_time_ms += duration_ms
        self.frame_seq += 1
        for finger in self.active_fingers.values():
            finger.contact_phase += 1

    def play(self):
        if not self.frames:
            return
        cmd_text = gen_cmd_str_by_list2(self.frames)
        self.dut_handle.run_cmd_by_file_with_ret(cmd_text)

    def finger_down(self, finger_id: int, x0: int, y0: int):
        if finger_id in self.active_fingers:
            raise ValueError(f"finger_id={finger_id} 已经处于按下状态，不能重复 down")

        x_abs, y_abs = self._pixel_to_abs(x0, y0)
        tracking_id = self._ensure_tracking_id(finger_id)
        self.active_fingers[finger_id] = FingerState(
            finger_id=finger_id,
            tracking_id=tracking_id,
            x_abs=x_abs,
            y_abs=y_abs,
            x_px=int(x0),
            y_px=int(y0),
        )
        self.last_changed_finger_id = finger_id

    def finger_move(self, finger_id: int, x0: int, y0: int):
        if finger_id not in self.active_fingers:
            raise ValueError(f"finger_id={finger_id} 尚未按下，不能 move")

        x_abs, y_abs = self._pixel_to_abs(x0, y0)
        finger = self.active_fingers[finger_id]
        finger.x_abs = x_abs
        finger.y_abs = y_abs
        finger.x_px = int(x0)
        finger.y_px = int(y0)
        self.last_changed_finger_id = finger_id

    def finger_up(self, finger_id: int) -> bool:
        if finger_id not in self.active_fingers:
            raise ValueError(f"finger_id={finger_id} 当前不在按下状态，不能 up")

        self.active_fingers.pop(finger_id)
        self.finger_tracking_map.pop(finger_id, None)
        if self.active_fingers:
            remaining_ids = list(self.active_fingers.keys())
            self.last_changed_finger_id = remaining_ids[0]
        else:
            self.last_changed_finger_id = None
        return len(self.active_fingers) == 0


class FingerReleaseScheduler:
    """后台调度手指延迟释放，避免在业务线程里阻塞等待。"""

    def __init__(self, release_callback):
        self.release_callback = release_callback
        self._condition = threading.Condition()
        self._jobs = {}
        self._versions = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="FingerReleaseScheduler",
        )
        self._thread.start()

    def schedule(self, finger_id: int, delay_ms: int) -> int:
        delay_ms = max(0, int(delay_ms))
        deadline = time.monotonic() + (delay_ms / 1000.0)
        with self._condition:
            version = self._versions.get(finger_id, 0) + 1
            self._versions[finger_id] = version
            self._jobs[finger_id] = (deadline, version)
            self._condition.notify_all()
            return version

    def cancel(self, finger_id: int):
        with self._condition:
            self._versions[finger_id] = self._versions.get(finger_id, 0) + 1
            self._jobs.pop(finger_id, None)
            self._condition.notify_all()

    def cancel_many(self, finger_ids):
        with self._condition:
            for finger_id in finger_ids:
                self._versions[finger_id] = self._versions.get(finger_id, 0) + 1
                self._jobs.pop(finger_id, None)
            self._condition.notify_all()

    def cancel_all(self):
        with self._condition:
            for finger_id in list(self._jobs.keys()):
                self._versions[finger_id] = self._versions.get(finger_id, 0) + 1
            self._jobs.clear()
            self._condition.notify_all()

    def is_current(self, finger_id: int, version: int) -> bool:
        with self._condition:
            return self._versions.get(finger_id) == version

    def get_jobs(self, finger_ids=None):
        with self._condition:
            if finger_ids is None:
                return dict(self._jobs)
            return {
                finger_id: self._jobs[finger_id]
                for finger_id in finger_ids
                if finger_id in self._jobs
            }

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._jobs.clear()
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self):
        while True:
            with self._condition:
                while not self._closed and not self._jobs:
                    self._condition.wait()
                if self._closed:
                    return

                next_deadline = min(deadline for deadline, _ in self._jobs.values())
                timeout = max(0.0, next_deadline - time.monotonic())
                if timeout > 0:
                    self._condition.wait(timeout=timeout)
                    continue

                now = time.monotonic()
                due_jobs = []
                for finger_id, (deadline, version) in list(self._jobs.items()):
                    if deadline <= now:
                        due_jobs.append((finger_id, version))
                        self._jobs.pop(finger_id, None)

            for finger_id, version in due_jobs:
                try:
                    self.release_callback(finger_id, version)
                except Exception as exc:
                    print(f"[FingerReleaseScheduler] finger_id={finger_id} 自动释放失败: {exc}")


class SendEventController:
    """参考 GameFrameWorker.Controller 风格封装 sendevent/EventRecord。"""

    def __init__(self, device_id="", dut_handle=None, frame_interval_ms: int = 16,
                 auto_prepare: bool = True, max_step_px: int = 100, trajectory: str = "linear"):
        self.dut_handle = dut_handle or get_dut_controller_handle(device_id)
        if auto_prepare:
            check_tool_exist(self.dut_handle)
        self.mt = MultiTouchController(self.dut_handle, frame_interval_ms=frame_interval_ms)
        self._legacy_pressed_fingers = set()
        self.max_step_px = max(1, int(max_step_px))
        self.trajectory = trajectory
        self._op_lock = threading.RLock()
        self._release_scheduler = FingerReleaseScheduler(self._handle_scheduled_release)

    def _flush(self):
        self.mt.play()
        self.mt.reset_frames()

    def close(self):
        wait_deadline = None
        with self._op_lock:
            active_finger_ids = list(self.mt.active_fingers.keys())
            pending_jobs = self._release_scheduler.get_jobs(active_finger_ids)
            if pending_jobs:
                latest_release_time = max(deadline for deadline, _ in pending_jobs.values())
                wait_deadline = latest_release_time + 0.1

        if wait_deadline is not None:
            while time.monotonic() < wait_deadline:
                with self._op_lock:
                    active_finger_ids = list(self.mt.active_fingers.keys())
                    if not active_finger_ids:
                        break
                    if not self._release_scheduler.get_jobs(active_finger_ids):
                        break
                time.sleep(0.01)

        with self._op_lock:
            self._release_scheduler.cancel_all()
            active_finger_ids = sorted(self.mt.active_fingers.keys())
            for finger_id in active_finger_ids:
                self._move_up_locked(finger_id)
        self._release_scheduler.close()

    def _cancel_pending_releases(self, finger_ids):
        self._release_scheduler.cancel_many(finger_ids)

    def _schedule_release(self, finger_id: int, delay_ms: int):
        self._release_scheduler.schedule(finger_id, delay_ms)

    def _handle_scheduled_release(self, finger_id: int, version: int):
        with self._op_lock:
            if not self._release_scheduler.is_current(finger_id, version):
                return
            if finger_id not in self.mt.active_fingers:
                return
            self._move_up_locked(finger_id)

    def _release_fingers_locked(self, finger_ids):
        for finger_id in finger_ids:
            if finger_id not in self.mt.active_fingers:
                continue
            was_last = len(self.mt.active_fingers) == 1
            self.mt.finger_up(finger_id)
            self._send_immediate_frame(include_btn_touch_up=was_last)

    def ensure_screen_mapping(self):
        self.mt.refresh_screen_mapping()

    def get_pixel_to_abs_trace(self, x0: int, y0: int):
        self.ensure_screen_mapping()
        _, trace = self.mt._pixel_to_abs(x0, y0, return_trace=True)
        return trace

    @staticmethod
    def _to_unsigned32_decimal(value: int) -> int:
        return int(value) & 0xffffffff

    def _build_immediate_contact_commands(self, input_prefix: str, finger: FingerState):
        # 这里刻意对齐 send_tp_data_demo 的即时 sendevent 方案：
        # 不带 002a/002b vendor 字段，接触参数使用抓包验证过的固定值。
        return [
            f"{input_prefix} 3 53 {int(finger.x_abs)}",
            f"{input_prefix} 3 54 {int(finger.y_abs)}",
            f"{input_prefix} 3 58 512",
            f"{input_prefix} 3 57 {int(finger.tracking_id)}",
            f"{input_prefix} 3 48 126",
            f"{input_prefix} 3 49 115",
            f"{input_prefix} 3 52 4294967283",
            f"{input_prefix} 3 56 0",
            f"{input_prefix} 0 2 0",
        ]

    def _advance_immediate_frame_state(self):
        self.mt.frame_seq += 1
        for finger in self.mt.active_fingers.values():
            finger.contact_phase += 1

    def _build_immediate_frame_command(self, include_btn_touch_down: bool = False, include_btn_touch_up: bool = False):
        self.ensure_screen_mapping()
        input_prefix = f"sendevent /dev/input/{self.mt.input_device}"
        cmd_list = []

        report_fingers = self.mt._get_report_fingers()
        for finger in report_fingers:
            cmd_list.extend(self._build_immediate_contact_commands(input_prefix, finger))

        if include_btn_touch_down:
            cmd_list.append(f"{input_prefix} 1 330 1")

        if include_btn_touch_up:
            cmd_list.append(f"{input_prefix} 0 2 0")
            cmd_list.append(f"{input_prefix} 1 330 0")

        cmd_list.append(f"{input_prefix} 0 0 0")
        return gen_cmd_str_by_list(cmd_list)

    def _send_immediate_frame(self, include_btn_touch_down: bool = False, include_btn_touch_up: bool = False):
        cmd_text = self._build_immediate_frame_command(
            include_btn_touch_down=include_btn_touch_down,
            include_btn_touch_up=include_btn_touch_up,
        )
        self.dut_handle.run_cmd_with_ret(cmd_text)
        self._advance_immediate_frame_state()

    def _quick_tap_locked(self, finger_id: int, pos):
        self._cancel_pending_releases([finger_id])
        self.ensure_screen_mapping()
        x0, y0 = int(pos[0]), int(pos[1])

        if not self.mt.active_fingers:
            self.mt.reset_frames()

        self._ensure_fingers_available([finger_id])
        had_active_fingers = bool(self.mt.active_fingers)

        self.mt.finger_down(finger_id, x0, y0)
        down_cmd = self._build_immediate_frame_command(include_btn_touch_down=not had_active_fingers)
        self._advance_immediate_frame_state()

        was_last = len(self.mt.active_fingers) == 1
        self.mt.finger_up(finger_id)
        up_cmd = self._build_immediate_frame_command(include_btn_touch_up=was_last)
        self._advance_immediate_frame_state()

        self.dut_handle.run_cmd_with_ret(gen_cmd_str_by_list([down_cmd, up_cmd]))

    def _build_legacy_single_touch_text(self, x0, y0, duration_ms):
        x_abs, y_abs = self.mt._pixel_to_abs(x0, y0)
        input_device = self.mt.input_device

        move_time_ms = max(5, int(duration_ms))
        move_count = max(2, round(move_time_ms / 5))
        point_list = gen_slider_points_by_bezier((x_abs, y_abs), (x_abs, y_abs), move_count)

        cmd_list = []
        for point_idx, point in enumerate(point_list):
            button_status = "no"
            if point_idx == 0:
                button_status = "down"
            elif point_idx == len(point_list) - 1:
                button_status = "up"
            cmd_str = gen_single_move_cmd_str2(
                point_idx,
                input_device,
                int(point[0]),
                int(point[1]),
                button_status,
            )
            cmd_list.append(cmd_str)
        return gen_cmd_str_by_list2(cmd_list)

    def _build_legacy_down_text(self, x0, y0):
        x_abs, y_abs = self.mt._pixel_to_abs(x0, y0)
        input_device = self.mt.input_device
        return gen_single_move_cmd_str2(0, input_device, int(x_abs), int(y_abs), "down")

    def _build_legacy_up_text(self):
        input_device = self.mt.input_device
        prefix = f"[{0:>8}.000000] /dev/input/{input_device}"
        return gen_up_cmd2(prefix)

    def _build_legacy_tap_single_text(self, x0, y0, x1, y1, wait_ms, move_time_ms, release=True):
        x0_abs, y0_abs = self.mt._pixel_to_abs(x0, y0)
        x1_abs, y1_abs = self.mt._pixel_to_abs(x1, y1)
        input_device = self.mt.input_device

        cmd_list = []
        point_idx = 0
        cmd_list.append(gen_single_move_cmd_str2(point_idx, input_device, int(x0_abs), int(y0_abs), "down"))
        point_idx += 1

        move_time_ms = max(5, int(move_time_ms))
        move_count = max(2, round(move_time_ms / 5))
        point_list = gen_slider_points_by_bezier((x0_abs, y0_abs), (x1_abs, y1_abs), move_count)

        for point in point_list:
            cmd_list.append(
                gen_single_move_cmd_str2(
                    point_idx,
                    input_device,
                    int(point[0]),
                    int(point[1]),
                    "no",
                )
            )
            point_idx += 1

        hold_count = max(0, round(max(0, int(wait_ms)) / 5))
        for _ in range(hold_count):
            cmd_list.append(gen_single_move_cmd_str2(point_idx, input_device, int(x1_abs), int(y1_abs), "no"))
            point_idx += 1

        if release:
            cmd_list.append(gen_single_move_cmd_str2(point_idx, input_device, int(x1_abs), int(y1_abs), "up"))

        return gen_cmd_str_by_list2(cmd_list)

    def _ensure_fingers_available(self, finger_ids):
        busy_ids = [finger_id for finger_id in finger_ids if finger_id in self.mt.active_fingers]
        if busy_ids:
            raise RuntimeError(f"finger_id 已处于按下状态，不能重复占用: {busy_ids}")

    def _wait_for_releasable_fingers(self, finger_ids):
        finger_ids = list(dict.fromkeys(int(finger_id) for finger_id in finger_ids))
        while True:
            with self._op_lock:
                busy_ids = [finger_id for finger_id in finger_ids if finger_id in self.mt.active_fingers]
                if not busy_ids:
                    return

                pending_jobs = self._release_scheduler.get_jobs(busy_ids)
                non_releasable_ids = [finger_id for finger_id in busy_ids if finger_id not in pending_jobs]
                if non_releasable_ids:
                    raise RuntimeError(f"finger_id 已处于按下状态，不能重复占用: {non_releasable_ids}")

                next_deadline = min(deadline for deadline, _ in pending_jobs.values())

            sleep_s = max(0.005, min(0.02, next_deadline - time.monotonic()))
            time.sleep(sleep_s)

    def _normalize_duration(self, duration_ms):
        if duration_ms is None:
            return self.mt.frame_interval_ms
        return max(0, int(duration_ms))

    def _resolve_trajectory(self, trajectory):
        return trajectory or self.trajectory

    def _resolve_max_step_px(self, max_step_px):
        if max_step_px is None:
            return self.max_step_px
        return max(1, int(max_step_px))

    def _calc_move_frame_count(self, target_map, max_step_px, duration_ms):
        max_distance = 0.0
        for fid, target_pos in target_map.items():
            finger = self.mt.active_fingers[fid]
            dx = int(target_pos[0]) - int(finger.x_px)
            dy = int(target_pos[1]) - int(finger.y_px)
            max_distance = max(max_distance, math.hypot(dx, dy))

        if max_distance > 0:
            return max(1, int(math.ceil(max_distance / float(max_step_px))))

        normalized_duration = self._normalize_duration(duration_ms)
        if normalized_duration <= 0:
            return 1
        return max(1, round(normalized_duration / self.mt.frame_interval_ms))

    def _sync_paths(self, path_map):
        max_len = max(len(path) for path in path_map.values())
        synced = {}
        for finger_id, path in path_map.items():
            if len(path) == max_len:
                synced[finger_id] = path
                continue
            synced[finger_id] = path + [path[-1]] * (max_len - len(path))
        return synced

    def _calc_path_frame_count_from_points(self, start_points, end_points, max_step_px, duration_ms):
        max_distance = 0.0
        for finger_id, start_pos in start_points.items():
            end_pos = end_points[finger_id]
            dx = int(end_pos[0]) - int(start_pos[0])
            dy = int(end_pos[1]) - int(start_pos[1])
            max_distance = max(max_distance, math.hypot(dx, dy))

        if max_distance > 0:
            return max(1, int(math.ceil(max_distance / float(max_step_px))))

        normalized_duration = self._normalize_duration(duration_ms)
        if normalized_duration <= 0:
            return 1
        return max(1, round(normalized_duration / self.mt.frame_interval_ms))

    def _run_multi_finger_path(self, start_points, end_points, wait_ms, move_time_ms, release_map=None,
                               trajectory=None, max_step_px=None):
        self.ensure_screen_mapping()
        finger_ids = sorted(start_points.keys())
        self._ensure_fingers_available(finger_ids)
        release_map = release_map or {finger_id: True for finger_id in finger_ids}
        had_active_fingers = bool(self.mt.active_fingers)

        self.mt.reset_frames()
        for idx, finger_id in enumerate(finger_ids):
            x0, y0 = start_points[finger_id]
            self.mt.finger_down(finger_id, x0, y0)
            self.mt.commit_frame(
                include_btn_touch_down=(not had_active_fingers and idx == 0),
                duration_ms=0,
            )

        move_time_ms = self._normalize_duration(move_time_ms)
        move_frame_count = self._calc_path_frame_count_from_points(
            start_points,
            end_points,
            self._resolve_max_step_px(max_step_px),
            move_time_ms,
        )
        point_count = move_frame_count + 1
        trajectory = self._resolve_trajectory(trajectory)
        path_map = {}
        for finger_id in finger_ids:
            path_map[finger_id] = gen_slider_points(
                start_points[finger_id],
                end_points[finger_id],
                point_count,
                trajectory=trajectory,
            )

        synced_paths = self._sync_paths(path_map)
        per_frame_ms = max(1, round(max(1, move_time_ms) / max(1, point_count - 1)))

        for step_idx in range(1, max(len(path) for path in synced_paths.values())):
            for finger_id in finger_ids:
                x, y = synced_paths[finger_id][step_idx]
                self.mt.finger_move(finger_id, x, y)
            self.mt.commit_frame(duration_ms=per_frame_ms)

        normalized_wait_ms = self._normalize_duration(wait_ms)
        if normalized_wait_ms > 0:
            self.mt.commit_frame(duration_ms=normalized_wait_ms)

        released_finger_ids = []
        for finger_id in finger_ids:
            if release_map.get(finger_id, True):
                self.mt.finger_up(finger_id)
                released_finger_ids.append(finger_id)

        if released_finger_ids:
            self.mt.commit_frame(
                include_btn_touch_up=len(self.mt.active_fingers) == 0,
                duration_ms=self.mt.frame_interval_ms,
            )
        elif any(not release_map.get(finger_id, True) for finger_id in finger_ids):
            # 对于“移动后保持按下”的场景，再补一帧终点接触态，避免设备没有稳定接住按住状态。
            self.mt.commit_frame(duration_ms=self.mt.frame_interval_ms)
        self._flush()

    def click(self, x0, y0, x_bias=0, y_bias=0, finger_id: int = 0, duration_ms: int = 0,
              trajectory=None, max_step_px=None):
        self._wait_for_releasable_fingers([finger_id])
        target_x = x0 + x_bias
        target_y = y0 + y_bias
        normalized_duration = self._normalize_duration(duration_ms)
        if normalized_duration <= 0:
            with self._op_lock:
                self._quick_tap_locked(finger_id, (target_x, target_y))
            return

        self.move_press(finger_id, (target_x, target_y))
        time.sleep(normalized_duration / 1000.0)
        self.move_up(finger_id)

    def click_down(self, x0, y0, x_bias=0, y_bias=0, dura=0, finger_id: int = 0,
                   trajectory=None, max_step_px=None):
        self._wait_for_releasable_fingers([finger_id])
        target_x = x0 + x_bias
        target_y = y0 + y_bias
        self.move_press(finger_id, (target_x, target_y))
        if dura > 0:
            self._schedule_release(finger_id, dura)

    def move_press(self, finger_id: int, pos):
        with self._op_lock:
            self._cancel_pending_releases([finger_id])
            self.ensure_screen_mapping()
            x0, y0 = int(pos[0]), int(pos[1])
            if finger_id in self.mt.active_fingers:
                self.mt.finger_move(finger_id, x0, y0)
                self._send_immediate_frame()
                return
            if not self.mt.active_fingers:
                self.mt.reset_frames()
            self._ensure_fingers_available([finger_id])
            had_active_fingers = bool(self.mt.active_fingers)
            self.mt.finger_down(finger_id, x0, y0)
            self._send_immediate_frame(include_btn_touch_down=not had_active_fingers)

    def move_to(self, finger_id, pos=None, duration_ms: int = 160,
                trajectory=None, max_step_px=None):
        with self._op_lock:
            if isinstance(finger_id, dict) and pos is None:
                target_map = finger_id
            else:
                target_map = {finger_id: pos}

            self._cancel_pending_releases(target_map.keys())

            missing_fingers = [fid for fid in target_map if fid not in self.mt.active_fingers]
            if missing_fingers:
                raise ValueError(f"finger_id={missing_fingers} 尚未按下，不能 move_to")
            missing_targets = [fid for fid, target_pos in target_map.items() if target_pos is None]
            if missing_targets:
                raise ValueError(f"finger_id={missing_targets} 缺少 move_to 目标位置")

            self.ensure_screen_mapping()
            path_map = {}
            move_frame_count = self._calc_move_frame_count(
                target_map,
                self._resolve_max_step_px(max_step_px),
                duration_ms,
            )
            point_count = move_frame_count + 1
            trajectory = self._resolve_trajectory(trajectory)
            for fid, target_pos in target_map.items():
                x0, y0 = int(target_pos[0]), int(target_pos[1])
                finger = self.mt.active_fingers[fid]
                path_map[fid] = gen_slider_points(
                    (finger.x_px, finger.y_px),
                    (x0, y0),
                    point_count,
                    trajectory=trajectory,
                )

            for step_idx in range(1, point_count):
                for fid, path in path_map.items():
                    x_px, y_px = path[step_idx]
                    x_abs, y_abs = self.mt._pixel_to_abs(x_px, y_px)
                    finger = self.mt.active_fingers[fid]
                    finger.x_abs = int(x_abs)
                    finger.y_abs = int(y_abs)
                    finger.x_px = int(x_px)
                    finger.y_px = int(y_px)
                    self.mt.last_changed_finger_id = fid
                self._send_immediate_frame()

    def move_up(self, finger_id: int, duration_ms: int = 0):
        with self._op_lock:
            self._cancel_pending_releases([finger_id])
            self._move_up_locked(finger_id)

    def _move_up_locked(self, finger_id: int):
        if finger_id not in self.mt.active_fingers:
            return
        self._release_fingers_locked([finger_id])

    def click_up(self, finger_id: int = 0, duration_ms: int = 0):
        self.move_up(finger_id, duration_ms=duration_ms)

    def tap_single(self, x0, y0, wait=100, dura=500, x_bias=0, y_bias=1, finger_id: int = 0,
                   release: bool = True, trajectory=None, max_step_px=None):
        normalized_wait = self._normalize_duration(wait)
        self._wait_for_releasable_fingers([finger_id])
        start_pos = (int(x0), int(y0))
        end_pos = (int(x0 + x_bias), int(y0 + y_bias))
        self.move_press(finger_id, start_pos)
        self.move_to(
            finger_id,
            end_pos,
            duration_ms=dura,
            trajectory=trajectory,
            max_step_px=max_step_px,
        )
        if not release:
            return
        if normalized_wait > 0:
            self._schedule_release(finger_id, normalized_wait)
            return
        self.move_up(finger_id)

    def tap_double(self, x1, y1, x2, y2, wait=100, dura=500,
                   x1_bias=0, y1_bias=1, x2_bias=0, y2_bias=1, finger_id: int = 0,
                   release1: bool = True, release2: bool = True, trajectory=None, max_step_px=None):
        second_finger_id = finger_id + 1
        if second_finger_id > 9:
            raise ValueError("tap_double 的 finger_id 最大只能到 8，否则第二根手指会超出设备支持范围")

        normalized_wait = self._normalize_duration(wait)
        self._wait_for_releasable_fingers([finger_id, second_finger_id])
        start_pos_map = {
            finger_id: (int(x1), int(y1)),
            second_finger_id: (int(x2), int(y2)),
        }
        end_pos_map = {
            finger_id: (int(x1 + x1_bias), int(y1 + y1_bias)),
            second_finger_id: (int(x2 + x2_bias), int(y2 + y2_bias)),
        }
        self.move_press(finger_id, start_pos_map[finger_id])
        self.move_press(second_finger_id, start_pos_map[second_finger_id])
        self.move_to(
            end_pos_map,
            duration_ms=dura,
            trajectory=trajectory,
            max_step_px=max_step_px,
        )
        if normalized_wait > 0:
            if release1:
                self._schedule_release(finger_id, normalized_wait)
            if release2:
                self._schedule_release(second_finger_id, normalized_wait)
            return
        if release1:
            self.move_up(finger_id)
        if release2:
            self.move_up(second_finger_id)


class Controller:
    """操作层：负责绝对坐标换算，并根据后端发送触控指令。"""

    def __init__(self, driver, worker, stage_info_raw, backend="uinput", backend_options=None):
        self.buttons = extract_absolute_points(stage_info_raw)
        self.driver = driver
        self.worker = worker
        self.backend = backend
        self.backend_options = backend_options or {}
        self.touch_backend = None
        self._cached_resolution = None
        self._cached_rotation = None

        if self.backend == "sendevent":
            self.touch_backend = SendEventController(**self.backend_options)
        elif self.backend != "uinput":
            raise ValueError(f"不支持的触控后端: {self.backend}")

    def _run_hdc(self, cmd):
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()

    def _require_sendevent_backend(self):
        if self.backend != "sendevent" or self.touch_backend is None:
            raise RuntimeError("当前控制器未启用 sendevent 后端，请在 aw/autogame/config/config.json 中将 touch_backend 设为 'sendevent'")

    def close(self):
        if self.touch_backend and hasattr(self.touch_backend, "close"):
            self.touch_backend.close()

    def _get_cached_resolution(self):
        if self._cached_resolution is None:
            env_w = os.environ.get("AUTOGAME_SCREEN_WIDTH")
            env_h = os.environ.get("AUTOGAME_SCREEN_HEIGHT")
            if env_w and env_h:
                try:
                    self._cached_resolution = (int(env_w), int(env_h))
                    return self._cached_resolution
                except ValueError:
                    pass

            res_w, res_h = get_resolution()
            if res_w is not None and res_h is not None:
                self._cached_resolution = (int(res_w), int(res_h))
        return self._cached_resolution

    def _get_cached_rotation(self):
        if self._cached_rotation is None:
            self._cached_rotation = normalize_rotation(get_display_rotation())
        return self._cached_rotation

    def _get_current_frame_size(self):
        frame = getattr(self.worker, "frame", None)
        if frame is None:
            return None
        try:
            height, width = frame.shape[:2]
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    def _transform_runtime_point(self, x, y, normalized=False, x_bias=0, y_bias=0, return_trace: bool = False):
        current_res = self._get_cached_resolution()
        frame_size = self._get_current_frame_size()
        trace = {
            "trace_type": "runtime_point",
            "input_xy": (x, y),
            "normalized": bool(normalized),
            "bias": (x_bias, y_bias),
            "cached_resolution": current_res,
            "frame_size": frame_size,
            "backend": self.backend,
        }

        if current_res is not None and current_res[0] is not None and current_res[1] is not None:
            screen_width, screen_height = int(current_res[0]), int(current_res[1])
        elif frame_size is not None:
            screen_width, screen_height = int(frame_size[0]), int(frame_size[1])
        else:
            screen_width, screen_height = None, None

        if normalized:
            if screen_width is None or screen_height is None:
                result = (int(round(x + x_bias)), int(round(y + y_bias)))
                trace["display_output"] = result
                return (result, trace) if return_trace else result
            x = float(x) * float(screen_width)
            y = float(y) * float(screen_height)
            trace["scaled_from_normalized"] = (x, y)

        if self.backend != "sendevent":
            result = (int(round(x + x_bias)), int(round(y + y_bias)))
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        if screen_width is None or screen_height is None:
            result = (int(round(x + x_bias)), int(round(y + y_bias)))
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        current_rotation = self._get_cached_rotation()
        trace["display_rotation"] = current_rotation
        trace["screen_size"] = (int(screen_width), int(screen_height))
        trace["pre_rotation_xy"] = (x + x_bias, y + y_bias)
        result = convert_display_point_by_rotation(
            x + x_bias, y + y_bias,
            int(screen_width), int(screen_height),
            current_rotation,
        )
        trace["rotation_applied_in_controller"] = True
        trace["display_output"] = result
        return (result, trace) if return_trace else result

    def _get_abs_pos(self, btn_input, x_bias=0, y_bias=0, return_trace: bool = False):
        if isinstance(btn_input, (list, tuple)) and len(btn_input) == 2:
            val_x, val_y = btn_input

            if 0 <= val_x <= 1.0 and 0 <= val_y <= 1.0:
                result = self._transform_runtime_point(
                    val_x, val_y, normalized=True, x_bias=x_bias, y_bias=y_bias, return_trace=return_trace
                )
                desc = f"Norm({val_x}, {val_y})"
            else:
                result = self._transform_runtime_point(
                    val_x, val_y, normalized=False, x_bias=x_bias, y_bias=y_bias, return_trace=return_trace
                )
                desc = f"Abs({val_x}, {val_y})"

            if return_trace:
                pos, trace = result
                desc = f"Norm({val_x}, {val_y})" if 0 <= val_x <= 1.0 and 0 <= val_y <= 1.0 else f"Abs({pos[0]}, {pos[1]})"
                return pos, desc, trace
            abs_x, abs_y = result
            return (abs_x, abs_y), desc

        if isinstance(btn_input, str):
            stage = self.worker.get_stage()
            if not stage:
                if return_trace:
                    return None, None, None
                return None, None
            full_key = f"{stage}_{btn_input}"
            button_data = self.buttons.get(full_key)
            if return_trace:
                pos, trace = self._transform_button_pos(button_data, x_bias=x_bias, y_bias=y_bias, return_trace=True)
                return pos, full_key, trace
            return self._transform_button_pos(button_data, x_bias=x_bias, y_bias=y_bias), full_key

        if return_trace:
            return None, None, None
        return None, None

    def _transform_button_pos(self, button_data, x_bias=0, y_bias=0, return_trace: bool = False):
        trace = {
            "trace_type": "static_button",
            "bias": (x_bias, y_bias),
            "backend": self.backend,
        }
        if button_data is None:
            if return_trace:
                trace["error"] = "button_data is None"
                return None, trace
            return None

        if isinstance(button_data, (list, tuple)) and len(button_data) == 2:
            result = (int(button_data[0] + x_bias), int(button_data[1] + y_bias))
            trace["input_pos"] = tuple(button_data)
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        if not isinstance(button_data, dict):
            if return_trace:
                trace["error"] = f"unexpected button_data type: {type(button_data).__name__}"
                return None, trace
            return None

        pos = button_data.get("pos")
        if not pos or len(pos) != 2:
            if return_trace:
                trace["error"] = "button_data.pos missing"
                return None, trace
            return None

        x, y = int(pos[0]), int(pos[1])
        trace["input_pos"] = (x, y)
        norm_pos = button_data.get("norm_pos")
        if norm_pos and len(norm_pos) == 2:
            trace["norm_pos"] = (float(norm_pos[0]), float(norm_pos[1]))
        rect = button_data.get("rect")
        if rect and len(rect) == 4:
            trace["rect"] = list(rect)
        current_res = self._get_cached_resolution()
        trace["cached_resolution"] = current_res
        if not current_res or current_res[0] is None or current_res[1] is None:
            result = (x + x_bias, y + y_bias)
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        src_width = int(button_data.get("scene_width") or 0)
        src_height = int(button_data.get("scene_height") or 0)
        dst_width, dst_height = int(current_res[0]), int(current_res[1])
        trace["scene_size"] = (src_width, src_height)
        trace["screen_size"] = (dst_width, dst_height)
        if src_width <= 0 or src_height <= 0:
            result = (x + x_bias, y + y_bias)
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        if "anchor" in button_data or (rect and len(rect) == 4):
            area_config = button_data if "anchor" in button_data else {"rect": rect}
            mapped_rect = resolve_area_rect_for_frame(
                dst_width,
                dst_height,
                area_config,
                dst_width,
                dst_height,
                src_width,
                src_height,
            )
            scaled_x = int(round((mapped_rect[0] + mapped_rect[2]) / 2.0))
            scaled_y = int(round((mapped_rect[1] + mapped_rect[3]) / 2.0))
            scaled_x = min(max(scaled_x, 0), max(dst_width - 1, 0))
            scaled_y = min(max(scaled_y, 0), max(dst_height - 1, 0))
            trace["mapped_rect_to_screen"] = mapped_rect
        elif norm_pos and len(norm_pos) == 2:
            mapped_x = float(norm_pos[0]) * float(dst_width)
            mapped_y = float(norm_pos[1]) * float(dst_height)
            scaled_x = int(round(mapped_x))
            scaled_y = int(round(mapped_y))
            scaled_x = min(max(scaled_x, 0), max(dst_width - 1, 0))
            scaled_y = min(max(scaled_y, 0), max(dst_height - 1, 0))
            trace["mapped_from_norm_to_screen"] = (mapped_x, mapped_y)
            trace["mapped_rect_to_screen"] = None
        else:
            scaled_x, scaled_y = scale_point(
                x, y,
                src_width, src_height,
                dst_width, dst_height,
            )
            trace["mapped_from_norm_to_screen"] = None
            trace["mapped_rect_to_screen"] = None

        trace["scaled_xy"] = (scaled_x, scaled_y)
        trace["pre_rotation_xy"] = (scaled_x + x_bias, scaled_y + y_bias)
        if self.backend != "sendevent":
            result = (scaled_x + x_bias, scaled_y + y_bias)
            trace["display_output"] = result
            return (result, trace) if return_trace else result

        current_rotation = self._get_cached_rotation()
        trace["display_rotation"] = current_rotation
        result = convert_display_point_by_rotation(
            scaled_x + x_bias,
            scaled_y + y_bias,
            dst_width,
            dst_height,
            current_rotation,
        )
        trace["rotation_applied_in_controller"] = True
        trace["display_output"] = result
        return (result, trace) if return_trace else result

    def _resolve_pos(self, btn_input, x_bias=0, y_bias=0, return_trace: bool = False):
        if return_trace:
            pos, label, trace = self._get_abs_pos(btn_input, x_bias=x_bias, y_bias=y_bias, return_trace=True)
            if not pos:
                return None, None, trace
            return pos, label, trace

        pos, label = self._get_abs_pos(btn_input, x_bias=x_bias, y_bias=y_bias)
        if not pos:
            return None, None
        return pos, label

    def tap_double(self, btn1, btn2, wait=100, dura=500,
                   x1_bias=0, y1_bias=1, x2_bias=0, y2_bias=1, finger_id=0,
                   release1=True, release2=True, trajectory=None, max_step_px=None):
        pos1, label1 = self._resolve_pos(btn1)
        pos2, label2 = self._resolve_pos(btn2)
        if pos1 and pos2:
            x1, y1 = pos1
            x2, y2 = pos2
            print(f"执行双指操作: {label1} @({x1},{y1}), {label2} @({x2},{y2})")
            if self.backend == "sendevent":
                end1, _ = self._resolve_pos(btn1, x_bias=x1_bias, y_bias=y1_bias)
                end2, _ = self._resolve_pos(btn2, x_bias=x2_bias, y_bias=y2_bias)
                if not end1 or not end2:
                    return
                self.touch_backend.tap_double(
                    x1,
                    y1,
                    x2,
                    y2,
                    wait=wait,
                    dura=dura,
                    x1_bias=end1[0] - x1,
                    y1_bias=end1[1] - y1,
                    x2_bias=end2[0] - x2,
                    y2_bias=end2[1] - y2,
                    finger_id=finger_id,
                    release1=release1,
                    release2=release2,
                    trajectory=trajectory,
                    max_step_px=max_step_px,
                )
            else:
                cmd = (
                    f"hdc shell uinput -T -m {x1} {y1} {x1 + x1_bias} {y1 + y1_bias} "
                    f"{x2} {y2} {x2 + x2_bias} {y2 + y2_bias} -k {wait} {dura}"
                )
                self._run_hdc(cmd)

    def tap_single(self, btn, wait=100, dura=500, x_bias=0, y_bias=1, finger_id=0,
                   release=True, trajectory=None, max_step_px=None):
        pos, label = self._resolve_pos(btn)
        if pos:
            x, y = pos
            print(f"执行单指操作: {label} @({x},{y})")
            if self.backend == "sendevent":
                end_pos, _ = self._resolve_pos(btn, x_bias=x_bias, y_bias=y_bias)
                if not end_pos:
                    return
                self.touch_backend.tap_single(
                    x,
                    y,
                    wait=wait,
                    dura=dura,
                    x_bias=end_pos[0] - x,
                    y_bias=end_pos[1] - y,
                    finger_id=finger_id,
                    release=release,
                    trajectory=trajectory,
                    max_step_px=max_step_px,
                )
            else:
                cmd = f"hdc shell uinput -T -m {x} {y} {x + x_bias} {y + y_bias} -k {wait} {dura}"
                self._run_hdc(cmd)

    def click_down(self, btn, x_bias=0, y_bias=0, dura=0, finger_id=0,
                   trajectory=None, max_step_px=None):
        pos, label = self._resolve_pos(btn, x_bias=x_bias, y_bias=y_bias)
        if pos:
            x, y = pos
            print(f"执行按下: {label} @({x},{y})")
            if self.backend == "sendevent":
                self.touch_backend.click_down(
                    x,
                    y,
                    dura=dura,
                    finger_id=finger_id,
                    trajectory=trajectory,
                    max_step_px=max_step_px,
                )
            else:
                if dura == 0:
                    cmd = f"hdc shell uinput -T -d {x} {y}"
                else:
                    cmd = f"hdc shell uinput -T -d {x} {y} -i {dura} -u {x} {y}"
                self._run_hdc(cmd)

    def click(self, btn, x_bias=0, y_bias=0, finger_id=0, duration_ms=0,
              trajectory=None, max_step_px=None):
        pos, label = self._resolve_pos(btn, x_bias=x_bias, y_bias=y_bias)
        if pos:
            x, y = pos
            print(f"执行点击: {label} @({x},{y})")
            if self.backend == "sendevent":
                self.touch_backend.click(
                    x,
                    y,
                    finger_id=finger_id,
                    duration_ms=duration_ms,
                    trajectory=trajectory,
                    max_step_px=max_step_px,
                )
            else:
                self._run_hdc(f"hdc shell uinput -T -c {x} {y}")

    def move_press(self, finger_id, pos, x_bias=0, y_bias=0):
        self._require_sendevent_backend()
        target, label = self._get_abs_pos(pos, x_bias=x_bias, y_bias=y_bias)
        if not target:
            raise ValueError("move_press 无法解析目标位置，请检查 pos/x_bias/y_bias")

        x, y = target
        desc = label or pos
        print(f"执行 move_press: finger_id={finger_id} @({x},{y})")
        print(f"move_press 目标: {desc}")
        self.touch_backend.move_press(finger_id, (x, y))

    def move_to(self, finger_id, pos, x_bias=0, y_bias=0, duration_ms=160,
                trajectory=None, max_step_px=None):
        self._require_sendevent_backend()
        target, label = self._get_abs_pos(pos, x_bias=x_bias, y_bias=y_bias)
        if not target:
            raise ValueError("move_to 无法解析目标位置，请检查 pos/x_bias/y_bias")

        x, y = target
        desc = label or pos
        print(f"执行 move_to: finger_id={finger_id} @({x},{y})")
        print(f"move_to 目标: {desc}")
        self.touch_backend.move_to(
            finger_id,
            (x, y),
            duration_ms=duration_ms,
            trajectory=trajectory,
            max_step_px=max_step_px,
        )

    def move_up(self, finger_id, duration_ms=0):
        self._require_sendevent_backend()
        print(f"执行 move_up: finger_id={finger_id}")
        self.touch_backend.move_up(finger_id, duration_ms=duration_ms)

class FrameWorker(threading.Thread):
    LAUNCHER_INACTIVITY_TIMEOUT_SECONDS = 5 * 60
    WATCHDOG_CHECK_INTERVAL_SECONDS = 1.0

    def __init__(self, buffer, driver=None, logger=None, controller_backend=None, controller_options=None):
        super().__init__()
        self.frame_index = 0
        self.viz_queue = mp.Queue(maxsize=5)
        self.viz_proc = None
        self.thread = None

        project_case = os.environ.get("TARGET_PROJECT_CASE")
        if not project_case:
            raise ValueError("TARGET_PROJECT_CASE 未设置")

        info_path = f"aw.autogame.customs_examples.{project_case}.info"
        info_module = importlib.import_module(info_path)
        self.stage_dict = getattr(info_module, "STAGE_DICT")
        raw_stage_info = getattr(info_module, "STAGE_INFO")

        case_name = os.environ.get("TARGET_GAME_CASE")
        if not case_name:
            raise ValueError("TARGET_GAME_CASE 未设置")

        logic_path = f"aw.autogame.customs_game_examples.{project_case}.{case_name}"
        try:
            logic_module = importlib.import_module(logic_path)
            self.on_stage_logic = getattr(logic_module, "on_stage")
            print(f"成功加载业务逻辑: {logic_path}")
        except Exception as exc:
            print(f"加载业务逻辑失败: {exc}")
            raise

        self.buffer = buffer
        self.driver = driver
        self.logger = logger
        self.running = False
        self.finished = False
        self.failed = False
        self.failure_code = None
        self.failure_reason = None
        self.failure_details = {}
        self._failure_lock = threading.Lock()
        self.last_control_action_time = time.monotonic()
        self.launcher_watchdog_enabled = self._is_launcher_mode()
        self.launcher_inactivity_timeout_seconds = self._resolve_launcher_inactivity_timeout_seconds()
        self._watchdog_stop_event = threading.Event()
        self._watchdog_thread = None

        # 触控后端统一从 config.json 读取，controller_backend 仅保留兼容旧调用签名。
        touch_backend = get_touch_backend()
        self.controller = Controller(
            driver,
            self,
            raw_stage_info,
            backend=touch_backend,
            backend_options=controller_options,
        )
        self.stage_resolver = StageLogicController()
        self.stage_info = {}
        self.current_stage = None
        self.frame = None
        self.last_gc_time = time.time()

        self.click = self._wrap_control_action("click", self.controller.click)
        self.click_down = self._wrap_control_action("click_down", self.controller.click_down)
        self.tap_single = self._wrap_control_action("tap_single", self.controller.tap_single)
        self.tap_double = self._wrap_control_action("tap_double", self.controller.tap_double)
        self.move_press = self._wrap_control_action("move_press", self.controller.move_press)
        self.move_to = self._wrap_control_action("move_to", self.controller.move_to)
        self.move_up = self._wrap_control_action("move_up", self.controller.move_up)

    def _is_launcher_mode(self):
        source = os.environ.get("AUTOGAME_RUN_SOURCE", "").strip().lower()
        vis_mode = os.environ.get("AUTOGAME_VIS_MODE", "").strip().lower()
        return source == "launcher" or vis_mode == "launcher"

    def _resolve_launcher_inactivity_timeout_seconds(self):
        raw_value = os.environ.get("AUTOGAME_LAUNCHER_INACTIVITY_TIMEOUT_MINUTES", "").strip()
        if not raw_value:
            return float(self.LAUNCHER_INACTIVITY_TIMEOUT_SECONDS)
        try:
            minutes = float(raw_value)
        except ValueError:
            return float(self.LAUNCHER_INACTIVITY_TIMEOUT_SECONDS)
        return max(0.0, minutes * 60.0)

    def _wrap_control_action(self, action_name, action):
        def _wrapped(*args, **kwargs):
            result = action(*args, **kwargs)
            self._record_control_action(action_name)
            return result
        return _wrapped

    def _record_control_action(self, action_name=None):
        self.last_control_action_time = time.monotonic()

    def mark_failed(self, code, reason, **details):
        with self._failure_lock:
            if self.failed:
                return False
            self.failed = True
            self.failure_code = str(code or "unknown_failure")
            self.failure_reason = str(reason or "自动化执行失败")
            self.failure_details = dict(details or {})
        print(f"[FrameWorker] 标记失败: code={self.failure_code}, reason={self.failure_reason}")
        return True

    def _resolve_launcher_run_archive_dir(self):
        run_index_text = os.environ.get("AUTOGAME_RUN_INDEX", "").strip()
        batch_start_timestamp = os.environ.get("AUTOGAME_BATCH_START_TIMESTAMP", "").strip()
        run_start_timestamp = os.environ.get("AUTOGAME_RUN_START_TIMESTAMP", "").strip()
        if not run_index_text:
            return None

        try:
            run_index = int(run_index_text)
        except ValueError:
            return None

        extra_metadata = {}
        if batch_start_timestamp:
            extra_metadata["batch_start_timestamp"] = batch_start_timestamp
        if run_start_timestamp:
            extra_metadata["run_start_timestamp"] = run_start_timestamp

        return resolve_run_archive_dir(run_index, extra_metadata=extra_metadata, create=True)

    def _capture_launcher_unknown_screenshot(self):
        archive_dir = self._resolve_launcher_run_archive_dir()
        if archive_dir is None:
            archive_dir = TEMP_DIR

        screenshot_dir = os.path.join(str(archive_dir), "unknow_screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        remote_path = f"/data/local/tmp/unknow_screen_{timestamp}.jpeg"
        local_path = os.path.join(screenshot_dir, f"unknow_screen_{timestamp}.jpeg")
        need_remote_rm = False

        try:
            snap_result = subprocess.run(
                ["hdc", "shell", "snapshot_display", "-f", remote_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            if snap_result.returncode != 0:
                raise RuntimeError(snap_result.stderr.strip() or snap_result.stdout.strip())
            need_remote_rm = True

            recv_result = subprocess.run(
                ["hdc", "file", "recv", remote_path, local_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            if recv_result.returncode != 0:
                raise RuntimeError(recv_result.stderr.strip() or recv_result.stdout.strip())

            print(f"[FrameWorker] 已抓取卡死截图: {local_path}")
            return local_path
        except Exception as exc:
            print(f"[FrameWorker] 抓取卡死截图失败: {exc}")
            return None
        finally:
            if need_remote_rm:
                try:
                    subprocess.run(
                        ["hdc", "shell", "rm", remote_path],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=5,
                    )
                except Exception:
                    pass

    def _handle_launcher_inactivity_timeout(self):
        idle_seconds = max(0.0, time.monotonic() - self.last_control_action_time)
        screenshot_path = self._capture_launcher_unknown_screenshot()
        reason = (
            f"launcher 模式下连续 {int(idle_seconds)} 秒未执行操控，"
            f"已判定当前用例卡在未知界面并主动结束。"
        )
        if screenshot_path:
            reason = f"{reason} 截图: {screenshot_path}"

        if self.mark_failed(
            "launcher_inactivity_timeout",
            reason,
            idle_seconds=idle_seconds,
            screenshot_path=screenshot_path,
        ):
            self.running = False
            self.finished = True

    def _watch_launcher_inactivity(self):
        while not self._watchdog_stop_event.wait(self.WATCHDOG_CHECK_INTERVAL_SECONDS):
            if not self.running:
                return
            if self.failed:
                return
            if not self.launcher_watchdog_enabled:
                return
            if self.launcher_inactivity_timeout_seconds <= 0:
                return
            idle_seconds = time.monotonic() - self.last_control_action_time
            if idle_seconds < self.launcher_inactivity_timeout_seconds:
                continue
            self._handle_launcher_inactivity_timeout()
            return

    def loop(self):
        print("GameFrameWorker 引擎已启动")
        while self.running:
            frame = self.buffer.get_latest()
            if frame is None:
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - self.last_gc_time > 30:
                gc.collect()
                self.last_gc_time = current_time

            try:
                self.frame = np.array(frame, copy=True)
                self.current_stage = self.get_stage()
                self.stage_info = self.stage_resolver.process_frame(self.frame, self.current_stage)

                if not self.viz_queue.full() and self.viz_proc:
                    self.viz_queue.put((self.frame.copy(), self.current_stage, self.stage_info, self.frame_index))
                    self.frame_index += 1

                self.on_stage_logic(self)
                time.sleep(0.05)
            except Exception as exc:
                print(f"[Loop Error] 运行时异常: {exc}")
                time.sleep(1)

    def start(self):
        self.running = True
        self.finished = False
        self.failed = False
        self.failure_code = None
        self.failure_reason = None
        self.failure_details = {}
        self.last_control_action_time = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

        if self.launcher_watchdog_enabled:
            self._watchdog_stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watch_launcher_inactivity,
                daemon=True,
                name="LauncherInactivityWatchdog",
            )
            self._watchdog_thread.start()

        self.viz_proc = mp.Process(target=visualizer_process, args=(self.viz_queue,), daemon=True)
        self.viz_proc.start()

    def stop(self):
        print("主动结束游戏自动化中......")
        already_stopped = not self.running

        self.running = False
        self.finished = True
        self._watchdog_stop_event.set()

        if self._watchdog_thread and threading.current_thread() is not self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.0)
        self._watchdog_thread = None

        if self.viz_proc:
            try:
                self.viz_queue.put_nowait("STOP")
            except Exception:
                pass

            self.viz_proc.join(timeout=1.0)
            if self.viz_proc.is_alive():
                self.viz_proc.terminate()
                self.viz_proc.join(timeout=0.5)

            self.viz_queue.cancel_join_thread()

        try:
            self.controller.close()
        except Exception as exc:
            print(f"[FrameWorker] 关闭控制器失败: {exc}")

        if already_stopped:
            print("GameFrameWorker 已停止")
            return

        print("GameFrameWorker 已停止")

    def get_stage(self):
        for stage, active in self.stage_dict.items():
            if active:
                return stage
        return None

    def get_info(self, area_name):
        suffix = f"__{area_name}"
        for key, value in self.stage_info.items():
            if key.endswith(suffix):
                return value
        return None

    def change_stage(self, stage_name):
        if stage_name not in self.stage_dict:
            print(f"\n[ERROR] 切换失败：阶段 '{stage_name}' 不在 STAGE_DICT 中！")
            return

        old_stage = self.current_stage
        for key in self.stage_dict.keys():
            self.stage_dict[key] = False
        self.stage_dict[stage_name] = True
        self.current_stage = stage_name

        print("\n" + ">" * 40)
        print(f"  STATUS CHANGE: [{old_stage}] -> [{stage_name}]")
        print(">" * 40 + "\n")

    def refresh_frame(self):
        frame = self.buffer.get_latest(must_new=True)
        if frame is None:
            print("[FrameWorker] 刷新失败：缓冲区暂无数据")
            return False

        self.frame = np.array(frame, copy=True)
        self.current_stage = self.get_stage()
        self.stage_info = self.stage_resolver.process_frame(self.frame, self.current_stage)

        if not self.viz_queue.full():
            self.viz_queue.put((self.frame.copy(), self.current_stage, self.stage_info, self.frame_index))
            self.frame_index += 1

        return True
