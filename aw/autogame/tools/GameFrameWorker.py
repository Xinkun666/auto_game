import os
import gc
import json
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
import logging

from aw.autogame.tools.Utils import *
from aw.autogame.tools.Utils import _parse_display_rotation
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
from aw.autogame.tools.GameSceneHandler import DEFAULT_GROUP_NAME, StageLogicController
from aw.autogame.tools.ProcessUtils import hdc_command_args, hidden_subprocess_kwargs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)

def _is_timed_special_info(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    timing = value[1]
    if not isinstance(timing, (list, tuple)) or len(timing) != 1:
        return False
    try:
        float(timing[0])
    except (TypeError, ValueError):
        return False
    return True


def _unwrap_timed_special_info(value):
    if _is_timed_special_info(value):
        return value[0]
    return value


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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            **hidden_subprocess_kwargs(),
        )

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            **hidden_subprocess_kwargs(),
        )

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
                width, height = int(match.group(1)), int(match.group(2))
                if rotation is None:
                    return max(width, height), min(width, height)
                return normalize_resolution_by_rotation(width, height, rotation)
        except Exception:
            pass

        try:
            ret = self.run_cmd_with_ret("hidumper -s RenderService -a screen")
            match = re.search(r"activeMode:\s*(\d+)\s*x\s*(\d+)", ret)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                if rotation is None:
                    return max(width, height), min(width, height)
                return normalize_resolution_by_rotation(width, height, rotation)
        except Exception:
            pass

        raise RuntimeError("获取分辨率失败，请检查 hdc shell 输出")

    def get_screen_rotation(self):
        candidates = [
            "hidumper -s DisplayManagerService -a -a",
            "hidumper -s WindowManagerService -a '-a'",
            "snapshot_display",
            "hidumper -s RenderService -a screen",
        ]

        for cmd in candidates:
            try:
                ret = self.run_cmd_with_ret(cmd)
                rotation = _parse_display_rotation(ret)
                if rotation is None:
                    match = re.search(r"rotation[^0-9]*(\d+)", ret, re.IGNORECASE)
                    if not match:
                        continue
                    rotation = normalize_rotation(match.group(1))
                if rotation is None:
                    continue

                return rotation
            except Exception:
                continue

        return None


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
    rotation = normalize_rotation(dut_handle.get_screen_rotation())
    if rotation is None:
        try:
            pixel_w, pixel_h = dut_handle.get_resolution()
            rotation = infer_landscape_rotation(pixel_w, pixel_h)
        except Exception:
            rotation = 0
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
