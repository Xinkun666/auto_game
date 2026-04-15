import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import (
    calculate_move_count,
)
from aw.autogame.tools.Utils import get_resolution, get_wh

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class HouseExitManager:
    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2

    DOOR_CLASS_IDS = {0, 4}
    WINDOW_CLASS_IDS = {2}

    SCAN_OFFSETS = [0, 45, -45, 90, -90, 135, -135, 180]

    def __init__(self):
        self.screen_w, self.screen_h = get_resolution()
        self.scan_anchor_direction: Optional[float] = None
        self.scan_index = 0

    def reset(self):
        self.scan_anchor_direction = None
        self.scan_index = 0

    def process(self, w: "FrameWorker") -> bool:
        house_scene = self._get_house_scene(w)
        if house_scene != self.HOUSE_INDOOR:
            if house_scene == self.HOUSE_OUTDOOR:
                return self._verify_exit_success(w)
            return False

        self._ensure_scan_anchor(w)
        detections = self._get_forward_scene(w)

        door = self._find_largest_target(detections, self.DOOR_CLASS_IDS)
        if door:
            print("[HouseExit] 发现门，准备出门")
            return self._exit_via_door(w, door)

        window = self._find_largest_target(detections, self.WINDOW_CLASS_IDS)
        if window:
            print("[HouseExit] 发现窗，准备翻窗")
            return self._exit_via_window(w, window)

        self._search_for_exit(w)
        return False

    def _get_house_scene(self, w: "FrameWorker") -> Optional[int]:
        value = w.get_info("house_scene")
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_forward_scene(self, w: "FrameWorker") -> List[List[float]]:
        scene = w.get_info("forward_scene")
        if not scene or isinstance(scene, bool):
            return []
        return list(scene)

    def _find_largest_target(
        self,
        detections: Sequence[Sequence[float]],
        class_ids: set,
    ) -> Optional[Sequence[float]]:
        candidates = [
            obj for obj in detections
            if len(obj) >= 6 and int(obj[5]) in class_ids
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def _exit_via_door(self, w: "FrameWorker", door: Sequence[float]) -> bool:
        self._align_to_target(w, door)
        w.refresh_frame()

        if w.get_info("开门"):
            print("[HouseExit] 门已关闭，先开门")
            w.click("开门")
            time.sleep(0.8)
            w.refresh_frame()

        print("[HouseExit] 小步靠近门口")
        self._move_forward(w, dura=220, wait=380)
        return self._verify_exit_success(w)

    def _exit_via_window(self, w: "FrameWorker", window: Sequence[float]) -> bool:
        for _ in range(4):
            self._align_to_target(w, window)
            w.refresh_frame()

            if w.get_info("跳跃"):
                print("[HouseExit] 靠窗后检测到跳跃，执行翻窗")
                w.click("跳跃")
                time.sleep(0.2)
                self._move_forward(w, dura=650, wait=1200)
                return self._verify_exit_success(w)

            print("[HouseExit] 小步靠近窗户")
            self._move_forward(w, dura=180, wait=320)
            w.refresh_frame()

            window = self._find_largest_target(self._get_forward_scene(w), self.WINDOW_CLASS_IDS)
            if not window:
                break

        return False

    def _verify_exit_success(self, w: "FrameWorker") -> bool:
        w.refresh_frame()
        if self._get_house_scene(w) != self.HOUSE_OUTDOOR:
            return False

        print("[HouseExit] 首次判定已在屋外，执行二次确认")
        self._move_forward(w, dura=420, wait=900)
        current_dir = w.get_info("direction")
        if current_dir is None:
            return False

        self._align_direction_blocking(w, current_dir, (current_dir + 180) % 360)
        w.refresh_frame()

        if self._get_house_scene(w) == self.HOUSE_OUTDOOR:
            print("[HouseExit] 出房成功")
            self.reset()
            return True

        print("[HouseExit] 二次确认失败，继续尝试出房")
        return False

    def _search_for_exit(self, w: "FrameWorker"):
        current_dir = w.get_info("direction")
        if current_dir is None:
            return

        target_dir = (self.scan_anchor_direction + self.SCAN_OFFSETS[self.scan_index]) % 360
        print(f"[HouseExit] 扫视角度 {target_dir:.1f}")
        self._align_direction_blocking(w, current_dir, target_dir, tolerance=8)
        w.refresh_frame()

        self.scan_index += 1
        if self.scan_index >= len(self.SCAN_OFFSETS):
            self.scan_index = 0
            self.scan_anchor_direction = None
            print("[HouseExit] 一圈未发现门窗，向前走并换个朝向继续搜索")
            self._move_forward(w, dura=320, wait=500)
            self._rotate_search_view(w)
            w.refresh_frame()

    def _rotate_search_view(self, w: "FrameWorker"):
        current_dir = w.get_info("direction")
        if current_dir is None:
            return
        self._align_direction_blocking(w, current_dir, (current_dir + 60) % 360, tolerance=10)

    def _ensure_scan_anchor(self, w: "FrameWorker"):
        if self.scan_anchor_direction is None:
            self.scan_anchor_direction = w.get_info("direction")
            self.scan_index = 0

    def _move_forward(self, w: "FrameWorker", dura=350, wait=600):
        w.tap_single("摇杆", y_bias=-300, dura=dura, wait=wait)

    def _align_to_target(self, w: "FrameWorker", target: Sequence[float], tolerance_px=80):
        for _ in range(6):
            inf_w, inf_h = get_wh()
            frame_w = max(inf_w, inf_h)
            scale = self.screen_w / frame_w
            target_center_x = (target[0] + target[2]) / 2
            offset_real = (target_center_x - (frame_w / 2)) * scale

            if abs(offset_real) <= tolerance_px:
                return True

            adjust_val = int(offset_real * 0.33)
            adjust_val = max(-400, min(400, adjust_val))
            w.tap_single("视角", x_bias=adjust_val, dura=500, wait=500)
            w.refresh_frame()

            detections = self._get_forward_scene(w)
            refreshed_target = self._find_largest_target(
                detections,
                self.DOOR_CLASS_IDS if int(target[5]) in self.DOOR_CLASS_IDS else self.WINDOW_CLASS_IDS,
            )
            if refreshed_target is None:
                return False
            target = refreshed_target
        return False

    def _align_direction_blocking(self, w: "FrameWorker", current_dir, target_angle, tolerance=5):
        for _ in range(10):
            turn_dir, px, diff = calculate_move_count(current_dir, target_angle)
            if diff <= tolerance:
                return True
            x_bias = px if turn_dir == "right" else -px
            w.tap_single("视角", x_bias=int(x_bias), dura=800, wait=500)
            w.refresh_frame()
            current_dir = w.get_info("direction")
            if current_dir is None:
                return False
        return False
