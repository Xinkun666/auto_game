import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_search_manager import (
    HouseSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    get_distance,
)

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class HouseSceneSearchManager(HouseSearchManager):
    """基于 house_scene 五分类的新搜房入口逻辑。

    这套逻辑替换“到达进门点后如何进门”和“进房后如何旋转搜房”的流程，
    选点、导航和出房兜底仍复用旧 HouseSearchManager 的成熟能力，便于新旧逻辑并存和回滚。
    """

    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2
    HOUSE_NEAR_DOOR = 3
    HOUSE_NEAR_WALL = 4
    HOUSE_NEAR_ENTRY_SCENES = {HOUSE_NEAR_DOOR, HOUSE_NEAR_WALL}

    STATUS_SCENE_ENTRY = "SCENE_ENTRY"

    ENTRY_APPROACH_MAX_STEPS = 10
    ENTRY_APPROACH_FORWARD_Y_BIAS = -420
    ENTRY_APPROACH_FORWARD_DURA = 650
    ENTRY_APPROACH_FORWARD_WAIT = 850

    SWEEP_STEP_MS = 250
    BUTTON_SWEEP_MAX_STEPS = 16
    BUTTON_SWEEP_X_BIAS = 320
    BUTTON_SWEEP_WAIT_PAD = 260

    ENTRY_SWEEP_MAX_STEPS = 14
    ENTRY_SWEEP_X_BIAS = 320
    ENTRY_SWEEP_Y_BIAS = -420
    ENTRY_SWEEP_WAIT_PAD = 320
    OPEN_DOOR_SETTLE_SECONDS = 0.8

    ROTATE_SEARCH_MOVE_DURA = 1000
    ROTATE_SEARCH_MOVE_WAIT_PAD = 260
    ROTATE_SEARCH_X_BIAS = 330
    ROTATE_SEARCH_Y_BIAS = -430
    ROTATE_SEARCH_TURN_DEGREES = 90
    ROTATE_SEARCH_HIT_SWITCH_COUNT = 6
    ROTATE_SEARCH_MAX_STEPS = 80
    ROTATE_SEARCH_RECOVER_STEP_MS = 300
    ROTATE_SEARCH_RECOVER_MAX_MS = 1800
    ROTATE_SEARCH_RECOVER_X_BIAS = 330
    ROTATE_FRAME_COMPARE_SIZE = (160, 90)
    ROTATE_FRAME_COMPARE_ROI = (0.18, 0.16, 0.82, 0.78)
    ROTATE_FRAME_MEAN_DIFF_THRESHOLD = 3.5
    ROTATE_FRAME_CHANGED_RATIO_THRESHOLD = 0.02
    ROTATE_FRAME_CHANGED_PIXEL_THRESHOLD = 12

    def searching_logic(self, w: "FrameWorker", current_loc, current_direction):
        if self._should_abort(w):
            return

        if self.searching_number == 5:
            self._continue_searching_until_timer(w, "已经搜满5个房间")
            return

        house_scene = self._get_house_scene(w)
        if house_scene == self.HOUSE_INDOOR and not self._is_entry_approach_status():
            self.indoor_stuck_frames += 1
            if self.indoor_stuck_frames > 30:
                print("[SceneSearch] 检测到长时间困在屋内，启动兜底出房策略")
                self.house_exit_manager.reset()
                for _ in range(20):
                    if self._should_abort(w):
                        return
                    if self.house_exit_manager.process(w):
                        print("[SceneSearch] 兜底出房成功，继续搜房计时")
                        self.indoor_stuck_frames = 0
                        self.searching_number = 0
                        self.completed_houses.add(self.current_house_id)
                        self.current_house_id = None
                        self.status = "IDLE"
                        self._continue_searching_until_timer(w, "兜底出房成功")
                        return
                print("[SceneSearch] 兜底出房失败，重置当前目标")
                self.indoor_stuck_frames = 0
                self.searching_number = 0
                self.current_house_id = None
                self.status = "IDLE"
                self._continue_searching_until_timer(w, "兜底出房失败")
                return
        else:
            self.indoor_stuck_frames = 0

        if self.current_house_id is None:
            if self.initial_target_pending:
                stable_loc = self._get_stable_initial_location(current_loc)
                if stable_loc is None:
                    self.stop_auto_forward(w)
                    w.refresh_frame()
                    return
                current_loc = stable_loc
                self.select_nearest_entry(current_loc)
                self.initial_target_pending = False
            else:
                self.select_smart_target(current_loc, current_direction)

            if not self.current_house_id:
                self._continue_searching_until_timer(w, "当前区域无合适目标或已搜完")
                return

            self.status = "FAST_NAV"
            target_dist = get_distance(current_loc, self.active_entry["location"])
            print(
                f"[SceneSearch] 锁定目标: {self.current_house_id} | "
                f"入口={self.active_entry['location']} | 距离={target_dist:.2f}"
            )
            self.history_locations = []

        target_loc = self.active_entry["location"]
        dist = get_distance(current_loc, target_loc)

        if self.status == "FAST_NAV":
            if self.update_and_check_stuck(current_loc):
                print("[SceneSearch] 快速导航检测到卡住，启动避障")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry["direction"])
                    self.status = "IDLE"
                self.history_locations = []
                return

            if dist <= self.ENTRY_AUTO_FORWARD_DISTANCE:
                print(f"[SceneSearch] 进入摇杆分段导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.align_direction(w, target_loc)

            if not self.auto_forward:
                w.click("自动前进")
                self.auto_forward = True

            self.handle_jump_logic(w)
            return

        if self.status == "PRECISE_NAV":
            if self.update_and_check_stuck(current_loc):
                print("[SceneSearch] 精细导航检测到卡住，启动避障")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry["direction"])
                    self.status = "IDLE"
                self.history_locations = []
                return

            if dist <= self.ENTRY_ARRIVAL_DISTANCE:
                print(f"[SceneSearch] 已到达进门点 (距离 {dist:.2f})，进入 house_scene 进门流程")
                self.stop_auto_forward(w)
                self.status = self.STATUS_SCENE_ENTRY
                return

            self.stop_auto_forward(w)
            self.align_direction(w, target_loc)
            y_bias, dura, wait = self._get_entry_move_params(dist)
            print(
                f"[SceneSearch] 分段推进到进门点: "
                f"dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
            )
            w.tap_single("摇杆", y_bias=y_bias, dura=dura, wait=wait)
            w.refresh_frame()
            self.handle_jump_logic(w)
            return

        if self.status == self.STATUS_SCENE_ENTRY:
            if not self._enter_house_by_scene(w):
                if self._should_abort(w):
                    return
                print("[SceneSearch] house_scene 进门失败，舍弃当前进门点")
                if self.active_entry:
                    self.handle_failed_entry_logic(self.active_entry["direction"])
                else:
                    self.current_house_id = None
                self.status = "IDLE"
                return

            if self._should_abort(w):
                return
            if not self.start_searching(w):
                return
            if w.current_stage != "搜房阶段":
                return

            self.completed_houses.add(self.current_house_id)
            self.searching_number += 1
            print(f"[SceneSearch] 房屋 {self.current_house_id} 完成，已搜 {self.searching_number}/5")
            w.refresh_frame()
            exit_direction = w.get_info("direction")
            self.prepare_next_target_logic(exit_direction)
            self.current_house_id = None
            self.status = "IDLE"

    def _is_entry_approach_status(self):
        return super()._is_entry_approach_status() or self.status == self.STATUS_SCENE_ENTRY

    def start_searching(self, w: "FrameWorker"):
        if self._should_abort(w):
            return False

        self._start_house_search_timer()
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        print("[SceneRotate] 进入房屋，启动 house_scene 旋转搜房")
        self._rotate_search_inside_house(w)

        if self._should_abort(w):
            self._clear_house_search_timer()
            return False

        self._clear_house_search_timer()
        w.refresh_frame()
        if self._get_house_scene(w) != self.HOUSE_INDOOR:
            print("[SceneRotate] 旋转搜房结束时已不在屋内，无需出房")
            return True

        print("[SceneRotate] 旋转搜房结束，开始出房")
        self._exit_house(w)
        return not self._should_abort(w)

    def _rotate_search_inside_house(self, w: "FrameWorker"):
        move_mode = "left_up"
        wall_hit_count = 0
        recover_ms = {
            "left_up": self.ROTATE_SEARCH_RECOVER_STEP_MS,
            "right_up": self.ROTATE_SEARCH_RECOVER_STEP_MS,
        }

        for step in range(self.ROTATE_SEARCH_MAX_STEPS):
            if self._should_abort(w) or self._house_search_timed_out():
                break

            w.refresh_frame()
            scene_before = self._get_house_scene(w)
            if scene_before is None:
                print("[SceneRotate] 当前 house_scene 暂未识别，继续按室内旋转搜房尝试")
            elif scene_before not in {
                self.HOUSE_INDOOR,
                self.HOUSE_NEAR_DOOR,
                self.HOUSE_NEAR_WALL,
            }:
                print(f"[SceneRotate] 当前 house_scene={scene_before}，停止室内旋转搜房")
                break

            before_frame = self._copy_current_frame(w)
            self._move_rotate_search_step(w, move_mode)
            after_frame = self._copy_current_frame(w)
            similar, mean_diff, changed_ratio = self._frames_are_similar(before_frame, after_frame)
            scene_after = self._get_house_scene(w)
            print(
                f"[SceneRotate] step={step + 1}, mode={move_mode}, "
                f"house_scene={scene_after}, frame_mean={mean_diff:.2f}, "
                f"changed={changed_ratio:.3f}, similar={similar}"
            )

            if scene_after in self.HOUSE_NEAR_ENTRY_SCENES:
                wall_hit_count += 1
                move_mode, wall_hit_count = self._handle_rotate_wall_hit(
                    w,
                    move_mode,
                    wall_hit_count,
                )
                continue

            if scene_after not in {self.HOUSE_INDOOR, None}:
                print(f"[SceneRotate] 推进后 house_scene={scene_after}，停止室内旋转搜房")
                break

            wall_hit_count = 0
            if similar:
                self._recover_rotate_search_stuck(w, move_mode, recover_ms[move_mode])
                recover_ms[move_mode] = min(
                    recover_ms[move_mode] + self.ROTATE_SEARCH_RECOVER_STEP_MS,
                    self.ROTATE_SEARCH_RECOVER_MAX_MS,
                )
            else:
                recover_ms[move_mode] = self.ROTATE_SEARCH_RECOVER_STEP_MS

    def _move_rotate_search_step(self, w: "FrameWorker", move_mode: str):
        x_bias = -self.ROTATE_SEARCH_X_BIAS if move_mode == "left_up" else self.ROTATE_SEARCH_X_BIAS
        label = "左上" if move_mode == "left_up" else "右上"
        print(f"[SceneRotate] 向{label}滑动 {self.ROTATE_SEARCH_MOVE_DURA}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.ROTATE_SEARCH_MOVE_DURA,
            wait=self.ROTATE_SEARCH_MOVE_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _handle_rotate_wall_hit(self, w: "FrameWorker", move_mode: str, wall_hit_count: int):
        label = "墙/门"
        if wall_hit_count >= self.ROTATE_SEARCH_HIT_SWITCH_COUNT:
            if move_mode == "left_up":
                print(f"[SceneRotate] 连续第{wall_hit_count}次撞{label}，逆时针90度并切到右上")
                self._turn(w, -self.ROTATE_SEARCH_TURN_DEGREES)
                return "right_up", 0

            print(f"[SceneRotate] 连续第{wall_hit_count}次撞{label}，顺时针90度并切到左上")
            self._turn(w, self.ROTATE_SEARCH_TURN_DEGREES)
            return "left_up", 0

        if move_mode == "left_up":
            print(f"[SceneRotate] 连续第{wall_hit_count}次撞{label}，顺时针90度后继续左上")
            self._turn(w, self.ROTATE_SEARCH_TURN_DEGREES)
        else:
            print(f"[SceneRotate] 连续第{wall_hit_count}次撞{label}，逆时针90度后继续右上")
            self._turn(w, -self.ROTATE_SEARCH_TURN_DEGREES)

        return move_mode, wall_hit_count

    def _recover_rotate_search_stuck(self, w: "FrameWorker", move_mode: str, dura_ms: int):
        if move_mode == "left_up":
            x_bias = self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "右"
        else:
            x_bias = -self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "左"

        print(f"[SceneRotate] 两帧过于相似，判定卡住，向{label}水平脱困 {dura_ms}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=0,
            dura=dura_ms,
            wait=dura_ms + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _copy_current_frame(self, w: "FrameWorker"):
        frame = getattr(w, "frame", None)
        if frame is None or not hasattr(frame, "shape"):
            return None
        return np.array(frame, copy=True)

    def _frames_are_similar(self, before_frame, after_frame):
        before = self._prepare_frame_for_compare(before_frame)
        after = self._prepare_frame_for_compare(after_frame)
        if before is None or after is None:
            return False, 999.0, 1.0

        diff = cv2.absdiff(before, after)
        mean_diff = float(np.mean(diff))
        changed_ratio = float(np.mean(diff > self.ROTATE_FRAME_CHANGED_PIXEL_THRESHOLD))
        similar = (
            mean_diff <= self.ROTATE_FRAME_MEAN_DIFF_THRESHOLD
            and changed_ratio <= self.ROTATE_FRAME_CHANGED_RATIO_THRESHOLD
        )
        return similar, mean_diff, changed_ratio

    def _prepare_frame_for_compare(self, frame):
        if frame is None or not hasattr(frame, "shape"):
            return None
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            return None

        h, w = gray.shape[:2]
        if h <= 1 or w <= 1:
            return None

        rx1, ry1, rx2, ry2 = self.ROTATE_FRAME_COMPARE_ROI
        x1 = max(0, min(w - 1, int(w * rx1)))
        y1 = max(0, min(h - 1, int(h * ry1)))
        x2 = max(x1 + 1, min(w, int(w * rx2)))
        y2 = max(y1 + 1, min(h, int(h * ry2)))
        crop = gray[y1:y2, x1:x2]
        return cv2.resize(crop, self.ROTATE_FRAME_COMPARE_SIZE, interpolation=cv2.INTER_AREA)

    def _enter_house_by_scene(self, w: "FrameWorker") -> bool:
        self.stop_auto_forward(w)
        if self.active_entry:
            ideal_angle = self.active_entry["direction"]
            print(f"[SceneEntry] 调整至进门方向: {ideal_angle}")
            self.align_direction_blocking(w, w.get_info("direction"), ideal_angle)

        w.refresh_frame()
        if self._is_indoor(w):
            print("[SceneEntry] 已是 indoor，直接开始屋内搜索")
            return True

        button_state = self._door_button_state(w)
        if button_state == "open":
            self._click_open_door(w)
            return self._enter_open_door_by_diagonal_sweep(w)
        if button_state == "close":
            print("[SceneEntry] 检测到关门按钮，门已打开，直接进门")
            return self._enter_open_door_by_diagonal_sweep(w)

        approach_result = self._approach_until_near_entry(w)
        if approach_result == "indoor":
            return True
        if approach_result == "open":
            self._click_open_door(w)
            return self._enter_open_door_by_diagonal_sweep(w)
        if approach_result == "close":
            print("[SceneEntry] 前推过程中检测到关门按钮，门已打开，直接进门")
            return self._enter_open_door_by_diagonal_sweep(w)
        if approach_result != "near":
            return False

        button_state = self._sweep_for_door_button(w)
        if button_state == "indoor":
            return True
        if button_state == "open":
            self._click_open_door(w)
            return self._enter_open_door_by_diagonal_sweep(w)
        if button_state == "close":
            print("[SceneEntry] 左右探测发现关门按钮，门已打开，直接进门")
            return self._enter_open_door_by_diagonal_sweep(w)

        print("[SceneEntry] 左右探测未找到开门/关门按钮")
        return False

    def _approach_until_near_entry(self, w: "FrameWorker") -> str:
        for step in range(self.ENTRY_APPROACH_MAX_STEPS):
            if self._should_abort(w):
                return "abort"

            w.refresh_frame()
            if self._is_indoor(w):
                print("[SceneEntry] 前推前已检测到 indoor")
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state:
                return button_state

            scene = self._get_house_scene(w)
            if scene in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[SceneEntry] 已到门/墙附近 house_scene={scene}")
                return "near"

            print(
                f"[SceneEntry] 正对门前推 {step + 1}/{self.ENTRY_APPROACH_MAX_STEPS}, "
                f"house_scene={scene}"
            )
            if w.get_info("跳跃"):
                self.handle_jump_logic(w)
            else:
                w.tap_single(
                    "摇杆",
                    y_bias=self.ENTRY_APPROACH_FORWARD_Y_BIAS,
                    dura=self.ENTRY_APPROACH_FORWARD_DURA,
                    wait=self.ENTRY_APPROACH_FORWARD_WAIT,
                )
                w.refresh_frame()

        w.refresh_frame()
        if self._is_indoor(w):
            return "indoor"
        if self._get_house_scene(w) in self.HOUSE_NEAR_ENTRY_SCENES:
            return "near"
        return self._door_button_state(w) or "failed"

    def _sweep_for_door_button(self, w: "FrameWorker") -> Optional[str]:
        blocked = {"left": False, "right": False}

        for step in range(self.BUTTON_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return None

            w.refresh_frame()
            if self._is_indoor(w):
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state:
                return button_state

            side = "left" if step % 2 == 0 else "right"
            if blocked[side]:
                print(f"[SceneEntry] {side} 侧已到 outdoor 临界点，跳过本次水平探测")
                continue

            dura = (step + 1) * self.SWEEP_STEP_MS
            x_bias = -self.BUTTON_SWEEP_X_BIAS if side == "left" else self.BUTTON_SWEEP_X_BIAS
            print(f"[SceneEntry] 水平向{self._side_label(side)}探门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=0,
                dura=dura,
                wait=dura + self.BUTTON_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            if self._is_indoor(w):
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state:
                return button_state

            scene = self._get_house_scene(w)
            if scene == self.HOUSE_OUTDOOR:
                blocked[side] = True
                print(f"[SceneEntry] 向{self._side_label(side)}已离开墙/门范围，记录临界点")

            if blocked["left"] and blocked["right"]:
                print("[SceneEntry] 左右两侧都到达 outdoor 临界点，停止探门")
                return None

        return None

    def _enter_open_door_by_diagonal_sweep(self, w: "FrameWorker") -> bool:
        for step in range(self.ENTRY_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_indoor(w):
                print("[SceneEntry] 已检测到 indoor，进门完成")
                return True

            if self._door_button_state(w) == "open":
                print("[SceneEntry] 进门过程中再次看到开门按钮，补点一次开门")
                self._click_open_door(w)

            side = "left" if step % 2 == 0 else "right"
            dura = (step + 1) * self.SWEEP_STEP_MS
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            print(f"[SceneEntry] 向{self._side_label(side)}上进门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.ENTRY_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.ENTRY_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            scene = self._get_house_scene(w)
            if scene == self.HOUSE_INDOOR:
                print("[SceneEntry] 左上/右上推进后 house_scene=indoor，进门完成")
                return True

        print("[SceneEntry] 左上/右上推进到上限，仍未进入 indoor")
        return False

    def _door_button_state(self, w: "FrameWorker") -> Optional[str]:
        if w.get_info("开门"):
            return "open"
        if w.get_info("关门"):
            return "close"
        return None

    def _click_open_door(self, w: "FrameWorker"):
        print("[SceneEntry] 检测到开门按钮，点击开门")
        w.click("开门")
        time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
        w.refresh_frame()

    def _is_indoor(self, w: "FrameWorker") -> bool:
        return self._get_house_scene(w) == self.HOUSE_INDOOR

    @staticmethod
    def _side_label(side: str) -> str:
        return "左" if side == "left" else "右"
