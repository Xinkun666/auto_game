import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_search_manager import (
    HouseSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    calculate_angle,
    calculate_move_count,
    check_location,
    get_adaptive_forward_motion,
    get_adaptive_side_motion,
    get_distance,
    update_adaptive_forward_motion,
    update_adaptive_side_motion,
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
    HOUSE_EXIT_SCENES = {HOUSE_OUTDOOR, HOUSE_ROOFTOP}

    STATUS_SCENE_ENTRY = "SCENE_ENTRY"
    ROTATE_RESULT_FINISHED = "finished"
    ROTATE_RESULT_EXITED = "exited"
    ROTATE_RESULT_FALLBACK_EXIT = "fallback_exit"
    ENTRY_DIRECTION_ALIGN_TOLERANCE = 3
    ENTRY_DIRECTION_ALIGN_MAX_STEPS = 8
    ENTRY_VISIBLE_DOOR_ALIGN_TOLERANCE_PX = 120
    ENTRY_ARRIVAL_DISTANCE = 0.0
    ENTRY_SIDE_ADJUST_MIN_DEGREES = 55
    ENTRY_SIDE_ADJUST_MAX_DEGREES = 125
    ENTRY_SIDE_ADJUST_X_BIAS = 230
    ENTRY_SIDE_ADJUST_BASE_DURA = 100
    ENTRY_SIDE_ADJUST_MAX_DURA = 420
    ENTRY_SIDE_ADJUST_WAIT_PAD = 240
    ENTRY_FORWARD_MAX_STEPS = 4
    ENTRY_FORWARD_STEP_Y_SCALE = 0.62
    ENTRY_FORWARD_STEP_MIN_DURA = 100
    ENTRY_FORWARD_STEP_WAIT_PAD = 220
    ENTRY_FORWARD_FAST_MODE = "fast"
    ENTRY_FORWARD_SLOW_MODE = "slow"

    ENTRY_APPROACH_MAX_STEPS = 4
    ENTRY_APPROACH_FORWARD_Y_BIAS = -280
    ENTRY_APPROACH_FORWARD_DURA = 360
    ENTRY_APPROACH_FORWARD_WAIT = 560

    SWEEP_STEP_MS = 100
    BUTTON_SWEEP_MAX_STEPS = 16
    BUTTON_SWEEP_X_BIAS = 240
    BUTTON_SWEEP_WAIT_PAD = 220

    ENTRY_SWEEP_MAX_STEPS = 14
    ENTRY_SWEEP_X_BIAS = 240
    ENTRY_SWEEP_Y_BIAS = -360
    ENTRY_SWEEP_WAIT_PAD = 260
    ENTRY_OPEN_SWEEP_BASE_DURA = 100
    ENTRY_OPEN_SWEEP_STEP_MS = 50
    ENTRY_OPEN_SWEEP_MAX_DURA = 750
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA = 250
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_MAX_DURA = 650
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_Y_BIAS = 360
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_WAIT = 360
    ENTRY_INDOOR_CONFIRM_FORWARD_Y_BIAS = -420
    ENTRY_INDOOR_CONFIRM_FORWARD_DURA = 650
    ENTRY_INDOOR_CONFIRM_FORWARD_WAIT = 850
    ENTRY_WINDOW_JUMP_SETTLE_SECONDS = 0.25
    OPEN_DOOR_SETTLE_SECONDS = 0.8

    ROTATE_SEARCH_MOVE_DURA = 1000
    ROTATE_SEARCH_MOVE_WAIT_PAD = 260
    ROTATE_SEARCH_X_BIAS = 330
    ROTATE_SEARCH_Y_BIAS = -430
    ROTATE_SEARCH_TURN_DEGREES = 90
    ROTATE_SEARCH_WALL_TURN_SEQUENCE = (90, 45, 22)
    ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS = 6
    ROTATE_SEARCH_HIT_SWITCH_COUNT = 6
    ROTATE_SEARCH_TURN_CORRECT_THRESHOLD = 6.0
    ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS = 2
    ROTATE_SEARCH_TURN_CORRECT_MAX_DEGREES = 45.0
    ROTATE_SEARCH_EXIT_FALLBACK_SWITCHES = 2
    ROTATE_SEARCH_MAX_STEPS = 80
    ROTATE_SEARCH_RECOVER_STEP_MS = 300
    ROTATE_SEARCH_RECOVER_MAX_MS = 1800
    ROTATE_SEARCH_RECOVER_X_BIAS = 330
    ROTATE_FRAME_COMPARE_SIZE = (160, 90)
    ROTATE_FRAME_COMPARE_ROI = (0.18, 0.16, 0.82, 0.78)
    ROTATE_FRAME_MEAN_DIFF_THRESHOLD = 3.5
    ROTATE_FRAME_CHANGED_RATIO_THRESHOLD = 0.02
    ROTATE_FRAME_CHANGED_PIXEL_THRESHOLD = 12

    EXIT_DOOR_CLASS_IDS = {0, 4}
    EXIT_WINDOW_CLASS_IDS = {2}
    EXIT_SEARCH_MAX_STEPS = 36
    EXIT_SEARCH_LEFT_UP_DURA = 1000
    EXIT_SEARCH_TURN_DEGREES = 60
    EXIT_DOOR_SWEEP_MAX_STEPS = 14
    EXIT_WINDOW_ALIGN_MAX_STEPS = 6
    EXIT_WINDOW_ALIGN_TOLERANCE_DEGREES = 3.0
    EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES = 20
    EXIT_WINDOW_FORWARD_MAX_STEPS = 3
    EXIT_WINDOW_FORWARD_Y_BIAS = -360
    EXIT_WINDOW_FORWARD_DURA = 360
    EXIT_WINDOW_FORWARD_WAIT = 520
    EXIT_WINDOW_JUMP_FORWARD_Y_BIAS = -430
    EXIT_WINDOW_JUMP_FORWARD_DURA = 650
    EXIT_WINDOW_JUMP_FORWARD_WAIT = 850

    def searching_logic(self, w: "FrameWorker", current_loc, current_direction):
        if self._should_abort(w):
            return

        house_scene = self._get_house_scene(w)
        if house_scene == self.HOUSE_INDOOR:
            if self._should_start_search_from_indoor():
                if self._confirm_indoor_before_search(w, "导航/进门过程中检测到 indoor"):
                    self._complete_current_house_search(w, "已在屋内，直接启动旋转搜房")
                return

            self._exit_unexpected_indoor(w)
            return

        if self.searching_number == 5:
            self._continue_searching_until_timer(w, "已经搜满5个房间")
            return

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

            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "FAST_NAV"):
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

            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "PRECISE_NAV"):
                return

            self.stop_auto_forward(w)
            if not self._move_precisely_to_entry_point(w, current_loc, target_loc, dist):
                self.align_direction(w, target_loc)
                y_bias, dura, wait = self._get_entry_move_params(dist)
                mode = self._entry_forward_mode(dist)
                print(
                    f"[SceneSearch] 分段推进到进门点: "
                    f"dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
                )
                self._tap_entry_forward_with_learning(w, target_loc, dist, mode, y_bias, dura, wait)
            self.handle_jump_logic(w)
            return

        if self.status == self.STATUS_SCENE_ENTRY:
            entry_result = self._enter_house_by_scene(w)
            if entry_result is None:
                return
            if not entry_result:
                if self._should_abort(w):
                    return
                print("[SceneSearch] house_scene 进门失败，舍弃当前进门点")
                if self.active_entry:
                    self.handle_failed_entry_logic(self.active_entry["direction"])
                else:
                    self.current_house_id = None
                self.status = "IDLE"
                return

            self._complete_current_house_search(w, "house_scene 进门成功")

    def _is_entry_approach_status(self):
        return super()._is_entry_approach_status() or self.status == self.STATUS_SCENE_ENTRY

    def _should_start_search_from_indoor(self) -> bool:
        return (
            self.searching_number < 5
            and self.current_house_id is not None
            and self.current_house_id not in self.completed_houses
        )

    def _confirm_indoor_before_search(self, w: "FrameWorker", reason: str) -> bool:
        if not w.get_info("跳跃"):
            return True

        print(f"[SceneSearch] {reason}，且检测到跳跃按钮，先按翻窗逻辑确认")
        self.handle_jump_logic(w)
        w.refresh_frame()
        if self._is_indoor(w):
            print("[SceneSearch] 跳跃后仍为 indoor，确认已进房")
            return True

        print(f"[SceneSearch] 跳跃后 house_scene={self._get_house_scene(w)}，暂不启动旋转搜房")
        return False

    def _complete_current_house_search(self, w: "FrameWorker", reason: str) -> bool:
        if self._should_abort(w):
            return False

        self.stop_auto_forward(w)
        self.indoor_stuck_frames = 0
        print(f"[SceneSearch] {reason}")

        if not self.start_searching(w):
            return False
        if w.current_stage != "搜房阶段":
            return False

        if self.current_house_id is not None:
            self.completed_houses.add(self.current_house_id)
        self.searching_number += 1
        print(f"[SceneSearch] 房屋 {self.current_house_id} 完成，已搜 {self.searching_number}/5")
        w.refresh_frame()
        exit_direction = w.get_info("direction")
        self.prepare_next_target_logic(exit_direction)
        self.current_house_id = None
        self.status = "IDLE"
        return True

    def _exit_unexpected_indoor(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        print("[SceneSearch] 已搜完或无待搜目标时检测到 indoor，优先执行出房")
        if self._exit_house(w):
            self.indoor_stuck_frames = 0
            self.current_house_id = None
            self.status = "IDLE"
            self._continue_searching_until_timer(w, "意外进房后已出房")

    def _move_precisely_to_entry_point(self, w: "FrameWorker", current_loc, target_loc, dist: float) -> bool:
        current_dir = w.get_info("direction")
        target_angle = calculate_angle(current_loc, target_loc)
        turn_dir, _, diff = calculate_move_count(current_dir, target_angle)
        if diff is None or turn_dir is None:
            return False

        if self.ENTRY_SIDE_ADJUST_MIN_DEGREES <= diff <= self.ENTRY_SIDE_ADJUST_MAX_DEGREES:
            side = "right" if turn_dir == "right" else "left"
            x_bias = self.ENTRY_SIDE_ADJUST_X_BIAS if side == "right" else -self.ENTRY_SIDE_ADJUST_X_BIAS
            dura = self._entry_micro_dura(
                dist,
                self.ENTRY_SIDE_ADJUST_BASE_DURA,
                self.ENTRY_SIDE_ADJUST_MAX_DURA,
            )
            wait = dura + self.ENTRY_SIDE_ADJUST_WAIT_PAD
            print(
                f"[SceneSearch] 进门点在{self._side_label(side)}侧，水平微调到点: "
                f"dist={dist:.2f}, diff={diff:.1f}, dura={dura}"
            )
            return self._tap_entry_side_with_learning(w, target_loc, dist, side, x_bias, dura, wait)

        self.align_direction(w, target_loc, threshold=5, max_steps=1)
        mode = self._entry_forward_mode(dist)
        y_bias, dura, wait = self._get_entry_move_params(dist)
        print(
            f"[SceneSearch] {self._entry_forward_mode_label(mode)}到进门点: "
            f"dist={dist:.2f}, target_angle={target_angle}, diff={diff:.1f}, "
            f"y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        return self._tap_entry_forward_with_learning(w, target_loc, dist, mode, y_bias, dura, wait)

    @staticmethod
    def _entry_micro_dura(dist: float, base_dura: int, max_dura: int) -> int:
        try:
            dist_val = max(0.0, float(dist))
        except (TypeError, ValueError):
            dist_val = 0.0
        return int(max(base_dura, min(max_dura, base_dura + dist_val * 18)))

    def _entry_forward_mode(self, dist: float) -> str:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            dist_val = 0.0
        if dist_val > self.ENTRY_COARSE_MOVE_DISTANCE:
            return self.ENTRY_FORWARD_FAST_MODE
        return self.ENTRY_FORWARD_SLOW_MODE

    @staticmethod
    def _entry_forward_mode_label(mode: str) -> str:
        return "快推" if mode == "fast" else "慢推"

    def _tap_entry_forward_with_learning(
        self,
        w: "FrameWorker",
        target_loc,
        desired_dist: float,
        mode: str,
        fallback_y_bias: int,
        fallback_dura: int,
        fallback_wait: int,
    ) -> bool:
        before_dist = self._get_current_entry_distance(w, target_loc)
        if before_dist is None:
            before_dist = desired_dist

        previous_dist = before_dist
        for step in range(self.ENTRY_FORWARD_MAX_STEPS):
            if self._should_abort(w):
                return False

            current_dist = self._get_current_entry_distance(w, target_loc)
            if current_dist is None:
                current_dist = previous_dist
            if current_dist <= self.ENTRY_ARRIVAL_DISTANCE:
                return True

            remaining_steps = max(1, self.ENTRY_FORWARD_MAX_STEPS - step)
            desired_step_dist = max(0.2, min(float(desired_dist), float(current_dist)) / remaining_steps)
            step_y_bias, step_dura, step_wait = self._entry_forward_step_fallback(
                fallback_y_bias,
                fallback_dura,
                fallback_wait,
            )
            y_bias, dura, wait, distance_key = get_adaptive_forward_motion(
                mode,
                desired_step_dist,
                step_y_bias,
                step_dura,
                step_wait,
            )

            print(
                f"[SceneSearch] 执行{self._entry_forward_mode_label(mode)}小步 "
                f"{step + 1}/{self.ENTRY_FORWARD_MAX_STEPS}: bin={distance_key}, "
                f"before={current_dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
            )
            w.tap_single("摇杆", y_bias=y_bias, dura=dura, wait=wait)
            w.refresh_frame()

            after_dist = self._get_current_entry_distance(w, target_loc)
            update_adaptive_forward_motion(
                mode,
                desired_step_dist,
                current_dist,
                after_dist,
                y_bias,
                dura,
                wait,
            )
            if after_dist is None:
                return True

            moved = current_dist - after_dist
            print(
                f"[SceneSearch] 推进反馈: mode={mode}, bin={distance_key}, "
                f"after={after_dist:.2f}, moved={moved:.2f}"
            )
            if after_dist <= self.ENTRY_ARRIVAL_DISTANCE or moved <= 0:
                break
            previous_dist = after_dist

        return True

    def _tap_entry_side_with_learning(
        self,
        w: "FrameWorker",
        target_loc,
        desired_dist: float,
        side: str,
        fallback_x_bias: int,
        fallback_dura: int,
        fallback_wait: int,
    ) -> bool:
        x_bias, dura, wait, distance_key = get_adaptive_side_motion(
            side,
            desired_dist,
            fallback_x_bias,
            fallback_dura,
            fallback_wait,
        )
        before_dist = self._get_current_entry_distance(w, target_loc)
        if before_dist is None:
            before_dist = desired_dist

        print(
            f"[SceneSearch] 执行{self._side_label(side)}滑微调: "
            f"bin={distance_key}, before={before_dist:.2f}, "
            f"x_bias={x_bias}, dura={dura}, wait={wait}"
        )
        w.tap_single("摇杆", x_bias=x_bias, y_bias=0, dura=dura, wait=wait)
        w.refresh_frame()

        after_dist = self._get_current_entry_distance(w, target_loc)
        update_adaptive_side_motion(
            side,
            desired_dist,
            before_dist,
            after_dist,
            x_bias,
            dura,
            wait,
        )
        if after_dist is not None:
            moved = before_dist - after_dist
            print(
                f"[SceneSearch] 侧滑反馈: side={side}, bin={distance_key}, "
                f"after={after_dist:.2f}, moved={moved:.2f}"
            )
        return True

    def _entry_forward_step_fallback(self, y_bias: int, dura: int, wait: int):
        step_y = int(round(float(y_bias) * self.ENTRY_FORWARD_STEP_Y_SCALE))
        step_dura = max(
            self.ENTRY_FORWARD_STEP_MIN_DURA,
            int(round(float(dura) / self.ENTRY_FORWARD_MAX_STEPS)),
        )
        step_wait = max(
            step_dura + self.ENTRY_FORWARD_STEP_WAIT_PAD,
            int(round(float(wait) / self.ENTRY_FORWARD_MAX_STEPS)),
        )
        if y_bias < 0:
            step_y = min(-80, step_y)
        else:
            step_y = max(80, step_y)
        return step_y, step_dura, step_wait

    def _get_current_entry_distance(self, w: "FrameWorker", target_loc) -> Optional[float]:
        raw = w.get_info("location")
        if not raw:
            return None

        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            current_loc = check_location(raw[0])
        else:
            current_loc = check_location(raw)
        if current_loc is None:
            return None

        dist = get_distance(current_loc, target_loc)
        if dist < 0:
            return None
        return dist

    def start_searching(self, w: "FrameWorker"):
        if self._should_abort(w):
            return False

        self._clear_house_search_timer()
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        print("[SceneRotate] 进入房屋，启动 house_scene 旋转搜房")
        rotate_result = self._rotate_search_inside_house(w)

        if self._should_abort(w):
            return False

        w.refresh_frame()
        if rotate_result == self.ROTATE_RESULT_EXITED or self._is_out_of_house(w):
            print("[SceneRotate] 旋转搜房过程中已出房，房屋搜索完成")
            return True

        if rotate_result == self.ROTATE_RESULT_FALLBACK_EXIT:
            print("[SceneRotate] 两轮撞墙循环仍未自然出房，开始执行出房策略")
        else:
            print("[SceneRotate] 旋转搜房结束，开始出房")

        if self._exit_house(w):
            print("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True
        if self._should_abort(w):
            return False

        w.refresh_frame()
        if self._is_out_of_house(w):
            print("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True

        print(f"[SceneRotate] 出房策略后仍未确认出房 house_scene={self._get_house_scene(w)}")
        return False

    def _house_search_timed_out(self):
        return False

    def _rotate_search_inside_house(self, w: "FrameWorker"):
        move_mode = "left_up"
        wall_hit_count = 0
        wall_switch_cycles = 0
        recover_ms = {
            "left_up": self.ROTATE_SEARCH_RECOVER_STEP_MS,
            "right_up": self.ROTATE_SEARCH_RECOVER_STEP_MS,
        }

        for step in range(self.ROTATE_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                break

            w.refresh_frame()
            scene_before = self._get_house_scene(w)
            if scene_before is None:
                print("[SceneRotate] 当前 house_scene 暂未识别，继续按室内旋转搜房尝试")
            elif scene_before in self.HOUSE_EXIT_SCENES:
                print(f"[SceneRotate] 滑动前已判定出房 house_scene={scene_before}")
                return self.ROTATE_RESULT_EXITED
            elif scene_before not in {
                self.HOUSE_INDOOR,
                self.HOUSE_NEAR_DOOR,
                self.HOUSE_NEAR_WALL,
            }:
                print(f"[SceneRotate] 当前 house_scene={scene_before}，停止室内旋转搜房")
                return self.ROTATE_RESULT_FINISHED

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

            if scene_after in self.HOUSE_EXIT_SCENES:
                print(f"[SceneRotate] {self._move_mode_label(move_mode)}滑动后判定出房 house_scene={scene_after}")
                return self.ROTATE_RESULT_EXITED

            if scene_after in self.HOUSE_NEAR_ENTRY_SCENES:
                wall_hit_count += 1
                move_mode, wall_hit_count, switched = self._handle_rotate_wall_hit(
                    w,
                    move_mode,
                    wall_hit_count,
                )
                if switched:
                    wall_switch_cycles += 1
                    print(
                        f"[SceneRotate] 撞墙循环切换 {wall_switch_cycles}/"
                        f"{self.ROTATE_SEARCH_EXIT_FALLBACK_SWITCHES}"
                    )
                    if wall_switch_cycles >= self.ROTATE_SEARCH_EXIT_FALLBACK_SWITCHES:
                        return self.ROTATE_RESULT_FALLBACK_EXIT
                continue

            if scene_after not in {self.HOUSE_INDOOR, None}:
                print(f"[SceneRotate] 推进后 house_scene={scene_after}，停止室内旋转搜房")
                return self.ROTATE_RESULT_FINISHED

            wall_hit_count = 0
            if similar:
                self._recover_rotate_search_stuck(w, move_mode, recover_ms[move_mode])
                recover_ms[move_mode] = min(
                    recover_ms[move_mode] + self.ROTATE_SEARCH_RECOVER_STEP_MS,
                    self.ROTATE_SEARCH_RECOVER_MAX_MS,
                )
            else:
                recover_ms[move_mode] = self.ROTATE_SEARCH_RECOVER_STEP_MS

        return self.ROTATE_RESULT_FINISHED

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
        current_mode = move_mode
        if wall_hit_count >= self.ROTATE_SEARCH_HIT_SWITCH_COUNT:
            if current_mode == "left_up":
                print(f"[SceneRotate] 左上撞{label}已达{wall_hit_count}次，立即切到右上，并改为向左补转")
                self._turn_until_not_near_entry(w, -1)
                return "right_up", 0, True

            print(f"[SceneRotate] 右上撞{label}已达{wall_hit_count}次，立即切到左上，并改为向右补转")
            self._turn_until_not_near_entry(w, 1)
            return "left_up", 0, True

        turn_sign = 1 if current_mode == "left_up" else -1
        turn_label = "向右" if turn_sign > 0 else "向左"
        print(f"[SceneRotate] 撞{label}后{turn_label}补转，直到不再贴墙/门")
        self._turn_until_not_near_entry(w, turn_sign)
        return move_mode, wall_hit_count, False

    def _turn_until_not_near_entry(self, w: "FrameWorker", turn_sign: int) -> bool:
        for attempt in range(self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            base_index = min(attempt, len(self.ROTATE_SEARCH_WALL_TURN_SEQUENCE) - 1)
            angle = self.ROTATE_SEARCH_WALL_TURN_SEQUENCE[base_index]
            signed_angle = turn_sign * angle
            print(
                f"[SceneRotate] 撞墙补转 {attempt + 1}/"
                f"{self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS}: {signed_angle}°"
            )
            self._turn_with_direction_correction(w, signed_angle)
            w.refresh_frame()
            scene = self._get_house_scene(w)
            if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[SceneRotate] 补转后 house_scene={scene}，继续当前滑动方向")
                return True

            print(f"[SceneRotate] 补转后仍贴墙/门 house_scene={scene}，继续缩小角度补转")

        print("[SceneRotate] 多次补转后仍贴墙/门，交给下一轮移动继续尝试")
        return False

    def _turn_with_direction_correction(self, w: "FrameWorker", signed_angle: float):
        before_dir = self._direction_as_float(w.get_info("direction"))
        self._turn(w, signed_angle)
        w.refresh_frame()
        if before_dir is None:
            print("[SceneRotate] 当前 direction 无效，跳过本次补角到位校验")
            return

        target_dir = (before_dir + float(signed_angle)) % 360.0
        for step in range(self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS):
            current_dir = self._direction_as_float(w.get_info("direction"))
            turn_dir, _, diff = calculate_move_count(current_dir, target_dir)
            if diff is None or turn_dir is None:
                print("[SceneRotate] 补角后 direction 无效，无法继续校验角度")
                return
            if diff <= self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
                return

            correction = min(float(diff), self.ROTATE_SEARCH_TURN_CORRECT_MAX_DEGREES)
            signed_correction = correction if turn_dir == "right" else -correction
            print(
                f"[SceneRotate] 补角未到位，二次校正 {step + 1}/"
                f"{self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS}: "
                f"current={current_dir:.1f}, target={target_dir:.1f}, "
                f"remaining={diff:.1f}, turn={signed_correction:.1f}°"
            )
            self._turn(w, signed_correction)
            w.refresh_frame()

        current_dir = self._direction_as_float(w.get_info("direction"))
        _, _, remaining = calculate_move_count(current_dir, target_dir)
        if remaining is not None and remaining > self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
            print(
                f"[SceneRotate] 二次校正后仍有角度偏差: "
                f"current={current_dir:.1f}, target={target_dir:.1f}, remaining={remaining:.1f}"
            )

    @staticmethod
    def _direction_as_float(direction) -> Optional[float]:
        try:
            return float(direction) % 360.0
        except (TypeError, ValueError):
            return None

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

    def _exit_house(self, w: "FrameWorker") -> bool:
        return self._exit_house_by_scene_strategy(w)

    def _exit_house_by_scene_strategy(self, w: "FrameWorker") -> bool:
        print("[SceneExit] 启动 house_scene 多路径出房策略")

        for step in range(self.EXIT_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 出房策略开始前已判定在屋外")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

            button_state = self._door_button_state(w)
            if button_state and self._exit_via_door_button(w, button_state):
                return True

            print(f"[SceneExit] 左上绕圈找出口 {step + 1}/{self.EXIT_SEARCH_MAX_STEPS}")
            self._move_exit_search_left_up(w)
            if self._is_out_of_house(w):
                print("[SceneExit] 左上滑动时意外出房，出房成功")
                return True

            button_state = self._door_button_state(w)
            if button_state and self._exit_via_door_button(w, button_state):
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

            print(f"[SceneExit] 左上后向右调整视角 {self.EXIT_SEARCH_TURN_DEGREES}° 继续绕圈")
            self._turn(w, self.EXIT_SEARCH_TURN_DEGREES)
            w.refresh_frame()

        print("[SceneExit] 多路径出房策略达到步数上限，仍未确认出房")
        return self._is_out_of_house(w)

    def _move_exit_search_left_up(self, w: "FrameWorker"):
        w.tap_single(
            "摇杆",
            x_bias=-self.ROTATE_SEARCH_X_BIAS,
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.EXIT_SEARCH_LEFT_UP_DURA,
            wait=self.EXIT_SEARCH_LEFT_UP_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _exit_via_door_button(self, w: "FrameWorker", button_state: str) -> bool:
        if button_state == "open":
            print("[SceneExit] 发现开门按钮，点击开门后尝试出门")
            w.click("开门")
            time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
            w.refresh_frame()
        else:
            print("[SceneExit] 发现关门按钮，门已打开，直接尝试出门")

        return self._exit_open_door_by_diagonal_sweep(w)

    def _exit_open_door_by_diagonal_sweep(self, w: "FrameWorker") -> bool:
        for step in range(self.EXIT_DOOR_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 门口推进前已在屋外")
                return True

            if self._door_button_state(w) == "open":
                print("[SceneExit] 门口推进前再次看到开门按钮，补点一次开门")
                w.click("开门")
                time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
                w.refresh_frame()

            side = "left" if step % 2 == 0 else "right"
            dura = min(
                self.ENTRY_OPEN_SWEEP_BASE_DURA + step * self.ENTRY_OPEN_SWEEP_STEP_MS,
                self.ENTRY_OPEN_SWEEP_MAX_DURA,
            )
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            print(f"[SceneExit] 门已打开，向{self._side_label(side)}上小步尝试出门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.ENTRY_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.ENTRY_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            if self._is_out_of_house(w):
                print("[SceneExit] 左上/右上门口推进后出房成功")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

        print("[SceneExit] 门口左上/右上推进到上限，未确认出房")
        return False

    def _exit_via_window_by_scene(self, w: "FrameWorker", window) -> bool:
        rel_angle = self._target_relative_angle(window)
        print(f"[SceneExit] 发现窗户，准备对齐 rel_angle={rel_angle}")
        align_state = self._align_to_exit_window(w, window)
        if align_state == "lost":
            return self._push_until_jump_and_exit_window(w, "窗户对齐过程中目标丢失")
        if align_state == "aligned":
            return self._push_until_jump_and_exit_window(w, "窗户已对齐")
        return False

    def _align_to_exit_window(self, w: "FrameWorker", window) -> str:
        target = window
        for step in range(self.EXIT_WINDOW_ALIGN_MAX_STEPS):
            if self._should_abort(w):
                return "abort"

            rel_angle = self._target_relative_angle(target)
            if rel_angle is None:
                return "lost"
            if abs(rel_angle) <= self.EXIT_WINDOW_ALIGN_TOLERANCE_DEGREES:
                print(f"[SceneExit] 窗户已对齐 rel_angle={rel_angle:.1f}")
                return "aligned"

            turn_angle = max(
                -self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES,
                min(self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES, rel_angle),
            )
            side = "右" if turn_angle > 0 else "左"
            print(
                f"[SceneExit] 窗户在{side}侧，对齐 {step + 1}/"
                f"{self.EXIT_WINDOW_ALIGN_MAX_STEPS}: turn={turn_angle:.1f}"
            )
            self._turn(w, turn_angle)
            w.refresh_frame()

            refreshed = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if not refreshed:
                print("[SceneExit] 对齐窗户时目标丢失，改为前推找跳跃按钮")
                return "lost"
            target = refreshed

        print("[SceneExit] 窗户对齐达到步数上限，按已接近窗户处理")
        return "aligned"

    def _push_until_jump_and_exit_window(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[SceneExit] {reason}，最多前推 {self.EXIT_WINDOW_FORWARD_MAX_STEPS} 次找跳跃")
        for step in range(self.EXIT_WINDOW_FORWARD_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 靠窗前推前已出房")
                return True

            if w.get_info("跳跃"):
                if self._jump_forward_exit_window(w, step + 1):
                    return True
                continue

            print(f"[SceneExit] 靠窗前推找跳跃 {step + 1}/{self.EXIT_WINDOW_FORWARD_MAX_STEPS}")
            w.tap_single(
                "摇杆",
                y_bias=self.EXIT_WINDOW_FORWARD_Y_BIAS,
                dura=self.EXIT_WINDOW_FORWARD_DURA,
                wait=self.EXIT_WINDOW_FORWARD_WAIT,
            )
            w.refresh_frame()

            if self._is_out_of_house(w):
                print("[SceneExit] 靠窗前推时意外出房")
                return True

            if w.get_info("跳跃") and self._jump_forward_exit_window(w, step + 1):
                return True

        print("[SceneExit] 靠窗前推 3 次仍未出现可用跳跃，放弃该窗户")
        return False

    def _jump_forward_exit_window(self, w: "FrameWorker", step: int) -> bool:
        print(f"[SceneExit] 检测到跳跃按钮，尝试翻窗出房 step={step}")
        w.click("跳跃")
        time.sleep(self.ENTRY_WINDOW_JUMP_SETTLE_SECONDS)
        w.tap_single(
            "摇杆",
            y_bias=self.EXIT_WINDOW_JUMP_FORWARD_Y_BIAS,
            dura=self.EXIT_WINDOW_JUMP_FORWARD_DURA,
            wait=self.EXIT_WINDOW_JUMP_FORWARD_WAIT,
        )
        w.refresh_frame()
        if self._is_out_of_house(w):
            print("[SceneExit] 翻窗后出房成功")
            return True
        return False

    def _find_largest_forward_target(self, w: "FrameWorker", class_ids: set):
        candidates = []
        for obj in self._get_forward_scene(w):
            try:
                if len(obj) < 6 or int(obj[5]) not in class_ids:
                    continue
                area = max(0.0, float(obj[2]) - float(obj[0])) * max(0.0, float(obj[3]) - float(obj[1]))
            except (TypeError, ValueError):
                continue
            candidates.append((area, obj))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _target_relative_angle(self, target) -> Optional[float]:
        try:
            center_x = (float(target[0]) + float(target[2])) / 2.0
        except (TypeError, ValueError, IndexError):
            return None
        return self.pixel_to_angle(center_x)

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

    def _enter_house_by_scene(self, w: "FrameWorker") -> Optional[bool]:
        self.stop_auto_forward(w)
        if self.active_entry:
            ideal_angle = self.active_entry["direction"]
            print(f"[SceneEntry] 调整至进门方向: {ideal_angle}")
            aligned = self.align_direction_blocking(
                w,
                w.get_info("direction"),
                ideal_angle,
                threshold=self.ENTRY_DIRECTION_ALIGN_TOLERANCE,
                max_steps=self.ENTRY_DIRECTION_ALIGN_MAX_STEPS,
            )
            if not aligned:
                print("[SceneEntry] 进门方向尚未对准，继续对准后再探门")
                return None

        w.refresh_frame()
        if self._is_indoor(w):
            print("[SceneEntry] 已是 indoor，直接开始屋内搜索")
            return True

        self._align_visible_entry_door_before_push(w)
        if self._is_indoor(w):
            print("[SceneEntry] 对齐视觉门后已是 indoor，直接开始屋内搜索")
            return True

        button_state = self._door_button_state(w)
        if button_state:
            return self._handle_entry_door_button_then_forward(w, button_state, "进门前")

        approach_result = self._approach_until_near_entry(w, force_first_forward=True)
        if approach_result == "indoor":
            print("[SceneEntry] 前推/跳跃后已确认 indoor，直接开始屋内搜索")
            return True
        if approach_result in {"open", "close"}:
            return self._handle_entry_door_button_then_forward(w, approach_result, "前推过程中")
        if approach_result != "near":
            return False

        button_state = self._sweep_for_door_button(w)
        if button_state == "indoor":
            return self._confirm_indoor_by_forward_push(w, "左右探门时检测到 indoor")
        if button_state in {"open", "close"}:
            return self._handle_entry_door_button_then_forward(w, button_state, "左右探门时")

        print("[SceneEntry] 左右探测未找到开门/关门按钮")
        return False

    def _handle_entry_door_button_then_forward(
        self,
        w: "FrameWorker",
        button_state: str,
        reason: str,
    ) -> bool:
        self.stop_auto_forward(w)
        if button_state == "open":
            print(f"[SceneEntry] {reason}发现开门按钮，点击开门后停止贴门前推")
            self._click_open_door(w)
        else:
            print(f"[SceneEntry] {reason}发现关门按钮，门已打开，不点击关门，停止贴门前推")

        self._realign_to_entry_direction(w, "门按钮出现后")

        if self._confirm_indoor_by_forward_push(w, "门按钮处理后重新对准进门方向"):
            return True

        print("[SceneEntry] 正前推后仍未 indoor，开始左前/右前小步进门")
        return self._enter_open_door_by_diagonal_sweep(w)

    def _realign_to_entry_direction(self, w: "FrameWorker", reason: str) -> bool:
        if not self.active_entry:
            return False

        ideal_angle = self.active_entry["direction"]
        print(f"[SceneEntry] {reason}，重新修正至进门方向: {ideal_angle}")
        aligned = self.align_direction_blocking(
            w,
            w.get_info("direction"),
            ideal_angle,
            threshold=self.ENTRY_DIRECTION_ALIGN_TOLERANCE,
            max_steps=self.ENTRY_DIRECTION_ALIGN_MAX_STEPS,
        )
        if not aligned:
            print("[SceneEntry] 门按钮后进门方向未完全对准，仍按当前方向尝试一次正前推")
        return aligned

    def _align_visible_entry_door_before_push(self, w: "FrameWorker") -> bool:
        door = self.find_largest_door(w)
        if door is None:
            print("[SceneEntry] 进门方向上未识别到门，直接按进门方向前推")
            return False

        print("[SceneEntry] 进门方向上识别到门，先粗略对齐门再前推")
        aligned = self._align_to_door_detection(
            w,
            door,
            tolerance_px=self.ENTRY_VISIBLE_DOOR_ALIGN_TOLERANCE_PX,
        )
        w.refresh_frame()
        if not aligned:
            print("[SceneEntry] 视觉门粗对齐未完全成功，继续按当前方向前推试探")
        return aligned

    def _approach_until_near_entry(self, w: "FrameWorker", force_first_forward: bool = False) -> str:
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
            if scene in self.HOUSE_NEAR_ENTRY_SCENES and not (force_first_forward and step == 0):
                print(f"[SceneEntry] 已到门/墙附近 house_scene={scene}")
                return "near"

            print(
                f"[SceneEntry] 正对门前推 {step + 1}/{self.ENTRY_APPROACH_MAX_STEPS}, "
                f"house_scene={scene}"
            )
            if w.get_info("跳跃"):
                self.handle_jump_logic(w)
                w.refresh_frame()
                if self._is_indoor(w):
                    print("[SceneEntry] 前推过程中跳跃后变为 indoor，确认翻窗进房")
                    return "indoor"
            else:
                w.tap_single(
                    "摇杆",
                    y_bias=self.ENTRY_APPROACH_FORWARD_Y_BIAS,
                    dura=self.ENTRY_APPROACH_FORWARD_DURA,
                    wait=self.ENTRY_APPROACH_FORWARD_WAIT,
                )
                w.refresh_frame()
                if self._is_indoor(w):
                    print("[SceneEntry] 前推后变为 indoor，确认已进房")
                    return "indoor"

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
                if self._confirm_indoor_by_forward_push(w, "小步推进前检测到 indoor"):
                    return True
                continue

            if self._door_button_state(w) == "open":
                print("[SceneEntry] 小步推进前再次看到开门按钮，补点一次开门")
                self._click_open_door(w)

            side = "left" if step % 2 == 0 else "right"
            dura = min(
                self.ENTRY_OPEN_SWEEP_BASE_DURA + step * self.ENTRY_OPEN_SWEEP_STEP_MS,
                self.ENTRY_OPEN_SWEEP_MAX_DURA,
            )
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            print(f"[SceneEntry] 门已打开，向{self._side_label(side)}上小步推进 {dura}ms")
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
                if self._confirm_indoor_by_forward_push(w, "左上/右上小步推进后检测到 indoor"):
                    return True
                continue

            if scene in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[SceneEntry] 小步推进后仍贴墙/门 house_scene={scene}，继续换边")
            elif scene == self.HOUSE_OUTDOOR:
                print("[SceneEntry] 小步推进后变为 outdoor，先反向斜后回拉，再换边进门")
                if self._backoff_from_outdoor_side(w, side, dura):
                    return True

        print("[SceneEntry] 左上/右上小步推进到上限，仍未进入 indoor")
        return False

    def _backoff_from_outdoor_side(self, w: "FrameWorker", side: str, dura_ms: Optional[int] = None) -> bool:
        opposite_x = self.ENTRY_SWEEP_X_BIAS if side == "left" else -self.ENTRY_SWEEP_X_BIAS
        try:
            backoff_dura = int(dura_ms or self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA)
        except (TypeError, ValueError):
            backoff_dura = self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA
        backoff_dura = min(
            max(self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA, backoff_dura),
            self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_MAX_DURA,
        )
        opposite_side = "right" if side == "left" else "left"
        print(
            f"[SceneEntry] outdoor 临界点回拉，向{self._side_label(opposite_side)}后 "
            f"{backoff_dura}ms 回到墙/门附近"
        )
        w.tap_single(
            "摇杆",
            x_bias=opposite_x,
            y_bias=self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_Y_BIAS,
            dura=backoff_dura,
            wait=backoff_dura + self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_WAIT,
        )
        w.refresh_frame()

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_INDOOR:
            return self._confirm_indoor_by_forward_push(w, "outdoor 回退后检测到 indoor")
        print(f"[SceneEntry] outdoor 回拉后 house_scene={scene}，下一步换边继续尝试")
        return False

    def _confirm_indoor_by_forward_push(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[SceneEntry] {reason}，前推确认入房")
        if w.get_info("跳跃"):
            print("[SceneEntry] indoor 信号伴随跳跃按钮，按翻窗逻辑点击跳跃后前推")
            w.click("跳跃")
            time.sleep(self.ENTRY_WINDOW_JUMP_SETTLE_SECONDS)

        w.tap_single(
            "摇杆",
            y_bias=self.ENTRY_INDOOR_CONFIRM_FORWARD_Y_BIAS,
            dura=self.ENTRY_INDOOR_CONFIRM_FORWARD_DURA,
            wait=self.ENTRY_INDOOR_CONFIRM_FORWARD_WAIT,
        )
        w.refresh_frame()

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_INDOOR:
            print("[SceneEntry] 前推后 house_scene=indoor，确认已进房")
            return True

        print(f"[SceneEntry] 前推后 house_scene={scene}，indoor 信号未确认")
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

    def _is_out_of_house(self, w: "FrameWorker") -> bool:
        return self._get_house_scene(w) in self.HOUSE_EXIT_SCENES

    @staticmethod
    def _move_mode_label(move_mode: str) -> str:
        return "左上" if move_mode == "left_up" else "右上"

    @staticmethod
    def _side_label(side: str) -> str:
        return "左" if side == "left" else "右"
