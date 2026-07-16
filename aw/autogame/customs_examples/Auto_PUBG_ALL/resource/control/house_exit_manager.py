import random
import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    execute_view_turn,
)
from aw.autogame.tools.Utils import get_resolution, get_wh

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class HouseExitManager:
    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2
    HOUSE_NEAR_DOOR = 3
    HOUSE_NEAR_WALL = 4
    HOUSE_EXIT_SCENES = {HOUSE_OUTDOOR, HOUSE_ROOFTOP}
    HOUSE_ACTIVE_EXIT_SCENES = {HOUSE_INDOOR, HOUSE_NEAR_DOOR, HOUSE_NEAR_WALL}

    DOOR_CLASS_IDS = {0, 4}
    WINDOW_CLASS_IDS = {2}

    SCAN_OFFSETS = [-45, 45]
    DOOR_LOST_FORWARD_DURA = 260
    DOOR_LOST_FORWARD_WAIT = 450
    DOOR_RECOVER_BACK_DURA = 380
    DOOR_RECOVER_BACK_WAIT = 650
    DOOR_DIAGONAL_SWEEP_STEPS = 8
    DOOR_DIAGONAL_SWEEP_X_BIAS = 240
    DOOR_DIAGONAL_SWEEP_Y_BIAS = -320
    DOOR_DIAGONAL_SWEEP_BASE_DURA = 160
    DOOR_DIAGONAL_SWEEP_STEP_DURA = 60
    DOOR_DIAGONAL_SWEEP_MAX_DURA = 580
    DOOR_DIAGONAL_SWEEP_WAIT_PAD = 180
    WINDOW_APPROACH_STEPS = 3
    WINDOW_APPROACH_DURA = 260
    WINDOW_APPROACH_WAIT = 360
    WINDOW_JUMP_FORWARD_DURA = 520
    WINDOW_JUMP_FORWARD_WAIT = 650
    WINDOW_LATERAL_RECOVER_STEPS = 6
    WINDOW_LATERAL_RECOVER_X_BIAS = 150
    WINDOW_LATERAL_RECOVER_DURA = 180
    WINDOW_LATERAL_RECOVER_WAIT = 160
    EXIT_CONFIRM_FORWARD_DURA = 220
    EXIT_CONFIRM_FORWARD_WAIT = 350
    WALL_BACKOFF_DURA = 420
    WALL_BACKOFF_WAIT = 560
    WALL_TURN_BACK_DEGREES = 180
    NO_EXIT_ESCAPE_DURA = 5000
    NO_EXIT_ESCAPE_WAIT = 5200
    NO_EXIT_ESCAPE_X_BIAS = 360
    NO_EXIT_ESCAPE_Y_BIAS = -460
    DEAD_END_ESCAPE_STEPS = 10
    DEAD_END_ESCAPE_TURN_X_BIAS = 500
    DEAD_END_ESCAPE_TURN_DURA = 280
    DEAD_END_ESCAPE_TURN_WAIT = 120
    DEAD_END_ESCAPE_SIDE_X_BIAS = 260
    DEAD_END_ESCAPE_SIDE_DURA = 260
    DEAD_END_ESCAPE_SIDE_WAIT = 120
    TARGET_ALIGN_MAX_STEPS = 3

    def __init__(self):
        self.screen_w, self.screen_h = get_resolution()
        self.scan_anchor_direction: Optional[float] = None
        self.scan_index = 0
        self.trusted_exit_signal = False
        self.auto_forward_enabled = False

    def reset(self):
        self.scan_anchor_direction = None
        self.scan_index = 0
        self.trusted_exit_signal = False
        self.auto_forward_enabled = False


    def process(self, w: "FrameWorker") -> bool:
        if self._handle_terminal_state(w, "检测到死亡或结算界面，结束出房兜底流程"):
            return True

        house_scene = self._get_house_scene(w)
        w.frame_log(
            f"[HouseExit] 本帧状态: scene={house_scene}, scan={self.scan_index}/{len(self.SCAN_OFFSETS)}, "
            f"auto_forward={self.auto_forward_enabled}"
        )
        if house_scene not in self.HOUSE_ACTIVE_EXIT_SCENES:
            if house_scene in self.HOUSE_EXIT_SCENES:
                w.frame_log(f"[HouseExit] scene={house_scene}，检测到已在室外/楼顶，开始二次确认")
                return self._verify_exit_success(w)
            w.frame_log(f"[HouseExit] scene={house_scene} 不属于出房处理场景，本帧不接管")
            return False

        self.trusted_exit_signal = False
        self._ensure_scan_anchor(w)
        visible_result = self._try_visible_exit_target(w)
        if visible_result is not None:
            return visible_result

        w.frame_log(f"[HouseExit] scene={house_scene}，当前视野未发现门窗，转入视角扫描")
        return self._search_for_exit(w)

    def _try_visible_exit_target(self, w: "FrameWorker") -> Optional[bool]:
        detections = self._get_forward_scene(w)

        door = self._find_largest_target(detections, self.DOOR_CLASS_IDS)
        if door:
            w.frame_log(f"[HouseExit] 发现门: {self._describe_target(door)}，停止自动前进并对准出门")
            self._stop_auto_forward(w)
            return self._exit_via_door(w, door)

        window = self._find_largest_target(detections, self.WINDOW_CLASS_IDS)
        if window:
            w.frame_log(f"[HouseExit] 发现窗: {self._describe_target(window)}，停止自动前进并对准翻窗")
            self._stop_auto_forward(w)
            return self._exit_via_window(w, window)

        return None

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

    def _is_terminal_state(self, w: "FrameWorker") -> bool:
        return bool(
            w.get_info("变身")
            or w.get_info("红色血条")
            or w.get_info("个人排名")
            or w.get_info("队伍排名")
        )

    def _handle_terminal_state(self, w: "FrameWorker", reason: str) -> bool:
        if not self._is_terminal_state(w):
            return False
        w.frame_log(f"[HouseExit] {reason}")
        self._stop_auto_forward(w)
        self.reset()
        try:
            w.change_stage("结束阶段")
        except Exception:
            pass
        return True

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

    @staticmethod
    def _describe_target(target: Sequence[float]) -> str:
        center_x = (float(target[0]) + float(target[2])) * 0.5
        center_y = (float(target[1]) + float(target[3])) * 0.5
        width = max(0.0, float(target[2]) - float(target[0]))
        height = max(0.0, float(target[3]) - float(target[1]))
        confidence = float(target[4]) if len(target) > 4 else 0.0
        return (
            f"center=({center_x:.0f},{center_y:.0f}), "
            f"size={width:.0f}x{height:.0f}, conf={confidence:.2f}"
        )

    def _exit_via_door(self, w: "FrameWorker", door: Sequence[float]) -> bool:
        self.trusted_exit_signal = False
        self._align_to_target(w, door, max_steps=self.TARGET_ALIGN_MAX_STEPS)
        w.refresh_frame()
        if self._handle_terminal_state(w, "对准门时检测到死亡或结算界面，结束出房流程"):
            return True

        if w.get_info("开门"):
            w.frame_log("[HouseExit] 检测到开门按钮，门处于关闭状态，点击开门")
            w.click("开门")
            self.trusted_exit_signal = True
            time.sleep(0.8)
            w.refresh_frame()
            if self._handle_terminal_state(w, "开门后检测到死亡或结算界面，结束出房流程"):
                return True
        elif w.get_info("关门"):
            w.frame_log("[HouseExit] 检测到关门按钮，门已打开，继续前推出门")
            self.trusted_exit_signal = True

        approached = self._approach_door_with_recovery(w, door)
        if self._handle_terminal_state(w, "靠近门过程中检测到死亡或结算界面，结束出房流程"):
            return True
        if not approached:
            return False
        if self._verify_exit_success(w):
            return True
        return self._door_diagonal_sweep(w)

    def _approach_door_with_recovery(self, w: "FrameWorker", door: Sequence[float]) -> bool:
        for step in range(3):
            if self._handle_terminal_state(w, "靠近门前检测到死亡或结算界面，结束出房流程"):
                return True

            refreshed = self._find_largest_target(self._get_forward_scene(w), self.DOOR_CLASS_IDS)
            if refreshed:
                door = refreshed
                w.frame_log(f"[HouseExit] 靠门第 {step + 1}/3 步重新识别到门，先修正视角")
                self._align_to_target(w, door, max_steps=self.TARGET_ALIGN_MAX_STEPS)
                w.refresh_frame()
                if self._handle_terminal_state(w, "修正门位置后检测到死亡或结算界面，结束出房流程"):
                    return True
                if w.get_info("开门"):
                    w.click("开门")
                    self.trusted_exit_signal = True
                    time.sleep(0.6)
                    w.refresh_frame()
                    if self._handle_terminal_state(w, "补点开门后检测到死亡或结算界面，结束出房流程"):
                        return True
                elif w.get_info("关门"):
                    self.trusted_exit_signal = True
                w.frame_log("[HouseExit] 门已对准，前推 240ms 尝试穿过门口")
                self._move_forward(w, dura=240, wait=420)
                w.refresh_frame()
                if self._handle_terminal_state(w, "门口前推后检测到死亡或结算界面，结束出房流程"):
                    return True
                if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                    return True
                if self._get_house_scene(w) in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
                    w.frame_log("[HouseExit] 前推后仍处于贴门/贴墙状态，转入左右斜向顶出")
                    return self._door_diagonal_sweep(w)
                continue

            w.frame_log(
                f"[HouseExit] 靠门时目标丢失，先前推 {self.DOOR_LOST_FORWARD_DURA}ms 试探"
            )
            self._move_forward(w, dura=self.DOOR_LOST_FORWARD_DURA, wait=self.DOOR_LOST_FORWARD_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门丢失试错前推后检测到死亡或结算界面，结束出房流程"):
                return True
            if w.get_info("开门") or w.get_info("关门"):
                self.trusted_exit_signal = True
                return True
            if self._find_largest_target(self._get_forward_scene(w), self.DOOR_CLASS_IDS):
                continue

            w.frame_log(
                f"[HouseExit] 前推后仍未找到门，后退 {self.DOOR_RECOVER_BACK_DURA}ms 重新定位"
            )
            self._move_backward(w, dura=self.DOOR_RECOVER_BACK_DURA, wait=self.DOOR_RECOVER_BACK_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门口后退重定位时检测到死亡或结算界面，结束出房流程"):
                return True
            recovered = self._find_largest_target(self._get_forward_scene(w), self.DOOR_CLASS_IDS)
            if not recovered:
                w.frame_log("[HouseExit] 后退重定位后仍未找到门，本轮门口出房失败")
                return False
            self._align_to_target(w, recovered, max_steps=self.TARGET_ALIGN_MAX_STEPS)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门口重定位后检测到死亡或结算界面，结束出房流程"):
                return True

        return True

    def _door_diagonal_sweep(self, w: "FrameWorker") -> bool:
        w.frame_log(f"[HouseExit] 门口受阻，开始左右斜向顶出，共 {self.DOOR_DIAGONAL_SWEEP_STEPS} 次")
        for step in range(self.DOOR_DIAGONAL_SWEEP_STEPS):
            if self._handle_terminal_state(w, "门口顶出前检测到死亡或结算界面，结束出房流程"):
                return True
            side_sign = -1 if step % 2 == 0 else 1
            dura = min(
                self.DOOR_DIAGONAL_SWEEP_BASE_DURA + step * self.DOOR_DIAGONAL_SWEEP_STEP_DURA,
                self.DOOR_DIAGONAL_SWEEP_MAX_DURA,
            )
            if w.get_info("开门"):
                w.frame_log("[HouseExit] 斜向顶出前再次检测到开门按钮，补点开门")
                w.click("开门")
                self.trusted_exit_signal = True
                time.sleep(0.35)
                w.refresh_frame()
                if self._handle_terminal_state(w, "门口补开门后检测到死亡或结算界面，结束出房流程"):
                    return True
            side_name = "左上" if side_sign < 0 else "右上"
            w.frame_log(
                f"[HouseExit] 斜向顶出 {step + 1}/{self.DOOR_DIAGONAL_SWEEP_STEPS}: "
                f"方向={side_name}, x_bias={side_sign * self.DOOR_DIAGONAL_SWEEP_X_BIAS}, "
                f"y_bias={self.DOOR_DIAGONAL_SWEEP_Y_BIAS}, dura={dura}ms"
            )
            w.tap_single(
                "摇杆",
                x_bias=side_sign * self.DOOR_DIAGONAL_SWEEP_X_BIAS,
                y_bias=self.DOOR_DIAGONAL_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.DOOR_DIAGONAL_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()
            if self._handle_terminal_state(w, "门口顶出过程中检测到死亡或结算界面，结束出房流程"):
                return True
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                w.frame_log(f"[HouseExit] 第 {step + 1} 次斜向顶出后检测到室外信号")
                return self._verify_exit_success(w)
        w.frame_log("[HouseExit] 左右斜向顶出结束，仍未离开房屋")
        return False

    def _exit_via_window(self, w: "FrameWorker", window: Sequence[float]) -> bool:
        self.trusted_exit_signal = False
        self._align_to_target(w, window, max_steps=self.TARGET_ALIGN_MAX_STEPS)
        w.refresh_frame()
        if self._handle_terminal_state(w, "对准窗户时检测到死亡或结算界面，结束出房流程"):
            return True

        for step in range(self.WINDOW_APPROACH_STEPS):
            if self._handle_terminal_state(w, "靠窗前检测到死亡或结算界面，结束出房流程"):
                return True
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                return self._verify_exit_success(w)

            if w.get_info("跳跃"):
                w.frame_log(
                    f"[HouseExit] 靠窗 {step + 1}/{self.WINDOW_APPROACH_STEPS}: "
                    f"检测到跳跃按钮，点击跳跃并前推 {self.WINDOW_JUMP_FORWARD_DURA}ms"
                )
                w.click("跳跃")
                time.sleep(0.12)
                self._move_forward(w, dura=self.WINDOW_JUMP_FORWARD_DURA, wait=self.WINDOW_JUMP_FORWARD_WAIT)
                w.refresh_frame()
                if self._handle_terminal_state(w, "翻窗前推后检测到死亡或结算界面，结束出房流程"):
                    return True
                if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                    self.trusted_exit_signal = True
                    return self._verify_exit_success(w)
                window = self._find_largest_target(self._get_forward_scene(w), self.WINDOW_CLASS_IDS)
                if not window:
                    recovered, window = self._recover_window_with_lateral_sweep(w)
                    if recovered:
                        return True
                    if not window:
                        return False
                continue

            w.frame_log(
                f"[HouseExit] 靠窗 {step + 1}/{self.WINDOW_APPROACH_STEPS}: "
                f"未检测到跳跃按钮，前推 {self.WINDOW_APPROACH_DURA}ms 继续靠近"
            )
            self._move_forward(w, dura=self.WINDOW_APPROACH_DURA, wait=self.WINDOW_APPROACH_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "靠窗前推后检测到死亡或结算界面，结束出房流程"):
                return True
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                return self._verify_exit_success(w)
            if self._get_house_scene(w) in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
                w.frame_log("[HouseExit] 靠窗前推后贴墙，转入左右横移重新定位窗户")
                recovered, window = self._recover_window_with_lateral_sweep(w)
                if recovered:
                    return True
                if not window:
                    return False

            window = self._find_largest_target(self._get_forward_scene(w), self.WINDOW_CLASS_IDS)
            if not window:
                recovered, window = self._recover_window_with_lateral_sweep(w)
                if recovered:
                    return True
                if not window:
                    break
            else:
                self._align_to_target(w, window, max_steps=1)
                w.refresh_frame()
                if self._handle_terminal_state(w, "重新对准窗户后检测到死亡或结算界面，结束出房流程"):
                    return True

        w.frame_log("[HouseExit] 靠窗尝试结束，仍未成功翻出")
        return False

    def _recover_window_with_lateral_sweep(self, w: "FrameWorker"):
        for step in range(self.WINDOW_LATERAL_RECOVER_STEPS):
            if self._handle_terminal_state(w, "窗边微调前检测到死亡或结算界面，结束出房流程"):
                return True, None
            side_sign = -1 if step % 2 == 0 else 1
            side_name = "左" if side_sign < 0 else "右"
            w.frame_log(
                f"[HouseExit] 窗边横移 {step + 1}/{self.WINDOW_LATERAL_RECOVER_STEPS}: "
                f"方向={side_name}, x_bias={side_sign * self.WINDOW_LATERAL_RECOVER_X_BIAS}, "
                f"dura={self.WINDOW_LATERAL_RECOVER_DURA}ms"
            )
            w.tap_single(
                "摇杆",
                x_bias=side_sign * self.WINDOW_LATERAL_RECOVER_X_BIAS,
                y_bias=0,
                dura=self.WINDOW_LATERAL_RECOVER_DURA,
                wait=self.WINDOW_LATERAL_RECOVER_WAIT,
            )
            w.refresh_frame()
            if self._handle_terminal_state(w, "窗边微调时检测到死亡或结算界面，结束出房流程"):
                return True, None
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                return True, None
            if w.get_info("跳跃"):
                w.frame_log(
                    f"[HouseExit] 横移后检测到跳跃按钮，点击跳跃并前推 "
                    f"{self.WINDOW_JUMP_FORWARD_DURA}ms"
                )
                w.click("跳跃")
                time.sleep(0.12)
                self._move_forward(w, dura=self.WINDOW_JUMP_FORWARD_DURA, wait=self.WINDOW_JUMP_FORWARD_WAIT)
                w.refresh_frame()
                if self._handle_terminal_state(w, "窗边跳跃前推后检测到死亡或结算界面，结束出房流程"):
                    return True, None
                if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                    self.trusted_exit_signal = True
                    return self._verify_exit_success(w), None
            window = self._find_largest_target(self._get_forward_scene(w), self.WINDOW_CLASS_IDS)
            if window:
                w.frame_log(f"[HouseExit] 横移后重新找到窗: {self._describe_target(window)}")
                return False, window
        w.frame_log("[HouseExit] 窗边横移结束，仍未重新找到窗户")
        return False, None

    def _verify_exit_success(self, w: "FrameWorker") -> bool:
        w.refresh_frame()
        if self._handle_terminal_state(w, "确认出房时检测到死亡或结算界面，结束出房流程"):
            return True
        if self._get_house_scene(w) not in self.HOUSE_EXIT_SCENES:
            return False

        signal_source = "门窗流程" if self.trusted_exit_signal else "场景识别"
        w.frame_log(
            f"[HouseExit] {signal_source}检测到室外信号，先前推 "
            f"{self.EXIT_CONFIRM_FORWARD_DURA}ms，再回看 180° 二次确认"
        )
        self._move_forward(w, dura=self.EXIT_CONFIRM_FORWARD_DURA, wait=self.EXIT_CONFIRM_FORWARD_WAIT)
        current_dir = w.get_info("direction")
        if current_dir is None:
            w.frame_log("[HouseExit] 二次确认时方向无效，无法回看180°，本次确认失败")
            return False

        self._align_direction_blocking(
            w,
            current_dir,
            (current_dir + 180) % 360,
            tolerance=15,
            max_steps=5,
            dura=450,
            wait=260,
        )
        w.refresh_frame()
        if self._handle_terminal_state(w, "二次确认出房时检测到死亡或结算界面，结束出房流程"):
            return True

        if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
            w.frame_log(f"[HouseExit] 二次确认成功，scene={self._get_house_scene(w)}，确定已离开房屋")
            restored_dir = w.get_info("direction")
            if restored_dir is not None:
                w.frame_log(f"[HouseExit] 恢复出房方向: {restored_dir}° → {current_dir}°")
                self._align_direction_blocking(
                    w,
                    restored_dir,
                    current_dir,
                    tolerance=15,
                    max_steps=5,
                    dura=450,
                    wait=260,
                )
                w.refresh_frame()
                if self._handle_terminal_state(w, "恢复出房方向时检测到死亡或结算界面，结束出房流程"):
                    return True
            self.reset()
            return True

        w.frame_log(f"[HouseExit] 回看180°后 scene={self._get_house_scene(w)}，二次确认失败，继续出房")
        return False

    def _search_for_exit(self, w: "FrameWorker"):
        if self._handle_terminal_state(w, "扫描出口前检测到死亡或结算界面，结束出房流程"):
            return True
        current_dir = w.get_info("direction")
        if current_dir is None:
            w.frame_log("[HouseExit] 扫描出口时方向无效，本帧无法转动视角")
            return False

        if self.scan_index >= len(self.SCAN_OFFSETS):
            self.scan_index = 0
            self.scan_anchor_direction = None
            return self._handle_no_exit_after_scan(w)

        target_dir = (self.scan_anchor_direction + self.SCAN_OFFSETS[self.scan_index]) % 360
        w.frame_log(
            f"[HouseExit] 扫描门窗 {self.scan_index + 1}/{len(self.SCAN_OFFSETS)}: "
            f"current={current_dir}°, anchor={self.scan_anchor_direction}°, target={target_dir:.1f}°"
        )
        self._align_direction_blocking(w, current_dir, target_dir, tolerance=8)
        w.refresh_frame()
        if self._handle_terminal_state(w, "扫描出口时检测到死亡或结算界面，结束出房流程"):
            return True

        self.scan_index += 1
        visible_result = self._try_visible_exit_target(w)
        if visible_result is not None:
            return visible_result

        return False

    def _handle_no_exit_after_scan(self, w: "FrameWorker") -> bool:
        scene = self._get_house_scene(w)
        w.frame_log(f"[HouseExit] 一轮视角扫描结束仍未发现门窗，scene={scene}")
        if scene in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
            if self._recover_from_wall_and_turn_back(w):
                return True
        return self._escape_dead_end_randomly(w)

    def _recover_from_wall_and_turn_back(self, w: "FrameWorker") -> bool:
        w.frame_log(
            f"[HouseExit] 扫描后仍贴墙，先后退 {self.WALL_BACKOFF_DURA}ms，"
            f"再掉头 {self.WALL_TURN_BACK_DEGREES}°"
        )
        self._move_backward(w, dura=self.WALL_BACKOFF_DURA, wait=self.WALL_BACKOFF_WAIT)
        w.refresh_frame()
        if self._handle_terminal_state(w, "撞墙后拉时检测到死亡或结算界面，结束出房流程"):
            return True

        current_dir = w.get_info("direction")
        if current_dir is not None:
            self._align_direction_blocking(
                w,
                current_dir,
                (current_dir + self.WALL_TURN_BACK_DEGREES) % 360,
                tolerance=12,
                max_steps=8,
            )
            w.refresh_frame()
            if self._handle_terminal_state(w, "撞墙后转身时检测到死亡或结算界面，结束出房流程"):
                return True

        visible_result = self._try_visible_exit_target(w)
        if visible_result is not None:
            return visible_result
        if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
            return self._verify_exit_success(w)
        return False

    def _escape_dead_end_randomly(self, w: "FrameWorker") -> bool:
        w.frame_log(
            f"[HouseExit] 未发现门窗，启动自动前进随机脱困，"
            f"最多尝试 {self.DEAD_END_ESCAPE_STEPS} 步"
        )
        self._start_auto_forward(w)
        for step in range(self.DEAD_END_ESCAPE_STEPS):
            w.refresh_frame()
            if self._handle_terminal_state(w, "死胡同随机跑动中检测到死亡或结算界面，结束出房流程"):
                return True
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                self._stop_auto_forward(w)
                return self._verify_exit_success(w)

            detections = self._get_forward_scene(w)
            if (
                self._find_largest_target(detections, self.DOOR_CLASS_IDS)
                or self._find_largest_target(detections, self.WINDOW_CLASS_IDS)
            ):
                w.frame_log("[HouseExit] 随机脱困中重新发现门窗，停止自动前进并切回目标出房")
                self._stop_auto_forward(w)
                visible_result = self._try_visible_exit_target(w)
                return bool(visible_result)

            turn_sign = random.choice((-1, 1))
            side_sign = random.choice((-1, 1))
            w.frame_log(
                f"[HouseExit] 随机脱困 {step + 1}/{self.DEAD_END_ESCAPE_STEPS}: "
                f"视角x={turn_sign * self.DEAD_END_ESCAPE_TURN_X_BIAS}, "
                f"摇杆x={side_sign * self.DEAD_END_ESCAPE_SIDE_X_BIAS}, y=-300"
            )
            w.tap_single(
                "视角",
                x_bias=turn_sign * self.DEAD_END_ESCAPE_TURN_X_BIAS,
                dura=self.DEAD_END_ESCAPE_TURN_DURA,
                wait=self.DEAD_END_ESCAPE_TURN_WAIT,
            )
            w.tap_single(
                "摇杆",
                x_bias=side_sign * self.DEAD_END_ESCAPE_SIDE_X_BIAS,
                y_bias=-300,
                dura=self.DEAD_END_ESCAPE_SIDE_DURA,
                wait=self.DEAD_END_ESCAPE_SIDE_WAIT,
            )
            if self._handle_terminal_state(w, "死胡同随机转向后检测到死亡或结算界面，结束出房流程"):
                return True
        self._stop_auto_forward(w)
        w.frame_log("[HouseExit] 随机脱困达到最大步数，仍未离开房屋")
        return False

    def _escape_after_failed_exit_scan(self, w: "FrameWorker") -> bool:
        if self._handle_terminal_state(w, "旧兜底出房前检测到死亡或结算界面，结束出房流程"):
            return True
        current_dir = w.get_info("direction")
        if current_dir is not None:
            target_dir = (current_dir + 180) % 360
            w.frame_log(f"[HouseExit] 旧兜底扫描失败，先掉头: {current_dir}° → {target_dir:.1f}°")
            self._align_direction_blocking(w, current_dir, target_dir, tolerance=12, max_steps=8)
            w.refresh_frame()
            if self._handle_terminal_state(w, "旧兜底转身后检测到死亡或结算界面，结束出房流程"):
                return True

        w.frame_log(
            f"[HouseExit] 旧兜底左上长推: x={-self.NO_EXIT_ESCAPE_X_BIAS}, "
            f"y={self.NO_EXIT_ESCAPE_Y_BIAS}, dura={self.NO_EXIT_ESCAPE_DURA}ms"
        )
        w.tap_single(
            "摇杆",
            x_bias=-self.NO_EXIT_ESCAPE_X_BIAS,
            y_bias=self.NO_EXIT_ESCAPE_Y_BIAS,
            dura=self.NO_EXIT_ESCAPE_DURA,
            wait=self.NO_EXIT_ESCAPE_WAIT,
        )
        w.refresh_frame()
        if self._handle_terminal_state(w, "旧兜底左上推进后检测到死亡或结算界面，结束出房流程"):
            return True

        w.frame_log(
            f"[HouseExit] 旧兜底右上长推: x={self.NO_EXIT_ESCAPE_X_BIAS}, "
            f"y={self.NO_EXIT_ESCAPE_Y_BIAS}, dura={self.NO_EXIT_ESCAPE_DURA}ms"
        )
        w.tap_single(
            "摇杆",
            x_bias=self.NO_EXIT_ESCAPE_X_BIAS,
            y_bias=self.NO_EXIT_ESCAPE_Y_BIAS,
            dura=self.NO_EXIT_ESCAPE_DURA,
            wait=self.NO_EXIT_ESCAPE_WAIT,
        )
        w.refresh_frame()
        if self._handle_terminal_state(w, "旧兜底右上推进后检测到死亡或结算界面，结束出房流程"):
            return True

        if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
            return self._verify_exit_success(w)

        self._rotate_search_view(w)
        w.refresh_frame()
        if self._handle_terminal_state(w, "旧兜底转视角后检测到死亡或结算界面，结束出房流程"):
            return True
        return False

    def _rotate_search_view(self, w: "FrameWorker"):
        current_dir = w.get_info("direction")
        if current_dir is None:
            return
        self._align_direction_blocking(w, current_dir, (current_dir + 60) % 360, tolerance=10)

    def _ensure_scan_anchor(self, w: "FrameWorker"):
        if self.scan_anchor_direction is None:
            self.scan_anchor_direction = w.get_info("direction")
            self.scan_index = 0
            if self.scan_anchor_direction is None:
                w.frame_log("[HouseExit] 当前方向无效，扫描锚点尚未建立")
            else:
                w.frame_log(f"[HouseExit] 建立扫描锚点: {self.scan_anchor_direction}°")

    def _start_auto_forward(self, w: "FrameWorker"):
        if self.auto_forward_enabled:
            return
        w.frame_log("[HouseExit] 点击自动前进，开始持续移动")
        w.click("自动前进")
        self.auto_forward_enabled = True

    def _stop_auto_forward(self, w: "FrameWorker"):
        if not self.auto_forward_enabled:
            return
        w.frame_log("[HouseExit] 再次点击自动前进，停止持续移动")
        w.click("自动前进")
        self.auto_forward_enabled = False

    def _move_forward(self, w: "FrameWorker", dura=350, wait=600):
        w.tap_single("摇杆", y_bias=-300, dura=dura, wait=wait)

    def _move_backward(self, w: "FrameWorker", dura=350, wait=600):
        w.tap_single("摇杆", y_bias=300, dura=dura, wait=wait)

    def _align_to_target(
        self,
        w: "FrameWorker",
        target: Sequence[float],
        tolerance_px=80,
        max_steps: Optional[int] = None,
    ):
        steps = self.TARGET_ALIGN_MAX_STEPS if max_steps is None else max_steps
        target_name = "门" if int(target[5]) in self.DOOR_CLASS_IDS else "窗"
        for step in range(steps):
            inf_w, inf_h = get_wh()
            frame_w = max(inf_w, inf_h)
            scale = self.screen_w / frame_w
            target_center_x = (target[0] + target[2]) / 2
            offset_real = (target_center_x - (frame_w / 2)) * scale

            if abs(offset_real) <= tolerance_px:
                w.frame_log(
                    f"[HouseExit] {target_name}已对准: offset={offset_real:.0f}px, "
                    f"tolerance={tolerance_px}px"
                )
                return True

            adjust_val = int(offset_real * 0.33)
            adjust_val = max(-400, min(400, adjust_val))
            w.frame_log(
                f"[HouseExit] 对准{target_name} {step + 1}/{steps}: "
                f"offset={offset_real:.0f}px, 视角x_bias={adjust_val}"
            )
            w.tap_single("视角", x_bias=adjust_val, dura=500, wait=500)
            w.refresh_frame()
            if self._handle_terminal_state(w, "对准门窗过程中检测到死亡或结算界面，结束出房流程"):
                return False

            detections = self._get_forward_scene(w)
            refreshed_target = self._find_largest_target(
                detections,
                self.DOOR_CLASS_IDS if int(target[5]) in self.DOOR_CLASS_IDS else self.WINDOW_CLASS_IDS,
            )
            if refreshed_target is None:
                w.frame_log(f"[HouseExit] 调整视角后{target_name}丢失，停止本次对准")
                return False
            target = refreshed_target
        w.frame_log(f"[HouseExit] {target_name}对准达到最大步数，仍未进入容差范围")
        return False

    def _align_direction_blocking(
        self,
        w: "FrameWorker",
        current_dir,
        target_angle,
        tolerance=5,
        max_steps=10,
        dura=800,
        wait=500,
    ):
        return execute_view_turn(
            w,
            current_dir,
            target_angle,
            threshold=tolerance,
            max_steps=max_steps,
            wait=wait,
            fallback_dura=dura,
            log_prefix="[HouseExit]",
        )
