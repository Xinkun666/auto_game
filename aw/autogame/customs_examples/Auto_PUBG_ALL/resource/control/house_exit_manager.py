import builtins
import random
import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    execute_view_turn,
)
from aw.autogame.tools.Utils import get_resolution, get_wh

def print(*values, sep=" ", end="\n", file=None, flush=False, **_kwargs):
    builtins.print(*values, sep=sep, end=end, file=file, flush=flush)

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

    def _set_frame_decision(
        self,
        w: "FrameWorker",
        observation: str,
        decision: str,
        action: Optional[str] = None,
        method: str = "",
        result: str = "",
        target: str = "出房兜底阶段",
    ):
        setter = getattr(w, "set_frame_decision", None)
        if not callable(setter):
            return
        setter(
            observation=observation,
            target=target,
            decision=decision,
            action=action or decision,
            method=method,
            result=result,
        )

    def process(self, w: "FrameWorker") -> bool:
        if self._handle_terminal_state(w, "检测到死亡或结算界面，结束出房兜底流程"):
            return True

        house_scene = self._get_house_scene(w)
        self._set_frame_decision(
            w,
            f"出房兜底：当前 house_scene={house_scene}",
            "根据门、窗、贴墙/贴门和室外信号选择出房动作",
            action="执行出房兜底决策",
            method="HouseExitManager.process()",
            result="本帧继续找门窗或确认已出房",
        )
        if house_scene not in self.HOUSE_ACTIVE_EXIT_SCENES:
            if house_scene in self.HOUSE_EXIT_SCENES:
                self._set_frame_decision(
                    w,
                    f"出房兜底检测到室外/楼顶 house_scene={house_scene}",
                    "复核出房成功，停止出房兜底流程",
                    action="确认出房成功",
                    method="_verify_exit_success()",
                    result="回到后续阶段",
                )
                return self._verify_exit_success(w)
            self._set_frame_decision(
                w,
                f"出房兜底当前 house_scene={house_scene}，不在可出房处理场景",
                "本帧不接管，让上层搜房/跑图逻辑继续处理",
                action="不接管本帧",
                method="return False",
                result="上层逻辑继续决策",
            )
            return False

        self.trusted_exit_signal = False
        self._ensure_scan_anchor(w)
        visible_result = self._try_visible_exit_target(w)
        if visible_result is not None:
            return visible_result

        self._set_frame_decision(
            w,
            f"出房兜底未直接看到门/窗，house_scene={house_scene}",
            "按扫描锚点继续找出口，必要时转向、绕圈或随机冲出",
            action="搜索出口",
            method="_search_for_exit()",
            result="下一帧继续确认门窗或室外信号",
        )
        return self._search_for_exit(w)

    def _try_visible_exit_target(self, w: "FrameWorker") -> Optional[bool]:
        detections = self._get_forward_scene(w)

        door = self._find_largest_target(detections, self.DOOR_CLASS_IDS)
        if door:
            print("[HouseExit] 发现门，准备出门")
            self._set_frame_decision(
                w,
                f"当前帧 forward_scene 发现门，door={door}",
                "停止自动前进，对准门，必要时开门后前推出房",
                action="对准门并出门",
                method="_exit_via_door()",
                result="下一帧确认是否已到室外",
            )
            self._stop_auto_forward(w)
            return self._exit_via_door(w, door)

        window = self._find_largest_target(detections, self.WINDOW_CLASS_IDS)
        if window:
            print("[HouseExit] 发现窗，准备翻窗")
            self._set_frame_decision(
                w,
                f"当前帧 forward_scene 发现窗，window={window}",
                "停止自动前进，对准窗，找跳跃按钮并前推翻窗",
                action="对准窗并翻窗",
                method="_exit_via_window()",
                result="下一帧确认是否已到室外",
            )
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
        print(f"[HouseExit] {reason}")
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

    def _exit_via_door(self, w: "FrameWorker", door: Sequence[float]) -> bool:
        self.trusted_exit_signal = False
        self._align_to_target(w, door, max_steps=self.TARGET_ALIGN_MAX_STEPS)
        w.refresh_frame()
        if self._handle_terminal_state(w, "对准门时检测到死亡或结算界面，结束出房流程"):
            return True

        if w.get_info("开门"):
            print("[HouseExit] 门已关闭，先开门")
            self._set_frame_decision(
                w,
                "当前帧检测到开门按钮，判断门处于关闭状态",
                "先点击开门，再靠近门口前推出房",
                action="点击开门",
                method="w.click(开门)",
                result="下一帧确认门是否打开并继续前推",
            )
            w.click("开门")
            self.trusted_exit_signal = True
            time.sleep(0.8)
            w.refresh_frame()
            if self._handle_terminal_state(w, "开门后检测到死亡或结算界面，结束出房流程"):
                return True
        elif w.get_info("关门"):
            print("[HouseExit] 检测到关门按钮，表示门已打开，直接出门")
            self._set_frame_decision(
                w,
                "当前帧检测到关门按钮，判断门已经打开",
                "不再点击门按钮，直接靠近门口前推出房",
                action="前推出门",
                method="_approach_door_with_recovery()",
                result="下一帧确认是否已到室外",
            )
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
                print(f"[HouseExit] 靠近门口前修正门位置 step={step + 1}")
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
                self._move_forward(w, dura=240, wait=420)
                w.refresh_frame()
                if self._handle_terminal_state(w, "门口前推后检测到死亡或结算界面，结束出房流程"):
                    return True
                if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                    return True
                if self._get_house_scene(w) in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
                    print("[HouseExit] 门口前推后贴墙/门，转入左上/右上小幅顶出")
                    return self._door_diagonal_sweep(w)
                continue

            print("[HouseExit] 前推时门丢失，先给一次前推试错")
            self._move_forward(w, dura=self.DOOR_LOST_FORWARD_DURA, wait=self.DOOR_LOST_FORWARD_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门丢失试错前推后检测到死亡或结算界面，结束出房流程"):
                return True
            if w.get_info("开门") or w.get_info("关门"):
                self.trusted_exit_signal = True
                return True
            if self._find_largest_target(self._get_forward_scene(w), self.DOOR_CLASS_IDS):
                continue

            print("[HouseExit] 试错后仍未找到门，后退并重新定位门")
            self._move_backward(w, dura=self.DOOR_RECOVER_BACK_DURA, wait=self.DOOR_RECOVER_BACK_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门口后退重定位时检测到死亡或结算界面，结束出房流程"):
                return True
            recovered = self._find_largest_target(self._get_forward_scene(w), self.DOOR_CLASS_IDS)
            if not recovered:
                return False
            self._align_to_target(w, recovered, max_steps=self.TARGET_ALIGN_MAX_STEPS)
            w.refresh_frame()
            if self._handle_terminal_state(w, "门口重定位后检测到死亡或结算界面，结束出房流程"):
                return True

        return True

    def _door_diagonal_sweep(self, w: "FrameWorker") -> bool:
        print("[HouseExit] 门口出房受阻，左右小幅上滑尝试顶出")
        for step in range(self.DOOR_DIAGONAL_SWEEP_STEPS):
            if self._handle_terminal_state(w, "门口顶出前检测到死亡或结算界面，结束出房流程"):
                return True
            side_sign = -1 if step % 2 == 0 else 1
            dura = min(
                self.DOOR_DIAGONAL_SWEEP_BASE_DURA + step * self.DOOR_DIAGONAL_SWEEP_STEP_DURA,
                self.DOOR_DIAGONAL_SWEEP_MAX_DURA,
            )
            if w.get_info("开门"):
                print("[HouseExit] 顶出前再次看到开门按钮，补点开门")
                self._set_frame_decision(
                    w,
                    "门口顶出前再次看到开门按钮，说明门又处于关闭/未开状态",
                    "补点开门后继续左右上小幅顶出",
                    action="补点开门",
                    method="w.click(开门)",
                    result="继续门口顶出流程",
                )
                w.click("开门")
                self.trusted_exit_signal = True
                time.sleep(0.35)
                w.refresh_frame()
                if self._handle_terminal_state(w, "门口补开门后检测到死亡或结算界面，结束出房流程"):
                    return True
            self._set_frame_decision(
                w,
                f"门口出房受阻，第 {step + 1} 次左右上顶出，side_sign={side_sign}",
                "用小幅左上/右上滑动尝试从门边挤出，避免继续原地撞门框",
                action="门口左右上顶出",
                method=(
                    "tap_single(摇杆, "
                    f"x_bias={side_sign * self.DOOR_DIAGONAL_SWEEP_X_BIAS}, "
                    f"y_bias={self.DOOR_DIAGONAL_SWEEP_Y_BIAS}, dura={dura})"
                ),
                result="下一帧确认 house_scene 是否变为室外/楼顶",
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
                return self._verify_exit_success(w)
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
                print(f"[HouseExit] 靠窗 step={step + 1}，点击跳跃并前推翻窗")
                self._set_frame_decision(
                    w,
                    f"靠窗 step={step + 1} 检测到跳跃按钮，判断可以翻窗",
                    "点击跳跃并前推翻窗出房",
                    action="跳跃前推翻窗",
                    method="w.click(跳跃); _move_forward()",
                    result="下一帧确认是否已到室外",
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

            print("[HouseExit] 小步靠近窗户")
            self._set_frame_decision(
                w,
                f"靠窗 step={step + 1} 未看到跳跃按钮",
                "先小步前推靠近窗户，继续寻找跳跃按钮",
                action="靠近窗户",
                method=f"_move_forward(dura={self.WINDOW_APPROACH_DURA})",
                result="下一帧继续检测跳跃/室外/贴墙信号",
            )
            self._move_forward(w, dura=self.WINDOW_APPROACH_DURA, wait=self.WINDOW_APPROACH_WAIT)
            w.refresh_frame()
            if self._handle_terminal_state(w, "靠窗前推后检测到死亡或结算界面，结束出房流程"):
                return True
            if self._get_house_scene(w) in self.HOUSE_EXIT_SCENES:
                return self._verify_exit_success(w)
            if self._get_house_scene(w) in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
                print("[HouseExit] 靠窗前推后贴墙，左右小幅移动重新定位窗户")
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

        return False

    def _recover_window_with_lateral_sweep(self, w: "FrameWorker"):
        for step in range(self.WINDOW_LATERAL_RECOVER_STEPS):
            if self._handle_terminal_state(w, "窗边微调前检测到死亡或结算界面，结束出房流程"):
                return True, None
            side_sign = -1 if step % 2 == 0 else 1
            print(f"[HouseExit] 靠窗左右小幅移动重新定位 {step + 1}/{self.WINDOW_LATERAL_RECOVER_STEPS}")
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
                print("[HouseExit] 小幅移动后出现跳跃，点击跳跃并前推")
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
                return False, window
        return False, None

    def _verify_exit_success(self, w: "FrameWorker") -> bool:
        w.refresh_frame()
        if self._handle_terminal_state(w, "确认出房时检测到死亡或结算界面，结束出房流程"):
            return True
        if self._get_house_scene(w) not in self.HOUSE_EXIT_SCENES:
            return False

        if self.trusted_exit_signal:
            print("[HouseExit] 已通过门/窗看到屋外信号，但仍执行二次确认，避免窗外/门外视野误判")
        else:
            print("[HouseExit] 首次判定已在屋外，执行二次确认")

        self._move_forward(w, dura=self.EXIT_CONFIRM_FORWARD_DURA, wait=self.EXIT_CONFIRM_FORWARD_WAIT)
        current_dir = w.get_info("direction")
        if current_dir is None:
            print("[HouseExit] 二次确认缺少方向，继续尝试出房")
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
            print("[HouseExit] 出房成功")
            restored_dir = w.get_info("direction")
            if restored_dir is not None:
                print("[HouseExit] 二次确认后转回出房方向，避免回头冲回房内")
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

        print("[HouseExit] 二次确认失败，继续尝试出房")
        return False

    def _search_for_exit(self, w: "FrameWorker"):
        if self._handle_terminal_state(w, "扫描出口前检测到死亡或结算界面，结束出房流程"):
            return True
        current_dir = w.get_info("direction")
        if current_dir is None:
            return False

        if self.scan_index >= len(self.SCAN_OFFSETS):
            self.scan_index = 0
            self.scan_anchor_direction = None
            return self._handle_no_exit_after_scan(w)

        target_dir = (self.scan_anchor_direction + self.SCAN_OFFSETS[self.scan_index]) % 360
        print(f"[HouseExit] 扫视角度 {target_dir:.1f}")
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
        if scene in {self.HOUSE_NEAR_DOOR, self.HOUSE_NEAR_WALL}:
            if self._recover_from_wall_and_turn_back(w):
                return True
        return self._escape_dead_end_randomly(w)

    def _recover_from_wall_and_turn_back(self, w: "FrameWorker") -> bool:
        print("[HouseExit] 左右扫视仍是墙，先后拉再向后调转方向")
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
        print("[HouseExit] 仍未发现门窗，启动自动前进随机跑动脱离死胡同")
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
                print("[HouseExit] 随机跑动中重新看到门/窗，停止自动前进并尝试出房")
                self._stop_auto_forward(w)
                visible_result = self._try_visible_exit_target(w)
                return bool(visible_result)

            turn_sign = random.choice((-1, 1))
            side_sign = random.choice((-1, 1))
            print(f"[HouseExit] 死胡同随机跑动 {step + 1}/{self.DEAD_END_ESCAPE_STEPS}")
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
        return False

    def _escape_after_failed_exit_scan(self, w: "FrameWorker") -> bool:
        if self._handle_terminal_state(w, "旧兜底出房前检测到死亡或结算界面，结束出房流程"):
            return True
        current_dir = w.get_info("direction")
        if current_dir is not None:
            target_dir = (current_dir + 180) % 360
            print(f"[HouseExit] 多角度未发现门窗，先调转 180 度到 {target_dir:.1f}")
            self._align_direction_blocking(w, current_dir, target_dir, tolerance=12, max_steps=8)
            w.refresh_frame()
            if self._handle_terminal_state(w, "旧兜底转身后检测到死亡或结算界面，结束出房流程"):
                return True

        print("[HouseExit] 未发现门窗，左上推动摇杆 5s 尝试出房")
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

        print("[HouseExit] 未发现门窗，右上推动摇杆 5s 尝试出房")
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

    def _start_auto_forward(self, w: "FrameWorker"):
        if self.auto_forward_enabled:
            return
        w.click("自动前进")
        self.auto_forward_enabled = True

    def _stop_auto_forward(self, w: "FrameWorker"):
        if not self.auto_forward_enabled:
            return
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
        for _ in range(steps):
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
            if self._handle_terminal_state(w, "对准门窗过程中检测到死亡或结算界面，结束出房流程"):
                return False

            detections = self._get_forward_scene(w)
            refreshed_target = self._find_largest_target(
                detections,
                self.DOOR_CLASS_IDS if int(target[5]) in self.DOOR_CLASS_IDS else self.WINDOW_CLASS_IDS,
            )
            if refreshed_target is None:
                return False
            target = refreshed_target
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
