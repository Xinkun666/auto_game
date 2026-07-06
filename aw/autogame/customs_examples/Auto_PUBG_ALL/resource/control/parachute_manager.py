import os
import cv2
import time
import subprocess

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import log_step
from typing import Tuple, Dict, Any, Optional, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 假设你的框架类定义在 framework.py 文件中
    from aw.autogame.tools.GameFrameWorker import FrameWorker

class ParachuteManager:
    """
    跳伞流程管理器
    职责：负责从点击开始游戏后，监控跳伞按钮、跟随状态、飞行距离，并执行跳伞和落地动作。
    """

    # --- 配置常量 (Configuration) ---
    TARGET_POS: Tuple[int, int] = (990, 757)  # 默认目标落点
    TRIGGER_DIST: int = 450  # 触发跳伞的距离阈值
    OVERSHOOT_INCREASE_FRAMES: int = 1  # 连续多少帧递增才判定为飞过最佳跳伞点
    DIVE_DURATION_MS: int = 47500  # 俯冲/滑行持续时间 (根据地图大小调整)
    JUMP_CONFIRM_TOLERANCE: int = 35  # 跳伞前后帧允许的小幅测距波动
    JUMP_LOCATION_CONTINUITY_MAX_STEP: int = 120  # 跳伞确认帧之间允许的最大位置跳变
    ROUTE_MISS_CONFIRM_TOLERANCE: int = 35  # 航线错过R城时，后一帧需要明显远离才确认重开
    SUSTAINED_ROUTE_MISS_INCREASE_FRAMES: int = 3  # 错过最近点后，连续递增多少帧才确认重开

    def __init__(self):
        self.is_active = False  # 是否处于监控跳伞距离的激活状态
        self.prior_dist = 0  # 历史最近距离（用于判断是否飞过了）
        self.last_dist: Optional[float] = None  # 上一帧距离（用于判断连续递增）
        self.last_location: Optional[Tuple[int, int]] = None  # 上一帧坐标（用于过滤坐标跳变）
        self.increase_streak = 0  # 连续递增帧数
        self.target_pos: Tuple[int, int] = self.TARGET_POS
        self.landing_stage: str = "搜房阶段"
        self.jump_confirm_distances: List[float] = []
        self.jump_confirm_locations: List[Tuple[int, int]] = []
        self.route_confirm_distances: List[float] = []
        self.route_confirm_locations: List[Tuple[int, int]] = []

    def reset(self):
        """重置跳伞管理器的内部状态"""
        self.is_active = False
        self.prior_dist = 0
        self.last_dist = None
        self.last_location = None
        self.increase_streak = 0
        self.jump_confirm_distances = []
        self.jump_confirm_locations = []
        self.route_confirm_distances = []
        self.route_confirm_locations = []
        print("[Parachute] 状态已重置!")

    def configure(
        self,
        target_pos: Optional[Tuple[int, int]] = None,
        landing_stage: str = "跑图阶段",
        dive_duration_ms: Optional[int] = None,
    ):
        if target_pos is not None:
            self.target_pos = target_pos
        self.landing_stage = landing_stage
        if dive_duration_ms is not None:
            self.DIVE_DURATION_MS = dive_duration_ms
        print(f"[Parachute] 配置更新: target={self.target_pos}, landing_stage={self.landing_stage}")

    def _set_frame_decision(
        self,
        w: 'FrameWorker',
        observation: str,
        decision: str,
        *,
        action: str,
        method: str,
        result: str,
        target: str = "跳伞阶段",
    ):
        log_step(
            f"当前跳伞帧日志：{observation}",
            target=target,
            action=action or decision,
            method=method,
            result=result or decision,
        )
        setter = getattr(w, "set_frame_decision", None)
        if callable(setter):
            setter(
                observation=observation,
                target=target,
                decision=decision,
                action=action,
                method=method,
                result=result,
            )

    def process(self, w: 'FrameWorker'):
        """
        执行跳伞逻辑的主入口
        :return: 状态变更字典 (用于更新 FSM 状态)
        """
        # 1. 如果检测到还在跟随队友，优先取消跟随
        if w.get_info('取消跟随'):
            print('[Parachute] 点击取消跟随!')
            self._set_frame_decision(
                w,
                "当前帧出现取消跟随",
                "点击取消跟随，解除队友跟随后继续判断跳伞",
                action="点击取消跟随",
                method="w.click(取消跟随)",
                result="等待下一帧确认跟随状态解除",
            )
            w.click(w.get_info('取消跟随'))
            time.sleep(1)

        # 2. 尝试激活监控状态 (当看到跳伞按钮且未激活时)
        if not self.is_active and w.get_info('离开'):
            self._set_frame_decision(
                w,
                "当前帧出现离开按钮，说明已进入可跳伞状态",
                "激活航线距离监控",
                action="开始监控R城距离",
                method="_activate_monitoring()",
                result="后续帧根据距离趋势决定跳伞",
            )
            self._activate_monitoring()

        # 3. 如果未激活监控，则无需后续操作
        if not self.is_active:
            return

        location = w.get_info('location')[0]

        # 持续修正飞机上的视角朝向，确保测距准确（假设依赖视角）
        align_direction(w, self.target_pos)

        current_dist = get_distance(location, self.target_pos)
        if not self._is_valid_distance(current_dist):
            print("[Parachute] 当前小地图坐标无效，暂不计算R城距离或触发跳伞")
            self._set_frame_decision(
                w,
                "当前帧小地图坐标无效",
                "暂不跳伞，等待下一帧重新识别坐标",
                action="等待下一帧",
                method="清空跳伞确认缓存",
                result="避免单帧异常导致误跳伞",
            )
            self.jump_confirm_distances = []
            self.jump_confirm_locations = []
            self.route_confirm_distances = []
            self.route_confirm_locations = []
            self.last_dist = None
            self.last_location = None
            return {}

        self._set_frame_decision(
            w,
            f"跳伞距离计算：current_loc={tuple(location)}，target_loc={self.target_pos}，"
            f"current_dist={current_dist:.2f}，trigger_dist={self.TRIGGER_DIST}，"
            f"prior_dist={self.prior_dist:.2f}，last_dist={self.last_dist}",
            "继续根据距离趋势判断是否到达跳伞窗口",
            action="保持跳伞监控",
            method="检查最近距离趋势、路线确认窗口和三帧跳伞窗口",
            result="未确认前不执行跳伞",
        )

        # 4. 距离趋势检查 (判断是否飞过了/飞远了)
        if self._check_flight_path(current_dist, location, w):
            return self._restart_match_for_bad_route(w)


        # 5. 判定是否到达跳伞点：用前后各一帧确认，避免单帧误判导致误跳伞
        if self._confirm_jump_window(current_dist, location):
            return self._perform_jump_sequence(w)

        return {}

    def _activate_monitoring(self):
        """激活跳伞监控模式"""
        self.is_active = True
        print("[Parachute] 检测到跳伞按钮，开始监控航线距离...")

    def _is_valid_distance(self, distance) -> bool:
        return distance is not None and distance >= 0

    def _check_flight_path(self, current_dist: float, location, w: 'FrameWorker') -> bool:
        """
        检查飞行路径状态。
        如果飞机已经越过最近点，且三帧动态窗口确认最近距离仍然大于阈值，则重开下一把。
        """
        self.route_confirm_distances.append(float(current_dist))
        self.route_confirm_locations.append(tuple(location))
        if len(self.route_confirm_distances) > 3:
            self.route_confirm_distances = self.route_confirm_distances[-3:]
            self.route_confirm_locations = self.route_confirm_locations[-3:]

        # 初始化最近距离
        if self.prior_dist == 0:
            self.prior_dist = current_dist
            self.last_dist = current_dist
            self.last_location = tuple(location)
            self.increase_streak = 0
            log_step(
                f"路线确认距离初始化：current_loc={tuple(location)}，current_dist={current_dist:.2f}，"
                f"trigger_dist={self.TRIGGER_DIST}",
                target="当前跳伞分支：路线确认初始化",
                action="记录第一帧最近距离",
                method="_check_flight_path 初始化 prior_dist/last_dist",
                result="等待后续帧判断航线是否靠近或远离目标",
            )
            return False
        else:
            last_dist_text = f"{self.last_dist:.2f}" if self.last_dist is not None else "None"
            print(
                f"[Parachute] 当前距离：{current_dist:.2f}, "
                f"历史最近距离：{self.prior_dist:.2f}, "
                f"上一帧距离：{last_dist_text}, "
                f"连续递增帧数：{self.increase_streak}"
            )

        if self.last_location is not None:
            location_step = get_distance(self.last_location, location)
            if (
                not self._is_valid_distance(location_step)
                or location_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
            ):
                print(
                    f"[Parachute] 当前坐标变化不连续，重置最近点趋势: "
                    f"last_location={self.last_location}, current_location={location}, "
                    f"step={location_step}, max_step={self.JUMP_LOCATION_CONTINUITY_MAX_STEP}"
                )
                self.prior_dist = current_dist
                self.last_dist = current_dist
                self.last_location = tuple(location)
                self.increase_streak = 0
                self.jump_confirm_distances = []
                self.jump_confirm_locations = []
                self.route_confirm_distances = [float(current_dist)]
                self.route_confirm_locations = [tuple(location)]
                log_step(
                    f"路线确认距离被重置：last_location={self.last_location}，"
                    f"current_loc={tuple(location)}，step={location_step}，"
                    f"max_step={self.JUMP_LOCATION_CONTINUITY_MAX_STEP}",
                    target="当前跳伞分支：坐标不连续",
                    action="重置路线和跳伞确认窗口",
                    method="_check_flight_path 坐标连续性过滤",
                    result="避免坐标跳变导致误判航线或误跳伞",
                )
                return False

        # 正常情况：距离在变小，更新最近距离
        if current_dist <= self.prior_dist:
            self.prior_dist = current_dist

        if self.last_dist is not None and current_dist > self.last_dist:
            self.increase_streak += 1
        else:
            self.increase_streak = 0

        self.last_dist = current_dist
        self.last_location = tuple(location)

        if (
            current_dist > self.TRIGGER_DIST
            and self.prior_dist > self.TRIGGER_DIST
            and self.increase_streak >= self.OVERSHOOT_INCREASE_FRAMES
        ):
            print(
                f"[Parachute] 已经过R城最近点，历史最近距离 {self.prior_dist:.2f} "
                f"> 跳伞阈值 {self.TRIGGER_DIST}，当前距离 {current_dist:.2f} "
                f"开始变大，等待动态窗口确认是否需要重开。"
            )
            log_step(
                f"路线确认距离进入重开候选：current_dist={current_dist:.2f}，"
                f"prior_dist={self.prior_dist:.2f}，last_dist={self.last_dist:.2f}，"
                f"increase_streak={self.increase_streak}，trigger_dist={self.TRIGGER_DIST}",
                target="当前跳伞分支：航线错过候选",
                action="等待三帧动态窗口确认是否重开",
                method="_confirm_bad_route_window()",
                result="确认航线最近点仍超过阈值才结束当前局",
            )
            if self._confirm_sustained_bad_route_increase():
                return True
            return self._confirm_bad_route_window()
        return False

    def _confirm_sustained_bad_route_increase(self) -> bool:
        if self.increase_streak < self.SUSTAINED_ROUTE_MISS_INCREASE_FRAMES:
            return False

        recent_distances = self.route_confirm_distances[-self.SUSTAINED_ROUTE_MISS_INCREASE_FRAMES:]
        recent_locations = self.route_confirm_locations[-self.SUSTAINED_ROUTE_MISS_INCREASE_FRAMES:]
        if len(recent_distances) < self.SUSTAINED_ROUTE_MISS_INCREASE_FRAMES:
            return False

        if any(distance <= self.TRIGGER_DIST for distance in recent_distances):
            return False

        if not all(
            next_distance > current_distance
            for current_distance, next_distance in zip(recent_distances, recent_distances[1:])
        ):
            return False

        if not self._confirm_location_continuity(recent_locations):
            return False

        print(
            f"[Parachute] 航线持续远离确认最近点仍超过跳伞阈值: "
            f"closest={self.prior_dist:.2f}, recent={recent_distances}, "
            f"threshold={self.TRIGGER_DIST}"
        )
        log_step(
            f"路线确认距离连续递增通过重开判断：closest={self.prior_dist:.2f}，"
            f"recent={recent_distances}，increase_streak={self.increase_streak}，"
            f"trigger_dist={self.TRIGGER_DIST}",
            target="当前跳伞分支：航线错过持续确认",
            action="放弃本局落点",
            method="_confirm_sustained_bad_route_increase 连续递增窗口",
            result="下一步切结束阶段重开",
        )
        return True

    def _confirm_bad_route_window(self) -> bool:
        if len(self.route_confirm_distances) < 3:
            log_step(
                f"路线确认距离不足三帧：distances={self.route_confirm_distances}",
                target="当前跳伞分支：航线确认等待",
                action="继续观察下一帧距离",
                method="_confirm_bad_route_window 三帧窗口",
                result="暂不重开",
            )
            return False

        prev_dist, candidate_dist, next_dist = self.route_confirm_distances
        if candidate_dist <= self.TRIGGER_DIST:
            return False

        if prev_dist < candidate_dist:
            print(
                f"[Parachute] 航线重开候选帧前一帧未靠近R城: "
                f"prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察"
            )
            return False

        if next_dist <= candidate_dist + self.ROUTE_MISS_CONFIRM_TOLERANCE:
            print(
                f"[Parachute] 航线重开候选帧后一帧未明显远离R城: "
                f"prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察"
            )
            return False

        if not self._confirm_location_continuity(self.route_confirm_locations):
            return False

        print(
            f"[Parachute] 航线动态窗口确认最近点仍超过跳伞阈值: "
            f"prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}, "
            f"threshold={self.TRIGGER_DIST}"
        )
        log_step(
            f"路线确认距离通过重开判断：prev={prev_dist:.2f}，candidate={candidate_dist:.2f}，"
            f"next={next_dist:.2f}，trigger_dist={self.TRIGGER_DIST}",
            target="当前跳伞分支：航线错过确认",
            action="放弃本局落点",
            method="_confirm_bad_route_window 动态窗口",
            result="下一步切结束阶段重开",
        )
        return True

    def _confirm_jump_window(self, current_dist: float, location) -> bool:
        """
        当前帧进入跳伞范围时不立刻跳，等待下一帧后用 [前一帧, 候选帧, 后一帧]
        做距离和坐标连贯性确认。这样可以过滤一帧定位/识别异常导致的距离突降。
        """
        self.jump_confirm_distances.append(float(current_dist))
        self.jump_confirm_locations.append(tuple(location))
        if len(self.jump_confirm_distances) > 3:
            self.jump_confirm_distances = self.jump_confirm_distances[-3:]
            self.jump_confirm_locations = self.jump_confirm_locations[-3:]

        if len(self.jump_confirm_distances) < 3:
            if current_dist <= self.TRIGGER_DIST:
                print(
                    f"[Parachute] 当前距离 {current_dist:.2f} 已到跳伞范围，"
                    "等待下一帧确认是否为连贯变化"
                )
                log_step(
                    f"跳伞开伞确认等待：distances={self.jump_confirm_distances}，"
                    f"locations={self.jump_confirm_locations}",
                    target="当前跳伞分支：跳伞确认等待",
                    action="等待下一帧补齐三帧窗口",
                    method="_confirm_jump_window 三帧窗口",
                    result="暂不点击跳伞",
                )
            return False

        prev_dist, candidate_dist, next_dist = self.jump_confirm_distances
        if candidate_dist > self.TRIGGER_DIST:
            return False

        if candidate_dist > prev_dist + self.JUMP_CONFIRM_TOLERANCE:
            print(
                f"[Parachute] 跳伞候选帧不连贯: prev={prev_dist:.2f}, "
                f"candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察"
            )
            return False

        if next_dist > candidate_dist + self.JUMP_CONFIRM_TOLERANCE:
            print(
                f"[Parachute] 跳伞候选帧后一帧明显反跳: prev={prev_dist:.2f}, "
                f"candidate={candidate_dist:.2f}, next={next_dist:.2f}，判定为单帧误判"
            )
            return False

        if not self._confirm_location_continuity(self.jump_confirm_locations):
            return False

        print(
            f"[Parachute] 跳伞三帧确认通过: prev={prev_dist:.2f}, "
            f"candidate={candidate_dist:.2f}, next={next_dist:.2f}"
        )
        log_step(
            f"跳伞开伞确认通过：prev={prev_dist:.2f}，candidate={candidate_dist:.2f}，"
            f"next={next_dist:.2f}，locations={self.jump_confirm_locations}",
            target="当前跳伞分支：跳伞窗口确认",
            action="执行跳伞和俯冲滑行",
            method="_confirm_jump_window 距离和坐标连续性确认",
            result="下一步点击跳伞并进入落地阶段",
        )
        return True

    def _confirm_location_continuity(self, locations) -> bool:
        if len(locations) < 3:
            return False

        prev_loc, candidate_loc, next_loc = locations
        prev_step = get_distance(prev_loc, candidate_loc)
        next_step = get_distance(candidate_loc, next_loc)
        if not self._is_valid_distance(prev_step) or not self._is_valid_distance(next_step):
            print(
                f"[Parachute] 跳伞确认坐标无效: "
                f"prev={prev_loc}, candidate={candidate_loc}, next={next_loc}"
            )
            return False

        if (
            prev_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
            or next_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
        ):
            print(
                f"[Parachute] 跳伞确认坐标不连续: "
                f"prev={prev_loc}, candidate={candidate_loc}, next={next_loc}, "
                f"prev_step={prev_step:.2f}, next_step={next_step:.2f}, "
                f"max_step={self.JUMP_LOCATION_CONTINUITY_MAX_STEP}"
            )
            return False

        return True

    def _restart_match_for_bad_route(self, w: 'FrameWorker'):
        print("[Parachute] 航线最近点超过阈值，放弃本局落点，进入结束阶段重开下一把")
        self._set_frame_decision(
            w,
            (
                f"动态窗口确认航线最近点仍超过 {self.TRIGGER_DIST}，"
                f"distances={self.route_confirm_distances}"
            ),
            "不跳伞，结束当前局并重开下一把",
            action="切换结束阶段",
            method="w.change_stage(结束阶段)",
            result="结束阶段返回大厅后重新开始下一把",
        )
        self.reset()
        w.change_stage("结束阶段")
        return {"bad_route_restart": True}


    def _perform_jump_sequence(self, w: 'FrameWorker'):
        """
        执行具体的：点击跳伞 -> 俯冲 -> 滑行 -> 落地 -> 切状态
        """
        print(f"[Parachute] 到达跳伞点，执行动作序列...")
        self._set_frame_decision(
            w,
            f"跳伞开伞确认已完成：target_loc={self.target_pos}，"
            f"distances={self.jump_confirm_distances}，dive_duration_ms={self.DIVE_DURATION_MS}",
            "点击跳伞并执行俯冲/滑行，落地后切换阶段",
            action="点击跳伞并压视角/摇杆俯冲",
            method="w.click(跳伞) + 视角/摇杆滑行序列",
            result=f"完成滑行后切换到 {self.landing_stage}",
        )
        w.click('跳伞')

        # 视角向下 (俯冲)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('摇杆', wait=self.DIVE_DURATION_MS, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=200)
        self.reset()
        w.change_stage(self.landing_stage)
