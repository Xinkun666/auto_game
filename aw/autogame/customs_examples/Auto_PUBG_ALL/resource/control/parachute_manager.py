import os
import cv2
import time
import subprocess

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print
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
    TRIGGER_DIST: int = 550  # 触发跳伞的距离阈值
    SAFE_BUFFER: int = 550  # 有效靠近的判定缓冲值
    OVERSHOOT_INCREASE_FRAMES: int = 5  # 连续多少帧递增才判定为飞过最佳跳伞点
    DIVE_DURATION_MS: int = 47500  # 俯冲/滑行持续时间 (根据地图大小调整)
    JUMP_CONFIRM_TOLERANCE: int = 35  # 跳伞前后帧允许的小幅测距波动

    def __init__(self):
        self.is_active = False  # 是否处于监控跳伞距离的激活状态
        self.prior_dist = 0  # 历史最近距离（用于判断是否飞过了）
        self.last_dist: Optional[float] = None  # 上一帧距离（用于判断连续递增）
        self.increase_streak = 0  # 连续递增帧数
        self.target_pos: Tuple[int, int] = self.TARGET_POS
        self.landing_stage: str = "搜房阶段"
        self.jump_confirm_distances: List[float] = []

    def reset(self):
        """重置跳伞管理器的内部状态"""
        self.is_active = False
        self.prior_dist = 0
        self.last_dist = None
        self.increase_streak = 0
        self.jump_confirm_distances = []
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

    def process(self, w: 'FrameWorker'):
        """
        执行跳伞逻辑的主入口
        :return: 状态变更字典 (用于更新 FSM 状态)
        """
        # 1. 如果检测到还在跟随队友，优先取消跟随
        if w.get_info('取消跟随'):
            print('[Parachute] 点击取消跟随!')
            w.click(w.get_info('取消跟随'))
            time.sleep(1)

        # 2. 尝试激活监控状态 (当看到跳伞按钮且未激活时)
        if not self.is_active and w.get_info('离开'):
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
            self.jump_confirm_distances = []
            self.last_dist = None
            return {}

        # 4. 距离趋势检查 (判断是否飞过了/飞远了)
        self._check_flight_path(current_dist, w)


        # 5. 判定是否到达跳伞点：用前后各一帧确认，避免单帧误判导致误跳伞
        if self._confirm_jump_window(current_dist):
            return self._perform_jump_sequence(w)

        return {}

    def _activate_monitoring(self):
        """激活跳伞监控模式"""
        self.is_active = True
        print("[Parachute] 检测到跳伞按钮，开始监控航线距离...")

    def _is_valid_distance(self, distance) -> bool:
        return distance is not None and distance >= 0

    def _check_flight_path(self, current_dist: float, w: 'FrameWorker'):
        """
        检查飞行路径状态。
        如果飞机正在远离目标且距离过远，判定为本局航线不佳或死亡。
        """
        # 初始化最近距离
        if self.prior_dist == 0:
            self.prior_dist = current_dist
            self.last_dist = current_dist
            self.increase_streak = 0
            return
        else:
            last_dist_text = f"{self.last_dist:.2f}" if self.last_dist is not None else "None"
            print(
                f"[Parachute] 当前距离：{current_dist:.2f}, "
                f"历史最近距离：{self.prior_dist:.2f}, "
                f"上一帧距离：{last_dist_text}, "
                f"连续递增帧数：{self.increase_streak}"
            )

        # 正常情况：距离在变小，更新最近距离
        if current_dist <= self.prior_dist:
            self.prior_dist = current_dist

        if self.last_dist is not None and current_dist > self.last_dist:
            self.increase_streak += 1
        else:
            self.increase_streak = 0

        self.last_dist = current_dist

        if (
            current_dist > self.TRIGGER_DIST
            and self.prior_dist > self.TRIGGER_DIST
            and self.increase_streak >= self.OVERSHOOT_INCREASE_FRAMES
        ):
            print(
                f"[Parachute] 警告：航线偏离。历史最近距离 {self.prior_dist:.2f} > 阈值 {self.TRIGGER_DIST}，"
                f"且已连续 {self.increase_streak} 帧远离目标，判定飞过最佳跳伞点。"
            )
            self.reset()
            w.change_stage('结束阶段')

    def _confirm_jump_window(self, current_dist: float) -> bool:
        """
        当前帧进入跳伞范围时不立刻跳，等待下一帧后用 [前一帧, 候选帧, 后一帧]
        做连贯性确认。这样可以过滤一帧定位/识别异常导致的距离突降。
        """
        self.jump_confirm_distances.append(float(current_dist))
        if len(self.jump_confirm_distances) > 3:
            self.jump_confirm_distances = self.jump_confirm_distances[-3:]

        if len(self.jump_confirm_distances) < 3:
            if current_dist <= self.TRIGGER_DIST:
                print(
                    f"[Parachute] 当前距离 {current_dist:.2f} 已到跳伞范围，"
                    "等待下一帧确认是否为连贯变化"
                )
            return False

        prev_dist, candidate_dist, next_dist = self.jump_confirm_distances
        if candidate_dist > self.TRIGGER_DIST:
            return False

        if min(prev_dist, candidate_dist, next_dist) >= self.SAFE_BUFFER:
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

        print(
            f"[Parachute] 跳伞三帧确认通过: prev={prev_dist:.2f}, "
            f"candidate={candidate_dist:.2f}, next={next_dist:.2f}"
        )
        return True


    def _perform_jump_sequence(self, w: 'FrameWorker'):
        """
        执行具体的：点击跳伞 -> 俯冲 -> 滑行 -> 落地 -> 切状态
        """
        print(f"[Parachute] 到达跳伞点，执行动作序列...")
        w.click('跳伞')

        # 视角向下 (俯冲)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('摇杆', wait=self.DIVE_DURATION_MS, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=200)
        self.reset()
        w.change_stage(self.landing_stage)
