import os
import cv2
import time
import subprocess

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import *
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
    TARGET_POS: Tuple[int, int] = (966, 755)  # 默认目标落点
    TRIGGER_DIST: int = 450  # 触发跳伞的距离阈值
    SAFE_BUFFER: int = 550  # 有效靠近的判定缓冲值
    DIVE_DURATION_MS: int = 47500  # 俯冲/滑行持续时间 (根据地图大小调整)

    def __init__(self):
        self.is_active = False  # 是否处于监控跳伞距离的激活状态
        self.prior_dist = 0  # 历史最近距离（用于判断是否飞过了）
        self.target_pos: Tuple[int, int] = self.TARGET_POS
        self.landing_stage: str = "搜房阶段"

    def reset(self):
        """重置跳伞管理器的内部状态"""
        self.is_active = False
        self.prior_dist = 0
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

        # 4. 距离趋势检查 (判断是否飞过了/飞远了)
        self._check_flight_path(current_dist, w)


        # 5. 判定是否到达跳伞点
        if current_dist <= self.TRIGGER_DIST and self.prior_dist <= self.TRIGGER_DIST + 50:
            # 安全检查：确保我们要么是初次测距，要么是一直在靠近
            if self.prior_dist < self.SAFE_BUFFER:
                return self._perform_jump_sequence(w)

        return {}

    def _activate_monitoring(self):
        """激活跳伞监控模式"""
        self.is_active = True
        print("[Parachute] 检测到跳伞按钮，开始监控航线距离...")

    def _check_flight_path(self, current_dist: float, w: 'FrameWorker'):
        """
        检查飞行路径状态。
        如果飞机正在远离目标且距离过远，判定为本局航线不佳或死亡。
        """
        # 初始化最近距离
        if self.prior_dist == 0:
            self.prior_dist = current_dist
            return
        else:
            print(f'[Parachute] 当前距离跳伞点距离：{current_dist:.2f}, 上一次距离跳伞点距离：{self.prior_dist:.2f}')

        # 正常情况：距离在变小，更新最近距离
        if current_dist <= self.prior_dist:
            self.prior_dist = current_dist

        if current_dist > self.prior_dist and self.prior_dist > self.TRIGGER_DIST:
            print(f'[Parachute] 警告：航线偏离。最近距离 {self.prior_dist} > 阈值 {self.TRIGGER_DIST}')
            self.reset()
            w.change_stage('结束阶段')


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
