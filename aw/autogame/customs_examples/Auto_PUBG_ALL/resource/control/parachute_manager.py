import os
import cv2
import time
import math
import subprocess

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import *
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
    TRIGGER_DIST: int = 470  # 触发跳伞的距离阈值
    OVERSHOOT_INCREASE_FRAMES: int = 1  # 连续多少帧递增才判定为飞过最佳跳伞点
    DIVE_DURATION_MS: int = 47500  # 俯冲/滑行持续时间 (根据地图大小调整)
    JUMP_CONFIRM_TOLERANCE: int = 35  # 跳伞前后帧允许的小幅测距波动
    JUMP_LOCATION_CONTINUITY_MAX_STEP: int = 120  # 跳伞确认帧之间允许的最大位置跳变
    ROUTE_MISS_CONFIRM_TOLERANCE: int = 35  # 航线错过目标时，后一帧需要明显远离才确认重开
    SUSTAINED_ROUTE_MISS_INCREASE_FRAMES: int = 3  # 错过最近点后，连续递增多少帧才确认重开
    ROUTE_MIN_STEP: float = 1.0  # 两个航线采样点之间至少要有可辨识位移
    PLANNED_DIRECTION_TOLERANCE: int = 5  # 提前对准计划跳伞方向的允许误差
    PLANNED_DIRECTION_MAX_STEPS: int = 2  # 每帧最多执行的方向校准步数

    def __init__(self):
        self._frame_worker = None
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
        self.jump_button_clicked = False
        self.target_candidates: Dict[str, Tuple[int, int]] = {}
        self.dynamic_target_selection = False
        self.route_samples: List[Tuple[int, int]] = []
        self.route_unit: Optional[Tuple[float, float]] = None
        self.selected_target_name: Optional[str] = None
        self.planned_jump_position: Optional[Tuple[int, int]] = None
        self.planned_jump_direction: Optional[int] = None
        self.direction_prepared = False
        self.target_selected_callback = None


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
        self.jump_button_clicked = False
        self.target_candidates = {}
        self.dynamic_target_selection = False
        self.route_samples = []
        self.route_unit = None
        self.selected_target_name = None
        self.planned_jump_position = None
        self.planned_jump_direction = None
        self.direction_prepared = False
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log('[Parachute] 状态已重置')

    def configure(
        self,
        target_pos: Optional[Tuple[int, int]] = None,
        landing_stage: str = "跑图阶段",
        dive_duration_ms: Optional[int] = None,
        target_candidates: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        normalized_candidates = {}
        for name, candidate in (target_candidates or {}).items():
            try:
                normalized_candidates[str(name)] = (
                    int(candidate[0]),
                    int(candidate[1]),
                )
            except (TypeError, ValueError, IndexError):
                continue

        self.target_candidates = normalized_candidates
        self.dynamic_target_selection = bool(normalized_candidates)
        self.route_samples = []
        self.route_unit = None
        self.selected_target_name = None
        self.planned_jump_position = None
        self.planned_jump_direction = None
        self.direction_prepared = False
        if self.dynamic_target_selection:
            self.target_pos = self.TARGET_POS
        elif target_pos is not None:
            self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self.landing_stage = landing_stage
        if dive_duration_ms is not None:
            self.DIVE_DURATION_MS = dive_duration_ms
        if getattr(self, "_frame_worker", None) is not None:
            target_text = (
                f"candidates={self.target_candidates}"
                if self.dynamic_target_selection
                else f"target={self.target_pos}"
            )
            self._frame_worker.frame_log(
                f'[Parachute] 配置更新: {target_text}, landing_stage={self.landing_stage}, '
                f'dive={self.DIVE_DURATION_MS}ms'
            )

    @staticmethod
    def _extract_location(w: 'FrameWorker') -> Optional[Tuple[int, int]]:
        raw_location = w.get_info('location')
        if isinstance(raw_location, (list, tuple)) and len(raw_location) == 1:
            raw_location = raw_location[0]
        try:
            x, y = raw_location[0], raw_location[1]
            if x is None or y is None:
                return None
            return int(round(float(x))), int(round(float(y)))
        except (TypeError, ValueError, IndexError):
            return None

    def _plan_target_on_route(
        self,
        route_origin: Tuple[int, int],
        route_unit: Tuple[float, float],
        target_name: str,
        target_pos: Tuple[int, int],
    ):
        ux, uy = route_unit
        dx = float(target_pos[0] - route_origin[0])
        dy = float(target_pos[1] - route_origin[1])
        along_distance = dx * ux + dy * uy
        target_distance_sq = dx * dx + dy * dy
        cross_distance_sq = max(
            0.0,
            target_distance_sq - along_distance * along_distance,
        )
        cross_distance = math.sqrt(cross_distance_sq)
        if cross_distance > self.TRIGGER_DIST:
            return None, (
                f"垂直距离={cross_distance:.2f} > {self.TRIGGER_DIST}"
            )

        half_chord = math.sqrt(
            max(0.0, self.TRIGGER_DIST ** 2 - cross_distance_sq)
        )
        entry_distance = along_distance - half_chord
        exit_distance = along_distance + half_chord
        if exit_distance < 0:
            return None, (
                f"目标470范围已在飞机后方 exit={exit_distance:.2f}"
            )

        travel_distance = max(0.0, entry_distance)
        jump_x = route_origin[0] + ux * travel_distance
        jump_y = route_origin[1] + uy * travel_distance
        jump_position = (int(round(jump_x)), int(round(jump_y)))
        jump_direction = calculate_angle(jump_position, target_pos)
        if jump_direction is None:
            return None, "无法计算计划跳伞方向"

        return {
            "name": target_name,
            "target": target_pos,
            "jump_position": jump_position,
            "jump_direction": int(jump_direction),
            "travel_distance": travel_distance,
            "cross_distance": cross_distance,
        }, None

    def _select_target_from_route(self, location: Tuple[int, int], w: 'FrameWorker'):
        if not self.route_samples:
            self.route_samples = [location]
            w.frame_log(
                f"[Parachute] 已取得第1个航线点 {location}，等待连续第2个移动点"
            )
            return {}

        previous = self.route_samples[-1]
        step = get_distance(previous, location)
        if (
            not self._is_valid_distance(step)
            or step < self.ROUTE_MIN_STEP
            or step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
        ):
            if self._is_valid_distance(step) and step >= self.ROUTE_MIN_STEP:
                self.route_samples = [location]
            w.frame_log(
                f"[Parachute] 第2个航线点暂不可用: prev={previous}, "
                f"current={location}, step={step:.2f}, "
                f"有效范围={self.ROUTE_MIN_STEP:.1f}-"
                f"{self.JUMP_LOCATION_CONTINUITY_MAX_STEP}；继续取点"
            )
            return {}

        self.route_samples = [previous, location]
        ux = (location[0] - previous[0]) / step
        uy = (location[1] - previous[1]) / step
        self.route_unit = (ux, uy)
        w.frame_log(
            f"[Parachute] 两点确认航线: {previous} -> {location}, "
            f"step={step:.2f}, unit=({ux:.4f},{uy:.4f})"
        )

        plans = []
        for target_name, target_pos in self.target_candidates.items():
            plan, reason = self._plan_target_on_route(
                location,
                self.route_unit,
                target_name,
                target_pos,
            )
            if plan is None:
                w.frame_log(
                    f"[Parachute] 航线目标不可达: {target_name}={target_pos}, "
                    f"原因={reason}"
                )
                continue
            plans.append(plan)
            w.frame_log(
                f"[Parachute] 航线目标可达: {target_name}={target_pos}, "
                f"沿航线还需={plan['travel_distance']:.2f}, "
                f"垂直距离={plan['cross_distance']:.2f}, "
                f"计划跳点={plan['jump_position']}, "
                f"计划方向={plan['jump_direction']}°"
            )

        if not plans:
            return self._restart_match_for_unreachable_targets(w)

        selected = min(
            plans,
            key=lambda item: (
                item["travel_distance"],
                item["cross_distance"],
                item["name"],
            ),
        )
        self.selected_target_name = selected["name"]
        self.target_pos = selected["target"]
        self.planned_jump_position = selected["jump_position"]
        self.planned_jump_direction = selected["jump_direction"]
        self.direction_prepared = False
        w.frame_log(
            f"[Parachute] 选择航线上最先可跳目标: {self.selected_target_name}, "
            f"target={self.target_pos}, jump_at={self.planned_jump_position}, "
            f"direction={self.planned_jump_direction}°, "
            f"along={selected['travel_distance']:.2f}"
        )

        callback = self.target_selected_callback
        if callable(callback):
            try:
                callback(self.selected_target_name, self.target_pos)
            except Exception as exc:
                w.frame_log(
                    f"[Parachute] 通知所选城区失败: {exc}"
                )

        self._prepare_planned_direction(w)
        return {}

    def _prepare_planned_direction(self, w: 'FrameWorker') -> bool:
        if self.planned_jump_direction is None:
            return False
        current_direction = w.get_info('direction')
        try:
            current_direction = float(current_direction)
        except (TypeError, ValueError):
            w.frame_log(
                f"[Parachute] 已选{self.selected_target_name}，但当前方向无效，"
                f"等待对准{self.planned_jump_direction}°"
            )
            return False

        aligned = execute_view_turn(
            w,
            current_direction,
            self.planned_jump_direction,
            threshold=self.PLANNED_DIRECTION_TOLERANCE,
            max_steps=self.PLANNED_DIRECTION_MAX_STEPS,
            wait=500,
            fallback_dura=800,
            log_prefix="[ParachuteTurn]",
        )
        if aligned:
            self.direction_prepared = True
            w.frame_log(
                f"[Parachute] 已提前锁定{self.selected_target_name}计划方向 "
                f"{self.planned_jump_direction}°，等待到达跳点"
            )
        return aligned


    def process(self, w: 'FrameWorker'):
        """
        执行跳伞逻辑的主入口
        :return: 状态变更字典 (用于更新 FSM 状态)
        """
        self._frame_worker = w

        # 1. 如果检测到还在跟随队友，优先取消跟随
        if w.get_info('取消跟随'):
            w.frame_log('[Parachute] 检测到跟随状态，点击取消跟随')
            w.click(w.get_info('取消跟随'))
            time.sleep(1)

        # 2. 尝试激活监控状态 (当看到跳伞按钮且未激活时)
        jump_icon = w.get_info('离开')
        if not self.is_active and jump_icon:
            self._activate_monitoring()

        # 3. 已经看到过跳伞图标，但本模块尚未点击跳伞时图标消失，
        #    说明没有完成一次正确的跳伞，直接退出当前局重新开始。
        if self.is_active and not self.jump_button_clicked and not jump_icon:
            return self._restart_match_for_missing_jump_icon(w)

        # 4. 如果未激活监控，则无需后续操作
        if not self.is_active:
            return

        location = self._extract_location(w)
        if location is None:
            w.frame_log("[Parachute] 小地图坐标无效，等待连续有效航线点")
            return {}

        if self.dynamic_target_selection:
            if self.selected_target_name is None:
                return self._select_target_from_route(location, w)
            if not self.direction_prepared:
                self._prepare_planned_direction(w)
                return {}
        else:
            # 固定落点兼容路径：继续按当前坐标持续对准目标。
            align_direction(w, self.target_pos)

        current_dist = get_distance(location, self.target_pos)
        if not self._is_valid_distance(current_dist):
            w.frame_log("[Parachute] 小地图距离无效，清空确认窗口并等待下一帧")
            self.jump_confirm_distances = []
            self.jump_confirm_locations = []
            self.route_confirm_distances = []
            self.route_confirm_locations = []
            self.last_dist = None
            self.last_location = None
            return {}

        last_dist_text = f"{self.last_dist:.2f}" if self.last_dist is not None else "None"
        w.frame_log(
            f"[ParachuteFrame] loc={tuple(location)}, target={self.target_pos}, "
            f"dist={current_dist:.2f}, threshold={self.TRIGGER_DIST}, "
            f"closest={self.prior_dist:.2f}, last={last_dist_text}, "
            f"away_streak={self.increase_streak}, "
            f"jump_window={len(self.jump_confirm_distances)}/3, "
            f"route_window={len(self.route_confirm_distances)}/3"
        )

        # 5. 距离趋势检查 (判断是否飞过了/飞远了)
        if self._check_flight_path(current_dist, location, w):
            return self._restart_match_for_bad_route(w)

        if (
            self.dynamic_target_selection
            and self.direction_prepared
            and current_dist <= self.TRIGGER_DIST
        ):
            w.frame_log(
                f"[Parachute] 已到{self.selected_target_name}计划跳点附近: "
                f"current={location}, planned={self.planned_jump_position}, "
                f"target_dist={current_dist:.2f} <= {self.TRIGGER_DIST}, "
                f"direction={self.planned_jump_direction}°；立即跳伞"
            )
            return self._perform_jump_sequence(w)

        # 6. 判定是否到达跳伞点：用前后各一帧确认，避免单帧误判导致误跳伞
        if self._confirm_jump_window(current_dist, location, w):
            return self._perform_jump_sequence(w)

        return {}

    def _activate_monitoring(self):
        """激活跳伞监控模式"""
        self.is_active = True
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(
                f'[Parachute] 跳伞监控已激活: target={self.target_pos}, threshold={self.TRIGGER_DIST}'
            )

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
            return False

        if self.last_location is not None:
            location_step = get_distance(self.last_location, location)
            if (
                not self._is_valid_distance(location_step)
                or location_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
            ):
                w.frame_log(
                    f"[Parachute] 坐标跳变，重置航线趋势: last={self.last_location}, "
                    f"current={tuple(location)}, step={location_step}, "
                    f"limit={self.JUMP_LOCATION_CONTINUITY_MAX_STEP}"
                )
                self.prior_dist = current_dist
                self.last_dist = current_dist
                self.last_location = tuple(location)
                self.increase_streak = 0
                self.jump_confirm_distances = []
                self.jump_confirm_locations = []
                self.route_confirm_distances = [float(current_dist)]
                self.route_confirm_locations = [tuple(location)]
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
            w.frame_log(
                f"[Parachute] 航线错过候选: closest={self.prior_dist:.2f}, "
                f"current={current_dist:.2f}, threshold={self.TRIGGER_DIST}, "
                f"away_streak={self.increase_streak}"
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

        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[Parachute] 航线持续远离确认最近点仍超过跳伞阈值: closest={self.prior_dist:.2f}, recent={recent_distances}, threshold={self.TRIGGER_DIST}')
        return True

    def _confirm_bad_route_window(self) -> bool:
        if len(self.route_confirm_distances) < 3:
            return False

        prev_dist, candidate_dist, next_dist = self.route_confirm_distances
        if candidate_dist <= self.TRIGGER_DIST:
            return False

        if prev_dist < candidate_dist:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Parachute] 航线重开候选帧前一帧未靠近目标: prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察')
            return False

        if next_dist <= candidate_dist + self.ROUTE_MISS_CONFIRM_TOLERANCE:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Parachute] 航线重开候选帧后一帧未明显远离目标: prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察')
            return False

        if not self._confirm_location_continuity(self.route_confirm_locations):
            return False

        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[Parachute] 航线动态窗口确认最近点仍超过跳伞阈值: prev={prev_dist:.2f}, candidate={candidate_dist:.2f}, next={next_dist:.2f}, threshold={self.TRIGGER_DIST}')
        return True

    def _confirm_jump_window(self, current_dist: float, location, w: 'FrameWorker') -> bool:
        """
        当前帧从阈值外连续进入跳伞范围时立刻跳伞。
        如果没有可用的前一帧连续性证据，再回退到 [前一帧, 候选帧, 后一帧]
        做距离和坐标连贯性确认，过滤一帧定位/识别异常导致的距离突降。
        """
        self.jump_confirm_distances.append(float(current_dist))
        self.jump_confirm_locations.append(tuple(location))
        if len(self.jump_confirm_distances) > 3:
            self.jump_confirm_distances = self.jump_confirm_distances[-3:]
            self.jump_confirm_locations = self.jump_confirm_locations[-3:]

        if (
            len(self.jump_confirm_distances) == 2
            and self.jump_confirm_distances[0] > self.TRIGGER_DIST
            and current_dist <= self.TRIGGER_DIST
        ):
            prev_loc, current_loc = self.jump_confirm_locations
            location_step = get_distance(prev_loc, current_loc)
            if (
                self._is_valid_distance(location_step)
                and location_step <= self.JUMP_LOCATION_CONTINUITY_MAX_STEP
            ):
                w.frame_log(
                    f"[Parachute] 连续进入跳伞范围: prev={self.jump_confirm_distances[0]:.2f}, "
                    f"current={current_dist:.2f}, step={location_step:.2f}, "
                    f"threshold={self.TRIGGER_DIST}"
                )
                return True


        if len(self.jump_confirm_distances) < 3:
            if current_dist <= self.TRIGGER_DIST:
                w.frame_log(
                    f"[Parachute] 已进入跳伞范围 dist={current_dist:.2f}，等待连续性确认"
                )
            return False

        prev_dist, candidate_dist, next_dist = self.jump_confirm_distances
        if candidate_dist > self.TRIGGER_DIST:
            return False

        if candidate_dist > prev_dist + self.JUMP_CONFIRM_TOLERANCE:
            w.frame_log(
                f"[Parachute] 跳伞候选帧不连贯: prev={prev_dist:.2f}, "
                f"candidate={candidate_dist:.2f}, next={next_dist:.2f}，继续观察"
            )
            return False

        if next_dist > candidate_dist + self.JUMP_CONFIRM_TOLERANCE:
            w.frame_log(
                f"[Parachute] 跳伞候选帧后一帧明显反跳: prev={prev_dist:.2f}, "
                f"candidate={candidate_dist:.2f}, next={next_dist:.2f}，判定为单帧误判"
            )
            return False

        if not self._confirm_location_continuity(self.jump_confirm_locations):
            return False

        w.frame_log(
            f"[Parachute] 三帧跳伞确认通过: "
            f"{prev_dist:.2f} -> {candidate_dist:.2f} -> {next_dist:.2f}"
        )
        return True

    def _confirm_location_continuity(self, locations) -> bool:
        if len(locations) < 3:
            return False

        prev_loc, candidate_loc, next_loc = locations
        prev_step = get_distance(prev_loc, candidate_loc)
        next_step = get_distance(candidate_loc, next_loc)
        if not self._is_valid_distance(prev_step) or not self._is_valid_distance(next_step):
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Parachute] 跳伞确认坐标无效: prev={prev_loc}, candidate={candidate_loc}, next={next_loc}')
            return False

        if (
            prev_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
            or next_step > self.JUMP_LOCATION_CONTINUITY_MAX_STEP
        ):
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Parachute] 跳伞确认坐标不连续: prev={prev_loc}, candidate={candidate_loc}, next={next_loc}, prev_step={prev_step:.2f}, next_step={next_step:.2f}, max_step={self.JUMP_LOCATION_CONTINUITY_MAX_STEP}')
            return False

        return True

    def _restart_match_for_bad_route(self, w: 'FrameWorker'):
        w.frame_log(
            f"[Parachute] 航线确认错过目标 {self.selected_target_name or self.target_pos}: "
            f"closest={self.prior_dist:.2f}, "
            f"threshold={self.TRIGGER_DIST}, recent={self.route_confirm_distances}；切换结束阶段"
        )
        self.reset()
        w.change_stage("结束阶段")
        return {"bad_route_restart": True}

    def _restart_match_for_unreachable_targets(self, w: 'FrameWorker'):
        w.frame_log(
            f"[Parachute] 两点航线确认后所有目标均不可达: "
            f"route={self.route_samples}, candidates={self.target_candidates}, "
            f"threshold={self.TRIGGER_DIST}；切换结束阶段开始下一把"
        )
        self.reset()
        w.change_stage("结束阶段")
        return {"unreachable_route_restart": True}

    def _restart_match_for_missing_jump_icon(self, w: 'FrameWorker'):
        w.frame_log(
            "[Parachute] 尚未点击跳伞时离开图标消失，切换结束阶段重开"
        )
        self.reset()
        w.change_stage("结束阶段")
        return {"missing_jump_icon_restart": True}


    def _perform_jump_sequence(self, w: 'FrameWorker'):
        """
        执行具体的：点击跳伞 -> 俯冲 -> 滑行 -> 落地 -> 切状态
        """
        w.frame_log(
            f"[Parachute] 执行跳伞: target={self.target_pos}, "
            f"window={self.jump_confirm_distances}, dive={self.DIVE_DURATION_MS}ms, "
            f"landing_stage={self.landing_stage}"
        )
        self.jump_button_clicked = True
        w.click('跳伞')

        # 视角向下 (俯冲)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('摇杆', wait=self.DIVE_DURATION_MS, dura=400, x_bias=0, y_bias=-500)
        w.tap_single('视角', wait=100, dura=400, x_bias=0, y_bias=200)
        self.reset()
        w.change_stage(self.landing_stage)
