import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.map_navigator import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.obstacle_analyzer import (
    ObstacleAvoidanceAnalyzer,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import (
    calculate_angle,
    calculate_move_count,
    check_location,
    draw_points_with_arrows,
    get_distance,
    is_location_stagnant,
)

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


@dataclass
class DriveContext:
    location: Tuple[int, int]
    direction: float
    speed: Optional[int]
    decision: str
    obstacle_info: Dict[str, Any]


class DrivingManager:
    # 首次出库必须对齐到的固定角度
    TARGET_FIRST_DIR = 115
    FIRST_ALIGN_TOLERANCE = 1
    ESCAPE_ALIGN_TOLERANCE = 8

    # 车辆方向与目标方向的允许误差；小于该值时直接视为已对齐
    ALIGN_THRESHOLD = 8
    # 到路径点多近时认为已经到达该路径点
    WAYPOINT_TOLERANCE = 5.0

    # 连续多少帧位置几乎不变时，判定车辆被卡住
    STUCK_REPEAT_LIMIT = 7
    # 两次位置变化小于该距离时，视为基本没动
    STUCK_LOCATION_EPS = 0.2
    # 困死判定窗口：连续多少帧都在局部很小范围打转才算真正困死
    TRAPPED_HISTORY_LEN = 80
    # 困死判定时，轨迹围绕中心点打转的半径范围
    TRAPPED_RADIUS_MIN = 1.0
    TRAPPED_RADIUS_MAX = 2.0
    # 已经进入不可通行区域后，连续脱困失败多少次就直接结束当前局
    FORBIDDEN_ESCAPE_FAIL_LIMIT = 8
    # 前方黑区检测距离：基础值，以及速度 2/3 时更远的提前量
    FORBIDDEN_BASE_CHECK_DISTANCE = 10
    FORBIDDEN_SPEED2_CHECK_DISTANCE = 14
    FORBIDDEN_SPEED3_CHECK_DISTANCE = 20
    # 高速接近黑区时，额外预刹车的持续时间
    FORBIDDEN_HIGH_SPEED_BRAKE_WAIT = 1400

    # 速度 2 时自动刹车的冷却时间，避免频繁点刹
    BRAKE_COOLDOWN_SPEED2 = 2.0

    # 驾驶结束时，停车和下车前的操作时长
    TIME_BRAKE = 3000
    TIME_OFF_CAR = 500

    # 不同时期的进圈目标距离，单位是地图坐标距离
    STAGE1_DIS = 600
    STAGE2_DIS = 400
    STAGE3_DIS = 220
    # 上面三段进圈距离各自对应的时间分界点，单位秒
    STAGE1_TIME = 11 * 60
    STAGE2_TIME = 16 * 60

    # 默认整段开车阶段的最大时长，单位秒
    DEFAULT_MAX_DRIVING_TIME = 10 * 60
    # 车前方障碍物识别使用的画面 ROI 区域
    DRIVE_VIEW_RECT = (0.2797, 0.2826, 0.7203, 0.6175)

    # 内部阶段名称：首次出库 / 正常巡航 / 驾驶结束
    STAGE_EXIT_GARAGE = "出库阶段"
    STAGE_CRUISE = "巡航阶段"
    STAGE_FINISH = "驾驶结束阶段"

    # 开车阶段按钮名称兼容表，用于适配不同标注名称
    CONTROL_ALIASES = {
        "up": ("油门", "前进", "up"),
        "down": ("倒车", "后退", "down"),
        "left": ("向左", "左转", "left"),
        "right": ("向右", "右转", "right"),
        "brake": ("急刹", "刹车", "制动", "brake"),
        "auto_forward": ("自动前进", "auto_forward"),
        "off_car": ("下车", "离开载具", "off_car"),
        "spectate": ("观战", "guanzhan"),
    }

    def __init__(
        self,
        map_tool: Optional[MapNavigator] = None,
        max_driving_time: int = DEFAULT_MAX_DRIVING_TIME,
    ):
        self.map_tool = map_tool or MapNavigator()
        self.max_driving_time = max_driving_time
        self.obstacle_analyzer: Optional[ObstacleAvoidanceAnalyzer] = None

        self.game_time: Optional[float] = None
        self.driving_start_time: Optional[float] = None

        self.is_first_car = True
        self.current_stage = self.STAGE_EXIT_GARAGE
        self.exit_garage_warmup_done = False

        self.circle_angles: List[float] = []
        self.stable_circle_angle: Optional[float] = None
        self.last_planned_circle_angle: Optional[float] = None
        self.road_list: List[Tuple[int, int]] = []

        self.prior_angle: Optional[float] = None
        self.prior_location: Optional[Tuple[int, int]] = None

        self.history_locations: List[Tuple[int, int]] = []
        self.trapped = False
        self.forbidden_escape_failures = 0

        self.last_motion_mode: Optional[str] = None
        self.last_motion_steer: Optional[str] = None
        self.last_motion_location: Optional[Tuple[int, int]] = None
        self.last_motion_started_at = 0.0
        self.blocked_motion_count = 0
        self.last_auto_brake_time = 0.0
        self.allow_running_fallback = True

        self._frame_action_executed = False

    def set_game_time(self, game_time: Optional[float] = None):
        self.game_time = time.time() if game_time is None else game_time
        print(f"[Driving] 游戏开始时间设置为：{self.game_time:.3f}")

    def get_elapsed_time(self) -> float:
        if self.game_time is None:
            return 0.0
        return max(0.0, time.time() - self.game_time)

    def get_driving_elapsed_time(self) -> float:
        if self.driving_start_time is None:
            return 0.0
        return max(0.0, time.time() - self.driving_start_time)

    def set_remaining_drive_time(self, remaining_time: Optional[float]):
        if remaining_time is None:
            return
        self.max_driving_time = max(0.0, float(remaining_time))
        print(f"[Driving] 剩余开车时长设置为：{self.max_driving_time:.2f}s")

    def reset(self, max_driving_time: Optional[int] = None):
        if max_driving_time is not None:
            self.max_driving_time = max_driving_time

        self.driving_start_time = None
        self.is_first_car = True
        self.current_stage = self.STAGE_EXIT_GARAGE
        self.exit_garage_warmup_done = False

        self.circle_angles = []
        self.stable_circle_angle = None
        self.last_planned_circle_angle = None
        self.road_list = []

        self.prior_angle = None
        self.prior_location = None

        self.history_locations = []
        self.trapped = False
        self.forbidden_escape_failures = 0

        self.last_motion_mode = None
        self.last_motion_steer = None
        self.last_motion_location = None
        self.last_motion_started_at = 0.0
        self.blocked_motion_count = 0
        self.last_auto_brake_time = 0.0
        self.allow_running_fallback = True

        self._frame_action_executed = False
        print("[Driving] 状态已重置!")

    def set_running_fallback_enabled(self, enabled: bool):
        self.allow_running_fallback = bool(enabled)

    def process(self, w: "FrameWorker"):
        self._begin_frame()

        if self.driving_start_time is None:
            self.driving_start_time = time.time()
            print(f"[Driving] 本次驾驶计时开始：{self.driving_start_time:.3f}")

        if self._has_rank_info(w):
            print("[Driving] 检测到个人排名或队伍排名，进入结束阶段")
            self._handle_rank_finish(w)
            self._finalize_frame(w)
            return

        if self._is_dead(w):
            print("[Driving] 检测到死亡，结束当前局")
            self._handle_death(w)
            self._finalize_frame(w)
            return

        context = self._build_context(w)
        if context is None:
            print("[Driving] 当前位置或朝向无效，等待下一帧")
            self._finalize_frame(w)
            return

        if self.game_time is None:
            self.set_game_time()

        self._update_circle_angle(w.get_info("white_angle"))
        self._update_trapped_state(context.location)

        if self.current_stage == self.STAGE_EXIT_GARAGE:
            print("[Driving] 当前阶段: 出库阶段")
            self._handle_first_car_alignment(w, context.direction)
            self._finalize_frame(w)
            return

        if self.get_driving_elapsed_time() >= self.max_driving_time:
            print("[Driving] 驾驶时长已到，结束驾驶阶段")
            self._run_stage_finish(w)
            self._finalize_frame(w)
            return

        if self.trapped:
            print("[Driving] 车辆长时间在 1~2 距离范围内打转，判定困死")
            if self.allow_running_fallback:
                self._exit_vehicle_to_running(w, "车辆困死，切回跑图阶段")
            else:
                self._handle_death(w)
            self._finalize_frame(w)
            return

        self.current_stage = self.STAGE_CRUISE
        target_direction = self._resolve_target_direction(context.location)

        if self._handle_forbidden_escape(w, context):
            self._finalize_frame(w)
            return

        if self._check_motion_block(context):
            print("[Driving] 检测到连续7帧位置不变，执行倒车避障")
            self._handle_motion_block(w, context)
            self._finalize_frame(w)
            return

        if self._handle_forbidden_ahead(w, context):
            self._finalize_frame(w)
            return

        if self._handle_visual_avoidance(w, context):
            self._finalize_frame(w)
            return

        self._drive_toward_target(w, context, target_direction)
        self._finalize_frame(w)

    def _log_drive_state(
        self,
        situation: str,
        context: DriveContext,
        decision: str,
        target_direction: Optional[float] = None,
    ):
        circle_text = (
            f"{self.stable_circle_angle:.1f}"
            if self.stable_circle_angle is not None
            else "None"
        )
        target_text = (
            f"{target_direction:.1f}"
            if target_direction is not None
            else "None"
        )
        speed_text = "None" if context.speed is None else str(context.speed)
        obstacle_count = int(context.obstacle_info.get("obstacles_count", 0) or 0)
        print(
            f"[情况:{situation}] "
            f"[状态: speed={speed_text}, loc={context.location}, dir={context.direction:.1f}, "
            f"circle={circle_text}, target={target_text}, obstacle_count={obstacle_count}, "
            f"vision={context.decision}] "
            f"[决策:{decision}]"
        )

    def _begin_frame(self):
        self._frame_action_executed = False

    def _finalize_frame(self, w: "FrameWorker"):
        if self._frame_action_executed:
            w.refresh_frame()

    def _build_context(self, w: "FrameWorker") -> Optional[DriveContext]:
        location = self._use_valid_location(self._get_location(w))
        direction = self._use_valid_direction(self._get_scalar(w.get_info("direction")))
        if location is None or direction is None:
            return None

        obstacle_info = self._analyze_obstacles(w)
        speed = self._get_speed(w)
        return DriveContext(
            location=location,
            direction=direction,
            speed=speed,
            decision=obstacle_info.get("decision", "straight"),
            obstacle_info=obstacle_info,
        )

    def _resolve_target_direction(self, location: Tuple[int, int]) -> Optional[float]:
        if self.stable_circle_angle is None:
            self.road_list = []
            self.last_planned_circle_angle = None
            return None

        if self._should_plan_route():
            self._load_path(location)

        self._consume_waypoints(location)
        if self.road_list:
            target = self.road_list[0]
            target_direction = calculate_angle(location, target)
            print(
                f"[Driving] 路径巡航: loc={location}, target={target}, "
                f"target_direction={target_direction}"
            )
            return target_direction

        print(f"[Driving] 自由巡航: stable_circle_angle={self.stable_circle_angle}")
        return self.stable_circle_angle

    def _consume_waypoints(self, location: Tuple[int, int]):
        while self.road_list:
            dist = get_distance(location, self.road_list[0])
            if dist > self.WAYPOINT_TOLERANCE:
                break
            reached = self.road_list.pop(0)
            print(f"[Driving] 已到达路径点: {reached}")

    def _should_plan_route(self) -> bool:
        if self.stable_circle_angle is None:
            return False
        if not self.road_list:
            return True
        if self.last_planned_circle_angle is None:
            return True
        return self._angle_diff(self.last_planned_circle_angle, self.stable_circle_angle) >= 10

    def _load_path(self, location: Tuple[int, int]):
        target_dist = self._get_circle_distance()
        target_point = self.map_tool.get_target_point(location, self.stable_circle_angle, target_dist)
        planned_path = self.map_tool.plan_path(location, target_point) if target_point is not None else []
        self.road_list = [tuple(map(int, point)) for point in planned_path if point is not None]
        self.last_planned_circle_angle = self.stable_circle_angle

        if self.road_list:
            print(f"[Driving] 路径已加载: {self.road_list}")
            try:
                draw_points_with_arrows(self.road_list)
            except Exception as exc:
                print(f"[Driving] 绘制路径调试图失败: {exc}")
        else:
            print("[Driving] 路径规划失败，回退自由巡航")

    def _handle_first_car_alignment(self, w: "FrameWorker", direction: float):
        if not self.exit_garage_warmup_done:
            print("[Driving] 首次出库先前探 700ms")
            self._tap_single_control(w, "up", wait=700, dura=100)
            self.exit_garage_warmup_done = True
            return

        turn_dir, _, diff = calculate_move_count(direction, self.TARGET_FIRST_DIR)
        if turn_dir is not None and diff > self.FIRST_ALIGN_TOLERANCE:
            duration = self._get_exit_alignment_duration(diff)
            if diff <= 12:
                action = "brake_turn_left" if turn_dir == "left" else "brake_turn_right"
                print(
                    f"[Driving] 出库精调角度: current={direction:.1f}, "
                    f"target={self.TARGET_FIRST_DIR}, diff={diff:.2f}, action={action}, duration={duration}"
                )
                self._execute_maneuver(w, action, duration=duration, brake_with_steer=False)
            else:
                action = "forward_turn_left" if turn_dir == "left" else "forward_turn_right"
                print(
                    f"[Driving] 出库转向修正: current={direction:.1f}, "
                    f"target={self.TARGET_FIRST_DIR}, diff={diff:.2f}, action={action}, duration={duration}"
                )
                self._execute_maneuver(w, action, duration=duration, brake_with_steer=True)
            return

        self._tap_single_control(w, "up", wait=5000, dura=100)
        self.is_first_car = False
        self.current_stage = self.STAGE_CRUISE
        self.exit_garage_warmup_done = False
        self.last_motion_location = None
        self.blocked_motion_count = 0
        print("[Driving] 出库完成，进入巡航阶段")

    def _handle_forbidden_escape(self, w: "FrameWorker", context: DriveContext) -> bool:
        if self.map_tool.is_walkable(context.location):
            self.forbidden_escape_failures = 0
            return False

        safe_point = self.map_tool.get_nearest_safe_point(context.location, max_search_dist=120)
        self.forbidden_escape_failures += 1
        print(f"[Driving] 已进入不可通行区域，执行角度脱离 attempt={self.forbidden_escape_failures}")

        if self.forbidden_escape_failures >= self.FORBIDDEN_ESCAPE_FAIL_LIMIT:
            print("[Driving] 不可通行区域脱困多次失败，结束当前局")
            self._handle_death(w)
            return True

        if safe_point is None:
            print("[Driving] 黑区内未找到安全点，先执行保守倒车脱离")
            self._log_drive_state("已进入不可通行区域", context, "backward(900ms)", self.stable_circle_angle)
            self._tap_single_control(w, "down", wait=900, dura=100)
            return True

        target_direction = calculate_angle(context.location, safe_point)
        turn_dir, _, diff = calculate_move_count(context.direction, target_direction)
        use_backward = diff > 90
        duration = self._get_turn_duration(diff, fine=False, reverse=use_backward)

        if use_backward:
            if turn_dir is None or diff <= self.ESCAPE_ALIGN_TOLERANCE:
                action_text = "backward(900ms)"
                print(
                    f"[Driving] 黑区脱离: safe_point={safe_point}, target={target_direction:.1f}, "
                    f"diff={diff:.2f}, action={action_text}"
                )
                self._log_drive_state("已进入不可通行区域", context, action_text, target_direction)
                self._tap_single_control(w, "down", wait=900, dura=100)
                return True

            action = f"backward_turn_{turn_dir}"
        else:
            if turn_dir is None or diff <= self.ESCAPE_ALIGN_TOLERANCE:
                action_text = "forward(700ms)"
                print(
                    f"[Driving] 黑区脱离: safe_point={safe_point}, target={target_direction:.1f}, "
                    f"diff={diff:.2f}, action={action_text}"
                )
                self._log_drive_state("已进入不可通行区域", context, action_text, target_direction)
                self._tap_single_control(w, "up", wait=700, dura=100)
                return True

            action = f"forward_turn_{turn_dir}"

        action_text = f"{action}({duration}ms)"
        self._log_drive_state(
            "已进入不可通行区域",
            context,
            action_text,
            target_direction,
        )
        print(
            f"[Driving] 黑区脱离: safe_point={safe_point}, target={target_direction:.1f}, "
            f"diff={diff:.2f}, action={action}, duration={duration}, mode={'backward' if use_backward else 'forward'}"
        )
        self._execute_maneuver(w, action, speed=context.speed, duration=duration, brake_with_steer=True)
        return True

    def _handle_forbidden_ahead(self, w: "FrameWorker", context: DriveContext) -> bool:
        if not self._has_forbidden_ahead(context):
            return False

        sector = self.map_tool.get_avoidance_action(context.location, context.direction)
        action_map = {
            1: ("forward_turn_right", 300),
            2: ("forward_turn_right", 700),
            3: ("backward_turn_right", 2000),
            4: ("forward_turn_left", 300),
            5: ("forward_turn_left", 700),
            6: ("backward_turn_left", 2000),
        }
        action = action_map.get(sector)
        if action is None:
            return False

        duration = action[1]
        if context.speed is not None and context.speed >= 3:
            print("[Driving] 高速接近不可通行区域，先执行预刹车")
            self._tap_single_control(w, "brake", wait=self.FORBIDDEN_HIGH_SPEED_BRAKE_WAIT)
            if action[0].startswith("forward_turn_"):
                duration += 250
            elif action[0].startswith("backward_turn_"):
                duration += 400

        print(f"[Driving] 前方不可通行，地图规避 sector={sector}, action={action}")
        self._log_drive_state(
            "前方有不可通行区域",
            context,
            f"{action[0]}({duration}ms)",
            self.stable_circle_angle,
        )
        self._execute_maneuver(
            w,
            action=action[0],
            speed=context.speed,
            duration=duration,
            brake_with_steer=True,
        )
        return True

    def _handle_visual_avoidance(self, w: "FrameWorker", context: DriveContext) -> bool:
        decision = context.decision or "straight"
        if decision == "straight":
            return False

        action_map = {
            "slight_left": ("forward_turn_left", 250),
            "small_left": ("forward_turn_left", 450),
            "large_left": ("forward_turn_left", 700),
            "slight_right": ("forward_turn_right", 250),
            "small_right": ("forward_turn_right", 450),
            "large_right": ("forward_turn_right", 700),
            "reverse_and_left": ("reverse_and_left", 1000),
            "reverse_and_right": ("reverse_and_right", 1000),
        }
        action = action_map.get(decision)
        if action is None:
            return False

        situation_map = {
            "slight_left": "前方有小障碍物",
            "slight_right": "前方有小障碍物",
            "small_left": "前方有中障碍物",
            "small_right": "前方有中障碍物",
            "large_left": "前方有大障碍物",
            "large_right": "前方有大障碍物",
            "reverse_and_left": "前方障碍物过大需要倒车",
            "reverse_and_right": "前方障碍物过大需要倒车",
        }
        print(f"[Driving] 视觉避障 decision={decision}, action={action}")
        self._log_drive_state(
            situation_map.get(decision, "前方有障碍物"),
            context,
            f"{action[0]}({action[1]}ms)",
            self.stable_circle_angle,
        )
        self._execute_maneuver(w, action=action[0], speed=context.speed, duration=action[1], brake_with_steer=True)
        return True

    def _drive_toward_target(
        self,
        w: "FrameWorker",
        context: DriveContext,
        target_direction: Optional[float],
    ):
        if target_direction is None:
            print("[Driving] 无进圈目标角度，保持直行")
            self._log_drive_state("前方无障碍物", context, "straight", target_direction)
            self._execute_maneuver(w, "straight", context.speed)
            return

        turn_dir, _, diff = calculate_move_count(context.direction, target_direction)
        print(
            f"[Driving] 目标对齐: current={context.direction}, "
            f"target={target_direction}, diff={diff:.2f}"
        )

        if turn_dir is None or diff <= self.ALIGN_THRESHOLD:
            self._log_drive_state("前方无障碍物", context, "straight", target_direction)
            self._execute_maneuver(w, "straight", context.speed)
            return

        action = f"forward_turn_{turn_dir}"
        duration = self._get_turn_duration(diff, fine=False)

        self._log_drive_state("发现进圈角度，进行对齐", context, f"{action}({duration}ms)", target_direction)
        self._execute_maneuver(w, action, speed=context.speed, duration=duration, brake_with_steer=False)

    def _get_exit_alignment_duration(self, diff: float) -> int:
        if diff <= 0:
            return 0
        return max(50, min(600, int(round(diff * 10))))

    def _get_turn_duration(
        self,
        diff: float,
        fine: bool = False,
        reverse: bool = False,
    ) -> int:
        if diff <= 0:
            return 0

        if fine:
            if diff <= 6:
                return 120
            if diff <= 10:
                return 180
            if diff <= 18:
                return 240
            if diff <= 28:
                return 320
            if diff <= 40:
                return 420
            return min(560, int(diff * 10))

        if reverse:
            if diff <= 15:
                return 300
            if diff <= 35:
                return 500
            if diff <= 60:
                return 700
            if diff <= 90:
                return 900
            return min(1200, int(diff * 10))

        if diff <= 12:
            return 180
        if diff <= 25:
            return 280
        if diff <= 45:
            return 420
        if diff <= 70:
            return 560
        return min(760, int(diff * 8))

    def _check_motion_block(self, context: DriveContext) -> bool:
        if self.current_stage in (self.STAGE_EXIT_GARAGE, self.STAGE_FINISH):
            self.last_motion_location = context.location
            self.blocked_motion_count = 1
            return False

        if self.last_motion_location is None:
            self.last_motion_location = context.location
            self.blocked_motion_count = 1
            return False

        if get_distance(context.location, self.last_motion_location) <= self.STUCK_LOCATION_EPS:
            self.blocked_motion_count += 1
        else:
            self.blocked_motion_count = 1

        self.last_motion_location = context.location
        return self.blocked_motion_count >= self.STUCK_REPEAT_LIMIT

    def _handle_motion_block(self, w: "FrameWorker", context: DriveContext):
        steer = self.last_motion_steer or self._decision_to_steer(context.decision) or "right"
        action = f"backward_turn_{steer}"

        print(
            f"[Driving] 快速脱困: stuck_frames={self.blocked_motion_count}, "
            f"steer={steer}, action={action}"
        )
        self._log_drive_state("车辆连续多帧位置不变", context, f"{action}(1000ms)", self.stable_circle_angle)
        self.blocked_motion_count = 0
        self._execute_maneuver(w, action, speed=context.speed, duration=1000, brake_with_steer=True)

    def _decision_to_steer(self, decision: str) -> Optional[str]:
        if "left" in decision:
            return "left"
        if "right" in decision:
            return "right"
        return None

    def _has_front_obstacle(self, context: DriveContext) -> bool:
        obstacle_count = int(context.obstacle_info.get("obstacles_count", 0) or 0)
        if obstacle_count > 0:
            return True
        if context.decision != "straight":
            return True
        return self._has_forbidden_ahead(context)

    def _get_forbidden_check_distance(self, speed: Optional[int]) -> int:
        if speed is None:
            return self.FORBIDDEN_BASE_CHECK_DISTANCE
        if speed >= 3:
            return self.FORBIDDEN_SPEED3_CHECK_DISTANCE
        if speed == 2:
            return self.FORBIDDEN_SPEED2_CHECK_DISTANCE
        return self.FORBIDDEN_BASE_CHECK_DISTANCE

    def _has_forbidden_ahead(self, context: DriveContext) -> bool:
        check_distance = self._get_forbidden_check_distance(context.speed)
        return (
            self.map_tool.check_safety_ahead(
                context.location,
                context.direction,
                distance=check_distance,
            )
            == "Pause"
        )

    def _update_trapped_state(self, location: Tuple[int, int]):
        self.history_locations.append(location)
        if len(self.history_locations) > self.TRAPPED_HISTORY_LEN:
            self.history_locations.pop(0)

        if len(self.history_locations) < self.TRAPPED_HISTORY_LEN:
            self.trapped = False
            return

        valid_points = [
            pt for pt in self.history_locations
            if pt is not None and pt[0] is not None and pt[1] is not None
        ]
        if len(valid_points) < self.TRAPPED_HISTORY_LEN:
            self.trapped = False
            return

        centroid_x = sum(pt[0] for pt in valid_points) / len(valid_points)
        centroid_y = sum(pt[1] for pt in valid_points) / len(valid_points)
        max_radius = max(get_distance(pt, (centroid_x, centroid_y)) for pt in valid_points)

        self.trapped = (
            is_location_stagnant(valid_points)
            and self.TRAPPED_RADIUS_MIN <= max_radius <= self.TRAPPED_RADIUS_MAX
        )
        if self.trapped:
            print(
                f"[Driving] 严格困死判定成立: radius={max_radius:.2f}, "
                f"frames={len(valid_points)}"
            )

    def _run_stage_finish(self, w: "FrameWorker"):
        self.current_stage = self.STAGE_FINISH
        self._tap_single_control(w, "brake", wait=self.TIME_BRAKE, dura=100)
        self._click_control(w, "off_car")
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("跑图阶段")

    def _exit_vehicle_to_running(self, w: "FrameWorker", reason: str):
        print(f"[Driving] {reason}")
        self.current_stage = self.STAGE_FINISH
        self._tap_single_control(w, "brake", wait=800, dura=80)
        self._click_control(w, "off_car")
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("跑图阶段")

    def _handle_death(self, w: "FrameWorker"):
        spectate = w.get_info("观战对手") or w.get_info("观战")
        if spectate:
            self._frame_action_executed = True
            w.click(spectate)
            time.sleep(2)
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("结束阶段")

    def _handle_rank_finish(self, w: "FrameWorker"):
        spectate = w.get_info("观战对手")
        if spectate:
            self._frame_action_executed = True
            w.click(spectate)
            time.sleep(2)
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("结束阶段")

    def _update_circle_angle(self, value):
        angle = self._get_scalar(value)
        if angle is None:
            return

        self.circle_angles.append(angle)
        if len(self.circle_angles) < 30:
            self.stable_circle_angle = None
            return

        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.utils import stable_angle

        self.stable_circle_angle = stable_angle(self.circle_angles)
        if self.stable_circle_angle is None:
            self.circle_angles.pop(0)
        else:
            self.circle_angles = []

    def _execute_maneuver(
        self,
        w: "FrameWorker",
        action: str,
        speed: Optional[int] = None,
        duration: int = 0,
        brake_with_steer: bool = False,
    ):
        action = self._normalize_motion_action(action)
        steer_key = "left" if "left" in action else ("right" if "right" in action else None)

        if self._needs_pre_brake(action, speed):
            brake_wait = 1000 if speed == 2 else 2000
            if brake_with_steer and steer_key is not None:
                self._tap_double_control(w, "brake", steer_key, wait=brake_wait)
            else:
                self._tap_single_control(w, "brake", wait=brake_wait)

        simple_actions = {
            "straight": lambda: self._click_down_control(w, "up"),
            "forward": lambda: self._tap_single_control(w, "up", wait=duration),
            "backward": lambda: self._tap_single_control(w, "down", wait=duration),
            "backward_turn_left": lambda: self._tap_double_control(w, "down", "right", wait=duration),
            "backward_turn_right": lambda: self._tap_double_control(w, "down", "left", wait=duration),
            "forward_turn_left": lambda: self._tap_double_control(w, "up", "left", wait=duration),
            "forward_turn_right": lambda: self._tap_double_control(w, "up", "right", wait=duration),
            "brake_turn_left": lambda: self._tap_double_control(w, "brake", "left", wait=duration or 300),
            "brake_turn_right": lambda: self._tap_double_control(w, "brake", "right", wait=duration or 300),
            "reverse_and_left": lambda: self._tap_double_control(w, "down", "right", wait=duration or 1000),
            "reverse_and_right": lambda: self._tap_double_control(w, "down", "left", wait=duration or 1000),
        }
        executor = simple_actions.get(action)
        if executor is None:
            print(f"[Driving] 未支持的驾驶动作: {action}")
            return

        executor()
        self._record_motion_action(action)

    def _needs_pre_brake(self, action: str, speed: Optional[int]) -> bool:
        if speed != 3:
            return False
        if self.current_stage == self.STAGE_EXIT_GARAGE:
            return True
        if action == "straight":
            return False
        return True

    def _normalize_motion_action(self, action: str) -> str:
        if action == "forward":
            return f"forward_turn_{self._get_default_steer()}"
        if action == "backward":
            return f"backward_turn_{self._get_default_steer()}"
        return action

    def _get_default_steer(self) -> str:
        return self.last_motion_steer or "right"

    def _should_trigger_auto_brake(self) -> bool:
        now = time.time()
        if now - self.last_auto_brake_time < self.BRAKE_COOLDOWN_SPEED2:
            return False
        self.last_auto_brake_time = now
        return True

    def _record_motion_action(self, action: str):
        self.last_motion_mode = None
        self.last_motion_steer = None
        self.last_motion_started_at = time.time()

        if action in ("straight", "forward", "forward_turn_left", "forward_turn_right"):
            self.last_motion_mode = "forward"
        elif action in (
            "backward",
            "backward_turn_left",
            "backward_turn_right",
            "reverse_and_left",
            "reverse_and_right",
        ):
            self.last_motion_mode = "backward"

        if "left" in action:
            self.last_motion_steer = "left"
        elif "right" in action:
            self.last_motion_steer = "right"

    def _get_circle_distance(self) -> int:
        elapsed = self.get_elapsed_time()
        if elapsed <= self.STAGE1_TIME:
            return self.STAGE1_DIS
        if elapsed <= self.STAGE2_TIME:
            return self.STAGE2_DIS
        return self.STAGE3_DIS

    def _get_location(self, w: "FrameWorker") -> Optional[Tuple[int, int]]:
        info = w.get_info("location")
        if info is None:
            return None
        if isinstance(info, (list, tuple)):
            if len(info) >= 2 and not isinstance(info[0], (list, tuple)):
                return check_location(info)
            if len(info) > 0:
                return check_location(info[0])
        return None

    def _get_speed(self, w: "FrameWorker") -> Optional[int]:
        speed = self._get_scalar(w.get_info("speed"))
        return int(speed) if speed is not None else None

    def _get_scalar(self, value) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, (int, float)):
                return float(first)
        return None

    def _use_valid_location(self, location: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if location is not None:
            self.prior_location = location
            return location
        if self.prior_location is not None:
            print("[Driving] 当前坐标无效，回退到上一帧坐标")
        return self.prior_location

    def _use_valid_direction(self, direction: Optional[float]) -> Optional[float]:
        if direction is not None:
            self.prior_angle = direction
            return direction
        if self.prior_angle is not None:
            print("[Driving] 当前角度无效，回退到上一帧角度")
        return self.prior_angle

    def _is_dead(self, w: "FrameWorker") -> bool:
        if bool(w.get_info("变身")):
            return True
        if bool(w.get_info("红色血条")):
            return True
        return False

    def _has_rank_info(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("个人排名")) or bool(w.get_info("队伍排名"))

    def _analyze_obstacles(self, w: "FrameWorker") -> Dict[str, Any]:
        detections = w.get_info("forward_scene2")
        if not detections:
            return {"decision": "straight", "obstacles_count": 0}

        frame = getattr(w, "frame", None)
        if frame is None or not hasattr(frame, "shape"):
            return {"decision": "straight", "obstacles_count": 0}

        frame_h, frame_w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.DRIVE_VIEW_RECT
        roi_x = int(frame_w * rx1)
        roi_y = int(frame_h * ry1)
        roi_w = max(1, int(frame_w * (rx2 - rx1)))
        roi_h = max(1, int(frame_h * (ry2 - ry1)))

        if self.obstacle_analyzer is None:
            self.obstacle_analyzer = ObstacleAvoidanceAnalyzer(width=roi_w, height=roi_h)

        local_dets = []
        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            local_dets.append([x1 - roi_x, y1 - roi_y, x2 - roi_x, y2 - roi_y, conf, cls_id])

        if not local_dets:
            return {"decision": "straight", "obstacles_count": 0}

        return self.obstacle_analyzer.analyze(local_dets)

    def _angle_diff(self, angle1: float, angle2: float) -> float:
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    def _resolve_control_name(self, w: "FrameWorker", key: str) -> str:
        aliases = self.CONTROL_ALIASES.get(key, (key,))
        stage = w.current_stage
        buttons = getattr(getattr(w, "controller", None), "buttons", {})
        if stage and buttons:
            for alias in aliases:
                if f"{stage}_{alias}" in buttons:
                    return alias
        return aliases[0]

    def _tap_single_control(self, w: "FrameWorker", key: str, **kwargs):
        self._frame_action_executed = True
        w.tap_single(self._resolve_control_name(w, key), **kwargs)

    def _tap_double_control(self, w: "FrameWorker", key1: str, key2: str, **kwargs):
        self._frame_action_executed = True
        w.tap_double(self._resolve_control_name(w, key1), self._resolve_control_name(w, key2), **kwargs)

    def _click_control(self, w: "FrameWorker", key: str, **kwargs):
        self._frame_action_executed = True
        w.click(self._resolve_control_name(w, key), **kwargs)

    def _click_down_control(self, w: "FrameWorker", key: str, **kwargs):
        self._frame_action_executed = True
        w.click_down(self._resolve_control_name(w, key), **kwargs)
