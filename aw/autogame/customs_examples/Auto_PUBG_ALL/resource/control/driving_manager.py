import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager import HouseExitManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.obstacle_analyzer import (
    ObstacleAvoidanceAnalyzer,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    calculate_angle,
    calculate_move_count,
    check_location,
    draw_points_with_arrows,
    get_distance,
    is_location_stagnant,
    load_adaptive_motion_section,
    persist_adaptive_motion_section,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.timing import Cooldown, Stopwatch
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import log_step

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


@dataclass
class DriveContext:
    location: Tuple[int, int]
    direction: float
    speed: Optional[int]
    decision: str
    obstacle_info: Dict[str, Any]


class TurnCalibration:
    SECTION = "drive_turn"
    DEFAULT_DEG_PER_MS = 0.08
    MIN_DEG_PER_MS = 0.02
    MAX_DEG_PER_MS = 0.20
    MIN_DURATION_MS = 120
    MAX_DURATION_MS = 650
    LEARNING_RATE = 0.2
    MIN_OBSERVED_DEG = 3.0
    MAX_OBSERVED_DEG = 120.0

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        raw = load_adaptive_motion_section(self.SECTION)
        if isinstance(raw, dict):
            self.data = {
                str(key): value
                for key, value in raw.items()
                if isinstance(value, dict)
            }

    def save(self):
        persist_adaptive_motion_section(self.SECTION, self.data)

    def estimate_duration(
        self,
        action: str,
        speed: Optional[int],
        diff: float,
        fallback_ms: int,
        max_duration_ms: Optional[int] = None,
    ) -> int:
        key = self._key(action, speed)
        rate = self._get_rate(key)
        estimated = int(round(float(diff) / rate))
        duration = self._clamp_duration(estimated, max_duration_ms=max_duration_ms)
        if key not in self.data:
            duration = min(
                duration,
                self._clamp_duration(fallback_ms, max_duration_ms=max_duration_ms),
            )
        return duration

    def observe(
        self,
        action: str,
        speed: Optional[int],
        before_angle: float,
        after_angle: float,
        duration_ms: int,
    ) -> bool:
        if duration_ms <= 0:
            return False

        observed_deg = self._observed_turn_degrees(action, before_angle, after_angle)
        if observed_deg is None:
            return False
        if not (self.MIN_OBSERVED_DEG <= observed_deg <= self.MAX_OBSERVED_DEG):
            return False

        observed_rate = observed_deg / float(duration_ms)
        observed_rate = self._clamp_rate(observed_rate)
        key = self._key(action, speed)
        old_rate = self._get_rate(key)
        samples = int(self.data.get(key, {}).get("samples", 0) or 0)
        alpha = self.LEARNING_RATE if samples > 0 else 1.0
        new_rate = self._clamp_rate(old_rate * (1.0 - alpha) + observed_rate * alpha)
        self.data[key] = {
            "deg_per_ms": new_rate,
            "samples": samples + 1,
            "updated_at": time.time(),
        }
        self.save()
        print(
            f"[TurnCalibration] 更新转向标定: key={key}, observed={observed_deg:.1f}deg/"
            f"{duration_ms}ms, rate={new_rate:.4f}, samples={samples + 1}"
        )
        return True

    def _key(self, action: str, speed: Optional[int]) -> str:
        return f"{action}:{self._speed_bucket(speed)}"

    def _speed_bucket(self, speed: Optional[int]) -> str:
        if speed is None or speed <= 1:
            return "speed_low"
        if speed == 2:
            return "speed_2"
        return "speed_3"

    def _get_rate(self, key: str) -> float:
        value = self.data.get(key, {}).get("deg_per_ms", self.DEFAULT_DEG_PER_MS)
        try:
            return self._clamp_rate(float(value))
        except (TypeError, ValueError):
            return self.DEFAULT_DEG_PER_MS

    def _clamp_rate(self, value: float) -> float:
        return max(self.MIN_DEG_PER_MS, min(self.MAX_DEG_PER_MS, value))

    def _clamp_duration(self, value: int, max_duration_ms: Optional[int] = None) -> int:
        max_duration = self.MAX_DURATION_MS if max_duration_ms is None else int(max_duration_ms)
        return max(self.MIN_DURATION_MS, min(max_duration, int(value)))

    def _observed_turn_degrees(
        self,
        action: str,
        before_angle: float,
        after_angle: float,
    ) -> Optional[float]:
        right_delta = (float(after_angle) - float(before_angle)) % 360.0
        left_delta = (float(before_angle) - float(after_angle)) % 360.0
        if action.endswith("_right"):
            return right_delta if right_delta <= 180.0 else None
        if action.endswith("_left"):
            return left_delta if left_delta <= 180.0 else None
        return None


class DrivingManager:
    ESCAPE_ALIGN_TOLERANCE = 8
    EXIT_GARAGE_INITIAL_FORWARD_MS = 3000
    EXIT_GARAGE_INITIAL_REVERSE_LEFT_MS = 1000
    EXIT_GARAGE_VISUAL_FORWARD_MS = 900
    EXIT_GARAGE_VISUAL_TURN_MS = 500
    EXIT_GARAGE_CENTERING_FORWARD_MS = 650
    EXIT_GARAGE_CENTERING_MIN_TURN_MS = 180
    EXIT_GARAGE_CENTERING_MAX_TURN_MS = 360
    EXIT_GARAGE_CLEAR_ROUNDS_TO_CRUISE = 3
    EXIT_GARAGE_SUCCESS_DISTANCE = 8.0
    EXIT_GARAGE_WALL_CLASS_IDS = {9, 19}
    EXIT_GARAGE_WALL_MIN_CONF = 0.35
    EXIT_GARAGE_WALL_MIN_AREA_RATIO = 0.003
    EXIT_GARAGE_WALL_MIN_BOTTOM_RATIO = 0.30
    EXIT_GARAGE_WALL_MIN_WIDTH_RATIO = 0.05
    EXIT_GARAGE_WALL_BLOCK_RATIO = 0.18
    EXIT_GARAGE_GATE_MIN_WIDTH_RATIO = 0.18
    EXIT_GARAGE_GATE_CENTER_DEADZONE = 0.05
    EXIT_GARAGE_SECTORS = {
        "left": (0.0, 0.38),
        "center": (0.28, 0.72),
        "right": (0.62, 1.0),
    }

    # 车辆方向与目标方向的允许误差；小于该值时直接视为已对齐
    ALIGN_THRESHOLD = 8
    # 到路径点多近时认为已经到达该路径点
    WAYPOINT_TOLERANCE = 5.0

    # 连续多少帧位置几乎不变时，判定车辆被卡住
    STUCK_REPEAT_LIMIT = 7
    # 连续按前进/前进+方向但位置不变时，更早认为可能撞到未识别障碍物。
    FORWARD_BLOCK_REPEAT_LIMIT = 2
    # 两次位置变化小于该距离时，视为基本没动
    STUCK_LOCATION_EPS = 0.2
    # 前进连续短时不动时只做轻量倒车微调，避免倒车转向过久直接掉头。
    FORWARD_BLOCK_BACKWARD_TURN_MS = 700
    FORWARD_BLOCK_TURN_MS = 600
    # 空旷区域连续前进不动到该帧数后，认为车辆可能没油，切回跑图处理。
    NO_FUEL_FORWARD_STALL_LIMIT = 4
    NO_FUEL_MAX_OBSTACLE_COVERAGE = 0.02
    MOTION_BLOCK_BACKWARD_TURN_MS = 800
    ROUTE_DEVIATION_REPLAN_DISTANCE = 35.0
    ROUTE_DEVIATION_LOOKAHEAD_POINTS = 8
    ROUTE_REPLAN_COOLDOWN_S = 3.0
    LOCAL_AVOIDANCE_REPEAT_DISTANCE = 12.0
    LOCAL_AVOIDANCE_REPEAT_LIMIT = 2
    ROUTE_SKIP_AFTER_AVOIDANCE = 5
    ROUTE_BLOCKED_POINT_RADIUS = 25.0
    ROUTE_TURN_MAX_DURATION_MS = 650
    FORBIDDEN_TURN_MAX_DURATION_MS = 1500
    VISUAL_AVOIDANCE_TURN_MAX_DURATION_MS = 1200
    VISUAL_CLOSE_REVERSE_MS = 1050
    VISUAL_CLOSE_BOTTOM_RATIO = 0.82
    VISUAL_CLOSE_AREA_RATIO = 0.10
    VISUAL_CLOSE_COVERAGE_RATIO = 0.52
    VISUAL_AVOIDANCE_ANGLE_MAP = {
        "slight_left": 12,
        "slight_right": 12,
        "small_left": 24,
        "small_right": 24,
        "large_left": 40,
        "large_right": 40,
        "reverse_and_left": 50,
        "reverse_and_right": 50,
    }
    # 困死判定窗口：连续多少帧都在局部很小范围打转才算真正困死
    TRAPPED_HISTORY_LEN = 80
    # 困死判定时，轨迹围绕中心点打转的半径范围
    TRAPPED_RADIUS_MIN = 1.0
    TRAPPED_RADIUS_MAX = 2.0
    # 已经进入不可通行区域后，连续多少次“没有朝安全点靠近”才结束当前局
    FORBIDDEN_ESCAPE_FAIL_LIMIT = 16
    FORBIDDEN_ESCAPE_OBSTACLE_REVERSE_AFTER = 2
    # 黑区脱离时，只要到最近安全点的距离缩短超过这个值，就认为本轮有进展
    FORBIDDEN_ESCAPE_PROGRESS_EPS = 1.0
    # 前方黑区检测距离：基础值，以及速度 2/3 时更远的提前量
    FORBIDDEN_BASE_CHECK_DISTANCE = 10
    FORBIDDEN_SPEED2_CHECK_DISTANCE = 14
    FORBIDDEN_SPEED3_CHECK_DISTANCE = 20
    # 高速接近黑区时，额外预刹车的持续时间
    FORBIDDEN_HIGH_SPEED_BRAKE_WAIT = 1400

    # 速度 2 时自动刹车的冷却时间，避免频繁点刹
    BRAKE_COOLDOWN_SPEED2 = 2.0
    # 前方无障碍直行时，点自动前进开/关，避免长按油门期间错过新障碍物。
    STRAIGHT_AUTO_FORWARD_ON_S = 4.0
    STRAIGHT_AUTO_FORWARD_PAUSE_S = 1.0
    STRAIGHT_AUTO_FORWARD_OBSTACLE_POLL_S = 0.25

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
    # 车前方障碍物识别使用的画面 ROI 外接矩形；实际视觉避障会在该矩形内再收窄为透视梯形。
    DRIVE_VIEW_RECT = (0.2797, 0.2826, 0.7203, 0.6175)
    DRIVE_VIEW_TRAPEZOID_TOP_Y = 0.18
    DRIVE_VIEW_TRAPEZOID_TOP_WIDTH_SCALE = 0.45
    # 第三人称驾驶时，自车常从 ROI 底部进入画面，并且横向覆盖会从中部向两侧扩散。
    # 因此只有 car 检测框满足“触底 + 横向跨过中线且左右两侧都有一定覆盖”时，才视为自车。
    SELF_VEHICLE_CLASS_IDS = {7}
    SELF_VEHICLE_BOTTOM_INTERSECT_TOLERANCE = 2.0
    SELF_VEHICLE_MIN_SIDE_COVER_RATIO = 0.06
    SELF_VEHICLE_MIN_TOTAL_WIDTH_RATIO = 0.18
    HORN_MISSING_EXIT_FRAME_LIMIT = 5

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
        "auto_forward": ("开车自动前进", "自动前进", "auto_forward"),
        "off_car": ("下车", "离开载具", "off_car"),
        "spectate": ("观战对手", "观战", "guanzhan"),
    }

    def __init__(
        self,
        map_tool: Optional[MapNavigator] = None,
        max_driving_time: int = DEFAULT_MAX_DRIVING_TIME,
    ):
        self.map_tool = map_tool or MapNavigator()
        self.max_driving_time = max_driving_time
        self.obstacle_analyzer: Optional[ObstacleAvoidanceAnalyzer] = None
        self.house_exit_manager = HouseExitManager()
        self.turn_calibration = TurnCalibration()

        self.match_clock = Stopwatch()
        self.driving_clock = Stopwatch()
        self.route_replan_cooldown = Cooldown()
        self.auto_brake_cooldown = Cooldown()

        self.is_first_car = True
        self.current_stage = self.STAGE_EXIT_GARAGE
        self.exit_garage_phase = 0
        self.exit_garage_clear_rounds = 0
        self.exit_garage_start_location: Optional[Tuple[int, int]] = None

        self.circle_angles: List[float] = []
        self.latest_circle_angle: Optional[float] = None
        self.stable_circle_angle: Optional[float] = None
        self.last_planned_circle_angle: Optional[float] = None
        self.road_list: List[Tuple[int, int]] = []
        self.last_avoidance_location: Optional[Tuple[int, int]] = None
        self.local_avoidance_fail_count = 0

        self.prior_angle: Optional[float] = None
        self.prior_location: Optional[Tuple[int, int]] = None

        self.history_locations: List[Tuple[int, int]] = []
        self.trapped = False
        self.forbidden_escape_failures = 0
        self.forbidden_escape_last_distance: Optional[float] = None
        self.forbidden_escape_last_action: Optional[str] = None
        self.forbidden_escape_action_repeats = 0

        self.last_motion_mode: Optional[str] = None
        self.last_motion_steer: Optional[str] = None
        self.last_motion_location: Optional[Tuple[int, int]] = None
        self.blocked_motion_count = 0
        self.no_fuel_stall_count = 0
        self.motion_stalled_this_frame = False
        self.forward_block_recovery_active = False
        self.allow_running_fallback = True
        self.pause_sp_callback: Optional[Callable] = None
        self.terminal_state_callback: Optional[Callable] = None
        self.next_running_finding_car: Optional[bool] = None
        self.horn_missing_frames = 0
        self.drive_auto_forward_active = False
        self.drive_auto_forward_started_at: Optional[float] = None

        self._frame_action_executed = False

    def set_game_time(self, game_time: Optional[float] = None):
        started_at = self.match_clock.start(game_time)
        print(f"[Driving] 游戏开始时间设置为：{started_at:.3f}")

    def get_elapsed_time(self) -> float:
        return self.match_clock.elapsed()

    def get_driving_elapsed_time(self) -> float:
        return self.driving_clock.elapsed()

    def set_remaining_drive_time(self, remaining_time: Optional[float]):
        if remaining_time is None:
            return
        self.max_driving_time = max(0.0, float(remaining_time))
        print(f"[Driving] 剩余开车时长设置为：{self.max_driving_time:.2f}s")

    def reset(self, max_driving_time: Optional[int] = None):
        if max_driving_time is not None:
            self.max_driving_time = max_driving_time

        self.driving_clock.reset()
        self.is_first_car = True
        self.current_stage = self.STAGE_EXIT_GARAGE
        self.exit_garage_phase = 0
        self.exit_garage_clear_rounds = 0
        self.exit_garage_start_location = None

        self.circle_angles = []
        self.latest_circle_angle = None
        self.stable_circle_angle = None
        self.last_planned_circle_angle = None
        self.road_list = []
        self.route_replan_cooldown.reset()
        self.last_avoidance_location = None
        self.local_avoidance_fail_count = 0

        self.prior_angle = None
        self.prior_location = None

        self.history_locations = []
        self.trapped = False
        self.forbidden_escape_failures = 0
        self.forbidden_escape_last_distance = None
        self.forbidden_escape_last_action = None
        self.forbidden_escape_action_repeats = 0

        self.last_motion_mode = None
        self.last_motion_steer = None
        self.last_motion_location = None
        self.blocked_motion_count = 0
        self.no_fuel_stall_count = 0
        self.motion_stalled_this_frame = False
        self.forward_block_recovery_active = False
        self.auto_brake_cooldown.reset()
        self.allow_running_fallback = True
        self.next_running_finding_car = None
        self.horn_missing_frames = 0
        self.drive_auto_forward_active = False
        self.drive_auto_forward_started_at = None

        self._frame_action_executed = False
        self.house_exit_manager.reset()
        print("[Driving] 状态已重置!")

    def set_running_fallback_enabled(self, enabled: bool):
        self.allow_running_fallback = bool(enabled)

    def consume_running_transition_finding_car(self, default: bool) -> bool:
        finding_car = self.next_running_finding_car
        self.next_running_finding_car = None
        if finding_car is None:
            return bool(default)
        return bool(finding_car)

    def skip_initial_exit_garage(self, reason: str = "roadside vehicle"):
        self.is_first_car = False
        self.current_stage = self.STAGE_CRUISE
        self.exit_garage_phase = 0
        self.exit_garage_clear_rounds = 0
        self.exit_garage_start_location = None
        self.last_motion_location = None
        self.blocked_motion_count = 0
        print(f"[Driving] 本次上车来源={reason}，跳过首次出库，直接进入巡航阶段")

    def process(self, w: "FrameWorker"):
        self._frame_worker = w
        self._begin_frame()
        w.frame_log("进入开车模块：这一帧先判断是否还在车上，再读取车辆位置、方向、速度和障碍信息")

        if self._handle_terminal_state(w, "开车模块入口"):
            self._finalize_frame(w)
            return

        if self.driving_clock.ensure_started():
            print(f"[Driving] 本次驾驶计时开始：{self.driving_clock.started_at:.3f}")
            w.frame_log("开车记录：这是本轮驾驶计时首次启动，后续会用这个时间判断开车阶段是否结束")

        if self._is_out_of_vehicle(w):
            print("[Driving] 检测到人物已下车，切回跑图阶段")
            w.frame_log("开车观察：当前帧车内控制 UI 消失或步行 UI 成立，所以判断人物已下车并切回跑图")
            log_step(
                "当前开车帧日志：开车阶段入口检测到人物已下车，车辆控制UI消失或步行UI成立",
                target="当前开车分支：已下车",
                action="停止驾驶并交给 RunningManager",
                method="_handle_unexpected_vehicle_exit(w)",
                result="下一帧重新按跑图阶段处理",
            )
            if hasattr(w, "set_frame_decision"):
                w.set_frame_decision(
                    observation="开车阶段当前帧检测到人物已下车",
                    target="开车阶段",
                    decision="切回跑图阶段",
                    action="停止驾驶并交给 RunningManager",
                    method="_handle_unexpected_vehicle_exit(w)",
                    result="下一帧重新按跑图阶段处理",
                )
            self._handle_unexpected_vehicle_exit(w)
            self._finalize_frame(w)
            return

        context = self._build_context(w)
        if context is None:
            print("[Driving] 当前位置或朝向无效，等待下一帧")
            w.frame_log("开车观察：当前帧没有有效车辆位置或方向，所以不下发驾驶动作，等下一帧重新识别")
            log_step(
                "当前开车帧日志：开车阶段当前位置或朝向无效，无法计算目标方向/障碍转向",
                target="当前开车分支：位置或朝向无效",
                action="暂不下发驾驶动作",
                method="_build_context(w)",
                result="等待下一帧重新识别车辆位置和方向",
            )
            if hasattr(w, "set_frame_decision"):
                w.set_frame_decision(
                    observation="开车阶段当前位置或朝向无效",
                    target="开车阶段",
                    decision="等待下一帧重新识别车辆位置和方向",
                    action="暂不下发驾驶动作",
                    method="_build_context(w)",
                    result="避免基于无效坐标误操作",
                )
            self._finalize_frame(w)
            return

        w.frame_log(
            f"开车观察：车辆位置={context.location}，方向={context.direction:.1f}，"
            f"速度={context.speed}，视觉避障建议={context.decision}"
        )
        log_step(
            f"当前开车帧日志：开车阶段入口观察：当前位置={context.location}，"
            f"当前方位={context.direction:.1f}，速度={context.speed}，"
            f"驾驶子状态={self.current_stage}，视觉决策={context.decision}，"
            f"障碍数量={int(context.obstacle_info.get('obstacles_count', 0) or 0)}",
            target="开车阶段",
            action="根据道路、目标点、障碍和车辆状态继续驾驶",
            method="DrivingManager.process",
            result="本帧继续推进驾驶路线",
        )
        if hasattr(w, "set_frame_decision"):
            w.set_frame_decision(
                observation=(
                    f"开车阶段：当前位置={context.location}，当前方位={context.direction}，"
                    f"驾驶子状态={self.current_stage}"
                ),
                target="开车阶段",
                decision="根据道路、目标点、障碍和车辆状态继续驾驶",
                action="执行驾驶控制",
                method="DrivingManager.process",
                result="本帧继续推进驾驶路线",
            )

        if not self.match_clock.is_running():
            self.set_game_time()

        self._update_circle_angle(w.get_info("white_angle"))
        self._update_trapped_state(context.location)

        if self.current_stage == self.STAGE_EXIT_GARAGE:
            print("[Driving] 当前阶段: 出库阶段")
            w.frame_log("开车决策：当前仍在首次出库阶段，所以优先执行出库对齐，不进入普通巡航")
            self._handle_first_car_alignment(w, context)
            self._finalize_frame(w)
            return

        if self.get_driving_elapsed_time() >= self.max_driving_time:
            print("[Driving] 驾驶时长已到，结束驾驶阶段")
            w.frame_log("开车决策：驾驶阶段计时已用完，所以结束驾驶并交回总流程")
            self._run_stage_finish(w)
            self._finalize_frame(w)
            return

        if self.trapped:
            if self._try_house_exit_when_indoor(w, "车辆困死"):
                w.frame_log("开车决策：车辆困死且可能卡在屋内，所以先交给出房模块脱离")
                self._finalize_frame(w)
                return
            print("[Driving] 车辆长时间在 1~2 距离范围内打转，判定困死")
            if self.allow_running_fallback:
                w.frame_log("开车决策：车辆长期困死且允许跑图兜底，所以停车下车切回跑图")
                self._exit_vehicle_to_running(
                    w,
                    "车辆困死，切回跑图阶段",
                    finding_car=self._should_continue_finding_car_after_running_fallback("车辆困死"),
                )
            else:
                w.frame_log("开车决策：车辆长期困死且不允许跑图兜底，所以结束当前局")
                self._handle_death(w)
            self._finalize_frame(w)
            return

        self.current_stage = self.STAGE_CRUISE
        target_direction = self._resolve_target_direction(context.location)
        w.frame_log(f"开车目标：当前目标方向={target_direction}，接下来会先处理黑区、卡住和障碍，再决定是否直行")

        w.frame_log("开车检查：先判断车辆当前位置是否在不可通行区域，命中时优先脱离")
        if self._handle_forbidden_escape(w, context):
            self._finalize_frame(w)
            return

        motion_blocked = self._check_motion_block(context)
        if self._check_probable_no_fuel(context):
            finding_car = self._should_find_car_after_no_fuel()
            w.frame_log("开车决策：空旷区域连续前进但车辆不动，疑似没油，所以切回跑图继续找车或进圈")
            self._exit_vehicle_to_running(
                w,
                "空旷区域连续前进不动，疑似车辆没油，切回跑图阶段",
                finding_car=finding_car,
            )
            self._finalize_frame(w)
            return

        if self._check_forward_motion_block(context):
            print("[Driving] 连续前进2帧位置不变，执行前进卡住恢复")
            w.frame_log("开车决策：连续前进但位置不变，所以先执行前进卡住恢复动作")
            self._handle_forward_motion_block(w, context)
            self._finalize_frame(w)
            return

        if motion_blocked:
            print("[Driving] 检测到连续7帧位置不变，执行倒车避障")
            w.frame_log("开车决策：车辆连续多帧位置不动，所以执行倒车避障恢复")
            self._handle_motion_block(w, context)
            self._finalize_frame(w)
            return

        w.frame_log("开车检查：再判断车头前方地图区域是否不可通行，避免继续向障碍开")
        if self._handle_forbidden_ahead(w, context):
            self._finalize_frame(w)
            return

        w.frame_log("开车检查：最后看视觉避障结果，如果识别到障碍就按避障动作处理")
        if self._handle_visual_avoidance(w, context):
            self._finalize_frame(w)
            return

        w.frame_log("开车决策：没有触发黑区、卡住或视觉避障，所以按目标方向正常驾驶")
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
        obstacle_classes = ",".join(context.obstacle_info.get("classes", [])[:4]) or "None"
        print(
            f"[情况:{situation}] "
            f"[状态: speed={speed_text}, loc={context.location}, dir={context.direction:.1f}, "
            f"circle={circle_text}, target={target_text}, obstacle_count={obstacle_count}, "
            f"vision={context.decision}, classes={obstacle_classes}] "
            f"[决策:{decision}]"
        )
        observation = (
            f"开车阶段：驾驶子状态={self.current_stage}，情况={situation}，"
            f"当前位置={context.location}，当前方位={context.direction:.1f}，"
            f"速度={speed_text}，圈角={circle_text}，目标角={target_text}，"
            f"障碍数量={obstacle_count}，视觉决策={context.decision}，障碍类别={obstacle_classes}"
        )
        method = (
            "DrivingManager._log_drive_state "
            f"target_direction={target_text}, speed={speed_text}, obstacle_count={obstacle_count}"
        )
        log_step(
            f"当前开车帧日志：{observation}",
            target=f"当前开车分支：{situation}",
            action=decision,
            method=method,
            result="等待本帧驾驶动作执行后由下一帧重新识别路况/位置",
        )
        worker = getattr(self, "_frame_worker", None)
        frame_logger = getattr(worker, "frame_log", None)
        if callable(frame_logger):
            frame_logger(
                f"开车内部判断：{situation}；车辆位置={context.location}，方向={context.direction:.1f}，"
                f"目标方向={target_text}，障碍数={obstacle_count}，视觉建议={context.decision}；接下来{decision}"
            )
        setter = getattr(worker, "set_frame_decision", None)
        if callable(setter):
            setter(
                observation=observation,
                target=f"当前开车分支：{situation}",
                decision=decision,
                action=decision,
                method=method,
                result="等待本帧驾驶动作执行后由下一帧重新识别路况/位置",
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
        self._repair_route_after_deviation(location)
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
        log_step(
            f"当前开车要规划进圈路径：current_loc={location}, circle_angle={self.stable_circle_angle}, "
            f"target_distance={target_dist:.2f}, target_loc={target_point}",
            target="开车路径规划",
            action="先按圈角和目标距离计算地图目标点，再调用 A* 规划路线",
            method="get_target_point() + MapNavigator.plan_path()",
            result="规划成功后使用第一个路径点作为驾驶目标方向",
        )
        planned_path = self.map_tool.plan_path(location, target_point) if target_point is not None else []
        self.road_list = [tuple(map(int, point)) for point in planned_path if point is not None]
        self.last_planned_circle_angle = self.stable_circle_angle

        if self.road_list:
            log_step(
                f"开车路径规划成功：current_loc={location}, target_loc={target_point}, "
                f"path_points={len(self.road_list)}, first_waypoint={self.road_list[0]}",
                target="开车路径规划",
                action="进入路径巡航",
                method="self.road_list = planned_path",
                result="下一帧会用 first_waypoint 计算 target_direction",
            )
            print(f"[Driving] 路径已加载: {self.road_list}")
            try:
                draw_points_with_arrows(self.road_list)
            except Exception as exc:
                print(f"[Driving] 绘制路径调试图失败: {exc}")
        else:
            log_step(
                f"开车路径规划失败：current_loc={location}, target_loc={target_point}",
                target="开车路径规划",
                action="回退自由巡航",
                method="MapNavigator.plan_path()",
                result="本帧直接使用 stable_circle_angle 作为目标方向",
            )
            print("[Driving] 路径规划失败，回退自由巡航")

    def _repair_route_after_deviation(self, location: Tuple[int, int]):
        if not self.road_list:
            return

        check_points = self.road_list[:self.ROUTE_DEVIATION_LOOKAHEAD_POINTS]
        nearest_idx, nearest_dist = min(
            enumerate(check_points),
            key=lambda item: get_distance(location, item[1]),
        )

        if nearest_idx > 0:
            skipped = self.road_list[:nearest_idx]
            del self.road_list[:nearest_idx]
            print(
                f"[Driving] 车辆已越过部分路径点，跳过 {len(skipped)} 个旧路径点，"
                f"nearest_dist={nearest_dist:.2f}"
            )

        if nearest_dist <= self.ROUTE_DEVIATION_REPLAN_DISTANCE:
            return

        if not self.route_replan_cooldown.try_acquire(self.ROUTE_REPLAN_COOLDOWN_S):
            return

        print(
            f"[Driving] 当前位置偏离规划路线过远 nearest_dist={nearest_dist:.2f}，"
            "从当前位置重规划进圈路线"
        )
        self._load_path(location)

    def _record_local_avoidance(self, context: DriveContext, reason: str):
        location = context.location
        if self.last_avoidance_location is None:
            self.local_avoidance_fail_count = 1
        elif get_distance(location, self.last_avoidance_location) <= self.LOCAL_AVOIDANCE_REPEAT_DISTANCE:
            self.local_avoidance_fail_count += 1
        else:
            self.local_avoidance_fail_count = 1

        self.last_avoidance_location = location
        print(
            f"[Driving] 记录局部避障: reason={reason}, "
            f"count={self.local_avoidance_fail_count}, loc={location}"
        )

        if self.local_avoidance_fail_count < self.LOCAL_AVOIDANCE_REPEAT_LIMIT:
            return

        self.local_avoidance_fail_count = 0
        self._recover_route_after_repeated_avoidance(location, reason)

    def _recover_route_after_repeated_avoidance(self, location: Tuple[int, int], reason: str):
        if not self.road_list:
            self.last_planned_circle_angle = None
            print(f"[Driving] 避障反复失败但当前无路径点，等待下一轮重规划: reason={reason}")
            return

        skipped = 0
        min_skip = min(self.ROUTE_SKIP_AFTER_AVOIDANCE, len(self.road_list))
        while self.road_list and skipped < min_skip:
            self.road_list.pop(0)
            skipped += 1

        while self.road_list and get_distance(location, self.road_list[0]) <= self.ROUTE_BLOCKED_POINT_RADIUS:
            self.road_list.pop(0)
            skipped += 1

        self.route_replan_cooldown.trigger()
        if not self.road_list:
            self.last_planned_circle_angle = None

        print(
            f"[Driving] 同一区域避障反复失败，跳过疑似受阻路径点: "
            f"reason={reason}, skipped={skipped}, remain={len(self.road_list)}"
        )

    def _handle_first_car_alignment(self, w: "FrameWorker", context: DriveContext):
        if self.exit_garage_start_location is None:
            self.exit_garage_start_location = context.location

        if self.exit_garage_phase == 0:
            print(f"[Driving] 首次出库第 1 步：固定前进 {self.EXIT_GARAGE_INITIAL_FORWARD_MS}ms 后点击刹车")
            self._log_drive_state(
                "首次出库",
                context,
                f"forward({self.EXIT_GARAGE_INITIAL_FORWARD_MS}ms)+brake_click",
                self.stable_circle_angle,
            )
            self._tap_single_control(w, "up", wait=self.EXIT_GARAGE_INITIAL_FORWARD_MS, dura=100)
            self._click_control(w, "brake")
            self.exit_garage_phase = 1
            return

        if self.exit_garage_phase == 1:
            print(f"[Driving] 首次出库第 2 步：倒车并向左打方向 {self.EXIT_GARAGE_INITIAL_REVERSE_LEFT_MS}ms")
            self._log_drive_state(
                "首次出库",
                context,
                f"reverse_turn_left({self.EXIT_GARAGE_INITIAL_REVERSE_LEFT_MS}ms)",
                self.stable_circle_angle,
            )
            self._tap_double_control(w, "down", "left", wait=self.EXIT_GARAGE_INITIAL_REVERSE_LEFT_MS)
            self.exit_garage_phase = 2
            return

        distance_from_start = get_distance(context.location, self.exit_garage_start_location)
        wall_state = self._analyze_exit_garage_stone_walls(w)
        front_blocked = wall_state["center_blocked"]
        left_blocked = wall_state["left_blocked"]
        right_blocked = wall_state["right_blocked"]

        print(
            "[Driving] 出库视觉判断: "
            f"front_blocked={front_blocked}, left_blocked={left_blocked}, right_blocked={right_blocked}, "
            f"gate_found={wall_state['gate_found']}, gate_offset={wall_state['gate_center_offset']:.3f}, "
            f"gate_width={wall_state['gate_width_ratio']:.3f}, "
            f"distance={distance_from_start:.2f}, clear_rounds={self.exit_garage_clear_rounds}"
        )

        if (
            distance_from_start >= self.EXIT_GARAGE_SUCCESS_DISTANCE
            or (
                not front_blocked
                and self.exit_garage_clear_rounds >= self.EXIT_GARAGE_CLEAR_ROUNDS_TO_CRUISE
                and distance_from_start >= (self.EXIT_GARAGE_SUCCESS_DISTANCE * 0.5)
            )
        ):
            self._finish_exit_garage()
            print("[Driving] 出库完成，进入巡航阶段")
            return

        if not front_blocked:
            self.exit_garage_clear_rounds += 1
            self._log_drive_state(
                "出库阶段前方无石墙",
                context,
                f"forward({self.EXIT_GARAGE_VISUAL_FORWARD_MS}ms)",
                self.stable_circle_angle,
            )
            self._tap_single_control(w, "up", wait=self.EXIT_GARAGE_VISUAL_FORWARD_MS, dura=100)
            return

        if wall_state["gate_found"]:
            self.exit_garage_clear_rounds = 0
            offset = float(wall_state["gate_center_offset"])

            if abs(offset) <= self.EXIT_GARAGE_GATE_CENTER_DEADZONE:
                self._log_drive_state(
                    "出库阶段出口居中",
                    context,
                    f"forward({self.EXIT_GARAGE_CENTERING_FORWARD_MS}ms)",
                    self.stable_circle_angle,
                )
                print(
                    "[Driving] 出库阶段检测到左右石墙形成出口，出口基本居中，"
                    f"gate_offset={offset:.3f}"
                )
                self._tap_single_control(
                    w,
                    "up",
                    wait=self.EXIT_GARAGE_CENTERING_FORWARD_MS,
                    dura=100,
                )
                return

            steer = "right" if offset > 0 else "left"
            duration = self._get_exit_gate_centering_duration(offset)
            self._log_drive_state(
                "出库阶段对准车库出口",
                context,
                f"forward_turn_{steer}({duration}ms)",
                self.stable_circle_angle,
            )
            print(
                f"[Driving] 出库阶段检测到左右石墙形成出口，微调对准中间: "
                f"steer={steer}, duration={duration}, gate_offset={offset:.3f}, "
                f"gate_width={wall_state['gate_width_ratio']:.3f}"
            )
            self._tap_double_control(w, "up", steer, wait=duration)
            return

        self.exit_garage_clear_rounds = 0

        left_score = float(wall_state["left_score"])
        right_score = float(wall_state["right_score"])
        if not left_blocked and right_blocked:
            steer = "left"
        elif not right_blocked and left_blocked:
            steer = "right"
        else:
            steer = "left" if left_score < right_score else "right"

        self._log_drive_state(
            "出库阶段前方存在石墙",
            context,
            f"forward_turn_{steer}({self.EXIT_GARAGE_VISUAL_TURN_MS}ms)",
            self.stable_circle_angle,
        )
        print(
            f"[Driving] 出库阶段根据石墙分布调整方向: steer={steer}, "
            f"left_score={left_score:.3f}, right_score={right_score:.3f}"
        )
        self._tap_double_control(w, "up", steer, wait=self.EXIT_GARAGE_VISUAL_TURN_MS)

    def _get_exit_gate_centering_duration(self, offset: float) -> int:
        ratio = min(1.0, abs(offset) / 0.25)
        span = self.EXIT_GARAGE_CENTERING_MAX_TURN_MS - self.EXIT_GARAGE_CENTERING_MIN_TURN_MS
        return int(round(self.EXIT_GARAGE_CENTERING_MIN_TURN_MS + span * ratio))

    def _handle_forbidden_escape(self, w: "FrameWorker", context: DriveContext) -> bool:
        if self.map_tool.is_walkable(context.location):
            self.forbidden_escape_failures = 0
            self.forbidden_escape_last_distance = None
            self._reset_forbidden_escape_action_state()
            return False

        self._record_local_avoidance(context, "已进入不可通行区域")
        safe_point = self.map_tool.get_nearest_safe_point(context.location, max_search_dist=120)
        progress_text = ""

        if safe_point is None:
            self.forbidden_escape_failures += 1
            self.forbidden_escape_last_distance = None
            progress_text = "safe_point=None"
        else:
            safe_distance = get_distance(context.location, safe_point)
            if self.forbidden_escape_last_distance is None:
                self.forbidden_escape_failures = 0
                self._reset_forbidden_escape_action_state()
                progress_text = f"safe_dist={safe_distance:.2f}, first_track"
            elif safe_distance <= self.forbidden_escape_last_distance - self.FORBIDDEN_ESCAPE_PROGRESS_EPS:
                self.forbidden_escape_failures = 0
                self._reset_forbidden_escape_action_state()
                progress_text = (
                    f"safe_dist={safe_distance:.2f}, "
                    f"prev={self.forbidden_escape_last_distance:.2f}, progress"
                )
            else:
                self.forbidden_escape_failures += 1
                progress_text = (
                    f"safe_dist={safe_distance:.2f}, "
                    f"prev={self.forbidden_escape_last_distance:.2f}, no_progress"
                )
            self.forbidden_escape_last_distance = safe_distance

        print(
            "[Driving] 已进入不可通行区域，执行角度脱离 "
            f"attempt={self.forbidden_escape_failures}, {progress_text}"
        )

        if self.forbidden_escape_failures >= self.FORBIDDEN_ESCAPE_FAIL_LIMIT:
            print("[Driving] 不可通行区域脱困多次失败，结束当前局")
            self._handle_death(w)
            return True

        if safe_point is None:
            print("[Driving] 黑区内未找到安全点，先执行保守倒车脱离")
            self._log_drive_state("已进入不可通行区域", context, "backward(900ms)", self.stable_circle_angle)
            self._record_forbidden_escape_action("backward")
            self._tap_single_control(w, "down", wait=900, dura=100)
            return True

        target_direction = calculate_angle(context.location, safe_point)
        turn_dir, _, diff = calculate_move_count(context.direction, target_direction)
        use_backward = diff > 90

        if use_backward:
            if turn_dir is None or diff <= self.ESCAPE_ALIGN_TOLERANCE:
                action_text = "backward(900ms)"
                print(
                    f"[Driving] 黑区脱离: safe_point={safe_point}, target={target_direction:.1f}, "
                    f"diff={diff:.2f}, action={action_text}"
                )
                self._log_drive_state("已进入不可通行区域", context, action_text, target_direction)
                self._record_forbidden_escape_action("backward")
                self._tap_single_control(w, "down", wait=900, dura=100)
                return True

            action = f"backward_turn_{turn_dir}"
            fallback_duration = self._get_turn_duration(diff, fine=False, reverse=True)
            max_duration = self.FORBIDDEN_TURN_MAX_DURATION_MS
        else:
            if turn_dir is None or diff <= self.ESCAPE_ALIGN_TOLERANCE:
                action_text = "forward(700ms)"
                print(
                    f"[Driving] 黑区脱离: safe_point={safe_point}, target={target_direction:.1f}, "
                    f"diff={diff:.2f}, action={action_text}"
                )
                action = "forward"
                duration = 700
                recovery_action = self._forbidden_escape_obstacle_recovery_action(context, action)
                if recovery_action:
                    duration = self.FORWARD_BLOCK_BACKWARD_TURN_MS
                    action_text = f"{recovery_action}({duration}ms)"
                    print(f"[Driving] 黑区脱离前进受阻，切换倒车语义避障: {action_text}")
                    self._log_drive_state("黑区脱离前进受阻", context, action_text, target_direction)
                    self._record_forbidden_escape_action(recovery_action)
                    self._execute_maneuver(
                        w,
                        recovery_action,
                        speed=context.speed,
                        duration=duration,
                        brake_with_steer=True,
                    )
                    return True

                self._log_drive_state("已进入不可通行区域", context, action_text, target_direction)
                self._record_forbidden_escape_action(action)
                self._tap_single_control(w, "up", wait=duration, dura=100)
                return True

            action = f"forward_turn_{turn_dir}"
            fallback_duration = self._get_turn_duration(diff, fine=False)
            max_duration = self.ROUTE_TURN_MAX_DURATION_MS

        duration = self._get_calibrated_turn_duration(
            action,
            context.speed,
            diff,
            fallback_duration,
            max_duration,
        )
        recovery_action = self._forbidden_escape_obstacle_recovery_action(context, action)
        if recovery_action:
            print(
                f"[Driving] 黑区脱离连续 {action} 未摆脱前方障碍，"
                f"切换为语义 {recovery_action}"
            )
            action = recovery_action
            duration = self.FORWARD_BLOCK_BACKWARD_TURN_MS
            max_duration = self.FORBIDDEN_TURN_MAX_DURATION_MS

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
        self._record_forbidden_escape_action(action)
        self._execute_calibrated_turn(
            w,
            context,
            action,
            duration,
            brake_with_steer=True,
            skip_obstacle_learning=False,
        )
        return True

    def _handle_forbidden_ahead(self, w: "FrameWorker", context: DriveContext) -> bool:
        if not self._has_forbidden_ahead(context):
            return False

        self._record_local_avoidance(context, "前方不可通行区域")
        sector = self.map_tool.get_avoidance_action(context.location, context.direction)
        action_map = {
            1: ("forward_turn_right", 220),
            2: ("forward_turn_right", 450),
            3: ("backward_turn_right", 1500),
            4: ("forward_turn_left", 220),
            5: ("forward_turn_left", 450),
            6: ("backward_turn_left", 1500),
        }
        action = action_map.get(sector)
        if action is None:
            return False

        action_name = action[0]
        desired_degrees = self._avoidance_action_degrees(action_name, action[1])
        max_duration = (
            self.FORBIDDEN_TURN_MAX_DURATION_MS
            if action_name.startswith("backward_turn_")
            else self.ROUTE_TURN_MAX_DURATION_MS
        )
        duration = self._get_calibrated_turn_duration(
            action_name,
            context.speed,
            desired_degrees,
            action[1],
            max_duration,
        )
        if context.speed is not None and context.speed >= 3:
            print("[Driving] 高速接近不可通行区域，先执行预刹车")
            self._tap_single_control(w, "brake", wait=self.FORBIDDEN_HIGH_SPEED_BRAKE_WAIT)
            if action_name.startswith("forward_turn_"):
                duration += 150
            elif action_name.startswith("backward_turn_"):
                duration += 250
            duration = min(duration, max_duration)

        print(f"[Driving] 前方不可通行，地图规避 sector={sector}, action={action}")
        self._log_drive_state(
            "前方有不可通行区域",
            context,
            f"{action_name}({duration}ms)",
            self.stable_circle_angle,
        )
        self._execute_calibrated_turn(
            w,
            context,
            action_name,
            duration=duration,
            brake_with_steer=True,
            skip_obstacle_learning=False,
        )
        return True

    def _handle_visual_avoidance(self, w: "FrameWorker", context: DriveContext) -> bool:
        decision = context.decision or "straight"
        if decision == "straight":
            return False

        hard_obstacle_count = int(context.obstacle_info.get("hard_obstacles_count", 0) or 0)
        coverage_ratio = float(context.obstacle_info.get("coverage_ratio", 0.0) or 0.0)
        severe_block = hard_obstacle_count > 0 or coverage_ratio >= 0.45
        action_map = {
            "slight_left": ("forward_turn_left", 220 if severe_block else 160),
            "small_left": ("forward_turn_left", 380 if severe_block else 300),
            "large_left": ("forward_turn_left", 620 if severe_block else 500),
            "slight_right": ("forward_turn_right", 220 if severe_block else 160),
            "small_right": ("forward_turn_right", 380 if severe_block else 300),
            "large_right": ("forward_turn_right", 620 if severe_block else 500),
            "reverse_and_left": ("reverse_and_left", 1100 if severe_block else 750),
            "reverse_and_right": ("reverse_and_right", 1100 if severe_block else 750),
        }
        action = action_map.get(decision)
        if action is None:
            return False

        self._record_local_avoidance(context, f"视觉避障:{decision}")
        action_name = action[0]
        if action_name.startswith("forward_turn_") and self._is_close_visual_block(context):
            steer = self._decision_to_steer(decision) or self._choose_less_obstructed_side(w)
            action_name = f"reverse_and_{steer}"
            print(
                f"[Driving] 障碍物已贴近车头，放弃前进转向，先倒车脱离: "
                f"decision={decision}, steer={steer}, "
                f"bottom={float(context.obstacle_info.get('max_bottom_ratio', 0.0) or 0.0):.2f}, "
                f"area={float(context.obstacle_info.get('max_area_ratio', 0.0) or 0.0):.2f}, "
                f"coverage={coverage_ratio:.2f}, stalled={self.motion_stalled_this_frame}"
            )
        desired_degrees = self.VISUAL_AVOIDANCE_ANGLE_MAP.get(decision, 25)
        max_duration = (
            self.VISUAL_AVOIDANCE_TURN_MAX_DURATION_MS
            if action_name.startswith(("backward_turn_", "reverse_and_"))
            else self.ROUTE_TURN_MAX_DURATION_MS
        )
        duration = self._get_calibrated_turn_duration(
            action_name,
            context.speed,
            desired_degrees,
            max(action[1], self.VISUAL_CLOSE_REVERSE_MS) if action_name.startswith("reverse_and_") else action[1],
            max_duration,
        )
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
            f"{action_name}({duration}ms)",
            self.stable_circle_angle,
        )
        self._execute_calibrated_turn(
            w,
            context,
            action_name,
            duration=duration,
            brake_with_steer=True,
            skip_obstacle_learning=False,
        )
        return True

    def _is_close_visual_block(self, context: DriveContext) -> bool:
        obstacle_info = context.obstacle_info or {}
        coverage_ratio = float(obstacle_info.get("coverage_ratio", 0.0) or 0.0)
        max_bottom_ratio = float(obstacle_info.get("max_bottom_ratio", 0.0) or 0.0)
        max_area_ratio = float(obstacle_info.get("max_area_ratio", 0.0) or 0.0)
        center_blocked = bool(obstacle_info.get("center_blocked"))
        near_center_bottom_blocked = bool(obstacle_info.get("near_center_bottom_blocked"))
        hard_obstacle_count = int(obstacle_info.get("hard_obstacles_count", 0) or 0)

        if coverage_ratio >= self.VISUAL_CLOSE_COVERAGE_RATIO:
            return True
        if near_center_bottom_blocked:
            return True
        if (
            max_bottom_ratio >= self.VISUAL_CLOSE_BOTTOM_RATIO
            and (max_area_ratio >= self.VISUAL_CLOSE_AREA_RATIO or center_blocked or hard_obstacle_count > 0)
        ):
            return True
        if self.motion_stalled_this_frame and (center_blocked or coverage_ratio >= 0.30):
            return True
        if self.blocked_motion_count >= self.FORWARD_BLOCK_REPEAT_LIMIT and (center_blocked or coverage_ratio >= 0.30):
            return True
        return False

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
        log_step(
            f"当前开车要计算目标方向：current_loc={context.location}, "
            f"current_dir={context.direction}, target_direction={target_direction}, "
            f"angle_diff={diff}, turn_dir={turn_dir}, speed={context.speed}",
            target="开车目标方向计算",
            action="判断直行还是带方向转向",
            method="calculate_move_count(context.direction, target_direction)",
            result="角度差小于阈值则直行，否则按校准表计算转向时长",
        )
        print(
            f"[Driving] 目标对齐: current={context.direction}, "
            f"target={target_direction}, diff={diff:.2f}"
        )

        if turn_dir is None or diff <= self.ALIGN_THRESHOLD:
            self._log_drive_state("前方无障碍物", context, "straight", target_direction)
            self._execute_maneuver(w, "straight", context.speed)
            return

        action = f"forward_turn_{turn_dir}"
        fallback_duration = self._get_turn_duration(diff, fine=False)
        duration = self._get_calibrated_turn_duration(
            action,
            context.speed,
            diff,
            fallback_duration,
            self.ROUTE_TURN_MAX_DURATION_MS,
        )

        self._log_drive_state("发现进圈角度，进行对齐", context, f"{action}({duration}ms)", target_direction)
        self._execute_calibrated_turn(
            w,
            context,
            action,
            duration,
            brake_with_steer=False,
            skip_obstacle_learning=True,
        )

    def _get_calibrated_turn_duration(
        self,
        action: str,
        speed: Optional[int],
        desired_degrees: float,
        fallback_duration: int,
        max_duration: int,
    ) -> int:
        return self.turn_calibration.estimate_duration(
            action,
            speed,
            desired_degrees,
            fallback_duration,
            max_duration_ms=max_duration,
        )

    def _avoidance_action_degrees(self, action: str, fallback_duration: int) -> float:
        if action.startswith("forward_turn_"):
            return max(8.0, min(70.0, float(fallback_duration) / 10.0))
        if action.startswith(("backward_turn_", "reverse_and_")):
            return max(15.0, min(90.0, float(fallback_duration) / 14.0))
        return max(8.0, min(70.0, float(fallback_duration) / 10.0))

    def _execute_calibrated_turn(
        self,
        w: "FrameWorker",
        context: DriveContext,
        action: str,
        duration: int,
        brake_with_steer: bool = False,
        skip_obstacle_learning: bool = False,
    ):
        before_direction = context.direction
        self._execute_maneuver(
            w,
            action,
            speed=context.speed,
            duration=duration,
            brake_with_steer=brake_with_steer,
        )

        if not w.refresh_frame():
            return
        self._frame_action_executed = False

        after_direction = self._get_scalar(w.get_info("direction"))
        if after_direction is None:
            print("[TurnCalibration] 本次转向后方向无效，跳过学习")
            return
        if skip_obstacle_learning and self._has_front_obstacle(context):
            print("[TurnCalibration] 本次转向前存在障碍/黑区风险，跳过学习")
            return

        self.turn_calibration.observe(
            action,
            context.speed,
            before_direction,
            after_direction,
            duration,
        )

    def _get_exit_alignment_duration(self, diff: float) -> int:
        if diff <= 0:
            return 0
        return max(50, min(600, int(round(diff * 10))))

    def _execute_exit_garage_alignment(self, w: "FrameWorker", turn_dir: str, duration: int):
        steer_key = "left" if turn_dir == "left" else "right"
        self._tap_double_control(w, "up", steer_key, wait=duration)
        self._tap_single_control(w, "brake", wait=500)

    def _finish_exit_garage(self):
        self.is_first_car = False
        self.current_stage = self.STAGE_CRUISE
        self.exit_garage_phase = 0
        self.exit_garage_clear_rounds = 0
        self.exit_garage_start_location = None
        self.last_motion_location = None
        self.blocked_motion_count = 0

    def _analyze_exit_garage_stone_walls(self, w: "FrameWorker") -> Dict[str, float]:
        detections = w.get_info("forward_scene")
        frame = getattr(w, "frame", None)
        if not detections or frame is None or not hasattr(frame, "shape"):
            return {
                "left_score": 0.0,
                "center_score": 0.0,
                "right_score": 0.0,
                "left_blocked": False,
                "center_blocked": False,
                "right_blocked": False,
                "gate_found": False,
                "gate_center_offset": 0.0,
                "gate_width_ratio": 0.0,
            }

        frame_h, frame_w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.DRIVE_VIEW_RECT
        roi_x1 = float(frame_w) * float(rx1)
        roi_y1 = float(frame_h) * float(ry1)
        roi_x2 = float(frame_w) * float(rx2)
        roi_y2 = float(frame_h) * float(ry2)
        roi_w = max(1.0, roi_x2 - roi_x1)
        roi_h = max(1.0, roi_y2 - roi_y1)

        sector_scores = {name: 0.0 for name in self.EXIT_GARAGE_SECTORS}
        left_wall_inner_edge: Optional[float] = None
        right_wall_inner_edge: Optional[float] = None

        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 6:
                continue

            x1, y1, x2, y2, conf, cls_id = det[:6]
            if int(cls_id) not in self.EXIT_GARAGE_WALL_CLASS_IDS:
                continue
            if float(conf) < self.EXIT_GARAGE_WALL_MIN_CONF:
                continue

            inter_x1 = max(float(x1), roi_x1)
            inter_y1 = max(float(y1), roi_y1)
            inter_x2 = min(float(x2), roi_x2)
            inter_y2 = min(float(y2), roi_y2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            local_x1 = inter_x1 - roi_x1
            local_x2 = inter_x2 - roi_x1
            local_y2 = inter_y2 - roi_y1
            width_ratio = (local_x2 - local_x1) / roi_w
            area_ratio = ((local_x2 - local_x1) * (inter_y2 - inter_y1)) / (roi_w * roi_h)
            bottom_ratio = local_y2 / roi_h

            if width_ratio < self.EXIT_GARAGE_WALL_MIN_WIDTH_RATIO:
                continue
            if area_ratio < self.EXIT_GARAGE_WALL_MIN_AREA_RATIO:
                continue
            if bottom_ratio < self.EXIT_GARAGE_WALL_MIN_BOTTOM_RATIO:
                continue

            center_ratio = ((local_x1 + local_x2) * 0.5) / roi_w
            if center_ratio < 0.5:
                left_wall_inner_edge = (
                    local_x2
                    if left_wall_inner_edge is None
                    else max(left_wall_inner_edge, local_x2)
                )
            else:
                right_wall_inner_edge = (
                    local_x1
                    if right_wall_inner_edge is None
                    else min(right_wall_inner_edge, local_x1)
                )

            for name, (start_ratio, end_ratio) in self.EXIT_GARAGE_SECTORS.items():
                sector_x1 = roi_w * float(start_ratio)
                sector_x2 = roi_w * float(end_ratio)
                overlap = max(0.0, min(local_x2, sector_x2) - max(local_x1, sector_x1))
                if overlap <= 0:
                    continue
                sector_width = max(1.0, sector_x2 - sector_x1)
                sector_scores[name] += overlap / sector_width

        gate_found = False
        gate_center_offset = 0.0
        gate_width_ratio = 0.0
        if left_wall_inner_edge is not None and right_wall_inner_edge is not None:
            gate_width = max(0.0, right_wall_inner_edge - left_wall_inner_edge)
            gate_width_ratio = gate_width / roi_w
            if gate_width_ratio >= self.EXIT_GARAGE_GATE_MIN_WIDTH_RATIO:
                gate_center = (left_wall_inner_edge + right_wall_inner_edge) * 0.5
                gate_center_offset = (gate_center / roi_w) - 0.5
                gate_found = True

        return {
            "left_score": sector_scores["left"],
            "center_score": sector_scores["center"],
            "right_score": sector_scores["right"],
            "left_blocked": sector_scores["left"] >= self.EXIT_GARAGE_WALL_BLOCK_RATIO,
            "center_blocked": sector_scores["center"] >= self.EXIT_GARAGE_WALL_BLOCK_RATIO,
            "right_blocked": sector_scores["right"] >= self.EXIT_GARAGE_WALL_BLOCK_RATIO,
            "gate_found": gate_found,
            "gate_center_offset": gate_center_offset,
            "gate_width_ratio": gate_width_ratio,
        }

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
        self.motion_stalled_this_frame = False
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
            self.motion_stalled_this_frame = True
        else:
            self.blocked_motion_count = 1

        self.last_motion_location = context.location
        return self.blocked_motion_count >= self.STUCK_REPEAT_LIMIT

    def _check_forward_motion_block(self, context: DriveContext) -> bool:
        if self.current_stage in (self.STAGE_EXIT_GARAGE, self.STAGE_FINISH):
            return False
        if self.forward_block_recovery_active:
            return False
        if self.last_motion_mode != "forward":
            return False
        if self.last_motion_location is None:
            return False

        stuck = get_distance(context.location, self.last_motion_location) <= self.STUCK_LOCATION_EPS
        return stuck and self.blocked_motion_count >= self.FORWARD_BLOCK_REPEAT_LIMIT

    def _check_probable_no_fuel(self, context: DriveContext) -> bool:
        if self.current_stage in (self.STAGE_EXIT_GARAGE, self.STAGE_FINISH):
            self.no_fuel_stall_count = 0
            return False
        if self.last_motion_mode != "forward":
            self.no_fuel_stall_count = 0
            return False
        if not self._is_open_forward_stall(context):
            self.no_fuel_stall_count = 0
            return False
        if not self.motion_stalled_this_frame:
            self.no_fuel_stall_count = 0
            return False

        self.no_fuel_stall_count += 1
        if self.no_fuel_stall_count < self.NO_FUEL_FORWARD_STALL_LIMIT:
            return False

        print(
            "[Driving] 疑似没油判定成立: "
            f"stall_frames={self.no_fuel_stall_count}, loc={context.location}, "
            f"circle={self._current_circle_angle_text()}"
        )
        log_step(
            f"当前开车帧日志：疑似车辆没油：连续前进不动帧数={self.no_fuel_stall_count}，"
            f"当前位置={context.location}，当前方位={context.direction:.1f}，"
            f"速度={context.speed}，圈角={self._current_circle_angle_text()}，"
            f"视觉决策={context.decision}",
            target="当前开车分支：疑似车辆没油",
            action="准备刹车下车并切回跑图",
            method=(
                "_check_probable_no_fuel: last_motion_mode=forward, open_forward_stall=True, "
                f"motion_stalled_this_frame={self.motion_stalled_this_frame}"
            ),
            result="交给 _exit_vehicle_to_running 决定是否继续寻车",
        )
        return True

    def _is_open_forward_stall(self, context: DriveContext) -> bool:
        if not self.map_tool.is_walkable(context.location):
            return False
        if self._has_forbidden_ahead(context):
            return False
        if (context.decision or "straight") != "straight":
            return False

        obstacle_count = int(context.obstacle_info.get("obstacles_count", 0) or 0)
        hard_obstacle_count = int(context.obstacle_info.get("hard_obstacles_count", 0) or 0)
        coverage_ratio = float(context.obstacle_info.get("coverage_ratio", 0.0) or 0.0)
        if obstacle_count > 0 or hard_obstacle_count > 0:
            return False
        return coverage_ratio <= self.NO_FUEL_MAX_OBSTACLE_COVERAGE

    def _should_find_car_after_no_fuel(self) -> bool:
        if self._should_continue_finding_car_after_running_fallback("疑似没油"):
            print("[Driving] 疑似没油但开车阶段未完成，切跑图后继续沿指定路线找车")
            return True

        print("[Driving] 疑似没油且开车阶段已完成，切跑图后再考虑进圈")
        return False

    def _should_continue_finding_car_after_running_fallback(self, reason: str) -> bool:
        remaining = max(0.0, self.max_driving_time - self.get_driving_elapsed_time())
        should_find = remaining > 0.0
        print(
            f"[Driving] {reason}回退跑图决策: "
            f"driving_remaining={remaining:.2f}s, "
            f"next={'继续找车' if should_find else '跑图/进圈'}"
        )
        return should_find

    def _current_circle_angle_text(self) -> str:
        circle_angle = self.stable_circle_angle
        if circle_angle is None:
            circle_angle = self.latest_circle_angle
        return "None" if circle_angle is None else f"{circle_angle:.1f}"

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

    def _try_house_exit_when_indoor(self, w: "FrameWorker", reason: str) -> bool:
        if self._get_house_scene(w) != HouseExitManager.HOUSE_INDOOR:
            return False

        print(f"[Driving] {reason}且 house_scene=indoor，优先使用 HouseExitManager 脱困")
        self._tap_single_control(w, "brake", wait=300, dura=80)
        self.house_exit_manager.reset()
        for _ in range(20):
            if self.house_exit_manager.process(w):
                print("[Driving] HouseExitManager 出房成功，切回跑图阶段")
                finding_car = self._should_continue_finding_car_after_running_fallback("室内脱困完成")
                self.reset(max_driving_time=self.max_driving_time)
                self.next_running_finding_car = finding_car
                w.change_stage("跑图阶段")
                return True

        print("[Driving] HouseExitManager 暂未出房，切回跑图阶段继续脱困")
        finding_car = self._should_continue_finding_car_after_running_fallback("室内脱困未完成")
        self.reset(max_driving_time=self.max_driving_time)
        self.next_running_finding_car = finding_car
        w.change_stage("跑图阶段")
        return True

    def _handle_forward_motion_block(self, w: "FrameWorker", context: DriveContext):
        if self._try_house_exit_when_indoor(w, "前进卡住"):
            return

        self.forward_block_recovery_active = True
        self.blocked_motion_count = 0

        decision = context.decision or "straight"
        if decision != "straight":
            print(f"[Driving] 前进卡住且当前检测到障碍物，转入视觉避障 decision={decision}")
            self._handle_visual_avoidance(w, context)
        else:
            steer = self._choose_less_obstructed_side(w)
            print(f"[Driving] 前进卡住但前方未检测到明确障碍，倒车并向 {steer} 规避")
            self._record_local_avoidance(context, "前进卡住")
            self._log_drive_state(
                "前进卡住后全场景择路",
                context,
                f"backward_turn_{steer}({self.FORWARD_BLOCK_BACKWARD_TURN_MS}ms)",
                self.stable_circle_angle,
            )
            self._execute_maneuver(
                w,
                f"backward_turn_{steer}",
                speed=context.speed,
                duration=self.FORWARD_BLOCK_BACKWARD_TURN_MS,
                brake_with_steer=True,
            )
            w.refresh_frame()
            updated_context = self._build_context(w) or context
            if (updated_context.decision or "straight") != "straight":
                print(f"[Driving] 倒车规避后检测到障碍物，继续视觉避障 decision={updated_context.decision}")
                self._handle_visual_avoidance(w, updated_context)
            else:
                forward_steer = self._choose_less_obstructed_side(w)
                print(f"[Driving] 倒车后仍无明确障碍，按障碍较少侧前进: {forward_steer}")
                self._log_drive_state(
                    "倒车规避后全场景择路",
                    updated_context,
                    f"forward_turn_{forward_steer}({self.FORWARD_BLOCK_TURN_MS}ms)",
                    self.stable_circle_angle,
                )
                self._execute_maneuver(
                    w,
                    f"forward_turn_{forward_steer}",
                    speed=updated_context.speed,
                    duration=self.FORWARD_BLOCK_TURN_MS,
                    brake_with_steer=True,
                )

        self.forward_block_recovery_active = False

    def _choose_less_obstructed_side(self, w: "FrameWorker") -> str:
        detections = w.get_info("forward_scene")
        frame = getattr(w, "frame", None)
        if not detections or frame is None or not hasattr(frame, "shape"):
            return self._get_opposite_steer(self.last_motion_steer or self._get_default_steer())

        frame_h, frame_w = frame.shape[:2]
        center_x = float(frame_w) / 2.0
        left_score = 0.0
        right_score = 0.0

        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            if int(cls_id) not in ObstacleAvoidanceAnalyzer.CLASS_CONFIG:
                continue
            if float(conf) < 0.25:
                continue

            box_w = max(0.0, float(x2) - float(x1))
            box_h = max(0.0, float(y2) - float(y1))
            if box_w <= 0 or box_h <= 0:
                continue

            area_ratio = (box_w * box_h) / float(max(1, int(frame_w) * int(frame_h)))
            box_center_x = (float(x1) + float(x2)) / 2.0
            if box_center_x < center_x:
                left_score += area_ratio
            else:
                right_score += area_ratio

        if left_score == right_score:
            return self._get_opposite_steer(self.last_motion_steer or self._get_default_steer())
        return "left" if left_score < right_score else "right"

    def _get_opposite_steer(self, steer: Optional[str]) -> str:
        return "left" if steer == "right" else "right"

    def _reset_forbidden_escape_action_state(self):
        self.forbidden_escape_last_action = None
        self.forbidden_escape_action_repeats = 0

    def _record_forbidden_escape_action(self, action: str):
        action = str(action or "")
        if action == self.forbidden_escape_last_action:
            self.forbidden_escape_action_repeats += 1
        else:
            self.forbidden_escape_last_action = action
            self.forbidden_escape_action_repeats = 1

    def _forbidden_escape_obstacle_recovery_action(self, context: DriveContext, action: str) -> Optional[str]:
        if self.forbidden_escape_failures < self.FORBIDDEN_ESCAPE_OBSTACLE_REVERSE_AFTER:
            return None
        if not self._has_visual_front_obstacle(context):
            return None
        if not action.startswith("forward"):
            return None
        if action == "forward":
            steer = self._decision_to_steer(context.decision) or self.last_motion_steer or self._get_default_steer()
        elif action.endswith("_left"):
            steer = "left"
        elif action.endswith("_right"):
            steer = "right"
        else:
            steer = self._get_default_steer()

        if self.forbidden_escape_last_action != action:
            return None
        if self.forbidden_escape_action_repeats < self.FORBIDDEN_ESCAPE_OBSTACLE_REVERSE_AFTER:
            return None
        return f"backward_turn_{steer}"

    @staticmethod
    def _has_visual_front_obstacle(context: DriveContext) -> bool:
        obstacle_count = int(context.obstacle_info.get("obstacles_count", 0) or 0)
        hard_obstacle_count = int(context.obstacle_info.get("hard_obstacles_count", 0) or 0)
        coverage_ratio = float(context.obstacle_info.get("coverage_ratio", 0.0) or 0.0)
        center_blocked = bool(context.obstacle_info.get("center_blocked"))
        near_center_bottom_blocked = bool(context.obstacle_info.get("near_center_bottom_blocked"))
        return (
            obstacle_count > 0
            or hard_obstacle_count > 0
            or coverage_ratio >= 0.20
            or center_blocked
            or near_center_bottom_blocked
            or (context.decision or "straight") != "straight"
        )

    def _handle_motion_block(self, w: "FrameWorker", context: DriveContext):
        if self._try_house_exit_when_indoor(w, "连续多帧位置不变"):
            return

        steer = self.last_motion_steer or self._decision_to_steer(context.decision) or "right"
        action = f"backward_turn_{steer}"

        print(
            f"[Driving] 快速脱困: stuck_frames={self.blocked_motion_count}, "
            f"steer={steer}, action={action}"
        )
        self._record_local_avoidance(context, "连续多帧位置不变")
        self._log_drive_state(
            "车辆连续多帧位置不变",
            context,
            f"{action}({self.MOTION_BLOCK_BACKWARD_TURN_MS}ms)",
            self.stable_circle_angle,
        )
        self.blocked_motion_count = 0
        self._execute_maneuver(
            w,
            action,
            speed=context.speed,
            duration=self.MOTION_BLOCK_BACKWARD_TURN_MS,
            brake_with_steer=True,
        )

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
        self.next_running_finding_car = False
        w.change_stage("跑图阶段")

    def _exit_vehicle_to_running(
        self,
        w: "FrameWorker",
        reason: str,
        finding_car: bool = False,
    ):
        print(f"[Driving] {reason}")
        log_step(
            f"当前开车帧日志：准备从开车切回跑图，原因={reason}，finding_car={finding_car}，"
            f"current_stage={self.current_stage}",
            target="当前开车分支：下车切跑图",
            action="刹车、下车、切换跑图阶段",
            method="_exit_vehicle_to_running: brake -> off_car -> reset -> change_stage(跑图阶段)",
            result="下一帧由 RunningManager 接管，必要时继续寻车",
        )
        self.current_stage = self.STAGE_FINISH
        self._tap_single_control(w, "brake", wait=800, dura=80)
        self._click_control(w, "off_car")
        self.reset(max_driving_time=self.max_driving_time)
        self.next_running_finding_car = bool(finding_car)
        w.change_stage("跑图阶段")

    def _handle_death(self, w: "FrameWorker"):
        if self._click_control_if_configured(w, "spectate"):
            time.sleep(2)
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("结束阶段")

    def _handle_rank_finish(self, w: "FrameWorker"):
        if callable(self.pause_sp_callback):
            self.pause_sp_callback(w)
        else:
            self._frame_action_executed = True
            w.click("sp")
            time.sleep(0.5)
        time.sleep(2)
        self._frame_action_executed = True
        w.click("观战对手")
        self.reset(max_driving_time=self.max_driving_time)
        w.change_stage("结束阶段")

    def _update_circle_angle(self, value):
        angle = self._get_scalar(value)
        if angle is None:
            return

        self.latest_circle_angle = angle
        self.circle_angles.append(angle)
        if len(self.circle_angles) < 30:
            self.stable_circle_angle = None
            return

        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils import stable_angle

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

        if action != "straight":
            self._cancel_drive_auto_forward(w, "执行非直行动作前取消自动前进")

        if self._needs_pre_brake(action, speed):
            brake_wait = 1000 if speed == 2 else 2000
            if brake_with_steer and steer_key is not None:
                self._tap_double_control(w, "brake", steer_key, wait=brake_wait)
            else:
                self._tap_single_control(w, "brake", wait=brake_wait)

        # backward_turn_left/right 和 reverse_and_left/right 的命名语义
        # 表示“车辆实际后退偏向的方向”，不是“倒车时方向键按下的方向”。
        # 因此后退向左脱离会落到 down + right，后退向右脱离会落到 down + left。
        simple_actions = {
            "straight": lambda: self._drive_straight_pulse(w),
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

    def _drive_straight_pulse(self, w: "FrameWorker"):
        print(
            f"[Driving] 前方无障碍，自动前进分段直行: "
            f"on={self.STRAIGHT_AUTO_FORWARD_ON_S:.1f}s, pause={self.STRAIGHT_AUTO_FORWARD_PAUSE_S:.1f}s, "
            f"poll={self.STRAIGHT_AUTO_FORWARD_OBSTACLE_POLL_S:.2f}s"
        )
        self._start_drive_auto_forward(w, "前方无障碍，点击自动前进")

        if self._monitor_drive_auto_forward_for_obstacles(w):
            return

        self._cancel_drive_auto_forward(w, "自动前进4s到时，点击取消")
        if self._wait_drive_auto_forward_pause(w):
            return

        if not self._auto_forward_detected_obstacle(w):
            self._start_drive_auto_forward(w, "空1s后再次点击自动前进")

    def _start_drive_auto_forward(self, w: "FrameWorker", reason: str):
        if self.drive_auto_forward_active:
            return
        print(f"[Driving] {reason}")
        self._click_control(w, "auto_forward")
        self.drive_auto_forward_active = True
        self.drive_auto_forward_started_at = time.monotonic()

    def _cancel_drive_auto_forward(self, w: "FrameWorker", reason: str):
        if not self.drive_auto_forward_active:
            return
        print(f"[Driving] {reason}")
        self._click_control(w, "auto_forward")
        self.drive_auto_forward_active = False
        self.drive_auto_forward_started_at = None

    def _monitor_drive_auto_forward_for_obstacles(self, w: "FrameWorker") -> bool:
        while self.drive_auto_forward_active:
            if self._handle_terminal_state(w, "车辆自动前进监控中"):
                self._cancel_drive_auto_forward(w, "检测到死亡或排名界面，取消车辆自动前进")
                return True

            started_at = self.drive_auto_forward_started_at or time.monotonic()
            elapsed = time.monotonic() - started_at
            remaining = self.STRAIGHT_AUTO_FORWARD_ON_S - elapsed
            if remaining <= 0:
                return False

            time.sleep(min(self.STRAIGHT_AUTO_FORWARD_OBSTACLE_POLL_S, remaining))
            if self._auto_forward_detected_obstacle(w):
                self._cancel_drive_auto_forward(w, "自动前进中实时检测到障碍物，立即取消并交给避障")
                return True

        return False

    def _wait_drive_auto_forward_pause(self, w: "FrameWorker") -> bool:
        deadline = time.monotonic() + self.STRAIGHT_AUTO_FORWARD_PAUSE_S
        while time.monotonic() < deadline:
            time.sleep(min(self.STRAIGHT_AUTO_FORWARD_OBSTACLE_POLL_S, max(0.0, deadline - time.monotonic())))
            if self._handle_terminal_state(w, "车辆自动前进暂停等待中"):
                return True
            if self._auto_forward_detected_obstacle(w):
                print("[Driving] 自动前进暂停期间检测到障碍物，暂不重新开启自动前进")
                return True
        return False

    def _auto_forward_detected_obstacle(self, w: "FrameWorker") -> bool:
        if not w.refresh_frame():
            return False

        obstacle_info = self._analyze_obstacles(w)
        decision = obstacle_info.get("decision", "straight")
        obstacle_count = int(obstacle_info.get("obstacles_count", 0) or 0)
        hard_obstacle_count = int(obstacle_info.get("hard_obstacles_count", 0) or 0)
        coverage_ratio = float(obstacle_info.get("coverage_ratio", 0.0) or 0.0)
        center_blocked = bool(obstacle_info.get("center_blocked"))
        near_center_bottom_blocked = bool(obstacle_info.get("near_center_bottom_blocked"))

        blocked = (
            decision != "straight"
            or obstacle_count > 0
            or hard_obstacle_count > 0
            or coverage_ratio >= 0.20
            or center_blocked
            or near_center_bottom_blocked
        )
        if blocked:
            print(
                f"[Driving] 自动前进轮询发现障碍: decision={decision}, "
                f"obstacles={obstacle_count}, hard={hard_obstacle_count}, "
                f"coverage={coverage_ratio:.2f}, center={center_blocked}, "
                f"near_bottom={near_center_bottom_blocked}"
            )
        return blocked

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
        return self.auto_brake_cooldown.try_acquire(self.BRAKE_COOLDOWN_SPEED2)

    def _record_motion_action(self, action: str):
        self.last_motion_mode = None
        self.last_motion_steer = None

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

    def _handle_terminal_state(self, w: "FrameWorker", context: str) -> bool:
        callback = getattr(self, "terminal_state_callback", None)
        if callable(callback) and callback(w, context):
            self._frame_action_executed = True
            return True

        if self._has_rank_info(w):
            print(f"[Driving] {context}检测到个人排名或队伍排名，进入结束阶段")
            self._handle_rank_finish(w)
            return True

        if self._is_dead(w):
            print(f"[Driving] {context}检测到死亡，结束当前局")
            self._handle_death(w)
            return True

        return False

    def _is_out_of_vehicle(self, w: "FrameWorker") -> bool:
        if self._get_configured_info(w, "驾驶"):
            print("[Driving] 检测到驾驶按钮，判定已意外下车")
            return True

        strong_vehicle_ui_visible = self._has_visible_info(
            w,
            ("喇叭", "speed", "漂移", "加速"),
        )

        if self._is_configured_info(w, "喇叭"):
            if self._get_configured_info(w, "喇叭"):
                self.horn_missing_frames = 0
            else:
                self.horn_missing_frames += 1
                if (
                    self.horn_missing_frames >= self.HORN_MISSING_EXIT_FRAME_LIMIT
                    and not strong_vehicle_ui_visible
                ):
                    print(
                        f"[Driving] 喇叭连续 {self.horn_missing_frames} 帧消失，"
                        "判定可能已下车"
                    )
                    return True
        else:
            self.horn_missing_frames = 0

        on_foot_ui_visible = self._has_visible_info(
            w,
            ("驾驶", "左拳头", "子弹", "跳跃"),
        )
        if not on_foot_ui_visible:
            return False

        vehicle_ui_visible = self._has_visible_info(
            w,
            ("喇叭", "speed", "漂移", "加速"),
        ) or strong_vehicle_ui_visible
        if vehicle_ui_visible:
            return False

        print("[Driving] 检测到步行UI，判定已意外下车")
        return True

    def _is_configured_info(self, w: "FrameWorker", area_name: str) -> bool:
        stage = getattr(w, "current_stage", None)
        resolver = getattr(w, "stage_resolver", None)
        raw_stage_info = getattr(resolver, "stage_info", {})
        stage_data = raw_stage_info.get(stage, {}) if isinstance(raw_stage_info, dict) else {}
        for scene_info in stage_data.get("scenes", {}).values():
            for variant in self._iter_scene_variants(scene_info):
                if area_name in (variant.get("areas") or {}):
                    return True
                if area_name in (variant.get("special_areas") or {}):
                    return True

        suffix = f"__{area_name}"
        return any(
            str(key).endswith(suffix)
            for key in getattr(w, "stage_info", {}).keys()
        )

    def _iter_scene_variants(self, scene_info: Dict[str, Any]):
        resolutions = scene_info.get("resolutions")
        if isinstance(resolutions, dict):
            yield from (item for item in resolutions.values() if isinstance(item, dict))
        elif isinstance(scene_info, dict):
            yield scene_info

    def _get_configured_info(self, w: "FrameWorker", area_name: str):
        if not self._is_configured_info(w, area_name):
            return None
        return w.get_info(area_name)

    def _has_visible_info(self, w: "FrameWorker", names: Tuple[str, ...]) -> bool:
        return any(self._get_configured_info(w, name) is not None for name in names)

    def _has_configured_control(self, w: "FrameWorker", key: str) -> bool:
        aliases = self.CONTROL_ALIASES.get(key, (key,))
        stage = getattr(w, "current_stage", None)
        buttons = getattr(getattr(w, "controller", None), "buttons", {})
        return bool(stage and buttons and any(f"{stage}_{alias}" in buttons for alias in aliases))

    def _click_control_if_configured(self, w: "FrameWorker", key: str, **kwargs) -> bool:
        if not self._has_configured_control(w, key):
            return False
        self._click_control(w, key, **kwargs)
        return True

    def _handle_unexpected_vehicle_exit(self, w: "FrameWorker"):
        finding_car = self._should_continue_finding_car_after_running_fallback("意外下车")
        log_step(
            f"当前开车帧日志：检测到意外下车，finding_car={finding_car}，"
            f"current_stage={self.current_stage}",
            target="当前开车分支：意外下车",
            action="重置开车状态并切回跑图",
            method="_handle_unexpected_vehicle_exit: reset -> change_stage(跑图阶段)",
            result="下一帧由 RunningManager 按步行状态继续处理",
        )
        self.reset(max_driving_time=self.max_driving_time)
        self.next_running_finding_car = finding_car
        w.change_stage("跑图阶段")

    def _analyze_obstacles(self, w: "FrameWorker") -> Dict[str, Any]:
        detections = w.get_info("forward_scene")
        if not detections:
            return {"decision": "straight", "obstacles_count": 0}

        frame = getattr(w, "frame", None)
        if frame is None or not hasattr(frame, "shape"):
            return {"decision": "straight", "obstacles_count": 0}

        frame_h, frame_w = frame.shape[:2]
        roi_x, roi_y, roi_w, roi_h, trapezoid = self._get_drive_obstacle_roi(frame_w, frame_h)

        if (
            self.obstacle_analyzer is None
            or getattr(self.obstacle_analyzer, "W", None) != roi_w
            or getattr(self.obstacle_analyzer, "H", None) != roi_h
        ):
            self.obstacle_analyzer = ObstacleAvoidanceAnalyzer(width=roi_w, height=roi_h)

        local_dets = []
        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            inter_x1 = max(float(x1), float(roi_x))
            inter_y1 = max(float(y1), float(roi_y))
            inter_x2 = min(float(x2), float(roi_x + roi_w))
            inter_y2 = min(float(y2), float(roi_y + roi_h))

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            trap_x1, trap_x2 = self._drive_trapezoid_x_bounds(trapezoid, inter_y2)
            inter_x1 = max(inter_x1, trap_x1)
            inter_x2 = min(inter_x2, trap_x2)
            if inter_x2 <= inter_x1:
                continue

            local_det = [
                inter_x1 - roi_x,
                inter_y1 - roi_y,
                inter_x2 - roi_x,
                inter_y2 - roi_y,
                conf,
                cls_id,
            ]
            if self._is_self_vehicle_detection(local_det, roi_w, roi_h):
                continue

            local_dets.append(local_det)

        if not local_dets:
            return {"decision": "straight", "obstacles_count": 0}

        obstacle_info = self.obstacle_analyzer.analyze(local_dets)
        self._attach_close_obstacle_metrics(obstacle_info, local_dets, roi_w, roi_h)
        return obstacle_info

    def _get_drive_obstacle_roi(self, frame_w: int, frame_h: int):
        rx1, ry1, rx2, ry2 = self.DRIVE_VIEW_RECT
        top_y = min(float(ry1), float(self.DRIVE_VIEW_TRAPEZOID_TOP_Y))
        bottom_y = float(ry2)
        bottom_left_x = float(rx1)
        bottom_right_x = float(rx2)
        center_x = (bottom_left_x + bottom_right_x) * 0.5
        bottom_width = max(0.0, bottom_right_x - bottom_left_x)
        top_width = bottom_width * float(self.DRIVE_VIEW_TRAPEZOID_TOP_WIDTH_SCALE)
        top_left_x = center_x - top_width * 0.5
        top_right_x = center_x + top_width * 0.5

        roi_x = int(frame_w * min(top_left_x, bottom_left_x))
        roi_y = int(frame_h * top_y)
        roi_x2 = int(frame_w * max(top_right_x, bottom_right_x))
        roi_y2 = int(frame_h * bottom_y)
        roi_w = max(1, roi_x2 - roi_x)
        roi_h = max(1, roi_y2 - roi_y)

        trapezoid = {
            "top_y": float(frame_h) * top_y,
            "bottom_y": float(frame_h) * bottom_y,
            "top_left_x": float(frame_w) * top_left_x,
            "top_right_x": float(frame_w) * top_right_x,
            "bottom_left_x": float(frame_w) * bottom_left_x,
            "bottom_right_x": float(frame_w) * bottom_right_x,
        }
        return roi_x, roi_y, roi_w, roi_h, trapezoid

    @staticmethod
    def _drive_trapezoid_x_bounds(trapezoid: Dict[str, float], y: float):
        top_y = float(trapezoid["top_y"])
        bottom_y = float(trapezoid["bottom_y"])
        progress = (float(y) - top_y) / max(1.0, bottom_y - top_y)
        progress = max(0.0, min(1.0, progress))
        left = trapezoid["top_left_x"] + (
            trapezoid["bottom_left_x"] - trapezoid["top_left_x"]
        ) * progress
        right = trapezoid["top_right_x"] + (
            trapezoid["bottom_right_x"] - trapezoid["top_right_x"]
        ) * progress
        return left, right

    def _attach_close_obstacle_metrics(self, obstacle_info: Dict[str, Any], local_dets, roi_w: int, roi_h: int):
        max_bottom_ratio = 0.0
        max_area_ratio = 0.0
        near_center_bottom_blocked = False
        center_x = float(roi_w) / 2.0

        for det in local_dets:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cfg = ObstacleAvoidanceAnalyzer.CLASS_CONFIG.get(int(cls_id))
            if not cfg or float(conf) < 0.25:
                continue

            box_w = max(0.0, float(x2) - float(x1))
            box_h = max(0.0, float(y2) - float(y1))
            if box_w <= 0 or box_h <= 0:
                continue

            bottom_ratio = float(y2) / float(max(1, roi_h))
            area_ratio = (box_w * box_h) / float(max(1, roi_w * roi_h))
            max_bottom_ratio = max(max_bottom_ratio, bottom_ratio)
            max_area_ratio = max(max_area_ratio, area_ratio)

            if float(x1) <= center_x <= float(x2) and bottom_ratio >= self.VISUAL_CLOSE_BOTTOM_RATIO:
                near_center_bottom_blocked = True

        obstacle_info["max_bottom_ratio"] = max_bottom_ratio
        obstacle_info["max_area_ratio"] = max_area_ratio
        obstacle_info["near_center_bottom_blocked"] = near_center_bottom_blocked

    def _is_self_vehicle_detection(self, det, roi_w: int, roi_h: int) -> bool:
        if int(det[5]) not in self.SELF_VEHICLE_CLASS_IDS:
            return False

        x1, _, x2, y2 = map(float, det[:4])
        roi_w = float(max(1, roi_w))
        roi_h = float(max(1, roi_h))

        if y2 < roi_h - self.SELF_VEHICLE_BOTTOM_INTERSECT_TOLERANCE:
            return False

        center_x = roi_w / 2.0
        if not (x1 < center_x < x2):
            return False

        left_cover = center_x - x1
        right_cover = x2 - center_x
        min_side_cover = roi_w * self.SELF_VEHICLE_MIN_SIDE_COVER_RATIO
        min_total_width = roi_w * self.SELF_VEHICLE_MIN_TOTAL_WIDTH_RATIO

        return (
            left_cover >= min_side_cover
            and right_cover >= min_side_cover
            and (x2 - x1) >= min_total_width
        )

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

    def _mark_drive_auto_forward_cancelled_by_manual_control(self, *keys: str):
        if not self.drive_auto_forward_active:
            return
        if keys and all(key == "auto_forward" for key in keys):
            return
        print("[Driving] 检测到其他驾驶按键，标记自动前进已取消")
        self.drive_auto_forward_active = False
        self.drive_auto_forward_started_at = None

    def _tap_single_control(self, w: "FrameWorker", key: str, **kwargs):
        self._mark_drive_auto_forward_cancelled_by_manual_control(key)
        self._frame_action_executed = True
        w.tap_single(self._resolve_control_name(w, key), **kwargs)

    def _tap_double_control(self, w: "FrameWorker", key1: str, key2: str, **kwargs):
        self._mark_drive_auto_forward_cancelled_by_manual_control(key1, key2)
        self._frame_action_executed = True
        w.tap_double(self._resolve_control_name(w, key1), self._resolve_control_name(w, key2), **kwargs)

    def _click_control(self, w: "FrameWorker", key: str, **kwargs):
        self._mark_drive_auto_forward_cancelled_by_manual_control(key)
        self._frame_action_executed = True
        w.click(self._resolve_control_name(w, key), **kwargs)

    def _click_down_control(self, w: "FrameWorker", key: str, **kwargs):
        self._mark_drive_auto_forward_cancelled_by_manual_control(key)
        self._frame_action_executed = True
        w.click_down(self._resolve_control_name(w, key), **kwargs)
