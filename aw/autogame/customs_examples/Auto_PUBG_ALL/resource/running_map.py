import time
from typing import List, Optional, Tuple, TYPE_CHECKING

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.map_navigator import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import (
    calculate_angle,
    calculate_move_count,
    check_location,
    get_distance,
    is_location_stagnant, draw_points_with_arrows,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.utils import find_path

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class RunningManager:
    """
    跑图/徒步管理器。

    在新框架下，本类只依赖 FrameWorker 暴露的统一接口:
    - w.get_info(name)
    - w.click(name)
    - w.tap_single(name, ...)
    - w.change_stage(name)
    - w.refresh_frame()
    """

    STUCK_HISTORY_LEN = 10
    TRAPPED_HISTORY_LEN = 50
    WAYPOINT_TOLERANCE = 2.0
    LOCATION_JUMP_THRESHOLD = 25.0
    LOCATION_JUMP_REPLAN_COOLDOWN = 1.5

    STAGE1_DIS = 600
    STAGE2_DIS = 400
    STAGE3_DIS = 220
    STAGE1_TIME = 11 * 60
    STAGE2_TIME = 16 * 60

    R_CITY = (1136, 783)
    CAR_ENTRY_POINT = (1131, 763)
    CAR_FACE_DIRECTION = 265
    PRECISE_FACE_DIRECTIONS = [265, 270, 275, 280, 285, 290]
    CIRCLE_REPLAN_THRESHOLD = 10
    PRECISE_FORWARD_BIAS_Y = -220
    PRECISE_FORWARD_DURA = 550
    PRECISE_FORWARD_WAIT = 1800
    PRECISE_LATERAL_STEP_BIAS = 150
    PRECISE_LATERAL_STEP_DURA = 220
    PRECISE_LATERAL_STEP_WAIT = 700
    PRECISE_RESET_CENTER_BIAS = -300
    PRECISE_RESET_CENTER_DURA = 260
    PRECISE_RESET_CENTER_WAIT = 700

    def __init__(self, map_tool: Optional[MapNavigator] = None):
        self.map_tool = map_tool or MapNavigator()
        self.game_time : Optional[float] = None

        self.road_list: List[Tuple[int, int]] = []
        self.locations: List[Tuple[int, int]] = []
        self.history_locations: List[Tuple[int, int]] = []

        self.auto_forward = False
        self.stuck = False
        self.trapped = False

        self.stable_circle_angle: Optional[float] = None
        self.find_car_times = 0
        self.correct_position_times = 0
        self.finding_car = True
        self.loading_road = False
        self.precise_entering_car = False
        self.precise_last_distance: Optional[float] = None
        self.precise_idle_rounds = 0
        self.precise_face_attempt_index = 0
        self.last_valid_location: Optional[Tuple[int, int]] = None
        self.last_jump_replan_time: float = 0.0

    def reset(self, finding_car: bool = True):
        self.road_list = []
        self.locations = []
        self.history_locations = []
        self.auto_forward = False
        self.stuck = False
        self.trapped = False
        self.stable_circle_angle = None
        self.find_car_times = 0
        self.correct_position_times = 0
        self.finding_car = finding_car
        self.loading_road = False
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_face_attempt_index = 0
        self.last_valid_location = None
        self.last_jump_replan_time = 0.0
        print("[Running] 状态已重置!")

    def set_game_time(self, game_time: Optional[float] = None):
        self.game_time = time.time() if game_time is None else game_time
        print(f'[Running] 游戏开始时间设置为： {self.game_time:.3f}')

    def get_elapsed_time(self) -> float:
        if self.game_time is None:
            return 0.0
        return max(0.0, time.time() - self.game_time)

    def process(self, w: "FrameWorker"):
        location = self._get_location(w)
        if location is None:
            print("[Running] 位置无效，尝试小幅前探刷新坐标...")
            w.tap_single("摇杆", y_bias=-250, dura=250, wait=500)
            return

        direction = self._get_scalar(w.get_info("direction"))
        self._update_circle_angle(w.get_info("white_angle"))

        if self._is_dead(w):
            print("[Running] 检测到死亡!")
            self._handle_death(w)
            return

        if self._is_in_vehicle(w):
            print("[Running] 检测到已经上车，切换到开车阶段")
            self._log_running_state("检测到已上车", location, direction, "切换到开车阶段")
            self.stop_auto_forward(w)
            self.reset(finding_car=False)
            w.change_stage("开车阶段")
            return

        if self.find_car_times >= len(self.PRECISE_FACE_DIRECTIONS):
            print(f"[Running] 已连续{len(self.PRECISE_FACE_DIRECTIONS)}次未成功上车，结束当前局")
            self._log_running_state("上车尝试已达上限", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.correct_position_times >= 5:
            print("[Running] 连续5次未找到车辆交互点，结束当前局")
            self._log_running_state("精调阶段长时间无进展", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.precise_entering_car:
            self._process_precise_entry(w, location, direction)
            return

        if self._handle_location_jump(location):
            if not self.loading_road or not self.road_list:
                self._load_path(location)
            if not self.road_list:
                print("[Running] 位置跳变后路径重规划失败")
                return

        self._check_if_stuck(location)
        self._check_if_trapped(location)

        if self.trapped:
            print("[Running] 人物长时间在局部区域打转，结束当前局")
            self._log_running_state("人物困死", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.stuck:
            print("[Running] 人物卡住，执行脱困")
            self._log_running_state("人物卡死", location, direction, "执行脱困")
            self._perform_unstuck_action(w, location)
            return

        if not self.loading_road or not self.road_list:
            self._load_path(location)

        if not self.road_list:
            print("[Running] 当前没有可执行路径")
            return

        target = self.road_list[0]
        dist = get_distance(location, target)
        print(f"[Running] Loc: {location}, Target: {target}, Dist: {dist:.2f}")

        if 0 <= dist <= self.WAYPOINT_TOLERANCE:
            print(f"[Running] 到达 {target} 点附近")
            self._handle_waypoint_arrival(w, location, direction, target, dist)
            return

        self._move_towards_target(w, location, direction, target)

    def _log_running_state(
        self,
        situation: str,
        location: Tuple[int, int],
        direction: Optional[float],
        decision: str,
        target: Optional[Tuple[int, int]] = None,
        dist: Optional[float] = None,
    ):
        stage = "寻车跑图" if self.finding_car else "普通跑图"
        direction_text = "None" if direction is None else f"{direction:.1f}"
        circle_text = "None" if self.stable_circle_angle is None else f"{self.stable_circle_angle:.1f}"
        target_text = str(target) if target is not None else "None"
        dist_text = "None" if dist is None else f"{dist:.2f}"
        print(
            f"[情况:{situation}] "
            f"[状态: mode={stage}, loc={location}, dir={direction_text}, circle={circle_text}, "
            f"target={target_text}, dist={dist_text}, auto_forward={self.auto_forward}, "
            f"path_len={len(self.road_list)}, precise={self.precise_entering_car}] "
            f"[决策:{decision}]"
        )

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

    def _get_scalar(self, value):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, (int, float)):
                return int(first)
        return None

    def _angle_diff(self, angle1: float, angle2: float) -> float:
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    def _update_circle_angle(self, white_angle):
        angle = self._get_scalar(white_angle)
        if angle is None:
            return

        if self.stable_circle_angle is None:
            self.stable_circle_angle = angle
            print(f"[Running] 获取到进圈方向: {angle:.1f}")
            if not self.finding_car:
                self.loading_road = False
            return

        if self._angle_diff(self.stable_circle_angle, angle) >= self.CIRCLE_REPLAN_THRESHOLD:
            print(f"[Running] 进圈方向更新: {self.stable_circle_angle:.1f} -> {angle:.1f}")
            self.stable_circle_angle = angle
            if not self.finding_car:
                self.loading_road = False

    def _is_in_vehicle(self, w: "FrameWorker") -> bool:
        return not w.get_info("左拳头") and not w.get_info("子弹")

    def _is_dead(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("变身")) or bool(w.get_info("红色血条"))

    def _handle_death(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        self.reset()
        w.change_stage("结束阶段")

    def _check_if_stuck(self, location: Tuple[int, int]):
        self.locations.append(location)
        if len(self.locations) > self.STUCK_HISTORY_LEN:
            self.locations.pop(0)

        if len(self.locations) >= self.STUCK_HISTORY_LEN:
            self.stuck = all(loc == self.locations[0] for loc in self.locations)
            if self.stuck:
                print("[Running] 检测到短时卡死")
        else:
            self.stuck = False

    def _check_if_trapped(self, location: Tuple[int, int]):
        self.history_locations.append(location)
        if len(self.history_locations) > self.TRAPPED_HISTORY_LEN:
            self.history_locations.pop(0)

        if len(self.history_locations) == self.TRAPPED_HISTORY_LEN:
            self.trapped = is_location_stagnant(self.history_locations)

    def _handle_location_jump(self, location: Tuple[int, int]) -> bool:
        if self.last_valid_location is None:
            self.last_valid_location = location
            return False

        previous_location = self.last_valid_location
        jump_dist = get_distance(location, previous_location)
        self.last_valid_location = location

        if jump_dist < self.LOCATION_JUMP_THRESHOLD:
            return False

        now = time.time()
        if now - self.last_jump_replan_time < self.LOCATION_JUMP_REPLAN_COOLDOWN:
            return False

        route_type = "寻车路径" if self.finding_car else "跑图路径"
        print(
            f"[Running] 检测到位置跳变: prev={previous_location}, "
            f"current={location}, jump_dist={jump_dist:.2f}，重新规划{route_type}"
        )
        self._log_running_state("位置跳变", location, None, f"重新规划{route_type}")
        self.last_jump_replan_time = now
        self.loading_road = False
        self.road_list = []
        self.locations = [location]
        self.history_locations = [location]
        self.stuck = False
        self.trapped = False
        return True

    def _load_path(self, location: Tuple[int, int]):
        if self.finding_car:
            print("[Running] 正在加载寻车路径...")
            self._log_running_state("正在加载寻车路径", location, None, "规划路径")
            if get_distance(location, self.R_CITY) > 50:
                approach_path = self.map_tool.plan_path(location, self.R_CITY)
                garage_path = find_path(self.R_CITY) or []
                self.road_list = self._merge_paths(approach_path, garage_path)
            else:
                self.road_list = find_path(location) or []
        else:
            if self.stable_circle_angle is None:
                print("[Running] 未获取到进圈方向，加载随机巡逻路径")
                self._log_running_state("未获取到进圈角度", location, None, "加载随机巡逻路径")
                self.road_list = self.map_tool.get_random_visible_points(location)
            else:
                print("[Running] 正在加载进圈路径...")
                self._log_running_state("正在加载进圈路径", location, None, "规划进圈路径")
                elapsed = self.get_elapsed_time()
                if elapsed <= self.STAGE1_TIME:
                    target_dist = self.STAGE1_DIS
                elif elapsed <= self.STAGE2_TIME:
                    target_dist = self.STAGE2_DIS
                else:
                    target_dist = self.STAGE3_DIS

                target_point = self.map_tool.get_target_point(location, self.stable_circle_angle, target_dist)
                self.road_list = self.map_tool.plan_path(location, target_point)

        self.road_list = [tuple(map(int, p)) for p in self.road_list if p is not None]
        self.loading_road = bool(self.road_list)
        if self.loading_road:
            print(f"[Running] 路径已加载: {self.road_list}")
            draw_points_with_arrows(self.road_list)
        else:
            print("[Running] 路径加载失败")

    def _merge_paths(self, path1, path2):
        merged = list(path1 or [])
        for point in path2 or []:
            if not merged or merged[-1] != point:
                merged.append(point)
        return merged

    def _perform_unstuck_action(self, w: "FrameWorker", current_loc: Tuple[int, int]):
        self.stop_auto_forward(w)

        if w.get_info("跳跃"):
            print("[Running] 尝试跳跃脱困")
            self._log_running_state("人物卡死", current_loc, None, "尝试跳跃脱困")
            w.click("跳跃")
            time.sleep(0.2)
            w.tap_single("摇杆", y_bias=-300, dura=400, wait=1000)
            w.refresh_frame()
            new_loc = self._get_location(w)
            if new_loc and get_distance(current_loc, new_loc) > 0.5:
                print("[Running] 跳跃脱困成功")
                self.stuck = False
                self.loading_road = False
                return

        print("[Running] 跳跃无效，执行 U 型脱困")
        self._log_running_state("跳跃脱困失败", current_loc, None, "执行U型脱困")
        for bias_x, bias_y in ((0, 300), (300, 0), (-300, 0), (0, -300)):
            w.tap_single("摇杆", x_bias=bias_x, y_bias=bias_y, dura=300, wait=1200)
            w.refresh_frame()
            new_loc = self._get_location(w)
            if new_loc and get_distance(current_loc, new_loc) > 0.5:
                print("[Running] 位移恢复，重新规划路径")
                self.stuck = False
                self.loading_road = False
                return

        self.stuck = False
        self.loading_road = False

    def _handle_waypoint_arrival(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Tuple[int, int],
        dist: float,
    ):
        if self.finding_car and len(self.road_list) <= 1:
            print("[Running] 已到达车库点，进入上车精调阶段")
            self._log_running_state("已到达车库点", location, direction, "进入上车精调阶段", target, dist)
            self._enter_precise_entry_mode(w)
            self._process_precise_entry(w, location, direction)
            return

        if self.road_list:
            self.road_list.pop(0)

        if not self.road_list:
            self.loading_road = False
            print("[Running] 当前路径已走完，准备重新规划")

    def _enter_precise_entry_mode(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        self.precise_entering_car = True
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_face_attempt_index = 0
        self.locations = []
        self.history_locations = []
        self.stuck = False
        self.trapped = False
        self.loading_road = False

    def _process_precise_entry(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        self.stop_auto_forward(w)

        if direction is None:
            print("[Running] 精调阶段当前朝向无效，等待下一帧")
            return

        dist_to_entry = get_distance(location, self.CAR_ENTRY_POINT)
        print(f"[Running] 精调上车中，当前位置 {location}，上车点 {self.CAR_ENTRY_POINT}，距离 {dist_to_entry:.2f}")
        self._log_running_state(
            "正在精调上车",
            location,
            direction,
            f"尝试对齐并接近上车点 angle={self._get_current_precise_face_direction()}",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )

        if dist_to_entry > 0:
            self._update_precise_progress(dist_to_entry)
            self._align_to_point(w, location, direction, self.CAR_ENTRY_POINT, threshold=3)
            w.tap_single("摇杆", y_bias=-120, dura=180, wait=350)
            w.refresh_frame()
            return

        target_face_direction = self._get_current_precise_face_direction()
        print(f"[Running] 当前入库尝试 {self.precise_face_attempt_index + 1}/5，目标朝向 {target_face_direction}")
        face_aligned = self._align_to_direction(w, direction, target_face_direction, threshold=3)
        if not face_aligned:
            return

        print("[Running] 已对准车头，缓慢前进尝试触发驾驶按钮")
        self._log_running_state(
            "已对准入库角度",
            location,
            direction,
            f"缓慢前进尝试上车 angle={target_face_direction}",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )
        w.tap_single(
            "摇杆",
            y_bias=self.PRECISE_FORWARD_BIAS_Y,
            dura=self.PRECISE_FORWARD_DURA,
            wait=self.PRECISE_FORWARD_WAIT,
        )
        w.refresh_frame()

        if self._click_drive_if_present(w):
            return

        for step in range(2):
            print(f"[Running] 正前未出现驾驶按钮，向右第 {step + 1}/2 次小幅试探")
            self._log_running_state(
                "正前未出现驾驶按钮",
                location,
                direction,
                f"向右小幅试探 {step + 1}/2",
                self.CAR_ENTRY_POINT,
                dist_to_entry,
            )
            w.tap_single(
                "摇杆",
                x_bias=self.PRECISE_LATERAL_STEP_BIAS,
                dura=self.PRECISE_LATERAL_STEP_DURA,
                wait=self.PRECISE_LATERAL_STEP_WAIT,
            )
            w.refresh_frame()
            if self._click_drive_if_present(w):
                return

        print("[Running] 右侧两次试探均未出现驾驶按钮，先大幅回到中心")
        w.tap_single(
            "摇杆",
            x_bias=self.PRECISE_RESET_CENTER_BIAS,
            dura=self.PRECISE_RESET_CENTER_DURA,
            wait=self.PRECISE_RESET_CENTER_WAIT,
        )
        w.refresh_frame()

        for step in range(2):
            print(f"[Running] 开始向左第 {step + 1}/2 次小幅试探")
            self._log_running_state(
                "右侧试探失败后改向左试探",
                location,
                direction,
                f"向左小幅试探 {step + 1}/2",
                self.CAR_ENTRY_POINT,
                dist_to_entry,
            )
            w.tap_single(
                "摇杆",
                x_bias=-self.PRECISE_LATERAL_STEP_BIAS,
                dura=self.PRECISE_LATERAL_STEP_DURA,
                wait=self.PRECISE_LATERAL_STEP_WAIT,
            )
            w.refresh_frame()
            if self._click_drive_if_present(w):
                return

        self._handle_precise_attempt_failure(w)

    def _update_precise_progress(self, dist_to_entry: float):
        if self.precise_last_distance is None:
            self.precise_last_distance = dist_to_entry
            self.precise_idle_rounds = 0
            return

        if dist_to_entry < self.precise_last_distance:
            self.precise_last_distance = dist_to_entry
            self.precise_idle_rounds = 0
            return

        self.precise_idle_rounds += 1
        if self.precise_idle_rounds >= 8:
            self.correct_position_times += 1
            self.precise_idle_rounds = 0
            print(f"[Running] 精调阶段长时间无进展，累计失败 {self.correct_position_times}")

    def _click_drive_if_present(self, w: "FrameWorker") -> bool:
        drive_btn = w.get_info("驾驶")
        if not drive_btn:
            return False

        print("[Running] 检测到驾驶按钮，执行上车")
        location = self._get_location(w)
        if location is not None:
            self._log_running_state("检测到驾驶按钮", location, self._get_scalar(w.get_info("direction")), "点击上车")
        w.click(drive_btn)
        time.sleep(1)
        w.refresh_frame()
        if self._is_in_vehicle(w):
            print("[Running] 上车成功")
            self.precise_entering_car = False
            self.stop_auto_forward(w)
            self.reset(finding_car=False)
            w.change_stage("开车阶段")
            return True

        print("[Running] 点击驾驶后仍未上车")
        return False

    def _get_current_precise_face_direction(self) -> int:
        index = min(self.precise_face_attempt_index, len(self.PRECISE_FACE_DIRECTIONS) - 1)
        return self.PRECISE_FACE_DIRECTIONS[index]

    def _handle_precise_attempt_failure(self, w: "FrameWorker"):
        current_direction = self._get_current_precise_face_direction()
        self.precise_face_attempt_index += 1
        self.find_car_times = self.precise_face_attempt_index

        if self.precise_face_attempt_index >= len(self.PRECISE_FACE_DIRECTIONS):
            print(f"[Running] 入库角度 {current_direction} 尝试失败，已达到 5 次上车尝试")
            return

        next_direction = self._get_current_precise_face_direction()
        print(
            f"[Running] 入库角度 {current_direction} 尝试失败，"
            f"退回原位后切换到角度 {next_direction} 再试"
        )
        self._return_to_entry_point(w)

    def _return_to_entry_point(self, w: "FrameWorker"):
        self.precise_last_distance = None
        self.precise_idle_rounds = 0

        for _ in range(6):
            location = self._get_location(w)
            direction = self._get_scalar(w.get_info("direction"))
            if location is None or direction is None:
                print("[Running] 回退车库点时位置或朝向无效")
                return

            dist_to_entry = get_distance(location, self.CAR_ENTRY_POINT)
            print(f"[Running] 回退车库点中，当前位置 {location}，距离 {dist_to_entry:.2f}")
            if dist_to_entry <= 0:
                return

            aligned = self._align_to_point(w, location, direction, self.CAR_ENTRY_POINT, threshold=3)
            if not aligned:
                continue

            w.tap_single("摇杆", y_bias=-120, dura=180, wait=500)
            w.refresh_frame()

    def _align_to_point(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: float,
        target: Tuple[int, int],
        threshold: int = 5,
    ) -> bool:
        target_dir = calculate_angle(location, target)
        turn_dir, pixel, diff = calculate_move_count(direction, target_dir)

        if turn_dir is None:
            return False

        if diff <= threshold:
            return True

        x_bias = pixel if turn_dir == "right" else -pixel
        dura_time = max(250, int(pixel * 1.5))
        print(f"[Running] 调整方向: current={direction}, target={target_dir}, pixel={pixel}")
        w.tap_single("视角", x_bias=int(x_bias), dura=dura_time, wait=250)
        w.refresh_frame()
        return False

    def _align_to_direction(
        self,
        w: "FrameWorker",
        direction: float,
        target_direction: float,
        threshold: int = 5,
    ) -> bool:
        turn_dir, pixel, diff = calculate_move_count(direction, target_direction)

        if turn_dir is None:
            return False

        if diff <= threshold:
            return True

        x_bias = pixel if turn_dir == "right" else -pixel
        dura_time = max(250, int(pixel * 1.5))
        print(f"[Running] 调整朝向: current={direction}, target={target_direction}, pixel={pixel}")
        w.tap_single("视角", x_bias=int(x_bias), dura=dura_time, wait=250)
        w.refresh_frame()
        return False

    def _move_towards_target(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Tuple[int, int],
    ):
        if not self.auto_forward:
            w.click("自动前进")
            self.auto_forward = True

        if direction is None:
            print("[Running] 当前朝向无效，等待下一帧")
            self._log_running_state("当前朝向无效", location, None, "等待下一帧", target)
            return

        target_dir = calculate_angle(location, target)
        turn_dir, pixel, diff = calculate_move_count(direction, target_dir)
        if abs(diff) <= 5:
            self._log_running_state("前方路径正常", location, direction, "保持自动前进", target, get_distance(location, target))
            return

        x_bias = pixel if turn_dir == "right" else -pixel
        dura_time = max(400, int(pixel * 1.5))
        print(
            f'[Correct Dire] current_dire : {direction}, target_dire : {target_dir}, 向 {target_dir} 调整 {pixel} 像素, 用时 {dura_time} ms')
        self._log_running_state(
            "跑图方向偏移",
            location,
            direction,
            f"{turn_dir}调整视角 {pixel}px/{dura_time}ms",
            target,
            get_distance(location, target),
        )
        w.tap_single("视角", x_bias=int(x_bias), dura=dura_time, wait=300)


    def stop_auto_forward(self, w: "FrameWorker"):
        if self.auto_forward:
            w.click("自动前进")
            self.auto_forward = False
