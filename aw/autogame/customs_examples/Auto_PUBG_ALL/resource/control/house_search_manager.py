import random
import time
from typing import TYPE_CHECKING
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager import HouseExitManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.timing import TimeoutTracker
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print
from aw.autogame.tools.Utils import *

if TYPE_CHECKING:
    # 假设你的框架类定义在 framework.py 文件中
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class HouseSearchManager:
    VISUAL_APPROACH_MAX_ATTEMPTS = 12
    UNSTUCK_MAX_CYCLES = 6
    UNSTUCK_FORWARD_STEPS = 5
    PICKUP_MAX_PER_DIRECTION = 3
    INITIAL_LOCATION_MIN_SAMPLES = 3
    INITIAL_LOCATION_MAX_SAMPLES = 6
    INITIAL_LOCATION_STABLE_DISTANCE = 2.5
    INITIAL_LOCATION_JUMP_RESET_DISTANCE = 8.0
    ENTRY_AUTO_FORWARD_DISTANCE = 30.0
    ENTRY_COARSE_MOVE_DISTANCE = 10.0
    ENTRY_ARRIVAL_DISTANCE = 1.0
    ENTRY_COARSE_Y_BIAS = -430
    ENTRY_COARSE_DURA = 1300
    ENTRY_FINE_Y_BIAS = -220
    ENTRY_FINE_DURA = 480
    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2
    HOUSE_NEAR_DOOR = 3
    HOUSE_NEAR_WALL = 4
    HOUSE_CLASS_IDS = {8}
    WINDOW_CLASS_IDS = {2}
    STONE_WALL_CLASS_IDS = {9}
    HOUSE_ENTRY_CLASS_IDS = {0, 2, 4}
    DOOR_CLASS_IDS = {0, 4}
    HOUSE_BLOCK_CENTER_OVERLAP = 0.12
    HOUSE_BLOCK_LOWER_OVERLAP = 0.18
    HOUSE_BLOCK_AREA_RATIO = 0.015
    HOUSE_BYPASS_SIDE_STEPS = 4
    HOUSE_BYPASS_FORWARD_STEPS = 3
    HOUSE_PROACTIVE_BYPASS_MIN_DISTANCE = 18.0
    HOUSE_PROACTIVE_BYPASS_SIDE_STEPS = 3
    HOUSE_PROACTIVE_BYPASS_FORWARD_STEPS = 2
    HOUSE_PROACTIVE_BYPASS_SIDE_BIAS = 300
    HOUSE_PROACTIVE_BYPASS_SIDE_DURA = 380
    HOUSE_PROACTIVE_BYPASS_SIDE_WAIT = 650
    HOUSE_PROACTIVE_BYPASS_FORWARD_Y_BIAS = -260
    HOUSE_PROACTIVE_BYPASS_FORWARD_DURA = 320
    HOUSE_PROACTIVE_BYPASS_FORWARD_WAIT = 700
    HOUSE_PROACTIVE_BYPASS_NEAR_ENTRY_SCENES = {3, 4}
    HOUSE_SEARCH_BYPASS_MIN_ENTRY_DISTANCE = 10.0
    HOUSE_OBSTACLE_TURN_STEP_DEGREES = 30
    HOUSE_OBSTACLE_MAX_TURN_DEGREES = 90
    HOUSE_OBSTACLE_FORWARD_Y_BIAS = -300
    HOUSE_OBSTACLE_FORWARD_DURA = 500
    HOUSE_OBSTACLE_FORWARD_WAIT = 3000
    HOUSE_BYPASS_UNSTUCK_PAUSE_SECONDS = 5.0
    STONE_WALL_FORWARD_Y_BIAS = -200
    STONE_WALL_FORWARD_DURA = 200
    STONE_WALL_FORWARD_WAIT = 500
    STONE_WALL_JUMP_FORWARD_Y_BIAS = -300
    STONE_WALL_JUMP_FORWARD_DURA = 300
    STONE_WALL_JUMP_FORWARD_WAIT = 900
    STONE_WALL_JUMP_SETTLE_SECONDS = 0.15
    VISIBLE_DOOR_CENTER_MAX_STEPS = 6
    VISIBLE_DOOR_CENTER_SIDE_BIAS = 240
    VISIBLE_DOOR_CENTER_SIDE_DURA = 260
    VISIBLE_DOOR_CENTER_SIDE_WAIT = 420
    VISIBLE_DOOR_FORWARD_Y_BIAS = -320
    VISIBLE_DOOR_FORWARD_DURA = 420
    VISIBLE_DOOR_FORWARD_WAIT = 800
    ACCIDENTAL_HOUSE_MATCH_MAX_DISTANCE = 22.0
    ROUTE_STUCK_TURN_DEGREES = 90
    ROUTE_STUCK_REPEAT_RADIUS = 4.0
    ROUTE_STUCK_MAX_TURN_DEGREES = 150
    ROUTE_STUCK_TURN_ESCALATE_STEP = 30
    ROUTE_STUCK_BYPASS_FORWARD_Y_BIAS = -300
    ROUTE_STUCK_BYPASS_FORWARD_DURA = 300
    ROUTE_STUCK_BYPASS_FORWARD_DURA_STEP = 160
    ROUTE_STUCK_BYPASS_FORWARD_MAX_DURA = 900
    ROUTE_STUCK_BYPASS_FORWARD_BASE_WAIT = 700
    ROUTE_STUCK_BYPASS_FORWARD_STEP_WAIT = 450
    ROUTE_STUCK_BYPASS_FORWARD_MAX_WAIT = 2600
    ROUTE_STUCK_BACKOFF_Y_BIAS = 300
    ROUTE_STUCK_BACKOFF_BASE_DURA = 450
    ROUTE_STUCK_BACKOFF_DURA_STEP = 180
    ROUTE_STUCK_BACKOFF_MAX_DURA = 900
    ROUTE_STUCK_BACKOFF_BASE_WAIT = 850
    ROUTE_STUCK_BACKOFF_WAIT_STEP = 300
    ROUTE_STUCK_BACKOFF_MAX_WAIT = 1800
    HOUSE_SEARCH_TIMEOUT_SECONDS = 60
    ENTRY_NEAR_MICRO_ADJUST_DISTANCE = 1.5
    ENTRY_NEAR_MICRO_DONE_DISTANCE = 0.25
    ENTRY_NEAR_MICRO_MAX_ATTEMPTS = 3
    ENTRY_NEAR_MICRO_X_BIAS = 120
    ENTRY_NEAR_MICRO_Y_BIAS = 120
    ENTRY_NEAR_MICRO_DURA = 160
    ENTRY_NEAR_MICRO_WAIT = 320
    ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS = 4
    ENTRY_DOOR_FINAL_LATERAL_X_BIAS = 450
    ENTRY_DOOR_FINAL_LATERAL_DURA = 180
    ENTRY_DOOR_FINAL_LATERAL_WAIT = 360
    ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX = 55
    ENTRY_DOOR_DIRECT_CENTER_MIN_RATIO = 0.40
    ENTRY_DOOR_DIRECT_CENTER_MAX_RATIO = 0.60
    ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS = -200
    ENTRY_DOOR_DIRECT_FORWARD_DURA = 200
    ENTRY_DOOR_DIRECT_FORWARD_WAIT = 3000
    ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS = 200
    ENTRY_DOOR_DIRECT_BACKOFF_DURA = 200
    ENTRY_DOOR_DIRECT_BACKOFF_WAIT = 3000
    ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE = 2
    ENTRY_DOOR_DIRECT_MAX_FAILURES = 3
    ENTRY_DOOR_DIRECT_REALIGN_MAX_ATTEMPTS = 3
    ENTRY_NEAR_LATERAL_CORRECT_MIN_RELATIVE_DEGREES = 5
    ENTRY_NEAR_LATERAL_CORRECT_MAX_RELATIVE_DEGREES = 175
    ENTRY_NEAR_LATERAL_CORRECT_X_BIAS = 120
    ENTRY_NEAR_LATERAL_CORRECT_DURA = 160
    ENTRY_NEAR_LATERAL_CORRECT_WAIT = 320
    ENTRY_NEAR_ALIGN_TOLERANCE = 5
    ENTRY_NEAR_ALIGN_MAX_STEPS = 2
    ENTRY_NEAR_ALIGN_MIN_DURA = 300
    ENTRY_NEAR_ALIGN_MAX_DURA = 300
    ENTRY_NEAR_ALIGN_MAX_BIAS = 120
    ENTRY_NEAR_ALIGN_WAIT = 100
    ENTRY_WALL_BACKOFF_DURA = 520
    ENTRY_WALL_BACKOFF_WAIT = 900
    EXCLUDED_ENTRY_LOCATIONS = {
        (1006, 706),
        (991, 709),
        (1010, 705),
    }
    ENTRY_CONFIRM_MAX_ATTEMPTS = 8
    ENTRY_CONFIRM_FORWARD_Y_BIAS = -420
    ENTRY_CONFIRM_FORWARD_DURA = 650
    ENTRY_CONFIRM_FORWARD_WAIT = 850
    ENTRY_CONFIRM_SIDE_X_BIAS = 260
    ALIGN_MAX_BIAS = 460
    ALIGN_MIN_DURA = 180
    ALIGN_MAX_DURA = 650
    ALIGN_WAIT = 220

    def __init__(self):
        self.map_tool = MapNavigator()
        self.house_data = load_json(
            r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/house_entry/house_entries_summary.json')
        self.excluded_house_ids = self._build_excluded_house_ids()

        self.completed_houses = set()
        self.current_house_id = None
        self.active_entry = None

        # 状态机: IDLE -> FAST_NAV -> PRECISE_NAV -> SCANNING -> VISUAL_APPROACH -> INTERACT -> FINAL_ENTRY
        self.status = "IDLE"

        # 辅助变量
        self.first_view = False
        self.auto_forward = False
        self.screen_w, self.screen_h = get_resolution()

        # 用于智能选点的临时黑名单 (本轮循环跳过，不永久删除)
        self.temp_skip_houses = set()

        # --- 卡顿检测相关变量 ---
        self.history_locations = []
        self.max_history_len = 5  # 记录最近10次位置
        self.stuck_threshold = 0.5  # 判定卡住的距离阈值

        self.searching_number = 0

        # 用于搜索房屋使用到的辅助变量
        self.supplies = []  # [(绝对角度, 框高)]
        self.doors = []  # [(绝对角度, 框高)]
        self.player_yaw = 0.0  # 累计旋转角度（0° = 进入房间时的朝向）
        self.last_target_bbox = None

        self.rooms_searched = 0

        self.entrance_doors = []  # 入口房间门列表 [(rel_angle, box_h), ...]
        self.a_door_sign = None  # 入口A门特征 (rel_angle, box_h)
        self.sub_rooms_info = []  # 已进入的子房间信息
        self.visited_doors = set()
        self.sub_rooms = []
        self.rooms_done = 0

        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.visited_abs = []
        self.visited_doors_info = []
        self.sub_room_area = None
        self.visited_sub_doors = []
        self.sub_rooms_entered = 0

        self.house_exit_manager = HouseExitManager()
        self.indoor_stuck_frames = 0
        self.house_search_timer = TimeoutTracker(
            self.HOUSE_SEARCH_TIMEOUT_SECONDS,
            monotonic=True,
        )
        self.abort_callback = None
        self.can_finish_callback = None
        self.finish_callback = None
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self.initial_target_pending = True
        self.location_missing_frames = 0
        self.initial_location_samples = []
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self.entry_near_micro_adjust_attempts = 0

    def _entry_location_tuple(self, entry):
        try:
            location = entry.get('location')
            return (int(location[0]), int(location[1]))
        except (TypeError, ValueError, IndexError, AttributeError):
            return None

    def _is_excluded_entry(self, entry) -> bool:
        return self._entry_location_tuple(entry) in self.EXCLUDED_ENTRY_LOCATIONS

    def _build_excluded_house_ids(self):
        excluded = set()
        for house_id, entries in self.house_data.items():
            if any(self._is_excluded_entry(entry) for entry in entries):
                excluded.add(house_id)
        if excluded:
            print(
                f"[Searching] 已过滤指定进门点 {sorted(self.EXCLUDED_ENTRY_LOCATIONS)} "
                f"对应房屋 {sorted(excluded)}"
            )
        return excluded

    def _is_excluded_house(self, house_id) -> bool:
        return house_id in self.excluded_house_ids

    def reset(self):
        self.completed_houses = set()
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.first_view = False
        self.auto_forward = False
        self.temp_skip_houses = set()
        self.history_locations = []
        self.searching_number = 0
        self.supplies = []
        self.doors = []
        self.player_yaw = 0.0
        self.last_target_bbox = None
        self.rooms_searched = 0
        self.entrance_doors = []
        self.a_door_sign = None
        self.sub_rooms_info = []
        self.visited_doors = set()
        self.sub_rooms = []
        self.rooms_done = 0
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.visited_abs = []
        self.visited_doors_info = []
        self.sub_room_area = None
        self.visited_sub_doors = []
        self.sub_rooms_entered = 0

        self.house_exit_manager.reset()
        self.indoor_stuck_frames = 0
        self.house_search_timer.reset()
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self.initial_target_pending = True
        self.location_missing_frames = 0
        self.initial_location_samples = []
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self.entry_near_micro_adjust_attempts = 0

    def process(self, w: 'FrameWorker'):
        if self._should_abort(w):
            return

        # 0. 基础设置：落地后首帧刷新画面 + 切第一人称
        if not self.first_view:
            w.refresh_frame()
            w.refresh_frame()
            w.click('人称')
            self.first_view = True

        location_raw = w.get_info('location')
        if location_raw is None:
            self.location_missing_frames += 1
            print('位置值是None，等待位置刷新...')
            w.refresh_frame()
            if self.location_missing_frames >= 3:
                print('位置连续缺失，轻微移动以刷新位置...')
                w.tap_single('摇杆', y_bias=-120, wait=300)
            return
        location = check_location(location_raw[0])
        direction = w.get_info('direction')

        if location is None:
            self.location_missing_frames += 1
            print('位置值无效，等待位置刷新...')
            w.refresh_frame()
            if self.location_missing_frames >= 3:
                print('位置连续无效，轻微移动以刷新位置...')
                w.tap_single('摇杆', y_bias=-120, wait=300)
            return

        self.location_missing_frames = 0
        self.searching_logic(w, location, direction)

    def _should_abort(self, w: 'FrameWorker'):
        callback = getattr(self, "abort_callback", None)
        if callback is None:
            return False
        try:
            return bool(callback(w))
        except Exception as exc:
            print(f"[Searching] 中断检查失败: {exc}")
            return False

    def _can_finish_searching(self, w: 'FrameWorker'):
        callback = getattr(self, "can_finish_callback", None)
        if callback is None:
            return True
        try:
            return bool(callback(w))
        except Exception as exc:
            print(f"[Searching] 结束条件检查失败: {exc}")
            return False

    def _finish_searching_phase(self, w: 'FrameWorker', reason: str):
        callback = getattr(self, "finish_callback", None)
        if callback is not None:
            try:
                return bool(callback(w, reason))
            except Exception as exc:
                print(f"[Searching] 搜房结束回调失败: {exc}")

        w.change_stage('跑图阶段')
        return True

    def _continue_searching_until_timer(self, w: 'FrameWorker', reason: str):
        self.stop_auto_forward(w)
        if self._can_finish_searching(w):
            print(f"[Searching] {reason}，搜房计时已满，切换到跑图阶段")
            self.searching_number = 0
            self.current_house_id = None
            self.active_entry = None
            self.status = "IDLE"
            self.avoid_angle_ref = None
            self.avoid_mode = None
            self.initial_target_pending = True
            self.location_missing_frames = 0
            self.initial_location_samples = []
            self._reset_entry_near_micro_adjust()
            return self._finish_searching_phase(w, reason)

        print(f"[Searching] {reason}，但搜房未满10分钟，重置本轮目标继续搜房")
        self.searching_number = 0
        self.completed_houses.clear()
        self.temp_skip_houses.clear()
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.history_locations = []
        self.indoor_stuck_frames = 0
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self.initial_target_pending = True
        self.location_missing_frames = 0
        self.initial_location_samples = []
        self._reset_entry_near_micro_adjust()
        return False

    def _get_forward_scene(self, w: 'FrameWorker'):
        scene = w.get_info('forward_scene')
        if isinstance(scene, (list, tuple)):
            return scene
        return []

    def _get_house_scene(self, w: 'FrameWorker'):
        value = w.get_info('house_scene')
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_location_value(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return check_location(value)

    def _reset_route_stuck_bypass(self):
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self._reset_entry_near_micro_adjust()

    def _reset_entry_near_micro_adjust(self):
        self.entry_near_micro_adjust_attempts = 0

    def _pause_unstuck_for_house_bypass(self, phase_label='NAV'):
        self.house_bypass_unstuck_pause_until = (
            time.monotonic() + self.HOUSE_BYPASS_UNSTUCK_PAUSE_SECONDS
        )
        self.history_locations = []
        print(f"[NavBypass] {phase_label} 绕房调整视角/前推期间暂停通用避障")

    def _is_house_bypass_unstuck_paused(self) -> bool:
        pause_until = getattr(self, "house_bypass_unstuck_pause_until", 0.0)
        if pause_until and time.monotonic() < pause_until:
            return True
        if pause_until:
            self.house_bypass_unstuck_pause_until = 0.0
        return False

    def _resolve_house_by_location(self, current_loc, max_distance=None):
        loc = self._normalize_location_value(current_loc)
        if loc is None:
            return None

        limit = self.ACCIDENTAL_HOUSE_MATCH_MAX_DISTANCE if max_distance is None else float(max_distance)
        best = None
        for house_id, entries in self.house_data.items():
            for entry in entries:
                entry_loc = self._entry_location_tuple(entry)
                if entry_loc is None:
                    continue
                dist = get_distance(loc, entry_loc)
                if best is None or dist < best[2]:
                    best = (house_id, entry, dist)

        if best and best[2] <= limit:
            return best
        return None

    def _confirm_indoor_before_search(self, w: 'FrameWorker', reason: str) -> bool:
        return True

    def _complete_current_house_search(self, w: 'FrameWorker', reason: str) -> bool:
        if self._should_abort(w):
            return False

        self.stop_auto_forward(w)
        self.indoor_stuck_frames = 0
        print(f"[Searching] {reason}")

        if not self.start_searching(w):
            return False
        if w.current_stage != '搜房阶段':
            return False

        if self.current_house_id is not None:
            self.completed_houses.add(self.current_house_id)
        self.searching_number += 1
        print(f"[Searching] 房屋 {self.current_house_id} 完成，累计已搜 {self.searching_number} 个")

        w.refresh_frame()
        exit_direction = w.get_info('direction')
        self.prepare_next_target_logic(exit_direction)
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.history_locations = []
        self._reset_route_stuck_bypass()
        return True

    def _exit_current_indoor_house(self, w: 'FrameWorker', reason: str) -> bool:
        self.stop_auto_forward(w)
        self._clear_house_search_timer()
        print(f"[Searching] {reason}，执行快速出房")

        result = self._exit_house(w)
        if result is None:
            w.refresh_frame()
            result = self._get_house_scene(w) != 0

        self.indoor_stuck_frames = 0
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.history_locations = []
        self._reset_route_stuck_bypass()

        if result:
            print("[Searching] 快速出房完成，继续寻找下一个进门点")
            return True
        print("[Searching] 快速出房暂未确认成功，下一轮继续兜底")
        return False

    def _handle_indoor_during_entry_route(self, w: 'FrameWorker', current_loc, reason: str) -> bool:
        if self._get_house_scene(w) != 0:
            return False

        current_stage = getattr(w, "current_stage", None)
        if current_stage and current_stage != '搜房阶段':
            return False

        self.stop_auto_forward(w)
        matched = self._resolve_house_by_location(current_loc)
        matched_house_id = None
        matched_entry = None

        if matched:
            matched_house_id, matched_entry, matched_dist = matched
            print(
                f"[Searching] {reason}，当前位置匹配到房屋 {matched_house_id}，"
                f"nearest_entry_dist={matched_dist:.2f}"
            )
            if matched_house_id in self.completed_houses or self._is_excluded_house(matched_house_id):
                return self._exit_current_indoor_house(
                    w,
                    f"误入房屋 {matched_house_id}，该房屋已搜过或被排除",
                )

            self.current_house_id = matched_house_id
            self.active_entry = matched_entry
        else:
            print(f"[Searching] {reason}，当前位置未匹配到房屋列表，搜完后不写入完成房屋")
            self.current_house_id = None
            self.active_entry = None

        if not self._confirm_indoor_before_search(w, reason):
            return True

        return self._complete_current_house_search(w, reason)

    def _start_house_search_timer(self):
        self.house_search_timer.start()

    def _clear_house_search_timer(self):
        self.house_search_timer.reset()

    def _house_search_timed_out(self):
        if self.house_search_timer.should_report_expired():
            print(f"[搜房] 入屋搜房已超过{self.HOUSE_SEARCH_TIMEOUT_SECONDS}s，停止搜房并执行出房策略")
        return self.house_search_timer.expired()

    def _should_stop_house_search(self, w: 'FrameWorker'):
        return self._should_abort(w) or self._house_search_timed_out()

    def _force_exit_after_search_timeout(self, w: 'FrameWorker'):
        self.stop_auto_forward(w)
        self._clear_house_search_timer()
        w.refresh_frame()

        if self._get_house_scene(w) != 0:
            print("[搜房] 超时时已不在屋内，视为出房完成")
            return True

        print("[搜房] 超时兜底：启动 HouseExitManager 直接出房")
        self.house_exit_manager.reset()
        for _ in range(30):
            if self._should_abort(w):
                return False
            if self.house_exit_manager.process(w):
                print("[搜房] 超时兜底出房成功")
                return True

        print("[搜房] HouseExitManager 未出房，回退到原出房策略")
        self._exit_house(w)
        return not self._should_abort(w) and self._get_house_scene(w) != 0

    def _get_frame_size(self):
        inf_w, inf_h = get_wh()
        return max(inf_w, inf_h), min(inf_w, inf_h)

    def _is_house_detection(self, det):
        try:
            return len(det) >= 6 and int(det[5]) in self.HOUSE_CLASS_IDS
        except (TypeError, ValueError):
            return False

    def _is_house_entry_detection(self, det):
        try:
            return len(det) >= 6 and int(det[5]) in self.HOUSE_ENTRY_CLASS_IDS
        except (TypeError, ValueError):
            return False

    def _detection_class_id(self, det):
        try:
            if len(det) < 6:
                return None
            return int(det[5])
        except (TypeError, ValueError):
            return None

    def _is_detection_class(self, det, class_ids):
        cls_id = self._detection_class_id(det)
        return cls_id in class_ids if cls_id is not None else False

    def _is_house_like_detection(self, det):
        return self._is_house_detection(det) or self._is_house_entry_detection(det)

    def _detection_area(self, det):
        try:
            x1, y1, x2, y2 = [float(v) for v in det[:4]]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _front_house_blocking(self, w: 'FrameWorker'):
        scene = self._get_forward_scene(w)
        if not scene:
            return None

        entry_candidates = [det for det in scene if self._is_house_entry_detection(det)]
        if entry_candidates:
            return max(entry_candidates, key=self._detection_area)

        frame_w, frame_h = self._get_frame_size()
        center_l = frame_w * 0.38
        center_r = frame_w * 0.62
        lower_t = frame_h * 0.35
        center_band_w = max(center_r - center_l, 1)
        lower_band_h = max(frame_h - lower_t, 1)
        candidates = []

        for det in scene:
            if not self._is_house_detection(det):
                continue
            x1, y1, x2, y2 = [float(v) for v in det[:4]]
            x1, x2 = max(0, min(x1, frame_w)), max(0, min(x2, frame_w))
            y1, y2 = max(0, min(y1, frame_h)), max(0, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue

            center_overlap = max(0, min(x2, center_r) - max(x1, center_l)) / center_band_w
            lower_overlap = max(0, min(y2, frame_h) - max(y1, lower_t)) / lower_band_h
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
            if (center_overlap >= self.HOUSE_BLOCK_CENTER_OVERLAP
                    and lower_overlap >= self.HOUSE_BLOCK_LOWER_OVERLAP
                    and area_ratio >= self.HOUSE_BLOCK_AREA_RATIO):
                center_x = (x1 + x2) / 2
                center_score = 1 - min(abs(center_x - frame_w / 2) / max(frame_w / 2, 1), 1)
                score = center_overlap * 2 + lower_overlap + area_ratio * 4 + center_score
                candidates.append((score, det))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _front_path_detection(self, scene, class_ids):
        if not scene:
            return None

        frame_w, frame_h = self._get_frame_size()
        center_l = frame_w * 0.35
        center_r = frame_w * 0.65
        lower_t = frame_h * 0.30
        center_band_w = max(center_r - center_l, 1)
        lower_band_h = max(frame_h - lower_t, 1)
        candidates = []

        for det in scene:
            if not self._is_detection_class(det, class_ids):
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in det[:4]]
            except (TypeError, ValueError):
                continue

            x1, x2 = max(0, min(x1, frame_w)), max(0, min(x2, frame_w))
            y1, y2 = max(0, min(y1, frame_h)), max(0, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue

            center_overlap = max(0, min(x2, center_r) - max(x1, center_l)) / center_band_w
            lower_overlap = max(0, min(y2, frame_h) - max(y1, lower_t)) / lower_band_h
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
            if center_overlap <= 0 and lower_overlap <= 0:
                continue
            score = center_overlap * 2 + lower_overlap + area_ratio * 4
            candidates.append((score, det))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _front_route_obstacle_summary(self, w: 'FrameWorker'):
        scene = self._get_forward_scene(w)
        summary = {
            "has_house": False,
            "has_window": False,
            "has_door": False,
            "stone_wall": None,
        }
        if not scene:
            return summary

        for det in scene:
            cls_id = self._detection_class_id(det)
            if cls_id is None:
                continue
            if cls_id in self.HOUSE_CLASS_IDS:
                summary["has_house"] = True
            if cls_id in self.WINDOW_CLASS_IDS:
                summary["has_window"] = True
            if cls_id in self.DOOR_CLASS_IDS:
                summary["has_door"] = True

        summary["stone_wall"] = self._front_path_detection(scene, self.STONE_WALL_CLASS_IDS)
        return summary

    def _house_side_block_score(self, scene, lane_left, lane_right, frame_h):
        lane_w = max(lane_right - lane_left, 1)
        lower_top = frame_h * 0.35
        lane_area = lane_w * max(frame_h - lower_top, 1)
        score = 0.0
        for det in scene:
            if not self._is_house_like_detection(det):
                continue
            x1, y1, x2, y2 = [float(v) for v in det[:4]]
            overlap_w = max(0, min(x2, lane_right) - max(x1, lane_left))
            overlap_h = max(0, min(y2, frame_h) - max(y1, lower_top))
            score += (overlap_w * overlap_h) / lane_area
        return score

    def _choose_house_bypass_side(self, w: 'FrameWorker'):
        scene = self._get_forward_scene(w)
        frame_w, frame_h = self._get_frame_size()
        left_score = self._house_side_block_score(scene, frame_w * 0.16, frame_w * 0.46, frame_h)
        right_score = self._house_side_block_score(scene, frame_w * 0.54, frame_w * 0.84, frame_h)
        side = "right" if right_score <= left_score else "left"
        print(f"[Unstuck] 房体绕行空隙判断：left={left_score:.2f}, right={right_score:.2f}，选择{side}")
        return side

    def _bypass_front_house_block(self, w: 'FrameWorker', current_loc, safe_get_loc):
        print("[Unstuck] 室外卡住，先后退确认前方是否为房体阻挡")
        w.tap_single('摇杆', y_bias=300, dura=450, wait=900)
        w.refresh_frame()

        if not self._front_house_blocking(w):
            print("[Unstuck] 后退后前方未确认房体，交给通用避障")
            return False

        if self._try_lock_visible_door_after_block(w):
            return True

        first_side = self._choose_house_bypass_side(w)
        sides = [first_side, "left" if first_side == "right" else "right"]
        back_loc = safe_get_loc() or current_loc

        for side in sides:
            if self._should_abort(w):
                return False
            bias = 300 if side == "right" else -300
            print(f"[Unstuck] 前方房体挡路，尝试向{side}侧滑绕房")
            side_base_loc = safe_get_loc() or back_loc

            for _ in range(self.HOUSE_BYPASS_SIDE_STEPS):
                if self._should_abort(w):
                    return False
                w.tap_single('摇杆', x_bias=bias, dura=450, wait=700)
                w.refresh_frame()
                if not self._front_house_blocking(w):
                    break

            side_loc = safe_get_loc()
            if not side_loc or not side_base_loc or get_distance(side_base_loc, side_loc) <= 0.5:
                print(f"[Unstuck] {side}侧滑位移不足，尝试另一侧")
                continue

            for _ in range(self.HOUSE_BYPASS_FORWARD_STEPS):
                if self._should_abort(w):
                    return False
                w.tap_single('摇杆', y_bias=-300, dura=400, wait=900)
                w.refresh_frame()
                forward_loc = safe_get_loc()
                if forward_loc and get_distance(current_loc, forward_loc) > self.stuck_threshold:
                    print("[Unstuck] 绕房通过成功")
                    return True
                if self._front_house_blocking(w):
                    w.tap_single('摇杆', x_bias=bias, dura=350, wait=500)
                    w.refresh_frame()

            print(f"[Unstuck] {side}侧仍未绕开，尝试另一侧")

        print("[Unstuck] 房体绕行未成功，回退到通用避障")
        return False

    def _is_searching_stage_frame(self, w: 'FrameWorker') -> bool:
        current_stage = getattr(w, "current_stage", None)
        return current_stage is None or current_stage == '搜房阶段'

    def _safe_get_frame_location(self, w: 'FrameWorker'):
        raw = w.get_info('location')
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            return check_location(raw[0])
        return check_location(raw)

    def _is_route_close_to_current_entry(self, target_loc, dist_val):
        if not self.current_house_id or not self.active_entry:
            return False
        entry_loc = self._entry_location_tuple(self.active_entry)
        target = check_location(target_loc)
        if entry_loc is None or target is None:
            return False
        if get_distance(entry_loc, target) > 2.0:
            return False
        return dist_val <= self.ENTRY_AUTO_FORWARD_DISTANCE

    def _rotate_view_until_house_clear(self, w: 'FrameWorker', side: str, phase_label: str):
        turned = 0
        direction = w.get_info('direction')
        while turned < self.HOUSE_OBSTACLE_MAX_TURN_DEGREES:
            if self._should_abort(w):
                return True
            if not self._front_house_blocking(w):
                return True

            step = min(
                self.HOUSE_OBSTACLE_TURN_STEP_DEGREES,
                self.HOUSE_OBSTACLE_MAX_TURN_DEGREES - turned,
            )
            if direction is None:
                x_bias = 300 if side == "right" else -300
                print(f"[NavBypass] {phase_label} 缺少方向角，直接向{side}拨视角")
                w.tap_single('视角', x_bias=x_bias, dura=520, wait=500)
            else:
                target_direction = (float(direction) + (step if side == "right" else -step)) % 360
                if target_direction == 0:
                    target_direction = 360
                print(
                    f"[NavBypass] {phase_label} 前方仍有房体，向{side}转{step}度避让"
                )
                self.align_direction_blocking(
                    w,
                    direction,
                    target_direction,
                    threshold=8,
                    max_steps=2,
                    wait=260,
                )

            turned += step
            w.refresh_frame()
            direction = w.get_info('direction')
        return True

    def _bypass_front_house_by_view_turn(self, w: 'FrameWorker', target_loc, phase_label='NAV'):
        self._pause_unstuck_for_house_bypass(phase_label)
        self.stop_auto_forward(w)
        side = self._choose_house_bypass_side(w)
        print(
            f"[NavBypass] {phase_label} 前方检测到房体/门窗，"
            f"固定向{side}侧转向绕行，最大{self.HOUSE_OBSTACLE_MAX_TURN_DEGREES}度"
        )

        self._rotate_view_until_house_clear(w, side, phase_label)
        if self._should_abort(w):
            return True

        print(f"[NavBypass] {phase_label} 绕行视角已处理，前推3秒后继续导航")
        w.tap_single(
            '摇杆',
            y_bias=self.HOUSE_OBSTACLE_FORWARD_Y_BIAS,
            dura=self.HOUSE_OBSTACLE_FORWARD_DURA,
            wait=self.HOUSE_OBSTACLE_FORWARD_WAIT,
        )
        w.refresh_frame()
        if hasattr(self, "history_locations"):
            self.history_locations = []
        self._pause_unstuck_for_house_bypass(phase_label)
        return True

    def _center_visible_door_by_lateral_move(self, w: 'FrameWorker', door):
        for _ in range(self.VISIBLE_DOOR_CENTER_MAX_STEPS):
            inf_w, inf_h = get_wh()
            frame_w = max(inf_w, inf_h)
            left_bound = frame_w / 3
            right_bound = frame_w * 2 / 3
            door_center_x = (door[0] + door[2]) / 2
            if left_bound <= door_center_x <= right_bound:
                return door

            x_bias = (
                -self.VISIBLE_DOOR_CENTER_SIDE_BIAS
                if door_center_x < left_bound
                else self.VISIBLE_DOOR_CENTER_SIDE_BIAS
            )
            print("[NavBypass] 非目标房门不在中间1/3，横向调整人物位置")
            w.tap_single(
                '摇杆',
                x_bias=x_bias,
                dura=self.VISIBLE_DOOR_CENTER_SIDE_DURA,
                wait=self.VISIBLE_DOOR_CENTER_SIDE_WAIT,
            )
            w.refresh_frame()
            door = self.find_largest_door(w)
            if door is None:
                return None
        return door

    def _try_enter_visible_non_target_house(self, w: 'FrameWorker', current_loc, phase_label='NAV'):
        door = self.find_largest_door(w)
        if door is None:
            return False

        print(f"[NavBypass] {phase_label} 前方不是当前目标但看到门，尝试顺路进房")
        self.stop_auto_forward(w)
        door = self._center_visible_door_by_lateral_move(w, door)
        if door is None:
            print("[NavBypass] 横向调整后门目标丢失，改走绕房策略")
            return False

        if not self._align_to_door_detection(w, door):
            print("[NavBypass] 门视角对齐失败，改走绕房策略")
            return False

        for _ in range(3):
            if self._should_abort(w):
                return True
            if w.get_info('开门'):
                w.click('开门')
                time.sleep(1)
            w.tap_single(
                '摇杆',
                y_bias=self.VISIBLE_DOOR_FORWARD_Y_BIAS,
                dura=self.VISIBLE_DOOR_FORWARD_DURA,
                wait=self.VISIBLE_DOOR_FORWARD_WAIT,
            )
            w.refresh_frame()
            if self._get_house_scene(w) == 0:
                indoor_loc = self._safe_get_frame_location(w) or current_loc
                return self._handle_indoor_during_entry_route(
                    w,
                    indoor_loc,
                    "前方非目标房门顺路进房",
                )

        print("[NavBypass] 顺路进房未确认 indoor，改走绕房策略")
        return False

    def _handle_front_stone_wall_on_search_route(self, w: 'FrameWorker', current_loc, phase_label='NAV') -> bool:
        self.stop_auto_forward(w)
        print(
            f"[NavBypass] {phase_label} 前方发现 stone_wall，"
            f"先短前推 y_bias={self.STONE_WALL_FORWARD_Y_BIAS}, wait={self.STONE_WALL_FORWARD_WAIT}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.STONE_WALL_FORWARD_Y_BIAS,
            dura=self.STONE_WALL_FORWARD_DURA,
            wait=self.STONE_WALL_FORWARD_WAIT,
        )
        w.refresh_frame()
        if self._get_house_scene(w) == self.HOUSE_INDOOR:
            indoor_loc = self._safe_get_frame_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                indoor_loc,
                "stone_wall 短前推后确认进房",
            )

        if w.get_info('跳跃'):
            print(f"[NavBypass] {phase_label} stone_wall 前推后出现跳跃按钮，点击跳跃")
            w.click('跳跃')
            time.sleep(self.STONE_WALL_JUMP_SETTLE_SECONDS)
        else:
            print(f"[NavBypass] {phase_label} stone_wall 前推后未识别到跳跃按钮，仍尝试跳跃前推")

        w.tap_single(
            '摇杆',
            y_bias=self.STONE_WALL_JUMP_FORWARD_Y_BIAS,
            dura=self.STONE_WALL_JUMP_FORWARD_DURA,
            wait=self.STONE_WALL_JUMP_FORWARD_WAIT,
        )
        w.refresh_frame()
        if self._get_house_scene(w) == self.HOUSE_INDOOR:
            indoor_loc = self._safe_get_frame_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                indoor_loc,
                "stone_wall 跳跃前推后确认进房",
            )

        if hasattr(self, "history_locations"):
            self.history_locations = []
        return True

    def _maybe_bypass_front_house_on_route(self, w: 'FrameWorker', current_loc, target_loc, dist, phase_label='NAV'):
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            return False

        house_scene = self._get_house_scene(w)
        if (
            dist_val <= self.ENTRY_AUTO_FORWARD_DISTANCE
            and house_scene in self.HOUSE_PROACTIVE_BYPASS_NEAR_ENTRY_SCENES
        ):
            print(f"[NavBypass] {phase_label} 已接近门/墙，跳过主动绕房")
            return False

        self.align_direction(w, target_loc, threshold=10, max_steps=1)
        w.refresh_frame()
        front_summary = self._front_route_obstacle_summary(w)

        if self._is_searching_stage_frame(w) and front_summary["stone_wall"] is not None:
            return self._handle_front_stone_wall_on_search_route(w, current_loc, phase_label)

        front_block = self._front_house_blocking(w)
        if not front_block:
            return False

        if not self._is_searching_stage_frame(w):
            return self._bypass_front_house_by_view_turn(w, target_loc, phase_label)

        if dist_val <= self.HOUSE_SEARCH_BYPASS_MIN_ENTRY_DISTANCE:
            print(
                f"[NavBypass] {phase_label} 前方有房体但距离进门点 {dist_val:.2f}<=10，"
                f"按当前目标入口处理，不主动绕房"
            )
            return False

        if front_summary["has_door"] or w.get_info('开门') or w.get_info('关门') or self.find_largest_door(w):
            print(
                f"[NavBypass] {phase_label} 前方有房体且距离进门点 {dist_val:.2f}>10，"
                f"但已定位到门，改走对准门前推逻辑"
            )
            if self._try_enter_visible_non_target_house(w, current_loc, phase_label):
                return True
            print(f"[NavBypass] {phase_label} 本轮对门前推未进房，不主动绕房，下一轮继续识别")
            return False

        if self._is_route_close_to_current_entry(target_loc, dist_val):
            print(f"[NavBypass] {phase_label} 前方可能是当前目标入门点房体，交给进门流程")
            return False

        if not front_summary["has_house"] and not front_summary["has_window"]:
            print(f"[NavBypass] {phase_label} 前方阻挡不是房子/窗户组合，不主动绕房")
            return False

        print(
            f"[NavBypass] {phase_label} 距离进门点 {dist_val:.2f}>10，"
            f"前方只有房体/窗且未看到门，执行绕房"
        )
        return self._bypass_front_house_by_view_turn(w, target_loc, phase_label)

    def _try_lock_visible_door_after_block(self, w: 'FrameWorker') -> bool:
        door = self.find_largest_door(w)
        if door is None:
            return False

        print("[Unstuck] 后退后前方是房子且定位到门，直接锁门进入交互流程")
        self.stop_auto_forward(w)
        self._align_to_door_detection(w, door)
        w.refresh_frame()

        if w.get_info('开门') or w.get_info('关门'):
            self.status = "INTERACT"
        else:
            self.status = "VISUAL_APPROACH"

        self.history_locations = []
        return True

    def _entry_near_micro_move_params(self, current_dir, target_angle):
        if current_dir is None or target_angle is None:
            return None
        try:
            current_dir = float(current_dir)
            target_angle = float(target_angle)
        except (TypeError, ValueError):
            return None

        relative = (target_angle - current_dir + 540) % 360 - 180
        if abs(relative) <= 45:
            return "forward", 0, -self.ENTRY_NEAR_MICRO_Y_BIAS, relative
        if abs(relative) >= 135:
            return "back", 0, self.ENTRY_NEAR_MICRO_Y_BIAS, relative
        if relative < 0:
            return "left", -self.ENTRY_NEAR_MICRO_X_BIAS, 0, relative
        return "right", self.ENTRY_NEAR_MICRO_X_BIAS, 0, relative

    @staticmethod
    def _entry_micro_direction_label(direction: str) -> str:
        labels = {
            "forward": "上方",
            "back": "后方",
            "left": "左边",
            "right": "右边",
        }
        return labels.get(direction, direction)

    @staticmethod
    def _door_center_x(door):
        try:
            return (float(door[0]) + float(door[2])) / 2
        except (TypeError, ValueError, IndexError):
            return None

    def _entry_door_frame_width(self):
        inf_w, inf_h = get_wh()
        return max(int(inf_w or 0), int(inf_h or 0))

    def _door_center_ratio(self, door, frame_w=None):
        if frame_w is None:
            frame_w = self._entry_door_frame_width()
        if frame_w <= 0:
            return None

        center_x = self._door_center_x(door)
        if center_x is None:
            return None
        return center_x / frame_w

    def _is_entry_door_roughly_centered(self, door, frame_w=None):
        ratio = self._door_center_ratio(door, frame_w)
        if ratio is None:
            return False
        return self.ENTRY_DOOR_DIRECT_CENTER_MIN_RATIO <= ratio <= self.ENTRY_DOOR_DIRECT_CENTER_MAX_RATIO

    def _align_visible_entry_door_for_direct_push(self, w: 'FrameWorker', door, phase_label='Nav'):
        """Move laterally first, then turn view until the door is roughly centered."""
        for step in range(self.ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS):
            frame_w = self._entry_door_frame_width()
            if frame_w <= 0:
                return None

            center_x = self._door_center_x(door)
            if center_x is None:
                return None

            left_third = frame_w / 3
            right_third = frame_w * 2 / 3
            screen_center = frame_w / 2

            if center_x < left_third:
                print(
                    f"[{phase_label}] 到达进门点后门在左侧1/3，"
                    f"先向左轻微横移对齐 {step + 1}/{self.ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS}"
                )
                w.tap_single(
                    '摇杆',
                    x_bias=-self.ENTRY_DOOR_FINAL_LATERAL_X_BIAS,
                    y_bias=0,
                    dura=self.ENTRY_DOOR_FINAL_LATERAL_DURA,
                    wait=self.ENTRY_DOOR_FINAL_LATERAL_WAIT,
                )
                w.refresh_frame()
                door = self.find_largest_door(w)
                if door is None:
                    print(f"[{phase_label}] 横移后门目标丢失，继续原进门流程")
                    return None
                continue

            if center_x > right_third:
                print(
                    f"[{phase_label}] 到达进门点后门在右侧1/3，"
                    f"先向右轻微横移对齐 {step + 1}/{self.ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS}"
                )
                w.tap_single(
                    '摇杆',
                    x_bias=self.ENTRY_DOOR_FINAL_LATERAL_X_BIAS,
                    y_bias=0,
                    dura=self.ENTRY_DOOR_FINAL_LATERAL_DURA,
                    wait=self.ENTRY_DOOR_FINAL_LATERAL_WAIT,
                )
                w.refresh_frame()
                door = self.find_largest_door(w)
                if door is None:
                    print(f"[{phase_label}] 横移后门目标丢失，继续原进门流程")
                    return None
                continue

            if left_third <= center_x <= right_third:
                if self._is_entry_door_roughly_centered(door, frame_w):
                    print(f"[{phase_label}] 门中心已大致在屏幕1/2附近，准备自动开门直推")
                    return door

                side = "左" if center_x < screen_center else "右"
                print(
                    f"[{phase_label}] 门中心已进入屏幕中间区域，"
                    f"位于{side}侧1/3到1/2范围，调整视角正对门"
                )
                self._align_to_door_detection(
                    w,
                    door,
                    tolerance_px=self.ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX,
                )
                w.refresh_frame()
                door = self.find_largest_door(w)
                if door is None:
                    print(f"[{phase_label}] 视角调整后门目标丢失，继续原进门流程")
                    return None
                if self._is_entry_door_roughly_centered(door):
                    print(f"[{phase_label}] 视角调整后门已大致居中，准备自动开门直推")
                    return door

        print(f"[{phase_label}] 门框横向预对齐达到步数上限，继续原进门流程")
        return None

    def _backoff_after_centered_entry_push_failure(self, w: 'FrameWorker', phase_label: str, failures: int, reason: str):
        print(
            f"[{phase_label}] {reason}，后拉记为进门失败 "
            f"{failures}/{self.ENTRY_DOOR_DIRECT_MAX_FAILURES}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_DIRECT_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_DIRECT_BACKOFF_WAIT,
        )
        w.refresh_frame()

    def _push_centered_entry_door_without_button(self, w: 'FrameWorker', phase_label='Nav', initial_door=None) -> str:
        failures = 0
        direct_started = False
        door = initial_door
        realign_attempts = 0

        while failures < self.ENTRY_DOOR_DIRECT_MAX_FAILURES:
            pushes_this_failure = 0
            while True:
                if self._should_abort(w):
                    return "aborted"

                w.refresh_frame()
                scene = self._get_house_scene(w)
                if scene == self.HOUSE_INDOOR:
                    print(f"[{phase_label}] 直推前已是 indoor，启动搜房策略")
                    return "indoor"

                visible_door = self.find_largest_door(w)
                if visible_door is not None:
                    if direct_started:
                        if realign_attempts >= self.ENTRY_DOOR_DIRECT_REALIGN_MAX_ATTEMPTS:
                            failures += 1
                            self._backoff_after_centered_entry_push_failure(
                                w,
                                phase_label,
                                failures,
                                f"前推后仍能看到门但重新对齐已达 {self.ENTRY_DOOR_DIRECT_REALIGN_MAX_ATTEMPTS} 次",
                            )
                            break
                        realign_attempts += 1
                        print(
                            f"[{phase_label}] 前推后仍能定位到门，继续调整视角对齐 "
                            f"{realign_attempts}/{self.ENTRY_DOOR_DIRECT_REALIGN_MAX_ATTEMPTS}"
                        )

                    aligned_door = self._align_visible_entry_door_for_direct_push(w, visible_door, phase_label)
                    if aligned_door is None:
                        if not direct_started:
                            return "not_ready"
                        print(
                            f"[{phase_label}] 重新对齐过程中门目标丢失，"
                            f"不记失败，沿当前方向继续前推"
                        )
                        door = None
                    else:
                        door = aligned_door
                elif not direct_started and door is None:
                    print(f"[{phase_label}] 门未进入视野，继续原进门流程")
                    return "not_ready"
                else:
                    if pushes_this_failure >= self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE:
                        failures += 1
                        self._backoff_after_centered_entry_push_failure(
                            w,
                            phase_label,
                            failures,
                            f"连续直推 {self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE} 次仍未进房且未再看到门",
                        )
                        break
                    print(f"[{phase_label}] 本次前推前未识别到门，沿当前进门方向继续直推")

                print(
                    f"[{phase_label}] 自动开门策略直推进门: "
                    f"y_bias={self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS}, "
                    f"dura={self.ENTRY_DOOR_DIRECT_FORWARD_DURA}, "
                    f"wait={self.ENTRY_DOOR_DIRECT_FORWARD_WAIT}"
                )
                w.tap_single(
                    '摇杆',
                    y_bias=self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS,
                    dura=self.ENTRY_DOOR_DIRECT_FORWARD_DURA,
                    wait=self.ENTRY_DOOR_DIRECT_FORWARD_WAIT,
                )
                direct_started = True
                pushes_this_failure += 1
                w.refresh_frame()

                scene = self._get_house_scene(w)
                if scene == self.HOUSE_INDOOR:
                    print(f"[{phase_label}] 自动开门直推后 house_scene=indoor，启动搜房策略")
                    return "indoor"

                if scene == self.HOUSE_NEAR_WALL:
                    failures += 1
                    self._backoff_after_centered_entry_push_failure(
                        w,
                        phase_label,
                        failures,
                        "直推后检测到贴墙/撞墙",
                    )
                    break

                visible_after_push = self.find_largest_door(w)
                if visible_after_push is not None:
                    door = visible_after_push
                    print(
                        f"[{phase_label}] 直推后仍能定位到门，"
                        f"下一轮继续调整视角后再前推"
                    )
                    continue

                if pushes_this_failure >= self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE:
                    failures += 1
                    self._backoff_after_centered_entry_push_failure(
                        w,
                        phase_label,
                        failures,
                        f"连续直推 {self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE} 次仍未进房且未再看到门",
                    )
                    break

                print(
                    f"[{phase_label}] 直推后暂未进房 house_scene={scene}，"
                    f"本轮已推 {pushes_this_failure}/{self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE}"
                )

        print(f"[{phase_label}] 自动开门直推累计失败 {failures} 次，判定当前进门点失败")
        return "failed"

    def _align_entry_door_after_arrival(self, w: 'FrameWorker', phase_label='Nav') -> str:
        """Arrived at the entry point: align the visible door, then push through auto-open."""
        door = self.find_largest_door(w)
        if door is None:
            print(f"[{phase_label}] 到达进门点后未识别到门，继续原进门流程")
            return "not_visible"

        self.stop_auto_forward(w)
        aligned_door = self._align_visible_entry_door_for_direct_push(w, door, phase_label)
        if aligned_door is None:
            return "not_ready"

        return self._push_centered_entry_door_without_button(w, phase_label, aligned_door)

    def _align_entry_direction_at_near_point(self, w: 'FrameWorker', phase_label='Nav') -> bool:
        ideal_angle = self.active_entry.get('direction') if self.active_entry else None
        if ideal_angle is None:
            return True

        print(
            f"[{phase_label}] 距离进门点<={self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:g}，"
            f"先对准进门点方向: {ideal_angle}"
        )
        aligned = self._align_near_entry_direction(w, ideal_angle)
        if aligned:
            w.refresh_frame()
        return aligned

    def _align_near_entry_direction(self, w: 'FrameWorker', ideal_angle) -> bool:
        return execute_view_turn(
            w,
            w.get_info('direction'),
            ideal_angle,
            threshold=getattr(self, 'ENTRY_DIRECTION_ALIGN_TOLERANCE', self.ENTRY_NEAR_ALIGN_TOLERANCE),
            max_steps=getattr(self, 'ENTRY_DIRECTION_ALIGN_MAX_STEPS', self.ENTRY_NEAR_ALIGN_MAX_STEPS),
            wait=self.ENTRY_NEAR_ALIGN_WAIT,
            min_dura=self.ENTRY_NEAR_ALIGN_MIN_DURA,
            max_dura=self.ENTRY_NEAR_ALIGN_MAX_DURA,
            max_px=self.ENTRY_NEAR_ALIGN_MAX_BIAS,
            log_prefix="[EntryNearAlign]",
        )

    def _correct_near_entry_lateral_position_once(
        self,
        w: 'FrameWorker',
        current_loc,
        target_loc,
        dist: float,
        phase_label='Nav',
    ) -> bool:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            return False

        if dist_val <= self.ENTRY_NEAR_MICRO_DONE_DISTANCE:
            print(f"[{phase_label}] 距离进门点 {dist_val:.2f}，无需左右位置修正")
            return False

        refreshed_loc = self._get_current_location(w) or current_loc
        current_dir = w.get_info('direction')
        target_angle = calculate_angle(refreshed_loc, target_loc)
        if current_dir is None or target_angle is None:
            print(f"[{phase_label}] 近门左右位置修正缺少方向/坐标，跳过")
            return False

        try:
            relative = (float(target_angle) - float(current_dir) + 540) % 360 - 180
        except (TypeError, ValueError):
            print(f"[{phase_label}] 近门左右位置修正角度无效，跳过")
            return False

        abs_relative = abs(relative)
        if (
            abs_relative <= self.ENTRY_NEAR_LATERAL_CORRECT_MIN_RELATIVE_DEGREES
            or abs_relative >= self.ENTRY_NEAR_LATERAL_CORRECT_MAX_RELATIVE_DEGREES
        ):
            print(
                f"[{phase_label}] 近门位置偏差主要在前后方向，"
                f"不做前后修正 relative={relative:.1f}"
            )
            return False

        side = "右" if relative > 0 else "左"
        x_bias = (
            self.ENTRY_NEAR_LATERAL_CORRECT_X_BIAS
            if relative > 0
            else -self.ENTRY_NEAR_LATERAL_CORRECT_X_BIAS
        )
        print(
            f"[{phase_label}] 对准入门方向后做一次左右位置修正: "
            f"dist={dist_val:.2f}, target_angle={target_angle:.1f}, "
            f"current_dir={float(current_dir):.1f}, relative={relative:.1f}, 向{side}推"
        )
        w.tap_single(
            '摇杆',
            x_bias=x_bias,
            y_bias=0,
            dura=self.ENTRY_NEAR_LATERAL_CORRECT_DURA,
            wait=self.ENTRY_NEAR_LATERAL_CORRECT_WAIT,
        )
        w.refresh_frame()
        self.history_locations = []
        return True

    def _handle_near_entry_point(self, w: 'FrameWorker', current_loc, target_loc, dist: float, phase_label='Nav') -> str:
        self.stop_auto_forward(w)
        if not self._align_entry_direction_at_near_point(w, phase_label):
            print(f"[{phase_label}] 进门点方向尚未对准，等待下一轮继续对准")
            return "aligning"

        self._correct_near_entry_lateral_position_once(w, current_loc, target_loc, dist, phase_label)

        arrival_result = self._align_entry_door_after_arrival(w, phase_label)
        if arrival_result == "not_visible":
            return "not_ready"

        if arrival_result != "not_ready":
            self._reset_entry_near_micro_adjust()
            return arrival_result

        self._reset_entry_near_micro_adjust()
        return "not_ready"

    def _micro_adjust_near_entry_point(self, w: 'FrameWorker', current_loc, target_loc, dist: float) -> bool:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            return False

        if dist_val > self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:
            self._reset_entry_near_micro_adjust()
            return False
        if dist_val <= self.ENTRY_NEAR_MICRO_DONE_DISTANCE:
            return False
        if self.entry_near_micro_adjust_attempts >= self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS:
            print(
                f"[Nav] 进门点近距离微调已达上限 "
                f"{self.entry_near_micro_adjust_attempts}/{self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS}，进入进门流程"
            )
            return False

        ideal_angle = self.active_entry.get('direction') if self.active_entry else None
        if ideal_angle is not None:
            self._align_near_entry_direction(w, ideal_angle)
            w.refresh_frame()

        refreshed_loc = self._get_current_location(w) or current_loc
        current_dir = w.get_info('direction') or ideal_angle
        target_angle = calculate_angle(refreshed_loc, target_loc)
        move_params = self._entry_near_micro_move_params(current_dir, target_angle)
        if move_params is None:
            return False

        direction, x_bias, y_bias, relative = move_params
        self.entry_near_micro_adjust_attempts += 1
        print(
            f"[Nav] 距离进门点 {dist_val:.2f}<={self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:g}，"
            f"未识别到门，已对准进门方向 {ideal_angle}，"
            f"目标点在{self._entry_micro_direction_label(direction)}，轻推摇杆微调 "
            f"{self.entry_near_micro_adjust_attempts}/{self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS} "
            f"(relative={relative:.1f}, x_bias={x_bias}, y_bias={y_bias})"
        )
        w.tap_single(
            '摇杆',
            x_bias=x_bias,
            y_bias=y_bias,
            dura=self.ENTRY_NEAR_MICRO_DURA,
            wait=self.ENTRY_NEAR_MICRO_WAIT,
        )
        w.refresh_frame()
        self.history_locations = []
        return True

    def searching_logic(self, w: 'FrameWorker', current_loc, current_direction):
        if self._should_abort(w):
            return

        # --- 屋内卡死兜底检测 ---
        house_scene = self._get_house_scene(w)
        if house_scene == 0 and self._is_entry_approach_status():
            if self._handle_indoor_during_entry_route(w, current_loc, "前往进门点途中检测到 indoor"):
                return

        if house_scene == 0 and not self._is_entry_approach_status():
            self.indoor_stuck_frames += 1
            if self.indoor_stuck_frames > 30:
                print('[Searching] 检测到长时间困在屋内 (house_scene=0)，启动兜底出房策略')
                self.house_exit_manager.reset()
                for _ in range(20):
                    if self._should_abort(w):
                        return
                    if self.house_exit_manager.process(w):
                        print('[Searching] 兜底出房成功，切换到跑图阶段')
                        self.indoor_stuck_frames = 0
                        self.searching_number = 0
                        self.completed_houses.add(self.current_house_id)
                        self.current_house_id = None
                        self.status = "IDLE"
                        self._continue_searching_until_timer(w, '兜底出房成功')
                        return
                print('[Searching] 兜底出房失败，强制重置状态切跑图')
                self.indoor_stuck_frames = 0
                self.searching_number = 0
                self.current_house_id = None
                self.status = "IDLE"
                self._continue_searching_until_timer(w, '兜底出房失败')
                return
        else:
            self.indoor_stuck_frames = 0

        # --- 智能选点 ---
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
            target_dist = get_distance(current_loc, self.active_entry['location'])
            print(
                f"[Searching] 锁定目标: {self.current_house_id} | "
                f"入口={self.active_entry['location']} | 距离={target_dist:.2f}"
            )
            self.history_locations = []  # 切换目标时清空历史

        target_loc = self.active_entry['location']
        dist = get_distance(current_loc, target_loc)

        # --- 快速前进 (距离 > 30.0 才使用自动前进) ---
        if self.status == "FAST_NAV":
            if self._jump_forward_if_visible_near_house(w, "FAST_NAV 靠近房子"):
                return

            # 卡顿检测逻辑
            if self._is_house_bypass_unstuck_paused():
                self.history_locations = []
                print("[Nav] 绕房视角调整/前推冷却中，跳过通用避障检测")
            elif self.update_and_check_stuck(current_loc):
                if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "FAST_NAV"):
                    return
                print("[Nav] 检测到人物卡死，启动避障程序...")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                self.history_locations = []
                return

            if dist <= self.ENTRY_AUTO_FORWARD_DISTANCE:
                print(f"[Nav] 进入摇杆分段导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.align_direction(w, target_loc)
            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "FAST_NAV"):
                return

            if not self.auto_forward:
                w.click('自动前进')
                self.auto_forward = True

            self.handle_jump_logic(w)

        # --- 分段摇杆逼近 ---
        elif self.status == "PRECISE_NAV":
            if self._jump_forward_if_visible_near_house(w, "PRECISE_NAV 靠近房子"):
                return

            # --- [修改 1] 在精细导航阶段加入卡顿检测 ---
            # 原因：即使在慢速移动时，也可能卡在树根或小障碍物上
            if self._is_house_bypass_unstuck_paused():
                self.history_locations = []
                print("[Nav] (Precise) 绕房视角调整/前推冷却中，跳过通用避障检测")
            elif self.update_and_check_stuck(current_loc):
                if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "PRECISE_NAV"):
                    return
                print("[Nav] (Precise) 检测到人物卡死，启动避障程序...")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                self.history_locations = []  # 清空历史，防止重复触发
                return
            # ----------------------------------------

            if dist <= self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:
                near_result = self._handle_near_entry_point(w, current_loc, target_loc, dist, "Nav")
                if near_result == "adjusting":
                    self.handle_jump_logic(w)
                    return

                print(f"[Nav] 已到达进门点 (距离 {dist:.2f})")
                if near_result == "indoor":
                    self._complete_current_house_search(w, "自动开门直推进房成功")
                    return
                if near_result == "failed":
                    if self.active_entry:
                        self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                    return
                if near_result in {"aborted", "aligning"}:
                    return
                self._reset_entry_near_micro_adjust()
                self.status = "SCANNING"
                return

            self._reset_entry_near_micro_adjust()

            if dist <= self.ENTRY_ARRIVAL_DISTANCE:
                print(f"[Nav] 已到达进门点 (距离 {dist:.2f})")
                arrival_result = self._align_entry_door_after_arrival(w, "Nav")
                if arrival_result == "indoor":
                    self._complete_current_house_search(w, "自动开门直推进房成功")
                    return
                if arrival_result == "failed":
                    if self.active_entry:
                        self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                    return
                if arrival_result == "aborted":
                    return
                self.status = "SCANNING"
                return

            self.stop_auto_forward(w)
            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "PRECISE_NAV"):
                return

            self.align_direction(w, target_loc)
            before_dist = dist
            mode = self._entry_forward_mode(dist)
            y_bias, dura, wait = self._get_entry_move_params(dist)
            print(f"[Nav] 分段推进到进门点: dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}")
            w.tap_single('摇杆', y_bias=y_bias, dura=dura, wait=wait)
            w.refresh_frame()
            after_loc = self._get_current_location(w)
            after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
            update_adaptive_forward_motion(mode, before_dist, before_dist, after_dist, y_bias, dura, wait)
            self.handle_jump_logic(w)

        # --- 进门点扫描 ---
        elif self.status == "SCANNING":
            print("[Scan] 到达点位，开始门检测...")
            ideal_angle = self.active_entry['direction']
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if self.check_and_lock_door(w):
                self.status = "VISUAL_APPROACH"
                return

            scan_offsets = [30, -30]
            found_door = False
            for offset in scan_offsets:
                target_angle = (ideal_angle + offset) % 360
                print(f"[Scan] 尝试角度: {target_angle} (偏移 {offset})")
                self.align_direction_blocking(w, w.get_info('direction'), target_angle)
                w.refresh_frame()

                if self.check_and_lock_door(w):
                    found_door = True
                    self.status = "VISUAL_APPROACH"
                    break
                else:
                    print(f"[Data] 角度 {target_angle} 未发现门，保存样本")
                    self.save_dataset_image(w.frame, f"no_door_offset_{offset}")

            if not found_door:
                print("[Scan] All angles scanned, door not found. Discarding current point.")
                self.completed_houses.add(self.current_house_id)
                self.handle_failed_entry_logic(ideal_angle)
                self.status = "IDLE"

        # --- 视觉对齐与推进 ---
        elif self.status == "VISUAL_APPROACH":
            for _ in range(self.VISUAL_APPROACH_MAX_ATTEMPTS):
                if self._should_abort(w):
                    return
                door = self.find_largest_door(w)
                if not door:
                    print("[Visual] 丢失目标，回到扫描")
                    self.status = "SCANNING"
                    return

                inf_w, inf_h = get_wh()
                frame_w = max(inf_w, inf_h)
                scale = self.screen_w / frame_w
                door_center_x = (door[0] + door[2]) / 2
                offset_real = (door_center_x - (frame_w / 2)) * scale

                if abs(offset_real) <= 80:
                    print("[Visual] 对齐完成，尝试交互")
                    self.status = "INTERACT"
                    break

                adjust_val = int(offset_real * 0.33)
                adjust_val = max(-400, min(400, adjust_val))
                w.tap_single('视角', x_bias=adjust_val, dura=500, wait=500)
                w.refresh_frame()
            else:
                print("[Visual] 多次视觉对齐失败，舍弃当前进门点")
                self.handle_failed_entry_logic(self.active_entry['direction'])
                self.status = "IDLE"
                return

        # --- 交互逻辑 ---
        elif self.status == "INTERACT":
            print(f"[Interact] 尝试在 {self.current_house_id} 寻找交互按钮...")
            success = False
            for i in range(10):
                if self._should_abort(w):
                    return
                w.refresh_frame()

                # --- [修改 2] 交互前移时加入跳跃检测 ---
                # 原因：门前可能有台阶或门槛，不跳跃无法靠近
                if w.get_info('跳跃'):
                    print("[Interact] 门前检测到障碍，尝试跳跃")
                    self.handle_jump_logic(w)  # 执行跳跃并前冲
                    w.refresh_frame()
                    continue  # 跳跃动作较大，跳过本次微调，直接进入下一次循环检查按钮
                # -----------------------------------

                if w.get_info('开门'):
                    w.click('开门')
                    time.sleep(1)
                    success = True
                    break
                if w.get_info('关门'):
                    print("[Interact] 检测到关门按钮，表示门已打开，不点击关门，直接准备入户")
                    success = True
                    break
                if not self._advance_towards_entry_door(w):
                    print("[Interact] 门目标丢失且兜底恢复失败")
                    break

            if success:
                print("[Interact] 交互成功，准备入户")
                self.status = "FINAL_ENTRY"
            else:
                print(f"[Interact] 警告：交互失败，舍弃进门点")
                if self.active_entry:
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                else:
                    self.current_house_id = None
                self.status = "IDLE"
                return

        # --- 最终入户 ---
        elif self.status == "FINAL_ENTRY":
            if self.active_entry:
                ideal_angle = self.active_entry['direction']
                print(f"[Entry] 调整至进门角度: {ideal_angle}")
                self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)
            print("[Entry] 进门并确认 house_scene")
            if not self._push_until_entered_house(w):
                print("[Entry] 多次推进后仍未进入房屋，舍弃当前进门点")
                if self.active_entry:
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                else:
                    self.current_house_id = None
                self.status = "IDLE"
                return

            if self._should_abort(w):
                return
            if not self.start_searching(w):
                return
            if w.current_stage != '搜房阶段':
                return
            self.completed_houses.add(self.current_house_id)
            self.searching_number += 1
            print(f"[Finish] 房屋 {self.current_house_id} 完成，累计已搜 {self.searching_number} 个")
            w.refresh_frame()
            exit_direction = w.get_info('direction')
            self.prepare_next_target_logic(exit_direction)
            self.current_house_id = None
            self.active_entry = None
            self.status = "IDLE"
            self.history_locations = []
            self._reset_route_stuck_bypass()

    def update_and_check_stuck(self, current_loc):
        self.history_locations.append(current_loc)
        if len(self.history_locations) > self.max_history_len:
            self.history_locations.pop(0)

        if len(self.history_locations) < self.max_history_len:
            return False

        x_coords = [loc[0] for loc in self.history_locations]
        y_coords = [loc[1] for loc in self.history_locations]
        max_dist = math.sqrt((max(x_coords) - min(x_coords)) ** 2 + (max(y_coords) - min(y_coords)) ** 2)
        return max_dist < self.stuck_threshold

    def _get_stable_initial_location(self, current_loc):
        """落地后等待小地图坐标稳定，避免沿用跳伞前旧位置选错最近入口。"""
        loc = tuple(current_loc)
        if self.initial_location_samples:
            prev = self.initial_location_samples[-1]
            jump_dist = get_distance(prev, loc)
            if jump_dist >= self.INITIAL_LOCATION_JUMP_RESET_DISTANCE:
                print(
                    f"[Searching] 落地坐标跳变 {jump_dist:.2f}，"
                    f"丢弃旧样本 prev={prev}, current={loc}"
                )
                self.initial_location_samples = [loc]
                return None

        self.initial_location_samples.append(loc)
        if len(self.initial_location_samples) > self.INITIAL_LOCATION_MAX_SAMPLES:
            self.initial_location_samples.pop(0)

        if len(self.initial_location_samples) < self.INITIAL_LOCATION_MIN_SAMPLES:
            print(
                f"[Searching] 等待落地位置稳定 "
                f"{len(self.initial_location_samples)}/{self.INITIAL_LOCATION_MIN_SAMPLES}: {loc}"
            )
            return None

        x_coords = [item[0] for item in self.initial_location_samples]
        y_coords = [item[1] for item in self.initial_location_samples]
        spread = math.sqrt((max(x_coords) - min(x_coords)) ** 2 + (max(y_coords) - min(y_coords)) ** 2)
        if spread <= self.INITIAL_LOCATION_STABLE_DISTANCE:
            print(f"[Searching] 落地位置已稳定: {loc}, spread={spread:.2f}")
            return loc

        if len(self.initial_location_samples) >= self.INITIAL_LOCATION_MAX_SAMPLES:
            print(f"[Searching] 落地位置仍有波动，使用最新坐标: {loc}, spread={spread:.2f}")
            return loc

        print(f"[Searching] 落地位置仍在刷新: latest={loc}, spread={spread:.2f}")
        return None

    def _entry_forward_mode(self, dist: float) -> str:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            dist_val = 0.0
        return "fast" if dist_val > self.ENTRY_COARSE_MOVE_DISTANCE else "slow"

    def _get_entry_move_params(self, dist):
        mode = self._entry_forward_mode(dist)
        try:
            dist_val = max(0.0, float(dist))
        except (TypeError, ValueError):
            dist_val = 0.0
        fallback_y_bias = -500 if mode == "fast" else -100
        fallback_dura = 300
        fallback_wait = int(max(
            180,
            min(7000, dist_val * (32 if mode == "fast" else 60) + (220 if mode == "fast" else 300)),
        ))
        y_bias, dura, wait, _ = get_adaptive_forward_motion(
            mode,
            dist_val,
            fallback_y_bias,
            fallback_dura,
            fallback_wait,
        )
        return y_bias, dura, wait

    def _get_current_location(self, w: 'FrameWorker'):
        raw = w.get_info('location')
        return self._normalize_location_value(raw)

    def _push_until_entered_house(self, w: 'FrameWorker') -> bool:
        if self._get_house_scene(w) == 0:
            print("[Entry] 已检测到 house_scene=0，确认已进屋")
            return True

        ideal_angle = self.active_entry['direction'] if self.active_entry else None
        for attempt in range(self.ENTRY_CONFIRM_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            if ideal_angle is not None:
                self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if attempt == 0:
                x_bias = 0
                print(f"[Entry] 正前推进确认入屋 {attempt + 1}/{self.ENTRY_CONFIRM_MAX_ATTEMPTS}")
            else:
                x_bias = self.ENTRY_CONFIRM_SIDE_X_BIAS if attempt % 2 == 1 else -self.ENTRY_CONFIRM_SIDE_X_BIAS
                side = "右前方" if x_bias > 0 else "左前方"
                print(f"[Entry] house_scene 仍非 0，向{side}推进确认入屋 {attempt + 1}/{self.ENTRY_CONFIRM_MAX_ATTEMPTS}")

            w.tap_single(
                '摇杆',
                x_bias=x_bias,
                y_bias=self.ENTRY_CONFIRM_FORWARD_Y_BIAS,
                dura=self.ENTRY_CONFIRM_FORWARD_DURA,
                wait=self.ENTRY_CONFIRM_FORWARD_WAIT,
            )
            w.refresh_frame()
            time.sleep(0.2)

            house_scene = self._get_house_scene(w)
            if house_scene == 0:
                print("[Entry] 推进后 house_scene=0，确认已进屋")
                return True

        return False

    def _is_entry_approach_status(self):
        return self.status in {"FAST_NAV", "PRECISE_NAV", "SCANNING", "VISUAL_APPROACH", "INTERACT", "FINAL_ENTRY"}

    def _backoff_and_recheck_house_scene(self, w: 'FrameWorker'):
        print("[Unstuck] house_scene=indoor，可能是贴墙误判，先后退复核室内/室外")
        w.tap_single(
            '摇杆',
            y_bias=300,
            dura=self.ENTRY_WALL_BACKOFF_DURA,
            wait=self.ENTRY_WALL_BACKOFF_WAIT,
        )
        w.refresh_frame()
        return self._get_house_scene(w)

    def _next_route_stuck_attempt(self, current_loc):
        loc = self._normalize_location_value(current_loc)
        if loc is None:
            self.route_stuck_bypass_attempts += 1
            return self.route_stuck_bypass_attempts

        if (
            self.route_stuck_reference_loc is not None
            and get_distance(self.route_stuck_reference_loc, loc) <= self.ROUTE_STUCK_REPEAT_RADIUS
        ):
            self.route_stuck_bypass_attempts += 1
        else:
            self.route_stuck_reference_loc = loc
            self.route_stuck_bypass_attempts = 1
        return self.route_stuck_bypass_attempts

    def _route_stuck_forward_wait(self, attempt: int) -> int:
        return int(min(
            self.ROUTE_STUCK_BYPASS_FORWARD_MAX_WAIT,
            self.ROUTE_STUCK_BYPASS_FORWARD_BASE_WAIT
            + max(0, attempt - 1) * self.ROUTE_STUCK_BYPASS_FORWARD_STEP_WAIT,
        ))

    def _route_stuck_turn_degrees(self, attempt: int) -> int:
        return int(min(
            self.ROUTE_STUCK_MAX_TURN_DEGREES,
            self.ROUTE_STUCK_TURN_DEGREES
            + max(0, attempt - 1) * self.ROUTE_STUCK_TURN_ESCALATE_STEP,
        ))

    def _route_stuck_backoff_motion(self, attempt: int):
        level = max(0, attempt - 1)
        dura = int(min(
            self.ROUTE_STUCK_BACKOFF_MAX_DURA,
            self.ROUTE_STUCK_BACKOFF_BASE_DURA + level * self.ROUTE_STUCK_BACKOFF_DURA_STEP,
        ))
        wait = int(min(
            self.ROUTE_STUCK_BACKOFF_MAX_WAIT,
            self.ROUTE_STUCK_BACKOFF_BASE_WAIT + level * self.ROUTE_STUCK_BACKOFF_WAIT_STEP,
        ))
        return self.ROUTE_STUCK_BACKOFF_Y_BIAS, dura, wait

    def _route_stuck_forward_motion(self, attempt: int):
        level = max(0, attempt - 1)
        dura = int(min(
            self.ROUTE_STUCK_BYPASS_FORWARD_MAX_DURA,
            self.ROUTE_STUCK_BYPASS_FORWARD_DURA + level * self.ROUTE_STUCK_BYPASS_FORWARD_DURA_STEP,
        ))
        wait = self._route_stuck_forward_wait(attempt)
        return self.ROUTE_STUCK_BYPASS_FORWARD_Y_BIAS, dura, wait

    def _resume_entry_direction_after_bypass(self, w: 'FrameWorker', target_loc):
        current_loc = self._get_current_location(w)
        if current_loc is None or target_loc is None:
            return

        self.align_direction(w, target_loc, threshold=10, max_steps=1)
        dist = get_distance(current_loc, target_loc)
        if dist <= self.ENTRY_ARRIVAL_DISTANCE:
            return

        mode = self._entry_forward_mode(dist)
        y_bias, dura, wait = self._get_entry_move_params(dist)
        print(
            f"[Unstuck] 绕障后恢复朝进门点推进: "
            f"dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        before_dist = dist
        w.tap_single('摇杆', y_bias=y_bias, dura=dura, wait=wait)
        w.refresh_frame()
        after_loc = self._get_current_location(w)
        after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
        update_adaptive_forward_motion(mode, before_dist, before_dist, after_dist, y_bias, dura, wait)

    def _recover_route_stuck_by_side_forward(
        self,
        w: 'FrameWorker',
        current_loc,
        target_loc,
        backoff_first: bool = True,
    ) -> bool:
        self.stop_auto_forward(w)
        attempt = self._next_route_stuck_attempt(current_loc)

        if backoff_first:
            backoff_y_bias, backoff_dura, backoff_wait = self._route_stuck_backoff_motion(attempt)
            print(
                f"[Unstuck] 前往进门点卡住，先后退复位 attempt={attempt}, "
                f"y_bias={backoff_y_bias}, dura={backoff_dura}, wait={backoff_wait}"
            )
            w.tap_single('摇杆', y_bias=backoff_y_bias, dura=backoff_dura, wait=backoff_wait)
            w.refresh_frame()
            if self._get_house_scene(w) == 0:
                loc_after_back = self._get_current_location(w) or current_loc
                return self._handle_indoor_during_entry_route(
                    w,
                    loc_after_back,
                    "卡住后后退复核确认误入房",
                )

        side = self._choose_house_bypass_side(w)
        current_dir = w.get_info('direction')
        turn_degrees = self._route_stuck_turn_degrees(attempt)
        if current_dir is not None:
            turn_delta = turn_degrees if side == "right" else -turn_degrees
            target_dir = (float(current_dir) + turn_delta) % 360
            print(
                f"[Unstuck] 视野向{side}侧转 {turn_degrees}° "
                f"复核室内/室外: attempt={attempt}, target={target_dir:.1f}"
            )
            self.align_direction_blocking(
                w,
                current_dir,
                target_dir,
                threshold=10,
                max_steps=4,
            )
            w.refresh_frame()

        if self._get_house_scene(w) == 0:
            loc_after_turn = self._get_current_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                loc_after_turn,
                "卡住后转向复核确认误入房",
            )

        forward_y_bias, forward_dura, wait = self._route_stuck_forward_motion(attempt)
        print(
            f"[Unstuck] 确认为室外卡住，沿{side}侧前推绕开障碍 "
            f"attempt={attempt}, y_bias={forward_y_bias}, dura={forward_dura}, wait={wait}"
        )
        w.tap_single(
            '摇杆',
            y_bias=forward_y_bias,
            dura=forward_dura,
            wait=wait,
        )
        w.refresh_frame()

        if self._get_house_scene(w) == 0:
            loc_after_forward = self._get_current_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                loc_after_forward,
                "绕障前推后确认误入房",
            )

        self._resume_entry_direction_after_bypass(w, target_loc)
        self.history_locations = []
        return True

    def execute_unstuck_logic(self, w: 'FrameWorker', current_loc):
        self.stop_auto_forward(w)
        target_loc = self.active_entry['location'] if self.active_entry else None

        def _safe_get_loc():
            return self._get_current_location(w)

        if self._get_house_scene(w) == 0:
            house_scene_after_backoff = self._backoff_and_recheck_house_scene(w)
            if house_scene_after_backoff != 0:
                print("[Unstuck] 后退复核后已不判定为室内，按室外卡住绕障")
                return self._recover_route_stuck_by_side_forward(
                    w,
                    _safe_get_loc() or current_loc,
                    target_loc,
                    backoff_first=False,
                )

            return self._handle_indoor_during_entry_route(
                w,
                _safe_get_loc() or current_loc,
                "后退复核后仍为 indoor",
            )

        if self._recover_route_stuck_by_side_forward(w, current_loc, target_loc):
            return True

        if self._bypass_front_house_block(w, current_loc, _safe_get_loc):
            return True

        if w.get_info('跳跃'):
            print("[Unstuck] 尝试跳跃脱困")
            self.handle_jump_logic(w)
            w.tap_single('摇杆', y_bias=-300, dura=500, wait=1000)
            w.refresh_frame()
            loc_raw = w.get_info('location')
            if loc_raw is not None:
                new_loc = check_location(loc_raw[0])
                if new_loc and get_distance(current_loc, new_loc) > self.stuck_threshold:
                    print("[Unstuck] 跳跃脱困成功")
                    return True

        print("[Unstuck] 跳跃无效，进入 U 型避障移动...")
        for _ in range(self.UNSTUCK_MAX_CYCLES):
            if self._should_abort(w):
                return False
            print("[Unstuck] 后退...")
            w.tap_single('摇杆', y_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_back = _safe_get_loc()
            if not loc_after_back:
                continue

            print("[Unstuck] 右移试探...")
            w.tap_single('摇杆', x_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_right = _safe_get_loc()

            side_way_clear = False
            last_valid_loc = loc_after_back

            if loc_after_right and get_distance(loc_after_back, loc_after_right) > 0.5:
                print("[Unstuck] 右侧可通行")
                side_way_clear = True
                last_valid_loc = loc_after_right
            else:
                print("[Unstuck] 右侧受阻，左移试探...")
                w.tap_single('摇杆', x_bias=-300, dura=300, wait=1500)
                w.refresh_frame()
                loc_after_left = _safe_get_loc()

                side_base_loc = loc_after_right or loc_after_back
                if loc_after_left and get_distance(side_base_loc, loc_after_left) > 0.5:
                    print("[Unstuck] 左侧可通行")
                    side_way_clear = True
                    last_valid_loc = loc_after_left

            if not side_way_clear:
                print("[Unstuck] 左右均受阻 (U型死角)，再次后退...")
                continue

            print("[Unstuck] 尝试向前突破...")
            for _ in range(self.UNSTUCK_FORWARD_STEPS):
                if self._should_abort(w):
                    return False
                w.tap_single('摇杆', y_bias=-300, dura=300, wait=2000)
                w.refresh_frame()
                loc_after_forward = _safe_get_loc()

                if loc_after_forward and get_distance(last_valid_loc, loc_after_forward) > 0.5:
                    print("[Unstuck] 脱困成功！")
                    return True
                else:
                    print("[Unstuck] 前方依然受阻，继续侧向移动...")
                    moved_side = False
                    for bias in [300, -300]:
                        if self._should_abort(w):
                            return False
                        w.tap_single('摇杆', x_bias=bias, dura=300, wait=1500)
                        w.refresh_frame()
                        temp_loc = _safe_get_loc()
                        move_base_loc = loc_after_forward or last_valid_loc
                        if temp_loc and get_distance(move_base_loc, temp_loc) > 0.5:
                            last_valid_loc = temp_loc
                            moved_side = True
                            break

                    if not moved_side:
                        print("[Unstuck] 前方死路，重新执行后退逻辑")
                        break
        print("[Unstuck] 脱困超过最大尝试次数，放弃当前进门点")
        return False

    def handle_jump_logic(self, w: 'FrameWorker'):
        if w.get_info('跳跃'):
            print("[Jump] 检测到障碍，执行跳跃")
            self.stop_auto_forward(w)
            w.click('跳跃')
            time.sleep(0.2)
            w.tap_single('摇杆', y_bias=-400, dura=100, wait=300)
            w.refresh_frame()

    def _jump_forward_if_visible_near_house(self, w: 'FrameWorker', phase_label: str) -> bool:
        if not w.get_info('跳跃'):
            return False

        print(f"[Jump] {phase_label} 检测到跳跃按钮，立即跳跃并前推")
        self.handle_jump_logic(w)
        self.history_locations = []
        return True

    def select_nearest_entry(self, current_loc):
        """落地后根据当前位置，从 house_data 中计算距离最近的进门点。"""
        best_dist = float('inf')
        best_id = None
        best_entry = None

        for house_id, entries in self.house_data.items():
            if self._is_excluded_house(house_id):
                continue
            if house_id in self.completed_houses:
                continue
            for entry in entries:
                dist = get_distance(current_loc, entry['location'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = house_id
                    best_entry = entry

        self.current_house_id = best_id
        self.active_entry = best_entry
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self._reset_route_stuck_bypass()

    def select_smart_target(self, current_loc, current_direction):
        best_dist = float('inf')
        best_id = None
        best_entry = None
        avoid_angle = getattr(self, 'avoid_angle_ref', None)
        avoid_mode = getattr(self, 'avoid_mode', None)

        for house_id, entries in self.house_data.items():
            if self._is_excluded_house(house_id): continue
            if house_id in self.completed_houses: continue
            if house_id in self.temp_skip_houses: continue

            for entry in entries:
                dist = get_distance(current_loc, entry['location'])
                if avoid_angle is not None:
                    angle_to_target = calculate_angle(current_loc, entry['location'])
                    if angle_to_target is None:
                        continue
                    diff = abs(angle_to_target - avoid_angle)
                    if diff > 180: diff = 360 - diff
                    if avoid_mode == 'SAME' and diff < 45: continue
                    if avoid_mode == 'OPPOSITE' and diff > 135: continue

                if dist < best_dist:
                    best_dist = dist
                    best_id = house_id
                    best_entry = entry

        self.current_house_id = best_id
        self.active_entry = best_entry
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self.temp_skip_houses.clear()
        self._reset_route_stuck_bypass()

    def handle_failed_entry_logic(self, failed_entry_angle):
        print(f"[Smart] 进门失败，临时跳过 {self.current_house_id}")
        self.temp_skip_houses.add(self.current_house_id)
        self.current_house_id = None
        self.avoid_angle_ref = failed_entry_angle
        self.avoid_mode = 'SAME'
        self._reset_entry_near_micro_adjust()

    def prepare_next_target_logic(self, exit_direction):
        self.avoid_angle_ref = exit_direction
        self.avoid_mode = 'OPPOSITE'

    def check_and_lock_door(self, w):
        if self.find_largest_door(w):
            return True
        return False

    def save_dataset_image(self, frame, suffix):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = f"temp/no_door/{timestamp}_{suffix}.jpg"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"[Data] 已保存图片: {path}")
        except Exception as e:
            print(f"[Data] 保存图片失败: {e}")

    def stop_auto_forward(self, w):
        if self.auto_forward:
            w.click('自动前进')
            self.auto_forward = False

    def align_direction_blocking(
        self,
        w,
        current_dir,
        target_angle,
        threshold=5,
        max_steps=10,
        wait=None,
        min_dura=None,
    ):
        return execute_view_turn(
            w,
            current_dir,
            target_angle,
            threshold=threshold,
            max_steps=max_steps,
            wait=self.ALIGN_WAIT if wait is None else wait,
            min_dura=self.ALIGN_MIN_DURA if min_dura is None else min_dura,
            max_dura=self.ALIGN_MAX_DURA,
            max_px=self.ALIGN_MAX_BIAS,
            log_prefix="[NavAlign]",
        )

    def align_direction(self, w, tar_loc, threshold=8, max_steps=1, wait=None):
        location_raw = w.get_info('location')
        if location_raw is None:
            return False
        cur_loc = check_location(location_raw[0])
        cur_dir = w.get_info('direction')
        if cur_loc is None or cur_dir is None:
            return False
        target_angle = calculate_angle(cur_loc, tar_loc)
        if target_angle is None:
            return False
        return execute_view_turn(
            w,
            cur_dir,
            target_angle,
            threshold=threshold,
            max_steps=max_steps,
            wait=self.ALIGN_WAIT if wait is None else wait,
            min_dura=self.ALIGN_MIN_DURA,
            max_dura=self.ALIGN_MAX_DURA,
            max_px=self.ALIGN_MAX_BIAS,
            log_prefix="[Nav]",
        )

    def find_largest_door(self, w):
        """
          0: door
          1: object
          2: window
          3: pick_menu
          4: open_door
        """
        scene = self._get_forward_scene(w)
        if not scene: return None
        doors = []
        for obj in scene:
            try:
                if int(obj[5]) in self.DOOR_CLASS_IDS:
                    doors.append(obj)
            except (IndexError, TypeError, ValueError):
                continue
        if not doors: return None
        return max(doors, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def _align_to_door_detection(self, w, door, tolerance_px=80):
        for _ in range(4):
            inf_w, inf_h = get_wh()
            frame_w = max(inf_w, inf_h)
            scale = self.screen_w / frame_w
            door_center_x = (door[0] + door[2]) / 2
            offset_real = (door_center_x - (frame_w / 2)) * scale
            if abs(offset_real) <= tolerance_px:
                return True

            adjust_val = int(offset_real * 0.33)
            adjust_val = max(-400, min(400, adjust_val))
            w.tap_single('视角', x_bias=adjust_val, dura=500, wait=500)
            w.refresh_frame()
            refreshed = self.find_largest_door(w)
            if refreshed is None:
                return False
            door = refreshed
        return False

    def _advance_towards_entry_door(self, w):
        door = self.find_largest_door(w)
        if door is not None:
            print("[Interact] 前推前重新定位门并修正视角")
            self._align_to_door_detection(w, door)
            w.tap_single('摇杆', y_bias=-320, dura=320, wait=320)
            w.refresh_frame()
            return True

        print("[Interact] 前推时门目标丢失，先给一次前推试错")
        w.tap_single('摇杆', y_bias=-260, dura=260, wait=450)
        w.refresh_frame()
        if w.get_info('开门') or w.get_info('关门') or self.find_largest_door(w):
            return True

        print("[Interact] 试错后仍无门/交互按钮，后退重找门")
        w.tap_single('摇杆', y_bias=300, dura=380, wait=650)
        w.refresh_frame()
        recovered = self.find_largest_door(w)
        if recovered is None:
            return False

        self._align_to_door_detection(w, recovered)
        w.refresh_frame()
        return True

    def start_searching(self, w):
        if self._should_abort(w):
            return False

        self._start_house_search_timer()
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        print("[搜房]入口房间搜集物资。。。")
        self.collect_supplies_in_room(w)
        if self._house_search_timed_out():
            return self._force_exit_after_search_timeout(w)
        if self._should_abort(w):
            self._clear_house_search_timer()
            return False

        self.house_entry_yaw = self.global_yaw
        a_door_abs_yaw = (self.house_entry_yaw + 180) % 360
        self.visited_sub_doors.append((a_door_abs_yaw, 999))
        print("[搜房] 已记录入口A门方向，防止误入")

        door_info = self._find_open_door_in_view(w)
        if not door_info: door_info = self._scan_for_open_door(w, 360)
        if self._house_search_timed_out():
            return self._force_exit_after_search_timeout(w)

        while door_info and self.sub_rooms_entered < 2:
            if self._house_search_timed_out():
                return self._force_exit_after_search_timeout(w)
            if self._should_abort(w):
                self._clear_house_search_timer()
                return False
            rel_ang, bh = door_info
            if self._enter_sub_room_and_collect(w, rel_ang, bh):
                if self._house_search_timed_out():
                    return self._force_exit_after_search_timeout(w)
                self.sub_rooms_entered += 1
                door_info = self._find_open_door_in_view(w)
                if not door_info: door_info = self._scan_for_open_door(w, 360)
            else:
                if self._house_search_timed_out():
                    return self._force_exit_after_search_timeout(w)
                break

        # 4. 退出房屋
        self._clear_house_search_timer()
        self._exit_house(w)
        return not self._should_abort(w)

    def _find_closed_door_in_view(self, w):
        doors = self.new_targets_of_class(w, [0])
        if not doors: return None
        best = max(doors, key=lambda x: x[1])
        return (best[0], best[1])

    def _scan_for_closed_door(self, w, max_rotate=360):
        total = 0
        while total < max_rotate:
            if self._should_stop_house_search(w):
                return None
            self._turn(w, 30)
            total += 30
            time.sleep(0.2)
            res = self._find_closed_door_in_view(w)
            if res: return res
        return None

    def _enter_closed_door(self, w, rel_angle, rush_time=1.0):
        # 对关门贴脸时不需要盲冲(传0)，贴脸后点击开门，待门开后再盲冲
        approached = self._robust_pass_through_door(w, rel_angle, [0], rush_time=0.0)
        if approached:
            if w.get_info('开门'):
                w.click('开门')
                time.sleep(1)
            time.sleep(0.5)
            w.tap_single('摇杆', y_bias=-400, dura=1000)
            w.refresh_frame()
            time.sleep(0.2)
            return True
        return False

    def _exit_house(self, w):

        print("\n>>> 准备退出房屋")
        trusted_exit_route = False

        # 策略1：入口房间关闭门
        print("[出口] 策略1：在入口房间寻找关闭的门")
        closed = self._find_closed_door_in_view(w)
        if not closed: closed = self._scan_for_closed_door(w, 360)
        if closed:
            rel_ang, _ = closed
            print(f"[出口] 发现入口房间关闭门，推开离开！")
            if self._enter_closed_door(w, rel_ang, rush_time=1.2):
                w.refresh_frame()
                if self._get_house_scene(w) != 0:
                    trusted_exit_route = True
                    return

        # 策略2：进子房间找关闭门
        print("[出口] 策略2：入口无关闭门，进入子房间寻找")
        if self._should_abort(w):
            return
        open_door = self._find_open_door_in_view(w)
        if not open_door: open_door = self._scan_for_open_door(w, 360)

        if open_door:
            rel_ang, bh = open_door
            print(f"[出口] 进子房间找关闭门")
            self._pass_through_open_door(w, rel_ang, rush_time=0.8)
            self.room_yaw = 0.0

            closed_in_sub = self._find_closed_door_in_view(w)
            if not closed_in_sub: closed_in_sub = self._scan_for_closed_door(w, 360)

            if closed_in_sub:
                c_rel_ang, _ = closed_in_sub
                print(f"[出口] 发现子房间关闭门，推开离开！")
                if self._enter_closed_door(w, c_rel_ang, rush_time=1.2):
                    w.refresh_frame()
                    if self._get_house_scene(w) != 0:
                        trusted_exit_route = True
                        return

            # 子房间没找到出口，退回入口房间
            print("[出口] 子房间无关闭门，扇区快搜退回入口房间")
            exit_door = self._find_open_door_in_view(w, ignore_visited=True)
            if not exit_door: exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)
            if exit_door: self._pass_through_open_door(w, exit_door[0], rush_time=0.8)

        # 策略3：从入口A门原路返回
        print("[出口] 从入口A门原路返回")
        if self._should_abort(w):
            return
        a_door = self._find_open_door_in_view(w, ignore_visited=True)
        if not a_door: a_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if a_door:
            print("[出口] 发现A门，穿过离开！")
            self._pass_through_open_door(w, a_door[0], rush_time=1.2)
            trusted_exit_route = True
        else:
            print("[出口] 极端情况：找不到A门，执行无门窗逃逸")
            self.house_exit_manager.reset()
            if self.house_exit_manager._escape_after_failed_exit_scan(w):
                return

        # 策略4：所有策略均失败，启动HouseExitManager兜底
        w.refresh_frame()
        if self._get_house_scene(w) == 0:
            print("[出口] 策略3后仍在屋内，启动HouseExitManager兜底出房")
            self.house_exit_manager.reset()
            for _ in range(30):
                if self._should_abort(w):
                    return
                if self.house_exit_manager.process(w):
                    print("[出口] 兜底出房成功")
                    return
            print("[出口] 兜底出房也失败，强制前进冲出")
            for _ in range(5):
                w.tap_single('摇杆', y_bias=-500, dura=300)
                w.refresh_frame()
                time.sleep(0.3)
            if self._get_house_scene(w) != 0:
                self.house_exit_manager.reset()
                self.house_exit_manager.process(w)
        elif not trusted_exit_route:
            print("[出口] 未经过明确门窗动作但已到屋外，执行二次确认")
            self.house_exit_manager.reset()
            self.house_exit_manager.process(w)

    def _calc_abs_angle(self, rel_ang):

        return (self.global_yaw + rel_ang) % 360

    def _robust_pass_through_door(self, w, rel_angle, target_classes=None, rush_time=1.0):

        if target_classes is None:
            target_classes = [4]
        self._visual_align(w, rel_angle, target_classes)
        inf_w, inf_h = get_wh()
        frame_w = max(inf_w, inf_h)
        center_x = frame_w / 2

        for _ in range(30):
            if self._should_stop_house_search(w):
                return False
            doors = self.new_targets_of_class(w, target_classes)
            if not doors:
                print("  [搜房] 警告：未检测到门，尝试盲冲补救")
                break

            best = max(doors, key=lambda x: x[1])
            rel_ang, bh, _, det = best
            cx = (det[0] + det[2]) / 2
            offset_px = cx - center_x

            inf_w, inf_h = get_wh()
            frame_h = min(inf_w, inf_h)

            # 贴脸判定
            if bh > frame_h * 0.6:
                print(f"  [搜房] 已贴脸门框(高度比:{bh / frame_h:.2f})，准备盲冲穿过！")
                break

            if abs(offset_px) > 5:
                self._turn(w, self.pixel_to_angle(cx) * 0.6)
                time.sleep(0.05)
                continue

            # 轨迹笔直，允许前进
            w.tap_single('摇杆', y_bias=-400, dura=300)
            w.refresh_frame()
            time.sleep(0.2)

        print(f"  [鲁棒穿门] 执行盲冲，时间: {rush_time}s")
        move_ms = max(0, int(float(rush_time) * 1000))
        if move_ms > 0:
            w.tap_single('摇杆', y_bias=-500, dura=move_ms)
            w.refresh_frame()
            time.sleep(0.2)
        return True

    def _pass_through_open_door(self, w, rel_angle, rush_time=1.0):
        return self._robust_pass_through_door(w, rel_angle, [4], rush_time)

    def _enter_sub_room_and_collect(self, w, rel_angle, box_h):
        """子房间完整交互流程：记录特征 -> 鲁棒穿门 -> 战术搜物资 -> 扇区回搜退门"""
        print("\n[子房间] 进入...")
        if self._should_stop_house_search(w):
            return False
        # 1. 记录进门绝对特征并去重
        abs_ang_enter = self._calc_abs_angle(rel_angle)
        self.visited_sub_doors.append((abs_ang_enter, box_h))

        # 2. 记录进门前的全局朝向，用于退出时计算反向扇区
        enter_yaw = self.global_yaw

        # 3. 穿门进入
        if not self._pass_through_open_door(w, rel_angle, rush_time=1.0):
            print("[错误] 进入失败")
            return False

        self.room_yaw = 0.0  # 重置局部坐标系
        # 4. 搜集物资（内部自带战术复位）
        self._search_supplies(w)
        if self._should_stop_house_search(w):
            return False

        # 5. 扇区快搜退出门
        print("[子房间] 搜集完毕，扇区快搜退出门...")
        target_exit_yaw = (enter_yaw + 180) % 360  # 计算进门背后的朝向
        # ignore_visited必须为True！因为进来的门已被标记，不忽略会看不到它
        exit_door = self._sector_scan_for_open_door(w, target_exit_yaw, sector_angle=120, ignore_visited=True)

        # 扇区兜底：如果扇区没找到，进行360全图扫描
        if not exit_door:
            print("[子房间] 未找到，360度兜底扫描...")
            exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if exit_door:
            rel_exit, _ = exit_door
            print(f"[子房间] 发现退出门，退出...")
            self._pass_through_open_door(w, rel_exit, rush_time=0.8)

            # 退回入口房间后，更新该门的特征以防重复进入
            time.sleep(0.2)
            doors = self.new_targets_of_class(w, [4])
            if doors:
                best = max(doors, key=lambda x: x[1])
                back_abs = self._calc_abs_angle(best[0])
                if not self._is_door_visited(w, back_abs, best[1]):
                    self.visited_sub_doors.append((back_abs, best[1]))
            return True

        print("[错误] 找不到退出门")
        return False

    def _sector_scan_for_open_door(self, w, center_yaw, sector_angle=120, ignore_visited=True):

        print(f"  [搜房] 中心朝向:{center_yaw:.0f}°, 扫描范围:{sector_angle}°")
        if self._should_stop_house_search(w):
            return None

        # 计算并转向目标中心朝向（处理最短路径旋转）
        delta = center_yaw - self.global_yaw
        if delta > 180: delta -= 360
        if delta < -180: delta += 360
        self._turn(w, delta)
        time.sleep(0.2)

        # 1. 检查中心点
        res = self._find_open_door_in_view(w, ignore_visited)
        if res: return res

        # 2. 左右扇区扫描
        half_sector = sector_angle // 2
        steps = half_sector // 30

        for i in range(1, steps + 1):  # 向左扫
            if self._should_stop_house_search(w):
                return None
            self._turn(w, 30)
            time.sleep(0.1)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res

        self._turn(w, - (half_sector))  # 瞬间归位中心
        time.sleep(0.2)
        for i in range(1, steps + 1):  # 向右扫
            if self._should_stop_house_search(w):
                return None
            self._turn(w, -30)
            time.sleep(0.1)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res

        return None

    def _scan_for_open_door(self, w, max_rotate=360, ignore_visited=False):

        total = 0
        while total < max_rotate:
            if self._should_stop_house_search(w):
                return None
            self._turn(w, 30)
            total += 30
            time.sleep(0.2)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res
        return None

    def _find_open_door_in_view(self, w, ignore_visited=False):

        doors = self.new_targets_of_class(w, [4])
        if not doors: return None
        doors.sort(key=lambda x: x[1], reverse=True)  # 框高越大越近，优先进入最近的门
        for rel_ang, bh, _, _ in doors:
            abs_ang = self._calc_abs_angle(rel_ang)
            if not ignore_visited and self._is_door_visited(w, abs_ang, bh):
                continue
            return (rel_ang, bh)
        return None

    def _is_door_visited(self, w, abs_ang, bh):

        for v_ang, v_bh in self.visited_sub_doors:
            angle_diff = abs(abs_ang - v_ang)
            angle_diff = min(angle_diff, 360 - angle_diff)  # 处理圆周折返
            if angle_diff < 20 and abs(bh - v_bh) < 50:  # 角度容差20度，框高容差50像素
                return True
        return False

    def collect_supplies_in_room(self, w):

        collected = []  # 已拾取的 (abs_angle, box_h)
        player_yaw = 0.0

        def calc_abs(rel_angle, box_h):
            return ((player_yaw + rel_angle) % 360, box_h)

        def is_duplicate(abs_ang, box_h):
            for a, h in collected:
                angle_diff = abs((abs_ang - a + 180) % 360 - 180)
                if angle_diff < 8 and abs(box_h - h) < 25:
                    return True
            return False

        def pickup_one_in_current_view(w):
            """在当前画面拾取一个未拾取过的物资，成功返回 True，否则 False"""
            if self._should_stop_house_search(w):
                return False
            # 获取当前画面所有物资，按面积取最近（最大）的一个
            scene = self._get_forward_scene(w)
            supplies = [obj for obj in scene if int(obj[5]) in [1]]

            if not supplies:
                return False
            # 选择面积最大的
            best = max(supplies, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            cx = (best[0] + best[2]) / 2
            rel_ang = self.pixel_to_angle(cx)
            box_h = best[3] - best[1]
            abs_ang = (player_yaw + rel_ang) % 360

            if is_duplicate(abs_ang, box_h):
                return False

            # 执行对准和拾取
            print(f"  发现物资（绝对{abs_ang:.1f}° 框高{box_h}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, box_h))
                return True
            return False

        # ---------- 方向序列 ----------
        print("======[搜资] 检查初始方向 (0°)，在刚进入房屋的视角下检查是否有物资，有则搜集======")
        for _ in range(self.PICKUP_MAX_PER_DIRECTION):
            if self._should_stop_house_search(w) or not pickup_one_in_current_view(w):
                break
            time.sleep(0.2)
        if self._should_stop_house_search(w):
            return len(collected)

        print("======[搜资] 左转45°检查是否有物资，有则收集======")
        self.turn_by_angle(w, -45, 300)
        player_yaw = (player_yaw - 45) % 360
        time.sleep(0.3)
        for _ in range(self.PICKUP_MAX_PER_DIRECTION):
            if self._should_stop_house_search(w) or not pickup_one_in_current_view(w):
                break
            time.sleep(0.2)
        if self._should_stop_house_search(w):
            return len(collected)

        print("======[搜资] 左转45°后回正，右转45度检查是否有物资，有则收集======")
        self.turn_by_angle(w, 45, 300)  # 回到 0°
        player_yaw = (player_yaw + 45) % 360
        time.sleep(0.3)
        self.turn_by_angle(w, 45, 300)  # 右转 45°
        player_yaw = (player_yaw + 45) % 360
        time.sleep(0.3)
        for _ in range(self.PICKUP_MAX_PER_DIRECTION):
            if self._should_stop_house_search(w) or not pickup_one_in_current_view(w):
                break
            time.sleep(0.2)

        print(f"[搜资] 结束，共拾取 {len(collected)} 个物资")
        self.turn_by_angle(w, -45, 300)
        print("========回正方向==============")
        return len(collected)

    def approach_and_pickup(self, w, initial_bbox, target_class, rel_ang):
        """
        小步靠近物资，并拾取
        返回是否成功拾取。
        """
        if self._should_stop_house_search(w):
            return False

        if abs(rel_ang) > 2:
            self.turn_by_angle(w, rel_ang, 200)
            time.sleep(1)

        for i in range(30):
            if self._should_stop_house_search(w):
                return False
            w.refresh_frame()
            scene = self._get_forward_scene(w)
            pick_menu = [obj for obj in scene if int(obj[5]) in [3]]

            print("当前是否有物资提示信息{}".format(pick_menu))
            if pick_menu:
                print("检查到附近有物资")
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                time.sleep(1)
                # 关闭附近弹窗，不影响继续旋转角度查找物资点
                if w.get_info("关闭附近"):
                    print("检测到关闭附近按钮。。。")
                    w.click(w.get_info("关闭附近"))
                    time.sleep(0.5)
                    w.refresh_frame()
                i = 30
                return True
            # 走到物资点后，检测到
            if w.get_info("关闭附近"):
                print("检查到附近有物资")
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                time.sleep(1)
                if w.get_info("关闭附近"):
                    print("检测到关闭附近按钮。。。")
                    w.click(w.get_info("关闭附近"))
                    time.sleep(0.5)
                    w.refresh_frame()
                i = 30
                return True

            else:
                print("======识别到物资后，视角对准，往前靠近{}步，最大移动距离30步======".format(i + 1))
                w.tap_single('摇杆', y_bias=-20, dura=300)
                time.sleep(0.5)
                w.refresh_frame()
                i += 1

            time.sleep(1)
        print("当前已移动完成30步或者已经拾取完物资")
        return False

    def pixel_to_angle(self, cx):
        inf_w, inf_h = get_wh()
        frame_w = max(inf_w, inf_h)
        center = frame_w / 2
        if frame_w <= 0: return 0.0
        return (cx - center) / center * (80 / 2)

    def turn_by_angle(self, w, delta_angle, duration_ms=200):
        try:
            delta_angle = float(delta_angle)
        except (TypeError, ValueError):
            return
        if abs(delta_angle) < 1.0:
            return
        before_dir = w.get_info('direction')
        if before_dir is None:
            return
        target_dir = (float(before_dir) + delta_angle) % 360.0
        execute_view_turn(
            w,
            before_dir,
            target_dir,
            threshold=1,
            max_steps=1,
            wait=20,
            fallback_dura=800,
            log_prefix="[SearchTurn]",
        )

    def targets_of_class(self, w, target_class=None):
        if target_class is None:
            target_class = [4]
        scene = self._get_forward_scene(w)
        dets = [obj for obj in scene if int(obj[5]) in target_class]
        infos = []
        for d in dets:
            if d[5] in [0, 1, 2, 3, 4]:
                cx = (d[0] + d[2]) / 2
                bh = d[3] - d[1]
                angle = self.pixel_to_angle(cx)
                area = (d[2] - d[0]) * (d[3] - d[1])
                infos.append((angle, bh, d, area))
        return infos

    def new_targets_of_class(self, w, target_class=None):
        if target_class is None:
            target_class = [4]
        scene = self._get_forward_scene(w)
        dets = [obj for obj in scene if int(obj[5]) in target_class]
        infos = []
        for d in dets:
            if d[5] in [0, 1, 2, 3, 4]:
                cx = (d[0] + d[2]) / 2
                bh = d[3] - d[1]
                angle = self.pixel_to_angle(cx)
                area = (d[2] - d[0]) * (d[3] - d[1])
                infos.append((angle, bh, d[5], d))
        return infos

    def _approach_door(self, w, rel_ang, is_sub_room=False):
        """
        从 initial_bbox 开始，视觉对准 + 小步靠近 + 拾取。
        返回是否成功拾取。
        """
        print("出子房间的门之前的角度{}".format(rel_ang))
        last_door = []
        # 调整角度
        if abs(rel_ang) > 2:

            if rel_ang > 0:
                print("向右滑动调整视角，角度有偏差，添加5度的偏差")
                rel_ang += 5
            else:
                print("角度微微调整")
                rel_ang += 6
            print("出子房间的门的进行调整的角度{}".format(rel_ang))
            if abs(rel_ang) > 45:

                count = int(abs(rel_ang) / 45)
                count_ang = abs(rel_ang) % 45
                print("角度大于45度，拆分成多次来旋转，拆分成{}次，是否有多余的{}".format(count, count_ang))
                for i in range(count):
                    if rel_ang > 0:
                        self.turn_by_angle(w, 45, 200)
                    else:
                        self.turn_by_angle(w, -45, 200)

            else:
                self.turn_by_angle(w, rel_ang, 200)

            time.sleep(1)

        # 调整角度结束后，往前移动靠近
        for i in range(30):
            if self._should_abort(w):
                return False
            w.tap_single('摇杆', y_bias=-20, dura=300)
            i += 1
            w.refresh_frame()
            time.sleep(1)

            scene = self._get_forward_scene(w)
            open_door1 = [obj for obj in scene if int(obj[5]) in [4]]

            if open_door1:
                last_door = max(open_door1, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                # 当前还在画面中可以检测到
                inf_w, inf_h = get_wh()
                frame_w = max(inf_w, inf_h)
                center = frame_w / 2
                print("向门靠近并移动的过程中门的信息{}，门的中心点位置{},屏幕的中心点位置{}".format(open_door1, (
                        open_door1[0][2] - open_door1[0][0]) / 2, center))
                # 移动靠近的过程中y1会逐渐减小，小于等于10 的时候，人物靠近门，这个时候暂停移动

            else:
                # 检测不到当前视角中的门的时候，当前已经靠近门边，直接往前走，可能会出现擦着墙边过的情况
                print("当前已经靠近房间的门,微调角度处理。。。。")
                door = last_door
                if door:
                    inf_w, inf_h = get_wh()
                    frame_w = max(inf_w, inf_h)
                    scale = self.screen_w / frame_w
                    door_center_x = (door[0] + door[2]) / 2
                    offset_real = (door_center_x - (frame_w / 2)) * scale

                    adjust_val = int(offset_real * 0.33)
                    adjust_val = max(-400, min(400, adjust_val))
                    print("当前微调视角，水平滑动{}".format(adjust_val))
                    w.tap_single('视角', x_bias=int(adjust_val), dura=500, wait=500)
                    w.refresh_frame()
                    time.sleep(0.5)
                    time.sleep(5)

                if w.get_info('开门'):
                    w.click('开门')
                    time.sleep(1)
                print("靠近门后，微调结束，直走进入房间。。。")
                w.tap_single('摇杆', y_bias=-400, dura=300)
                w.refresh_frame()
                w.tap_single('摇杆', y_bias=-400, dura=300)
                w.refresh_frame()
                print("靠近门后往前移动俩步结束，不在往前移动")
                return True
        time.sleep(1)
        print("当前已移动完成30步")
        return False

    def _collect_in_direction(self, w, avoid_door_abs=None):
        collected = []
        if self._should_stop_house_search(w):
            return
        supplies = self.new_targets_of_class(w, target_class=[1])
        print("子房间查找物资的信息{}".format(supplies))
        print("子房间查找物资的信息{}".format(supplies))

        if supplies:

            # 选择面积最大的
            best = max(supplies, key=lambda d: d[1])
            rel_ang = best[0]
            abs_ang = (self.room_yaw + rel_ang) % 360

            print(f"  发现物资（绝对{abs_ang:.1f}° 框高{best[1]}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, best[1]))
        else:
            print("当前子房间内未找到物资信息,继续下一次视角中获取物资...")
            time.sleep(1)

        if len(collected) == 2:
            print("当前物资已拾满")

    def _search_supplies(self, w, avoid_door_abs=None):
        print("[物资] 方向扫描...")
        if self._should_stop_house_search(w):
            return
        self._collect_in_direction(w, avoid_door_abs)  # 正前
        if self._should_stop_house_search(w):
            return
        self._turn(w, -45)
        if self._should_stop_house_search(w):
            return
        self._collect_in_direction(w, avoid_door_abs)  # 左45°
        if self._should_stop_house_search(w):
            return
        self._turn(w, 45)
        time.sleep(5)
        if self._should_stop_house_search(w):
            return
        self._turn(w, 45)
        if self._should_stop_house_search(w):
            return
        self._collect_in_direction(w, avoid_door_abs)  # 右45°
        if self._should_stop_house_search(w):
            return
        self._turn(w, -45)  # 回正

    def _visual_align(self, w, target_angle, target_class=None):
        print("开始调整。。。{}".format(target_angle))
        for _ in range(6):
            if self._should_stop_house_search(w):
                return
            if abs(target_angle) <= 1.5:
                return
            step = max(-30, min(30, target_angle))
            self._turn(w, step)
            time.sleep(0.15)
            targets = self.new_targets_of_class(w, target_class=target_class)
            if not targets:
                print("  [对准] 目标丢失")
                return
            best = max(targets, key=lambda x: x[1])
            target_angle = best[0]

    def _turn(self, w, delta):
        self.turn_by_angle(w, delta)
        self.room_yaw = (self.room_yaw + delta) % 360
        self.global_yaw = (self.global_yaw + delta) % 360
