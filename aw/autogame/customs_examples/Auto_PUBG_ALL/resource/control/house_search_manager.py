import json
import os
import time
from typing import TYPE_CHECKING, Optional
import cv2
import numpy as np
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import (
    MapNavigator,
    smooth_path_by_line_of_sight,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager import HouseExitManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.timing import TimeoutTracker
from aw.autogame.tools.Utils import *

if TYPE_CHECKING:
    # 假设你的框架类定义在 framework.py 文件中
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class HouseSearchManager:
    VISUAL_APPROACH_MAX_ATTEMPTS = 12
    UNSTUCK_MAX_CYCLES = 6
    UNSTUCK_FORWARD_STEPS = 5
    UNSTUCK_SAME_POINT_RADIUS = 2.0
    UNSTUCK_BACK_Y_BIAS = 300
    UNSTUCK_SIDE_X_BIAS = 300
    UNSTUCK_FORWARD_Y_BIAS = -300
    UNSTUCK_STEP_DURA = 300
    UNSTUCK_STEP_WAIT = 1500
    UNSTUCK_FORWARD_WAIT = 2000
    UNSTUCK_LARGE_VIEW_X_BIAS = 720
    UNSTUCK_LARGE_VIEW_DURA = 850
    UNSTUCK_LARGE_VIEW_WAIT = 500
    UNSTUCK_LARGE_AUTO_FORWARD_SECONDS = 3.0
    UNSTUCK_LARGE_SIDE_WAIT = 3000
    PICKUP_MAX_PER_DIRECTION = 3
    INITIAL_LOCATION_MIN_SAMPLES = 3
    INITIAL_LOCATION_MAX_SAMPLES = 6
    INITIAL_LOCATION_STABLE_DISTANCE = 2.5
    INITIAL_LOCATION_JUMP_RESET_DISTANCE = 8.0
    LANDING_LOCATION_PROBE_Y_BIAS = -120
    LANDING_LOCATION_PROBE_DURA = 180
    LANDING_LOCATION_PROBE_WAIT = 360
    LANDING_LOCATION_STABLE_MAX_JUMP_DISTANCE = 100.0
    VALID_FRAME_LOCATION_RECOVERY_Y_BIAS = -120
    VALID_FRAME_LOCATION_RECOVERY_WAIT = 300
    VALID_FRAME_DIRECTION_RECOVERY_X_BIAS = 180
    VALID_FRAME_DIRECTION_RECOVERY_DURA = 180
    VALID_FRAME_DIRECTION_RECOVERY_WAIT = 300
    ENTRY_AUTO_FORWARD_DISTANCE = 30.0
    ENTRY_COARSE_MOVE_DISTANCE = 15.0
    ENTRY_ARRIVAL_DISTANCE = 1.0
    ENTRY_COARSE_Y_BIAS = -430
    ENTRY_COARSE_DURA = 1300
    ENTRY_FINE_Y_BIAS = -220
    ENTRY_FINE_DURA = 480
    JUMP_FORWARD_SETTLE_SECONDS = 0.5
    JUMP_FORWARD_Y_BIAS = -180
    JUMP_FORWARD_DURA = 160
    JUMP_FORWARD_WAIT = 320
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
    ROUTE_STUCK_REPEAT_RADIUS = 2.0
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
    STUCK_HISTORY_LEN = 7
    ROUTE_STUCK_HOUSE_BYPASS_TURN_DEGREES = 60
    ROUTE_STUCK_HOUSE_BYPASS_VIEW_X_BIAS = 360
    ROUTE_STUCK_HOUSE_BYPASS_VIEW_DURA = 300
    ROUTE_STUCK_HOUSE_BYPASS_VIEW_WAIT = 500
    ROUTE_STUCK_HOUSE_BYPASS_FORWARD_Y_BIAS = -520
    ROUTE_STUCK_HOUSE_BYPASS_FORWARD_DURA = 650
    ROUTE_STUCK_HOUSE_BYPASS_FORWARD_WAIT = 1000
    HOUSE_SEARCH_TIMEOUT_SECONDS = 60
    ENTRY_NEAR_MICRO_ADJUST_DISTANCE = 2.5
    ENTRY_NEAR_MICRO_DONE_DISTANCE = 0.25
    ENTRY_NEAR_MICRO_MAX_ATTEMPTS = 8
    ENTRY_NEAR_MICRO_X_BIAS = 120
    ENTRY_NEAR_MICRO_Y_BIAS = 120
    ENTRY_NEAR_MICRO_DURA = 160
    ENTRY_NEAR_MICRO_WAIT = 320
    ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS = 4
    ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX = 55
    ENTRY_DOOR_VIEW_ADJUST_REFRESH_SETTLE_SECONDS = 0.2
    ENTRY_DOOR_ALIGN_CENTER_THRESHOLD = 80
    ENTRY_DOOR_ALIGN_CLOSE_CENTER_THRESHOLD = 140
    ENTRY_DOOR_ALIGN_NEAR_CENTER_THRESHOLD = 220
    ENTRY_DOOR_ALIGN_VERY_NEAR_CENTER_THRESHOLD = 300
    ENTRY_DOOR_ALIGN_STEP_RATIO = 0.33
    ENTRY_DOOR_ALIGN_MAX_BIAS = 400
    ENTRY_DOOR_ALIGN_DURA = 500
    ENTRY_DOOR_ALIGN_WAIT = 500
    ENTRY_DOOR_ALIGN_CLOSE_AREA_RATIO = 0.030
    ENTRY_DOOR_ALIGN_NEAR_AREA_RATIO = 0.055
    ENTRY_DOOR_ALIGN_VERY_NEAR_AREA_RATIO = 0.090
    ENTRY_DOOR_DIRECT_CENTER_MIN_RATIO = 0.40
    ENTRY_DOOR_DIRECT_CENTER_MAX_RATIO = 0.60
    ENTRY_DOOR_EDGE_LATERAL_LEFT_RATIO = 0.25
    ENTRY_DOOR_EDGE_LATERAL_RIGHT_RATIO = 0.75
    ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS = -200
    ENTRY_DOOR_DIRECT_FORWARD_DURA = 200
    ENTRY_DOOR_DIRECT_FORWARD_WAIT = 3000
    ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS = 200
    ENTRY_DOOR_DIRECT_BACKOFF_DURA = 200
    ENTRY_DOOR_DIRECT_BACKOFF_WAIT = 3000
    ENTRY_DOOR_MISSING_BACKOFF_Y_BIAS = 320
    ENTRY_DOOR_MISSING_BACKOFF_DURA = 300
    ENTRY_DOOR_MISSING_BACKOFF_WAIT = 520
    ENTRY_DOOR_MISSING_STRICT_FORWARD_Y_BIAS = -420
    ENTRY_DOOR_MISSING_STRICT_FORWARD_DURA = 650
    ENTRY_DOOR_MISSING_STRICT_FORWARD_WAIT = 850
    ENTRY_DOOR_MISSING_WALL_BACKOFF_Y_BIAS = 200
    ENTRY_DOOR_MISSING_WALL_BACKOFF_DURA = 200
    ENTRY_DOOR_MISSING_WALL_BACKOFF_WAIT = 520
    ENTRY_NEAR_WALL_SIDE_ESCAPE_X_BIAS = 120
    ENTRY_NEAR_WALL_SIDE_ESCAPE_DURA = 160
    ENTRY_NEAR_WALL_SIDE_ESCAPE_WAIT = 320
    SWEEP_STEP_MS = 100
    BUTTON_SWEEP_MAX_STEPS = 16
    BUTTON_SWEEP_X_BIAS = 240
    BUTTON_SWEEP_WAIT_PAD = 220
    ENTRY_DOOR_JUMP_FORWARD_Y_BIAS = -360
    ENTRY_DOOR_JUMP_FORWARD_DURA = 320
    ENTRY_DOOR_JUMP_FORWARD_WAIT = 720
    ENTRY_DOOR_ALIGNED_PUSH_MAX_ATTEMPTS = 3
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
    ALIGN_MAX_BIAS = 800
    ALIGN_MIN_DURA = 180
    ALIGN_MAX_DURA = 1000
    ALIGN_WAIT = 30
    OUTDOOR_NEAR_WALL_ENTRY_EXEMPT_DISTANCE = 3.0
    OUTDOOR_NEAR_WALL_RECOVERY_MAX_CYCLES = 2
    OUTDOOR_NEAR_WALL_BACK_Y_BIAS = 300
    OUTDOOR_NEAR_WALL_BACK_DURA = 360
    OUTDOOR_NEAR_WALL_BACK_WAIT = 600
    OUTDOOR_NEAR_WALL_LEFT_VIEW_X_BIAS = -800
    OUTDOOR_NEAR_WALL_LEFT_VIEW_DURA = 420
    OUTDOOR_NEAR_WALL_LEFT_VIEW_WAIT = 30
    OUTDOOR_NEAR_WALL_FORWARD_Y_BIAS = -360
    OUTDOOR_NEAR_WALL_FORWARD_DURA = 2000
    OUTDOOR_NEAR_WALL_FORWARD_WAIT = 2000
    OUTDOOR_NEAR_WALL_CLEAR_FORWARD_DURA = 1000
    OUTDOOR_NEAR_WALL_CLEAR_FORWARD_WAIT = 1000

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
        self.temp_skip_entries = set()

        # --- 卡顿检测相关变量 ---
        self.history_locations = []
        self.max_history_len = self.STUCK_HISTORY_LEN  # 连续7帧坐标完全相同才判定卡死
        self.stuck_threshold = 0.5  # 脱困/绕障后的位移复核阈值

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
        self.direction_missing_frames = 0
        self.last_valid_location = None
        self.initial_location_samples = []
        self.landing_location_confirmed = False
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self.entry_near_micro_adjust_attempts = 0
        self.entry_door_last_area_ratio = None
        self._entry_door_force_strict_align_once = False
        self._jump_forward_guard = False
        self._jump_forward_wait_until_hidden = False

    def _entry_location_tuple(self, entry):
        try:
            location = entry.get('location')
            return (int(location[0]), int(location[1]))
        except (TypeError, ValueError, IndexError, AttributeError):
            return None

    def _is_excluded_entry(self, entry) -> bool:
        return self._entry_location_tuple(entry) in self.EXCLUDED_ENTRY_LOCATIONS

    def _is_temp_skipped_entry(self, entry) -> bool:
        return self._entry_location_tuple(entry) in self.temp_skip_entries

    def _build_excluded_house_ids(self):
        excluded = set()
        for house_id, entries in self.house_data.items():
            if any(self._is_excluded_entry(entry) for entry in entries):
                excluded.add(house_id)
        if excluded:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Searching] 已过滤指定进门点 {sorted(self.EXCLUDED_ENTRY_LOCATIONS)} 对应房屋 {sorted(excluded)}')
        return excluded

    def _is_excluded_house(self, house_id) -> bool:
        return house_id in self.excluded_house_ids

    def _mark_current_entry_failed(self, reason: str):
        entry_loc = self._entry_location_tuple(self.active_entry) if self.active_entry else None
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[EntryPoint] {reason}，临时舍弃当前入门点 house={self.current_house_id}, entry={entry_loc}；同一房子的其他入门点继续保留')
        if entry_loc is not None:
            self.temp_skip_entries.add(entry_loc)
        self.current_house_id = None
        self.active_entry = None
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self._reset_entry_near_micro_adjust()
        self._reset_route_stuck_bypass()

    def reset(self):
        self.completed_houses = set()
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.first_view = False
        self.auto_forward = False
        self.temp_skip_houses = set()
        self.temp_skip_entries = set()
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
        self.direction_missing_frames = 0
        self.last_valid_location = None
        self.initial_location_samples = []
        self.landing_location_confirmed = False
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self.entry_near_micro_adjust_attempts = 0
        self._jump_forward_guard = False
        self._jump_forward_wait_until_hidden = False



    def _house_scene_label(self, scene) -> str:
        labels = {
            self.HOUSE_INDOOR: "indoor",
            self.HOUSE_OUTDOOR: "outdoor",
            self.HOUSE_ROOFTOP: "rooftop",
            self.HOUSE_NEAR_DOOR: "near_door",
            self.HOUSE_NEAR_WALL: "near_wall",
        }
        return labels.get(scene, f"unknown({scene})")



    def process(self, w: 'FrameWorker'):
        self._frame_worker = w
        if self._should_abort(w):
            return

        confirmed_landing_location = None

        # 0. 基础设置：落地后先用当前位置做一次前探刷新，确认坐标稳定后再切第一人称。
        if not self.first_view:
            confirmed_landing_location = self._confirm_landing_location_before_first_view(w)
            if confirmed_landing_location is None:
                return

        location_raw = (
            [confirmed_landing_location]
            if confirmed_landing_location is not None
            else w.get_info('location')
        )
        if location_raw is None:
            self.location_missing_frames += 1
            w.frame_log('位置值是None，等待位置刷新...')
            w.frame_log("搜房观察：当前位置为空，所以轻推摇杆前探并刷新画面，下一帧重新读取坐标")
            self._reset_stuck_history_for_frame_wait("搜房阶段当前位置缺失")
            w.tap_single('摇杆', y_bias=self.VALID_FRAME_LOCATION_RECOVERY_Y_BIAS, wait=self.VALID_FRAME_LOCATION_RECOVERY_WAIT)
            self._refresh_frame_and_handle_jump(w)
            return
        location = self._remember_valid_location(location_raw)
        direction = w.get_info('direction')

        if location is None:
            self.location_missing_frames += 1
            w.frame_log('位置值无效，等待位置刷新...')
            w.frame_log(f"搜房观察：当前位置值无效 raw={location_raw}，所以轻推摇杆前探并刷新坐标")
            self._reset_stuck_history_for_frame_wait("搜房阶段位置值无效")
            w.tap_single('摇杆', y_bias=self.VALID_FRAME_LOCATION_RECOVERY_Y_BIAS, wait=self.VALID_FRAME_LOCATION_RECOVERY_WAIT)
            self._refresh_frame_and_handle_jump(w)
            return

        self.location_missing_frames = 0
        try:
            float(direction)
        except (TypeError, ValueError):
            self.direction_missing_frames = getattr(self, "direction_missing_frames", 0) + 1
            x_bias = self.VALID_FRAME_DIRECTION_RECOVERY_X_BIAS
            if self.direction_missing_frames % 2 == 0:
                x_bias = -x_bias
            w.frame_log("方向值无效，轻微滑动视角刷新方向...")
            w.frame_log(f"搜房观察：当前位置有效但方向值无效 direction={direction}，轻滑视角后下一帧重读")
            self._reset_stuck_history_for_frame_wait("搜房阶段方向值无效")
            w.tap_single(
                '视角',
                x_bias=x_bias,
                dura=self.VALID_FRAME_DIRECTION_RECOVERY_DURA,
                wait=self.VALID_FRAME_DIRECTION_RECOVERY_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            return

        self.direction_missing_frames = 0
        w.frame_log('[Action] 进入搜房决策逻辑')
        self.searching_logic(w, location, direction)

    def _confirm_landing_location_before_first_view(self, w: 'FrameWorker'):
        first_loc = self._remember_valid_location(w.get_info('location'))
        if first_loc is None:
            self.location_missing_frames += 1
            w.frame_log('落地首帧位置值无效，先刷新等待位置稳定...')
            w.frame_log("搜房观察：落地首帧当前位置为空，所以先刷新画面；连续缺失时轻推摇杆刷新小地图坐标")
            self._reset_stuck_history_for_frame_wait("落地首帧坐标缺失")
            self._refresh_frame_and_handle_jump(w, "落地首帧坐标缺失")
            if self.location_missing_frames >= 3:
                w.tap_single('摇杆', y_bias=-120, wait=300)
            return None

        self.location_missing_frames = 0
        w.frame_log(f"[Searching] 落地首帧当前位置={first_loc}，轻推摇杆后刷新第二帧确认坐标稳定")
        w.tap_single(
            '摇杆',
            y_bias=self.LANDING_LOCATION_PROBE_Y_BIAS,
            dura=self.LANDING_LOCATION_PROBE_DURA,
            wait=self.LANDING_LOCATION_PROBE_WAIT,
        )
        self._refresh_frame_and_handle_jump(w, "落地首帧前推后确认坐标")

        second_loc = self._remember_valid_location(w.get_info('location'))
        if second_loc is None:
            w.frame_log("[Searching] 落地前推后第二帧位置仍无效，等待下一帧重新确认")
            self._reset_stuck_history_for_frame_wait("落地前推后第二帧位置无效")
            return None

        location_delta = get_distance(first_loc, second_loc)
        if location_delta > self.LANDING_LOCATION_STABLE_MAX_JUMP_DISTANCE:
            w.frame_log(
                f"[Searching] 落地前推后坐标跳变 {location_delta:.2f} > "
                f"{self.LANDING_LOCATION_STABLE_MAX_JUMP_DISTANCE:.1f}，暂不切人称或选入门点"
            )
            self.initial_location_samples = [second_loc]
            self._reset_stuck_history_for_frame_wait("落地前推后坐标跳变过大")
            return None

        self.initial_location_samples = [second_loc]
        self.landing_location_confirmed = True
        w.frame_log('[Action] 点击人称并进入搜房判断')
        w.click('人称')
        self.first_view = True
        return second_loc

    def _should_abort(self, w: 'FrameWorker'):
        callback = getattr(self, "abort_callback", None)
        if callback is None:
            return False
        try:
            return bool(callback(w))
        except Exception as exc:
            w.frame_log(f"[Searching] 中断检查失败: {exc}")
            return False

    def _can_finish_searching(self, w: 'FrameWorker'):
        callback = getattr(self, "can_finish_callback", None)
        if callback is None:
            return True
        try:
            return bool(callback(w))
        except Exception as exc:
            w.frame_log(f"[Searching] 结束条件检查失败: {exc}")
            return False

    def _finish_searching_phase(self, w: 'FrameWorker', reason: str):
        callback = getattr(self, "finish_callback", None)
        if callback is not None:
            try:
                return bool(callback(w, reason))
            except Exception as exc:
                w.frame_log(f"[Searching] 搜房结束回调失败: {exc}")

        w.change_stage('跑图阶段')
        return True

    def _continue_searching_until_timer(self, w: 'FrameWorker', reason: str):
        self.stop_auto_forward(w)
        if self._can_finish_searching(w):
            w.frame_log(f"[Searching] {reason}，搜房计时已满，切换到跑图阶段")
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

        w.frame_log(f"[Searching] {reason}，但搜房未满10分钟，重置本轮目标继续搜房")
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

    def _refresh_frame_and_handle_jump(self, w: 'FrameWorker', reason: str = ""):
        refreshed = w.refresh_frame()
        raw_location = w.get_info("location")
        current_loc = self._remember_valid_location(raw_location)
        location_source = "current"
        if current_loc is None:
            current_loc = self._last_valid_location()
            location_source = "last_valid" if current_loc is not None else "missing"

        current_dir = w.get_info("direction")
        house_scene = self._get_house_scene(w)
        if reason or not refreshed or location_source != "current":
            notes = []
            if not refreshed:
                notes.append("刷新失败")
            if location_source != "current":
                notes.append(f"location_source={location_source}")
            suffix = f"，{'，'.join(notes)}" if notes else ""
            w.frame_log(
                f"[Frame] {reason or '画面刷新'}：raw_location={raw_location}，当前位置={current_loc}，"
                f"方向={current_dir}，场景={self._house_scene_label(house_scene)}{suffix}"
            )

        jump_reason = reason or f"{getattr(self, 'status', 'UNKNOWN')} 刷新后检查"
        self._handle_global_jump_forward_if_visible(w, jump_reason)
        return refreshed

    def _handle_global_jump_forward_if_visible(self, w: 'FrameWorker', reason: str) -> bool:
        if getattr(self, "_jump_forward_guard", False):
            return False
        if getattr(w, "current_stage", None) != "搜房阶段":
            return False
        if not w.get_info('跳跃'):
            self._jump_forward_wait_until_hidden = False
            return False
        if getattr(self, "_jump_forward_wait_until_hidden", False):
            return False

        w.frame_log(f"[Jump] 搜房全局抢占：{reason} 检测到跳跃按钮，只点击一次跳跃并轻微前推")
        return self.handle_jump_logic(w, reason)

    def _normalize_location_value(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return check_location(value)

    def _remember_valid_location(self, value):
        loc = self._normalize_location_value(value)
        if loc is not None:
            self.last_valid_location = loc
        return loc

    def _last_valid_location(self):
        return self._normalize_location_value(getattr(self, "last_valid_location", None))

    def _reset_route_stuck_bypass(self):
        self.route_stuck_reference_loc = None
        self.route_stuck_bypass_attempts = 0
        self.house_bypass_unstuck_pause_until = 0.0
        self._reset_entry_near_micro_adjust()

    def _reset_stuck_history_for_frame_wait(self, reason: str = ""):
        if getattr(self, "history_locations", None):
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f"[Stuck] {reason or '等待刷新帧'}，清空卡住窗口，避免把等待帧误判为卡死")
        self.history_locations = []

    def _reset_entry_near_micro_adjust(self):
        self.entry_near_micro_adjust_attempts = 0

    def _pause_unstuck_for_house_bypass(self, phase_label='NAV'):
        self.house_bypass_unstuck_pause_until = (
            time.monotonic() + self.HOUSE_BYPASS_UNSTUCK_PAUSE_SECONDS
        )
        self.history_locations = []
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[NavBypass] {phase_label} 绕房调整视角/前推期间暂停通用避障')

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
        w.frame_log(f"[Searching] {reason}")

        if not self.start_searching(w):
            return False
        if w.current_stage != '搜房阶段':
            return False

        if self.current_house_id is not None:
            self.completed_houses.add(self.current_house_id)
        self.searching_number += 1
        w.frame_log(f"[Searching] 房屋 {self.current_house_id} 完成，累计已搜 {self.searching_number} 个")

        self._refresh_frame_and_handle_jump(w)
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
        w.frame_log(f"[Searching] {reason}，执行快速出房")

        result = self._exit_house(w)
        if result is None:
            self._refresh_frame_and_handle_jump(w)
            result = self._get_house_scene(w) != 0

        self.indoor_stuck_frames = 0
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.history_locations = []
        self._reset_route_stuck_bypass()

        if result:
            w.frame_log("[Searching] 快速出房完成，继续寻找下一个进门点")
            return True
        w.frame_log("[Searching] 快速出房暂未确认成功，下一轮继续兜底")
        return False

    def _handle_indoor_during_entry_route(self, w: 'FrameWorker', current_loc, reason: str) -> bool:
        if self._get_house_scene(w) != 0:
            return False

        current_stage = getattr(w, "current_stage", None)
        if current_stage and current_stage != '搜房阶段':
            return False

        self.stop_auto_forward(w)
        w.frame_log('[Action] 转入当前房搜房')
        matched = self._resolve_house_by_location(current_loc)
        matched_house_id = None
        matched_entry = None

        if matched:
            matched_house_id, matched_entry, matched_dist = matched
            w.frame_log(
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
            w.frame_log(f"[Searching] {reason}，当前位置未匹配到房屋列表，搜完后不写入完成房屋")
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
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[搜房] 入屋搜房已超过{self.HOUSE_SEARCH_TIMEOUT_SECONDS}s，停止搜房并执行出房策略')
        return self.house_search_timer.expired()

    def _should_stop_house_search(self, w: 'FrameWorker'):
        return self._should_abort(w) or self._house_search_timed_out()

    def _force_exit_after_search_timeout(self, w: 'FrameWorker'):
        self.stop_auto_forward(w)
        self._clear_house_search_timer()
        self._refresh_frame_and_handle_jump(w)

        if self._get_house_scene(w) != 0:
            w.frame_log("[搜房] 超时时已不在屋内，视为出房完成")
            return True

        w.frame_log("[搜房] 超时兜底：启动 HouseExitManager 直接出房")
        self.house_exit_manager.reset()
        for _ in range(30):
            if self._should_abort(w):
                return False
            if self.house_exit_manager.process(w):
                w.frame_log("[搜房] 超时兜底出房成功")
                return True

        w.frame_log("[搜房] HouseExitManager 未出房，回退到原出房策略")
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
        w.frame_log(f"[Unstuck] 房体绕行空隙判断：left={left_score:.2f}, right={right_score:.2f}，选择{side}")
        return side

    def _bypass_front_house_block(self, w: 'FrameWorker', current_loc, safe_get_loc):
        w.frame_log("[Unstuck] 室外卡住，先后退确认前方是否为房体阻挡")
        w.tap_single('摇杆', y_bias=300, dura=450, wait=900)
        self._refresh_frame_and_handle_jump(w)

        if not self._front_house_blocking(w):
            w.frame_log("[Unstuck] 后退后前方未确认房体，交给通用避障")
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
            w.frame_log(f"[Unstuck] 前方房体挡路，尝试向{side}侧滑绕房")
            side_base_loc = safe_get_loc() or back_loc

            for _ in range(self.HOUSE_BYPASS_SIDE_STEPS):
                if self._should_abort(w):
                    return False
                w.frame_log('[Action] ' + str((f'向{self._side_label(side)}侧滑绕房') or ('前方确认是房体阻挡，先向侧边滑动寻找绕行空间')))
                w.tap_single('摇杆', x_bias=bias, dura=450, wait=700)
                self._refresh_frame_and_handle_jump(w)
                if not self._front_house_blocking(w):
                    break

            side_loc = safe_get_loc()
            if not side_loc or not side_base_loc or get_distance(side_base_loc, side_loc) <= 0.5:
                w.frame_log(f"[Unstuck] {side}侧滑位移不足，尝试另一侧")
                continue

            for _ in range(self.HOUSE_BYPASS_FORWARD_STEPS):
                if self._should_abort(w):
                    return False
                w.frame_log('[Action] 绕房前推')
                w.tap_single('摇杆', y_bias=-300, dura=400, wait=900)
                self._refresh_frame_and_handle_jump(w)
                forward_loc = safe_get_loc()
                if forward_loc and get_distance(current_loc, forward_loc) > self.stuck_threshold:
                    w.frame_log("[Unstuck] 绕房通过成功")
                    return True
                if self._front_house_blocking(w):
                    w.tap_single('摇杆', x_bias=bias, dura=350, wait=500)
                    self._refresh_frame_and_handle_jump(w)

            w.frame_log(f"[Unstuck] {side}侧仍未绕开，尝试另一侧")

        w.frame_log("[Unstuck] 房体绕行未成功，回退到通用避障")
        return False

    def _is_searching_stage_frame(self, w: 'FrameWorker') -> bool:
        current_stage = getattr(w, "current_stage", None)
        return current_stage is None or current_stage == '搜房阶段'

    def _safe_get_frame_location(self, w: 'FrameWorker'):
        raw = w.get_info('location')
        if raw is None:
            return None
        return self._normalize_location_value(raw)

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
                w.frame_log(f"[NavBypass] {phase_label} 缺少方向角，直接向{side}拨视角")
                w.tap_single('视角', x_bias=x_bias, dura=520, wait=500)
            else:
                target_direction = (float(direction) + (step if side == "right" else -step)) % 360
                if target_direction == 0:
                    target_direction = 360
                w.frame_log(
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
            self._refresh_frame_and_handle_jump(w)
            direction = w.get_info('direction')
        return True

    def _bypass_front_house_by_view_turn(self, w: 'FrameWorker', target_loc, phase_label='NAV'):
        self._pause_unstuck_for_house_bypass(phase_label)
        self.stop_auto_forward(w)
        side = self._choose_house_bypass_side(w)
        w.frame_log(
            f"[NavBypass] {phase_label} 前方检测到房体/门窗，"
            f"固定向{side}侧转向绕行，最大{self.HOUSE_OBSTACLE_MAX_TURN_DEGREES}度"
        )

        self._rotate_view_until_house_clear(w, side, phase_label)
        if self._should_abort(w):
            return True

        w.frame_log(f"[NavBypass] {phase_label} 绕行视角已处理，前推3秒后继续导航")
        w.tap_single(
            '摇杆',
            y_bias=self.HOUSE_OBSTACLE_FORWARD_Y_BIAS,
            dura=self.HOUSE_OBSTACLE_FORWARD_DURA,
            wait=self.HOUSE_OBSTACLE_FORWARD_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
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
            w.frame_log("[NavBypass] 非目标房门不在中间1/3，横向调整人物位置")
            w.tap_single(
                '摇杆',
                x_bias=x_bias,
                dura=self.VISIBLE_DOOR_CENTER_SIDE_DURA,
                wait=self.VISIBLE_DOOR_CENTER_SIDE_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            door = self.find_largest_door(w)
            if door is None:
                return None
        return door

    def _try_enter_visible_non_target_house(self, w: 'FrameWorker', current_loc, phase_label='NAV'):
        door = self.find_largest_door(w)
        if door is None:
            return False

        w.frame_log(f"[NavBypass] {phase_label} 前方不是当前目标但看到门，尝试顺路进房")
        self.stop_auto_forward(w)
        door = self._center_visible_door_by_lateral_move(w, door)
        if door is None:
            w.frame_log("[NavBypass] 横向调整后门目标丢失，改走绕房策略")
            return False

        if not self._align_to_door_detection(w, door):
            w.frame_log("[NavBypass] 门视角对齐失败，改走绕房策略")
            return False

        for _ in range(3):
            if self._should_abort(w):
                return True
            if w.get_info('开门'):
                w.frame_log('[Action] 点击开门')
                w.click('开门')
                time.sleep(1)
            w.frame_log('[Action] 顺路进房前推')
            w.tap_single(
                '摇杆',
                y_bias=self.VISIBLE_DOOR_FORWARD_Y_BIAS,
                dura=self.VISIBLE_DOOR_FORWARD_DURA,
                wait=self.VISIBLE_DOOR_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            if self._get_house_scene(w) == 0:
                indoor_loc = self._safe_get_frame_location(w) or current_loc
                return self._handle_indoor_during_entry_route(
                    w,
                    indoor_loc,
                    "前方非目标房门顺路进房",
                )

        w.frame_log("[NavBypass] 顺路进房未确认 indoor，改走绕房策略")
        return False

    def _handle_front_stone_wall_on_search_route(self, w: 'FrameWorker', current_loc, phase_label='NAV') -> bool:
        self.stop_auto_forward(w)
        w.frame_log(
            f"[NavBypass] {phase_label} 前方发现 stone_wall，"
            f"先短前推 y_bias={self.STONE_WALL_FORWARD_Y_BIAS}, wait={self.STONE_WALL_FORWARD_WAIT}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.STONE_WALL_FORWARD_Y_BIAS,
            dura=self.STONE_WALL_FORWARD_DURA,
            wait=self.STONE_WALL_FORWARD_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        if self._get_house_scene(w) == self.HOUSE_INDOOR:
            indoor_loc = self._safe_get_frame_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                indoor_loc,
                "stone_wall 短前推后确认进房",
            )

        jump_handled = False
        if w.get_info('跳跃'):
            jump_handled = self.handle_jump_logic(w, f"{phase_label} stone_wall 前推后出现跳跃按钮")
        else:
            w.frame_log(f"[NavBypass] {phase_label} stone_wall 前推后未识别到跳跃按钮，仍尝试跳跃前推")

        if not jump_handled:
            w.tap_single(
                '摇杆',
                y_bias=self.STONE_WALL_JUMP_FORWARD_Y_BIAS,
                dura=self.STONE_WALL_JUMP_FORWARD_DURA,
                wait=self.STONE_WALL_JUMP_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
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
        w.frame_log(f"[{phase_label}] 主动绕房已取消，只在 near_wall 且确认卡住后触发脱困")
        return False

    def _try_lock_visible_door_after_block(self, w: 'FrameWorker') -> bool:
        door = self.find_largest_door(w)
        if door is None:
            return False

        w.frame_log("[Unstuck] 后退后前方是房子且定位到门，直接锁门进入交互流程")
        self.stop_auto_forward(w)
        self._align_to_door_detection(w, door)
        self._refresh_frame_and_handle_jump(w)

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
    def _side_label(side: str) -> str:
        return "左" if side == "left" else "右"

    @staticmethod
    def _door_center_x(door):
        try:
            return (float(door[0]) + float(door[2])) / 2
        except (TypeError, ValueError, IndexError):
            return None

    def _entry_door_frame_width(self):
        inf_w, inf_h = get_wh()
        return max(int(inf_w or 0), int(inf_h or 0))

    def _get_visual_frame_size(self, w):
        frame = getattr(w, "frame", None)
        if frame is not None and hasattr(frame, "shape"):
            try:
                frame_h, frame_w = frame.shape[:2]
                frame_w = int(frame_w)
                frame_h = int(frame_h)
                if frame_w > 0 and frame_h > 0:
                    return frame_w, frame_h
            except Exception:
                pass

        inf_w, inf_h = get_wh()
        frame_w = max(int(inf_w or 0), int(inf_h or 0))
        frame_h = min(int(inf_w or 0), int(inf_h or 0))
        if frame_w > 0 and frame_h > 0:
            return frame_w, frame_h
        return None

    def _get_detection_area_ratio(self, w, det):
        frame_size = self._get_visual_frame_size(w)
        if frame_size is None:
            return None
        frame_w, frame_h = frame_size
        try:
            box_w = max(0.0, float(det[2]) - float(det[0]))
            box_h = max(0.0, float(det[3]) - float(det[1]))
            return (box_w * box_h) / max(1.0, float(frame_w * frame_h))
        except (TypeError, ValueError, IndexError):
            return None

    def _get_visible_door_center_offset(self, w, door):
        frame_size = self._get_visual_frame_size(w)
        if frame_size is None:
            return None, None, None
        frame_w, _ = frame_size
        center_x = self._door_center_x(door)
        if center_x is None:
            return None, None, frame_w

        screen_w = self.screen_w
        if not screen_w:
            screen_w, _ = get_resolution()
            self.screen_w = screen_w
        if not screen_w:
            screen_w = frame_w

        door_area_ratio = self._get_detection_area_ratio(w, door)
        self.entry_door_last_area_ratio = door_area_ratio
        offset_real = (center_x - (frame_w / 2.0)) * (float(screen_w) / float(frame_w))
        return offset_real, door_area_ratio, frame_w

    def _get_door_align_center_threshold(self, tolerance_px=80):
        try:
            threshold = int(tolerance_px)
        except (TypeError, ValueError):
            threshold = self.ENTRY_DOOR_ALIGN_CENTER_THRESHOLD
        threshold = max(threshold, self.ENTRY_DOOR_ALIGN_CENTER_THRESHOLD)

        ratio = self.entry_door_last_area_ratio
        if ratio is None:
            return threshold
        if ratio >= self.ENTRY_DOOR_ALIGN_VERY_NEAR_AREA_RATIO:
            return max(threshold, self.ENTRY_DOOR_ALIGN_VERY_NEAR_CENTER_THRESHOLD)
        if ratio >= self.ENTRY_DOOR_ALIGN_NEAR_AREA_RATIO:
            return max(threshold, self.ENTRY_DOOR_ALIGN_NEAR_CENTER_THRESHOLD)
        if ratio >= self.ENTRY_DOOR_ALIGN_CLOSE_AREA_RATIO:
            return max(threshold, self.ENTRY_DOOR_ALIGN_CLOSE_CENTER_THRESHOLD)
        return threshold

    def _get_strict_door_align_center_threshold(self, tolerance_px=80):
        try:
            threshold = int(tolerance_px)
        except (TypeError, ValueError):
            threshold = self.ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX
        return max(1, threshold)

    def _mark_entry_door_strict_align_after_backoff(self):
        self._entry_door_force_strict_align_once = True

    def _consume_entry_door_strict_align_after_backoff(self) -> bool:
        if getattr(self, "_entry_door_force_strict_align_once", False):
            self._entry_door_force_strict_align_once = False
            return True
        return False

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

    def _shift_edge_visible_entry_door_by_lateral_move(self, w: 'FrameWorker', door, phase_label='Nav') -> bool:
        frame_size = self._get_visual_frame_size(w)
        if frame_size is None:
            return False

        frame_w, _ = frame_size
        ratio = self._door_center_ratio(door, frame_w)
        if ratio is None:
            return False

        if ratio <= self.ENTRY_DOOR_EDGE_LATERAL_LEFT_RATIO:
            side = "left"
            x_bias = -self.VISIBLE_DOOR_CENTER_SIDE_BIAS
        elif ratio >= self.ENTRY_DOOR_EDGE_LATERAL_RIGHT_RATIO:
            side = "right"
            x_bias = self.VISIBLE_DOOR_CENTER_SIDE_BIAS
        else:
            return False

        w.frame_log(
            f"[{phase_label}] 入门点附近最大门中心在画面{self._side_label(side)}侧边缘 "
            f"(ratio={ratio:.2f})，不转视角，改用摇杆横移对齐门"
        )
        w.tap_single(
            '摇杆',
            x_bias=x_bias,
            y_bias=0,
            dura=self.VISIBLE_DOOR_CENTER_SIDE_DURA,
            wait=self.VISIBLE_DOOR_CENTER_SIDE_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        self.history_locations = []
        return True

    def _align_visible_entry_door_for_direct_push(self, w: 'FrameWorker', door, phase_label='Nav'):
        """Use the same visual-center loop as car alignment before pushing through a door."""
        wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "门框视觉对齐前")
        if wall_result is not None:
            return wall_result if wall_result == "indoor" else "near_wall_backoff"

        if not self._align_to_door_detection(
            w,
            door,
            tolerance_px=self.ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX,
            phase_label=phase_label,
        ):
            w.frame_log(f"[{phase_label}] 视觉中心闭环对门失败，继续原进门流程")
            return None

        wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "门框视觉对齐后")
        if wall_result is not None:
            return wall_result if wall_result == "indoor" else "near_wall_backoff"

        door = self.find_largest_door(w)
        if door is None:
            w.frame_log(f"[{phase_label}] 视觉对门后门目标丢失，继续原进门流程")
            return None

        w.frame_log(f"[{phase_label}] 门已按视觉中心闭环对准，准备自动开门直推")
        return door

    def _backoff_after_centered_entry_push_failure(self, w: 'FrameWorker', phase_label: str, failures: int, reason: str):
        w.frame_log(
            f"[{phase_label}] {reason}，后拉记为进门失败 "
            f"{failures}/{self.ENTRY_DOOR_DIRECT_MAX_FAILURES}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_DIRECT_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_DIRECT_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        self._mark_entry_door_strict_align_after_backoff()

    def _handle_entry_near_wall_if_needed(self, w: 'FrameWorker', phase_label: str, reason: str):
        if self._get_house_scene(w) != self.HOUSE_NEAR_WALL:
            return None

        w.frame_log(
                    f"[{phase_label}] {reason}：near_wall，先右移 "
                    f"x={self.ENTRY_NEAR_WALL_SIDE_ESCAPE_X_BIAS}，再后拉 "
                    f"y={self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS}"
                )
        w.tap_single(
            '摇杆',
            x_bias=self.ENTRY_NEAR_WALL_SIDE_ESCAPE_X_BIAS,
            y_bias=0,
            dura=self.ENTRY_NEAR_WALL_SIDE_ESCAPE_DURA,
            wait=self.ENTRY_NEAR_WALL_SIDE_ESCAPE_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)

        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_DIRECT_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_DIRECT_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        self.history_locations = []

        if self._get_house_scene(w) == self.HOUSE_INDOOR:
            w.frame_log(f"[{phase_label}] near_wall 后拉后仍在 indoor，直接启动搜房策略")
            return "indoor"

        self._mark_entry_door_strict_align_after_backoff()
        scene_after_backoff = self._get_house_scene(w)
        if scene_after_backoff in {self.HOUSE_OUTDOOR, self.HOUSE_ROOFTOP, self.HOUSE_NEAR_DOOR}:
            w.frame_log(
                f"[{phase_label}] near_wall 后拉后已到屋外/门口 house_scene={scene_after_backoff}，"
                f"向左轻推抵消刚才右移后继续进门"
            )
            w.tap_single(
                '摇杆',
                x_bias=-self.ENTRY_NEAR_WALL_SIDE_ESCAPE_X_BIAS,
                y_bias=0,
                dura=self.ENTRY_NEAR_WALL_SIDE_ESCAPE_DURA,
                wait=self.ENTRY_NEAR_WALL_SIDE_ESCAPE_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            self.history_locations = []
            return "retry"

        w.frame_log(f"[{phase_label}] near_wall 后拉后 house_scene={scene_after_backoff}，等待下一轮继续调整")
        return "adjusting"

    def _backoff_entry_near_wall_if_needed(self, w: 'FrameWorker', phase_label: str, reason: str) -> bool:
        return self._handle_entry_near_wall_if_needed(w, phase_label, reason) is not None

    def _handle_nav_near_entry_scene_if_needed(
        self,
        w: 'FrameWorker',
        phase_label: str,
        reason: str,
        current_loc=None,
        target_loc=None,
    ):
        scene = self._get_house_scene(w)
        if scene not in {self.HOUSE_NEAR_WALL, self.HOUSE_NEAR_DOOR}:
            return None

        if scene == self.HOUSE_NEAR_WALL:
            recovery_result = self._recover_outdoor_near_wall(
                w,
                current_loc,
                target_loc,
                phase_label,
                reason,
            )
            if recovery_result is not None:
                return recovery_result

        scene_label = "near_wall" if scene == self.HOUSE_NEAR_WALL else "near_door"
        w.frame_log(
                    f"[{phase_label}] {reason}：{scene_label}，后拉脱离门墙后继续当前入口"
                )
        self.stop_auto_forward(w)
        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_DIRECT_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_DIRECT_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        self.history_locations = []

        scene_after_backoff = self._get_house_scene(w)
        if scene_after_backoff == self.HOUSE_INDOOR:
            w.frame_log(f"[{phase_label}] {scene_label} 后拉后已在 indoor，直接启动当前房搜房")
            return "indoor"

        self._mark_entry_door_strict_align_after_backoff()
        if w.get_info('跳跃'):
            w.frame_log(f"[{phase_label}] {scene_label} 后拉后仍有跳跃按钮，尝试跳过房体/石墙障碍")
            self.handle_jump_logic(w, f"{phase_label} {scene_label} 后拉后跳障")
            if self._get_house_scene(w) == self.HOUSE_INDOOR:
                w.frame_log(f"[{phase_label}] 跳障后已在 indoor，直接启动当前房搜房")
                return "indoor"

        return "adjusting"

    def _recover_outdoor_near_wall(self, w: 'FrameWorker', current_loc, target_loc, phase_label: str, reason: str):
        """Recover from an outdoor wall hit without disturbing indoor or close-entry handling."""
        if self._get_house_scene(w) != self.HOUSE_NEAR_WALL:
            return None

        target = self._normalize_location_value(target_loc)
        current = self._normalize_location_value(current_loc) or self._get_current_location(w)
        if target is None or current is None:
            return None

        distance = get_distance(current, target)
        if distance <= self.OUTDOOR_NEAR_WALL_ENTRY_EXEMPT_DISTANCE:
            return None

        if w.get_info('跳跃'):
            w.frame_log(f"[{phase_label}] {reason}室外 near_wall 且可跳跃，优先跳跃+前推")
            self.handle_jump_logic(w, f"{phase_label} 室外 near_wall")
            self.history_locations = []
            return "adjusting"

        original_direction = w.get_info('direction')
        for cycle in range(self.OUTDOOR_NEAR_WALL_RECOVERY_MAX_CYCLES):
            cycle_label = f"{cycle + 1}/{self.OUTDOOR_NEAR_WALL_RECOVERY_MAX_CYCLES}"
            w.frame_log(
                f"[{phase_label}] {reason}室外 near_wall 恢复 {cycle_label}: "
                "后拉 -> 左滑视野 -> 前推约2秒 -> 回正"
            )
            self.stop_auto_forward(w)
            w.tap_single(
                '摇杆',
                y_bias=self.OUTDOOR_NEAR_WALL_BACK_Y_BIAS,
                dura=self.OUTDOOR_NEAR_WALL_BACK_DURA,
                wait=self.OUTDOOR_NEAR_WALL_BACK_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            if self._get_house_scene(w) == self.HOUSE_INDOOR:
                return "indoor"
            if w.get_info('跳跃'):
                self.handle_jump_logic(w, f"{phase_label} 室外 near_wall 后拉后")
                self.history_locations = []
                return "adjusting"

            w.tap_single(
                '视角',
                x_bias=self.OUTDOOR_NEAR_WALL_LEFT_VIEW_X_BIAS,
                dura=self.OUTDOOR_NEAR_WALL_LEFT_VIEW_DURA,
                wait=self.OUTDOOR_NEAR_WALL_LEFT_VIEW_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            if w.get_info('跳跃'):
                self.handle_jump_logic(w, f"{phase_label} 室外 near_wall 左滑后")
                self.history_locations = []
                return "adjusting"

            w.tap_single(
                '摇杆',
                y_bias=self.OUTDOOR_NEAR_WALL_FORWARD_Y_BIAS,
                dura=self.OUTDOOR_NEAR_WALL_FORWARD_DURA,
                wait=self.OUTDOOR_NEAR_WALL_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            if w.get_info('跳跃'):
                self.handle_jump_logic(w, f"{phase_label} 室外 near_wall 左绕前推后")
                self.history_locations = []
                return "adjusting"

            current_direction = w.get_info('direction')
            if original_direction is not None and current_direction is not None:
                self.align_direction_blocking(
                    w,
                    current_direction,
                    original_direction,
                    threshold=8,
                    max_steps=3,
                )
                self._refresh_frame_and_handle_jump(w)

            if self._get_house_scene(w) == self.HOUSE_INDOOR:
                return "indoor"
            if w.get_info('跳跃'):
                self.handle_jump_logic(w, f"{phase_label} 室外 near_wall 回正后")
                self.history_locations = []
                return "adjusting"

            if self._get_house_scene(w) != self.HOUSE_NEAR_WALL:
                refreshed_loc = self._get_current_location(w)
                if refreshed_loc is not None:
                    if not self.align_direction(w, target, threshold=10, max_steps=3):
                        self.history_locations = []
                        return "adjusting"
                w.tap_single(
                    '摇杆',
                    y_bias=self.OUTDOOR_NEAR_WALL_FORWARD_Y_BIAS,
                    dura=self.OUTDOOR_NEAR_WALL_CLEAR_FORWARD_DURA,
                    wait=self.OUTDOOR_NEAR_WALL_CLEAR_FORWARD_WAIT,
                )
                self._refresh_frame_and_handle_jump(w)
                self.history_locations = []
                return "recovered"

        self.history_locations = []
        return "adjusting"

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

                self._refresh_frame_and_handle_jump(w)
                scene = self._get_house_scene(w)
                if scene == self.HOUSE_INDOOR:
                    w.frame_log(f"[{phase_label}] 直推前已是 indoor，启动搜房策略")
                    return "indoor"
                if scene == self.HOUSE_NEAR_WALL:
                    wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "直推前")
                    if wall_result == "indoor":
                        return "indoor"
                    return "adjusting"

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
                        w.frame_log(
                            f"[{phase_label}] 前推后仍能定位到门，继续调整视角对齐 "
                            f"{realign_attempts}/{self.ENTRY_DOOR_DIRECT_REALIGN_MAX_ATTEMPTS}"
                        )

                    aligned_door = self._align_visible_entry_door_for_direct_push(w, visible_door, phase_label)
                    if aligned_door == "indoor":
                        return "indoor"
                    if aligned_door == "near_wall_backoff":
                        return "adjusting"
                    if aligned_door is None:
                        if not direct_started:
                            return "not_ready"
                        w.frame_log(
                            f"[{phase_label}] 重新对齐过程中门目标丢失，"
                            f"不记失败，沿当前方向继续前推"
                        )
                        door = None
                    else:
                        door = aligned_door
                elif not direct_started and door is None:
                    w.frame_log(f"[{phase_label}] 门未进入视野，继续原进门流程")
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
                    w.frame_log(f"[{phase_label}] 本次前推前未识别到门，沿当前进门方向继续直推")

                w.frame_log(
                                    f"[{phase_label}] 直推进门：第 {pushes_this_failure + 1}/"
                                    f"{self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE} 次，"
                                    f"y={self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS}，"
                                    f"dura={self.ENTRY_DOOR_DIRECT_FORWARD_DURA}"
                                )
                w.tap_single(
                    '摇杆',
                    y_bias=self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS,
                    dura=self.ENTRY_DOOR_DIRECT_FORWARD_DURA,
                    wait=self.ENTRY_DOOR_DIRECT_FORWARD_WAIT,
                )
                direct_started = True
                pushes_this_failure += 1
                self._refresh_frame_and_handle_jump(w)

                scene = self._get_house_scene(w)
                if scene == self.HOUSE_INDOOR:
                    w.frame_log(f"[{phase_label}] 自动开门直推后 house_scene=indoor，启动搜房策略")
                    return "indoor"

                if scene == self.HOUSE_NEAR_WALL:
                    wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "直推后")
                    if wall_result == "indoor":
                        return "indoor"
                    return "adjusting"

                visible_after_push = self.find_largest_door(w)
                if visible_after_push is not None:
                    door = visible_after_push
                    w.frame_log(
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

                w.frame_log(
                    f"[{phase_label}] 直推后暂未进房 house_scene={scene}，"
                    f"本轮已推 {pushes_this_failure}/{self.ENTRY_DOOR_DIRECT_PUSHES_PER_FAILURE}"
                )

        w.frame_log(f"[{phase_label}] 自动开门直推累计失败 {failures} 次，判定当前进门点失败")
        return "failed"

    def _scan_entry_door_after_micro_adjust(self, w: 'FrameWorker', phase_label='Nav'):
        w.frame_log(f"[{phase_label}] 入门点距离已微调到 0/近似0，开始看正前方有没有门")
        door = self.find_largest_door(w)
        if door is not None:
            w.frame_log(f"[{phase_label}] 正前方看到了门，进入对准门流程: door={door}")
            return door

        w.frame_log(
            f"[{phase_label}] 入门点距离已为0但正前方没看到门，先后拉扩视野 "
            f"y={self.ENTRY_DOOR_MISSING_BACKOFF_Y_BIAS}，"
            f"dura={self.ENTRY_DOOR_MISSING_BACKOFF_DURA}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_MISSING_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_MISSING_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_MISSING_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        if self._is_indoor(w):
            w.frame_log(f"[{phase_label}] 后拉扩视野后 house_scene=indoor，直接启动搜房")
            return "indoor"
        door = self.find_largest_door(w)
        if door is not None:
            w.frame_log(f"[{phase_label}] 后拉扩视野后看到门，进入对准门流程: door={door}")
            self._mark_entry_door_strict_align_after_backoff()
            return door

        return self._push_missing_entry_door_after_strict_align(w, phase_label)

    def _push_missing_entry_door_after_strict_align(self, w: 'FrameWorker', phase_label='Nav'):
        ideal_angle = self.active_entry.get('direction') if self.active_entry else None
        if ideal_angle is None:
            w.frame_log(f"[{phase_label}] 后拉扩视野后仍没看到门，但缺少入门方向，舍弃当前入门点")
            return None

        w.frame_log(f"[{phase_label}] 后拉扩视野后仍没看到门，严格回正入门方向 ideal_angle={ideal_angle}")
        if not self._align_near_entry_direction(w, ideal_angle):
            w.frame_log(f"[{phase_label}] 入门方向仍未严格对齐，等待下一帧继续校准")
            return "adjusting"

        w.frame_log(
            f"[{phase_label}] 入门方向已严格回正但仍没门，执行大前推确认进门 "
            f"y={self.ENTRY_DOOR_MISSING_STRICT_FORWARD_Y_BIAS}，"
            f"dura={self.ENTRY_DOOR_MISSING_STRICT_FORWARD_DURA}"
        )
        w.tap_single(
            '摇杆',
            y_bias=self.ENTRY_DOOR_MISSING_STRICT_FORWARD_Y_BIAS,
            dura=self.ENTRY_DOOR_MISSING_STRICT_FORWARD_DURA,
            wait=self.ENTRY_DOOR_MISSING_STRICT_FORWARD_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        if self._is_indoor(w):
            w.frame_log(f"[{phase_label}] 大前推后 house_scene=indoor，确认已经进门")
            return "indoor"

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_NEAR_WALL:
            w.frame_log(
                f"[{phase_label}] 大前推后撞墙，稍微后拉复核是否已进门 "
                f"y={self.ENTRY_DOOR_MISSING_WALL_BACKOFF_Y_BIAS}，"
                f"dura={self.ENTRY_DOOR_MISSING_WALL_BACKOFF_DURA}"
            )
            w.tap_single(
                '摇杆',
                y_bias=self.ENTRY_DOOR_MISSING_WALL_BACKOFF_Y_BIAS,
                dura=self.ENTRY_DOOR_MISSING_WALL_BACKOFF_DURA,
                wait=self.ENTRY_DOOR_MISSING_WALL_BACKOFF_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            if self._is_indoor(w):
                w.frame_log(f"[{phase_label}] 撞墙后拉复核 house_scene=indoor，确认已经进门")
                return "indoor"

        w.frame_log(f"[{phase_label}] 大前推/撞墙后拉后仍未进房，舍弃当前入门点")
        return None

    def _push_aligned_entry_door_and_check_indoor(self, w: 'FrameWorker', phase_label='Nav', initial_door=None) -> str:
        door = initial_door
        for attempt in range(self.ENTRY_DOOR_ALIGNED_PUSH_MAX_ATTEMPTS):
            if self._should_abort(w):
                return "aborted"

            if door is None:
                w.frame_log(f"[{phase_label}] 前推前门目标丢失，重新正前方看门；没门就回正入门方向大前推确认")
                door = self._scan_entry_door_after_micro_adjust(w, phase_label)
                if door == "indoor":
                    return "indoor"
                if door == "adjusting":
                    return "adjusting"
                if door is None:
                    return "failed"

            w.frame_log(
                f"[{phase_label}] 第 {attempt + 1}/{self.ENTRY_DOOR_ALIGNED_PUSH_MAX_ATTEMPTS} 次对准门: "
                f"door={door}"
            )
            if self._shift_edge_visible_entry_door_by_lateral_move(w, door, phase_label):
                return "adjusting"

            if not self._align_to_door_detection(
                w,
                door,
                tolerance_px=self.ENTRY_DOOR_FINAL_VIEW_TOLERANCE_PX,
            ):
                w.frame_log(f"[{phase_label}] 对准门失败，重新获取门目标后继续")
                self._refresh_frame_and_handle_jump(w)
                door = self.find_largest_door(w)
                continue

            w.frame_log(
                            f"[{phase_label}] 门已对准，第 {attempt + 1}/"
                            f"{self.ENTRY_DOOR_ALIGNED_PUSH_MAX_ATTEMPTS} 次前推 "
                            f"y={self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS}，"
                            f"dura={self.ENTRY_DOOR_DIRECT_FORWARD_DURA}"
                        )
            w.tap_single(
                '摇杆',
                y_bias=self.ENTRY_DOOR_DIRECT_FORWARD_Y_BIAS,
                dura=self.ENTRY_DOOR_DIRECT_FORWARD_DURA,
                wait=self.ENTRY_DOOR_DIRECT_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)

            scene = self._get_house_scene(w)
            if scene == self.HOUSE_INDOOR:
                w.frame_log(f"[{phase_label}] 前推后 house_scene=indoor，启动搜房")
                return "indoor"

            if w.get_info('跳跃'):
                w.frame_log(
                    f"[{phase_label}] 前推后检测到跳跃按钮，判断前方有石墙/障碍，"
                    f"按一次跳跃并轻微前推"
                )
                self.handle_jump_logic(w, f"{phase_label} 前推后出现跳跃按钮")
                scene = self._get_house_scene(w)
                if scene == self.HOUSE_INDOOR:
                    w.frame_log(f"[{phase_label}] 跳跃翻障后 house_scene=indoor，启动搜房")
                    return "indoor"

                w.frame_log(f"[{phase_label}] 跳跃翻障后还未进屋，重新定位门并对准后前推")
                door = self.find_largest_door(w)
                continue

            door = self.find_largest_door(w)
            if door is not None:
                w.frame_log(f"[{phase_label}] 前推后仍能看到门，继续重新对准后前推")
                continue

            w.frame_log(f"[{phase_label}] 前推后未进屋且未再看到门，当前 house_scene={scene}，准备重试/失败")

        w.frame_log(f"[{phase_label}] 多次对门前推后仍未进入室内，判定当前入门点失败")
        return "failed"

    def _align_entry_door_after_arrival(self, w: 'FrameWorker', phase_label='Nav') -> str:
        """Arrived at the entry point: find the door, align it, then push through."""
        wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "到达进门点后")
        if wall_result == "indoor":
            return "indoor"
        if wall_result is not None:
            return "adjusting"

        door = self._scan_entry_door_after_micro_adjust(w, phase_label)
        if door == "indoor":
            return "indoor"
        if door == "adjusting":
            return "adjusting"
        if door is None:
            self._mark_current_entry_failed("入门点微调完成后仍未定位到门")
            return "failed"

        self.stop_auto_forward(w)
        result = self._push_aligned_entry_door_and_check_indoor(w, phase_label, door)
        if result == "failed":
            self._mark_current_entry_failed("对准门前推/跳跃翻障后仍未进入室内")
        return result

    def _align_entry_direction_at_near_point(self, w: 'FrameWorker', phase_label='Nav') -> bool:
        ideal_angle = self.active_entry.get('direction') if self.active_entry else None
        if ideal_angle is None:
            return True

        current_dir = w.get_info('direction')
        w.frame_log(
            f"[{phase_label}] 当前已在入门点附近，入门方向应为 {ideal_angle}，"
            f"开始把视角对齐到入门方向"
        )
        aligned = self._align_near_entry_direction(w, ideal_angle)
        if aligned:
            self._refresh_frame_and_handle_jump(w)
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

    def _try_visible_entry_door_before_micro_adjust(
        self,
        w: 'FrameWorker',
        target_loc,
        dist: float,
        phase_label='Nav',
    ) -> str:
        door = self.find_largest_door(w)
        if door is None:
            w.frame_log(
                f"[{phase_label}] 当前距离入门点 {target_loc} 为 {dist:.2f}，"
                "方向已对齐但还没看到门，继续慢速微调到入门点"
            )
            return "not_visible"

        w.frame_log(
            f"[{phase_label}] 当前距离入门点 {target_loc} 为 {dist:.2f}，"
            f"方向已对齐且已看到门，跳过微调到0，直接对准门前推: door={door}"
        )
        self.stop_auto_forward(w)
        result = self._push_aligned_entry_door_and_check_indoor(w, phase_label, door)
        if result == "failed":
            self._mark_current_entry_failed("入门点附近已见门但对准门前推/跳跃翻障后仍未进入室内")
        return result

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
            w.frame_log(
                f"[{phase_label}] 当前距离入门点 {target_loc} 为 {dist_val:.2f}，"
                f"已经足够贴近，不再左右微调摇杆"
            )
            return False

        refreshed_loc = self._get_current_location(w) or current_loc
        current_dir = w.get_info('direction')
        target_angle = calculate_angle(refreshed_loc, target_loc)
        if current_dir is None or target_angle is None:
            w.frame_log(f"[{phase_label}] 近门左右位置修正缺少方向/坐标，跳过")
            return False

        try:
            relative = (float(target_angle) - float(current_dir) + 540) % 360 - 180
        except (TypeError, ValueError):
            w.frame_log(f"[{phase_label}] 近门左右位置修正角度无效，跳过")
            return False

        abs_relative = abs(relative)
        if (
            abs_relative <= self.ENTRY_NEAR_LATERAL_CORRECT_MIN_RELATIVE_DEGREES
            or abs_relative >= self.ENTRY_NEAR_LATERAL_CORRECT_MAX_RELATIVE_DEGREES
        ):
            w.frame_log(
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
        w.frame_log(
                    f"[{phase_label}] 横向微调：距离={dist_val:.2f}，目标在{side}侧 "
                    f"{relative:.1f}°，x={x_bias}，dura={self.ENTRY_NEAR_LATERAL_CORRECT_DURA}"
                )
        w.tap_single(
            '摇杆',
            x_bias=x_bias,
            y_bias=0,
            dura=self.ENTRY_NEAR_LATERAL_CORRECT_DURA,
            wait=self.ENTRY_NEAR_LATERAL_CORRECT_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        self.history_locations = []
        return True

    def _handle_near_entry_point(self, w: 'FrameWorker', current_loc, target_loc, dist: float, phase_label='Nav') -> str:
        self.stop_auto_forward(w)
        w.frame_log(
            f"[{phase_label}] 当前距离入门点 {target_loc} 为 {dist:.2f} "
            f"<= {self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:g}，已经到达入门点附近，"
            f"停止自动前进，准备对齐入门方向并微调位置"
        )

        if not self._align_entry_direction_at_near_point(w, phase_label):
            w.frame_log(f"[{phase_label}] 进门点方向尚未对准，等待下一轮继续对准")
            return "aligning"

        visible_door_result = self._try_visible_entry_door_before_micro_adjust(
            w,
            target_loc,
            dist,
            phase_label,
        )
        if visible_door_result != "not_visible":
            self._reset_entry_near_micro_adjust()
            return visible_door_result

        wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "对准进门方向后")
        if wall_result == "indoor":
            return "indoor"
        if wall_result is not None:
            return "adjusting"

        micro_result = self._micro_adjust_near_entry_point(w, current_loc, target_loc, dist, phase_label)
        if micro_result == "adjusting":
            wall_result = self._handle_entry_near_wall_if_needed(w, phase_label, "入门点微调后")
            if wall_result == "indoor":
                return "indoor"
            if wall_result is not None:
                return "adjusting"
            w.frame_log(f"[{phase_label}] 入门点微调动作已执行，等待下一帧重新计算距离")
            return "adjusting"
        if micro_result == "failed":
            self._mark_current_entry_failed("入门点近距离微调多次后仍无法到达入门点")
            return "failed"
        if micro_result != "ready":
            return "adjusting"

        arrival_result = self._align_entry_door_after_arrival(w, phase_label)
        if arrival_result != "not_ready":
            self._reset_entry_near_micro_adjust()
            return arrival_result

        self._reset_entry_near_micro_adjust()
        return "not_ready"

    def _micro_adjust_near_entry_point(self, w: 'FrameWorker', current_loc, target_loc, dist: float, phase_label='Nav') -> str:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            w.frame_log(f"[{phase_label}] 入门点微调距离无效: dist={dist}，跳过微调")
            return "failed"

        if dist_val > self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:
            self._reset_entry_near_micro_adjust()
            return "outside"
        if dist_val <= self.ENTRY_NEAR_MICRO_DONE_DISTANCE:
            w.frame_log(
                f"[{phase_label}] 当前距离入门点 {target_loc} 为 {dist_val:.2f} "
                f"<= {self.ENTRY_NEAR_MICRO_DONE_DISTANCE:g}，按 0 处理，停止微调并开始看门"
            )
            return "ready"
        if self.entry_near_micro_adjust_attempts >= self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS:
            w.frame_log(
                f"[{phase_label}] 入门点近距离很慢微调已达上限 "
                f"{self.entry_near_micro_adjust_attempts}/{self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS}，"
                f"当前距离入门点 {target_loc} 仍为 {dist_val:.2f}，不能提前找门，舍弃当前入门点"
            )
            return "failed"

        ideal_angle = self.active_entry.get('direction') if self.active_entry else None
        refreshed_loc = self._get_current_location(w) or current_loc
        current_dir = ideal_angle if ideal_angle is not None else w.get_info('direction')
        target_angle = calculate_angle(refreshed_loc, target_loc)
        move_params = self._entry_near_micro_move_params(current_dir, target_angle)
        if move_params is None:
            w.frame_log(
                f"[{phase_label}] 入门点微调缺少有效方向/坐标: "
                f"current_loc={refreshed_loc}, target_loc={target_loc}, "
                f"current_dir={current_dir}, target_angle={target_angle}"
            )
            return "failed"

        direction, x_bias, y_bias, relative = move_params
        self.entry_near_micro_adjust_attempts += 1
        before_dist = get_distance(refreshed_loc, target_loc) if refreshed_loc is not None else dist_val
        desired_dist = max(0.2, float(before_dist))
        used_x_bias = x_bias
        used_y_bias = y_bias
        used_dura = self.ENTRY_NEAR_MICRO_DURA
        used_wait = self.ENTRY_NEAR_MICRO_WAIT
        distance_key = None
        if direction in ("left", "right"):
            used_x_bias, used_dura, used_wait, distance_key = get_adaptive_side_motion(
                direction,
                desired_dist,
                x_bias,
                self.ENTRY_NEAR_MICRO_DURA,
                self.ENTRY_NEAR_MICRO_WAIT,
            )
            used_y_bias = 0
        else:
            mode = "entry_back" if direction == "back" else "slow"
            used_y_bias, used_dura, used_wait, distance_key = get_adaptive_forward_motion(
                mode,
                desired_dist,
                y_bias,
                self.ENTRY_NEAR_MICRO_DURA,
                self.ENTRY_NEAR_MICRO_WAIT,
            )
            used_x_bias = 0
            if direction == "back":
                used_y_bias = abs(int(used_y_bias))

        w.frame_log(
                    f"[{phase_label}] 入门点微调 "
                    f"{self.entry_near_micro_adjust_attempts}/{self.ENTRY_NEAR_MICRO_MAX_ATTEMPTS}："
                    f"距离={dist_val:.2f}，方向={self._entry_micro_direction_label(direction)}，"
                    f"相对角={relative:.1f}°，摇杆=({used_x_bias},{used_y_bias})，"
                    f"dura={used_dura}，wait={used_wait}"
                )
        w.tap_single(
            '摇杆',
            x_bias=used_x_bias,
            y_bias=used_y_bias,
            dura=used_dura,
            wait=used_wait,
        )
        self._refresh_frame_and_handle_jump(w)
        after_loc = self._get_current_location(w) or refreshed_loc
        after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
        if direction in ("left", "right"):
            update_adaptive_side_motion(
                direction,
                desired_dist,
                before_dist,
                after_dist,
                used_x_bias,
                used_dura,
                used_wait,
            )
        else:
            mode = "entry_back" if direction == "back" else "slow"
            update_adaptive_forward_motion(
                mode,
                desired_dist,
                before_dist,
                after_dist,
                used_y_bias,
                used_dura,
                used_wait,
            )
        self.history_locations = []
        return "adjusting"

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
                w.frame_log('[Searching] 检测到长时间困在屋内 (house_scene=0)，启动兜底出房策略')
                self.house_exit_manager.reset()
                for _ in range(20):
                    if self._should_abort(w):
                        return
                    if self.house_exit_manager.process(w):
                        w.frame_log('[Searching] 兜底出房成功，切换到跑图阶段')
                        self.indoor_stuck_frames = 0
                        self.searching_number = 0
                        self.completed_houses.add(self.current_house_id)
                        self.current_house_id = None
                        self.status = "IDLE"
                        self._continue_searching_until_timer(w, '兜底出房成功')
                        return
                w.frame_log('[Searching] 兜底出房失败，强制重置状态切跑图')
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
                    self._reset_stuck_history_for_frame_wait("等待落地位置稳定")
                    w.frame_log('[Action] 停止并刷新等待稳定坐标')
                    self.stop_auto_forward(w)
                    self._refresh_frame_and_handle_jump(w)
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
            w.frame_log(
                f"[Searching] 锁定目标: {self.current_house_id} | "
                f"入口={self.active_entry['location']} | 距离={target_dist:.2f}"
            )
            self.history_locations = []  # 切换目标时清空历史

        target_loc = self.active_entry['location']
        dist = get_distance(current_loc, target_loc)

        # --- 快速导航：远距也使用角度校准 + 摇杆推进 ---
        if self.status == "FAST_NAV":
            w.frame_log('[Action] 对准入门点并摇杆推进')
            nav_scene_result = self._handle_nav_near_entry_scene_if_needed(w, "Nav快推段", "导航中")
            if nav_scene_result == "indoor":
                self._complete_current_house_search(w, "导航中贴门墙后进入房屋")
                return
            if nav_scene_result is not None:
                return

            if self._jump_forward_if_visible_near_house(w, "Nav快推段靠近房子"):
                return

            # 卡顿检测逻辑
            if self.update_and_check_stuck(current_loc):
                w.frame_log("[Nav] 检测到人物卡死，启动避障程序...")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                self.history_locations = []
                return

            if dist <= self.ENTRY_AUTO_FORWARD_DISTANCE:
                w.frame_log(f"[Nav] 进入摇杆分段导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.stop_auto_forward(w)
            aligned = self.align_direction(w, target_loc, threshold=10, max_steps=1)
            if not aligned:
                w.frame_log('[Action] 等待角度对齐')
                return

            before_dist = dist
            mode = self._entry_forward_mode(dist)
            y_bias, dura, wait = self._get_entry_move_params(dist)
            w.frame_log('[Action] ' + str((f'{self._entry_forward_mode_label(mode)}靠近入门点') or ('方向进入10度容差后，不再点击自动前进，而是按计算出的时长滑动摇杆推进')))
            w.tap_single('摇杆', y_bias=y_bias, dura=dura, wait=wait)
            self._refresh_frame_and_handle_jump(w)
            after_loc = self._get_current_location(w)
            after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
            update_adaptive_forward_motion(mode, before_dist, before_dist, after_dist, y_bias, dura, wait)

            self.handle_jump_logic(w)

        # --- 分段摇杆逼近 ---
        elif self.status == "PRECISE_NAV":
            w.frame_log('[Action] 精准推进入门点')
            nav_scene_result = self._handle_nav_near_entry_scene_if_needed(w, "Nav精推段", "导航中")
            if nav_scene_result == "indoor":
                self._complete_current_house_search(w, "导航中贴门墙后进入房屋")
                return
            if nav_scene_result is not None:
                return

            if self._jump_forward_if_visible_near_house(w, "Nav精推段靠近房子"):
                return

            if dist <= self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:
                near_result = self._handle_near_entry_point(w, current_loc, target_loc, dist, "Nav")
                if near_result == "adjusting":
                    self.handle_jump_logic(w)
                    return

                w.frame_log(f"[Nav] 当前距离入门点 {target_loc} 为 {dist:.2f}，近门处理结果={near_result}")
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
                w.frame_log(
                    f"[Nav] 当前距离入门点 {target_loc} 为 {dist:.2f} "
                    f"<= {self.ENTRY_ARRIVAL_DISTANCE:g}，已经完全到达入门点，开始找门并对准"
                )
                arrival_result = self._align_entry_door_after_arrival(w, "Nav")
                if arrival_result == "indoor":
                    self._complete_current_house_search(w, "自动开门直推进房成功")
                    return
                if arrival_result == "failed":
                    if self.active_entry:
                        self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                    return
                if arrival_result in {"aborted", "adjusting"}:
                    return
                self.status = "SCANNING"
                return

            self.stop_auto_forward(w)

            aligned = self.align_direction(w, target_loc, threshold=10, max_steps=1)
            if not aligned:
                self.history_locations = []
                return

            if self.update_and_check_stuck(current_loc):
                w.frame_log("[Nav] (Precise) 检测到人物卡死，启动避障程序...")
                if not self.execute_unstuck_logic(w, current_loc):
                    self.handle_failed_entry_logic(self.active_entry['direction'])
                    self.status = "IDLE"
                self.history_locations = []
                return

            before_dist = dist
            mode = self._entry_forward_mode(dist)
            y_bias, dura, wait = self._get_entry_move_params(dist)
            w.frame_log(
                f"[Nav] 当前距离入门点 {target_loc} 为 {dist:.2f}，"
                f"需要{self._entry_forward_mode_label(mode)}靠近入门点："
                f"y_bias={y_bias}, dura={dura}, wait={wait}"
            )
            w.tap_single('摇杆', y_bias=y_bias, dura=dura, wait=wait)
            self._refresh_frame_and_handle_jump(w)
            after_loc = self._get_current_location(w)
            after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
            update_adaptive_forward_motion(mode, before_dist, before_dist, after_dist, y_bias, dura, wait)
            self.handle_jump_logic(w)

        # --- 进门点扫描 ---
        elif self.status == "SCANNING":
            w.frame_log("[Scan] 到达点位，开始门检测...")
            ideal_angle = self.active_entry['direction']
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if self.check_and_lock_door(w):
                self.status = "VISUAL_APPROACH"
                return

            scan_offsets = [30, -30]
            found_door = False
            for offset in scan_offsets:
                target_angle = (ideal_angle + offset) % 360
                w.frame_log(f"[Scan] 尝试角度: {target_angle} (偏移 {offset})")
                self.align_direction_blocking(w, w.get_info('direction'), target_angle)
                self._refresh_frame_and_handle_jump(w)

                if self.check_and_lock_door(w):
                    found_door = True
                    self.status = "VISUAL_APPROACH"
                    break
                else:
                    w.frame_log(f"[Data] 角度 {target_angle} 未发现门，保存样本")
                    self.save_dataset_image(w.frame, f"no_door_offset_{offset}")

            if not found_door:
                w.frame_log("[Scan] 各角度均未发现门，舍弃当前入门点")
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
                    w.frame_log("[Visual] 丢失目标，回到扫描")
                    self.status = "SCANNING"
                    return

                if self._align_to_door_detection(
                    w,
                    door,
                    tolerance_px=self.ENTRY_DOOR_ALIGN_CENTER_THRESHOLD,
                    phase_label="VisualApproach",
                ):
                    w.frame_log("[Visual] 对齐完成，尝试交互")
                    self.status = "INTERACT"
                    break
            else:
                w.frame_log("[Visual] 多次视觉对齐失败，舍弃当前进门点")
                self.handle_failed_entry_logic(self.active_entry['direction'])
                self.status = "IDLE"
                return

        # --- 交互逻辑 ---
        elif self.status == "INTERACT":
            w.frame_log(f"[Interact] 尝试在 {self.current_house_id} 寻找交互按钮...")
            success = False
            for i in range(10):
                if self._should_abort(w):
                    return
                self._refresh_frame_and_handle_jump(w)

                # --- [修改 2] 交互前移时加入跳跃检测 ---
                # 原因：门前可能有台阶或门槛，不跳跃无法靠近
                if w.get_info('跳跃'):
                    w.frame_log("[Interact] 门前检测到障碍，尝试跳跃")
                    self.handle_jump_logic(w)  # 执行跳跃并前冲
                    self._refresh_frame_and_handle_jump(w)
                    continue  # 跳跃动作较大，跳过本次微调，直接进入下一次循环检查按钮
                # -----------------------------------

                if w.get_info('开门'):
                    w.frame_log('[Action] 点击开门')
                    w.click('开门')
                    time.sleep(1)
                    success = True
                    break
                if w.get_info('关门'):
                    w.frame_log("[Interact] 检测到关门按钮，表示门已打开，不点击关门，直接准备入户")
                    success = True
                    break
                if not self._advance_towards_entry_door(w):
                    w.frame_log("[Interact] 门目标丢失且兜底恢复失败")
                    break

            if success:
                w.frame_log("[Interact] 交互成功，准备入户")
                self.status = "FINAL_ENTRY"
            else:
                w.frame_log("[Interact] 交互失败，舍弃当前入门点")
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
                w.frame_log(f"[Entry] 调整至进门角度: {ideal_angle}")
                self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)
            w.frame_log("[Entry] 进门并确认 house_scene")
            if not self._push_until_entered_house(w):
                w.frame_log("[Entry] 多次推进后仍未进入房屋，舍弃当前进门点")
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
            w.frame_log(f"[Finish] 房屋 {self.current_house_id} 完成，累计已搜 {self.searching_number} 个")
            self._refresh_frame_and_handle_jump(w)
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

        try:
            first = (self.history_locations[0][0], self.history_locations[0][1])
            return all((loc[0], loc[1]) == first for loc in self.history_locations)
        except (TypeError, IndexError):
            return False

    def _get_stable_initial_location(self, current_loc):
        """落地后等待小地图坐标稳定，避免沿用跳伞前旧位置选错最近入口。"""
        loc = tuple(current_loc)
        if getattr(self, "landing_location_confirmed", False):
            self.landing_location_confirmed = False
            self.initial_location_samples = [loc]
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Searching] 落地位置已通过前推两帧确认: {loc}')
            return loc

        if self.initial_location_samples:
            prev = self.initial_location_samples[-1]
            jump_dist = get_distance(prev, loc)
            if jump_dist >= self.INITIAL_LOCATION_JUMP_RESET_DISTANCE:
                if getattr(self, "_frame_worker", None) is not None:
                    self._frame_worker.frame_log(f'[Searching] 落地坐标跳变 {jump_dist:.2f}，丢弃旧样本 prev={prev}, current={loc}')
                self.initial_location_samples = [loc]
                return None

        self.initial_location_samples.append(loc)
        if len(self.initial_location_samples) > self.INITIAL_LOCATION_MAX_SAMPLES:
            self.initial_location_samples.pop(0)

        if len(self.initial_location_samples) < self.INITIAL_LOCATION_MIN_SAMPLES:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Searching] 等待落地位置稳定 {len(self.initial_location_samples)}/{self.INITIAL_LOCATION_MIN_SAMPLES}: {loc}')
            return None

        x_coords = [item[0] for item in self.initial_location_samples]
        y_coords = [item[1] for item in self.initial_location_samples]
        spread = math.sqrt((max(x_coords) - min(x_coords)) ** 2 + (max(y_coords) - min(y_coords)) ** 2)
        if spread <= self.INITIAL_LOCATION_STABLE_DISTANCE:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Searching] 落地位置已稳定: {loc}, spread={spread:.2f}')
            return loc

        if len(self.initial_location_samples) >= self.INITIAL_LOCATION_MAX_SAMPLES:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Searching] 落地位置仍有波动，使用最新坐标: {loc}, spread={spread:.2f}')
            return loc

        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[Searching] 落地位置仍在刷新: latest={loc}, spread={spread:.2f}')
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
        current_loc = self._remember_valid_location(raw)
        if current_loc is not None:
            return current_loc
        return self._last_valid_location()

    def _push_until_entered_house(self, w: 'FrameWorker') -> bool:
        if self._get_house_scene(w) == 0:
            w.frame_log("[Entry] 已检测到 house_scene=0，确认已进屋")
            return True

        ideal_angle = self.active_entry['direction'] if self.active_entry else None
        for attempt in range(self.ENTRY_CONFIRM_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            if ideal_angle is not None:
                self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if attempt == 0:
                x_bias = 0
                w.frame_log(f"[Entry] 正前推进确认入屋 {attempt + 1}/{self.ENTRY_CONFIRM_MAX_ATTEMPTS}")
            else:
                x_bias = self.ENTRY_CONFIRM_SIDE_X_BIAS if attempt % 2 == 1 else -self.ENTRY_CONFIRM_SIDE_X_BIAS
                side = "右前方" if x_bias > 0 else "左前方"
                w.frame_log(f"[Entry] house_scene 仍非 0，向{side}推进确认入屋 {attempt + 1}/{self.ENTRY_CONFIRM_MAX_ATTEMPTS}")

            w.tap_single(
                '摇杆',
                x_bias=x_bias,
                y_bias=self.ENTRY_CONFIRM_FORWARD_Y_BIAS,
                dura=self.ENTRY_CONFIRM_FORWARD_DURA,
                wait=self.ENTRY_CONFIRM_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)
            time.sleep(0.2)

            house_scene = self._get_house_scene(w)
            if house_scene == 0:
                w.frame_log("[Entry] 推进后 house_scene=0，确认已进屋")
                return True

        return False

    def _is_entry_approach_status(self):
        return self.status in {"FAST_NAV", "PRECISE_NAV", "SCANNING", "VISUAL_APPROACH", "INTERACT", "FINAL_ENTRY"}

    def _backoff_and_recheck_house_scene(self, w: 'FrameWorker'):
        w.frame_log("[Unstuck] house_scene=indoor，可能是贴墙误判，先后退复核室内/室外")
        w.tap_single(
            '摇杆',
            y_bias=300,
            dura=self.ENTRY_WALL_BACKOFF_DURA,
            wait=self.ENTRY_WALL_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
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
        w.frame_log(
            f"[Unstuck] 绕障后恢复朝进门点推进: "
            f"dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        before_dist = dist
        w.tap_single('摇杆', y_bias=y_bias, dura=dura, wait=wait)
        self._refresh_frame_and_handle_jump(w)
        after_loc = self._get_current_location(w)
        after_dist = get_distance(after_loc, target_loc) if after_loc is not None else None
        update_adaptive_forward_motion(mode, before_dist, before_dist, after_dist, y_bias, dura, wait)

    def _choose_route_stuck_bypass_side_by_target_angle(self, w: 'FrameWorker', current_loc, target_loc):
        current_dir = w.get_info('direction')
        refreshed_loc = self._get_current_location(w) or current_loc
        target = check_location(target_loc)
        target_angle = calculate_angle(refreshed_loc, target) if refreshed_loc is not None and target is not None else None

        if current_dir is None or target_angle is None:
            fallback_side = self._choose_house_bypass_side(w)
            w.frame_log(
                f"[Unstuck] 缺少当前方向或目标坐标，无法按目的地角度选边，"
                f"回退使用房体空隙选择 side={fallback_side}"
            )
            return fallback_side, None, target_angle, current_dir

        try:
            current_dir_float = float(current_dir)
            target_angle_float = float(target_angle)
        except (TypeError, ValueError):
            fallback_side = self._choose_house_bypass_side(w)
            w.frame_log(
                f"[Unstuck] 当前方向/目标角度无效，回退使用房体空隙选择 side={fallback_side}: "
                f"current_dir={current_dir}, target_angle={target_angle}"
            )
            return fallback_side, None, target_angle, current_dir

        relative = (target_angle_float - current_dir_float + 540) % 360 - 180
        if abs(relative) <= 5:
            side = self._choose_house_bypass_side(w)
            w.frame_log(
                f"[Unstuck] 目的地基本在正前方 relative={relative:.1f}°，"
                f"按房体空隙选择 side={side}"
            )
            return side, relative, target_angle_float, current_dir_float

        side = "right" if relative > 0 else "left"
        side_label = "右" if side == "right" else "左"
        w.frame_log(
            f"[Unstuck] 结合目的地和当前方向选择绕房方向："
            f"current_dir={current_dir_float:.1f}, target_angle={target_angle_float:.1f}, "
            f"relative={relative:.1f}°，目的地在{side_label}侧，选择 side={side}"
        )
        return side, relative, target_angle_float, current_dir_float

    def _recover_route_stuck_by_side_forward(
        self,
        w: 'FrameWorker',
        current_loc,
        target_loc,
        backoff_first: bool = True,
    ) -> bool:
        self.stop_auto_forward(w)
        scene_before = self._get_house_scene(w)
        if scene_before != self.HOUSE_NEAR_WALL:
            w.frame_log(
                f"[Unstuck] 检测到卡住但 house_scene={scene_before}，"
                f"入口导航后拉避让只在 near_wall 触发，交给通用脱困"
            )
            return False

        outdoor_near_wall_result = self._recover_outdoor_near_wall(
            w,
            current_loc,
            target_loc,
            "Nav卡住恢复",
            "位置卡住时",
        )
        if outdoor_near_wall_result is not None:
            return True

        attempt = self._next_route_stuck_attempt(current_loc)

        backoff_y_bias, backoff_dura, backoff_wait = self._route_stuck_backoff_motion(attempt)
        w.frame_log(
            f"[Unstuck] 卡住且 house_scene=near_wall，只执行后拉避让，不再绕房: "
            f"attempt={attempt}, y_bias={backoff_y_bias}, dura={backoff_dura}, wait={backoff_wait}"
        )
        w.tap_single('摇杆', y_bias=backoff_y_bias, dura=backoff_dura, wait=backoff_wait)
        self._refresh_frame_and_handle_jump(w)
        if self._get_house_scene(w) == 0:
            loc_after_back = self._get_current_location(w) or current_loc
            return self._handle_indoor_during_entry_route(
                w,
                loc_after_back,
                "卡住后后退复核确认误入房",
            )

        loc_after_back = self._get_current_location(w)
        if loc_after_back is not None:
            current_loc = loc_after_back
        w.frame_log(f"[Unstuck] near_wall 后拉避让完成，从 {current_loc} 重新规划/继续入门点导航")
        self._mark_entry_door_strict_align_after_backoff()
        self.history_locations = []
        return True

    def _same_unstuck_point(self, origin, loc) -> bool:
        if loc is None:
            return True
        return get_distance(origin, loc) <= self.UNSTUCK_SAME_POINT_RADIUS

    def _unstuck_point_review_text(self, origin, loc) -> str:
        if loc is None:
            return f"origin={origin}, current=None, dist=None, same_point=True"
        dist = get_distance(origin, loc)
        return (
            f"origin={origin}, current={loc}, dist={dist:.2f}, "
            f"same_point={dist <= self.UNSTUCK_SAME_POINT_RADIUS}"
        )

    def _tap_unstuck_joystick(
        self,
        w: 'FrameWorker',
        current_loc,
        target_loc,
        branch: str,
        decision: str,
        action: str,
        x_bias: int,
        y_bias: int,
        wait: int,
    ):
        w.frame_log('[Action] ' + str((action) or (decision)))
        w.tap_single(
            '摇杆',
            x_bias=x_bias,
            y_bias=y_bias,
            dura=self.UNSTUCK_STEP_DURA,
            wait=wait,
        )
        self._refresh_frame_and_handle_jump(w)

    def _execute_u_unstuck_attempt(self, w: 'FrameWorker', current_loc, target_loc, side: str) -> bool:
        side_label = "左" if side == "left" else "右"
        side_bias = -self.UNSTUCK_SIDE_X_BIAS if side == "left" else self.UNSTUCK_SIDE_X_BIAS
        branch = "当前搜房分支：U型避障左U尝试" if side == "left" else "当前搜房分支：U型避障右U尝试"
        w.frame_log(f"[Unstuck] U型避障{side_label}U：后拉 -> {side_label}滑 -> 前冲")

        self._tap_unstuck_joystick(
            w,
            self._get_current_location(w) or current_loc,
            target_loc,
            branch,
            f"人物卡住，先后拉拉开空间，再尝试{side_label}侧U型绕开",
            "U型避障后拉",
            0,
            self.UNSTUCK_BACK_Y_BIAS,
            self.UNSTUCK_STEP_WAIT,
        )
        self._tap_unstuck_joystick(
            w,
            self._get_current_location(w) or current_loc,
            target_loc,
            branch,
            f"后拉后向{side_label}滑，寻找侧向出口",
            f"{side_label}滑试探",
            side_bias,
            0,
            self.UNSTUCK_STEP_WAIT,
        )
        self._tap_unstuck_joystick(
            w,
            self._get_current_location(w) or current_loc,
            target_loc,
            branch,
            f"{side_label}滑后向前冲，验证是否绕过障碍",
            "U型避障前冲",
            0,
            self.UNSTUCK_FORWARD_Y_BIAS,
            self.UNSTUCK_FORWARD_WAIT,
        )

        loc_after = self._get_current_location(w)
        review = self._unstuck_point_review_text(current_loc, loc_after)
        w.frame_log(f"[Unstuck] {side_label}U后卡点复核: {review}")
        return not self._same_unstuck_point(current_loc, loc_after)

    def _execute_large_area_unstuck(self, w: 'FrameWorker', current_loc, target_loc) -> bool:
        w.frame_log("[Unstuck] 当前搜房分支：大范围避障启动，左右U型仍卡住")
        w.tap_single(
            '视角',
            x_bias=self.UNSTUCK_LARGE_VIEW_X_BIAS,
            dura=self.UNSTUCK_LARGE_VIEW_DURA,
            wait=self.UNSTUCK_LARGE_VIEW_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)
        if not self.auto_forward:
            w.click('自动前进')
            self.auto_forward = True
        time.sleep(self.UNSTUCK_LARGE_AUTO_FORWARD_SECONDS)
        self._refresh_frame_and_handle_jump(w)

        for step in range(2):
            self._tap_unstuck_joystick(
                w,
                self._get_current_location(w) or current_loc,
                target_loc,
                "当前搜房分支：大范围避障左滑",
                f"自动前进后第 {step + 1}/2 次向左滑，扩大绕障半径",
                f"大范围左滑{step + 1}/2",
                -self.UNSTUCK_SIDE_X_BIAS,
                0,
                self.UNSTUCK_LARGE_SIDE_WAIT,
            )

        self.stop_auto_forward(w)
        if target_loc is not None:
            self.align_direction(w, target_loc)

        self._tap_unstuck_joystick(
            w,
            self._get_current_location(w) or current_loc,
            target_loc,
            "当前搜房分支：大范围避障后调整方向前冲",
            "大范围左滑后重新对准目标方向，向前冲出障碍区",
            "调整方向后前冲",
            0,
            self.UNSTUCK_FORWARD_Y_BIAS,
            self.UNSTUCK_FORWARD_WAIT,
        )
        loc_after = self._get_current_location(w)
        review = self._unstuck_point_review_text(current_loc, loc_after)
        w.frame_log('[Action] 复核大范围避障结果')
        return not self._same_unstuck_point(current_loc, loc_after)

    def execute_unstuck_logic(self, w: 'FrameWorker', current_loc):
        self.stop_auto_forward(w)
        target_loc = self.active_entry['location'] if self.active_entry else None

        def _safe_get_loc():
            return self._get_current_location(w)

        if self._get_house_scene(w) == 0:
            house_scene_after_backoff = self._backoff_and_recheck_house_scene(w)
            if house_scene_after_backoff != 0:
                w.frame_log("[Unstuck] 后退复核后已不判定为室内，按室外卡住绕障")
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

        w.frame_log("[Unstuck] 进入U型避障：先左U，仍在同一卡点再右U，仍卡住则大范围避障")
        if self._should_abort(w):
            return False
        if self._execute_u_unstuck_attempt(w, current_loc, target_loc, side="left"):
            return True

        if self._should_abort(w):
            return False
        w.frame_log("[Unstuck] 左U后仍在同一卡点，改为右U")
        if self._execute_u_unstuck_attempt(w, current_loc, target_loc, side="right"):
            return True

        if self._should_abort(w):
            return False
        w.frame_log("[Unstuck] 右U后仍在同一卡点，启动大范围避障")
        if self._execute_large_area_unstuck(w, current_loc, target_loc):
            return True

        w.frame_log("[Unstuck] 大范围避障后仍在同一卡点，放弃当前进门点")
        return False

    def handle_jump_logic(self, w: 'FrameWorker', reason: str = "检测到障碍") -> bool:
        if getattr(self, "_jump_forward_guard", False):
            return False
        if not w.get_info('跳跃'):
            self._jump_forward_wait_until_hidden = False
            return False
        if getattr(self, "_jump_forward_wait_until_hidden", False):
            return False

        w.frame_log(
            f"[Jump] {reason}，点击跳跃一次，等待 {self.JUMP_FORWARD_SETTLE_SECONDS:.1f}s 后轻微前推"
        )
        self._jump_forward_guard = True
        self._jump_forward_wait_until_hidden = True
        try:
            self.stop_auto_forward(w)
            w.click('跳跃')
            time.sleep(self.JUMP_FORWARD_SETTLE_SECONDS)
            w.tap_single(
                '摇杆',
                y_bias=self.JUMP_FORWARD_Y_BIAS,
                dura=self.JUMP_FORWARD_DURA,
                wait=self.JUMP_FORWARD_WAIT,
            )
            w.refresh_frame()
        finally:
            self._jump_forward_guard = False
        return True

    def _jump_forward_if_visible_near_house(self, w: 'FrameWorker', phase_label: str) -> bool:
        if not w.get_info('跳跃'):
            return False

        w.frame_log(f"[Jump] {phase_label} 检测到跳跃按钮，立即跳跃并前推")
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
                if self._is_temp_skipped_entry(entry):
                    continue
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
                if self._is_temp_skipped_entry(entry):
                    continue
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
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[Smart] 进门失败，临时跳过 {self.current_house_id}')
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
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Data] 已保存图片: {path}')
        except Exception as e:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[Data] 保存图片失败: {e}')

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
        w.frame_log('[Action] 阻塞式调整视角')
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
        cur_loc = self._normalize_location_value(location_raw)
        cur_dir = w.get_info('direction')
        if cur_loc is None or cur_dir is None:
            w.frame_log(
                f"[NavAlign] 无法计算目标角：raw_location={location_raw}，"
                f"current_loc={cur_loc}，current_dir={cur_dir}"
            )
            return False
        target_angle = calculate_angle(cur_loc, tar_loc)
        if target_angle is None:
            w.frame_log(
                f"[NavAlign] 目标角计算失败：raw_location={location_raw}，"
                f"current_loc={cur_loc}，target_loc={tar_loc}"
            )
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

    def _refresh_door_after_view_adjust(self, w, phase_label="DoorAlign"):
        w.frame_log(
            f"[{phase_label}] 视角/位置调整后等待最新帧再继续判断: "
            f"settle={self.ENTRY_DOOR_VIEW_ADJUST_REFRESH_SETTLE_SECONDS:.1f}s"
        )
        if self.ENTRY_DOOR_VIEW_ADJUST_REFRESH_SETTLE_SECONDS > 0:
            time.sleep(self.ENTRY_DOOR_VIEW_ADJUST_REFRESH_SETTLE_SECONDS)
        self._refresh_frame_and_handle_jump(w, f"{phase_label} 等待最新帧")
        return self.find_largest_door(w)

    def _align_to_door_detection(
        self,
        w,
        door,
        tolerance_px=80,
        phase_label="DoorAlign",
        return_state: bool = False,
        return_last_offset: bool = False,
    ):
        last_offset_real = None

        def finish(state: str):
            result = state if return_state else state == "aligned"
            if return_last_offset:
                return result, last_offset_real
            return result

        strict_after_backoff = self._consume_entry_door_strict_align_after_backoff()
        for step in range(self.ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS):
            offset_real, door_area_ratio, frame_w = self._get_visible_door_center_offset(w, door)
            if offset_real is None:
                return finish("lost")

            last_offset_real = offset_real

            if strict_after_backoff:
                center_threshold = self._get_strict_door_align_center_threshold(tolerance_px)
            else:
                center_threshold = self._get_door_align_center_threshold(tolerance_px)
            if abs(offset_real) <= center_threshold:
                w.frame_log(
                                    f"[{phase_label}] 门已对准：偏移={offset_real:.1f}px，"
                                    f"阈值={center_threshold}，面积占比={door_area_ratio}"
                                )
                return finish("aligned")

            adjust_val = int(offset_real * self.ENTRY_DOOR_ALIGN_STEP_RATIO)
            adjust_val = max(
                -self.ENTRY_DOOR_ALIGN_MAX_BIAS,
                min(self.ENTRY_DOOR_ALIGN_MAX_BIAS, adjust_val),
            )
            w.frame_log(
                            f"[{phase_label}] 对门 {step + 1}/{self.ENTRY_DOOR_FINAL_ALIGN_MAX_STEPS}："
                            f"偏移={offset_real:.1f}px，阈值={center_threshold}，"
                            f"视角x={adjust_val}"
                        )
            w.frame_log('[Action] 滑动视角微调门中心')
            w.tap_single(
                '视角',
                x_bias=adjust_val,
                dura=self.ENTRY_DOOR_ALIGN_DURA,
                wait=self.ENTRY_DOOR_ALIGN_WAIT,
            )
            refreshed = self._refresh_door_after_view_adjust(w, phase_label)
            if refreshed is None:
                return finish("lost")
            door = refreshed
        return finish("not_aligned")

    def _advance_towards_entry_door(self, w):
        door = self.find_largest_door(w)
        if door is not None:
            w.frame_log("[Interact] 前推前重新定位门并修正视角")
            self._align_to_door_detection(w, door)
            w.tap_single('摇杆', y_bias=-320, dura=320, wait=320)
            self._refresh_frame_and_handle_jump(w)
            return True

        w.frame_log("[Interact] 前推时门目标丢失，先给一次前推试错")
        w.tap_single('摇杆', y_bias=-260, dura=260, wait=450)
        self._refresh_frame_and_handle_jump(w)
        if w.get_info('开门') or w.get_info('关门') or self.find_largest_door(w):
            return True

        w.frame_log("[Interact] 试错后仍无门/交互按钮，后退重找门")
        w.tap_single('摇杆', y_bias=300, dura=380, wait=650)
        self._refresh_frame_and_handle_jump(w)
        recovered = self.find_largest_door(w)
        if recovered is None:
            return False

        self._align_to_door_detection(w, recovered)
        self._refresh_frame_and_handle_jump(w)
        return True

    def start_searching(self, w):
        if self._should_abort(w):
            return False

        self._start_house_search_timer()
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        w.frame_log("[Supply] 开始搜索入口房间物资")
        self.collect_supplies_in_room(w)
        if self._house_search_timed_out():
            return self._force_exit_after_search_timeout(w)
        if self._should_abort(w):
            self._clear_house_search_timer()
            return False

        self.house_entry_yaw = self.global_yaw
        a_door_abs_yaw = (self.house_entry_yaw + 180) % 360
        self.visited_sub_doors.append((a_door_abs_yaw, 999))
        w.frame_log("[搜房] 已记录入口A门方向，防止误入")

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
                w.frame_log('[Action] 点击开门')
                w.click('开门')
                time.sleep(1)
            time.sleep(0.5)
            w.frame_log('[Action] 前推穿门')
            w.tap_single('摇杆', y_bias=-400, dura=1000)
            self._refresh_frame_and_handle_jump(w)
            time.sleep(0.2)
            return True
        return False

    def _exit_house(self, w):

        w.frame_log("[Exit] 开始寻找出房路线")
        trusted_exit_route = False

        # 策略1：入口房间关闭门
        w.frame_log("[出口] 策略1：在入口房间寻找关闭的门")
        closed = self._find_closed_door_in_view(w)
        if not closed: closed = self._scan_for_closed_door(w, 360)
        if closed:
            rel_ang, _ = closed
            w.frame_log(f"[出口] 发现入口房间关闭门，推开离开！")
            if self._enter_closed_door(w, rel_ang, rush_time=1.2):
                self._refresh_frame_and_handle_jump(w)
                if self._get_house_scene(w) != 0:
                    trusted_exit_route = True
                    return

        # 策略2：进子房间找关闭门
        w.frame_log("[出口] 策略2：入口无关闭门，进入子房间寻找")
        if self._should_abort(w):
            return
        open_door = self._find_open_door_in_view(w)
        if not open_door: open_door = self._scan_for_open_door(w, 360)

        if open_door:
            rel_ang, bh = open_door
            w.frame_log(f"[出口] 进子房间找关闭门")
            self._pass_through_open_door(w, rel_ang, rush_time=0.8)
            self.room_yaw = 0.0

            closed_in_sub = self._find_closed_door_in_view(w)
            if not closed_in_sub: closed_in_sub = self._scan_for_closed_door(w, 360)

            if closed_in_sub:
                c_rel_ang, _ = closed_in_sub
                w.frame_log(f"[出口] 发现子房间关闭门，推开离开！")
                if self._enter_closed_door(w, c_rel_ang, rush_time=1.2):
                    self._refresh_frame_and_handle_jump(w)
                    if self._get_house_scene(w) != 0:
                        trusted_exit_route = True
                        return

            # 子房间没找到出口，退回入口房间
            w.frame_log("[出口] 子房间无关闭门，扇区快搜退回入口房间")
            exit_door = self._find_open_door_in_view(w, ignore_visited=True)
            if not exit_door: exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)
            if exit_door: self._pass_through_open_door(w, exit_door[0], rush_time=0.8)

        # 策略3：从入口A门原路返回
        w.frame_log("[出口] 从入口A门原路返回")
        if self._should_abort(w):
            return
        a_door = self._find_open_door_in_view(w, ignore_visited=True)
        if not a_door: a_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if a_door:
            w.frame_log("[出口] 发现A门，穿过离开！")
            self._pass_through_open_door(w, a_door[0], rush_time=1.2)
            trusted_exit_route = True
        else:
            w.frame_log("[出口] 极端情况：找不到A门，执行无门窗逃逸")
            self.house_exit_manager.reset()
            if self.house_exit_manager._escape_after_failed_exit_scan(w):
                return

        # 策略4：所有策略均失败，启动HouseExitManager兜底
        self._refresh_frame_and_handle_jump(w)
        if self._get_house_scene(w) == 0:
            w.frame_log("[出口] 策略3后仍在屋内，启动HouseExitManager兜底出房")
            self.house_exit_manager.reset()
            for _ in range(30):
                if self._should_abort(w):
                    return
                if self.house_exit_manager.process(w):
                    w.frame_log("[出口] 兜底出房成功")
                    return
            w.frame_log("[出口] 兜底出房也失败，强制前进冲出")
            for _ in range(5):
                w.tap_single('摇杆', y_bias=-500, dura=300)
                self._refresh_frame_and_handle_jump(w)
                time.sleep(0.3)
            if self._get_house_scene(w) != 0:
                self.house_exit_manager.reset()
                self.house_exit_manager.process(w)
        elif not trusted_exit_route:
            w.frame_log("[出口] 未经过明确门窗动作但已到屋外，执行二次确认")
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
                w.frame_log("  [搜房] 警告：未检测到门，尝试盲冲补救")
                break

            best = max(doors, key=lambda x: x[1])
            rel_ang, bh, _, det = best
            cx = (det[0] + det[2]) / 2
            offset_px = cx - center_x

            inf_w, inf_h = get_wh()
            frame_h = min(inf_w, inf_h)

            # 贴脸判定
            if bh > frame_h * 0.6:
                w.frame_log(f"  [搜房] 已贴脸门框(高度比:{bh / frame_h:.2f})，准备盲冲穿过！")
                break

            if abs(offset_px) > 5:
                self._turn(w, self.pixel_to_angle(cx) * 0.6)
                time.sleep(0.05)
                continue

            # 轨迹笔直，允许前进
            w.frame_log('[Action] 小步靠近门框')
            w.tap_single('摇杆', y_bias=-400, dura=300)
            self._refresh_frame_and_handle_jump(w)
            time.sleep(0.2)

        w.frame_log(f"  [鲁棒穿门] 执行盲冲，时间: {rush_time}s")
        move_ms = max(0, int(float(rush_time) * 1000))
        if move_ms > 0:
            w.tap_single('摇杆', y_bias=-500, dura=move_ms)
            self._refresh_frame_and_handle_jump(w)
            time.sleep(0.2)
        return True

    def _pass_through_open_door(self, w, rel_angle, rush_time=1.0):
        return self._robust_pass_through_door(w, rel_angle, [4], rush_time)

    def _enter_sub_room_and_collect(self, w, rel_angle, box_h):
        """子房间完整交互流程：记录特征 -> 鲁棒穿门 -> 战术搜物资 -> 扇区回搜退门"""
        w.frame_log("[SubRoom] 开始进入子房间")
        if self._should_stop_house_search(w):
            return False
        # 1. 记录进门绝对特征并去重
        abs_ang_enter = self._calc_abs_angle(rel_angle)
        self.visited_sub_doors.append((abs_ang_enter, box_h))

        # 2. 记录进门前的全局朝向，用于退出时计算反向扇区
        enter_yaw = self.global_yaw

        # 3. 穿门进入
        if not self._pass_through_open_door(w, rel_angle, rush_time=1.0):
            w.frame_log("[错误] 进入失败")
            return False

        self.room_yaw = 0.0  # 重置局部坐标系
        # 4. 搜集物资（内部自带战术复位）
        self._search_supplies(w)
        if self._should_stop_house_search(w):
            return False

        # 5. 扇区快搜退出门
        w.frame_log("[子房间] 搜集完毕，扇区快搜退出门...")
        target_exit_yaw = (enter_yaw + 180) % 360  # 计算进门背后的朝向
        # ignore_visited必须为True！因为进来的门已被标记，不忽略会看不到它
        exit_door = self._sector_scan_for_open_door(w, target_exit_yaw, sector_angle=120, ignore_visited=True)

        # 扇区兜底：如果扇区没找到，进行360全图扫描
        if not exit_door:
            w.frame_log("[子房间] 未找到，360度兜底扫描...")
            exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if exit_door:
            rel_exit, _ = exit_door
            w.frame_log(f"[子房间] 发现退出门，退出...")
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

        w.frame_log("[错误] 找不到退出门")
        return False

    def _sector_scan_for_open_door(self, w, center_yaw, sector_angle=120, ignore_visited=True):

        w.frame_log(f"  [搜房] 中心朝向:{center_yaw:.0f}°, 扫描范围:{sector_angle}°")
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
            w.frame_log(f"  发现物资（绝对{abs_ang:.1f}° 框高{box_h}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, box_h))
                return True
            return False

        # ---------- 方向序列 ----------
        w.frame_log("[Supply] 检查正前方物资")
        for _ in range(self.PICKUP_MAX_PER_DIRECTION):
            if self._should_stop_house_search(w) or not pickup_one_in_current_view(w):
                break
            time.sleep(0.2)
        if self._should_stop_house_search(w):
            return len(collected)

        w.frame_log("[Supply] 左转45°检查物资")
        self.turn_by_angle(w, -45, 300)
        player_yaw = (player_yaw - 45) % 360
        time.sleep(0.3)
        for _ in range(self.PICKUP_MAX_PER_DIRECTION):
            if self._should_stop_house_search(w) or not pickup_one_in_current_view(w):
                break
            time.sleep(0.2)
        if self._should_stop_house_search(w):
            return len(collected)

        w.frame_log("[Supply] 回正后右转45°检查物资")
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

        w.frame_log(f"[搜资] 结束，共拾取 {len(collected)} 个物资")
        self.turn_by_angle(w, -45, 300)
        w.frame_log("[Supply] 物资扫描结束，视角回正")
        return len(collected)

    def approach_and_pickup(self, w, initial_bbox, target_class, rel_ang):
        """
        小步靠近物资，让游戏内自动拾取接管。
        返回是否已经进入自动拾取可处理的范围。
        """
        if self._should_stop_house_search(w):
            return False

        if abs(rel_ang) > 2:
            w.frame_log('[Action] 调整视角对准物资')
            self.turn_by_angle(w, rel_ang, 200)
            time.sleep(1)

        for i in range(30):
            if self._should_stop_house_search(w):
                return False
            self._refresh_frame_and_handle_jump(w)
            scene = self._get_forward_scene(w)
            pick_menu = [obj for obj in scene if int(obj[5]) in [3]]

            if pick_menu:
                w.frame_log("检查到附近有物资")
                time.sleep(1)
                self._refresh_frame_and_handle_jump(w)
                return True

            w.frame_log(f"[Supply] 第 {i + 1}/30 步靠近物资")
            w.tap_single('摇杆', y_bias=-20, dura=300)
            time.sleep(0.5)
            self._refresh_frame_and_handle_jump(w)
            i += 1

            time.sleep(1)
        w.frame_log("当前已移动完成30步或者已经进入自动拾取范围")
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
        w.frame_log('[Action] 转动视角')
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
        w.frame_log("出子房间的门之前的角度{}".format(rel_ang))
        last_door = []
        # 调整角度
        if abs(rel_ang) > 2:

            if rel_ang > 0:
                w.frame_log("向右滑动调整视角，角度有偏差，添加5度的偏差")
                rel_ang += 5
            else:
                w.frame_log("角度微微调整")
                rel_ang += 6
            w.frame_log("出子房间的门的进行调整的角度{}".format(rel_ang))
            if abs(rel_ang) > 45:

                count = int(abs(rel_ang) / 45)
                count_ang = abs(rel_ang) % 45
                w.frame_log("角度大于45度，拆分成多次来旋转，拆分成{}次，是否有多余的{}".format(count, count_ang))
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
            w.frame_log('[Action] 小步靠近门')
            w.tap_single('摇杆', y_bias=-20, dura=300)
            i += 1
            self._refresh_frame_and_handle_jump(w)
            time.sleep(1)

            scene = self._get_forward_scene(w)
            open_door1 = [obj for obj in scene if int(obj[5]) in [4]]

            if open_door1:
                last_door = max(open_door1, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                # 当前还在画面中可以检测到
                inf_w, inf_h = get_wh()
                frame_w = max(inf_w, inf_h)
                center = frame_w / 2
                w.frame_log(
                    f"[Door] 靠近中仍识别到 {len(open_door1)} 个门目标，"
                    f"首个门中心={(open_door1[0][0] + open_door1[0][2]) / 2:.1f}，"
                    f"屏幕中心={center:.1f}"
                )
                # 移动靠近的过程中y1会逐渐减小，小于等于10 的时候，人物靠近门，这个时候暂停移动

            else:
                # 检测不到当前视角中的门的时候，当前已经靠近门边，直接往前走，可能会出现擦着墙边过的情况
                w.frame_log("当前已经靠近房间的门,微调角度处理。。。。")
                door = last_door
                if door:
                    self._align_to_door_detection(
                        w,
                        door,
                        tolerance_px=self.ENTRY_DOOR_ALIGN_CENTER_THRESHOLD,
                        phase_label="SearchDoorClose",
                    )
                    time.sleep(0.5)
                    time.sleep(5)

                if w.get_info('开门'):
                    w.click('开门')
                    time.sleep(1)
                w.frame_log("靠近门后，微调结束，直走进入房间。。。")
                w.tap_single('摇杆', y_bias=-400, dura=300)
                self._refresh_frame_and_handle_jump(w)
                w.tap_single('摇杆', y_bias=-400, dura=300)
                self._refresh_frame_and_handle_jump(w)
                w.frame_log("靠近门后往前移动俩步结束，不在往前移动")
                return True
        time.sleep(1)
        w.frame_log("当前已移动完成30步")
        return False

    def _collect_in_direction(self, w, avoid_door_abs=None):
        collected = []
        if self._should_stop_house_search(w):
            return
        supplies = self.new_targets_of_class(w, target_class=[1])

        if supplies:

            # 选择面积最大的
            best = max(supplies, key=lambda d: d[1])
            rel_ang = best[0]
            abs_ang = (self.room_yaw + rel_ang) % 360

            w.frame_log(f"  发现物资（绝对{abs_ang:.1f}° 框高{best[1]}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, best[1]))
        else:
            w.frame_log("当前子房间内未找到物资信息,继续下一次视角中获取物资...")
            time.sleep(1)

        if len(collected) == 2:
            w.frame_log("当前物资已拾满")

    def _search_supplies(self, w, avoid_door_abs=None):
        w.frame_log("[物资] 方向扫描...")
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
        w.frame_log(f"[Align] 开始对准目标，初始偏角={target_angle:.1f}°")
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
                w.frame_log("  [对准] 目标丢失")
                return
            best = max(targets, key=lambda x: x[1])
            target_angle = best[0]

    def _turn(self, w, delta):
        self.turn_by_angle(w, delta)
        self.room_yaw = (self.room_yaw + delta) % 360
        self.global_yaw = (self.global_yaw + delta) % 360


class HouseSceneSearchManager(HouseSearchManager):
    """基于 house_scene 五分类的新搜房入口逻辑。

    这套逻辑替换“到达进门点后如何进门”和“进房后如何旋转搜房”的流程，
    选点、导航和出房兜底仍复用旧 HouseSearchManager 的成熟能力，便于新旧逻辑并存和回滚。
    """

    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2
    HOUSE_NEAR_DOOR = 3
    HOUSE_NEAR_WALL = 4
    HOUSE_NEAR_ENTRY_SCENES = {HOUSE_NEAR_DOOR, HOUSE_NEAR_WALL}
    HOUSE_EXIT_SCENES = {HOUSE_OUTDOOR, HOUSE_ROOFTOP}

    R_CITY_FALLBACK_CENTER = (1036, 745)
    R_CITY_FALLBACK_LANDING_TARGET = (990, 757)
    R_CITY_DEFAULT_NEAR_DISTANCE = 50.0
    R_CITY_PRE_SEARCH_DISTANCE = 3.0
    R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE = 15.0
    R_CITY_ENTRY_MAP_NAV_DISTANCE = 15.0
    R_CITY_DEFAULT_HOUSE_ARRIVAL_DISTANCE = 2.0
    R_CITY_DEFAULT_EARLY_ENTRY_SCENE_DISTANCE = 5.0
    R_CITY_ROUTE_WAYPOINT_DISTANCE = 5.0
    R_CITY_ROUTE_PROJECTION_CORRIDOR = 12.0
    R_CITY_ROUTE_ENTRY_HANDOFF_DISTANCE = 15.0
    R_CITY_ROUTE_REPLAN_STUCK_CYCLES = 2
    R_CITY_FAILED_TARGET_LIMIT = 2
    R_CITY_NEAR_ENTRY_LOOK_DEGREES = 30
    ENTRY_AUTO_FORWARD_DISTANCE = 15.0
    ENTRY_COARSE_MOVE_DISTANCE = 10.0
    R_CITY_FORWARD_HOUSE_BYPASS_DISTANCE = 10.0
    R_CITY_BODY_ENTRY_DISTANCE = 4.0
    R_CITY_BODY_ENTRY_ALIGN_WAIT = 30

    STATUS_ROUTE_TO_R_CITY = "ROUTE_TO_R_CITY"
    ROTATE_RESULT_FINISHED = "finished"
    ROTATE_RESULT_EXITED = "exited"
    ROTATE_RESULT_FALLBACK_EXIT = "fallback_exit"
    ENTRY_DIRECTION_ALIGN_TOLERANCE = 5
    ENTRY_DIRECTION_ALIGN_MAX_STEPS = 8
    ENTRY_VISIBLE_DOOR_ALIGN_TOLERANCE_PX = 120
    ENTRY_ARRIVAL_DISTANCE = 0.0
    ENTRY_SIDE_ADJUST_MIN_DEGREES = 55
    ENTRY_SIDE_ADJUST_MAX_DEGREES = 125
    ENTRY_SIDE_ADJUST_X_BIAS = 230
    ENTRY_SIDE_ADJUST_BASE_DURA = 100
    ENTRY_SIDE_ADJUST_MAX_DURA = 420
    ENTRY_SIDE_ADJUST_WAIT_PAD = 240
    ENTRY_FORWARD_MAX_STEPS = 4
    ENTRY_FORWARD_STEP_Y_SCALE = 0.62
    ENTRY_FORWARD_STEP_MIN_DURA = 100
    ENTRY_FORWARD_STEP_WAIT_PAD = 220
    ENTRY_FORWARD_FAST_MODE = "fast"
    ENTRY_FORWARD_SLOW_MODE = "slow"

    ENTRY_APPROACH_MAX_STEPS = 4
    ENTRY_APPROACH_FORWARD_Y_BIAS = -280
    ENTRY_APPROACH_FORWARD_DURA = 360
    ENTRY_APPROACH_FORWARD_WAIT = 560

    SWEEP_STEP_MS = 100
    BUTTON_SWEEP_MAX_STEPS = 16
    BUTTON_SWEEP_X_BIAS = 240
    BUTTON_SWEEP_WAIT_PAD = 220

    ENTRY_SWEEP_MAX_STEPS = 14
    ENTRY_SWEEP_X_BIAS = 240
    ENTRY_SWEEP_Y_BIAS = -360
    ENTRY_SWEEP_WAIT_PAD = 260
    ENTRY_OPEN_SWEEP_BASE_DURA = 100
    ENTRY_OPEN_SWEEP_STEP_MS = 50
    ENTRY_OPEN_SWEEP_MAX_DURA = 750
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA = 250
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_MAX_DURA = 650
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_Y_BIAS = 360
    ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_WAIT = 360
    ENTRY_INDOOR_CONFIRM_FORWARD_Y_BIAS = -420
    ENTRY_INDOOR_CONFIRM_FORWARD_DURA = 650
    ENTRY_INDOOR_CONFIRM_FORWARD_WAIT = 850
    R_CITY_PRECISE_NAV_Y_BIAS = -200
    R_CITY_PRECISE_NAV_ALIGN_TOLERANCE = 10
    R_CITY_PRECISE_NAV_ALIGN_MAX_STEPS = 1
    R_CITY_PRECISE_NAV_MIN_WAIT = 300
    R_CITY_PRECISE_NAV_MAX_WAIT = 12000
    R_CITY_ENTRY_FORWARD_PROGRESS_MAX_STEPS = 12
    R_CITY_ENTRY_FAST_MIN_DURA = 520
    R_CITY_ENTRY_FAST_MIN_WAIT = 275
    R_CITY_ENTRY_SLOW_MIN_DURA = 460
    R_CITY_ENTRY_SLOW_MIN_WAIT = 325
    ENTRY_WINDOW_JUMP_SETTLE_SECONDS = 0.25
    OPEN_DOOR_SETTLE_SECONDS = 0.8
    ENTRY_OPEN_DOOR_SHORT_PUSH_RATIO = 1.0 / 3.0
    ENTRY_OPEN_DOOR_SHORT_PUSH_MIN_DURA = 120
    ENTRY_OPEN_DOOR_RELOCK_BACKOFF_Y_BIAS = 360
    ENTRY_OPEN_DOOR_RELOCK_MAX_ATTEMPTS = 5
    ENTRY_OPEN_DOOR_RELOCK_PUSH_WAITS = (300, 500, 600, 700, 800)
    ENTRY_OPEN_DOOR_RELOCK_PUSH_DURAS = (220, 300, 360, 420, 480)
    ENTRY_OPEN_DOOR_RELOCK_BACKOFF_WAITS = (500, 580, 650, 730, 800)
    ENTRY_OPEN_DOOR_RELOCK_BACKOFF_DURAS = (360, 420, 500, 580, 650)
    ENTRY_OPEN_DOOR_RELOCK_BACKOFF_DURA = 500
    ENTRY_OPEN_DOOR_RELOCK_BACKOFF_WAIT = 650
    ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_STEPS = 2
    ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_DURA = 260
    ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_WAIT = 420
    ENTRY_OPEN_DOOR_RELOCK_TOLERANCE_PX = 90

    ROTATE_SEARCH_MOVE_DURA = 1000
    ROTATE_SEARCH_MOVE_WAIT_PAD = 260
    ROTATE_SEARCH_X_BIAS = 200
    ROTATE_SEARCH_Y_BIAS = -290
    ROTATE_SEARCH_AUTO_TIMEOUT_SECONDS = 90
    ROTATE_SEARCH_AUTO_POLL_SECONDS = 0.35
    ROTATE_SEARCH_WALL_TURN_DEGREES = 60
    ROTATE_SEARCH_STUCK_SIMILAR_FRAMES = 5
    ROTATE_SEARCH_TURN_DEGREES = 90
    ROTATE_SEARCH_WALL_TURN_SEQUENCE = (90, 45, 22)
    ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS = 6
    ROTATE_SEARCH_HIT_SWITCH_COUNT = 6
    ROTATE_SEARCH_TURN_CORRECT_THRESHOLD = 6.0
    ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS = 2
    ROTATE_SEARCH_TURN_CORRECT_MAX_DEGREES = 45.0
    ROTATE_SEARCH_EXIT_FALLBACK_SWITCHES = 2
    ROTATE_SEARCH_MAX_STEPS = 80
    ROTATE_SEARCH_RECOVER_STEP_MS = 300
    ROTATE_SEARCH_RECOVER_MAX_MS = 1800
    ROTATE_SEARCH_RECOVER_X_BIAS = 330
    ROTATE_SEARCH_SWEEP_MOVE_DURA = 150
    ROTATE_SEARCH_SWEEP_MOVE_WAIT = 3000
    ROTATE_SEARCH_SWEEP_TURN_PX = 320
    ROTATE_SEARCH_SWEEP_TURN_DURA = 150
    ROTATE_SEARCH_SWEEP_TURN_WAIT = 3000
    ROTATE_EXIT_RECHECK_TURN_PX = 220
    ROTATE_SEARCH_SWEEP_CYCLES_PER_DIRECTION = 6
    ROTATE_FRAME_COMPARE_SIZE = (160, 90)
    ROTATE_FRAME_COMPARE_ROI = (0.18, 0.16, 0.82, 0.78)
    ROTATE_FRAME_MEAN_DIFF_THRESHOLD = 3.5
    ROTATE_FRAME_CHANGED_RATIO_THRESHOLD = 0.02
    ROTATE_FRAME_CHANGED_PIXEL_THRESHOLD = 12

    EXIT_DOOR_CLASS_IDS = {0, 4}
    EXIT_WINDOW_CLASS_IDS = {2}
    STONE_WALL_CLASS_IDS = {9}
    EXIT_SEARCH_MAX_STEPS = 36
    EXIT_SEARCH_LEFT_UP_DURA = 1000
    EXIT_SEARCH_TURN_DEGREES = 60
    EXIT_DOOR_SWEEP_MAX_STEPS = 14
    EXIT_WINDOW_ALIGN_MAX_STEPS = 6
    EXIT_WINDOW_ALIGN_TOLERANCE_DEGREES = 3.0
    EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES = 20
    EXIT_WINDOW_FORWARD_MAX_STEPS = 3
    EXIT_WINDOW_FORWARD_Y_BIAS = -360
    EXIT_WINDOW_FORWARD_DURA = 360
    EXIT_WINDOW_FORWARD_WAIT = 520
    EXIT_WINDOW_JUMP_FORWARD_Y_BIAS = -430
    EXIT_WINDOW_JUMP_FORWARD_DURA = 650
    EXIT_WINDOW_JUMP_FORWARD_WAIT = 850
    SCENE_EXIT_DOOR_SCAN_CYCLES = 1
    SCENE_EXIT_DOOR_SCAN_TURN_COUNT = 6
    SCENE_EXIT_SCAN_TURN_DEGREES = 60
    SCENE_EXIT_DOOR_ALIGN_TOLERANCE_PX = 120
    SCENE_EXIT_DOOR_FORWARD_Y_BIAS = -520
    SCENE_EXIT_DOOR_FORWARD_DURA = 5000
    SCENE_EXIT_DOOR_FORWARD_WAIT = 5200
    SCENE_EXIT_DOOR_WALL_SIDE_X_BIAS = 260
    SCENE_EXIT_DOOR_WALL_SIDE_Y_BIAS = -520
    SCENE_EXIT_DOOR_WALL_SIDE_DURA = 2000
    SCENE_EXIT_DOOR_WALL_SIDE_WAIT = 2200
    SCENE_EXIT_DOOR_LOST_SIDE_X_BIAS = 260
    SCENE_EXIT_DOOR_LOST_SIDE_Y_BIAS = -520
    SCENE_EXIT_DOOR_LOST_FIRST_SIDE_DURA = 2000
    SCENE_EXIT_DOOR_LOST_FIRST_SIDE_WAIT = 2200
    SCENE_EXIT_DOOR_LOST_SECOND_SIDE_DURA = 4000
    SCENE_EXIT_DOOR_LOST_SECOND_SIDE_WAIT = 4200
    SCENE_EXIT_WINDOW_SCAN_TURN_COUNT = 6
    SCENE_EXIT_WINDOW_APPROACH_MAX_STEPS = 3
    SCENE_EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES = 20
    SCENE_EXIT_WINDOW_GENTLE_FORWARD_Y_BIAS = -200
    SCENE_EXIT_WINDOW_GENTLE_FORWARD_DURA = 150
    SCENE_EXIT_WINDOW_GENTLE_FORWARD_WAIT = 300
    SCENE_EXIT_TARGET_ATTEMPT_LIMIT = 3
    SCENE_EXIT_EMERGENCY_TURN_DEGREES = 30
    SCENE_EXIT_EMERGENCY_MAX_STEPS = 60
    SCENE_EXIT_EMERGENCY_POLL_SECONDS = 0.3
    SCENE_EXIT_WALL_BACKOFF_Y_BIAS = 380
    SCENE_EXIT_WALL_BACKOFF_DURA = 420
    SCENE_EXIT_WALL_BACKOFF_WAIT = 620
    SCENE_EXIT_WALL_TURN_AROUND_PX = 700
    SCENE_EXIT_WALL_TURN_AROUND_DURA = 220
    SCENE_EXIT_WALL_TURN_AROUND_WAIT = 700

    WATER_FLOAT_DURA = 1000
    WATER_BACK_DURA = 650
    WATER_BACK_WAIT = 900
    WATER_SIDE_X_BIAS = 320
    WATER_SIDE_DURA = 900
    WATER_SIDE_WAIT = 1500
    WATER_FORWARD_Y_BIAS = -300
    WATER_FORWARD_DURA = 850
    WATER_FORWARD_WAIT = 1500
    WATER_SHORE_SIDE_SWIPES = 2
    WATER_ESCAPE_STUCK_FRAMES = 3
    WATER_ESCAPE_STUCK_DISTANCE = 0.6
    WATER_ESCAPE_SIDE_SWITCH_ATTEMPTS = 3
    WATER_ESCAPE_MAX_ATTEMPTS = 5
    WATER_FLOAT_RESET_MISSING_FRAMES = 5
    FORBIDDEN_ESCAPE_SEARCH_RADIUS = 120
    FORBIDDEN_ESCAPE_ARRIVAL_DISTANCE = 3.0
    FORBIDDEN_ESCAPE_FORWARD_DURA = 700
    FORBIDDEN_ESCAPE_FORWARD_WAIT = 900

    def __init__(self):
        super().__init__()
        self.r_city_config = {}
        self.r_city_center = self._load_r_city_center()
        self.r_city_landing_target = self._load_r_city_landing_target()
        self.r_city_near_distance = max(
            self.R_CITY_ENTRY_MAP_NAV_DISTANCE,
            self._load_r_city_threshold(
                "near_region_distance",
                self.R_CITY_DEFAULT_NEAR_DISTANCE,
            ),
        )
        self.r_city_house_arrival_distance = self._load_r_city_threshold(
            "house_arrival_distance",
            self.R_CITY_DEFAULT_HOUSE_ARRIVAL_DISTANCE,
        )
        self.r_city_early_entry_scene_distance = self._load_r_city_threshold(
            "early_entry_scene_distance",
            self.R_CITY_DEFAULT_EARLY_ENTRY_SCENE_DISTANCE,
        )
        self.r_city_side_candidate_ids = {}
        self.r_city_targets = self._build_r_city_targets()
        self.r_city_recovery_route_callback = None
        self.r_city_pre_search_target = None
        self.r_city_pre_search_distance = self.R_CITY_PRE_SEARCH_DISTANCE
        self.r_city_pre_search_route_callback = None
        self.r_city_entry_route_callback = None
        self.r_city_pre_search_completed = True
        self._reset_r_city_runtime()

    def reset(self):
        super().reset()
        self._reset_r_city_runtime()

    def _handle_nav_near_entry_scene_if_needed(
        self,
        w: "FrameWorker",
        phase_label: str,
        reason: str,
        current_loc=None,
        target_loc=None,
    ):
        """Confirm a visible door once after near-door/wall backoff, or skip it."""
        scene = self._get_house_scene(w)
        if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
            return super()._handle_nav_near_entry_scene_if_needed(
                w,
                phase_label,
                reason,
                current_loc=current_loc,
                target_loc=target_loc,
            )

        scene_label = "near_wall" if scene == self.HOUSE_NEAR_WALL else "near_door"
        w.frame_log(
            f"[RCitySearch] 当前入门点检测到{scene_label}，后拉后先确认门框，"
            "不再直接回到同一入口精推"
        )
        self.stop_auto_forward(w)
        w.tap_single(
            "摇杆",
            y_bias=self.ENTRY_DOOR_DIRECT_BACKOFF_Y_BIAS,
            dura=self.ENTRY_DOOR_DIRECT_BACKOFF_DURA,
            wait=self.ENTRY_DOOR_DIRECT_BACKOFF_WAIT,
        )
        # 这里只做视觉确认。不能走全局跳跃处理，否则“无门不前推”的
        # 分支可能因为跳跃按钮而意外产生一次前推。
        w.refresh_frame()
        if self._is_indoor(w):
            return "indoor"

        door = self.find_largest_door(w)
        if door is None:
            w.frame_log('[Action] 左转30度找门')
            self.turn_by_angle(w, -self.R_CITY_NEAR_ENTRY_LOOK_DEGREES)
            w.refresh_frame()
            if self._is_indoor(w):
                return "indoor"
            door = self.find_largest_door(w)

        if door is None:
            w.frame_log('[Action] 右转30度回正找门')
            self.turn_by_angle(w, self.R_CITY_NEAR_ENTRY_LOOK_DEGREES)
            w.refresh_frame()
            if self._is_indoor(w):
                return "indoor"
            door = self.find_largest_door(w)

        if door is not None:
            w.frame_log('[Action] 视觉对准门并前推')
            result = self._push_aligned_entry_door_and_check_indoor(w, phase_label, door)
            if result != "failed":
                return result
            failure_reason = "贴门墙后已确认门但视觉对门前推仍未进入室内"
        else:
            failure_reason = "贴门墙后正前方、左转30度和回正后均未看到门"

        w.frame_log(f"[RCitySearch] {failure_reason}，舍弃当前入门点并重新选点")
        self._mark_current_r_city_target_failed(failure_reason)
        self.status = "IDLE"
        return "failed"

    def searching_logic(self, w: "FrameWorker", current_loc, current_direction):
        if self._should_abort(w):
            return

        house_scene = self._get_house_scene(w)
        w.frame_log('[Action] 执行R城搜房主决策')
        if self._finish_callback_configured() and self._can_finish_searching(w):
            w.frame_log('[Action] 结束搜房并切跑图')
            self._continue_searching_until_timer(w, "R城搜房计时已满")
            return

        if self._is_in_water(w):
            w.frame_log('[Action] 执行水边脱困')
            if self._request_r_city_recovery_route(w, "落地后检测到人物在水里，切跑图阶段脱水并回R城"):
                return
            self._handle_water_escape(w, current_loc, current_direction)
            return

        if house_scene == self.HOUSE_INDOOR:
            w.frame_log('[Action] 切入当前房搜房')
            self._adopt_r_city_target_from_location(current_loc)
            self._handle_indoor_during_entry_route(w, current_loc, "导航/进门过程中检测到 indoor")
            return

        self.indoor_stuck_frames = 0

        if self.initial_target_pending:
            stable_loc = self._get_stable_initial_location(current_loc)
            if stable_loc is None:
                self._reset_stuck_history_for_frame_wait("R城等待落地初始位置稳定")
                w.frame_log('[Action] 停止并刷新等待稳定坐标')
                self.stop_auto_forward(w)
                self._refresh_frame_and_handle_jump(w)
                return
            current_loc = stable_loc
            self.initial_target_pending = False

        nearest_entry_loc = self._maintain_nearest_r_city_entry_reference(current_loc)
        if nearest_entry_loc is None:
            self._finish_r_city_searching(w, "R城没有可用入门点，无法维持最近入门点参考")
            return

        if self.status == self.STATUS_ROUTE_TO_R_CITY:
            self._handle_route_to_r_city(w, current_loc, current_direction)
            return

        if self.current_house_id is None:
            w.frame_log("[RCitySearch] 落地后直接从入门点列表选择最近目标，不再前往R城中转点")
            self._select_next_r_city_house(current_loc, current_direction)

            if not self.current_house_id:
                self._finish_r_city_searching(w, "R城房点已全部处理或均不可进入")
                return

            target_loc, target_dist = self._active_entry_point_distance(current_loc)
            route_name, route_loc, route_dist = self._active_entry_nearest_route_point(current_loc)
            route_context = ""
            if route_name is not None and route_loc is not None and route_dist is not None:
                route_context = f"，最近接入参考={route_name}{route_loc}，参考距离={route_dist:.2f}"
            w.frame_log(
                f"[RCitySearch] 落地后锁定最近入门点 {self.active_entry['location']}，"
                f"房屋={self.current_house_id}，入门方向={self.active_entry['direction']}，"
                f"真实入门点距离={target_dist:.2f}"
            )
            self.history_locations = []
            if (
                target_loc is not None
                and target_dist is not None
                and target_dist > self.R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE
            ):
                w.frame_log('[Action] 远距离转跑图阶段')
                if self._request_r_city_entry_route(w, target_loc, target_dist):
                    return
                if self._handle_far_entry_forbidden_escape(w, current_loc, current_direction, target_loc):
                    return
                if self._start_r_city_entry_map_navigation(w, current_loc, target_dist):
                    return
                self._defer_r_city_entry_until_nav_range(
                    w,
                    (
                        f"真实入门点 {target_loc} 仍在 {target_dist:.2f} 距离外，"
                        "跑图回调和地图路径都没有接管"
                    ),
                )
                return
            self.status = "PRECISE_NAV"

        target_loc = self.active_entry["location"]
        dist = get_distance(current_loc, target_loc)
        force_precise_entry_nav = (
            target_loc is not None
            and dist <= self.ENTRY_AUTO_FORWARD_DISTANCE
        )

        if force_precise_entry_nav:
            self.status = "PRECISE_NAV"
            self.r_city_route_path = []
            self.r_city_route_index = 0
            self.r_city_route_target = None
            self.r_city_route_stuck_cycles = 0
            self.forbidden_escape_region_anchor = None

        if not force_precise_entry_nav and not self._is_walkable(current_loc):
            forbidden_compare_entry = nearest_entry_loc
            if self._same_forbidden_region(current_loc, forbidden_compare_entry):
                w.frame_log(
                    f"[RCitySearch] 当前落点 {current_loc} 与最近入门点 {forbidden_compare_entry} 在同一不可通行区域，"
                    f"距离 {dist:.2f}，直接按入门点方向继续冲，不走最近安全点绕路"
                )
            elif self._is_reentered_forbidden_escape_region(current_loc):
                route_target = self.r_city_route_target or {}
                route_target_loc = route_target.get("approach_location") or route_target.get("location")
                w.frame_log(
                    f"[RCitySearch] 当前位置 {current_loc} 回到上次逃离的同一不可通行区域，"
                    "保留既有脱离后路线，不重复触发黑区脱离"
                )
            else:
                w.frame_log(
                    f"[RCitySearch] 当前落点 {current_loc} 不可通行，且与最近入门点 {forbidden_compare_entry} "
                    f"不是同一不可通行区域，先快速脱离当前黑区"
                )
                self._handle_forbidden_escape(
                    w,
                    current_loc,
                    current_direction,
                    target_loc=forbidden_compare_entry,
                )
                return

        route_waypoint = None
        following_route_waypoint = False
        if not force_precise_entry_nav:
            route_waypoint = self._current_r_city_route_waypoint(current_loc)
            following_route_waypoint = route_waypoint is not None
            if route_waypoint is not None:
                target_loc = route_waypoint
                dist = get_distance(current_loc, target_loc)
                w.frame_log('[Action] 沿规划路径前往入门点')
            elif self.r_city_route_path:
                self.r_city_route_path = []
                self.r_city_route_index = 0
                self.r_city_route_target = None
                self.forbidden_escape_region_anchor = None

        if (
            self.status in {"FAST_NAV", "PRECISE_NAV"}
            and not following_route_waypoint
            and not force_precise_entry_nav
        ):
            entry_loc, entry_dist = self._active_entry_point_distance(current_loc)
            route_name, route_loc, route_dist = self._active_entry_nearest_route_point(current_loc)
            route_context = ""
            if route_name is not None and route_loc is not None and route_dist is not None:
                route_context = f"，最近接入参考={route_name}{route_loc}，参考距离={route_dist:.2f}"
            if (
                entry_loc is not None
                and entry_dist is not None
                and entry_dist > self.R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE
            ):
                w.frame_log('[Action] 退回跑图阶段')
                if self._request_r_city_entry_route(w, entry_loc, entry_dist):
                    return
                if self._handle_far_entry_forbidden_escape(w, current_loc, current_direction, entry_loc):
                    return
                if self._start_r_city_entry_map_navigation(w, current_loc, entry_dist):
                    return
                self._defer_r_city_entry_until_nav_range(
                    w,
                    (
                        f"Nav复核时真实入门点 {entry_loc} 仍在 {entry_dist:.2f} 距离外，"
                        "跑图回调和地图路径都没有接管"
                    ),
                )
                return

        if self.status == "FAST_NAV":
            w.frame_log('[Action] 朝入门点摇杆推进')
            nav_scene_result = self._handle_nav_near_entry_scene_if_needed(w, "Nav快推段", "R城导航中")
            if nav_scene_result == "indoor":
                w.frame_log('[Action] 完成当前房搜房')
                self._complete_current_house_search(w, "R城导航中贴门墙后进入房屋")
                return
            if nav_scene_result is not None:
                return

            if self.update_and_check_stuck(current_loc):
                w.frame_log("[SceneSearch] 快速导航检测到卡住，启动避障")
                if not self._recover_r_city_navigation_stuck(w, current_loc):
                    self._mark_current_r_city_target_failed("快速导航卡住避障失败")
                    self.status = "IDLE"
                self.history_locations = []
                return

            if not following_route_waypoint and dist <= self.ENTRY_AUTO_FORWARD_DISTANCE:
                w.frame_log(f"[RCitySearch] 进入摇杆分段导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.stop_auto_forward(w)
            if self._move_precisely_to_entry_point(w, current_loc, target_loc, dist, phase_label="Nav快推段"):
                self.handle_jump_logic(w)
            return

        if self.status == "PRECISE_NAV":
            w.frame_log('[Action] 精准推进入门点')
            nav_scene_result = self._handle_nav_near_entry_scene_if_needed(
                w,
                "Nav精推段",
                "R城导航中",
                current_loc=current_loc,
                target_loc=target_loc,
            )
            if nav_scene_result == "indoor":
                w.frame_log('[Action] 完成当前房搜房')
                self._complete_current_house_search(w, "R城导航中贴门墙后进入房屋")
                return
            if nav_scene_result is not None:
                w.frame_log('[Action] 处理贴墙贴门干扰')
                return

            if dist <= self.ENTRY_NEAR_MICRO_ADJUST_DISTANCE:
                w.frame_log('[Action] 执行入门点近距建模流程')
                near_result = self._handle_near_entry_point(
                    w, current_loc, target_loc, dist, "RCityEntry"
                )
                if near_result == "indoor":
                    self._complete_current_house_search(w, "入门点+入门方向进门成功")
                    return
                if near_result in {"failed", "aborted"}:
                    self.status = "IDLE"
                    return
                return

            self.stop_auto_forward(w)
            if self._move_precisely_to_entry_point(w, current_loc, target_loc, dist, phase_label="Nav精推段"):
                self.handle_jump_logic(w)
                return

            if self.update_and_check_stuck(current_loc):
                w.frame_log("[SceneSearch] 精细导航检测到卡住，启动避障")
                if not self._recover_r_city_navigation_stuck(w, current_loc):
                    self._mark_current_r_city_target_failed("精细导航卡住避障失败")
                    self.status = "IDLE"
                self.history_locations = []
                return

            self.align_direction(
                w,
                target_loc,
                threshold=self.R_CITY_PRECISE_NAV_ALIGN_TOLERANCE,
                max_steps=self.R_CITY_PRECISE_NAV_ALIGN_MAX_STEPS,
            )
            y_bias, dura, wait = self._get_entry_move_params(dist)
            mode = self._entry_forward_mode(dist)
            w.frame_log(
                f"[SceneSearch] 当前距离入门点 {target_loc} 为 {dist:.2f}，"
                f"未完成精准推进，补一次{self._entry_forward_mode_label(mode)}："
                f"y_bias={y_bias}, dura={dura}, wait={wait}"
            )
            self._tap_entry_forward_with_learning(w, target_loc, dist, mode, y_bias, dura, wait)
            self.handle_jump_logic(w)
            return

    def _load_r_city_center(self):
        center = self.r_city_config.get("geometry", {}).get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return (int(center[0]), int(center[1]))
            except (TypeError, ValueError):
                pass
        return self.R_CITY_FALLBACK_CENTER

    def _load_r_city_landing_target(self):
        target = self.r_city_config.get("geometry", {}).get("landing_target")
        if isinstance(target, (list, tuple)) and len(target) >= 2:
            try:
                return (int(target[0]), int(target[1]))
            except (TypeError, ValueError):
                pass
        return self.R_CITY_FALLBACK_LANDING_TARGET

    def configure_r_city_landing_target(self, target):
        loc = self._location_tuple(target)
        if loc is not None:
            self.r_city_landing_target = loc

    def configure_r_city_pre_search_target(
        self,
        target,
        arrival_distance: float = R_CITY_PRE_SEARCH_DISTANCE,
    ):
        loc = self._location_tuple(target)
        if loc is None:
            return
        self.r_city_pre_search_target = loc
        self.r_city_pre_search_distance = max(0.0, float(arrival_distance))

    def _load_r_city_threshold(self, name: str, default: float) -> float:
        value = (
            self.r_city_config.get("geometry", {})
            .get("thresholds", {})
            .get(name, default)
        )
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _build_r_city_targets(self):
        targets = []
        for house_id in sorted(self.house_data, key=lambda value: int(value) if str(value).isdigit() else str(value)):
            entries = self.house_data.get(house_id) or []
            for entry_index, entry in enumerate(entries, start=1):
                loc = self._entry_location_tuple(entry)
                if loc is None or self._is_excluded_entry(entry):
                    continue
                approach_loc = self._resolve_r_city_approach_location(loc)
                try:
                    entry_direction = int(float(entry.get("direction"))) % 360
                except (TypeError, ValueError, AttributeError):
                    entry_direction = self._resolve_r_city_entry_direction(loc, None)

                target_id = f"house_{house_id}_entry_{entry_index}"
                targets.append(
                    {
                        "id": target_id,
                        "house_id": str(house_id),
                        "entry_index": entry_index,
                        "location": loc,
                        "approach_location": approach_loc,
                        "side": self._side_from_location(approach_loc),
                        "quality": "house_entry",
                        "entry_direction": entry_direction,
                        "nearest_existing_entry": entry,
                        "source": "house_entry",
                    }
                )
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[RCitySearch] 使用 house_entries_summary 入门点生成搜房目标: {len(targets)} 个')
        return targets

    @staticmethod
    def _r_city_target_house_id(target) -> Optional[str]:
        if not target:
            return None
        house_id = target.get("house_id") or target.get("existing_house_id") or target.get("id")
        return str(house_id) if house_id is not None else None

    def _r_city_target_entry_key(self, target):
        if not target:
            return None
        entry_loc = self._location_tuple(target.get("location"))
        if entry_loc is None:
            return None
        direction = target.get("entry_direction", target.get("direction"))
        try:
            entry_direction = int(float(direction)) % 360
        except (TypeError, ValueError):
            entry_direction = None
        return entry_loc, entry_direction

    def _is_r_city_target_completed(self, target) -> bool:
        target_id = target.get("id")
        house_id = self._r_city_target_house_id(target)
        entry_key = self._r_city_target_entry_key(target)
        completed_entry_keys = getattr(self, "r_city_completed_entry_keys", set())
        return (
            target_id in self.r_city_completed_targets
            or house_id in self.completed_houses
            or house_id in self.r_city_completed_targets
            or entry_key in completed_entry_keys
        )

    def _is_r_city_target_available(self, target) -> bool:
        if self._is_r_city_target_completed(target):
            return False
        entry_loc = self._location_tuple(target.get("location"))
        approach_loc = self._location_tuple(target.get("approach_location"))
        if entry_loc is not None and entry_loc in self.temp_skip_entries:
            return False
        if approach_loc is not None and approach_loc in self.temp_skip_entries:
            return False
        return self.r_city_failed_counts.get(target.get("id"), 0) < self.R_CITY_FAILED_TARGET_LIMIT

    def _mark_r_city_house_completed(self, house_id: Optional[str]):
        if house_id is None:
            return
        house_id = str(house_id)
        if not hasattr(self, "r_city_completed_entry_keys"):
            self.r_city_completed_entry_keys = set()
        self.completed_houses.add(house_id)
        for target in self.r_city_targets:
            if self._r_city_target_house_id(target) == house_id:
                self.r_city_completed_targets.add(target["id"])
                entry_key = self._r_city_target_entry_key(target)
                if entry_key is not None:
                    self.r_city_completed_entry_keys.add(entry_key)

    def _resolve_r_city_approach_location(self, loc):
        if self._is_walkable(loc):
            return loc
        finder = getattr(self.map_tool, "nearest_walkable_within_radius", None)
        if callable(finder):
            safe_point, _ = finder(loc, 12)
            safe_loc = self._location_tuple(safe_point)
            if safe_loc is not None:
                return safe_loc
        return loc

    def _resolve_r_city_entry_direction(self, approach_loc, configured_direction):
        try:
            return int(float(configured_direction)) % 360
        except (TypeError, ValueError):
            angle = calculate_angle(approach_loc, self.r_city_center)
            return int(angle or 0) % 360

    def _reset_r_city_runtime(self):
        self.r_city_completed_targets = set()
        self.r_city_completed_entry_keys = set()
        self.r_city_failed_counts = {}
        self.r_city_nearest_entry_location = None
        self.r_city_nearest_entry_id = None
        self.current_r_city_target = None
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.r_city_route_stuck_cycles = 0
        self.r_city_entry_large_backoff_count = 0
        self.r_city_side_probe_target = None
        self.r_city_side_probe_count = 0
        self.forbidden_escape_target = None
        self.forbidden_escape_region_anchor = None
        self.water_escape_side = None
        self.water_escape_side_attempts = 0
        self.water_escape_total_attempts = 0
        self.water_escape_last_loc = None
        self.water_escape_stuck_frames = 0
        self.water_float_pressed_in_episode = False
        self.water_float_missing_frames = self.WATER_FLOAT_RESET_MISSING_FRAMES
        self.r_city_pre_search_completed = True

    def _handle_r_city_pre_search_route(self, w: "FrameWorker", current_loc) -> bool:
        if self.r_city_pre_search_completed:
            return False

        target = self.r_city_pre_search_target
        if target is None:
            return False

        loc = self._location_tuple(current_loc)
        if loc is None:
            return False

        dist = get_distance(loc, target)
        if 0 <= dist <= self.r_city_pre_search_distance:
            self.r_city_pre_search_completed = True
            w.frame_log(
                f"[SceneSearch] 已到达R城搜房起点 {target}，"
                f"dist={dist:.2f} <= {self.r_city_pre_search_distance:.1f}，开始找房"
            )
            return False

        callback = self.r_city_pre_search_route_callback
        if not callable(callback):
            self.r_city_pre_search_completed = True
            w.frame_log("[SceneSearch] 未配置R城搜房起点跑图回调，跳过前置点直接找房")
            return False

        reason = (
            f"R城落地后先前往搜房起点 {target}，"
            f"dist={dist:.2f} > {self.r_city_pre_search_distance:.1f}"
        )
        w.frame_log(f"[SceneSearch] {reason}")
        self.stop_auto_forward(w)
        self.history_locations = []
        self.initial_location_samples = []
        self.initial_target_pending = True
        return bool(callback(w, target, reason, self.r_city_pre_search_distance))

    def _finish_callback_configured(self) -> bool:
        return getattr(self, "can_finish_callback", None) is not None

    @staticmethod
    def _location_tuple(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return (int(round(float(value[0]))), int(round(float(value[1]))))
        except (TypeError, ValueError):
            return None

    def _is_walkable(self, loc) -> bool:
        checker = getattr(self.map_tool, "is_walkable", None)
        if not callable(checker):
            return True
        try:
            return bool(checker(loc))
        except Exception as exc:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] 地图可通行检查失败，按可通行处理: {exc}')
            return True

    def _same_forbidden_region(self, current_loc, target_loc) -> bool:
        checker = getattr(self.map_tool, "same_forbidden_region", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(current_loc, target_loc))
        except Exception as exc:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] 不可通行区域连通判断失败，按不同区域处理: {exc}')
            return False

    def _line_only_crosses_target_forbidden_region(self, current_loc, target_loc) -> bool:
        checker = getattr(self.map_tool, "line_only_crosses_target_forbidden_region", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(current_loc, target_loc))
        except Exception as exc:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] 直线穿越目标黑区判断失败，按需要A*处理: {exc}')
            return False

    def _is_reentered_forbidden_escape_region(self, current_loc) -> bool:
        route_target = self.r_city_route_target or {}
        if route_target.get("id") != "forbidden_escape_to_entry" or not self.r_city_route_path:
            return False
        anchor = self._location_tuple(getattr(self, "forbidden_escape_region_anchor", None))
        if anchor is None:
            return False
        return self._same_forbidden_region(current_loc, anchor)

    def _distance_to_r_city(self, current_loc):
        loc = self._location_tuple(current_loc)
        if loc is None or not self.r_city_targets:
            return None, None
        best = min(
            self.r_city_targets,
            key=lambda item: get_distance(loc, item["location"]),
        )
        return get_distance(loc, best["location"]), best

    def _distance_to_r_city_landing_target(self, current_loc) -> Optional[float]:
        loc = self._location_tuple(current_loc)
        if loc is None or self.r_city_landing_target is None:
            return None
        return get_distance(loc, self.r_city_landing_target)

    def _request_r_city_recovery_route(self, w: "FrameWorker", reason: str) -> bool:
        callback = self.r_city_recovery_route_callback
        if not callable(callback):
            return False
        w.frame_log('[Action] 切换跑图恢复路线')
        self.stop_auto_forward(w)
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.forbidden_escape_region_anchor = None
        self.status = self.STATUS_ROUTE_TO_R_CITY
        callback(w, self.r_city_landing_target, reason)
        return True

    def _handle_far_entry_forbidden_escape(
        self,
        w: "FrameWorker",
        current_loc,
        current_direction,
        target_loc,
    ) -> bool:
        if target_loc is None:
            return False
        if self._is_walkable(current_loc):
            return False
        if self._same_forbidden_region(current_loc, target_loc):
            return False
        if self._is_reentered_forbidden_escape_region(current_loc):
            return False

        w.frame_log('[Action] 优先黑区脱离')
        self._handle_forbidden_escape(w, current_loc, current_direction, target_loc=target_loc)
        return True

    def _start_r_city_direct_entry_rush_if_clear(
        self,
        w: "FrameWorker",
        current_loc,
        target_loc,
        dist: float,
        phase_label: str,
    ) -> bool:
        start_loc = self._location_tuple(current_loc)
        target_loc = self._location_tuple(target_loc)
        if start_loc is None or target_loc is None:
            return False
        if not self._is_walkable(start_loc):
            return False
        if not self._line_only_crosses_target_forbidden_region(start_loc, target_loc):
            return False

        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.r_city_route_target = None
        self.forbidden_escape_region_anchor = None
        self.history_locations = []
        self.status = "FAST_NAV"
        w.frame_log(
            f"[RCitySearch] {phase_label}: start={start_loc}, entry={target_loc}, dist={dist:.2f}，"
            "直线只穿目标黑区，跳过A*直接冲入门点"
        )

        aligned = self.align_direction(w, target_loc, threshold=10, max_steps=1)
        if not aligned:
            self.stop_auto_forward(w)
            return True
        if not self.auto_forward:
            w.click("自动前进")
            self.auto_forward = True
        self.handle_jump_logic(w)
        return True

    def _start_r_city_entry_map_navigation(self, w: "FrameWorker", current_loc, route_dist: float) -> bool:
        if not self.active_entry:
            return False

        target_loc = self._location_tuple(self.active_entry.get("location"))
        route_target_loc = self._location_tuple(self.active_entry.get("approach_location")) or target_loc
        start_loc = self._location_tuple(current_loc)
        if target_loc is None or route_target_loc is None or start_loc is None:
            return False

        path = self._plan_path_safe(start_loc, route_target_loc)
        if not path:
            w.frame_log(
                f"[RCitySearch] 真实入门点距离 {route_dist:.2f} > {self.R_CITY_ENTRY_MAP_NAV_DISTANCE:.1f}，"
                "但 map_navigation 未能规划路径，本帧不进入Nav"
            )
            return False

        self.r_city_route_path = path
        self.r_city_route_index = 0
        self.r_city_route_target = {
            "id": self.active_entry.get("r_city_target_id", "nearest_entry"),
            "house_id": self.active_entry.get("house_id"),
            "location": target_loc,
            "approach_location": route_target_loc,
            "side": self._side_from_location(route_target_loc),
            "entry_direction": self.active_entry.get("direction", 0),
        }
        self.status = self.STATUS_ROUTE_TO_R_CITY
        self.forbidden_escape_region_anchor = None
        self.stop_auto_forward(w)
        self.history_locations = []
        self.r_city_route_stuck_cycles = 0
        w.frame_log(
                    f"[RCitySearch] 启动地图导航：当前位置={start_loc}，"
                    f"入口={target_loc}，接入点={route_target_loc}，路径点={len(path)}"
                )
        return True

    def _request_r_city_entry_route(self, w: "FrameWorker", target_loc, dist: float) -> bool:
        callback = self.r_city_entry_route_callback
        if not callable(callback):
            w.frame_log(
                f"[RCitySearch] 真实入门点 {target_loc} 距离 {dist:.2f} > "
                f"{self.R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE:.1f}，但未配置跑图回调，继续由搜房模块前往"
            )
            return False

        reason = (
            f"落地后真实入门点 {target_loc} 距离 {dist:.2f} > "
            f"{self.R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE:.1f}，先计入跑图时间前往入门点附近"
        )
        w.frame_log(f"[RCitySearch] {reason}")
        self.stop_auto_forward(w)
        self.history_locations = []
        self.initial_location_samples = []
        self.initial_target_pending = True
        approach_loc = None
        if self.active_entry:
            approach_loc = self._location_tuple(self.active_entry.get("approach_location"))
        if not callback(w, target_loc, reason, self.R_CITY_ENTRY_RUNNING_ROUTE_DISTANCE, approach_loc):
            return False
        w.frame_log(
            f"[RCitySearch] 跑图接管后继续锁定入门点 {target_loc}，"
            f"安全接入点={approach_loc}；黑区脱离/A*完成后只能回到该入门点，"
            "进入精推范围或该入口确认失败前不重新选点"
        )
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.forbidden_escape_region_anchor = None
        self.status = "IDLE"
        return True

    def _defer_r_city_entry_until_nav_range(self, w: "FrameWorker", reason: str) -> None:
        w.frame_log('[Action] 保持跑图/路线导航')
        self.stop_auto_forward(w)
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.r_city_route_stuck_cycles = 0
        self.forbidden_escape_region_anchor = None
        self.history_locations = []
        self.status = self.STATUS_ROUTE_TO_R_CITY

    def _side_from_location(self, loc):
        x, y = loc
        cx, cy = self.r_city_center
        dx = x - cx
        dy = y - cy
        if abs(dx) >= abs(dy):
            return "east" if dx > 0 else "west"
        return "south" if dy > 0 else "north"

    def _approach_side_from_current_location(self, current_loc):
        loc = self._location_tuple(current_loc)
        if loc is None:
            return "west"
        x, y = loc
        cx, cy = self.r_city_center
        dx = x - cx
        dy = y - cy
        if abs(dx) >= abs(dy):
            return "west" if dx < 0 else "east"
        return "north" if dy < 0 else "south"

    def _select_r_city_route_target(self, current_loc):
        side = self._approach_side_from_current_location(current_loc)
        side_ids = set(self.r_city_side_candidate_ids.get(side, []))
        candidates = [
            item for item in self.r_city_targets
            if item["id"] in side_ids or item["side"] == side
        ]
        if not candidates:
            candidates = list(self.r_city_targets)

        loc = self._location_tuple(current_loc)
        if loc is None:
            return candidates[0] if candidates else None

        reachable = []
        for target in candidates:
            path = self._plan_path_safe(loc, target["approach_location"])
            if path:
                reachable.append((self._path_length(path), target, path))
        if not reachable:
            for target in self.r_city_targets:
                path = self._plan_path_safe(loc, target["approach_location"])
                if path:
                    reachable.append((self._path_length(path), target, path))
        if reachable:
            _, target, path = min(reachable, key=lambda item: item[0])
            self.r_city_route_path = path
            self.r_city_route_index = 0
            return target

        return min(candidates, key=lambda item: get_distance(loc, item["approach_location"]))

    def _plan_path_safe(self, start_loc, target_loc):
        planner = getattr(self.map_tool, "plan_path", None)
        if not callable(planner):
            return [tuple(start_loc), tuple(target_loc)]
        try:
            path = planner(start_loc, target_loc)
        except Exception as exc:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] 规划路径失败: start={start_loc}, target={target_loc}, err={exc}')
            return []
        if not path:
            return []
        result = []
        for point in path:
            loc = self._location_tuple(point)
            if loc is not None:
                result.append(loc)
        return self._smooth_planned_path(result)

    def _smooth_planned_path(self, path):
        points = []
        for point in path or []:
            loc = self._location_tuple(point)
            if loc is not None:
                points.append(loc)
        line_checker = getattr(self.map_tool, "line_is_walkable", None)
        if not callable(line_checker):
            line_checker = getattr(self.map_tool, "_check_line_of_sight", None)
        if not callable(line_checker):
            return points
        return smooth_path_by_line_of_sight(points, line_checker)

    @staticmethod
    def _path_length(path) -> float:
        if len(path) < 2:
            return 0.0
        return sum(get_distance(path[i - 1], path[i]) for i in range(1, len(path)))

    def _handle_route_to_r_city(self, w: "FrameWorker", current_loc, current_direction):
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.status = self.STATUS_ROUTE_TO_R_CITY

        distance_to_r_city, nearest = self._distance_to_r_city(current_loc)
        w.frame_log('[Action] 执行R城路线导航')

        if not self.r_city_route_target or not self.r_city_route_path:
            self.r_city_route_target = self._select_r_city_route_target(current_loc)
            if not self.r_city_route_target and nearest:
                self.r_city_route_target = nearest
            if not self.r_city_route_target:
                w.frame_log('[Action] 结束R城搜房')
                self._finish_r_city_searching(w, "无法选择R城接入点")
                return
            if not self.r_city_route_path:
                self.r_city_route_path = self._plan_path_safe(
                    current_loc,
                    self.r_city_route_target["approach_location"],
                )
                self.r_city_route_index = 0
            w.frame_log(
                f"[RCityRoute] 规划去R城: side={self.r_city_route_target['side']}, "
                f"target={self.r_city_route_target['approach_location']}, "
                f"path_points={len(self.r_city_route_path)}"
            )

        if self._handoff_r_city_route_to_entry_nav_if_close(w, current_loc, current_direction):
            return

        waypoint = self._current_r_city_route_waypoint(current_loc)
        if waypoint is None:
            self.r_city_route_target = None
            self.r_city_route_path = []
            self.status = "IDLE"
            self.forbidden_escape_region_anchor = None
            return

        if self.update_and_check_stuck(current_loc):
            self.r_city_route_stuck_cycles += 1
            w.frame_log(
                f"[RCityRoute] 前往R城卡住 "
                f"{self.r_city_route_stuck_cycles}/{self.R_CITY_ROUTE_REPLAN_STUCK_CYCLES}"
            )
            self.stop_auto_forward(w)
            if self.r_city_route_stuck_cycles >= self.R_CITY_ROUTE_REPLAN_STUCK_CYCLES:
                self.r_city_route_target = None
                self.r_city_route_path = []
                self.r_city_route_stuck_cycles = 0
                self.forbidden_escape_region_anchor = None
            self.active_entry = {
                "location": waypoint,
                "direction": int(calculate_angle(current_loc, waypoint) or current_direction or 0),
            }
            self.execute_unstuck_logic(w, current_loc)
            self.active_entry = None
            self.history_locations = []
            return

        self.align_direction(w, waypoint)
        if not self.auto_forward:
            w.frame_log('[Action] 点击自动前进')
            w.click("自动前进")
            self.auto_forward = True
        self.handle_jump_logic(w)

    def _handoff_r_city_route_to_entry_nav_if_close(self, w: "FrameWorker", current_loc, current_direction) -> bool:
        loc = self._location_tuple(current_loc)
        if loc is None or not self.r_city_route_target:
            return False

        route_target = self.r_city_route_target
        entry_loc = self._location_tuple(route_target.get("location"))
        if entry_loc is None:
            return False

        trigger_dist = get_distance(loc, entry_loc)
        if trigger_dist > self.R_CITY_ROUTE_ENTRY_HANDOFF_DISTANCE:
            return False

        w.frame_log(
            f"[RCityRoute] 已接近真实入门点 {entry_loc} dist={trigger_dist:.2f}，"
            "停止路线自动前进，并按当前位置动态重选最近入门点"
        )
        self.stop_auto_forward(w)
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.r_city_route_stuck_cycles = 0
        self.forbidden_escape_region_anchor = None
        self.history_locations = []

        self._select_next_r_city_house(loc, current_direction)
        if not self.current_house_id:
            w.frame_log('[Action] 结束R城搜房')
            self._finish_r_city_searching(w, "R城路线交接后无可用入门点")
            return True

        target_loc = self.active_entry["location"]
        target_dist = get_distance(loc, target_loc)
        if target_dist > self.R_CITY_ROUTE_ENTRY_HANDOFF_DISTANCE:
            w.frame_log('[Action] 继续跑图接近入门点')
            if self._request_r_city_entry_route(w, target_loc, target_dist):
                return True
            if self._handle_far_entry_forbidden_escape(w, loc, current_direction, target_loc):
                return True
            if self._start_r_city_entry_map_navigation(w, loc, target_dist):
                return True
            self._defer_r_city_entry_until_nav_range(
                w,
                (
                    f"动态重选后的真实入门点 {target_loc} 仍在 {target_dist:.2f} 距离外，"
                    "跑图回调和地图路径都没有接管"
                ),
            )
            return True
        nav_phase = "Nav精推段"
        self.status = "PRECISE_NAV"
        w.frame_log('[Action] ' + str((f'切换{nav_phase}') or ('停止路线自动前进后，按人物实际位置重新选择最近入门点，再进入Nav')))
        return True

    def _current_r_city_route_waypoint(self, current_loc):
        if not self.r_city_route_path:
            return None
        loc = self._location_tuple(current_loc)
        if loc is None:
            return None
        self._advance_r_city_route_index_by_projection(loc)
        while self.r_city_route_index < len(self.r_city_route_path):
            waypoint = self.r_city_route_path[self.r_city_route_index]
            if get_distance(loc, waypoint) > self.R_CITY_ROUTE_WAYPOINT_DISTANCE:
                return waypoint
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCityRoute] 路径点 {waypoint} 已进入 {self.R_CITY_ROUTE_WAYPOINT_DISTANCE:g} 距离内，切换下一个路径点')
            self.r_city_route_index += 1
        return None

    def _advance_r_city_route_index_by_projection(self, current_loc):
        path = getattr(self, "r_city_route_path", None) or []
        start_index = int(getattr(self, "r_city_route_index", 0) or 0)
        if start_index < 0 or start_index >= len(path) - 1:
            return

        best_index = start_index
        best_ratio = 0.0
        best_dist = float("inf")
        for segment_index in range(start_index, len(path) - 1):
            ratio, dist = self._projection_ratio_and_distance(
                path[segment_index],
                path[segment_index + 1],
                current_loc,
            )
            if 0.0 <= ratio <= 1.0 and dist <= self.R_CITY_ROUTE_PROJECTION_CORRIDOR:
                best_index = segment_index + 1
                best_ratio = ratio
                best_dist = dist

        if best_index > start_index:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCityRoute] 当前位置 {current_loc} 已投影到后续路径线段，route_index {start_index}->{best_index}, ratio={best_ratio:.2f}, dist={best_dist:.2f}')
            self.r_city_route_index = best_index

    @staticmethod
    def _projection_ratio_and_distance(start, end, point):
        sx, sy = start
        ex, ey = end
        px, py = point
        vx = ex - sx
        vy = ey - sy
        length_sq = vx * vx + vy * vy
        if length_sq <= 0:
            return 0.0, get_distance(point, end)
        ratio = ((px - sx) * vx + (py - sy) * vy) / float(length_sq)
        clamped = max(0.0, min(1.0, ratio))
        proj_x = sx + vx * clamped
        proj_y = sy + vy * clamped
        return ratio, get_distance(point, (proj_x, proj_y))

    def _plan_route_from_escape_point_to_entry(self, safe_target, entry_target):
        safe_loc = self._location_tuple(safe_target)
        entry_loc = self._location_tuple(entry_target)
        if safe_loc is None or entry_loc is None:
            return []

        path = self._plan_path_safe(safe_loc, entry_loc)
        route_target = entry_loc
        if not path:
            approach_loc = self._resolve_r_city_approach_location(entry_loc)
            if approach_loc != entry_loc:
                path = self._plan_path_safe(safe_loc, approach_loc)
                route_target = approach_loc

        if not path:
            self.r_city_route_path = []
            self.r_city_route_index = 0
            self.r_city_route_target = None
            self.forbidden_escape_region_anchor = None
            return []

        self.r_city_route_path = path
        self.r_city_route_index = 0
        self.r_city_route_target = {
            "id": "forbidden_escape_to_entry",
            "approach_location": route_target,
            "location": entry_loc,
            "side": self._approach_side_from_current_location(safe_loc),
        }
        return path

    def _handle_forbidden_escape(self, w: "FrameWorker", current_loc, current_direction, target_loc=None):
        target_loc = self._location_tuple(target_loc)
        if target_loc is not None and not self._same_forbidden_region(current_loc, target_loc):
            dist = get_distance(current_loc, target_loc)
            w.frame_log(
                f"[RCityRoute] 当前不可通行，入门点 {target_loc} 不在同一不可通行区域，"
                f"距离 {dist:.2f}，先找最近可通行点脱离当前黑区"
            )

        safe_target = self.forbidden_escape_target
        if safe_target is None or get_distance(current_loc, safe_target) <= self.FORBIDDEN_ESCAPE_ARRIVAL_DISTANCE:
            finder = getattr(self.map_tool, "nearest_walkable_within_radius", None)
            if callable(finder):
                safe_target, _ = finder(current_loc, self.FORBIDDEN_ESCAPE_SEARCH_RADIUS)
            safe_target = self._location_tuple(safe_target)
            self.forbidden_escape_target = safe_target

        if safe_target is None:
            w.frame_log("[RCityRoute] 当前不可通行且未找到安全点，短后退后刷新")
            self.stop_auto_forward(w)
            w.frame_log('[Action] 短后退刷新')
            w.tap_single("摇杆", y_bias=300, dura=450, wait=850)
            self._refresh_frame_and_handle_jump(w)
            return

        planned_entry_path = self._plan_route_from_escape_point_to_entry(safe_target, target_loc)
        if planned_entry_path:
            self.forbidden_escape_region_anchor = self._location_tuple(current_loc)
        if target_loc is not None:
            pass

        w.frame_log(f"[RCityRoute] 当前不在可通行区域，先脱离到安全点 {safe_target}")
        self.stop_auto_forward(w)
        self.align_direction(w, safe_target)
        w.frame_log('[Action] 启动自动前进到安全点')
        if not self.auto_forward:
            w.click("自动前进")
            self.auto_forward = True
        self._refresh_frame_and_handle_jump(w)

    def _is_in_water(self, w: "FrameWorker") -> bool:
        visible = bool(w.get_info("上浮"))
        self._update_water_float_state(visible)
        return visible

    def _update_water_float_state(self, visible: bool):
        if visible:
            self.water_float_missing_frames = 0
            return
        missing_frames = getattr(
            self,
            "water_float_missing_frames",
            self.WATER_FLOAT_RESET_MISSING_FRAMES,
        )
        self.water_float_missing_frames = missing_frames + 1
        if self.water_float_missing_frames >= self.WATER_FLOAT_RESET_MISSING_FRAMES:
            self.water_float_pressed_in_episode = False

    def _current_navigation_target_location(self, current_loc=None):
        if self.current_r_city_target:
            return self.current_r_city_target["approach_location"]
        if self.r_city_route_target:
            return self.r_city_route_target["approach_location"]
        if current_loc is not None:
            _, nearest = self._distance_to_r_city(current_loc)
            if nearest:
                return nearest["approach_location"]
        return self.r_city_center

    def _handle_water_escape(self, w: "FrameWorker", current_loc, current_direction):
        target_loc = self._current_navigation_target_location(current_loc)

        float_visible = self._is_in_water(w)
        should_press_float = float_visible and not getattr(
            self,
            "water_float_pressed_in_episode",
            False,
        )
        if should_press_float:
            self.stop_auto_forward(w)
        before_loc = self._location_tuple(current_loc)
        w.frame_log(
            f"[RCityWater] 落水/水边受阻，先上浮并自动前进脱困 "
            f"attempt={self.water_escape_total_attempts + 1}, target={target_loc}"
        )

        if should_press_float:
            w.click("上浮", duration_ms=self.WATER_FLOAT_DURA)
            self.water_float_pressed_in_episode = True
            self._refresh_frame_and_handle_jump(w, "水区长按上浮后")
        elif float_visible:
            w.frame_log("[RCityWater] 本轮落水已长按过上浮，继续保持自动前进，不重复点击上浮")

        if not self._is_in_water(w):
            w.frame_log("[RCityWater] 上浮按钮已消失，认为已离开水域，回到陆地搜房逻辑")
            self._reset_water_escape_progress()
            return

        self.align_direction(w, target_loc, threshold=5, max_steps=1)
        w.frame_log('[Action] 点击自动前进')
        if not self.auto_forward:
            w.click("自动前进")
            self.auto_forward = True

        self._refresh_frame_and_handle_jump(w, "水区自动前进后")
        if not self._is_in_water(w):
            w.frame_log("[RCityWater] 自动前进后上浮按钮消失，回到陆地搜房逻辑")
            self._reset_water_escape_progress()
            return

        after_loc = self._get_current_location(w) or before_loc
        if self._record_water_escape_attempt(before_loc, after_loc):
            self._handle_water_shore_obstacle(w, after_loc, target_loc, current_direction)

    def _reset_water_escape_progress(self):
        self.water_escape_side = None
        self.water_escape_side_attempts = 0
        self.water_escape_stuck_frames = 0
        self.water_escape_last_loc = None

    def _handle_water_shore_obstacle(self, w: "FrameWorker", current_loc, target_loc, current_direction):
        side = self._choose_water_escape_side(current_loc, target_loc, current_direction)
        side_label = self._side_label(side)
        x_bias = -self.WATER_SIDE_X_BIAS if side == "left" else self.WATER_SIDE_X_BIAS

        w.frame_log(f"[RCityWater] 连续多帧无位移，沿{side_label}侧执行岸边避障")
        self.stop_auto_forward(w)
        for _ in range(self.WATER_SHORE_SIDE_SWIPES):
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                dura=self.WATER_SIDE_DURA,
                wait=self.WATER_SIDE_WAIT,
            )
            self._refresh_frame_and_handle_jump(w, "水区岸边侧滑后")
            if not self._is_in_water(w):
                w.frame_log("[RCityWater] 岸边侧滑后上浮按钮消失，回到陆地搜房逻辑")
                self._reset_water_escape_progress()
                return

        self.align_direction(w, target_loc, threshold=5, max_steps=1)
        if not self.auto_forward:
            w.click("自动前进")
            self.auto_forward = True
        self.water_escape_stuck_frames = 0
        self.water_escape_last_loc = self._get_current_location(w) or current_loc

    def _choose_water_escape_side(self, current_loc, target_loc, current_direction):
        if (
            self.water_escape_side
            and self.water_escape_side_attempts < self.WATER_ESCAPE_SIDE_SWITCH_ATTEMPTS
        ):
            return self.water_escape_side

        if self.water_escape_side and self.water_escape_side_attempts >= self.WATER_ESCAPE_SIDE_SWITCH_ATTEMPTS:
            self.water_escape_side = "right" if self.water_escape_side == "left" else "left"
            self.water_escape_side_attempts = 0
            return self.water_escape_side

        target_angle = calculate_angle(current_loc, target_loc)
        turn_dir, _, _ = calculate_move_count(current_direction, target_angle)
        self.water_escape_side = "left" if turn_dir == "left" else "right"
        self.water_escape_side_attempts = 0
        return self.water_escape_side

    def _record_water_escape_attempt(self, before_loc, after_loc):
        before = self._location_tuple(before_loc)
        after = self._location_tuple(after_loc)
        reference = self.water_escape_last_loc or before
        moved = 0.0 if reference is None or after is None else get_distance(reference, after)
        self.water_escape_total_attempts += 1
        self.water_escape_last_loc = after
        if moved >= self.WATER_ESCAPE_STUCK_DISTANCE:
            self.water_escape_stuck_frames = 0
            self.water_escape_side_attempts = 0
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCityWater] 水中自动前进反馈: moved={moved:.2f}, stuck_frames=0')
            return False

        self.water_escape_stuck_frames += 1
        self.water_escape_side_attempts += 1
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f'[RCityWater] 水中自动前进反馈: moved={moved:.2f}, stuck_frames={self.water_escape_stuck_frames}, side={self.water_escape_side}')
        if self.water_escape_stuck_frames >= self.WATER_ESCAPE_STUCK_FRAMES:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log('[RCityWater] 连续3帧几乎无位移，触发岸边侧滑避障')
            return True
        return False

    def _select_next_r_city_house(self, current_loc, current_direction):
        loc = self._location_tuple(current_loc)
        candidates = [
            item for item in self.r_city_targets
            if self._is_r_city_target_available(item)
        ]
        if not candidates or loc is None:
            self.current_house_id = None
            self.current_r_city_target = None
            self.active_entry = None
            return

        target = min(candidates, key=lambda item: get_distance(loc, item["location"]))
        self._lock_r_city_target(target)

    def _maintain_nearest_r_city_entry_reference(self, current_loc):
        """Refresh the nearest available entry used for forbidden-region checks.

        This is intentionally separate from ``active_entry``: a route already in
        progress can finish coherently, while every black-region decision still
        compares the player with the entry that is nearest at this exact position.
        """
        loc = self._location_tuple(current_loc)
        if loc is None:
            return getattr(self, "r_city_nearest_entry_location", None)

        candidates = [
            item for item in self.r_city_targets
            if self._is_r_city_target_available(item)
        ]
        if not candidates:
            self.r_city_nearest_entry_location = None
            self.r_city_nearest_entry_id = None
            return None

        nearest = min(candidates, key=lambda item: get_distance(loc, item["location"]))
        nearest_loc = self._location_tuple(nearest.get("location"))
        if nearest_loc is None:
            self.r_city_nearest_entry_location = None
            self.r_city_nearest_entry_id = None
            return None

        previous_id = getattr(self, "r_city_nearest_entry_id", None)
        self.r_city_nearest_entry_location = nearest_loc
        self.r_city_nearest_entry_id = nearest.get("id")
        if previous_id != self.r_city_nearest_entry_id:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] 更新最近入门点参考：id={self.r_city_nearest_entry_id}, location={nearest_loc}；不可通行区域将与该点比较')
        return nearest_loc

    def get_live_r_city_entry_for_route(self, current_loc):
        """Return the live nearest entry while the unfinished search is in running mode.

        The forced running route uses this callback before every black-region,
        direct-line, or A* decision.  Keep ``active_entry`` synchronized too,
        so returning to the search stage enters Nav for the same live target.
        """
        nearest_loc = self._maintain_nearest_r_city_entry_reference(current_loc)
        nearest_id = getattr(self, "r_city_nearest_entry_id", None)
        if nearest_loc is None or nearest_id is None:
            return None

        target = next(
            (
                item for item in self.r_city_targets
                if item.get("id") == nearest_id and self._is_r_city_target_available(item)
            ),
            None,
        )
        if target is None:
            return None

        active_id = (self.active_entry or {}).get("r_city_target_id")
        if active_id != target.get("id"):
            self._lock_r_city_target(target)
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f"[RCitySearch] 跑图接管期间同步动态最近入门点：id={target['id']}，location={target['location']}")

        return {
            "id": target["id"],
            "location": target["location"],
            "approach_location": target["approach_location"],
        }

    def _active_entry_nearest_route_point(self, current_loc):
        loc = self._location_tuple(current_loc)
        if loc is None or not self.active_entry:
            return None, None, None

        distances = []
        entry_loc = self._location_tuple(self.active_entry.get("location"))
        approach_loc = self._location_tuple(self.active_entry.get("approach_location"))
        if entry_loc is not None:
            distances.append(("入门点", entry_loc, get_distance(loc, entry_loc)))
        if approach_loc is not None:
            distances.append(("入门点安全点", approach_loc, get_distance(loc, approach_loc)))
        if not distances:
            return None, None, None
        return min(distances, key=lambda item: item[2])

    def _active_entry_point_distance(self, current_loc):
        loc = self._location_tuple(current_loc)
        if loc is None or not self.active_entry:
            return None, None
        entry_loc = self._location_tuple(self.active_entry.get("location"))
        if entry_loc is None:
            return None, None
        return entry_loc, get_distance(loc, entry_loc)

    def _lock_r_city_target(self, target):
        self.current_r_city_target = target
        self.current_house_id = self._r_city_target_house_id(target)
        self.active_entry = {
            "location": target["location"],
            "approach_location": target["approach_location"],
            "direction": target["entry_direction"],
            "r_city_target_id": target["id"],
            "house_id": self.current_house_id,
        }
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.forbidden_escape_region_anchor = None
        self.r_city_entry_large_backoff_count = 0
        self.r_city_side_probe_target = None
        self.r_city_side_probe_count = 0

    def _mark_current_r_city_target_failed(self, reason: str):
        if self.current_r_city_target:
            target_id = self.current_r_city_target["id"]
            self.r_city_failed_counts[target_id] = self.r_city_failed_counts.get(target_id, 0) + 1
            entry_loc = self.current_r_city_target.get("location")
            approach_loc = self.current_r_city_target.get("approach_location")
            if entry_loc is not None:
                self.temp_skip_entries.add(tuple(entry_loc))
            if approach_loc is not None:
                self.temp_skip_entries.add(tuple(approach_loc))
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] {reason}: 跳过当前入门点 {target_id} entry={entry_loc}, approach={approach_loc} fail={self.r_city_failed_counts[target_id]}/{self.R_CITY_FAILED_TARGET_LIMIT}；同房其他入门点继续保留')
        elif self.current_house_id:
            if getattr(self, "_frame_worker", None) is not None:
                self._frame_worker.frame_log(f'[RCitySearch] {reason}: {self.current_house_id}')
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.history_locations = []
        self._reset_entry_near_micro_adjust()
        self._reset_route_stuck_bypass()

    def _mark_current_entry_failed(self, reason: str):
        if self.current_r_city_target:
            self._mark_current_r_city_target_failed(reason)
            return
        super()._mark_current_entry_failed(reason)

    def _adopt_r_city_target_from_location(self, current_loc):
        if self.current_r_city_target or self.current_house_id:
            return
        loc = self._location_tuple(current_loc)
        if loc is None:
            return
        candidates = [
            item for item in self.r_city_targets
            if self._is_r_city_target_available(item)
            and get_distance(loc, item["location"]) <= self.r_city_early_entry_scene_distance
        ]
        if not candidates:
            return
        target = min(candidates, key=lambda item: get_distance(loc, item["location"]))
        if getattr(self, "_frame_worker", None) is not None:
            self._frame_worker.frame_log(f"[RCitySearch] 室内状态下匹配到R城房点 {target['id']}")
        self._lock_r_city_target(target)

    def _finish_r_city_searching(self, w: "FrameWorker", reason: str):
        self.stop_auto_forward(w)
        w.frame_log(f"[RCitySearch] {reason}，切换到跑图阶段")
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.status = "IDLE"
        return self._finish_searching_phase(w, reason)

    def _should_start_search_from_indoor(self) -> bool:
        return (
            self.current_house_id is not None
            and self.current_house_id not in self.completed_houses
        )

    def _confirm_indoor_before_search(self, w: "FrameWorker", reason: str) -> bool:
        if not w.get_info("跳跃"):
            return True

        w.frame_log(f"[SceneSearch] {reason}，且检测到跳跃按钮，先按翻窗逻辑确认")
        self.handle_jump_logic(w)
        self._refresh_frame_and_handle_jump(w)
        if self._is_indoor(w):
            w.frame_log("[SceneSearch] 跳跃后仍为 indoor，确认已进房")
            return True

        w.frame_log(f"[SceneSearch] 跳跃后 house_scene={self._get_house_scene(w)}，暂不启动旋转搜房")
        return False

    def _complete_current_house_search(self, w: "FrameWorker", reason: str) -> bool:
        if self._should_abort(w):
            return False

        self.stop_auto_forward(w)
        self.indoor_stuck_frames = 0
        w.frame_log(f"[SceneSearch] {reason}")

        if not self.start_searching(w):
            return False
        if w.current_stage != "搜房阶段":
            return False

        completed_house_id = self.current_house_id
        if completed_house_id is not None:
            self.completed_houses.add(str(completed_house_id))
        self.searching_number += 1
        completed_house_count = len(
            {
                self._r_city_target_house_id(target)
                for target in self.r_city_targets
                if self._is_r_city_target_completed(target)
            }
        )
        total_house_count = len(
            {
                self._r_city_target_house_id(target)
                for target in self.r_city_targets
                if self._r_city_target_house_id(target) is not None
            }
        )
        w.frame_log(
            f"[RCitySearch] 房屋 {completed_house_id} 完成，"
            f"已登记 {completed_house_count}/{total_house_count} 栋，后续跳过该 house_id"
        )
        self._refresh_frame_and_handle_jump(w)
        exit_direction = w.get_info("direction")
        self.prepare_next_target_logic(exit_direction)
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.status = "IDLE"
        self.history_locations = []
        self._reset_route_stuck_bypass()
        return True

    def _exit_unexpected_indoor(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        w.frame_log("[SceneSearch] 已搜完或无待搜目标时检测到 indoor，优先执行出房")
        if self._exit_house(w):
            self.indoor_stuck_frames = 0
            self.current_house_id = None
            self.status = "IDLE"
            self._continue_searching_until_timer(w, "意外进房后已出房")

    def _recover_r_city_navigation_stuck(self, w: "FrameWorker", current_loc) -> bool:
        if self._is_indoor(w):
            w.frame_log("[RCitySearch] 推进中卡住但已在室内，转入室内搜房/出房链路")
            return self._handle_indoor_during_entry_route(
                w,
                current_loc,
                "R城推进卡住时确认已进房",
            )
        w.frame_log("[RCitySearch] 推进中卡住且仍在室外，执行室外绕障")
        return self.execute_unstuck_logic(w, current_loc)

    def _move_precisely_to_entry_point(
        self,
        w: "FrameWorker",
        current_loc,
        target_loc,
        dist: float,
        phase_label: str = "Nav精推段",
    ) -> bool:
        current_dir = w.get_info("direction")
        target_angle = calculate_angle(current_loc, target_loc)
        turn_dir, _, diff = calculate_move_count(current_dir, target_angle)
        if diff is None or turn_dir is None:
            return False

        align_threshold = self.R_CITY_PRECISE_NAV_ALIGN_TOLERANCE
        align_max_steps = self.R_CITY_PRECISE_NAV_ALIGN_MAX_STEPS
        aligned = self.align_direction(
            w,
            target_loc,
            threshold=align_threshold,
            max_steps=align_max_steps,
        )
        if not aligned:
            w.frame_log(
                f"[SceneSearch] 当前距离入门点 {target_loc} 为 {dist:.2f}，"
                f"视角还没对准入门点，允许误差={align_threshold}，本轮不推摇杆"
            )
            return True
        mode = self._entry_forward_mode(dist)
        y_bias, dura, wait = self._get_entry_move_params(dist)
        w.frame_log(
                    f"[SceneSearch] {self._entry_forward_mode_label(mode)}靠近入口："
                    f"距离={dist:.2f}，目标角={target_angle}，角差={diff:.1f}°，"
                    f"y={y_bias}，dura={dura}"
                )
        return self._tap_entry_forward_with_learning(
            w,
            target_loc,
            dist,
            mode,
            y_bias,
            dura,
            wait,
            phase_label=phase_label,
        )

    @staticmethod
    def _entry_micro_dura(dist: float, base_dura: int, max_dura: int) -> int:
        try:
            dist_val = max(0.0, float(dist))
        except (TypeError, ValueError):
            dist_val = 0.0
        return int(max(base_dura, min(max_dura, base_dura + dist_val * 18)))

    def _entry_forward_mode(self, dist: float) -> str:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            dist_val = 0.0
        if dist_val > self.ENTRY_COARSE_MOVE_DISTANCE:
            return self.ENTRY_FORWARD_FAST_MODE
        return self.ENTRY_FORWARD_SLOW_MODE

    def _get_entry_move_params(self, dist):
        y_bias, dura, wait = super()._get_entry_move_params(dist)
        y_bias = self.R_CITY_PRECISE_NAV_Y_BIAS
        mode = self._entry_forward_mode(dist)
        if mode == self.ENTRY_FORWARD_FAST_MODE:
            return (
                y_bias,
                max(int(dura), self.R_CITY_ENTRY_FAST_MIN_DURA),
                max(int(wait), self.R_CITY_ENTRY_FAST_MIN_WAIT),
            )
        return (
            y_bias,
            max(int(dura), self.R_CITY_ENTRY_SLOW_MIN_DURA),
            max(int(wait), self.R_CITY_ENTRY_SLOW_MIN_WAIT),
        )

    def _precise_nav_wait_ms(self, dist, fallback_wait: int) -> int:
        try:
            wait = int(round(max(0.0, float(dist)) * 0.55 * 1000))
        except (TypeError, ValueError):
            wait = int(fallback_wait)
        return max(
            self.R_CITY_PRECISE_NAV_MIN_WAIT,
            min(self.R_CITY_PRECISE_NAV_MAX_WAIT, wait),
        )

    @staticmethod
    def _entry_forward_mode_label(mode: str) -> str:
        return "快推" if mode == "fast" else "慢推"

    def _tap_entry_forward_with_learning(
        self,
        w: "FrameWorker",
        target_loc,
        desired_dist: float,
        mode: str,
        fallback_y_bias: int,
        fallback_dura: int,
        fallback_wait: int,
        phase_label: str = "Nav精推段",
    ) -> bool:
        before_dist = self._get_current_entry_distance(w, target_loc)
        if before_dist is None:
            before_dist = desired_dist

        previous_dist = before_dist
        progress_max_steps = max(
            1,
            int(getattr(self, "R_CITY_ENTRY_FORWARD_PROGRESS_MAX_STEPS", self.ENTRY_FORWARD_MAX_STEPS)),
        )
        step = 0
        while step < progress_max_steps:
            step += 1
            if self._should_abort(w):
                return False

            current_dist = self._get_current_entry_distance(w, target_loc)
            if current_dist is None:
                current_dist = previous_dist
            if current_dist <= self.ENTRY_ARRIVAL_DISTANCE:
                w.frame_log(
                    f"[SceneSearch] 当前距离入门点 {target_loc} 为 {current_dist:.2f} "
                    f"<= {self.ENTRY_ARRIVAL_DISTANCE:g}，已经到达入门点，停止继续前推"
                )
                return True

            current_loc = self._get_current_location(w)
            current_dir = w.get_info("direction")
            target_angle = calculate_angle(current_loc, target_loc) if current_loc is not None else None
            aligned = self.align_direction(
                w,
                target_loc,
                threshold=self.R_CITY_PRECISE_NAV_ALIGN_TOLERANCE,
                max_steps=self.R_CITY_PRECISE_NAV_ALIGN_MAX_STEPS,
            )
            if not aligned:
                w.frame_log('[Action] 等待下一帧继续角度校准')
                return True

            desired_step_dist = max(0.2, float(current_dist))
            y_bias, dura, wait, distance_key = get_adaptive_forward_motion(
                mode,
                desired_step_dist,
                fallback_y_bias,
                fallback_dura,
                fallback_wait,
            )
            adaptive_y_bias = y_bias
            model_wait = wait
            wait = self._precise_nav_wait_ms(current_dist, wait)
            y_bias = self.R_CITY_PRECISE_NAV_Y_BIAS

            w.frame_log(
                            f"[SceneSearch] 入门推进 {step}/{progress_max_steps}："
                            f"距离={current_dist:.2f}，模式={self._entry_forward_mode_label(mode)}，"
                            f"档位={distance_key}，y={y_bias}，dura={dura}，wait={wait}"
                        )
            w.tap_single("摇杆", y_bias=y_bias, dura=dura, wait=wait)
            self._refresh_frame_and_handle_jump(w)

            after_dist = self._get_current_entry_distance(w, target_loc)
            update_adaptive_forward_motion(
                mode,
                desired_step_dist,
                current_dist,
                after_dist,
                y_bias,
                dura,
                wait,
            )
            if after_dist is None:
                return True

            moved = current_dist - after_dist
            w.frame_log(
                f"[SceneSearch] 推进入门点 {target_loc} 后，"
                f"距离从 {current_dist:.2f} 变为 {after_dist:.2f}，"
                f"实际靠近={moved:.2f}，模式={self._entry_forward_mode_label(mode)}，模型距离={distance_key}"
            )
            if after_dist <= self.ENTRY_ARRIVAL_DISTANCE or moved <= 0:
                break
            previous_dist = after_dist
        else:
            w.frame_log(
                f"[SceneSearch] 入门点 {target_loc} 已连续靠近 {progress_max_steps} 步，"
                "结束本轮精推进保护；下一帧继续按最新距离/角度判断是否前推"
            )

        return True

    def _tap_entry_side_with_learning(
        self,
        w: "FrameWorker",
        target_loc,
        desired_dist: float,
        side: str,
        fallback_x_bias: int,
        fallback_dura: int,
        fallback_wait: int,
    ) -> bool:
        x_bias, dura, wait, distance_key = get_adaptive_side_motion(
            side,
            desired_dist,
            fallback_x_bias,
            fallback_dura,
            fallback_wait,
        )
        before_dist = self._get_current_entry_distance(w, target_loc)
        if before_dist is None:
            before_dist = desired_dist

        w.frame_log(
            f"[SceneSearch] 执行{self._side_label(side)}滑微调: "
            f"bin={distance_key}, before={before_dist:.2f}, "
            f"x_bias={x_bias}, dura={dura}, wait={wait}"
        )
        w.tap_single("摇杆", x_bias=x_bias, y_bias=0, dura=dura, wait=wait)
        self._refresh_frame_and_handle_jump(w)

        after_dist = self._get_current_entry_distance(w, target_loc)
        update_adaptive_side_motion(
            side,
            desired_dist,
            before_dist,
            after_dist,
            x_bias,
            dura,
            wait,
        )
        if after_dist is not None:
            moved = before_dist - after_dist
            w.frame_log(
                f"[SceneSearch] 侧滑反馈: side={side}, bin={distance_key}, "
                f"after={after_dist:.2f}, moved={moved:.2f}"
            )
        return True

    def _entry_forward_step_fallback(self, y_bias: int, dura: int, wait: int):
        step_y = int(round(float(y_bias) * self.ENTRY_FORWARD_STEP_Y_SCALE))
        step_dura = max(
            self.ENTRY_FORWARD_STEP_MIN_DURA,
            int(round(float(dura) / self.ENTRY_FORWARD_MAX_STEPS)),
        )
        step_wait = max(
            step_dura + self.ENTRY_FORWARD_STEP_WAIT_PAD,
            int(round(float(wait) / self.ENTRY_FORWARD_MAX_STEPS)),
        )
        if y_bias < 0:
            step_y = min(-80, step_y)
        else:
            step_y = max(80, step_y)
        return step_y, step_dura, step_wait

    def _get_current_entry_distance(self, w: "FrameWorker", target_loc) -> Optional[float]:
        current_loc = self._get_current_location(w)
        if current_loc is None:
            return None

        dist = get_distance(current_loc, target_loc)
        if dist < 0:
            return None
        return dist

    def start_searching(self, w: "FrameWorker"):
        if self._should_abort(w):
            return False

        self._clear_house_search_timer()
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        w.frame_log("[SceneRotate] 进入房屋，启动 house_scene 旋转搜房")
        rotate_result = self._rotate_search_inside_house(w)

        if self._should_abort(w):
            return False

        self._refresh_frame_and_handle_jump(w)
        if rotate_result == self.ROTATE_RESULT_EXITED or self._is_out_of_house(w):
            w.frame_log("[SceneRotate] 旋转搜房过程中已出房，房屋搜索完成")
            return True

        if rotate_result == self.ROTATE_RESULT_FALLBACK_EXIT:
            w.frame_log("[SceneRotate] 两轮撞墙循环仍未自然出房，开始执行出房策略")
        else:
            w.frame_log("[SceneRotate] 旋转搜房结束，开始出房")

        if self._exit_house(w):
            w.frame_log("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True
        if self._should_abort(w):
            return False

        self._refresh_frame_and_handle_jump(w)
        if self._is_out_of_house(w):
            w.frame_log("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True

        w.frame_log(f"[SceneRotate] 出房策略后仍未确认出房 house_scene={self._get_house_scene(w)}")
        return False

    def _house_search_timed_out(self):
        return False

    def _rotate_search_inside_house(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        w.frame_log("[SceneRotate] 室内搜房改为固定推进转向：左上+右转一圈，随后出房")

        search_plan = (
            ("顺时针", "left_up", self.ROTATE_SEARCH_SWEEP_TURN_PX),
        )
        step_index = 0
        for phase_label, move_mode, turn_px in search_plan:
            for phase_step in range(self.ROTATE_SEARCH_SWEEP_CYCLES_PER_DIRECTION):
                if self._should_abort(w):
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_FINISHED

                self._refresh_frame_and_handle_jump(w)
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    if self._confirm_rotate_exit_or_continue(
                        w,
                        phase_label,
                        "推进前",
                        scene,
                        turn_px,
                    ):
                        return self.ROTATE_RESULT_EXITED
                    scene = self._get_house_scene(w)
                if scene is not None and scene not in {
                    self.HOUSE_INDOOR,
                    self.HOUSE_NEAR_DOOR,
                    self.HOUSE_NEAR_WALL,
                }:
                    w.frame_log(f"[SceneRotate] 当前 house_scene={scene}，停止室内旋转搜房")
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_FINISHED

                step_index += 1
                w.frame_log(
                    f"[SceneRotate] {phase_label}推进 {phase_step + 1}/"
                    f"{self.ROTATE_SEARCH_SWEEP_CYCLES_PER_DIRECTION}，全局步数={step_index}"
                )
                self._move_rotate_search_sweep_step(w, move_mode, phase_label)

                self._refresh_frame_and_handle_jump(w)
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    if self._confirm_rotate_exit_or_continue(
                        w,
                        phase_label,
                        "推进后",
                        scene,
                        turn_px,
                    ):
                        return self.ROTATE_RESULT_EXITED
                    continue

                w.frame_log(f"[SceneRotate] {phase_label}推进后原地转视角 {turn_px}px")
                self._turn_raw_pixels(w, turn_px)

                self._refresh_frame_and_handle_jump(w)
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    if self._confirm_rotate_exit_or_continue(
                        w,
                        phase_label,
                        "转向后",
                        scene,
                        -turn_px,
                    ):
                        return self.ROTATE_RESULT_EXITED
                    continue

        self.stop_auto_forward(w)
        w.frame_log("[SceneRotate] 左上+右转一圈后仍未出房，切出房策略")
        return self.ROTATE_RESULT_FALLBACK_EXIT

    def _confirm_rotate_exit_or_continue(
        self,
        w: "FrameWorker",
        phase_label: str,
        checkpoint: str,
        first_scene,
        recheck_turn_px: int,
    ) -> bool:
        turn_px = recheck_turn_px
        if not turn_px:
            turn_px = self.ROTATE_EXIT_RECHECK_TURN_PX
        if abs(turn_px) > self.ROTATE_EXIT_RECHECK_TURN_PX:
            turn_px = self.ROTATE_EXIT_RECHECK_TURN_PX if turn_px > 0 else -self.ROTATE_EXIT_RECHECK_TURN_PX

        w.frame_log(
            f"[SceneRotate] {phase_label}{checkpoint}疑似看到窗外/门外，"
            f"scene={first_scene}，转视角 x={turn_px} 复核"
        )
        self.stop_auto_forward(w)
        self._turn_raw_pixels(w, turn_px)
        self._refresh_frame_and_handle_jump(w)

        second_scene = self._get_house_scene(w)
        if second_scene in self.HOUSE_EXIT_SCENES:
            w.frame_log(
                f"[SceneRotate] {phase_label}{checkpoint}屋外信号复核仍为 house_scene={second_scene}，"
                "确认已出房，结束旋转搜房"
            )
            self.stop_auto_forward(w)
            return True

        w.frame_log(
            f"[SceneRotate] {phase_label}{checkpoint}屋外信号复核后为 house_scene={second_scene}，"
            "复核后仍在室内/门墙附近，继续未完成的旋转搜房"
        )
        return False

    def _ensure_rotate_auto_forward(self, w: "FrameWorker", reason: str):
        if self.auto_forward:
            return
        w.frame_log(f"[SceneRotate] {reason}")
        w.click("自动前进")
        self.auto_forward = True
        self._refresh_frame_and_handle_jump(w)

    def _recover_rotate_auto_forward_stuck(self, w: "FrameWorker", turn_sign: int):
        if turn_sign > 0:
            x_bias = -self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "左"
        else:
            x_bias = self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "右"

        w.frame_log(
            f"[SceneRotate] 连续 {self.ROTATE_SEARCH_STUCK_SIMILAR_FRAMES} 帧前景相似，"
            f"判定卡住，向{label}横拉一下"
        )
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=0,
            dura=self.ROTATE_SEARCH_RECOVER_STEP_MS,
            wait=self.ROTATE_SEARCH_RECOVER_STEP_MS + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        self.auto_forward = False
        self._refresh_frame_and_handle_jump(w)

    def _move_rotate_search_step(self, w: "FrameWorker", move_mode: str):
        x_bias = self._rotate_move_x_bias(move_mode)
        label = self._move_mode_label(move_mode)
        w.frame_log(f"[SceneRotate] 向{label}滑动 {self.ROTATE_SEARCH_MOVE_DURA}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.ROTATE_SEARCH_MOVE_DURA,
            wait=self.ROTATE_SEARCH_MOVE_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        self._refresh_frame_and_handle_jump(w)

    def _move_rotate_search_sweep_step(self, w: "FrameWorker", move_mode: str, phase_label: str):
        x_bias = self._rotate_move_x_bias(move_mode)
        label = self._move_mode_label(move_mode)
        w.frame_log(
            f"[SceneRotate] {phase_label}: 向{label}推进 "
            f"{self.ROTATE_SEARCH_SWEEP_MOVE_DURA}ms"
        )
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.ROTATE_SEARCH_SWEEP_MOVE_DURA,
            wait=self.ROTATE_SEARCH_SWEEP_MOVE_WAIT,
        )
        self._refresh_frame_and_handle_jump(w)

    def _turn_raw_pixels(self, w: "FrameWorker", signed_px: int):
        w.frame_log('[Action] 转动视角')
        w.tap_single(
            "视角",
            x_bias=int(signed_px),
            dura=self.ROTATE_SEARCH_SWEEP_TURN_DURA,
            wait=self.ROTATE_SEARCH_SWEEP_TURN_WAIT,
        )

    def _move_mode_turn_sign(self, move_mode: str) -> int:
        return 1 if move_mode == "left_up" else -1

    def _rotate_move_x_bias(self, move_mode: str) -> int:
        return -self.ROTATE_SEARCH_X_BIAS if move_mode == "left_up" else self.ROTATE_SEARCH_X_BIAS

    def _opposite_move_mode(self, move_mode: str) -> str:
        return "right_up" if move_mode == "left_up" else "left_up"

    def _handle_rotate_wall_hit(self, w: "FrameWorker", move_mode: str, wall_hit_count: int):
        label = "墙/门"
        current_mode = move_mode
        if wall_hit_count >= self.ROTATE_SEARCH_HIT_SWITCH_COUNT:
            if current_mode == "left_up":
                w.frame_log(f"[SceneRotate] 左上累计撞{label}已达{wall_hit_count}次，立即切到右上，并改为向左补转")
                self._turn_until_not_near_entry(w, -1)
                return "right_up", 0, True

            w.frame_log(f"[SceneRotate] 右上累计撞{label}已达{wall_hit_count}次，立即切到左上，并改为向右补转")
            self._turn_until_not_near_entry(w, 1)
            return "left_up", 0, True

        turn_sign = self._move_mode_turn_sign(current_mode)
        turn_label = "向右" if turn_sign > 0 else "向左"
        w.frame_log(f"[SceneRotate] 撞{label}后{turn_label}补转，直到不再贴墙/门")
        self._turn_until_not_near_entry(w, turn_sign)
        return move_mode, wall_hit_count, False

    def _turn_until_not_near_entry(self, w: "FrameWorker", turn_sign: int) -> bool:
        for attempt in range(self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            base_index = min(attempt, len(self.ROTATE_SEARCH_WALL_TURN_SEQUENCE) - 1)
            angle = self.ROTATE_SEARCH_WALL_TURN_SEQUENCE[base_index]
            signed_angle = turn_sign * angle
            w.frame_log(
                f"[SceneRotate] 撞墙补转 {attempt + 1}/"
                f"{self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS}: {signed_angle}°"
            )
            self._turn_with_direction_correction(w, signed_angle)
            self._refresh_frame_and_handle_jump(w)
            scene = self._get_house_scene(w)
            if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
                w.frame_log(f"[SceneRotate] 补转后 house_scene={scene}，继续当前滑动方向")
                return True

            w.frame_log(f"[SceneRotate] 补转后仍贴墙/门 house_scene={scene}，继续缩小角度补转")

        w.frame_log("[SceneRotate] 多次补转后仍贴墙/门，交给下一轮移动继续尝试")
        return False

    def _turn_with_direction_correction(self, w: "FrameWorker", signed_angle: float):
        before_dir = self._direction_as_float(w.get_info("direction"))
        self._turn(w, signed_angle)
        self._refresh_frame_and_handle_jump(w)
        if before_dir is None:
            w.frame_log("[SceneRotate] 当前 direction 无效，跳过本次补角到位校验")
            return

        target_dir = (before_dir + float(signed_angle)) % 360.0
        for step in range(self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS):
            current_dir = self._direction_as_float(w.get_info("direction"))
            turn_dir, _, diff = calculate_move_count(current_dir, target_dir)
            if diff is None or turn_dir is None:
                w.frame_log("[SceneRotate] 补角后 direction 无效，无法继续校验角度")
                return
            if diff <= self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
                return

            correction = min(float(diff), self.ROTATE_SEARCH_TURN_CORRECT_MAX_DEGREES)
            signed_correction = correction if turn_dir == "right" else -correction
            w.frame_log(
                            f"[SceneRotate] 补角校正 {step + 1}/"
                            f"{self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS}："
                            f"{current_dir:.1f}°→{target_dir:.1f}°，剩余={diff:.1f}°"
                        )
            self._turn(w, signed_correction)
            self._refresh_frame_and_handle_jump(w)

        current_dir = self._direction_as_float(w.get_info("direction"))
        _, _, remaining = calculate_move_count(current_dir, target_dir)
        if remaining is not None and remaining > self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
            w.frame_log(
                f"[SceneRotate] 二次校正后仍有角度偏差: "
                f"current={current_dir:.1f}, target={target_dir:.1f}, remaining={remaining:.1f}"
            )

    @staticmethod
    def _direction_as_float(direction) -> Optional[float]:
        try:
            return float(direction) % 360.0
        except (TypeError, ValueError):
            return None

    def _recover_rotate_search_stuck(self, w: "FrameWorker", move_mode: str, dura_ms: int):
        if move_mode == "left_up":
            x_bias = self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "右"
        else:
            x_bias = -self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "左"

        w.frame_log(f"[SceneRotate] 两帧过于相似，判定卡住，向{label}水平脱困 {dura_ms}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=0,
            dura=dura_ms,
            wait=dura_ms + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        self._refresh_frame_and_handle_jump(w)

    def _exit_house(self, w: "FrameWorker") -> bool:
        w.frame_log("[SceneExit] 出房开始：先按60度扫描找门；六次仍无门再找窗；三次穿门/翻窗失败后进入应急冲出")
        self._exit_target_attempts = 0
        self.stop_auto_forward(w)
        if self._exit_house_by_door_scan_strategy(w):
            w.frame_log("[SceneExit] 按门扫描出房成功")
            self.stop_auto_forward(w)
            return True

        self.stop_auto_forward(w)
        final_scene = self._get_house_scene(w)
        w.frame_log(f"[SceneExit] 门扫描出房未成功，停止移动后最终 house_scene={final_scene}")
        return self._is_out_of_house(w)

    def _exit_house_by_door_scan_strategy(self, w: "FrameWorker") -> bool:
        if self._try_exit_current_visible_door(w, "当前视角"):
            return True

        for turn_index in range(self.SCENE_EXIT_DOOR_SCAN_TURN_COUNT):
            if self._should_abort(w):
                return False
            w.frame_log(
                f"[SceneExit] 没看到门，顺时针转{self.SCENE_EXIT_SCAN_TURN_DEGREES}度找门 "
                f"{turn_index + 1}/{self.SCENE_EXIT_DOOR_SCAN_TURN_COUNT}"
            )
            self._turn(w, self.SCENE_EXIT_SCAN_TURN_DEGREES)
            self._refresh_frame_and_handle_jump(w, "出房转60度后复查门")
            if self._is_out_of_house(w):
                return True
            if self._try_exit_visible_door_from_current_frame(w, f"第{turn_index + 1}次60度转向后"):
                return True

        w.frame_log("[SceneExit] 连续6次60度转向仍未找到门，切换为按60度找窗翻出")
        if self._exit_house_by_window_scan_strategy(w):
            return True
        return self._emergency_exit_by_auto_forward_clockwise(w, "门窗均未成功出房")

    def _try_exit_current_visible_door(self, w: "FrameWorker", phase_label: str) -> bool:
        self._refresh_frame_and_handle_jump(w, f"出房{phase_label}查门")
        if self._is_out_of_house(w):
            w.frame_log(f"[SceneExit] {phase_label}查门前已双帧确认在屋外")
            return True

        return self._try_exit_visible_door_from_current_frame(w, phase_label)

    def _try_exit_visible_door_from_current_frame(self, w: "FrameWorker", phase_label: str) -> bool:
        door = self.find_largest_door(w)
        if not door:
            w.frame_log(f"[SceneExit] {phase_label}没有看到门，继续按出房扫描流程执行")
            return False

        w.frame_log(f"[SceneExit] {phase_label}看到门，先对准门再大幅前推: door={door}")
        align_state, last_door_offset = self._align_to_door_detection(
            w,
            door,
            tolerance_px=self.SCENE_EXIT_DOOR_ALIGN_TOLERANCE_PX,
            phase_label="SceneExitDoor",
            return_state=True,
            return_last_offset=True,
        )
        if align_state == "aligned":
            result = self._push_exit_door_and_check_out(w, f"{phase_label}门已对准")
        elif align_state == "lost":
            w.frame_log(
                f"[SceneExit] {phase_label}对准门时目标丢失，"
                f"保留丢失前门中心偏移={last_door_offset}，按该侧门口恢复"
            )
            result = self._recover_lost_exit_door_and_check_out(
                w,
                phase_label,
                last_door_offset,
            )
        else:
            w.frame_log(f"[SceneExit] {phase_label}门未进入容差，保留当前位置继续扫描找门")
            result = False

        if result:
            return True
        emergency_result = self._record_exit_target_attempt_failure(w, f"{phase_label}出门尝试失败")
        return False if emergency_result is None else emergency_result

    def _record_exit_target_attempt_failure(self, w: "FrameWorker", reason: str) -> Optional[bool]:
        self._exit_target_attempts = getattr(self, "_exit_target_attempts", 0) + 1
        w.frame_log(
            f"[SceneExit] {reason}，目标出房失败计数="
            f"{self._exit_target_attempts}/{self.SCENE_EXIT_TARGET_ATTEMPT_LIMIT}"
        )
        if self._exit_target_attempts < self.SCENE_EXIT_TARGET_ATTEMPT_LIMIT:
            return None
        return self._emergency_exit_by_auto_forward_clockwise(
            w,
            f"连续{self.SCENE_EXIT_TARGET_ATTEMPT_LIMIT}次出门/翻窗失败",
        )

    def _tap_exit_motion_and_check_out(
        self,
        w: "FrameWorker",
        label: str,
        *,
        x_bias: int = 0,
        y_bias: int,
        dura: int,
        wait: int,
    ) -> bool:
        w.frame_log(
            f"[SceneExit] {label}: x_bias={x_bias}, y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        w.tap_single("摇杆", x_bias=x_bias, y_bias=y_bias, dura=dura, wait=wait)
        self._refresh_frame_and_handle_jump(w, f"{label}后复核")
        return self._is_out_of_house(w)

    def _recover_lost_exit_door_and_check_out(
        self,
        w: "FrameWorker",
        phase_label: str,
        last_door_offset: Optional[float],
    ) -> bool:
        if self._tap_exit_motion_and_check_out(
            w,
            f"{phase_label}门丢失后直推",
            y_bias=self.SCENE_EXIT_DOOR_FORWARD_Y_BIAS,
            dura=self.SCENE_EXIT_DOOR_FORWARD_DURA,
            wait=self.SCENE_EXIT_DOOR_FORWARD_WAIT,
        ):
            return True
        scene_after_forward = self._get_house_scene(w)
        if scene_after_forward != self.HOUSE_NEAR_WALL:
            w.frame_log(
                f"[SceneExit] 门丢失后直推未出房，house_scene={scene_after_forward}；"
                "不是 near_wall，不做后拉，回到找门/找窗扫描"
            )
            return False

        first_side = "右" if last_door_offset is None or last_door_offset >= 0 else "左"
        second_side = "左" if first_side == "右" else "右"
        w.frame_log(
            f"[SceneExit] 门丢失后直推仍贴墙；丢失前门在{first_side}侧，"
            f"先{first_side}前推2秒，再{second_side}前推4秒"
        )
        for index, side in enumerate((first_side, second_side)):
            x_bias = self.SCENE_EXIT_DOOR_LOST_SIDE_X_BIAS if side == "右" else -self.SCENE_EXIT_DOOR_LOST_SIDE_X_BIAS
            dura = (
                self.SCENE_EXIT_DOOR_LOST_FIRST_SIDE_DURA
                if index == 0
                else self.SCENE_EXIT_DOOR_LOST_SECOND_SIDE_DURA
            )
            wait = (
                self.SCENE_EXIT_DOOR_LOST_FIRST_SIDE_WAIT
                if index == 0
                else self.SCENE_EXIT_DOOR_LOST_SECOND_SIDE_WAIT
            )
            if self._tap_exit_motion_and_check_out(
                w,
                f"门丢失直推贴墙后向{side}前推 {index + 1}/2",
                x_bias=x_bias,
                y_bias=self.SCENE_EXIT_DOOR_LOST_SIDE_Y_BIAS,
                dura=dura,
                wait=wait,
            ):
                return True

        if self._get_house_scene(w) != self.HOUSE_NEAR_WALL:
            return False

        w.frame_log("[SceneExit] 门丢失后直推及左右前推均未出房且仍贴墙，后拉掉头后重新找门/窗")
        self._recover_exit_wall_collision(w, "门丢失左右前推后")
        return self._is_out_of_house(w)

    def _push_exit_door_and_check_out(self, w: "FrameWorker", reason: str) -> bool:
        w.frame_log(
            f"[SceneExit] {reason}：门已对准，直推5秒出房 "
            f"y={self.SCENE_EXIT_DOOR_FORWARD_Y_BIAS}，"
            f"dura={self.SCENE_EXIT_DOOR_FORWARD_DURA}"
        )
        w.tap_single(
            "摇杆",
            y_bias=self.SCENE_EXIT_DOOR_FORWARD_Y_BIAS,
            dura=self.SCENE_EXIT_DOOR_FORWARD_DURA,
            wait=self.SCENE_EXIT_DOOR_FORWARD_WAIT,
        )
        self._refresh_frame_and_handle_jump(w, "出房对准门大幅前推后")

        if self._is_out_of_house(w):
            w.frame_log("[SceneExit] 对准门大幅前推后双帧确认已出房")
            return True
        if self._try_aligned_exit_door_side_pushes_and_check_out(w, reason):
            return True
        if self._recover_exit_wall_collision(w, "对准门大幅前推后"):
            return self._is_out_of_house(w)
        return False

    def _try_aligned_exit_door_side_pushes_and_check_out(
        self,
        w: "FrameWorker",
        reason: str,
    ) -> bool:
        scene = self._get_house_scene(w)
        if scene != self.HOUSE_NEAR_WALL:
            return False

        w.frame_log(
            f"[SceneExit] 直推后仍贴墙，先左前推、再右前推尝试顶出；"
            f"reason={reason}，x=±{self.SCENE_EXIT_DOOR_WALL_SIDE_X_BIAS}，"
            f"dura={self.SCENE_EXIT_DOOR_WALL_SIDE_DURA}"
        )
        for side, x_bias in (
            ("左", -self.SCENE_EXIT_DOOR_WALL_SIDE_X_BIAS),
            ("右", self.SCENE_EXIT_DOOR_WALL_SIDE_X_BIAS),
        ):
            if self._tap_exit_motion_and_check_out(
                w,
                f"对准门直推贴墙后向{side}前推顶出",
                x_bias=x_bias,
                y_bias=self.SCENE_EXIT_DOOR_WALL_SIDE_Y_BIAS,
                dura=self.SCENE_EXIT_DOOR_WALL_SIDE_DURA,
                wait=self.SCENE_EXIT_DOOR_WALL_SIDE_WAIT,
            ):
                w.frame_log(f"[SceneExit] 对准门直推贴墙后向{side}前推，双帧确认已出房")
                return True

        w.frame_log("[SceneExit] 对准门直推撞墙后左右前推均未出房，继续后拉掉头找门")
        return False

    def _exit_house_by_window_scan_strategy(self, w: "FrameWorker") -> bool:
        for turn_index in range(self.SCENE_EXIT_WINDOW_SCAN_TURN_COUNT + 1):
            if self._should_abort(w):
                return False
            window = self._find_center_nearest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window:
                if self._exit_via_center_nearest_window(w, window):
                    return True
                emergency_result = self._record_exit_target_attempt_failure(
                    w,
                    f"第{turn_index + 1}次窗户翻出尝试失败",
                )
                if emergency_result is not None:
                    return emergency_result

            if turn_index >= self.SCENE_EXIT_WINDOW_SCAN_TURN_COUNT:
                break
            w.frame_log(
                f"[SceneExit] 未定位到可用窗户，顺时针转{self.SCENE_EXIT_SCAN_TURN_DEGREES}度找窗 "
                f"{turn_index + 1}/{self.SCENE_EXIT_WINDOW_SCAN_TURN_COUNT}"
            )
            self._turn(w, self.SCENE_EXIT_SCAN_TURN_DEGREES)
            self._refresh_frame_and_handle_jump(w, "出房转60度后复查窗户")
            if self._is_out_of_house(w):
                return True
        return False

    def _exit_via_center_nearest_window(self, w: "FrameWorker", window) -> bool:
        target = window
        align_state = "approached"
        for step in range(self.SCENE_EXIT_WINDOW_APPROACH_MAX_STEPS):
            angle = self._target_relative_angle(target)
            if angle is None:
                break
            if abs(angle) <= self.EXIT_WINDOW_ALIGN_TOLERANCE_DEGREES:
                align_state = "aligned"
                break
            turn_angle = max(
                -self.SCENE_EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES,
                min(self.SCENE_EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES, angle),
            )
            w.frame_log(
                f"[SceneExit] 选择画面中心最近的窗户，靠近 {step + 1}/"
                f"{self.SCENE_EXIT_WINDOW_APPROACH_MAX_STEPS}: turn={turn_angle:.1f}"
            )
            self._turn(w, turn_angle)
            self._refresh_frame_and_handle_jump(w, "窗户靠近后重新识别")
            refreshed = self._find_center_nearest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if refreshed is None:
                align_state = "lost"
                break
            target = refreshed

        w.frame_log(
            f"[SceneExit] 画面中心最近窗户{('已对准' if align_state == 'aligned' else '最多靠近3次后停止调整')}，"
            "执行一次轻微前推并检查跳跃"
        )
        return self._push_exit_window_once(w, align_state)

    def _push_exit_window_once(self, w: "FrameWorker", align_state: str) -> bool:
        if w.get_info("跳跃"):
            w.frame_log("[SceneExit] 靠窗轻推前已出现跳跃，点击跳跃+前推翻出")
            return self._jump_forward_exit_window(w, 0)
        if self._tap_exit_motion_and_check_out(
            w,
            f"窗户{align_state}后轻微前推",
            y_bias=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_Y_BIAS,
            dura=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_DURA,
            wait=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_WAIT,
        ):
            return True
        if w.get_info("跳跃"):
            w.frame_log("[SceneExit] 窗户轻推后出现跳跃，停止前进并点击跳跃+前推翻出")
            self.stop_auto_forward(w)
            w.click("跳跃")
            w.tap_single(
                "摇杆",
                y_bias=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_Y_BIAS,
                dura=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_DURA,
                wait=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w, "翻窗跳跃前推后")
            if self._is_out_of_house(w):
                return True
        if self._get_house_scene(w) == self.HOUSE_NEAR_WALL:
            w.frame_log("[SceneExit] 靠窗轻推撞墙，后拉后重新定位窗户")
            self.stop_auto_forward(w)
            w.tap_single(
                "摇杆",
                y_bias=self.SCENE_EXIT_WALL_BACKOFF_Y_BIAS,
                dura=self.SCENE_EXIT_WALL_BACKOFF_DURA,
                wait=self.SCENE_EXIT_WALL_BACKOFF_WAIT,
            )
            self._refresh_frame_and_handle_jump(w, "翻窗撞墙后拉后")
        return self._is_out_of_house(w)

    def _emergency_exit_by_auto_forward_clockwise(self, w: "FrameWorker", reason: str) -> bool:
        w.frame_log(
            f"[SceneExit] 启动最终应急出房：{reason}；自动前进并持续顺时针每次转"
            f"{self.SCENE_EXIT_EMERGENCY_TURN_DEGREES}度，出现跳跃立即跳跃前推"
        )
        for step in range(self.SCENE_EXIT_EMERGENCY_MAX_STEPS):
            if self._should_abort(w):
                self.stop_auto_forward(w)
                return False
            if not self.auto_forward:
                w.click("自动前进")
                self.auto_forward = True
            self._turn(w, self.SCENE_EXIT_EMERGENCY_TURN_DEGREES)
            time.sleep(self.SCENE_EXIT_EMERGENCY_POLL_SECONDS)
            self._refresh_frame_and_handle_jump(w, f"最终应急顺时针第{step + 1}步")
            if self._is_out_of_house(w):
                self.stop_auto_forward(w)
                return True
            if w.get_info("跳跃"):
                w.frame_log("[SceneExit] 应急冲出中看到跳跃，立刻停止自动前进并跳跃+前推")
                self.stop_auto_forward(w)
                w.click("跳跃")
                w.tap_single(
                    "摇杆",
                    y_bias=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_Y_BIAS,
                    dura=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_DURA,
                    wait=self.SCENE_EXIT_WINDOW_GENTLE_FORWARD_WAIT,
                )
                self._refresh_frame_and_handle_jump(w, "最终应急跳跃前推后")
                if self._is_out_of_house(w):
                    return True
        self.stop_auto_forward(w)
        w.frame_log("[SceneExit] 最终应急本轮未出房，下一帧继续同一应急策略")
        return self._is_out_of_house(w)

    def _recover_exit_wall_collision(self, w: "FrameWorker", reason: str) -> bool:
        scene = self._get_house_scene(w)
        if scene != self.HOUSE_NEAR_WALL:
            return False

        w.frame_log(
            f"[SceneExit] {reason}：撞墙，先后拉 "
            f"y={self.SCENE_EXIT_WALL_BACKOFF_Y_BIAS}，"
            f"再将视角调转180度 x={self.SCENE_EXIT_WALL_TURN_AROUND_PX}"
        )
        self.stop_auto_forward(w)
        w.tap_single(
            "摇杆",
            y_bias=self.SCENE_EXIT_WALL_BACKOFF_Y_BIAS,
            dura=self.SCENE_EXIT_WALL_BACKOFF_DURA,
            wait=self.SCENE_EXIT_WALL_BACKOFF_WAIT,
        )
        self._refresh_frame_and_handle_jump(w, "出房撞墙后拉后")
        w.tap_single(
            "视角",
            x_bias=self.SCENE_EXIT_WALL_TURN_AROUND_PX,
            dura=self.SCENE_EXIT_WALL_TURN_AROUND_DURA,
            wait=self.SCENE_EXIT_WALL_TURN_AROUND_WAIT,
        )
        self._refresh_frame_and_handle_jump(w, "出房撞墙后视角180度复位")
        return True

    def _exit_house_by_scene_strategy(self, w: "FrameWorker") -> bool:
        w.frame_log("[SceneExit] 启动 house_scene 多路径出房策略")
        move_mode = "left_up"
        wall_hit_count = 0

        for step in range(self.EXIT_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                return False

            self._refresh_frame_and_handle_jump(w)
            if self._is_out_of_house(w):
                w.frame_log("[SceneExit] 出房策略开始前已判定在屋外")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

            button_state = self._door_button_state(w)
            if button_state and self._exit_via_door_button(w, button_state):
                return True

            label = self._move_mode_label(move_mode)
            w.frame_log(f"[SceneExit] {label}绕圈找出口 {step + 1}/{self.EXIT_SEARCH_MAX_STEPS}")
            self._move_exit_search_step(w, move_mode)
            if self._is_out_of_house(w):
                w.frame_log(f"[SceneExit] {label}滑动时意外出房，出房成功")
                return True

            button_state = self._door_button_state(w)
            if button_state and self._exit_via_door_button(w, button_state):
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

            scene = self._get_house_scene(w)
            if scene in self.HOUSE_NEAR_ENTRY_SCENES:
                wall_hit_count += 1
                w.frame_log(
                    f"[SceneExit] 累计撞墙/门 {wall_hit_count}/"
                    f"{self.ROTATE_SEARCH_HIT_SWITCH_COUNT}, mode={move_mode}"
                )
                if wall_hit_count >= self.ROTATE_SEARCH_HIT_SWITCH_COUNT:
                    move_mode = self._opposite_move_mode(move_mode)
                    wall_hit_count = 0
                    label = self._move_mode_label(move_mode)
                    w.frame_log(f"[SceneExit] 撞墙/门达到阈值，切换为{label}逆向绕圈")

            turn_sign = self._move_mode_turn_sign(move_mode)
            turn_label = "向右" if turn_sign > 0 else "向左"
            w.frame_log(
                f"[SceneExit] {self._move_mode_label(move_mode)}后{turn_label}调整视角 "
                f"{self.EXIT_SEARCH_TURN_DEGREES}° 继续绕圈"
            )
            self._turn(w, turn_sign * self.EXIT_SEARCH_TURN_DEGREES)
            self._refresh_frame_and_handle_jump(w)

        w.frame_log("[SceneExit] 多路径出房策略达到步数上限，仍未确认出房")
        return self._is_out_of_house(w)

    def _move_exit_search_step(self, w: "FrameWorker", move_mode: str):
        w.tap_single(
            "摇杆",
            x_bias=self._rotate_move_x_bias(move_mode),
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.EXIT_SEARCH_LEFT_UP_DURA,
            wait=self.EXIT_SEARCH_LEFT_UP_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        self._refresh_frame_and_handle_jump(w)

    def _exit_via_door_button(self, w: "FrameWorker", button_state: str) -> bool:
        if button_state == "open":
            w.frame_log("[SceneExit] 发现开门按钮，点击开门后尝试出门")
            w.click("开门")
            time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
            self._refresh_frame_and_handle_jump(w)
        else:
            w.frame_log("[SceneExit] 发现关门按钮，门已打开，直接尝试出门")

        return self._exit_open_door_by_diagonal_sweep(w)

    def _exit_open_door_by_diagonal_sweep(self, w: "FrameWorker") -> bool:
        for step in range(self.EXIT_DOOR_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return False

            self._refresh_frame_and_handle_jump(w)
            if self._is_out_of_house(w):
                w.frame_log("[SceneExit] 门口推进前已在屋外")
                return True

            if self._door_button_state(w) == "open":
                w.frame_log("[SceneExit] 门口推进前再次看到开门按钮，补点一次开门")
                w.click("开门")
                time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
                self._refresh_frame_and_handle_jump(w)

            side = "left" if step % 2 == 0 else "right"
            dura = min(
                self.ENTRY_OPEN_SWEEP_BASE_DURA + step * self.ENTRY_OPEN_SWEEP_STEP_MS,
                self.ENTRY_OPEN_SWEEP_MAX_DURA,
            )
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            w.frame_log(f"[SceneExit] 门已打开，向{self._side_label(side)}上小步尝试出门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.ENTRY_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.ENTRY_SWEEP_WAIT_PAD,
            )
            self._refresh_frame_and_handle_jump(w)

            if self._is_out_of_house(w):
                w.frame_log("[SceneExit] 左上/右上门口推进后出房成功")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

        w.frame_log("[SceneExit] 门口左上/右上推进到上限，未确认出房")
        return False

    def _exit_via_window_by_scene(self, w: "FrameWorker", window) -> bool:
        rel_angle = self._target_relative_angle(window)
        w.frame_log(f"[SceneExit] 发现窗户，准备对齐 rel_angle={rel_angle}")
        align_state = self._align_to_exit_window(w, window)
        if align_state == "lost":
            return self._push_until_jump_and_exit_window(w, "窗户对齐过程中目标丢失")
        if align_state == "aligned":
            return self._push_until_jump_and_exit_window(w, "窗户已对齐")
        return False

    def _align_to_exit_window(self, w: "FrameWorker", window) -> str:
        target = window
        for step in range(self.EXIT_WINDOW_ALIGN_MAX_STEPS):
            if self._should_abort(w):
                return "abort"

            rel_angle = self._target_relative_angle(target)
            if rel_angle is None:
                return "lost"
            if abs(rel_angle) <= self.EXIT_WINDOW_ALIGN_TOLERANCE_DEGREES:
                w.frame_log(f"[SceneExit] 窗户已对齐 rel_angle={rel_angle:.1f}")
                return "aligned"

            turn_angle = max(
                -self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES,
                min(self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES, rel_angle),
            )
            side = "右" if turn_angle > 0 else "左"
            w.frame_log(
                f"[SceneExit] 窗户在{side}侧，对齐 {step + 1}/"
                f"{self.EXIT_WINDOW_ALIGN_MAX_STEPS}: turn={turn_angle:.1f}"
            )
            self._turn(w, turn_angle)
            self._refresh_frame_and_handle_jump(w)

            refreshed = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if not refreshed:
                w.frame_log("[SceneExit] 对齐窗户时目标丢失，改为前推找跳跃按钮")
                return "lost"
            target = refreshed

        w.frame_log("[SceneExit] 窗户对齐达到步数上限，按已接近窗户处理")
        return "aligned"

    def _push_until_jump_and_exit_window(self, w: "FrameWorker", reason: str) -> bool:
        w.frame_log(f"[SceneExit] {reason}，最多前推 {self.EXIT_WINDOW_FORWARD_MAX_STEPS} 次找跳跃")
        for step in range(self.EXIT_WINDOW_FORWARD_MAX_STEPS):
            if self._should_abort(w):
                return False

            self._refresh_frame_and_handle_jump(w)
            if self._is_out_of_house(w):
                w.frame_log("[SceneExit] 靠窗前推前已出房")
                return True

            if w.get_info("跳跃"):
                if self._jump_forward_exit_window(w, step + 1):
                    return True
                continue

            w.frame_log(f"[SceneExit] 靠窗前推找跳跃 {step + 1}/{self.EXIT_WINDOW_FORWARD_MAX_STEPS}")
            w.tap_single(
                "摇杆",
                y_bias=self.EXIT_WINDOW_FORWARD_Y_BIAS,
                dura=self.EXIT_WINDOW_FORWARD_DURA,
                wait=self.EXIT_WINDOW_FORWARD_WAIT,
            )
            self._refresh_frame_and_handle_jump(w)

            if self._is_out_of_house(w):
                w.frame_log("[SceneExit] 靠窗前推时意外出房")
                return True

            if w.get_info("跳跃") and self._jump_forward_exit_window(w, step + 1):
                return True

        w.frame_log("[SceneExit] 靠窗前推 3 次仍未出现可用跳跃，放弃该窗户")
        return False

    def _jump_forward_exit_window(self, w: "FrameWorker", step: int) -> bool:
        w.frame_log(f"[SceneExit] 检测到跳跃按钮，尝试翻窗出房 step={step}")
        if not self.handle_jump_logic(w, f"SceneExit 翻窗出房 step={step}"):
            return False
        if self._is_out_of_house(w):
            w.frame_log("[SceneExit] 翻窗后出房成功")
            return True
        return False

    def _find_largest_forward_target(self, w: "FrameWorker", class_ids: set):
        candidates = []
        for obj in self._get_forward_scene(w):
            try:
                if len(obj) < 6 or int(obj[5]) not in class_ids:
                    continue
                area = max(0.0, float(obj[2]) - float(obj[0])) * max(0.0, float(obj[3]) - float(obj[1]))
            except (TypeError, ValueError):
                continue
            candidates.append((area, obj))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _find_center_nearest_forward_target(self, w: "FrameWorker", class_ids: set):
        frame_w, frame_h = self._get_frame_size()
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0
        candidates = []
        for obj in self._get_forward_scene(w):
            try:
                if len(obj) < 6 or int(obj[5]) not in class_ids:
                    continue
                target_x = (float(obj[0]) + float(obj[2])) / 2.0
                target_y = (float(obj[1]) + float(obj[3])) / 2.0
                center_distance = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
            except (TypeError, ValueError):
                continue
            candidates.append((center_distance, obj))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _target_relative_angle(self, target) -> Optional[float]:
        try:
            center_x = (float(target[0]) + float(target[2])) / 2.0
        except (TypeError, ValueError, IndexError):
            return None
        return self.pixel_to_angle(center_x)

    def _copy_current_frame(self, w: "FrameWorker"):
        frame = getattr(w, "frame", None)
        if frame is None or not hasattr(frame, "shape"):
            return None
        return np.array(frame, copy=True)

    def _frames_are_similar(self, before_frame, after_frame):
        before = self._prepare_frame_for_compare(before_frame)
        after = self._prepare_frame_for_compare(after_frame)
        if before is None or after is None:
            return False, 999.0, 1.0

        diff = cv2.absdiff(before, after)
        mean_diff = float(np.mean(diff))
        changed_ratio = float(np.mean(diff > self.ROTATE_FRAME_CHANGED_PIXEL_THRESHOLD))
        similar = (
            mean_diff <= self.ROTATE_FRAME_MEAN_DIFF_THRESHOLD
            and changed_ratio <= self.ROTATE_FRAME_CHANGED_RATIO_THRESHOLD
        )
        return similar, mean_diff, changed_ratio

    def _prepare_frame_for_compare(self, frame):
        if frame is None or not hasattr(frame, "shape"):
            return None
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            return None

        h, w = gray.shape[:2]
        if h <= 1 or w <= 1:
            return None

        rx1, ry1, rx2, ry2 = self.ROTATE_FRAME_COMPARE_ROI
        x1 = max(0, min(w - 1, int(w * rx1)))
        y1 = max(0, min(h - 1, int(h * ry1)))
        x2 = max(x1 + 1, min(w, int(w * rx2)))
        y2 = max(y1 + 1, min(h, int(h * ry2)))
        crop = gray[y1:y2, x1:x2]
        return cv2.resize(crop, self.ROTATE_FRAME_COMPARE_SIZE, interpolation=cv2.INTER_AREA)

    def _door_button_state(self, w: "FrameWorker") -> Optional[str]:
        if w.get_info("开门"):
            return "open"
        if w.get_info("关门"):
            return "close"
        return None

    def _click_open_door(self, w: "FrameWorker"):
        w.frame_log("[SceneEntry] 检测到开门按钮，点击开门")
        w.click("开门")
        time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
        self._refresh_frame_and_handle_jump(w)

    def _is_indoor(self, w: "FrameWorker") -> bool:
        return self._get_house_scene(w) == self.HOUSE_INDOOR

    def _is_out_of_house(self, w: "FrameWorker") -> bool:
        first_scene = self._get_house_scene(w)
        if first_scene not in self.HOUSE_EXIT_SCENES:
            return False

        self._refresh_frame_and_handle_jump(w, "出房单帧信号复核")
        second_scene = self._get_house_scene(w)
        if second_scene in self.HOUSE_EXIT_SCENES:
            w.frame_log(
                f"[SceneExit] 连续确认屋外信号 first={first_scene}, "
                f"second={second_scene}，判定已出房"
            )
            return True

        w.frame_log(
            f"[SceneExit] 单帧屋外信号 first={first_scene} 后复核为 "
            f"{second_scene}，判定仍未出房，继续出房逻辑"
        )
        return False

    @staticmethod
    def _move_mode_label(move_mode: str) -> str:
        return "左上" if move_mode == "left_up" else "右上"

    @staticmethod
    def _side_label(side: str) -> str:
        return "左" if side == "left" else "右"
