import json
import os
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_search_manager import (
    HouseSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    calculate_angle,
    calculate_move_count,
    check_location,
    get_adaptive_forward_motion,
    get_adaptive_side_motion,
    get_distance,
    update_adaptive_forward_motion,
    update_adaptive_side_motion,
)

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


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

    R_CITY_AREA_CONFIG_PATH = os.path.join(
        "aw",
        "autogame",
        "customs_examples",
        "Auto_PUBG_ALL",
        "resource",
        "house_entry",
        "r_city_house_area.json",
    )
    R_CITY_FALLBACK_CENTER = (1036, 745)
    R_CITY_DEFAULT_NEAR_DISTANCE = 30.0
    R_CITY_DEFAULT_HOUSE_ARRIVAL_DISTANCE = 2.0
    R_CITY_DEFAULT_EARLY_ENTRY_SCENE_DISTANCE = 5.0
    R_CITY_ROUTE_WAYPOINT_DISTANCE = 3.0
    R_CITY_ROUTE_REPLAN_STUCK_CYCLES = 2
    R_CITY_FAILED_TARGET_LIMIT = 2
    R_CITY_FORWARD_HOUSE_BYPASS_DISTANCE = 10.0
    R_CITY_BODY_ENTRY_DISTANCE = 4.0
    R_CITY_BODY_ENTRY_ALIGN_WAIT = 30

    STATUS_ROUTE_TO_R_CITY = "ROUTE_TO_R_CITY"
    STATUS_SCENE_ENTRY = "SCENE_ENTRY"
    ROTATE_RESULT_FINISHED = "finished"
    ROTATE_RESULT_EXITED = "exited"
    ROTATE_RESULT_FALLBACK_EXIT = "fallback_exit"
    ENTRY_DIRECTION_ALIGN_TOLERANCE = 3
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
    R_CITY_ENTRY_FREE_SEARCH_MAX_STEPS = 8
    R_CITY_ENTRY_WALL_SWEEP_STEPS = 8
    R_CITY_ENTRY_WALL_SWEEP_X_BIAS = 260
    R_CITY_ENTRY_WALL_SWEEP_DURA = 260
    R_CITY_ENTRY_WALL_SWEEP_WAIT = 50
    R_CITY_ENTRY_WALL_SWEEP_WAIT_STEP = 50
    R_CITY_ENTRY_BLIND_FORWARD_Y_BIAS = -320
    R_CITY_ENTRY_BLIND_FORWARD_DURA = 420
    R_CITY_ENTRY_BLIND_FORWARD_WAIT = 300
    R_CITY_ENTRY_WINDOW_FORWARD_Y_BIAS = -220
    R_CITY_ENTRY_WINDOW_FORWARD_DURA = 260
    R_CITY_ENTRY_WINDOW_FORWARD_WAIT = 220
    R_CITY_ENTRY_WINDOW_WALL_SWEEP_X_BIAS = 140
    R_CITY_ENTRY_WINDOW_WALL_SWEEP_DURA = 180
    R_CITY_ENTRY_JUMP_WALL_BACKOFF_Y_BIAS = 400
    R_CITY_ENTRY_JUMP_WALL_BACKOFF_DURA = 100
    R_CITY_ENTRY_JUMP_WALL_BACKOFF_WAIT = 500
    R_CITY_ENTRY_WALL_RELOCK_BACKOFF_Y_BIAS = 300
    R_CITY_ENTRY_WALL_RELOCK_BACKOFF_DURA = 320
    R_CITY_ENTRY_WALL_RELOCK_BACKOFF_WAIT = 450
    R_CITY_ENTRY_WALL_RELOCK_TURN_PX = 220
    R_CITY_ENTRY_WALL_RELOCK_TURN_DURA = 180
    R_CITY_ENTRY_WALL_RELOCK_TURN_WAIT = 300
    R_CITY_ENTRY_WALL_RELOCK_MAX_ATTEMPTS = 2
    R_CITY_ENTRY_STONE_WALL_FORWARD_Y_BIAS = -300
    R_CITY_ENTRY_STONE_WALL_FORWARD_DURA = 360
    R_CITY_ENTRY_STONE_WALL_FORWARD_WAIT = 260
    R_CITY_ENTRY_STONE_WALL_JUMP_SETTLE_SECONDS = 0.2
    R_CITY_ENTRY_TARGET_ALIGN_TOLERANCE = 8
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
    ROTATE_SEARCH_X_BIAS = 330
    ROTATE_SEARCH_Y_BIAS = -430
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
    ROTATE_SEARCH_SWEEP_TURN_PX = 350
    ROTATE_SEARCH_SWEEP_TURN_DURA = 150
    ROTATE_SEARCH_SWEEP_TURN_WAIT = 3000
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

    WATER_FLOAT_DURA = 2000
    WATER_BACK_DURA = 650
    WATER_BACK_WAIT = 900
    WATER_SIDE_X_BIAS = 320
    WATER_SIDE_DURA = 900
    WATER_SIDE_WAIT = 1500
    WATER_FORWARD_Y_BIAS = -300
    WATER_FORWARD_DURA = 850
    WATER_FORWARD_WAIT = 1500
    WATER_ESCAPE_STUCK_DISTANCE = 0.6
    WATER_ESCAPE_SIDE_SWITCH_ATTEMPTS = 3
    WATER_ESCAPE_MAX_ATTEMPTS = 5
    FORBIDDEN_ESCAPE_SEARCH_RADIUS = 120
    FORBIDDEN_ESCAPE_ARRIVAL_DISTANCE = 3.0
    FORBIDDEN_ESCAPE_FORWARD_DURA = 700
    FORBIDDEN_ESCAPE_FORWARD_WAIT = 900

    def __init__(self):
        super().__init__()
        self.r_city_config = self._load_r_city_area_config()
        self.r_city_center = self._load_r_city_center()
        self.r_city_near_distance = self._load_r_city_threshold(
            "near_region_distance",
            self.R_CITY_DEFAULT_NEAR_DISTANCE,
        )
        self.r_city_house_arrival_distance = self._load_r_city_threshold(
            "house_arrival_distance",
            self.R_CITY_DEFAULT_HOUSE_ARRIVAL_DISTANCE,
        )
        self.r_city_early_entry_scene_distance = self._load_r_city_threshold(
            "early_entry_scene_distance",
            self.R_CITY_DEFAULT_EARLY_ENTRY_SCENE_DISTANCE,
        )
        self.r_city_side_candidate_ids = (
            self.r_city_config.get("route_entry_strategy", {})
            .get("side_candidate_ids", {})
        )
        self.r_city_targets = self._build_r_city_targets()
        self._reset_r_city_runtime()

    def reset(self):
        super().reset()
        self._reset_r_city_runtime()

    def searching_logic(self, w: "FrameWorker", current_loc, current_direction):
        if self._should_abort(w):
            return

        house_scene = self._get_house_scene(w)
        if self._finish_callback_configured() and self._can_finish_searching(w):
            self._continue_searching_until_timer(w, "R城搜房计时已满")
            return

        if self._is_in_water(w):
            self._handle_water_escape(w, current_loc, current_direction)
            return

        if house_scene == self.HOUSE_INDOOR:
            self._adopt_r_city_target_from_location(current_loc)
            self._handle_indoor_during_entry_route(w, current_loc, "导航/进门过程中检测到 indoor")
            return

        self.indoor_stuck_frames = 0

        if self.initial_target_pending:
            stable_loc = self._get_stable_initial_location(current_loc)
            if stable_loc is None:
                self.stop_auto_forward(w)
                w.refresh_frame()
                return
            current_loc = stable_loc
            self.initial_target_pending = False

        distance_to_r_city, _ = self._distance_to_r_city(current_loc)
        if (
            distance_to_r_city is not None
            and distance_to_r_city > self.r_city_near_distance
        ):
            if not self._is_walkable(current_loc):
                self._handle_forbidden_escape(w, current_loc, current_direction)
                return
            self._handle_route_to_r_city(w, current_loc, current_direction)
            return

        if self.current_house_id is None:
            self._select_next_r_city_house(current_loc, current_direction)

            if not self.current_house_id:
                self._finish_r_city_searching(w, "R城房点已全部处理或均不可进入")
                return

            self.status = "FAST_NAV"
            target_dist = get_distance(current_loc, self.active_entry["location"])
            print(
                f"[RCitySearch] 锁定目标: {self.current_house_id} | "
                f"靠近点={self.active_entry['location']} | 距离={target_dist:.2f}"
            )
            self.history_locations = []

        target_loc = self.active_entry["location"]
        dist = get_distance(current_loc, target_loc)

        if self._maybe_start_entry_for_nearby_r_city_body(w, current_loc):
            return

        if self.status == "FAST_NAV":
            if self.update_and_check_stuck(current_loc):
                print("[SceneSearch] 快速导航检测到卡住，启动避障")
                if not self.execute_unstuck_logic(w, current_loc):
                    self._mark_current_r_city_target_failed("快速导航卡住避障失败")
                    self.status = "IDLE"
                self.history_locations = []
                return

            if dist <= self.ENTRY_AUTO_FORWARD_DISTANCE:
                print(f"[RCitySearch] 进入摇杆分段导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            if self._maybe_switch_to_front_r_city_house(w, current_loc):
                return

            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "FAST_NAV"):
                return

            self.align_direction(w, target_loc)

            if not self.auto_forward:
                w.click("自动前进")
                self.auto_forward = True

            self.handle_jump_logic(w)
            return

        if self.status == "PRECISE_NAV":
            if self.update_and_check_stuck(current_loc):
                print("[SceneSearch] 精细导航检测到卡住，启动避障")
                if not self.execute_unstuck_logic(w, current_loc):
                    self._mark_current_r_city_target_failed("精细导航卡住避障失败")
                    self.status = "IDLE"
                self.history_locations = []
                return

            if self._should_start_r_city_entry(current_loc, house_scene, dist):
                print(
                    f"[RCitySearch] 已到达房点/墙门附近 "
                    f"(dist={dist:.2f}, house_scene={house_scene})，进入 house_scene 进门流程"
                )
                self.stop_auto_forward(w)
                self.status = self.STATUS_SCENE_ENTRY
                return

            if self._maybe_switch_to_front_r_city_house(w, current_loc):
                return

            if self._maybe_bypass_front_house_on_route(w, current_loc, target_loc, dist, "PRECISE_NAV"):
                return

            self.stop_auto_forward(w)
            if not self._move_precisely_to_entry_point(w, current_loc, target_loc, dist):
                self.align_direction(w, target_loc)
                y_bias, dura, wait = self._get_entry_move_params(dist)
                mode = self._entry_forward_mode(dist)
                print(
                    f"[SceneSearch] 分段推进到进门点: "
                    f"dist={dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
                )
                self._tap_entry_forward_with_learning(w, target_loc, dist, mode, y_bias, dura, wait)
            self.handle_jump_logic(w)
            return

        if self.status == self.STATUS_SCENE_ENTRY:
            entry_result = self._enter_house_by_scene(w)
            if entry_result is None:
                return
            if not entry_result:
                if self._should_abort(w):
                    return
                print("[RCitySearch] house_scene 进门失败，临时跳过当前R城房点")
                self._mark_current_r_city_target_failed("house_scene 进门失败")
                self.status = "IDLE"
                return

            self._complete_current_house_search(w, "house_scene 进门成功")

    def _load_r_city_area_config(self):
        try:
            with open(self.R_CITY_AREA_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[RCitySearch] R城配置读取失败，使用空配置: {exc}")
            return {}

    def _load_r_city_center(self):
        center = self.r_city_config.get("geometry", {}).get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return (int(center[0]), int(center[1]))
            except (TypeError, ValueError):
                pass
        return self.R_CITY_FALLBACK_CENTER

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
        for index, point in enumerate(self.r_city_config.get("points", []), start=1):
            loc = self._location_tuple(point.get("location") or point.get("raw_location"))
            if loc is None:
                continue
            approach_loc = self._resolve_r_city_approach_location(loc)
            nearest_entry = point.get("nearest_existing_entry") or {}
            entry_direction = self._resolve_r_city_entry_direction(
                approach_loc,
                nearest_entry.get("direction"),
            )
            target = {
                "id": str(point.get("id") or f"r_city_{index:03d}"),
                "location": loc,
                "approach_location": approach_loc,
                "side": str(point.get("side") or self._side_from_location(loc)),
                "quality": str(point.get("quality") or "accepted"),
                "existing_house_id": nearest_entry.get("house_id"),
                "entry_direction": entry_direction,
                "nearest_existing_entry": nearest_entry,
            }
            targets.append(target)
        return targets

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
        self.r_city_failed_counts = {}
        self.current_r_city_target = None
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0
        self.r_city_route_stuck_cycles = 0
        self.forbidden_escape_target = None
        self.water_escape_side = None
        self.water_escape_side_attempts = 0
        self.water_escape_total_attempts = 0
        self.water_escape_last_loc = None

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
            print(f"[RCitySearch] 地图可通行检查失败，按可通行处理: {exc}")
            return True

    def _distance_to_r_city(self, current_loc):
        loc = self._location_tuple(current_loc)
        if loc is None or not self.r_city_targets:
            return None, None
        best = min(
            self.r_city_targets,
            key=lambda item: get_distance(loc, item["location"]),
        )
        return get_distance(loc, best["location"]), best

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
            print(f"[RCitySearch] 规划路径失败: start={start_loc}, target={target_loc}, err={exc}")
            return []
        if not path:
            return []
        result = []
        for point in path:
            loc = self._location_tuple(point)
            if loc is not None:
                result.append(loc)
        return result

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
        if distance_to_r_city is not None and distance_to_r_city <= self.r_city_near_distance:
            print(f"[RCityRoute] 已进入R城附近 dist={distance_to_r_city:.2f}，开始选房")
            self.stop_auto_forward(w)
            self.status = "IDLE"
            self.r_city_route_target = None
            self.r_city_route_path = []
            self.r_city_route_index = 0
            return

        if not self.r_city_route_target or not self.r_city_route_path:
            self.r_city_route_target = self._select_r_city_route_target(current_loc)
            if not self.r_city_route_target and nearest:
                self.r_city_route_target = nearest
            if not self.r_city_route_target:
                self._finish_r_city_searching(w, "无法选择R城接入点")
                return
            if not self.r_city_route_path:
                self.r_city_route_path = self._plan_path_safe(
                    current_loc,
                    self.r_city_route_target["approach_location"],
                )
                self.r_city_route_index = 0
            print(
                f"[RCityRoute] 规划去R城: side={self.r_city_route_target['side']}, "
                f"target={self.r_city_route_target['approach_location']}, "
                f"path_points={len(self.r_city_route_path)}"
            )

        waypoint = self._current_r_city_route_waypoint(current_loc)
        if waypoint is None:
            self.r_city_route_target = None
            self.r_city_route_path = []
            self.status = "IDLE"
            return

        if self.update_and_check_stuck(current_loc):
            self.r_city_route_stuck_cycles += 1
            print(
                f"[RCityRoute] 前往R城卡住 "
                f"{self.r_city_route_stuck_cycles}/{self.R_CITY_ROUTE_REPLAN_STUCK_CYCLES}"
            )
            self.stop_auto_forward(w)
            if self.r_city_route_stuck_cycles >= self.R_CITY_ROUTE_REPLAN_STUCK_CYCLES:
                self.r_city_route_target = None
                self.r_city_route_path = []
                self.r_city_route_stuck_cycles = 0
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
            w.click("自动前进")
            self.auto_forward = True
        self.handle_jump_logic(w)

    def _current_r_city_route_waypoint(self, current_loc):
        if not self.r_city_route_path:
            return None
        while self.r_city_route_index < len(self.r_city_route_path):
            waypoint = self.r_city_route_path[self.r_city_route_index]
            if get_distance(current_loc, waypoint) > self.R_CITY_ROUTE_WAYPOINT_DISTANCE:
                return waypoint
            self.r_city_route_index += 1
        return None

    def _handle_forbidden_escape(self, w: "FrameWorker", current_loc, current_direction):
        safe_target = self.forbidden_escape_target
        if safe_target is None or get_distance(current_loc, safe_target) <= self.FORBIDDEN_ESCAPE_ARRIVAL_DISTANCE:
            finder = getattr(self.map_tool, "nearest_walkable_within_radius", None)
            if callable(finder):
                safe_target, _ = finder(current_loc, self.FORBIDDEN_ESCAPE_SEARCH_RADIUS)
            safe_target = self._location_tuple(safe_target)
            self.forbidden_escape_target = safe_target

        if safe_target is None:
            print("[RCityRoute] 当前不可通行且未找到安全点，短后退后刷新")
            self.stop_auto_forward(w)
            w.tap_single("摇杆", y_bias=300, dura=450, wait=850)
            w.refresh_frame()
            return

        print(f"[RCityRoute] 当前不在可通行区域，先脱离到安全点 {safe_target}")
        self.stop_auto_forward(w)
        self.align_direction(w, safe_target)
        w.tap_single(
            "摇杆",
            y_bias=-300,
            dura=self.FORBIDDEN_ESCAPE_FORWARD_DURA,
            wait=self.FORBIDDEN_ESCAPE_FORWARD_WAIT,
        )
        w.refresh_frame()

    def _is_in_water(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("上浮"))

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
        side = self._choose_water_escape_side(current_loc, target_loc, current_direction)
        side_label = self._side_label(side)
        x_bias = -self.WATER_SIDE_X_BIAS if side == "left" else self.WATER_SIDE_X_BIAS

        self.stop_auto_forward(w)
        before_loc = self._location_tuple(current_loc)
        print(
            f"[RCityWater] 落水/水边受阻，沿{side_label}侧岸线脱困 "
            f"attempt={self.water_escape_total_attempts + 1}, target={target_loc}"
        )

        if w.get_info("上浮"):
            w.click("上浮")
            time.sleep(self.WATER_FLOAT_DURA / 1000.0)
            w.refresh_frame()

        w.tap_single("摇杆", y_bias=320, dura=self.WATER_BACK_DURA, wait=self.WATER_BACK_WAIT)
        w.refresh_frame()
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=0,
            dura=self.WATER_SIDE_DURA,
            wait=self.WATER_SIDE_WAIT,
        )
        w.refresh_frame()
        self.align_direction(w, target_loc)
        w.tap_single(
            "摇杆",
            y_bias=self.WATER_FORWARD_Y_BIAS,
            dura=self.WATER_FORWARD_DURA,
            wait=self.WATER_FORWARD_WAIT,
        )
        w.refresh_frame()

        after_loc = self._get_current_location(w) or before_loc
        self._record_water_escape_attempt(before_loc, after_loc)
        if self.water_escape_total_attempts >= self.WATER_ESCAPE_MAX_ATTEMPTS and self._finish_callback_configured():
            print("[RCityWater] 水边脱困多次失败，交给跑图阶段继续处理")
            self._finish_searching_phase(w, "R城搜房水边脱困失败")

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
        moved = 0.0 if before is None or after is None else get_distance(before, after)
        self.water_escape_total_attempts += 1
        self.water_escape_side_attempts += 1
        self.water_escape_last_loc = after
        print(
            f"[RCityWater] 水边脱困反馈: side={self.water_escape_side}, "
            f"moved={moved:.2f}, same_side_attempts={self.water_escape_side_attempts}"
        )
        if moved >= self.WATER_ESCAPE_STUCK_DISTANCE:
            return
        if self.water_escape_side_attempts >= self.WATER_ESCAPE_SIDE_SWITCH_ATTEMPTS:
            print("[RCityWater] 同方向多次几乎无位移，下轮换另一侧沿岸尝试")

    def _select_next_r_city_house(self, current_loc, current_direction):
        loc = self._location_tuple(current_loc)
        candidates = [
            item for item in self.r_city_targets
            if item["id"] not in self.r_city_completed_targets
            and self.r_city_failed_counts.get(item["id"], 0) < self.R_CITY_FAILED_TARGET_LIMIT
        ]
        if not candidates or loc is None:
            self.current_house_id = None
            self.current_r_city_target = None
            self.active_entry = None
            return

        def score(target):
            dist = get_distance(loc, target["approach_location"])
            angle_penalty = 0.0
            target_angle = calculate_angle(loc, target["approach_location"])
            _, _, diff = calculate_move_count(current_direction, target_angle)
            if diff is not None:
                angle_penalty = diff / 18.0
            failure_penalty = self.r_city_failed_counts.get(target["id"], 0) * 15.0
            return dist + angle_penalty + failure_penalty

        target = min(candidates, key=score)
        self._lock_r_city_target(target)

    def _lock_r_city_target(self, target):
        self.current_r_city_target = target
        self.current_house_id = target["id"]
        self.active_entry = {
            "location": target["approach_location"],
            "direction": target["entry_direction"],
            "r_city_target_id": target["id"],
        }
        self.r_city_route_target = None
        self.r_city_route_path = []
        self.r_city_route_index = 0

    def _maybe_start_entry_for_nearby_r_city_body(self, w: "FrameWorker", current_loc) -> bool:
        target, distance = self._nearest_r_city_body_target(
            current_loc,
            self.R_CITY_BODY_ENTRY_DISTANCE,
        )
        if target is None:
            return False

        if self.current_house_id != target["id"]:
            print(
                f"[RCitySearch] 人物已贴近另一栋R城房体，改锁 {target['id']} "
                f"body_dist={distance:.2f}"
            )
            self._lock_r_city_target(target)
        else:
            print(
                f"[RCitySearch] 人物已贴近当前R城房体 {target['id']} "
                f"body_dist={distance:.2f}"
            )

        self.stop_auto_forward(w)
        self.align_direction(
            w,
            target["location"],
            threshold=self.R_CITY_ENTRY_TARGET_ALIGN_TOLERANCE,
            max_steps=2,
            wait=self.R_CITY_BODY_ENTRY_ALIGN_WAIT,
        )
        self.status = self.STATUS_SCENE_ENTRY
        self.history_locations = []
        return True

    def _nearest_r_city_body_target(self, current_loc, max_distance: float):
        loc = self._location_tuple(current_loc)
        if loc is None:
            return None, None

        candidates = []
        for target in self.r_city_targets:
            target_id = target["id"]
            if target_id in self.r_city_completed_targets:
                continue
            if self.r_city_failed_counts.get(target_id, 0) >= self.R_CITY_FAILED_TARGET_LIMIT:
                continue
            distance = get_distance(loc, target["location"])
            if distance <= max_distance:
                candidates.append((distance, target))

        if not candidates:
            return None, None
        distance, target = min(candidates, key=lambda item: item[0])
        return target, distance

    def _mark_current_r_city_target_failed(self, reason: str):
        if self.current_r_city_target:
            target_id = self.current_r_city_target["id"]
            self.r_city_failed_counts[target_id] = self.r_city_failed_counts.get(target_id, 0) + 1
            print(
                f"[RCitySearch] {reason}: {target_id} "
                f"fail={self.r_city_failed_counts[target_id]}/{self.R_CITY_FAILED_TARGET_LIMIT}"
            )
        elif self.current_house_id:
            print(f"[RCitySearch] {reason}: {self.current_house_id}")
        self.temp_skip_houses.add(self.current_house_id)
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.history_locations = []

    def _should_start_r_city_entry(self, current_loc, house_scene, dist: float) -> bool:
        if dist <= self.r_city_house_arrival_distance:
            return True
        if (
            house_scene in self.HOUSE_NEAR_ENTRY_SCENES
            and self.current_r_city_target
            and get_distance(current_loc, self.current_r_city_target["location"])
            <= self.r_city_early_entry_scene_distance
        ):
            return True
        return False

    def _maybe_switch_to_front_r_city_house(self, w: "FrameWorker", current_loc) -> bool:
        scene = self._get_house_scene(w)
        if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
            return False
        loc = self._location_tuple(current_loc)
        if loc is None:
            return False
        candidates = [
            item for item in self.r_city_targets
            if item["id"] not in self.r_city_completed_targets
            and item["id"] != self.current_house_id
            and self.r_city_failed_counts.get(item["id"], 0) < self.R_CITY_FAILED_TARGET_LIMIT
            and get_distance(loc, item["location"]) <= self.R_CITY_FORWARD_HOUSE_BYPASS_DISTANCE
        ]
        if not candidates:
            return False
        target = min(candidates, key=lambda item: get_distance(loc, item["location"]))
        current_dist = (
            get_distance(loc, self.current_r_city_target["location"])
            if self.current_r_city_target else float("inf")
        )
        next_dist = get_distance(loc, target["location"])
        if current_dist <= 5 and next_dist >= current_dist:
            return False
        print(
            f"[RCitySearch] 途中已贴近另一栋R城房子，临时改搜 "
            f"{target['id']} dist={next_dist:.2f}"
        )
        self._lock_r_city_target(target)
        self.status = self.STATUS_SCENE_ENTRY
        self.stop_auto_forward(w)
        return True

    def _adopt_r_city_target_from_location(self, current_loc):
        if self.current_r_city_target or self.current_house_id:
            return
        loc = self._location_tuple(current_loc)
        if loc is None:
            return
        candidates = [
            item for item in self.r_city_targets
            if item["id"] not in self.r_city_completed_targets
            and get_distance(loc, item["location"]) <= self.r_city_early_entry_scene_distance
        ]
        if not candidates:
            return
        target = min(candidates, key=lambda item: get_distance(loc, item["location"]))
        print(f"[RCitySearch] 室内状态下匹配到R城房点 {target['id']}")
        self._lock_r_city_target(target)

    def _finish_r_city_searching(self, w: "FrameWorker", reason: str):
        self.stop_auto_forward(w)
        print(f"[RCitySearch] {reason}，切换到跑图阶段")
        self.current_house_id = None
        self.current_r_city_target = None
        self.active_entry = None
        self.status = "IDLE"
        return self._finish_searching_phase(w, reason)

    def _is_entry_approach_status(self):
        return super()._is_entry_approach_status() or self.status == self.STATUS_SCENE_ENTRY

    def _should_start_search_from_indoor(self) -> bool:
        return (
            self.current_house_id is not None
            and self.current_house_id not in self.r_city_completed_targets
        )

    def _confirm_indoor_before_search(self, w: "FrameWorker", reason: str) -> bool:
        if not w.get_info("跳跃"):
            return True

        print(f"[SceneSearch] {reason}，且检测到跳跃按钮，先按翻窗逻辑确认")
        self.handle_jump_logic(w)
        w.refresh_frame()
        if self._is_indoor(w):
            print("[SceneSearch] 跳跃后仍为 indoor，确认已进房")
            return True

        print(f"[SceneSearch] 跳跃后 house_scene={self._get_house_scene(w)}，暂不启动旋转搜房")
        return False

    def _complete_current_house_search(self, w: "FrameWorker", reason: str) -> bool:
        if self._should_abort(w):
            return False

        self.stop_auto_forward(w)
        self.indoor_stuck_frames = 0
        print(f"[SceneSearch] {reason}")

        if not self.start_searching(w):
            return False
        if w.current_stage != "搜房阶段":
            return False

        if self.current_house_id is not None:
            self.completed_houses.add(self.current_house_id)
            self.r_city_completed_targets.add(self.current_house_id)
        self.searching_number += 1
        print(
            f"[RCitySearch] 房屋 {self.current_house_id} 完成，"
            f"已搜 {len(self.r_city_completed_targets)}/{len(self.r_city_targets)}"
        )
        w.refresh_frame()
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
        print("[SceneSearch] 已搜完或无待搜目标时检测到 indoor，优先执行出房")
        if self._exit_house(w):
            self.indoor_stuck_frames = 0
            self.current_house_id = None
            self.status = "IDLE"
            self._continue_searching_until_timer(w, "意外进房后已出房")

    def _move_precisely_to_entry_point(self, w: "FrameWorker", current_loc, target_loc, dist: float) -> bool:
        current_dir = w.get_info("direction")
        target_angle = calculate_angle(current_loc, target_loc)
        turn_dir, _, diff = calculate_move_count(current_dir, target_angle)
        if diff is None or turn_dir is None:
            return False

        align_threshold = 2 if dist < 5 else 5
        align_max_steps = 3 if dist < 5 else 1
        aligned = self.align_direction(
            w,
            target_loc,
            threshold=align_threshold,
            max_steps=align_max_steps,
        )
        if not aligned:
            print(
                f"[SceneSearch] 距离进门点 {dist:.2f}，角度未对准 "
                f"threshold={align_threshold}，本轮不前推"
            )
            return True
        mode = self._entry_forward_mode(dist)
        y_bias, dura, wait = self._get_entry_move_params(dist)
        print(
            f"[SceneSearch] {self._entry_forward_mode_label(mode)}到进门点: "
            f"dist={dist:.2f}, target_angle={target_angle}, diff={diff:.1f}, "
            f"y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        return self._tap_entry_forward_with_learning(w, target_loc, dist, mode, y_bias, dura, wait)

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
    ) -> bool:
        before_dist = self._get_current_entry_distance(w, target_loc)
        if before_dist is None:
            before_dist = desired_dist

        previous_dist = before_dist
        for step in range(self.ENTRY_FORWARD_MAX_STEPS):
            if self._should_abort(w):
                return False

            current_dist = self._get_current_entry_distance(w, target_loc)
            if current_dist is None:
                current_dist = previous_dist
            if current_dist <= self.ENTRY_ARRIVAL_DISTANCE:
                return True

            desired_step_dist = max(0.2, float(current_dist))
            y_bias, dura, wait, distance_key = get_adaptive_forward_motion(
                mode,
                desired_step_dist,
                fallback_y_bias,
                fallback_dura,
                fallback_wait,
            )

            print(
                f"[SceneSearch] 执行{self._entry_forward_mode_label(mode)}小步 "
                f"{step + 1}/{self.ENTRY_FORWARD_MAX_STEPS}: model_dist={distance_key}, "
                f"before={current_dist:.2f}, y_bias={y_bias}, dura={dura}, wait={wait}"
            )
            w.tap_single("摇杆", y_bias=y_bias, dura=dura, wait=wait)
            w.refresh_frame()

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
            print(
                f"[SceneSearch] 推进反馈: mode={mode}, model_dist={distance_key}, "
                f"after={after_dist:.2f}, moved={moved:.2f}"
            )
            if after_dist <= self.ENTRY_ARRIVAL_DISTANCE or moved <= 0:
                break
            previous_dist = after_dist

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

        print(
            f"[SceneSearch] 执行{self._side_label(side)}滑微调: "
            f"bin={distance_key}, before={before_dist:.2f}, "
            f"x_bias={x_bias}, dura={dura}, wait={wait}"
        )
        w.tap_single("摇杆", x_bias=x_bias, y_bias=0, dura=dura, wait=wait)
        w.refresh_frame()

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
            print(
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
        raw = w.get_info("location")
        if not raw:
            return None

        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            current_loc = check_location(raw[0])
        else:
            current_loc = check_location(raw)
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

        print("[SceneRotate] 进入房屋，启动 house_scene 旋转搜房")
        rotate_result = self._rotate_search_inside_house(w)

        if self._should_abort(w):
            return False

        w.refresh_frame()
        if rotate_result == self.ROTATE_RESULT_EXITED or self._is_out_of_house(w):
            print("[SceneRotate] 旋转搜房过程中已出房，房屋搜索完成")
            return True

        if rotate_result == self.ROTATE_RESULT_FALLBACK_EXIT:
            print("[SceneRotate] 两轮撞墙循环仍未自然出房，开始执行出房策略")
        else:
            print("[SceneRotate] 旋转搜房结束，开始出房")

        if self._exit_house(w):
            print("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True
        if self._should_abort(w):
            return False

        w.refresh_frame()
        if self._is_out_of_house(w):
            print("[SceneRotate] 出房策略成功，房屋搜索完成")
            return True

        print(f"[SceneRotate] 出房策略后仍未确认出房 house_scene={self._get_house_scene(w)}")
        return False

    def _house_search_timed_out(self):
        return False

    def _rotate_search_inside_house(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        print("[SceneRotate] 室内搜房改为固定推进转向：顺时针 6 次，逆时针 6 次")

        search_plan = (
            ("顺时针", "left_up", self.ROTATE_SEARCH_SWEEP_TURN_PX),
            ("逆时针", "right_up", -self.ROTATE_SEARCH_SWEEP_TURN_PX),
        )
        step_index = 0
        for phase_label, move_mode, turn_px in search_plan:
            for phase_step in range(self.ROTATE_SEARCH_SWEEP_CYCLES_PER_DIRECTION):
                if self._should_abort(w):
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_FINISHED

                w.refresh_frame()
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    print(f"[SceneRotate] {phase_label}推进前已出房 house_scene={scene}")
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_EXITED
                if scene is not None and scene not in {
                    self.HOUSE_INDOOR,
                    self.HOUSE_NEAR_DOOR,
                    self.HOUSE_NEAR_WALL,
                }:
                    print(f"[SceneRotate] 当前 house_scene={scene}，停止室内旋转搜房")
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_FINISHED

                step_index += 1
                print(
                    f"[SceneRotate] {phase_label}推进 {phase_step + 1}/"
                    f"{self.ROTATE_SEARCH_SWEEP_CYCLES_PER_DIRECTION}，全局步数={step_index}"
                )
                self._move_rotate_search_sweep_step(w, move_mode, phase_label)

                w.refresh_frame()
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    print(f"[SceneRotate] {phase_label}推进后判定出房 house_scene={scene}")
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_EXITED

                print(f"[SceneRotate] {phase_label}推进后原地转视角 {turn_px}px")
                self._turn_raw_pixels(w, turn_px)

                w.refresh_frame()
                scene = self._get_house_scene(w)
                if scene in self.HOUSE_EXIT_SCENES:
                    print(f"[SceneRotate] {phase_label}转向后判定出房 house_scene={scene}")
                    self.stop_auto_forward(w)
                    return self.ROTATE_RESULT_EXITED

        self.stop_auto_forward(w)
        print("[SceneRotate] 顺时针+逆时针各一圈后仍未出房，切出房策略")
        return self.ROTATE_RESULT_FALLBACK_EXIT

    def _ensure_rotate_auto_forward(self, w: "FrameWorker", reason: str):
        if self.auto_forward:
            return
        print(f"[SceneRotate] {reason}")
        w.click("自动前进")
        self.auto_forward = True
        w.refresh_frame()

    def _recover_rotate_auto_forward_stuck(self, w: "FrameWorker", turn_sign: int):
        if turn_sign > 0:
            x_bias = -self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "左"
        else:
            x_bias = self.ROTATE_SEARCH_RECOVER_X_BIAS
            label = "右"

        print(
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
        w.refresh_frame()

    def _move_rotate_search_step(self, w: "FrameWorker", move_mode: str):
        x_bias = self._rotate_move_x_bias(move_mode)
        label = self._move_mode_label(move_mode)
        print(f"[SceneRotate] 向{label}滑动 {self.ROTATE_SEARCH_MOVE_DURA}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.ROTATE_SEARCH_MOVE_DURA,
            wait=self.ROTATE_SEARCH_MOVE_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _move_rotate_search_sweep_step(self, w: "FrameWorker", move_mode: str, phase_label: str):
        x_bias = self._rotate_move_x_bias(move_mode)
        label = self._move_mode_label(move_mode)
        print(
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
        w.refresh_frame()

    def _turn_raw_pixels(self, w: "FrameWorker", signed_px: int):
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
                print(f"[SceneRotate] 左上累计撞{label}已达{wall_hit_count}次，立即切到右上，并改为向左补转")
                self._turn_until_not_near_entry(w, -1)
                return "right_up", 0, True

            print(f"[SceneRotate] 右上累计撞{label}已达{wall_hit_count}次，立即切到左上，并改为向右补转")
            self._turn_until_not_near_entry(w, 1)
            return "left_up", 0, True

        turn_sign = self._move_mode_turn_sign(current_mode)
        turn_label = "向右" if turn_sign > 0 else "向左"
        print(f"[SceneRotate] 撞{label}后{turn_label}补转，直到不再贴墙/门")
        self._turn_until_not_near_entry(w, turn_sign)
        return move_mode, wall_hit_count, False

    def _turn_until_not_near_entry(self, w: "FrameWorker", turn_sign: int) -> bool:
        for attempt in range(self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            base_index = min(attempt, len(self.ROTATE_SEARCH_WALL_TURN_SEQUENCE) - 1)
            angle = self.ROTATE_SEARCH_WALL_TURN_SEQUENCE[base_index]
            signed_angle = turn_sign * angle
            print(
                f"[SceneRotate] 撞墙补转 {attempt + 1}/"
                f"{self.ROTATE_SEARCH_WALL_TURN_MAX_ATTEMPTS}: {signed_angle}°"
            )
            self._turn_with_direction_correction(w, signed_angle)
            w.refresh_frame()
            scene = self._get_house_scene(w)
            if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[SceneRotate] 补转后 house_scene={scene}，继续当前滑动方向")
                return True

            print(f"[SceneRotate] 补转后仍贴墙/门 house_scene={scene}，继续缩小角度补转")

        print("[SceneRotate] 多次补转后仍贴墙/门，交给下一轮移动继续尝试")
        return False

    def _turn_with_direction_correction(self, w: "FrameWorker", signed_angle: float):
        before_dir = self._direction_as_float(w.get_info("direction"))
        self._turn(w, signed_angle)
        w.refresh_frame()
        if before_dir is None:
            print("[SceneRotate] 当前 direction 无效，跳过本次补角到位校验")
            return

        target_dir = (before_dir + float(signed_angle)) % 360.0
        for step in range(self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS):
            current_dir = self._direction_as_float(w.get_info("direction"))
            turn_dir, _, diff = calculate_move_count(current_dir, target_dir)
            if diff is None or turn_dir is None:
                print("[SceneRotate] 补角后 direction 无效，无法继续校验角度")
                return
            if diff <= self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
                return

            correction = min(float(diff), self.ROTATE_SEARCH_TURN_CORRECT_MAX_DEGREES)
            signed_correction = correction if turn_dir == "right" else -correction
            print(
                f"[SceneRotate] 补角未到位，二次校正 {step + 1}/"
                f"{self.ROTATE_SEARCH_TURN_CORRECT_MAX_STEPS}: "
                f"current={current_dir:.1f}, target={target_dir:.1f}, "
                f"remaining={diff:.1f}, turn={signed_correction:.1f}°"
            )
            self._turn(w, signed_correction)
            w.refresh_frame()

        current_dir = self._direction_as_float(w.get_info("direction"))
        _, _, remaining = calculate_move_count(current_dir, target_dir)
        if remaining is not None and remaining > self.ROTATE_SEARCH_TURN_CORRECT_THRESHOLD:
            print(
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

        print(f"[SceneRotate] 两帧过于相似，判定卡住，向{label}水平脱困 {dura_ms}ms")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=0,
            dura=dura_ms,
            wait=dura_ms + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _exit_house(self, w: "FrameWorker") -> bool:
        print("[SceneExit] 统一委托 HouseExitManager 执行出房")
        self.stop_auto_forward(w)
        self.house_exit_manager.reset()
        for attempt in range(self.EXIT_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                return False
            if self.house_exit_manager.process(w):
                print(f"[SceneExit] HouseExitManager 出房成功 attempt={attempt + 1}")
                return True
            w.refresh_frame()
            if self._is_out_of_house(w):
                print(f"[SceneExit] HouseExitManager 后确认已出房 attempt={attempt + 1}")
                self.house_exit_manager.reset()
                return True
        print("[SceneExit] HouseExitManager 达到尝试上限，仍未确认出房")
        return self._is_out_of_house(w)

    def _exit_house_by_scene_strategy(self, w: "FrameWorker") -> bool:
        print("[SceneExit] 启动 house_scene 多路径出房策略")
        move_mode = "left_up"
        wall_hit_count = 0

        for step in range(self.EXIT_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 出房策略开始前已判定在屋外")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

            button_state = self._door_button_state(w)
            if button_state and self._exit_via_door_button(w, button_state):
                return True

            label = self._move_mode_label(move_mode)
            print(f"[SceneExit] {label}绕圈找出口 {step + 1}/{self.EXIT_SEARCH_MAX_STEPS}")
            self._move_exit_search_step(w, move_mode)
            if self._is_out_of_house(w):
                print(f"[SceneExit] {label}滑动时意外出房，出房成功")
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
                print(
                    f"[SceneExit] 累计撞墙/门 {wall_hit_count}/"
                    f"{self.ROTATE_SEARCH_HIT_SWITCH_COUNT}, mode={move_mode}"
                )
                if wall_hit_count >= self.ROTATE_SEARCH_HIT_SWITCH_COUNT:
                    move_mode = self._opposite_move_mode(move_mode)
                    wall_hit_count = 0
                    label = self._move_mode_label(move_mode)
                    print(f"[SceneExit] 撞墙/门达到阈值，切换为{label}逆向绕圈")

            turn_sign = self._move_mode_turn_sign(move_mode)
            turn_label = "向右" if turn_sign > 0 else "向左"
            print(
                f"[SceneExit] {self._move_mode_label(move_mode)}后{turn_label}调整视角 "
                f"{self.EXIT_SEARCH_TURN_DEGREES}° 继续绕圈"
            )
            self._turn(w, turn_sign * self.EXIT_SEARCH_TURN_DEGREES)
            w.refresh_frame()

        print("[SceneExit] 多路径出房策略达到步数上限，仍未确认出房")
        return self._is_out_of_house(w)

    def _move_exit_search_step(self, w: "FrameWorker", move_mode: str):
        w.tap_single(
            "摇杆",
            x_bias=self._rotate_move_x_bias(move_mode),
            y_bias=self.ROTATE_SEARCH_Y_BIAS,
            dura=self.EXIT_SEARCH_LEFT_UP_DURA,
            wait=self.EXIT_SEARCH_LEFT_UP_DURA + self.ROTATE_SEARCH_MOVE_WAIT_PAD,
        )
        w.refresh_frame()

    def _exit_via_door_button(self, w: "FrameWorker", button_state: str) -> bool:
        if button_state == "open":
            print("[SceneExit] 发现开门按钮，点击开门后尝试出门")
            w.click("开门")
            time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
            w.refresh_frame()
        else:
            print("[SceneExit] 发现关门按钮，门已打开，直接尝试出门")

        return self._exit_open_door_by_diagonal_sweep(w)

    def _exit_open_door_by_diagonal_sweep(self, w: "FrameWorker") -> bool:
        for step in range(self.EXIT_DOOR_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 门口推进前已在屋外")
                return True

            if self._door_button_state(w) == "open":
                print("[SceneExit] 门口推进前再次看到开门按钮，补点一次开门")
                w.click("开门")
                time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
                w.refresh_frame()

            side = "left" if step % 2 == 0 else "right"
            dura = min(
                self.ENTRY_OPEN_SWEEP_BASE_DURA + step * self.ENTRY_OPEN_SWEEP_STEP_MS,
                self.ENTRY_OPEN_SWEEP_MAX_DURA,
            )
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            print(f"[SceneExit] 门已打开，向{self._side_label(side)}上小步尝试出门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.ENTRY_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.ENTRY_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            if self._is_out_of_house(w):
                print("[SceneExit] 左上/右上门口推进后出房成功")
                return True

            window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if window and self._exit_via_window_by_scene(w, window):
                return True

        print("[SceneExit] 门口左上/右上推进到上限，未确认出房")
        return False

    def _exit_via_window_by_scene(self, w: "FrameWorker", window) -> bool:
        rel_angle = self._target_relative_angle(window)
        print(f"[SceneExit] 发现窗户，准备对齐 rel_angle={rel_angle}")
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
                print(f"[SceneExit] 窗户已对齐 rel_angle={rel_angle:.1f}")
                return "aligned"

            turn_angle = max(
                -self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES,
                min(self.EXIT_WINDOW_ALIGN_MAX_STEP_DEGREES, rel_angle),
            )
            side = "右" if turn_angle > 0 else "左"
            print(
                f"[SceneExit] 窗户在{side}侧，对齐 {step + 1}/"
                f"{self.EXIT_WINDOW_ALIGN_MAX_STEPS}: turn={turn_angle:.1f}"
            )
            self._turn(w, turn_angle)
            w.refresh_frame()

            refreshed = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
            if not refreshed:
                print("[SceneExit] 对齐窗户时目标丢失，改为前推找跳跃按钮")
                return "lost"
            target = refreshed

        print("[SceneExit] 窗户对齐达到步数上限，按已接近窗户处理")
        return "aligned"

    def _push_until_jump_and_exit_window(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[SceneExit] {reason}，最多前推 {self.EXIT_WINDOW_FORWARD_MAX_STEPS} 次找跳跃")
        for step in range(self.EXIT_WINDOW_FORWARD_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_out_of_house(w):
                print("[SceneExit] 靠窗前推前已出房")
                return True

            if w.get_info("跳跃"):
                if self._jump_forward_exit_window(w, step + 1):
                    return True
                continue

            print(f"[SceneExit] 靠窗前推找跳跃 {step + 1}/{self.EXIT_WINDOW_FORWARD_MAX_STEPS}")
            w.tap_single(
                "摇杆",
                y_bias=self.EXIT_WINDOW_FORWARD_Y_BIAS,
                dura=self.EXIT_WINDOW_FORWARD_DURA,
                wait=self.EXIT_WINDOW_FORWARD_WAIT,
            )
            w.refresh_frame()

            if self._is_out_of_house(w):
                print("[SceneExit] 靠窗前推时意外出房")
                return True

            if w.get_info("跳跃") and self._jump_forward_exit_window(w, step + 1):
                return True

        print("[SceneExit] 靠窗前推 3 次仍未出现可用跳跃，放弃该窗户")
        return False

    def _jump_forward_exit_window(self, w: "FrameWorker", step: int) -> bool:
        print(f"[SceneExit] 检测到跳跃按钮，尝试翻窗出房 step={step}")
        w.click("跳跃")
        time.sleep(self.ENTRY_WINDOW_JUMP_SETTLE_SECONDS)
        w.tap_single(
            "摇杆",
            y_bias=self.EXIT_WINDOW_JUMP_FORWARD_Y_BIAS,
            dura=self.EXIT_WINDOW_JUMP_FORWARD_DURA,
            wait=self.EXIT_WINDOW_JUMP_FORWARD_WAIT,
        )
        w.refresh_frame()
        if self._is_out_of_house(w):
            print("[SceneExit] 翻窗后出房成功")
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

    def _enter_r_city_house_by_free_search(self, w: "FrameWorker") -> Optional[bool]:
        self.stop_auto_forward(w)
        print("[RCityEntry] R城房点是房体坐标，启用自由找门/窗/按钮进房")

        for step in range(self.R_CITY_ENTRY_FREE_SEARCH_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_indoor(w):
                print("[RCityEntry] 已是 indoor，直接开始屋内搜索")
                return True

            if self._handle_r_city_visible_entry_opportunity(w):
                return True

            self._align_to_r_city_house_point(w)
            scene = self._get_house_scene(w)
            print(f"[RCityEntry] 自由进房轮次 {step + 1}: house_scene={scene}")

            if scene in self.HOUSE_NEAR_ENTRY_SCENES:
                if self._sweep_r_city_wall_for_entry(w):
                    return True
                continue

            if self._push_r_city_entry_forward_and_check_indoor(w, "未见门窗按钮，朝房体盲前推"):
                return True

            scene = self._get_house_scene(w)
            if scene in self.HOUSE_NEAR_ENTRY_SCENES and self._sweep_r_city_wall_for_entry(w):
                return True

        print("[RCityEntry] 多轮自由找门/窗/按钮仍未进房")
        return self._is_indoor(w)

    def _align_to_r_city_house_point(self, w: "FrameWorker") -> bool:
        target_loc = self._r_city_house_body_location()
        if target_loc is None:
            print("[RCityEntry] 当前R城房点缺少坐标，跳过朝房体对齐")
            return False
        print(f"[RCityEntry] 朝房体坐标对齐: {target_loc}")
        return self.align_direction(
            w,
            target_loc,
            threshold=self.R_CITY_ENTRY_TARGET_ALIGN_TOLERANCE,
            max_steps=2,
        )

    def _r_city_house_body_location(self):
        if self.current_r_city_target:
            return self.current_r_city_target.get("location")
        if self.active_entry:
            return self.active_entry.get("location")
        return None

    def _handle_r_city_visible_entry_opportunity(self, w: "FrameWorker") -> bool:
        if self._is_indoor(w):
            return True

        jump_result = self._press_r_city_jump_if_visible(w, "看到跳跃按钮，立即按翻窗/越障逻辑")
        if jump_result is not None:
            return jump_result or self._push_r_city_entry_forward_and_check_indoor(
                w,
                "跳跃后继续前推确认",
            )

        button_state = self._door_button_state(w)
        if button_state:
            return self._handle_r_city_door_button_then_forward(w, button_state)

        stone_wall = self._find_largest_forward_target(w, self.STONE_WALL_CLASS_IDS)
        if stone_wall is not None:
            return self._handle_r_city_stone_wall_entry(w)

        door = self.find_largest_door(w)
        if door is not None:
            print("[RCityEntry] 画面中发现门，先对门再前推找按钮")
            self._align_to_r_city_forward_target(w, door, "门")
            if self._push_r_city_entry_forward_and_check_indoor(w, "对门后前推"):
                return True
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True
            button_state = self._door_button_state(w)
            if button_state:
                return self._handle_r_city_door_button_then_forward(w, button_state)
            if self._get_house_scene(w) in self.HOUSE_NEAR_ENTRY_SCENES:
                print("[RCityEntry] 对门前推后撞墙，后拉调角并重新锁门")
                return self._recover_r_city_wall_hit_and_relock(
                    w,
                    label="门",
                    target_finder=self.find_largest_door,
                    push_reason="重锁门后再次前推",
                )

        window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
        if window is not None:
            print("[RCityEntry] 画面中发现窗，先对窗再短前推找跳跃")
            self._align_to_r_city_forward_target(w, window, "窗")
            if self._push_r_city_entry_forward_and_check_indoor(
                w,
                "对窗后短前推",
                y_bias=self.R_CITY_ENTRY_WINDOW_FORWARD_Y_BIAS,
                dura=self.R_CITY_ENTRY_WINDOW_FORWARD_DURA,
                wait=self.R_CITY_ENTRY_WINDOW_FORWARD_WAIT,
            ):
                return True
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True
            if self._get_house_scene(w) in self.HOUSE_NEAR_ENTRY_SCENES:
                print("[RCityEntry] 对窗短前推后撞墙，后拉调角并重新锁窗")
                return self._recover_r_city_wall_hit_and_relock(
                    w,
                    label="窗",
                    target_finder=lambda frame: self._find_largest_forward_target(
                        frame,
                        self.EXIT_WINDOW_CLASS_IDS,
                    ),
                    push_reason="重锁窗后短前推",
                    y_bias=self.R_CITY_ENTRY_WINDOW_FORWARD_Y_BIAS,
                    dura=self.R_CITY_ENTRY_WINDOW_FORWARD_DURA,
                    wait=self.R_CITY_ENTRY_WINDOW_FORWARD_WAIT,
                )

        return False

    def _handle_r_city_immediate_entry_opportunity(self, w: "FrameWorker") -> bool:
        if self._is_indoor(w):
            return True

        jump_result = self._press_r_city_jump_if_visible(w, "看到跳跃按钮，立即点击")
        if jump_result is not None:
            return jump_result or self._push_r_city_entry_forward_and_check_indoor(
                w,
                "跳跃后继续前推确认",
            )

        button_state = self._door_button_state(w)
        if button_state:
            return self._handle_r_city_door_button_then_forward(w, button_state)

        return False

    def _press_r_city_jump_if_visible(self, w: "FrameWorker", reason: str) -> Optional[bool]:
        if not w.get_info("跳跃"):
            return None
        print(f"[RCityEntry] {reason}")
        self.handle_jump_logic(w)
        w.refresh_frame()
        if self._is_indoor(w):
            print("[RCityEntry] 点击跳跃后 house_scene=indoor，进房成功")
            return True
        if self._get_house_scene(w) == self.HOUSE_NEAR_WALL:
            return self._recover_r_city_near_wall_after_jump(w)
        return False

    def _recover_r_city_near_wall_after_jump(self, w: "FrameWorker") -> bool:
        print("[RCityEntry] 跳跃后仍 near_wall，先后拉离墙再重新查看门/窗/indoor")
        w.tap_single(
            "摇杆",
            y_bias=self.R_CITY_ENTRY_JUMP_WALL_BACKOFF_Y_BIAS,
            dura=self.R_CITY_ENTRY_JUMP_WALL_BACKOFF_DURA,
            wait=self.R_CITY_ENTRY_JUMP_WALL_BACKOFF_WAIT,
        )
        w.refresh_frame()

        if self._is_indoor(w):
            print("[RCityEntry] 跳跃后后拉复查 house_scene=indoor，进房成功")
            return True

        button_state = self._door_button_state(w)
        if button_state:
            return self._handle_r_city_door_button_then_forward(w, button_state)

        door = self.find_largest_door(w)
        if door is not None:
            print("[RCityEntry] 跳跃后后拉重新看到门，重新对门前推")
            self._align_to_r_city_forward_target(w, door, "门")
            return self._push_r_city_entry_forward_and_check_indoor(w, "跳跃后后拉重看门前推")

        window = self._find_largest_forward_target(w, self.EXIT_WINDOW_CLASS_IDS)
        if window is not None:
            print("[RCityEntry] 跳跃后后拉重新看到窗，重新对窗短前推")
            self._align_to_r_city_forward_target(w, window, "窗")
            return self._push_r_city_entry_forward_and_check_indoor(
                w,
                "跳跃后后拉重看窗短前推",
                y_bias=self.R_CITY_ENTRY_WINDOW_FORWARD_Y_BIAS,
                dura=self.R_CITY_ENTRY_WINDOW_FORWARD_DURA,
                wait=self.R_CITY_ENTRY_WINDOW_FORWARD_WAIT,
            )

        return False

    def _handle_r_city_stone_wall_entry(self, w: "FrameWorker") -> bool:
        print("[RCityEntry] 前方识别到 stone_wall，前推、跳跃、再前推")
        if self._push_r_city_entry_forward_and_check_indoor(
            w,
            "stone_wall 前第一次前推",
            y_bias=self.R_CITY_ENTRY_STONE_WALL_FORWARD_Y_BIAS,
            dura=self.R_CITY_ENTRY_STONE_WALL_FORWARD_DURA,
            wait=self.R_CITY_ENTRY_STONE_WALL_FORWARD_WAIT,
        ):
            return True

        w.click("跳跃")
        time.sleep(self.R_CITY_ENTRY_STONE_WALL_JUMP_SETTLE_SECONDS)
        w.refresh_frame()

        return self._push_r_city_entry_forward_and_check_indoor(
            w,
            "stone_wall 跳跃后第二次前推",
            y_bias=self.R_CITY_ENTRY_STONE_WALL_FORWARD_Y_BIAS,
            dura=self.R_CITY_ENTRY_STONE_WALL_FORWARD_DURA,
            wait=self.R_CITY_ENTRY_STONE_WALL_FORWARD_WAIT,
        )

    def _handle_r_city_door_button_then_forward(self, w: "FrameWorker", button_state: str) -> bool:
        self.stop_auto_forward(w)
        if button_state == "open":
            print("[RCityEntry] 看到开门按钮，点击开门后前推")
            self._click_open_door(w)
        else:
            print("[RCityEntry] 看到关门按钮，说明门已打开，不点击关门，直接前推")
        return self._push_r_city_entry_forward_and_check_indoor(w, "门按钮处理后前推")

    def _align_to_r_city_forward_target(self, w: "FrameWorker", target, label: str) -> bool:
        relative_angle = self._target_relative_angle(target)
        if relative_angle is None:
            print(f"[RCityEntry] {label}目标角度无效，跳过视觉对齐")
            return False
        if abs(relative_angle) <= self.R_CITY_ENTRY_TARGET_ALIGN_TOLERANCE:
            print(f"[RCityEntry] {label}已在视野中央附近")
            return True

        limited_angle = max(-25.0, min(25.0, relative_angle))
        print(f"[RCityEntry] {label}偏移 {relative_angle:.1f}°，微调 {limited_angle:.1f}°")
        self._turn(w, limited_angle)
        time.sleep(0.12)
        w.refresh_frame()
        return True

    def _push_r_city_entry_forward_and_check_indoor(
        self,
        w: "FrameWorker",
        reason: str,
        y_bias: Optional[int] = None,
        dura: Optional[int] = None,
        wait: Optional[int] = None,
    ) -> bool:
        print(f"[RCityEntry] {reason}")
        w.tap_single(
            "摇杆",
            y_bias=self.R_CITY_ENTRY_BLIND_FORWARD_Y_BIAS if y_bias is None else y_bias,
            dura=self.R_CITY_ENTRY_BLIND_FORWARD_DURA if dura is None else dura,
            wait=self.R_CITY_ENTRY_BLIND_FORWARD_WAIT if wait is None else wait,
        )
        w.refresh_frame()

        if self._is_indoor(w):
            print("[RCityEntry] 前推后 house_scene=indoor，进房成功")
            return True
        jump_result = self._press_r_city_jump_if_visible(w, "前推后出现跳跃按钮，立即点击")
        if jump_result is not None:
            return bool(jump_result)
        if self._door_button_state(w):
            print("[RCityEntry] 前推后出现门按钮，下一轮立即处理")
        return False

    def _recover_r_city_wall_hit_and_relock(
        self,
        w: "FrameWorker",
        label: str,
        target_finder,
        push_reason: str,
        y_bias: Optional[int] = None,
        dura: Optional[int] = None,
        wait: Optional[int] = None,
    ) -> bool:
        for attempt in range(self.R_CITY_ENTRY_WALL_RELOCK_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True

            print(
                f"[RCityEntry] 撞墙恢复 {attempt + 1}/"
                f"{self.R_CITY_ENTRY_WALL_RELOCK_MAX_ATTEMPTS}: 先后拉离墙"
            )
            w.tap_single(
                "摇杆",
                y_bias=self.R_CITY_ENTRY_WALL_RELOCK_BACKOFF_Y_BIAS,
                dura=self.R_CITY_ENTRY_WALL_RELOCK_BACKOFF_DURA,
                wait=self.R_CITY_ENTRY_WALL_RELOCK_BACKOFF_WAIT,
            )
            w.refresh_frame()
            if self._is_indoor(w):
                return True
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True

            turn_px = self.R_CITY_ENTRY_WALL_RELOCK_TURN_PX
            if attempt % 2 == 1:
                turn_px = -turn_px
            print(f"[RCityEntry] 后拉后向{'右' if turn_px > 0 else '左'}调角 {abs(turn_px)}px")
            w.tap_single(
                "视角",
                x_bias=turn_px,
                dura=self.R_CITY_ENTRY_WALL_RELOCK_TURN_DURA,
                wait=self.R_CITY_ENTRY_WALL_RELOCK_TURN_WAIT,
            )
            w.refresh_frame()

            target = target_finder(w)
            if target is None:
                print(f"[RCityEntry] 调角后暂未重新看到{label}，继续下一次恢复")
                continue

            print(f"[RCityEntry] 调角后重新看到{label}，重新对准并尝试前进")
            self._align_to_r_city_forward_target(w, target, label)
            if self._push_r_city_entry_forward_and_check_indoor(
                w,
                push_reason,
                y_bias=y_bias,
                dura=dura,
                wait=wait,
            ):
                return True
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True
            if self._get_house_scene(w) not in self.HOUSE_NEAR_ENTRY_SCENES:
                return False
        return False

    def _sweep_r_city_wall_for_entry(
        self,
        w: "FrameWorker",
        small: bool = False,
        immediate_only: bool = False,
    ) -> bool:
        if small:
            print("[RCityEntry] 贴窗撞墙，左右小幅交替查找跳跃/开关门")
        else:
            print("[RCityEntry] 贴墙/门但未见按钮，左右交替短等待查找门窗/跳跃/开关门")

        x_bias_base = (
            self.R_CITY_ENTRY_WINDOW_WALL_SWEEP_X_BIAS
            if small else self.R_CITY_ENTRY_WALL_SWEEP_X_BIAS
        )
        dura = (
            self.R_CITY_ENTRY_WINDOW_WALL_SWEEP_DURA
            if small else self.R_CITY_ENTRY_WALL_SWEEP_DURA
        )
        for step in range(self.R_CITY_ENTRY_WALL_SWEEP_STEPS):
            if self._handle_r_city_immediate_entry_opportunity(w):
                return True
            side = "right" if step % 2 == 0 else "left"
            x_bias = x_bias_base if side == "right" else -x_bias_base
            side_label = self._side_label(side)
            wait = self.R_CITY_ENTRY_WALL_SWEEP_WAIT + step * self.R_CITY_ENTRY_WALL_SWEEP_WAIT_STEP
            if self._should_abort(w):
                return False
            print(
                f"[RCityEntry] 贴墙向{side_label}侧移 "
                f"{step + 1}/{self.R_CITY_ENTRY_WALL_SWEEP_STEPS}, wait={wait}"
            )
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=0,
                dura=dura,
                wait=wait,
            )
            w.refresh_frame()
            if self._is_indoor(w):
                print("[RCityEntry] 侧移后已进房")
                return True
            if immediate_only:
                entry_handled = self._handle_r_city_immediate_entry_opportunity(w)
            else:
                entry_handled = self._handle_r_city_visible_entry_opportunity(w)
            if entry_handled:
                return True

            scene = self._get_house_scene(w)
            if scene not in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[RCityEntry] 侧移后 house_scene={scene}，停止本轮贴墙交替侧移")
                return False
        return False

    def _enter_house_by_scene(self, w: "FrameWorker") -> Optional[bool]:
        self.stop_auto_forward(w)
        if self.current_r_city_target:
            return self._enter_r_city_house_by_free_search(w)

        if self.active_entry:
            ideal_angle = self.active_entry["direction"]
            print(f"[SceneEntry] 调整至进门方向: {ideal_angle}")
            aligned = self.align_direction_blocking(
                w,
                w.get_info("direction"),
                ideal_angle,
                threshold=self.ENTRY_DIRECTION_ALIGN_TOLERANCE,
                max_steps=self.ENTRY_DIRECTION_ALIGN_MAX_STEPS,
            )
            if not aligned:
                print("[SceneEntry] 进门方向尚未对准，继续对准后再探门")
                return None

        w.refresh_frame()
        if self._is_indoor(w):
            print("[SceneEntry] 已是 indoor，直接开始屋内搜索")
            return True

        self._align_visible_entry_door_before_push(w)
        if self._is_indoor(w):
            print("[SceneEntry] 对齐视觉门后已是 indoor，直接开始屋内搜索")
            return True

        approach_result = self._approach_until_near_entry(w, force_first_forward=True)
        if approach_result == "indoor":
            print("[SceneEntry] 前推/跳跃后已确认 indoor，直接开始屋内搜索")
            return True
        if approach_result in {"open", "close"}:
            return self._handle_entry_door_button_then_forward(w, approach_result, "前推过程中")
        if approach_result != "near":
            return False

        self._realign_to_entry_direction(w, "未发现门按钮，左右震荡探门前")
        button_state = self._sweep_for_door_button(w)
        if button_state == "indoor":
            print("[SceneEntry] 左右探门时检测到 indoor，确认已进房")
            return True
        if button_state in {"open", "close"}:
            return self._handle_entry_door_button_then_forward(w, button_state, "左右探门时")

        print("[SceneEntry] 左右探测未找到开门/关门按钮")
        return False

    def _handle_entry_door_button_then_forward(
        self,
        w: "FrameWorker",
        button_state: str,
        reason: str,
    ) -> bool:
        self.stop_auto_forward(w)
        if button_state == "open":
            print(f"[SceneEntry] {reason}发现开门按钮，点击开门后停止贴门前推")
            self._click_open_door(w)
            opened_now = True
        else:
            print(f"[SceneEntry] {reason}发现关门按钮，门已打开，不点击关门，停止贴门前推")
            opened_now = False

        self._realign_to_entry_direction(w, "门按钮出现后")

        if self._enter_after_door_button_relock_loop(w, reason, opened_now):
            return True

        print("[SceneEntry] 多次后拉、重锁门、短前推仍未 indoor，严格进门流程失败")
        return False

    def _enter_after_door_button_relock_loop(
        self,
        w: "FrameWorker",
        reason: str,
        opened_now: bool,
    ) -> bool:
        for attempt in range(self.ENTRY_OPEN_DOOR_RELOCK_MAX_ATTEMPTS):
            if self._should_abort(w):
                return False

            if attempt > 0:
                relock_result = self._backoff_and_relock_entry_door(w, reason, attempt)
                if relock_result == "indoor":
                    return True
                if relock_result != "aligned":
                    return False
            if self._is_indoor(w):
                print("[SceneEntry] 后拉/重锁门后已是 indoor，直接确认入房")
                return True

            if attempt == 0 and not opened_now:
                self._realign_to_entry_direction(w, "关门按钮出现后")

            push_dura, push_wait = self._entry_door_relock_push_motion(attempt)
            print(
                f"[SceneEntry] 门按钮后尝试进房 {attempt + 1}/"
                f"{self.ENTRY_OPEN_DOOR_RELOCK_MAX_ATTEMPTS}: "
                f"dura={push_dura}, wait={push_wait}"
            )
            if self._push_entry_forward_and_check_indoor(
                w,
                f"{reason}门按钮后第{attempt + 1}次前推",
                push_dura,
                push_wait,
            ):
                return True

            scene = self._get_house_scene(w)
            if scene not in {
                self.HOUSE_INDOOR,
                self.HOUSE_NEAR_DOOR,
                self.HOUSE_NEAR_WALL,
                self.HOUSE_OUTDOOR,
                None,
            }:
                print(f"[SceneEntry] 前推后 house_scene={scene}，停止重锁门循环")
                return False

        return False

    def _push_entry_forward_and_check_indoor(self, w: "FrameWorker", reason: str, dura: int, wait: int) -> bool:
        print(f"[SceneEntry] {reason}，前推确认入房")
        w.tap_single(
            "摇杆",
            y_bias=self.ENTRY_INDOOR_CONFIRM_FORWARD_Y_BIAS,
            dura=dura,
            wait=wait,
        )
        w.refresh_frame()

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_INDOOR:
            print("[SceneEntry] 前推后 house_scene=indoor，确认已进房")
            return True

        print(f"[SceneEntry] 前推后 house_scene={scene}，未进入 indoor")
        return False

    def _backoff_and_relock_entry_door(self, w: "FrameWorker", reason: str, attempt: int) -> str:
        backoff_dura, backoff_wait = self._entry_door_relock_backoff_motion(attempt)
        print(
            f"[SceneEntry] {reason}前推未进房，后拉后重新锁定门框 "
            f"{attempt + 1}/{self.ENTRY_OPEN_DOOR_RELOCK_MAX_ATTEMPTS}: "
            f"dura={backoff_dura}, wait={backoff_wait}"
        )
        w.tap_single(
            "摇杆",
            y_bias=self.ENTRY_OPEN_DOOR_RELOCK_BACKOFF_Y_BIAS,
            dura=backoff_dura,
            wait=backoff_wait,
        )
        w.refresh_frame()

        if self._is_indoor(w):
            print("[SceneEntry] 后拉复核时已是 indoor，直接确认入房")
            return "indoor"

        door = self._find_entry_door_after_backoff(w)
        if isinstance(door, str) and door == "indoor":
            return "indoor"
        if door is None:
            print("[SceneEntry] 后拉后仍未重新识别到门，停止本轮严格进门流程")
            return "failed"

        print("[SceneEntry] 后拉后重新锁定当前门，先对齐门框再前推")
        self._align_to_door_detection(
            w,
            door,
            tolerance_px=self.ENTRY_OPEN_DOOR_RELOCK_TOLERANCE_PX,
        )
        w.refresh_frame()

        if self._is_indoor(w):
            print("[SceneEntry] 重新锁门对齐后已是 indoor")
            return "indoor"

        return "aligned"

    def _find_entry_door_after_backoff(self, w: "FrameWorker"):
        door = self.find_largest_door(w)
        for step in range(self.ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_STEPS):
            if door is not None or self._should_abort(w):
                break

            print(
                f"[SceneEntry] 后拉后未看到门，再后拉一点寻找门框 "
                f"{step + 1}/{self.ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_STEPS}"
            )
            w.tap_single(
                "摇杆",
                y_bias=self.ENTRY_OPEN_DOOR_RELOCK_BACKOFF_Y_BIAS,
                dura=self.ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_DURA,
                wait=self.ENTRY_OPEN_DOOR_RELOCK_EXTRA_BACKOFF_WAIT,
            )
            w.refresh_frame()
            if self._is_indoor(w):
                return "indoor"
            door = self.find_largest_door(w)

        return door

    def _entry_door_relock_push_motion(self, attempt: int):
        dura = self._sequence_value(self.ENTRY_OPEN_DOOR_RELOCK_PUSH_DURAS, attempt)
        wait = self._sequence_value(self.ENTRY_OPEN_DOOR_RELOCK_PUSH_WAITS, attempt)
        dura = max(self.ENTRY_OPEN_DOOR_SHORT_PUSH_MIN_DURA, int(dura))
        wait = max(dura, int(wait))
        return dura, wait

    def _entry_door_relock_backoff_motion(self, attempt: int):
        index = max(0, int(attempt) - 1)
        dura = self._sequence_value(self.ENTRY_OPEN_DOOR_RELOCK_BACKOFF_DURAS, index)
        wait = self._sequence_value(self.ENTRY_OPEN_DOOR_RELOCK_BACKOFF_WAITS, index)
        dura = max(1, int(dura))
        wait = max(dura, int(wait))
        return dura, wait

    @staticmethod
    def _sequence_value(values, index: int):
        if not values:
            return 0
        index = max(0, min(int(index), len(values) - 1))
        return values[index]

    def _realign_to_entry_direction(self, w: "FrameWorker", reason: str) -> bool:
        if not self.active_entry:
            return False

        ideal_angle = self.active_entry["direction"]
        print(f"[SceneEntry] {reason}，重新修正至进门方向: {ideal_angle}")
        aligned = self.align_direction_blocking(
            w,
            w.get_info("direction"),
            ideal_angle,
            threshold=self.ENTRY_DIRECTION_ALIGN_TOLERANCE,
            max_steps=self.ENTRY_DIRECTION_ALIGN_MAX_STEPS,
        )
        if not aligned:
            print("[SceneEntry] 门按钮后进门方向未完全对准，仍按当前方向尝试一次正前推")
        return aligned

    def _align_visible_entry_door_before_push(self, w: "FrameWorker") -> bool:
        door = self.find_largest_door(w)
        if door is None:
            print("[SceneEntry] 进门方向上未识别到门，直接按进门方向前推")
            return False

        print("[SceneEntry] 进门方向上识别到门，先粗略对齐门再前推")
        aligned = self._align_to_door_detection(
            w,
            door,
            tolerance_px=self.ENTRY_VISIBLE_DOOR_ALIGN_TOLERANCE_PX,
        )
        w.refresh_frame()
        if not aligned:
            print("[SceneEntry] 视觉门粗对齐未完全成功，继续按当前方向前推试探")
        return aligned

    def _approach_until_near_entry(self, w: "FrameWorker", force_first_forward: bool = False) -> str:
        for step in range(self.ENTRY_APPROACH_MAX_STEPS):
            if self._should_abort(w):
                return "abort"

            w.refresh_frame()
            if self._is_indoor(w):
                print("[SceneEntry] 前推前已检测到 indoor")
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state and not (force_first_forward and step == 0):
                return button_state

            scene = self._get_house_scene(w)
            if scene in self.HOUSE_NEAR_ENTRY_SCENES and not (force_first_forward and step == 0):
                print(f"[SceneEntry] 已到门/墙附近 house_scene={scene}")
                return "near"

            print(
                f"[SceneEntry] 正对门前推 {step + 1}/{self.ENTRY_APPROACH_MAX_STEPS}, "
                f"house_scene={scene}"
            )
            if w.get_info("跳跃"):
                self.handle_jump_logic(w)
                w.refresh_frame()
                if self._is_indoor(w):
                    print("[SceneEntry] 前推过程中跳跃后变为 indoor，确认翻窗进房")
                    return "indoor"
            else:
                w.tap_single(
                    "摇杆",
                    y_bias=self.ENTRY_APPROACH_FORWARD_Y_BIAS,
                    dura=self.ENTRY_APPROACH_FORWARD_DURA,
                    wait=self.ENTRY_APPROACH_FORWARD_WAIT,
                )
                w.refresh_frame()
                if self._is_indoor(w):
                    print("[SceneEntry] 前推后变为 indoor，确认已进房")
                    return "indoor"

        w.refresh_frame()
        if self._is_indoor(w):
            return "indoor"
        if self._get_house_scene(w) in self.HOUSE_NEAR_ENTRY_SCENES:
            return "near"
        return self._door_button_state(w) or "failed"

    def _sweep_for_door_button(self, w: "FrameWorker") -> Optional[str]:
        blocked = {"left": False, "right": False}

        for step in range(self.BUTTON_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return None

            w.refresh_frame()
            if self._is_indoor(w):
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state:
                return button_state

            side = "left" if step % 2 == 0 else "right"
            if blocked[side]:
                print(f"[SceneEntry] {side} 侧已到 outdoor 临界点，跳过本次水平探测")
                continue

            dura = (step + 1) * self.SWEEP_STEP_MS
            x_bias = -self.BUTTON_SWEEP_X_BIAS if side == "left" else self.BUTTON_SWEEP_X_BIAS
            print(f"[SceneEntry] 水平向{self._side_label(side)}探门 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=0,
                dura=dura,
                wait=dura + self.BUTTON_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            if self._is_indoor(w):
                return "indoor"

            button_state = self._door_button_state(w)
            if button_state:
                return button_state

            scene = self._get_house_scene(w)
            if scene == self.HOUSE_OUTDOOR:
                blocked[side] = True
                print(f"[SceneEntry] 向{self._side_label(side)}已离开墙/门范围，记录临界点")

            if blocked["left"] and blocked["right"]:
                print("[SceneEntry] 左右两侧都到达 outdoor 临界点，停止探门")
                return None

        return None

    def _enter_open_door_by_diagonal_sweep(self, w: "FrameWorker") -> bool:
        for step in range(self.ENTRY_SWEEP_MAX_STEPS):
            if self._should_abort(w):
                return False

            w.refresh_frame()
            if self._is_indoor(w):
                if self._confirm_indoor_by_forward_push(w, "小步推进前检测到 indoor"):
                    return True
                continue

            if self._door_button_state(w) == "open":
                print("[SceneEntry] 小步推进前再次看到开门按钮，补点一次开门")
                self._click_open_door(w)

            side = "left" if step % 2 == 0 else "right"
            dura = min(
                self.ENTRY_OPEN_SWEEP_BASE_DURA + step * self.ENTRY_OPEN_SWEEP_STEP_MS,
                self.ENTRY_OPEN_SWEEP_MAX_DURA,
            )
            x_bias = -self.ENTRY_SWEEP_X_BIAS if side == "left" else self.ENTRY_SWEEP_X_BIAS
            print(f"[SceneEntry] 门已打开，向{self._side_label(side)}上小步推进 {dura}ms")
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.ENTRY_SWEEP_Y_BIAS,
                dura=dura,
                wait=dura + self.ENTRY_SWEEP_WAIT_PAD,
            )
            w.refresh_frame()

            scene = self._get_house_scene(w)
            if scene == self.HOUSE_INDOOR:
                if self._confirm_indoor_by_forward_push(w, "左上/右上小步推进后检测到 indoor"):
                    return True
                continue

            if scene in self.HOUSE_NEAR_ENTRY_SCENES:
                print(f"[SceneEntry] 小步推进后仍贴墙/门 house_scene={scene}，继续换边")
            elif scene == self.HOUSE_OUTDOOR:
                print("[SceneEntry] 小步推进后变为 outdoor，先反向斜后回拉，再换边进门")
                if self._backoff_from_outdoor_side(w, side, dura):
                    return True

        print("[SceneEntry] 左上/右上小步推进到上限，仍未进入 indoor")
        return False

    def _backoff_from_outdoor_side(self, w: "FrameWorker", side: str, dura_ms: Optional[int] = None) -> bool:
        opposite_x = self.ENTRY_SWEEP_X_BIAS if side == "left" else -self.ENTRY_SWEEP_X_BIAS
        try:
            backoff_dura = int(dura_ms or self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA)
        except (TypeError, ValueError):
            backoff_dura = self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA
        backoff_dura = min(
            max(self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_DURA, backoff_dura),
            self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_MAX_DURA,
        )
        opposite_side = "right" if side == "left" else "left"
        print(
            f"[SceneEntry] outdoor 临界点回拉，向{self._side_label(opposite_side)}后 "
            f"{backoff_dura}ms 回到墙/门附近"
        )
        w.tap_single(
            "摇杆",
            x_bias=opposite_x,
            y_bias=self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_Y_BIAS,
            dura=backoff_dura,
            wait=backoff_dura + self.ENTRY_OPEN_SWEEP_OUTDOOR_BACKOFF_WAIT,
        )
        w.refresh_frame()

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_INDOOR:
            return self._confirm_indoor_by_forward_push(w, "outdoor 回退后检测到 indoor")
        print(f"[SceneEntry] outdoor 回拉后 house_scene={scene}，下一步换边继续尝试")
        return False

    def _confirm_indoor_by_forward_push(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[SceneEntry] {reason}，前推确认入房")
        if w.get_info("跳跃"):
            print("[SceneEntry] indoor 信号伴随跳跃按钮，按翻窗逻辑点击跳跃后前推")
            w.click("跳跃")
            time.sleep(self.ENTRY_WINDOW_JUMP_SETTLE_SECONDS)

        w.tap_single(
            "摇杆",
            y_bias=self.ENTRY_INDOOR_CONFIRM_FORWARD_Y_BIAS,
            dura=self.ENTRY_INDOOR_CONFIRM_FORWARD_DURA,
            wait=self.ENTRY_INDOOR_CONFIRM_FORWARD_WAIT,
        )
        w.refresh_frame()

        scene = self._get_house_scene(w)
        if scene == self.HOUSE_INDOOR:
            print("[SceneEntry] 前推后 house_scene=indoor，确认已进房")
            return True

        print(f"[SceneEntry] 前推后 house_scene={scene}，indoor 信号未确认")
        return False

    def _door_button_state(self, w: "FrameWorker") -> Optional[str]:
        if w.get_info("开门"):
            return "open"
        if w.get_info("关门"):
            return "close"
        return None

    def _click_open_door(self, w: "FrameWorker"):
        print("[SceneEntry] 检测到开门按钮，点击开门")
        w.click("开门")
        time.sleep(self.OPEN_DOOR_SETTLE_SECONDS)
        w.refresh_frame()

    def _is_indoor(self, w: "FrameWorker") -> bool:
        return self._get_house_scene(w) == self.HOUSE_INDOOR

    def _is_out_of_house(self, w: "FrameWorker") -> bool:
        return self._get_house_scene(w) in self.HOUSE_EXIT_SCENES

    @staticmethod
    def _move_mode_label(move_mode: str) -> str:
        return "左上" if move_mode == "left_up" else "右上"

    @staticmethod
    def _side_label(side: str) -> str:
        return "左" if side == "left" else "右"
