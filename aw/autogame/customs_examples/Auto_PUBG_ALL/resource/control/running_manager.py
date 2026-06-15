import json
import math
import os
import time
from typing import Callable, List, Optional, Set, Tuple, TYPE_CHECKING

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager import HouseExitManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    calculate_angle,
    calculate_move_count,
    check_location,
    execute_view_turn,
    get_distance,
    get_adaptive_forward_motion,
    is_location_stagnant, draw_points_with_arrows,
    update_adaptive_forward_motion,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils import find_path, get_resolution
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.timing import (
    ActiveWindow,
    Cooldown,
    Stopwatch,
    TimeoutTracker,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


RESOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROAD_DIR = os.path.join(RESOURCE_DIR, "road")


class RoadRouteHelper:
    """道路点路径助手。

    优先使用 road_topology 的拓扑路径；如果当前工程缺少 road_matrix/road_mask，
    则退化为读取红/蓝道路点，并用 MapNavigator 规划到下一个道路点。
    """

    INTERSECTION_NODE_FILES = ("red_coords.json",)
    ROUTE_NODE_FILES = ("blue_coords.json",)
    PATH_SAMPLE_INTERVAL = 30.0

    def __init__(self, map_tool: MapNavigator):
        self.map_tool = map_tool
        self._nodes: Optional[List[Tuple[int, int]]] = None
        self._intersection_nodes: Optional[List[Tuple[int, int]]] = None
        self._topo = None
        self._topo_load_attempted = False

    def get_nodes(self) -> List[Tuple[int, int]]:
        if self._nodes is not None:
            return self._nodes

        self._nodes = self._load_nodes(self.INTERSECTION_NODE_FILES + self.ROUTE_NODE_FILES)
        print(f"[RoadRoute] 已加载道路点 {len(self._nodes)} 个")
        return self._nodes

    def get_intersection_nodes(self) -> List[Tuple[int, int]]:
        if self._intersection_nodes is not None:
            return self._intersection_nodes

        self._intersection_nodes = self._load_nodes(self.INTERSECTION_NODE_FILES)
        print(f"[RoadRoute] 已加载道路红色 node {len(self._intersection_nodes)} 个")
        return self._intersection_nodes

    def topology_available(self) -> bool:
        return self._get_topo() is not None

    def _load_nodes(self, filenames) -> List[Tuple[int, int]]:
        nodes: List[Tuple[int, int]] = []
        seen = set()
        for filename in filenames:
            path = os.path.join(ROAD_DIR, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[RoadRoute] 读取道路点失败: {path}, err={exc}")
                continue

            for value in data.values():
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    continue
                point = (int(value[0]), int(value[1]))
                if point in seen:
                    continue
                seen.add(point)
                nodes.append(point)
        return nodes

    def nearest_node(
        self,
        point: Tuple[int, int],
        exclude: Optional[Set[Tuple[int, int]]] = None,
        min_distance: float = 0.0,
        topology_only: bool = False,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        nodes = self.get_intersection_nodes() if topology_only else self.get_nodes()
        if not nodes:
            return None, float("inf")

        exclude = exclude or set()
        candidates = [
            node for node in nodes
            if node not in exclude and get_distance(point, node) >= min_distance
        ]
        if not candidates:
            return None, float("inf")

        node = min(candidates, key=lambda item: get_distance(point, item))
        return node, get_distance(point, node)

    def center_biased_node(
        self,
        point: Tuple[int, int],
        center: Tuple[int, int],
        exclude: Optional[Set[Tuple[int, int]]] = None,
        min_distance: float = 0.0,
        topology_only: bool = False,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        nodes = self.get_intersection_nodes() if topology_only else self.get_nodes()
        if not nodes:
            return None, float("inf")

        exclude = exclude or set()
        candidates = [
            node for node in nodes
            if node not in exclude and get_distance(point, node) >= min_distance
        ]
        if not candidates:
            return None, float("inf")

        node = min(
            candidates,
            key=lambda item: (
                get_distance(item, center),
                get_distance(point, item),
            ),
        )
        return node, get_distance(point, node)

    def plan_to_node(
        self,
        start: Tuple[int, int],
        node: Tuple[int, int],
        allow_fallback: bool = True,
    ) -> List[Tuple[int, int]]:
        topo_path = self._try_topo_path(start, node)
        if topo_path:
            return self._sample_path(self._dedupe_path(topo_path))

        if not allow_fallback:
            return []

        planned = self.map_tool.plan_path(start, node)
        if not planned:
            planned = [node]
        elif tuple(map(int, planned[-1])) != node:
            planned.append(node)
        return self._sample_path(self._dedupe_path(planned))

    def plan_priority_road_path(
        self,
        start: Tuple[int, int],
        road_points: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]], float]:
        topo = self._get_topo()
        if topo is None:
            return [], None, float("inf")

        start_road_point, start_dist = topo.find_nearest_point_from_mask(start[0], start[1])
        if start_road_point is None:
            return [], None, float("inf")

        full_path: List[Tuple[int, int]] = []
        segment_start = tuple(map(int, start_road_point))
        for road_point in road_points:
            road_point = tuple(map(int, road_point))
            if get_distance(segment_start, road_point) <= 1.0:
                full_path = self._merge_dedupe_paths(full_path, [road_point])
                segment_start = road_point
                continue

            dest_key = self._find_topo_node_key(topo, road_point)
            if dest_key is None:
                print(f"[RoadRoute] 指定寻车点 {road_point} 不是 road_topology 红色节点")
                return [], segment_start, start_dist

            result = topo.shortest_path_from_point(
                int(segment_start[0]),
                int(segment_start[1]),
                int(dest_key[1:]) - 1,
            )
            if not result or len(result) < 3 or not result[2]:
                print(f"[RoadRoute] road_topology 无法规划 {segment_start} -> {road_point}")
                return [], segment_start, start_dist

            raw_segment = self._dedupe_path([tuple(map(int, point)) for point in result[2]])
            sampled_segment = self._sample_path(raw_segment)
            print(
                f"[RoadRoute] 指定寻车路段 {segment_start} -> {road_point}: "
                f"raw={len(raw_segment)}, sampled={len(sampled_segment)}, interval={self.PATH_SAMPLE_INTERVAL:.0f}"
            )
            full_path = self._merge_dedupe_paths(full_path, sampled_segment)
            if not full_path or full_path[-1] != road_point:
                full_path = self._merge_dedupe_paths(full_path, [road_point])
            segment_start = road_point

        return full_path, tuple(map(int, start_road_point)), float(start_dist)

    def _try_topo_path(self, start: Tuple[int, int], node: Tuple[int, int]) -> List[Tuple[int, int]]:
        topo = self._get_topo()
        if topo is None:
            return []

        dest_key = self._find_topo_node_key(topo, node)
        if dest_key is None:
            return []

        topo_nodes = [
            tuple(map(int, item))
            for item in (getattr(topo, "node_data", []) or []) + (getattr(topo, "route_node_data", []) or [])
        ]
        if not topo_nodes:
            return []

        start_node = min(topo_nodes, key=lambda item: get_distance(start, item))
        prefix = []
        if get_distance(start, start_node) > 1:
            prefix = self.map_tool.plan_path(start, start_node) or [start_node]

        try:
            result = topo.shortest_path_from_point(
                int(start_node[0]),
                int(start_node[1]),
                int(dest_key[1:]) - 1,
            )
        except Exception as exc:
            print(f"[RoadRoute] road_topology 规划失败，回退 A*: {exc}")
            return []

        if not result or len(result) < 3 or not result[2]:
            return []

        topo_path = [tuple(map(int, item)) for item in result[2]]
        return prefix + topo_path

    def _get_topo(self):
        if self._topo_load_attempted:
            return self._topo

        self._topo_load_attempted = True
        try:
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.road_topology import RoadTopo

            self._topo = RoadTopo()
            print("[RoadRoute] road_topology 拓扑加载成功")
        except Exception as exc:
            self._topo = None
            print(f"[RoadRoute] road_topology 拓扑不可用，使用道路点+A*兜底: {exc}")
        return self._topo

    def _find_topo_node_key(self, topo, node: Tuple[int, int]) -> Optional[str]:
        target = tuple(map(int, node))
        for key, value in getattr(topo, "node_dict", {}).items():
            if tuple(map(int, value)) == target:
                return key
        return None

    def _dedupe_path(self, path) -> List[Tuple[int, int]]:
        cleaned: List[Tuple[int, int]] = []
        for point in path or []:
            if point is None:
                continue
            item = tuple(map(int, point))
            if cleaned and cleaned[-1] == item:
                continue
            cleaned.append(item)
        return cleaned

    def _merge_dedupe_paths(self, path1, path2) -> List[Tuple[int, int]]:
        merged = self._dedupe_path(path1)
        for point in self._dedupe_path(path2):
            if not merged or merged[-1] != point:
                merged.append(point)
        return merged

    def _sample_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) <= 2:
            return path

        sampled = [path[0]]
        prev = path[0]
        distance_since_last = 0.0

        for point in path[1:-1]:
            distance_since_last += get_distance(prev, point)
            if distance_since_last >= self.PATH_SAMPLE_INTERVAL:
                sampled.append(point)
                distance_since_last = 0.0
            prev = point

        if sampled[-1] != path[-1]:
            sampled.append(path[-1])
        return sampled


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

    # 短时卡死判定窗口：连续多少帧位置不变算“卡住”
    STUCK_HISTORY_LEN = 10
    # 长时困死判定窗口：连续多少帧都在局部小范围打转算“困死”
    TRAPPED_HISTORY_LEN = 50
    # 距离目标点 10 内停止自动前进，改用短摇杆精确逼近；小于 5 才消费该路径点。
    WAYPOINT_TOLERANCE = 5.0
    WAYPOINT_PRECISE_APPROACH_DISTANCE = 10.0
    WAYPOINT_PRECISE_BIAS_Y = -120
    WAYPOINT_PRECISE_DURA = 180
    WAYPOINT_PRECISE_WAIT = 350
    UNSTUCK_MOVE_THRESHOLD = 0.8
    UNSTUCK_STEP_DURA = 360
    UNSTUCK_STEP_WAIT = 900
    UNSTUCK_TURN_BIAS = 360
    UNSTUCK_TURN_DURA = 420
    UNSTUCK_TURN_WAIT = 300
    WAYPOINT_PROJECTION_PASS_RATIO = 1.0
    WAYPOINT_PROJECTION_CORRIDOR = 12.0
    # 单帧位置跳变超过这个距离时，认为定位异常，需要重规划
    LOCATION_JUMP_THRESHOLD = 25.0
    # 位置跳变后，多久内不重复触发重规划
    LOCATION_JUMP_REPLAN_COOLDOWN = 1.5
    # OCR 偶发会把小地图坐标识别成左上角附近，直接用这种点规划会把路线带歪。
    LOCATION_MIN_VALID_COORD = 20

    # 不同时期的进圈目标距离，单位是地图坐标距离
    STAGE1_DIS = 600
    STAGE2_DIS = 400
    STAGE3_DIS = 220
    # 上面三段进圈距离各自对应的时间分界点，单位秒
    STAGE1_TIME = 11 * 60
    STAGE2_TIME = 16 * 60

    # R 城寻车的大致目标点
    R_CITY = (1136, 783)
    # 海岛地图中心附近。无进圈压力时，道路巡游优先往这里靠。
    MAP_CENTER = (1024, 1024)
    # 车库附近的精确上车点
    CAR_ENTRY_POINT = (1131, 763)
    # 距离车库上车点太远时，不再绕路去车库，直接切换沿路找车。
    GARAGE_SEARCH_MAX_DISTANCE = 50.0
    # 车库无车后，先准确离开到路边，再开始道路巡游找车。
    GARAGE_TO_ROADSIDE_POINTS = ((1134, 762), (1134, 771))
    GARAGE_TO_ROADSIDE_TOLERANCE = 2.0
    GARAGE_TO_ROADSIDE_FORWARD_BIAS_Y = -300
    GARAGE_TO_ROADSIDE_FORWARD_DURA = 300
    GARAGE_TO_ROADSIDE_FORWARD_WAIT = 5000
    # 历史保留字段，表示默认入库朝向
    CAR_FACE_DIRECTION = 265
    # 跑图/开车统一通过“人称”按钮切换视角。
    VIEW_SWITCH_BUTTON = "人称"
    VIEW_MODE_FIRST = "first"
    VIEW_MODE_THIRD = "third"
    # 入库失败后依次尝试的朝向序列
    PRECISE_FACE_DIRECTIONS = [265, 270, 275, 280, 285, 290]
    # 进圈角度变化超过这个阈值时，重新规划跑图路径
    CIRCLE_REPLAN_THRESHOLD = 10
    # 精调上车时，先向前顶车的摇杆偏移/持续时间/等待时间
    PRECISE_FORWARD_BIAS_Y = -220
    PRECISE_FORWARD_DURA = 550
    PRECISE_FORWARD_WAIT = 2500
    # 精调靠近车库上车点时也要有脱困保护，否则跑到车库背面会一直顶墙。
    PRECISE_ENTRY_STUCK_SWITCH_LIMIT = 3
    PRECISE_ENTRY_IDLE_UNSTUCK_ROUNDS = 4
    # 进入最后上车点附近后属于微调/找驾驶按钮，不再按位置不变判定卡死。
    PRECISE_ENTRY_MICRO_ADJUST_RADIUS = 5.0
    # 车库精调阶段方向值累计异常达到该次数后，不再继续车库找车。
    PRECISE_ENTRY_INVALID_DIRECTION_LIMIT = 5
    # 精调上车时，向右或向左单次小幅试探的摇杆参数
    PRECISE_LATERAL_STEP_BIAS = 150
    PRECISE_LATERAL_STEP_DURA = 220
    PRECISE_LATERAL_STEP_WAIT = 700
    # 右探后回到中间位置的大幅回位参数
    PRECISE_RESET_CENTER_BIAS = -300
    PRECISE_RESET_CENTER_DURA = 260
    PRECISE_RESET_CENTER_WAIT = 700
    # 视觉对车时的参数，沿用 searching_house 的“对准门”思路
    CAR_ALIGN_CENTER_THRESHOLD = 80
    CAR_ALIGN_STEP_RATIO = 0.33
    CAR_ALIGN_MAX_BIAS = 400
    CAR_ALIGN_DURA = 500
    CAR_ALIGN_WAIT = 500
    CAR_VISUAL_FORWARD_BIAS_Y = -220
    CAR_VISUAL_FORWARD_DURA = 320
    CAR_VISUAL_FORWARD_WAIT = 600
    CAR_VISUAL_SEARCH_MAX_STEPS = 4
    # 可切换实验方案：sendevent 持续前推摇杆，同时用 uinput 调整视角。
    # 默认回到原始视觉靠车前推方案；需要专项验证时再打开。
    CAR_APPROACH_USE_SENDEVENT_UINPUT = False
    CAR_APPROACH_FALLBACK_TO_LEGACY = True
    CAR_APPROACH_MIXED_MAX_STEPS = 10
    CAR_APPROACH_MIXED_MOVE_DURA = 80
    CAR_APPROACH_MIXED_CENTER_THRESHOLD = 45
    CAR_APPROACH_MIXED_VIEW_STEP_RATIO = 0.28
    CAR_APPROACH_MIXED_MAX_VIEW_BIAS = 260
    CAR_APPROACH_MIXED_VIEW_DURA = 120
    CAR_APPROACH_MIXED_VIEW_WAIT = 20
    CAR_VISUAL_DYNAMIC_FAR_AREA_RATIO = 0.0015
    CAR_VISUAL_DYNAMIC_MID_AREA_RATIO = 0.012
    CAR_VISUAL_DYNAMIC_CLOSE_AREA_RATIO = 0.030
    CAR_VISUAL_DYNAMIC_NEAR_AREA_RATIO = 0.045
    CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO = 0.08
    CAR_VISUAL_DYNAMIC_MIN_WAIT = 200
    CAR_VISUAL_DYNAMIC_VERY_NEAR_WAIT = 130
    CAR_VISUAL_DYNAMIC_NEAR_WAIT = 260
    CAR_VISUAL_DYNAMIC_CLOSE_WAIT = 650
    CAR_VISUAL_DYNAMIC_FAR_WAIT = 2200
    CAR_VISUAL_DYNAMIC_MAX_WAIT = 8500
    CAR_VISUAL_DYNAMIC_MIN_DURA = 65
    CAR_VISUAL_DYNAMIC_VERY_NEAR_DURA = 85
    CAR_VISUAL_DYNAMIC_NEAR_DURA = 120
    CAR_VISUAL_DYNAMIC_CLOSE_DURA = 180
    CAR_VISUAL_DYNAMIC_FAR_DURA = 260
    CAR_VISUAL_DYNAMIC_MIN_BIAS_Y = -85
    CAR_VISUAL_DYNAMIC_VERY_NEAR_BIAS_Y = -100
    CAR_VISUAL_DYNAMIC_NEAR_BIAS_Y = -130
    CAR_VISUAL_DYNAMIC_CLOSE_BIAS_Y = -175
    CAR_VISUAL_DYNAMIC_FAR_BIAS_Y = -220
    CAR_FORWARD_LOST_BACKOFF_Y_BIAS = 360
    CAR_FORWARD_LOST_BACKOFF_DURA = 300
    CAR_FORWARD_LOST_BACKOFF_WAIT = 500
    # 单轮寻车超过该时间仍未上车，则结束当前局；计时从进入/恢复寻车模式开始。
    CAR_SEARCH_TIMEOUT = 5 * 60
    # 路边发现远车后，允许跨帧追车，避免车辆框短暂丢失后又回头追原道路点。
    ROADSIDE_CAR_LOST_LIMIT = 5
    # 已经看到车并前推靠近后，若目标丢失，只额外前推一次；仍丢失则视为误识别。
    ROADSIDE_CAR_LOST_FORWARD_LIMIT = 1
    ROADSIDE_CAR_PURSUIT_STEP_LIMIT = 24
    ROADSIDE_CAR_MAX_ROAD_DISTANCE = 10.0
    ROADSIDE_CAR_MAX_PLAYER_ROAD_DISTANCE = 10.0
    ROADSIDE_CAR_ESTIMATE_FOV_DEGREES = 70.0
    ROADSIDE_CAR_DISTANCE_SCALE = 1.6
    ROADSIDE_CAR_MIN_ESTIMATED_DISTANCE = 5.0
    ROADSIDE_CAR_MAX_ESTIMATED_DISTANCE = 70.0
    ROADSIDE_CAR_OVERSHOOT_AREA_RATIO = 0.045
    ROADSIDE_CAR_REAR_TURN_ATTEMPTS = 6
    ROADSIDE_CAR_REAR_FORWARD_BIAS_Y = -110
    ROADSIDE_CAR_REAR_FORWARD_DURA = 120
    ROADSIDE_CAR_REAR_FORWARD_WAIT = 250
    POST_HOUSE_EXIT_CLEAR_X_BIAS = 260
    POST_HOUSE_EXIT_CLEAR_Y_BIAS = -220
    POST_HOUSE_EXIT_CLEAR_DURA = 450
    POST_HOUSE_EXIT_CLEAR_WAIT = 550
    # 落水后，上浮和向前划水脱离水面的操作参数
    WATER_FLOAT_DURA = 2000
    WATER_FORWARD_BIAS_Y = -280
    WATER_FORWARD_DURA = 1200
    WATER_FORWARD_WAIT = 5200
    WATER_EXIT_STUCK_FRAMES = 3
    WATER_EXIT_STUCK_DISTANCE = 0.6
    WATER_EXIT_STUCK_WINDOW = 18.0
    WATER_EXIT_BACK_DURA = 650
    WATER_EXIT_BACK_WAIT = 900
    WATER_EXIT_SIDE_DURA = 900
    WATER_EXIT_SIDE_WAIT = 1500
    HOUSE_SCENE_REAR_CONFIRM_TURNS = 3
    HOUSE_SCENE_RESTORE_TURNS = 2
    # 刚下车后，忽略附近车辆交互的保护时间，避免立刻又上车
    VEHICLE_EXIT_PROTECTION = 5.0
    # 下车后若仍贴着车，先短暂移动离开载具
    VEHICLE_EXIT_ESCAPE_DURA = 350
    VEHICLE_EXIT_ESCAPE_WAIT = 500
    # 人物落在不可通行区域时，先脱离黑区再规划
    FORBIDDEN_ESCAPE_SEARCH_DIST = 120
    FORBIDDEN_ESCAPE_FORWARD_DURA = 700
    FORBIDDEN_ESCAPE_FORWARD_WAIT = 900
    # 道路巡游/进圈策略
    ROAD_NODE_REACHED_TOLERANCE = 3.0
    ROAD_PATROL_MIN_NODE_DISTANCE = 8.0
    ROAD_CIRCLE_NODE_MAX_DISTANCE = 30.0
    CIRCLE_RANDOM_ROUTE_MIN_DIST = 25
    CIRCLE_RANDOM_ROUTE_MAX_DIST = 90
    CIRCLE_RANDOM_ROUTE_NUM_POINTS = 12
    VEHICLE_ENTRY_ROADSIDE = "roadside"
    VEHICLE_ENTRY_GARAGE = "garage"
    VEHICLE_ENTRY_UNKNOWN = "unknown"
    CAR_SEARCH_GARAGE = "garage"
    CAR_SEARCH_ROADSIDE = "roadside"
    PRIORITY_CAR_SEARCH_ANCHORS = ((1109, 792), (1189, 783), (1322, 960))
    RUNNING_ROUTE_CIRCLE = "circle"
    RUNNING_ROUTE_PATROL = "patrol"
    RUNNING_ROUTE_RANDOM_AROUND_CIRCLE = "random_around_circle"
    RUNNING_ROUTE_PRIORITY_CAR_SEARCH = "priority_car_search"
    RUNNING_ROUTE_FORCED = "forced_route"
    JUMP_CLICK_COOLDOWN = 0.8
    DEFAULT_FORCED_ROUTE_ARRIVAL_DISTANCE = 30.0

    def __init__(self, map_tool: Optional[MapNavigator] = None):
        self.map_tool = map_tool or MapNavigator()
        self.road_helper = RoadRouteHelper(self.map_tool)
        self.house_exit_manager = HouseExitManager()
        self.match_clock = Stopwatch()
        self.car_search_timer = TimeoutTracker(self.CAR_SEARCH_TIMEOUT)
        self.vehicle_ignore_window = ActiveWindow()
        self.jump_replan_cooldown = Cooldown()
        self.jump_click_cooldown = Cooldown()
        self.water_exit_clock = Stopwatch()
        self.screen_w, self.screen_h = get_resolution()

        self.road_list: List[Tuple[int, int]] = []
        self.locations: List[Tuple[int, int]] = []
        self.history_locations: List[Tuple[int, int]] = []
        self.current_segment_start: Optional[Tuple[int, int]] = None

        self.auto_forward = False
        self.stuck = False
        self.trapped = False

        self.stable_circle_angle: Optional[float] = None
        self.drive_required = True
        self.find_car_times = 0
        self.correct_position_times = 0
        self.finding_car = True
        self.loading_road = False
        self.precise_entering_car = False
        self.precise_last_distance: Optional[float] = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.precise_face_attempt_index = 0
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.current_view_mode = self.VIEW_MODE_THIRD
        self.last_valid_location: Optional[Tuple[int, int]] = None
        self.pause_sp_callback: Optional[Callable] = None
        self.visited_road_nodes: Set[Tuple[int, int]] = set()
        self.current_road_node: Optional[Tuple[int, int]] = None
        self.active_vehicle_entry_source: Optional[str] = None
        self.last_vehicle_entry_source: Optional[str] = None
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio: Optional[float] = None
        self.roadside_car_peak_area_ratio: Optional[float] = None
        self.roadside_car_last_forward_motion: Optional[Tuple[int, int, int]] = None
        self.current_running_route_kind: Optional[str] = None
        self.last_circle_target_point: Optional[Tuple[int, int]] = None
        self.circle_route_completed = False
        self.priority_car_search_active = False
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points: List[Tuple[int, int]] = []
        self.water_exit_last_location: Optional[Tuple[int, int]] = None
        self.water_exit_stuck_frames = 0
        self.water_exit_side_sign = 1
        self.water_escape_target: Optional[Tuple[int, int]] = None
        self.water_swim_last_location: Optional[Tuple[int, int]] = None
        self.water_swim_stuck_frames = 0
        self.forced_route_target: Optional[Tuple[int, int]] = None
        self.forced_route_finish_stage: Optional[str] = None
        self.forced_route_reason: Optional[str] = None
        self.forced_route_arrival_distance = self.DEFAULT_FORCED_ROUTE_ARRIVAL_DISTANCE

    def reset(self, finding_car: bool = True):
        self.road_list = []
        self.locations = []
        self.history_locations = []
        self.current_segment_start = None
        self.auto_forward = False
        self.stuck = False
        self.trapped = False
        self.stable_circle_angle = None
        self.drive_required = bool(finding_car)
        self.find_car_times = 0
        self.correct_position_times = 0
        self.finding_car = finding_car
        self.loading_road = False
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.precise_face_attempt_index = 0
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.current_view_mode = self.VIEW_MODE_THIRD
        self.last_valid_location = None
        self.jump_replan_cooldown.reset()
        self.jump_click_cooldown.reset()
        self.vehicle_ignore_window.reset()
        self.visited_road_nodes = set()
        self.current_road_node = None
        self.active_vehicle_entry_source = None
        self.last_vehicle_entry_source = None
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None
        self.current_running_route_kind = None
        self.last_circle_target_point = None
        self.circle_route_completed = False
        if finding_car:
            self.car_search_timer.start()
        else:
            self.car_search_timer.reset()
        self.priority_car_search_active = bool(finding_car)
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points = []
        self.water_exit_last_location = None
        self.water_exit_stuck_frames = 0
        self.water_exit_clock.reset()
        self.water_exit_side_sign = 1
        self.water_escape_target = None
        self.water_swim_last_location = None
        self.water_swim_stuck_frames = 0
        self._clear_forced_route()
        self.house_exit_manager.reset()
        print("[Running] 状态已重置!")

    def start_forced_route(
        self,
        target: Tuple[int, int],
        finish_stage: str,
        reason: str,
        arrival_distance: float = DEFAULT_FORCED_ROUTE_ARRIVAL_DISTANCE,
    ):
        self.forced_route_target = tuple(map(int, target))
        self.forced_route_finish_stage = finish_stage
        self.forced_route_reason = reason
        self.forced_route_arrival_distance = max(0.0, float(arrival_distance))
        self.drive_required = False
        self.finding_car = False
        self.car_search_timer.reset()
        self.priority_car_search_active = False
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points = []
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE
        self.loading_road = False
        self.road_list = []
        self.locations = []
        self.history_locations = []
        self.stuck = False
        self.trapped = False
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.current_road_node = None
        self.current_segment_start = None
        self.current_running_route_kind = self.RUNNING_ROUTE_FORCED
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        print(
            f"[Running] 启动临时跑图路线: reason={reason}, "
            f"target={self.forced_route_target}, finish_stage={finish_stage}, "
            f"arrival={self.forced_route_arrival_distance:.1f}"
        )

    def _has_forced_route(self) -> bool:
        return (
            self.forced_route_target is not None
            and self.forced_route_finish_stage is not None
        )

    def _clear_forced_route(self):
        self.forced_route_target = None
        self.forced_route_finish_stage = None
        self.forced_route_reason = None
        self.forced_route_arrival_distance = self.DEFAULT_FORCED_ROUTE_ARRIVAL_DISTANCE

    def set_game_time(self, game_time: Optional[float] = None):
        started_at = self.match_clock.start(game_time)
        print(f'[Running] 游戏开始时间设置为： {started_at:.3f}')

    def set_drive_required(self, required: bool):
        self.drive_required = bool(required)

    def get_elapsed_time(self) -> float:
        return self.match_clock.elapsed()

    def process(self, w: "FrameWorker"):
        location = self._get_location(w)
        if location is None:
            print("[Running] 位置无效，尝试小幅前探刷新坐标...")
            w.tap_single("摇杆", y_bias=-250, dura=250, wait=500)
            return

        direction = self._get_scalar(w.get_info("direction"))
        self._update_circle_angle(w.get_info("white_angle"))
        forced_route_active = self._has_forced_route()
        if not forced_route_active:
            self._refresh_finding_car_policy(w, location, direction)

        if self._is_dead(w):
            print("[Running] 检测到死亡!")
            self._handle_death(w)
            return

        if self._has_rank_info(w):
            print("[Running] 检测到个人排名或队伍排名，进入结束阶段")
            self._handle_rank_finish(w)
            return

        if self._handle_recent_vehicle_exit(w, location, direction):
            return

        if self._is_in_vehicle(w):
            print("[Running] 检测到已经上车，切换到开车阶段")
            self._log_running_state("检测到已上车", location, direction, "切换到开车阶段")
            entry_source = self.active_vehicle_entry_source or self.VEHICLE_ENTRY_UNKNOWN
            self._ensure_third_person_view(w, location, direction, "检测到已上车，切回第三人称")
            self.stop_auto_forward(w)
            self.reset(finding_car=False)
            self.last_vehicle_entry_source = entry_source
            w.change_stage("开车阶段")
            return

        if self._ensure_first_person_view(w, location, direction):
            return

        if self._is_in_water(w):
            self._handle_water_escape(w, location, direction)
            return

        if self._handle_recent_water_exit_stuck(w, location, direction):
            return

        if self._handle_forbidden_escape(w, location, direction):
            return

        self._click_jump_if_available(w, location, direction)

        if forced_route_active:
            self._process_forced_route(w, location, direction)
            return

        if self._handle_priority_car_route_finished(w, location, direction, "指定寻车路线已走完"):
            return

        if self._handle_car_search_timeout(w, location, direction):
            return

        if self.find_car_times >= len(self.PRECISE_FACE_DIRECTIONS):
            if self.finding_car and self.car_search_mode == self.CAR_SEARCH_GARAGE:
                self._switch_to_roadside_car_search("车库多角度视觉找车未成功")
                return
            print(f"[Running] 已连续{len(self.PRECISE_FACE_DIRECTIONS)}次未成功上车，结束当前局")
            self._log_running_state("上车尝试已达上限", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.correct_position_times >= 5:
            if self.finding_car and self.car_search_mode == self.CAR_SEARCH_GARAGE:
                self._switch_to_roadside_car_search("车库上车点精调长时间无进展")
                return
            print("[Running] 连续5次未找到车辆交互点，结束当前局")
            self._log_running_state("精调阶段长时间无进展", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.precise_entering_car:
            self._process_precise_entry(w, location, direction)
            return

        if (
            self.finding_car
            and self.car_search_mode == self.CAR_SEARCH_ROADSIDE
            and not self.garage_to_roadside_route_active
            and self._handle_roadside_vehicle_entry(w, location, direction)
        ):
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
            if self._try_house_exit_when_indoor(w, location, direction, "人物困死"):
                return
            if not self.trapped:
                print("[Running] 人物困死但后视角复核为室外，先按普通脱困处理")
                self._perform_unstuck_action(w, location)
                return
            print("[Running] 人物长时间在局部区域打转，结束当前局")
            self._log_running_state("人物困死", location, direction, "结束当前局")
            self._handle_death(w)
            return

        if self.stuck:
            if self._try_house_exit_when_indoor(w, location, direction, "人物卡住"):
                return
            print("[Running] 人物卡住，执行脱困")
            self._log_running_state("人物卡死", location, direction, "执行脱困")
            self._perform_unstuck_action(w, location)
            return

        if not self.loading_road or not self.road_list:
            self._load_path(location)

        if not self.road_list:
            print("[Running] 当前没有可执行路径")
            if self._handle_priority_car_route_finished(w, location, direction, "指定寻车路线已走完"):
                return
            return

        if not self.garage_to_roadside_route_active:
            self._advance_waypoint_by_projection(location)
        if not self.road_list:
            print("[Running] 当前路径已按投影走完，下一帧重新规划")
            if self._handle_priority_car_route_finished(w, location, direction, "指定寻车路线投影判定已走完"):
                return
            self._mark_running_route_completed_if_needed(location, "投影判定路径走完")
            self.loading_road = False
            return

        target = self.road_list[0]
        dist = get_distance(location, target)
        print(f"[Running] Loc: {location}, Target: {target}, Dist: {dist:.2f}")

        if self._handle_garage_to_roadside_forward_push(w, location, direction, target):
            return

        arrival_tolerance = self._get_current_waypoint_tolerance()
        if 0 <= dist < arrival_tolerance:
            print(f"[Running] 到达 {target} 点附近")
            self._handle_waypoint_arrival(w, location, direction, target, dist)
            return

        if (
            (self.garage_to_roadside_route_active or len(self.road_list) <= 1)
            and 0 <= dist <= self.WAYPOINT_PRECISE_APPROACH_DISTANCE
        ):
            print(f"[Running] 距离目标点 {dist:.2f}，切换精确逼近")
            self._precise_approach_waypoint(w, location, direction, target, dist)
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
        if self.finding_car:
            stage = "车库找车" if self.car_search_mode == self.CAR_SEARCH_GARAGE else "道路找车"
        else:
            stage = "普通跑图"
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

        location = None
        if isinstance(info, (list, tuple)):
            if len(info) >= 2 and not isinstance(info[0], (list, tuple)):
                location = check_location(info)
            elif len(info) > 0:
                location = check_location(info[0])

        if location is None:
            return None

        try:
            location = (int(location[0]), int(location[1]))
        except (TypeError, ValueError, IndexError):
            return None

        if not self._is_reasonable_location(location):
            print(f"[Running] 坐标疑似异常，忽略本帧 location={location}")
            return None

        return location

    def _is_reasonable_location(self, location: Tuple[int, int]) -> bool:
        x, y = location
        width = getattr(self.map_tool, "width", None)
        height = getattr(self.map_tool, "height", None)
        if width is not None and height is not None:
            if x < 0 or y < 0 or x >= int(width) or y >= int(height):
                return False

        return x > self.LOCATION_MIN_VALID_COORD and y > self.LOCATION_MIN_VALID_COORD

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
            self.circle_route_completed = False
            print(f"[Running] 获取到进圈方向: {angle:.1f}")
            if not self.finding_car:
                self.loading_road = False
            return

        if self._angle_diff(self.stable_circle_angle, angle) >= self.CIRCLE_REPLAN_THRESHOLD:
            print(f"[Running] 进圈方向更新: {self.stable_circle_angle:.1f} -> {angle:.1f}")
            self.stable_circle_angle = angle
            self.circle_route_completed = False
            if not self.finding_car:
                self.loading_road = False

    def _need_circle_now(self) -> bool:
        if self.drive_required:
            return False
        return self.stable_circle_angle is not None and not self.circle_route_completed

    def _refresh_finding_car_policy(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        if self.drive_required:
            needs_priority_route = (
                not self.finding_car
                or self.car_search_mode != self.CAR_SEARCH_ROADSIDE
                or not self.priority_car_search_active
                or self.current_running_route_kind
                in {
                    self.RUNNING_ROUTE_CIRCLE,
                    self.RUNNING_ROUTE_RANDOM_AROUND_CIRCLE,
                    self.RUNNING_ROUTE_PATROL,
                }
            )
            if not needs_priority_route:
                return

            print("[Running] 开车阶段未完成，优先沿指定路线找车，暂不处理进圈")
            self._log_running_state("恢复找车", location, direction, "沿指定路线继续找车")
            self.stop_auto_forward(w)
            self.finding_car = True
            self.car_search_timer.start()
            self.priority_car_search_active = True
            self.priority_car_search_finished = False
            self.priority_car_search_next_index = 0
            self.priority_car_search_road_points = []
            self.current_running_route_kind = None
            self._switch_to_roadside_car_search(
                "开车未完成，优先沿指定路线找车",
                leave_garage_route=False,
            )
            return

        if not self.finding_car:
            return

        print("[Running] 开车阶段已完成，停止找车，恢复跑图/进圈")
        self._log_running_state("停止找车", location, direction, "开车完成后恢复跑图/进圈")
        self.stop_auto_forward(w)
        self.finding_car = False
        self.car_search_timer.reset()
        self.priority_car_search_active = False
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.precise_face_attempt_index = 0
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None
        self.current_road_node = None
        self.active_vehicle_entry_source = None
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None

    def _is_in_vehicle(self, w: "FrameWorker") -> bool:
        if any(w.get_info(name) for name in ("漂移", "喇叭")):
            return True

        on_foot_ui_missing = not w.get_info("左拳头") and not w.get_info("子弹")
        vehicle_ui_visible = any(
            w.get_info(name)
            for name in ("自动前进", "急刹", "加速")
        )
        return on_foot_ui_missing and vehicle_ui_visible

    def _is_in_water(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("上浮"))

    def _is_dead(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("变身")) or bool(w.get_info("红色血条"))

    def _has_rank_info(self, w: "FrameWorker") -> bool:
        return bool(w.get_info("个人排名")) or bool(w.get_info("队伍排名"))

    def _handle_car_search_timeout(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if not self.finding_car:
            return False
        if self.priority_car_search_active and not self.priority_car_search_finished:
            return False

        self.car_search_timer.start_if_needed()

        elapsed = self.car_search_timer.elapsed()
        if elapsed < self.CAR_SEARCH_TIMEOUT:
            return False

        print(
            f"[Running] 本轮寻车已超过 {self.CAR_SEARCH_TIMEOUT:.0f}s 仍未上车，"
            "结束当前局并开始下一局"
        )
        self._log_running_state(
            "寻车超时",
            location,
            direction,
            f"本轮寻车耗时 {elapsed:.1f}s，结束当前局",
        )
        self._handle_death(w)
        return True

    def _handle_priority_car_route_finished(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        reason: str,
    ) -> bool:
        if (
            not self.finding_car
            or self.car_search_mode != self.CAR_SEARCH_ROADSIDE
            or self.current_running_route_kind != self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH
        ):
            return False

        if self.road_list:
            return False

        route_points = self._get_priority_car_search_road_points()
        final_point = route_points[-1] if route_points else tuple(map(int, self.PRIORITY_CAR_SEARCH_ANCHORS[-1]))
        final_dist = get_distance(location, final_point)
        if final_dist > max(self.WAYPOINT_TOLERANCE, self.ROAD_NODE_REACHED_TOLERANCE):
            return False

        print(f"[Running] {reason}，已到达指定终点 {final_point} 仍未上车，结束当前局并重开")
        self._log_running_state(
            "指定寻车路线结束",
            location,
            direction,
            "终点未上车，结束当前局",
            final_point,
            final_dist,
        )
        self.priority_car_search_finished = True
        self.priority_car_search_active = False
        self._handle_death(w)
        return True

    def _handle_death(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        self.reset()
        w.change_stage("结束阶段")

    def _handle_rank_finish(self, w: "FrameWorker"):
        if callable(self.pause_sp_callback):
            self.pause_sp_callback(w)
        else:
            w.click("sp")
            time.sleep(0.5)
        time.sleep(1)
        w.click("观战对手")
        self.reset()
        w.change_stage("结束阶段")

    def notify_vehicle_exit(
        self,
        cooldown: float = VEHICLE_EXIT_PROTECTION,
        finding_car: bool = False,
    ):
        self.drive_required = bool(finding_car)
        self.finding_car = bool(finding_car)
        if self.finding_car:
            self.car_search_timer.start()
        else:
            self.car_search_timer.reset()
        self.priority_car_search_active = bool(self.finding_car)
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points = []
        self.loading_road = False
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.road_list = []
        self.locations = []
        self.history_locations = []
        self.stuck = False
        self.trapped = False
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.current_view_mode = self.VIEW_MODE_THIRD
        self.vehicle_ignore_window.start(cooldown)
        self.jump_click_cooldown.reset()
        self.current_road_node = None
        self.current_segment_start = None
        self.active_vehicle_entry_source = None
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE if self.finding_car else self.CAR_SEARCH_GARAGE
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None
        self.current_running_route_kind = None
        print(
            f"[Running] 收到下车通知，载具交互保护期 {cooldown:.1f}s，"
            f"后续模式={'继续寻车' if self.finding_car else '纯跑图'}"
        )

    def notify_searching_exit(self, finding_car: bool = True):
        self.drive_required = bool(finding_car)
        self.finding_car = bool(finding_car)
        if self.finding_car:
            self.car_search_timer.start()
        else:
            self.car_search_timer.reset()
        self.priority_car_search_active = bool(self.finding_car)
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points = []
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE
        self.loading_road = False
        self.road_list = []
        self.locations = []
        self.history_locations = []
        self.stuck = False
        self.trapped = False
        self.current_road_node = None
        self.current_segment_start = None
        self.current_running_route_kind = None
        self.garage_to_roadside_route_active = False
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None
        print(
            "[Running] 收到搜房结束通知，"
            f"后续模式={'沿指定路线找车，开车完成后再跑图/进圈' if self.finding_car else '跑图/进圈'}"
        )

    def consume_vehicle_entry_source(self) -> Optional[str]:
        source = self.last_vehicle_entry_source
        self.last_vehicle_entry_source = None
        return source

    def _handle_recent_vehicle_exit(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if not self.vehicle_ignore_window.active():
            return False

        drive_btn = w.get_info("驾驶")
        still_vehicle_ui = self._is_in_vehicle(w)
        remaining = self.vehicle_ignore_window.remaining()

        if not drive_btn and not still_vehicle_ui:
            return False

        print(f"[Running] 刚下车，忽略载具交互 remaining={remaining:.2f}s")
        self._log_running_state("下车保护期", location, direction, "忽略上车/驾驶并先离开载具")
        self.stop_auto_forward(w)
        self.loading_road = False
        self.road_list = []
        w.tap_single(
            "摇杆",
            y_bias=-260,
            dura=self.VEHICLE_EXIT_ESCAPE_DURA,
            wait=self.VEHICLE_EXIT_ESCAPE_WAIT,
        )
        w.refresh_frame()
        return True

    def _handle_forbidden_escape(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if self.finding_car:
            print("[Running] 当前处于寻车阶段，跳过黑区脱离推理")
            return False

        if self.map_tool.is_walkable(location):
            return False

        self._check_if_stuck(location)
        if self.stuck:
            print("[Running] 当前位于不可通行区域且人物卡住，先执行避障脱困")
            self._log_running_state("不可通行区域卡住", location, direction, "调用避障脱困")
            self._perform_unstuck_action(w, location)
            return True

        safe_point = self.map_tool.get_nearest_safe_point(
            location,
            max_search_dist=self.FORBIDDEN_ESCAPE_SEARCH_DIST,
        )
        self.stop_auto_forward(w)
        self.loading_road = False
        self.road_list = []

        if safe_point is None:
            print("[Running] 当前位于不可通行区域，暂未找到安全点，先尝试直线脱离")
            self._log_running_state("人物位于不可通行区域", location, direction, "直线尝试脱离黑区")
            w.tap_single(
                "摇杆",
                y_bias=-300,
                dura=self.FORBIDDEN_ESCAPE_FORWARD_DURA,
                wait=self.FORBIDDEN_ESCAPE_FORWARD_WAIT,
            )
            w.refresh_frame()
            return True

        dist = get_distance(location, safe_point)
        print(f"[Running] 当前位于不可通行区域，先脱离到最近安全点 {safe_point}，距离 {dist:.2f}")
        self._log_running_state("人物位于不可通行区域", location, direction, "先脱离黑区再规划路径", safe_point, dist)

        if w.get_info("跳跃") and self.jump_click_cooldown.try_acquire(self.JUMP_CLICK_COOLDOWN):
            print("[Running] 不可通行区域发现跳跃键，先跳跃并朝安全点前推")
            w.click("跳跃")
            time.sleep(0.15)
            if direction is not None:
                self._align_to_point(w, location, direction, safe_point, threshold=8)
            w.tap_single(
                "摇杆",
                y_bias=-300,
                dura=self.FORBIDDEN_ESCAPE_FORWARD_DURA,
                wait=self.FORBIDDEN_ESCAPE_FORWARD_WAIT,
            )
            w.refresh_frame()
            return True

        if direction is not None:
            aligned = self._align_to_point(w, location, direction, safe_point, threshold=5)
            if not aligned:
                return True

        w.tap_single(
            "摇杆",
            y_bias=-300,
            dura=self.FORBIDDEN_ESCAPE_FORWARD_DURA,
            wait=self.FORBIDDEN_ESCAPE_FORWARD_WAIT,
        )
        w.refresh_frame()
        return True

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

        if not self.jump_replan_cooldown.try_acquire(self.LOCATION_JUMP_REPLAN_COOLDOWN):
            return False

        route_type = "寻车路径" if self.finding_car else "跑图路径"
        print(
            f"[Running] 检测到位置跳变: prev={previous_location}, "
            f"current={location}, jump_dist={jump_dist:.2f}，重新规划{route_type}"
        )
        self._log_running_state("位置跳变", location, None, f"重新规划{route_type}")
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None
        self.locations = [location]
        self.history_locations = [location]
        self.stuck = False
        self.trapped = False
        return True

    def _process_forced_route(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if not self._has_forced_route():
            return False

        target = self.forced_route_target
        finish_stage = self.forced_route_finish_stage
        dist_to_final = get_distance(location, target)
        print(
            f"[Running] 临时路线前往 {target}, dist={dist_to_final:.2f}, "
            f"reason={self.forced_route_reason}"
        )
        if 0 <= dist_to_final <= self.forced_route_arrival_distance:
            print(f"[Running] 已到达临时路线目标 {target}，切回 {finish_stage}")
            self.stop_auto_forward(w)
            self._clear_forced_route()
            self.loading_road = False
            self.road_list = []
            self.current_segment_start = None
            self.current_running_route_kind = None
            w.change_stage(finish_stage)
            return True

        if self._handle_location_jump(location):
            if not self.loading_road or not self.road_list:
                self._load_path(location)
            if not self.road_list:
                print("[Running] 临时路线位置跳变后重规划失败，等待下一帧")
                return True

        self._check_if_stuck(location)
        self._check_if_trapped(location)

        if self.trapped:
            print("[Running] 临时路线人物困住，按卡住脱困处理，继续前往目标")
            self._log_running_state("临时路线困住", location, direction, "执行脱困", target, dist_to_final)
            self._perform_unstuck_action(w, location)
            return True

        if self.stuck:
            print("[Running] 临时路线人物卡住，执行脱困")
            self._log_running_state("临时路线卡住", location, direction, "执行脱困", target, dist_to_final)
            self._perform_unstuck_action(w, location)
            return True

        if not self.loading_road or not self.road_list:
            self._load_path(location)

        if not self.road_list:
            print("[Running] 临时路线无可执行路径，下一帧重试")
            return True

        self._advance_waypoint_by_projection(location)
        if not self.road_list:
            self.loading_road = False
            return True

        waypoint = self.road_list[0]
        waypoint_dist = get_distance(location, waypoint)
        print(f"[Running] 临时路线 waypoint={waypoint}, dist={waypoint_dist:.2f}")

        if 0 <= waypoint_dist < self._get_current_waypoint_tolerance():
            self._discard_current_road_target()
            return True

        if len(self.road_list) <= 1 and 0 <= waypoint_dist <= self.WAYPOINT_PRECISE_APPROACH_DISTANCE:
            print(f"[Running] 临时路线距离目标点 {waypoint_dist:.2f}，切换精确逼近")
            self._precise_approach_waypoint(w, location, direction, waypoint, waypoint_dist)
            return True

        self._move_towards_target(w, location, direction, waypoint)
        return True

    def _load_path(self, location: Tuple[int, int]):
        if self.garage_to_roadside_route_active:
            print("[Running] 继续加载车库离库路线，先到路边再找车")
            self.road_list = list(self.GARAGE_TO_ROADSIDE_POINTS)
            self.current_road_node = None
            self.current_segment_start = None
            self.current_running_route_kind = None
        elif self._has_forced_route():
            self._load_forced_route_path(location)
        elif self.finding_car:
            if self.car_search_mode == self.CAR_SEARCH_GARAGE:
                if self._should_skip_garage_search(location):
                    self._switch_to_roadside_car_search(
                        "距离车库点过远",
                        leave_garage_route=False,
                    )
                    self._load_priority_car_search_path(location, reason="距离车库点过远，直接沿指定路线找车")
                else:
                    self._load_garage_find_path(location)
            else:
                self._load_priority_car_search_path(location, reason="寻车阶段沿指定路线巡游")
        else:
            self._load_running_path(location)

        self.road_list = [tuple(map(int, p)) for p in self.road_list if p is not None]
        self.loading_road = bool(self.road_list)
        self.current_segment_start = location if self.loading_road else None
        if self.loading_road:
            print(f"[Running] 路径已加载: {self.road_list}")
            try:
                draw_points_with_arrows(self.road_list)
            except Exception as exc:
                print(f"[Running] 绘制路径调试图失败: {exc}")
        else:
            print("[Running] 路径加载失败")

    def _load_forced_route_path(self, location: Tuple[int, int]):
        target = self.forced_route_target
        if target is None:
            self.road_list = []
            return

        print(f"[Running] 正在加载临时跑图路线: {location} -> {target}")
        self._log_running_state("临时路线规划", location, None, self.forced_route_reason or "前往目标", target)
        self.road_list = self.map_tool.plan_path(location, target) or [target]
        if not self.road_list or tuple(map(int, self.road_list[-1])) != target:
            self.road_list = self._merge_paths(self.road_list, [target])
        self.current_road_node = None
        self.current_running_route_kind = self.RUNNING_ROUTE_FORCED

    def _load_garage_find_path(self, location: Tuple[int, int]):
        print("[Running] 正在加载车库优先寻车路径...")
        self._log_running_state("正在加载车库寻车路径", location, None, "先去车库取车")
        self.current_road_node = None

        if get_distance(location, self.R_CITY) > 50:
            approach_path = self.map_tool.plan_path(location, self.R_CITY) or []
            garage_path = find_path(self.R_CITY) or self.map_tool.plan_path(self.R_CITY, self.CAR_ENTRY_POINT) or []
            self.road_list = self._merge_paths(approach_path, garage_path)
        else:
            self.road_list = find_path(location) or self.map_tool.plan_path(location, self.CAR_ENTRY_POINT) or []

        if not self.road_list or tuple(map(int, self.road_list[-1])) != self.CAR_ENTRY_POINT:
            self.road_list = self._merge_paths(self.road_list, [self.CAR_ENTRY_POINT])

    def _should_skip_garage_search(self, location: Tuple[int, int]) -> bool:
        dist_to_garage = get_distance(location, self.CAR_ENTRY_POINT)
        if dist_to_garage <= self.GARAGE_SEARCH_MAX_DISTANCE:
            return False
        print(
            f"[Running] 当前距离车库上车点 {dist_to_garage:.2f} "
            f"> {self.GARAGE_SEARCH_MAX_DISTANCE:.2f}，跳过车库找车"
        )
        self._log_running_state(
            "距离车库点过远",
            location,
            None,
            "跳过车库，直接切换到沿路找车",
            self.CAR_ENTRY_POINT,
            dist_to_garage,
        )
        return True

    def _merge_paths(self, path1, path2):
        merged = [tuple(map(int, point)) for point in (path1 or []) if point is not None]
        for point in path2 or []:
            if point is None:
                continue
            item = tuple(map(int, point))
            if not merged or merged[-1] != item:
                merged.append(item)
        return merged

    def _load_priority_car_search_path(self, location: Tuple[int, int], reason: str) -> bool:
        if (
            not self.finding_car
            or self.car_search_mode != self.CAR_SEARCH_ROADSIDE
            or not self.priority_car_search_active
            or self.priority_car_search_finished
        ):
            return False

        print(f"[Running] {reason}: road_points={self.PRIORITY_CAR_SEARCH_ANCHORS}")
        self._log_running_state(
            reason,
            location,
            None,
            "沿指定 road_topology 路线规划道路找车",
            self.PRIORITY_CAR_SEARCH_ANCHORS[-1],
        )

        self._sync_priority_car_route_progress(location)
        route_points = self._get_priority_car_search_road_points()
        remaining_points = route_points[self.priority_car_search_next_index:]
        if not remaining_points:
            self.priority_car_search_finished = True
            self.current_running_route_kind = self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH
            self.road_list = []
            return True

        route, start_road_point, start_road_dist = self.road_helper.plan_priority_road_path(
            location,
            remaining_points,
        )
        if not route:
            print("[Running] road_topology 指定寻车路线规划失败，等待下一帧重试")
            self.road_list = []
            self.current_running_route_kind = self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH
            return False

        print(
            f"[Running] 指定寻车路线起点 A={start_road_point}, "
            f"player_to_A={start_road_dist:.2f}, remaining={remaining_points}"
        )

        self.current_road_node = None
        self.road_list = route
        self.current_running_route_kind = self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH
        self.garage_to_roadside_route_active = False
        print(f"[Running] 指定寻车路线已生成: {self.road_list}")
        return True

    def _is_priority_route_segment_reasonable(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        segment: List[Tuple[int, int]],
    ) -> bool:
        if not segment:
            return False

        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        margin = 180
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        for point in segment:
            if point[0] < min_x or point[0] > max_x or point[1] < min_y or point[1] > max_y:
                return False

        direct_dist = max(get_distance(start, end), 1.0)
        path_dist = 0.0
        prev = start
        for point in segment:
            path_dist += get_distance(prev, point)
            prev = point
        return path_dist <= direct_dist * 4

    def _get_priority_car_search_road_points(self, use_topology_nodes: Optional[bool] = None) -> List[Tuple[int, int]]:
        if self.priority_car_search_road_points:
            return self.priority_car_search_road_points

        self.priority_car_search_road_points = [
            tuple(map(int, point))
            for point in self.PRIORITY_CAR_SEARCH_ANCHORS
        ]
        print(f"[Running] 指定寻车精确道路节点: {self.priority_car_search_road_points}")
        return self.priority_car_search_road_points

    def _sync_priority_car_route_progress(self, location: Tuple[int, int]):
        route_points = self._get_priority_car_search_road_points()
        while self.priority_car_search_next_index < len(route_points):
            road_point = tuple(map(int, route_points[self.priority_car_search_next_index]))
            if get_distance(location, road_point) > self.WAYPOINT_TOLERANCE:
                break
            self.priority_car_search_next_index += 1
            print(f"[Running] 指定寻车道路点已到达: {road_point}")

    def _mark_priority_car_waypoint_reached(self, point: Tuple[int, int]):
        route_points = self._get_priority_car_search_road_points()
        while self.priority_car_search_next_index < len(route_points):
            road_point = tuple(map(int, route_points[self.priority_car_search_next_index]))
            if get_distance(point, road_point) > self.WAYPOINT_TOLERANCE:
                break
            self.priority_car_search_next_index += 1
            print(f"[Running] 指定寻车道路点已到达: {road_point}")

    def _load_road_patrol_path(self, location: Tuple[int, int], reason: str):
        print(f"[Running] {reason}，规划道路点")
        use_topology_nodes = self.road_helper.topology_available()

        prefer_center = not self._need_circle_now()
        target_hint = self.MAP_CENTER if prefer_center else location
        target_text = "地图中心附近道路点" if prefer_center else "下一个道路点"
        self._log_running_state(reason, location, None, f"规划到{target_text}", target_hint)
        if prefer_center:
            node, node_dist = self.road_helper.center_biased_node(
                location,
                self.MAP_CENTER,
                exclude=self.visited_road_nodes,
                min_distance=0.0,
                topology_only=use_topology_nodes,
            )
        else:
            node, node_dist = self.road_helper.nearest_node(
                location,
                exclude=self.visited_road_nodes,
                min_distance=0.0,
                topology_only=use_topology_nodes,
            )

        if node is not None and node_dist <= self.ROAD_NODE_REACHED_TOLERANCE:
            self.visited_road_nodes.add(node)
            if prefer_center:
                node, node_dist = self.road_helper.center_biased_node(
                    location,
                    self.MAP_CENTER,
                    exclude=self.visited_road_nodes,
                    min_distance=self.ROAD_PATROL_MIN_NODE_DISTANCE,
                    topology_only=use_topology_nodes,
                )
            else:
                node, node_dist = self.road_helper.nearest_node(
                    location,
                    exclude=self.visited_road_nodes,
                    min_distance=self.ROAD_PATROL_MIN_NODE_DISTANCE,
                    topology_only=use_topology_nodes,
                )

        if node is None:
            if self.visited_road_nodes:
                print("[Running] 可巡游道路点已用完，清空已访问集合后重新选择")
                self.visited_road_nodes.clear()
                if prefer_center:
                    node, node_dist = self.road_helper.center_biased_node(
                        location,
                        self.MAP_CENTER,
                        min_distance=self.ROAD_PATROL_MIN_NODE_DISTANCE,
                        topology_only=use_topology_nodes,
                    )
                else:
                    node, node_dist = self.road_helper.nearest_node(
                        location,
                        min_distance=self.ROAD_PATROL_MIN_NODE_DISTANCE,
                        topology_only=use_topology_nodes,
                    )

        if node is None:
            print("[Running] 没有可用道路点，回退随机可视点巡逻")
            self.current_road_node = None
            self.road_list = self.map_tool.get_random_visible_points(location)
            self.current_running_route_kind = self.RUNNING_ROUTE_PATROL if not self.finding_car else None
            return

        self.current_road_node = node
        self.road_list = self.road_helper.plan_to_node(location, node)
        self.current_running_route_kind = self.RUNNING_ROUTE_PATROL if not self.finding_car else None
        center_dist = get_distance(node, self.MAP_CENTER)
        print(
            f"[Running] 道路巡游目标 node={node}, dist={node_dist:.2f}, "
            f"center_dist={center_dist:.2f}, prefer_center={prefer_center}"
        )

    def _load_running_path(self, location: Tuple[int, int]):
        if self.stable_circle_angle is None:
            print("[Running] 未获取到进圈方向，先沿临近道路点巡游")
            self._load_road_patrol_path(location, reason="纯跑图道路巡游")
            return

        if self.circle_route_completed and self.last_circle_target_point is not None:
            if self._load_circle_random_running_path(location):
                return
            print("[Running] 圈中心附近随机跑图规划失败，回退道路巡游")
            self._load_road_patrol_path(location, reason="圈中心随机规划失败，先道路巡游")
            return

        print("[Running] 正在加载进圈路径...")
        self._log_running_state("正在加载进圈路径", location, None, "优先规划到进圈点附近道路点")
        target_point = self._get_circle_target_point(location)
        if target_point is None:
            self._load_road_patrol_path(location, reason="进圈目标无效，先道路巡游")
            return

        road_node, road_dist = self.road_helper.nearest_node(
            target_point,
            topology_only=self.road_helper.topology_available(),
        )
        if road_node is not None and road_dist <= self.ROAD_CIRCLE_NODE_MAX_DISTANCE:
            self.current_road_node = road_node
            dist_to_node = get_distance(location, road_node)
            print(
                f"[Running] 进圈点 {target_point} 附近道路点 {road_node}, "
                f"road_dist={road_dist:.2f}, current_dist={dist_to_node:.2f}"
            )
            if dist_to_node <= self.WAYPOINT_TOLERANCE:
                self.road_list = self.map_tool.plan_path(location, target_point) or [target_point]
            else:
                self.road_list = self.road_helper.plan_to_node(location, road_node)
            self.last_circle_target_point = target_point
            self.current_running_route_kind = self.RUNNING_ROUTE_CIRCLE
            return

        print(
            f"[Running] 进圈点附近没有足够近的道路点 "
            f"(nearest={road_node}, dist={road_dist:.2f})，回退 A*+mask"
        )
        self.current_road_node = None
        self.road_list = self.map_tool.plan_path(location, target_point)
        self.last_circle_target_point = target_point
        self.current_running_route_kind = self.RUNNING_ROUTE_CIRCLE

    def _load_circle_random_running_path(self, location: Tuple[int, int]) -> bool:
        anchor = self.last_circle_target_point
        if anchor is None:
            return False

        print(f"[Running] 已完成进圈路线，围绕圈目标 {anchor} 随机规划跑图路线")
        self._log_running_state("圈内随机跑图", location, None, "围绕上次圈目标随机规划可通行路线", anchor)
        candidates = self.map_tool.get_random_visible_points(
            anchor,
            num_points=self.CIRCLE_RANDOM_ROUTE_NUM_POINTS,
            min_dist=self.CIRCLE_RANDOM_ROUTE_MIN_DIST,
            max_dist=self.CIRCLE_RANDOM_ROUTE_MAX_DIST,
        )

        for candidate in candidates:
            candidate = tuple(map(int, candidate))
            if get_distance(location, candidate) <= self.WAYPOINT_TOLERANCE:
                continue
            path = self.map_tool.plan_path(location, candidate)
            if path:
                self.current_road_node = None
                self.road_list = path
                self.current_running_route_kind = self.RUNNING_ROUTE_RANDOM_AROUND_CIRCLE
                print(f"[Running] 圈目标附近随机跑图目标 {candidate}, path_len={len(path)}")
                return True

        return False

    def _get_circle_target_point(self, location: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if self.stable_circle_angle is None:
            return None

        elapsed = self.get_elapsed_time()
        if elapsed <= self.STAGE1_TIME:
            target_dist = self.STAGE1_DIS
        elif elapsed <= self.STAGE2_TIME:
            target_dist = self.STAGE2_DIS
        else:
            target_dist = self.STAGE3_DIS
        return self.map_tool.get_target_point(location, self.stable_circle_angle, target_dist)

    def _handle_water_escape(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        self.stop_auto_forward(w)
        target = self._get_running_target(location)
        self.water_escape_target = target
        self._log_running_state("检测到落水", location, direction, "执行上浮脱困", target)

        print("[Running] 检测到上浮图标，先点击上浮再长按保持浮出水面")
        float_button = w.get_info("上浮")
        w.click("上浮")
        time.sleep(0.2)
        w.click_down(float_button or "上浮", dura=self.WATER_FLOAT_DURA)
        time.sleep(0.3)
        w.refresh_frame()

        updated_location = self._get_location(w) or location
        updated_direction = self._get_scalar(w.get_info("direction"))

        if target is not None and updated_direction is not None:
            aligned = self._align_to_point(w, updated_location, updated_direction, target, threshold=5)
            if not aligned:
                return

        print("[Running] 开始按目标方向长推摇杆脱离水面")
        before_forward_location = updated_location
        w.tap_single(
            "摇杆",
            y_bias=self.WATER_FORWARD_BIAS_Y,
            dura=self.WATER_FORWARD_DURA,
            wait=self.WATER_FORWARD_WAIT,
        )
        w.refresh_frame()

        if not self._is_in_water(w) or w.get_info("左拳头") or w.get_info("子弹"):
            print("[Running] 上浮图标已消失，已脱离水面，恢复正常跑图")
            new_location = self._get_location(w) or updated_location
            new_direction = self._get_scalar(w.get_info("direction"))
            self._log_running_state("已脱离水面", new_location, new_direction, "恢复正常跑图", target)
            self.water_exit_last_location = new_location
            self.water_exit_stuck_frames = 0
            self.water_exit_clock.start()
            self.water_swim_last_location = None
            self.water_swim_stuck_frames = 0
            self.loading_road = False
            self.current_segment_start = None
            return

        after_forward_location = self._get_location(w) or before_forward_location
        if self._handle_in_water_forward_stuck(w, after_forward_location, direction, target):
            return

        print("[Running] 仍在水中，下一帧继续执行脱水流程")

    def _handle_in_water_forward_stuck(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Optional[Tuple[int, int]],
    ) -> bool:
        if self.water_swim_last_location is None:
            self.water_swim_last_location = location
            self.water_swim_stuck_frames = 0
            return False

        moved = get_distance(self.water_swim_last_location, location)
        if moved > self.WATER_EXIT_STUCK_DISTANCE:
            self.water_swim_last_location = location
            self.water_swim_stuck_frames = 0
            return False

        self.water_swim_stuck_frames += 1
        if self.water_swim_stuck_frames < self.WATER_EXIT_STUCK_FRAMES:
            return False

        print("[Running] 水中朝目标前推仍无有效位移，侧移换上岸点后继续前推")
        self._log_running_state("水中前推卡住", location, direction, "侧移换上岸点", target)
        side_bias = 360 * self.water_exit_side_sign
        self.water_exit_side_sign *= -1
        w.tap_single(
            "摇杆",
            x_bias=side_bias,
            dura=self.WATER_EXIT_SIDE_DURA,
            wait=self.WATER_EXIT_SIDE_WAIT,
        )
        w.refresh_frame()

        updated_location = self._get_location(w) or location
        updated_direction = self._get_scalar(w.get_info("direction"))
        if target is not None and updated_direction is not None:
            self._align_to_point(w, updated_location, updated_direction, target, threshold=8)

        w.tap_single(
            "摇杆",
            y_bias=self.WATER_FORWARD_BIAS_Y,
            dura=self.WATER_FORWARD_DURA,
            wait=self.WATER_FORWARD_WAIT,
        )
        w.refresh_frame()
        self.water_swim_last_location = self._get_location(w) or updated_location
        self.water_swim_stuck_frames = 0
        self.loading_road = False
        self.current_segment_start = None
        return True

    def _handle_recent_water_exit_stuck(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if self.water_exit_last_location is None:
            return False

        if self.water_exit_clock.elapsed() > self.WATER_EXIT_STUCK_WINDOW:
            self.water_exit_last_location = None
            self.water_exit_stuck_frames = 0
            self.water_exit_clock.reset()
            return False

        moved = get_distance(self.water_exit_last_location, location)
        if moved > self.WATER_EXIT_STUCK_DISTANCE:
            self.water_exit_last_location = location
            self.water_exit_stuck_frames = 0
            self.water_exit_clock.start()
            return False

        self.water_exit_stuck_frames += 1
        if self.water_exit_stuck_frames < self.WATER_EXIT_STUCK_FRAMES:
            return False

        target = self.water_escape_target or self._get_running_target(location)
        print("[Running] 刚上岸后位置不动，疑似卡在岸边，换上岸点")
        self._log_running_state("岸边上岸卡住", location, direction, "后退并侧移更换上岸点", target)
        self.stop_auto_forward(w)

        w.tap_single(
            "摇杆",
            y_bias=280,
            dura=self.WATER_EXIT_BACK_DURA,
            wait=self.WATER_EXIT_BACK_WAIT,
        )
        w.refresh_frame()

        side_bias = 360 * self.water_exit_side_sign
        self.water_exit_side_sign *= -1
        w.tap_single(
            "摇杆",
            x_bias=side_bias,
            dura=self.WATER_EXIT_SIDE_DURA,
            wait=self.WATER_EXIT_SIDE_WAIT,
        )
        w.refresh_frame()

        new_location = self._get_location(w) or location
        new_direction = self._get_scalar(w.get_info("direction"))
        if target is not None and new_direction is not None:
            self._align_to_point(w, new_location, new_direction, target, threshold=8)

        w.tap_single(
            "摇杆",
            y_bias=self.WATER_FORWARD_BIAS_Y,
            dura=self.WATER_FORWARD_DURA,
            wait=self.WATER_FORWARD_WAIT,
        )
        w.refresh_frame()

        self.water_exit_last_location = self._get_location(w) or new_location
        self.water_exit_clock.start()
        self.water_exit_stuck_frames = 0
        self.loading_road = False
        self.current_segment_start = None
        return True

    def _get_running_target(self, location: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if self._has_forced_route():
            return self.forced_route_target

        if self.road_list:
            return self.road_list[0]

        if not self.loading_road:
            self._load_path(location)
            if self.road_list:
                return self.road_list[0]

        if self.finding_car:
            if self.car_search_mode == self.CAR_SEARCH_GARAGE:
                return self.CAR_ENTRY_POINT
            node, _ = self.road_helper.nearest_node(
                location,
                exclude=self.visited_road_nodes,
                topology_only=self.road_helper.topology_available(),
            )
            return node or self.R_CITY

        if self.stable_circle_angle is not None:
            return self._get_circle_target_point(location)

        return None

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

    def _confirm_indoor_after_rear_view(
        self,
        w: "FrameWorker",
        direction: Optional[float],
        reason: str,
    ) -> bool:
        front_scene = self._get_house_scene(w)
        if front_scene != HouseExitManager.HOUSE_INDOOR:
            return False

        print(f"[Running] {reason}时前景 house_scene=indoor，先转身复核，避免贴墙/路过房子误判")
        self.stop_auto_forward(w)
        original_direction = direction
        if original_direction is None:
            original_direction = self._get_scalar(w.get_info("direction"))

        if original_direction is None:
            print("[Running] 当前朝向无效，执行粗略后视角复核")
            w.tap_single("视角", x_bias=self.UNSTUCK_TURN_BIAS * 2, dura=850, wait=500)
            w.refresh_frame()
        else:
            rear_direction = (float(original_direction) + 180.0) % 360.0
            for _ in range(self.HOUSE_SCENE_REAR_CONFIRM_TURNS):
                current_direction = self._get_scalar(w.get_info("direction"))
                if current_direction is None:
                    break
                if self._align_to_direction(w, current_direction, rear_direction, threshold=8):
                    break
            w.refresh_frame()

        rear_scene = self._get_house_scene(w)
        if rear_scene == HouseExitManager.HOUSE_INDOOR:
            print("[Running] 后视角复核仍为 indoor，确认人物在屋内")
            return True

        print(f"[Running] 后视角复核 house_scene={rear_scene}，判定为室外贴墙/路过房子误判")
        if original_direction is not None:
            for _ in range(self.HOUSE_SCENE_RESTORE_TURNS):
                current_direction = self._get_scalar(w.get_info("direction"))
                if current_direction is None:
                    break
                if self._align_to_direction(w, current_direction, float(original_direction), threshold=10):
                    break
        self.stuck = False
        self.trapped = False
        return False

    def _try_house_exit_when_indoor(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        reason: str,
    ) -> bool:
        if not self._confirm_indoor_after_rear_view(w, direction, reason):
            return False

        print(f"[Running] {reason}且后视角确认 house_scene=indoor，使用 HouseExitManager 脱困")
        self._log_running_state(reason, location, direction, "卡住后确认屋内，优先出房")
        self.stop_auto_forward(w)
        self.house_exit_manager.reset()
        for _ in range(20):
            if self.house_exit_manager.process(w):
                self._clear_after_house_exit(w)
                new_location = self._get_location(w) or location
                self.locations = [new_location]
                self.history_locations = [new_location]
                self.stuck = False
                self.trapped = False
                self.loading_road = False
                self.road_list = []
                self.current_segment_start = None
                print("[Running] HouseExitManager 出房成功，重新规划跑图路线")
                return True

        print("[Running] HouseExitManager 暂未出房，下一帧继续判断")
        self.stuck = False
        self.trapped = False
        return True

    def _clear_after_house_exit(self, w: "FrameWorker"):
        print("[Running] 出房后执行左右斜向清位，避免掉头冲回房内")
        self.stop_auto_forward(w)
        for x_bias in (-self.POST_HOUSE_EXIT_CLEAR_X_BIAS, self.POST_HOUSE_EXIT_CLEAR_X_BIAS):
            w.tap_single(
                "摇杆",
                x_bias=x_bias,
                y_bias=self.POST_HOUSE_EXIT_CLEAR_Y_BIAS,
                dura=self.POST_HOUSE_EXIT_CLEAR_DURA,
                wait=self.POST_HOUSE_EXIT_CLEAR_WAIT,
            )
            w.refresh_frame()

    def _perform_unstuck_action(self, w: "FrameWorker", current_loc: Tuple[int, int]):
        if self._try_house_exit_when_indoor(w, current_loc, None, "脱困前检测"):
            return

        self.stop_auto_forward(w)

        if w.get_info("跳跃"):
            print("[Running] 尝试跳跃配合侧移脱困")
            self._log_running_state("人物卡死", current_loc, None, "跳跃后侧移/斜退脱困")
            w.click("跳跃")
            time.sleep(0.15)
            if self._try_unstuck_move(w, current_loc, "跳跃后左侧移", -280, 0):
                return
            if self._try_unstuck_move(w, current_loc, "跳跃后右侧移", 280, 0):
                return

        print("[Running] 执行侧退绕障脱困")
        self._log_running_state("人物卡死", current_loc, None, "执行侧退绕障脱困")
        for label, bias_x, bias_y in (
            ("左后撤", -240, 260),
            ("右后撤", 240, 260),
            ("左侧移", -300, 0),
            ("右侧移", 300, 0),
            ("后撤拉开距离", 0, 300),
        ):
            if self._try_unstuck_move(w, current_loc, label, bias_x, bias_y):
                return

        for label, turn_bias in (("左转绕开", -self.UNSTUCK_TURN_BIAS), ("右转绕开", self.UNSTUCK_TURN_BIAS)):
            print(f"[Running] {label}，避开原前方障碍后再试探前进")
            self._log_running_state("人物卡死", current_loc, None, label)
            w.tap_single("视角", x_bias=turn_bias, dura=self.UNSTUCK_TURN_DURA, wait=self.UNSTUCK_TURN_WAIT)
            w.refresh_frame()
            if self._try_unstuck_move(w, current_loc, f"{label}后前进", 0, -260):
                return

        print("[Running] 本轮脱困未产生有效位移，清空路径等待下一帧重新判断")
        self.stuck = False
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None

    def _try_unstuck_move(
        self,
        w: "FrameWorker",
        origin: Tuple[int, int],
        label: str,
        x_bias: int,
        y_bias: int,
    ) -> bool:
        print(f"[Running] 脱困动作: {label}, x_bias={x_bias}, y_bias={y_bias}")
        w.tap_single(
            "摇杆",
            x_bias=x_bias,
            y_bias=y_bias,
            dura=self.UNSTUCK_STEP_DURA,
            wait=self.UNSTUCK_STEP_WAIT,
        )
        w.refresh_frame()
        new_loc = self._get_location(w)
        if not new_loc:
            return False

        moved = get_distance(origin, new_loc)
        if moved < self.UNSTUCK_MOVE_THRESHOLD:
            return False

        print(f"[Running] 脱困产生有效位移 {moved:.2f}，从新位置重新规划路径")
        self.stuck = False
        self.trapped = False
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None
        self.locations = [new_loc]
        self.history_locations = [new_loc]
        self.last_valid_location = new_loc
        return True

    def _click_jump_if_available(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if not w.get_info("跳跃"):
            return False

        if not self.jump_click_cooldown.try_acquire(self.JUMP_CLICK_COOLDOWN):
            return False

        print("[Running] 发现跳跃键，点击跳跃")
        self._log_running_state("发现跳跃键", location, direction, "点击跳跃")
        w.click("跳跃")
        return True

    def _get_current_waypoint_tolerance(self) -> float:
        if self.garage_to_roadside_route_active:
            return self.GARAGE_TO_ROADSIDE_TOLERANCE
        return self.WAYPOINT_TOLERANCE

    def _handle_garage_to_roadside_forward_push(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Tuple[int, int],
    ) -> bool:
        if not self.garage_to_roadside_route_active or len(self.road_list) != 1:
            return False

        if tuple(map(int, target)) != tuple(map(int, self.GARAGE_TO_ROADSIDE_POINTS[-1])):
            return False

        if direction is None:
            print("[Running] 离库前推阶段当前朝向无效，等待下一帧")
            return True

        aligned = self._align_to_point(w, location, direction, target, threshold=3)
        if not aligned:
            self._log_running_state("车库离库前推", location, direction, "先对准路边方向", target)
            return True

        print(
            f"[Running] 已到达车库离库点，方向对准 {target}，"
            f"直接前推 {self.GARAGE_TO_ROADSIDE_FORWARD_WAIT}ms"
        )
        self._log_running_state(
            "车库离库前推",
            location,
            direction,
            f"前推 {self.GARAGE_TO_ROADSIDE_FORWARD_WAIT}ms 后开始道路找车",
            target,
        )
        w.tap_single(
            "摇杆",
            y_bias=self.GARAGE_TO_ROADSIDE_FORWARD_BIAS_Y,
            dura=self.GARAGE_TO_ROADSIDE_FORWARD_DURA,
            wait=self.GARAGE_TO_ROADSIDE_FORWARD_WAIT,
        )
        w.refresh_frame()
        self.garage_to_roadside_route_active = False
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None
        print("[Running] 车库离库前推完成，下一帧开始规划道路 node 找车")
        return True

    def _handle_waypoint_arrival(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Tuple[int, int],
        dist: float,
    ):
        if self.garage_to_roadside_route_active:
            if self.road_list:
                reached = self.road_list.pop(0)
                self.current_segment_start = reached
                print(f"[Running] 已到达车库离库点: {reached}")

            if not self.road_list:
                self.garage_to_roadside_route_active = False
                self.loading_road = False
                self.current_segment_start = None
                print("[Running] 车库离库路线已完成，下一帧开始规划道路 node 找车")
            return

        if (
            self.finding_car
            and self.car_search_mode == self.CAR_SEARCH_GARAGE
            and len(self.road_list) <= 1
        ):
            print("[Running] 已到达车库上车点，进入上车精调阶段")
            self._log_running_state("已到达车库上车点", location, direction, "进入上车精调阶段", target, dist)
            self._enter_precise_entry_mode(w)
            self._process_precise_entry(w, location, direction)
            return

        if self.road_list:
            reached = self.road_list.pop(0)
            self.current_segment_start = reached
            if self.current_running_route_kind == self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH:
                self._mark_priority_car_waypoint_reached(reached)
            if self.current_road_node is not None and get_distance(reached, self.current_road_node) <= self.ROAD_NODE_REACHED_TOLERANCE:
                self.visited_road_nodes.add(self.current_road_node)
                print(f"[Running] 已到达道路 node: {self.current_road_node}")
                self.current_road_node = None

        if not self.road_list:
            self.loading_road = False
            self.current_segment_start = None
            print("[Running] 当前路径已走完，准备重新规划")
            if self._handle_priority_car_route_finished(w, location, direction, "到达指定寻车路线终点"):
                return
            self._mark_running_route_completed_if_needed(location, "到达路径终点")

    def _mark_running_route_completed_if_needed(self, location: Tuple[int, int], reason: str):
        if self.finding_car or self.current_running_route_kind != self.RUNNING_ROUTE_CIRCLE:
            return

        self.circle_route_completed = True
        self.current_running_route_kind = None
        if self.last_circle_target_point is None:
            self.last_circle_target_point = location
        print(
            f"[Running] {reason}，本次进圈路线已完成；"
            f"后续跑图围绕圈目标 {self.last_circle_target_point} 随机规划"
        )

    def _advance_waypoint_by_projection(self, location: Tuple[int, int]):
        while len(self.road_list) >= 2:
            if self.current_segment_start is None:
                self.current_segment_start = location
                return

            target = self.road_list[0]
            next_target = self.road_list[1]
            passed_current = self._projection_ratio(self.current_segment_start, target, location)
            next_ratio, next_dist = self._projection_ratio_and_distance(target, next_target, location)

            should_advance = (
                passed_current >= self.WAYPOINT_PROJECTION_PASS_RATIO
                or (0.0 <= next_ratio <= 1.0 and next_dist <= self.WAYPOINT_PROJECTION_CORRIDOR)
            )
            if not should_advance:
                return

            reached = self.road_list.pop(0)
            self.current_segment_start = reached
            if self.current_running_route_kind == self.RUNNING_ROUTE_PRIORITY_CAR_SEARCH:
                self._mark_priority_car_waypoint_reached(reached)
            print(
                f"[Running] 投影已越过锚点，切换下一个目标: reached={reached}, "
                f"passed={passed_current:.2f}, next_ratio={next_ratio:.2f}, next_dist={next_dist:.2f}"
            )
            if self.current_road_node is not None and get_distance(reached, self.current_road_node) <= self.ROAD_NODE_REACHED_TOLERANCE:
                self.visited_road_nodes.add(self.current_road_node)
                self.current_road_node = None

        if not self.road_list:
            self.loading_road = False
            self.current_segment_start = None

    def _projection_ratio(self, start: Tuple[int, int], end: Tuple[int, int], point: Tuple[int, int]) -> float:
        ratio, _ = self._projection_ratio_and_distance(start, end, point)
        return ratio

    def _projection_ratio_and_distance(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        point: Tuple[int, int],
    ) -> Tuple[float, float]:
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

    def _enter_precise_entry_mode(self, w: "FrameWorker"):
        self.stop_auto_forward(w)
        self.precise_entering_car = True
        self.active_vehicle_entry_source = self.VEHICLE_ENTRY_GARAGE
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.precise_face_attempt_index = 0
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.locations = []
        self.history_locations = []
        self.stuck = False
        self.trapped = False
        self.loading_road = False

    def _handle_roadside_vehicle_entry(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if w.get_info("驾驶"):
            self.active_vehicle_entry_source = self.VEHICLE_ENTRY_ROADSIDE
        if self._attempt_drive_after_move(w, "跑图中检查驾驶按钮"):
            return True

        car = self._find_largest_car(w)
        if not self.roadside_car_pursuing and not car:
            return False

        if not self.roadside_car_pursuing:
            self.stop_auto_forward(w)
            if not self._is_roadside_car_candidate(w, location, direction, car):
                return True
            self._start_roadside_car_pursuit(w, location, direction)

        return self._process_roadside_car_pursuit(w, location, direction)

    def _start_roadside_car_pursuit(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        print("[Running] 道路巡游中发现车辆，进入路边追车模式")
        self._log_running_state("道路巡游发现车辆", location, direction, "锁定车辆并尝试靠近上车")
        self.stop_auto_forward(w)
        self.active_vehicle_entry_source = self.VEHICLE_ENTRY_ROADSIDE
        self.roadside_car_pursuing = True
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None
        self._discard_current_road_target()
        self._switch_view_mode(
            w,
            self.VIEW_MODE_FIRST,
            "道路巡游发现车辆，切换第一人称以便视觉对车",
        )

    def _process_roadside_car_pursuit(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        self.roadside_car_steps += 1

        if self._attempt_drive_after_move(
            w,
            f"路边追车前检查驾驶按钮 {self.roadside_car_steps}/{self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT}",
        ):
            return True

        aligned = self._align_to_visible_car(w)
        if aligned is None:
            self.roadside_car_lost_rounds += 1
            self.roadside_car_lost_after_forward_pushes += 1
            print(
                f"[Running] 路边追车中车辆暂时丢失 "
                f"{self.roadside_car_lost_rounds}/{self.ROADSIDE_CAR_LOST_LIMIT}，"
                f"前推后丢失 {self.roadside_car_lost_after_forward_pushes}/"
                f"{self.ROADSIDE_CAR_LOST_FORWARD_LIMIT}，保持方向继续靠近"
            )
            if self._should_try_rear_car_recovery():
                if self._try_recover_overshot_car_behind(w, location, direction):
                    return True
                self._give_up_roadside_car_pursuit("近距离前推后丢车，回头也未发现车辆")
                return True
            if self.roadside_car_lost_after_forward_pushes > self.ROADSIDE_CAR_LOST_FORWARD_LIMIT:
                self._give_up_roadside_car_pursuit("前推后连续丢失车辆，判定为误识别")
                return True
            if self.roadside_car_lost_rounds > self.ROADSIDE_CAR_LOST_LIMIT:
                self._give_up_roadside_car_pursuit("连续多帧未重新识别到车辆")
                return True
        elif not aligned:
            self.roadside_car_lost_rounds = 0
            self.roadside_car_lost_after_forward_pushes = 0
            return True
        else:
            self.roadside_car_lost_rounds = 0
            self.roadside_car_lost_after_forward_pushes = 0
            print(
                f"[Running] 路边追车已对准车辆，前推靠近 "
                f"{self.roadside_car_steps}/{self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT}"
            )

        if aligned is None and self.roadside_car_last_forward_motion is not None:
            forward_bias_y, forward_dura, forward_wait = self._get_lost_car_half_forward_motion()
            print(
                f"[Running] 路边追车丢车补前推使用上次一半时间: "
                f"y_bias={forward_bias_y}, dura={forward_dura}ms, wait={forward_wait}ms"
            )
        else:
            forward_bias_y, forward_dura, forward_wait = self._get_dynamic_car_forward_motion()
        print(
            f"[Running] 路边追车前推 y_bias={forward_bias_y}, "
            f"dura={forward_dura}ms, wait={forward_wait}ms, "
            f"car_area_ratio={self.roadside_car_last_area_ratio}"
        )
        w.tap_single(
            "摇杆",
            y_bias=forward_bias_y,
            dura=forward_dura,
            wait=forward_wait,
        )
        self.roadside_car_last_forward_motion = (forward_bias_y, forward_dura, forward_wait)
        w.refresh_frame()

        if self._attempt_drive_after_move(
            w,
            f"路边追车靠近后检查驾驶按钮 {self.roadside_car_steps}/{self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT}",
        ):
            return True

        if not self._find_largest_car(w):
            recover_result = self._recover_car_lost_after_forward_push(
                w,
                f"路边追车前推后车辆消失 {self.roadside_car_steps}/{self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT}",
            )
            if recover_result in {"entered", "visible"}:
                self.roadside_car_lost_rounds = 0
                self.roadside_car_lost_after_forward_pushes = 0
                return True
            self.roadside_car_lost_after_forward_pushes += 1
            return True

        if self._click_drive_directly_after_move(
            w,
            f"路边追车靠近后直接点击驾驶 {self.roadside_car_steps}/{self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT}",
        ):
            return True

        if self.roadside_car_steps >= self.ROADSIDE_CAR_PURSUIT_STEP_LIMIT:
            self._give_up_roadside_car_pursuit("追车步数达到上限")
        return True

    def _discard_current_road_target(self):
        if self.current_road_node is not None:
            self.visited_road_nodes.add(self.current_road_node)
            print(f"[Running] 放弃当前道路 node，避免追车失败后回头: {self.current_road_node}")
        self.current_road_node = None
        self.loading_road = False
        self.road_list = []
        self.current_segment_start = None

    def _give_up_roadside_car_pursuit(self, reason: str):
        print(f"[Running] 路边追车放弃: {reason}，从当前位置重新规划下一段道路")
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None
        self.active_vehicle_entry_source = None
        self._discard_current_road_target()

    def _switch_to_roadside_car_search(self, reason: str, leave_garage_route: bool = True):
        if leave_garage_route:
            print(f"[Running] {reason}，判定车库暂无可上车辆，先离开车库再切换到沿路找车")
        else:
            print(f"[Running] {reason}，直接切换到沿路找车")
        self.finding_car = True
        self.car_search_mode = self.CAR_SEARCH_ROADSIDE
        self.priority_car_search_active = True
        self.priority_car_search_finished = False
        self.priority_car_search_next_index = 0
        self.priority_car_search_road_points = []
        self.precise_entering_car = False
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.precise_stuck_recoveries = 0
        self.precise_face_attempt_index = 0
        self.precise_view_ready = False
        self.precise_invalid_direction_count = 0
        self.find_car_times = 0
        self.correct_position_times = 0
        self.loading_road = bool(leave_garage_route)
        self.road_list = list(self.GARAGE_TO_ROADSIDE_POINTS) if leave_garage_route else []
        self.current_segment_start = None
        self.current_road_node = None
        self.visited_road_nodes = set()
        self.active_vehicle_entry_source = None
        self.garage_to_roadside_route_active = bool(leave_garage_route)
        self.roadside_car_pursuing = False
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        self.roadside_car_steps = 0
        self.roadside_car_last_area_ratio = None
        self.roadside_car_peak_area_ratio = None
        self.roadside_car_last_forward_motion = None

    def _process_precise_entry(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        self.stop_auto_forward(w)
        self._click_jump_if_available(w, location, direction)

        if direction is None:
            print("[Running] 精调阶段当前朝向无效，等待下一帧")
            return

        dist_to_entry = get_distance(location, self.CAR_ENTRY_POINT)
        print(f"[Running] 精调上车中，当前位置 {location}，上车点 {self.CAR_ENTRY_POINT}，距离 {dist_to_entry:.2f}")
        if self._handle_precise_invalid_direction(location, direction, dist_to_entry):
            return
        self._log_running_state(
            "正在精调上车",
            location,
            direction,
            f"尝试对齐并接近上车点 angle={self._get_current_precise_face_direction()}",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )

        in_micro_adjust_zone = dist_to_entry <= self.PRECISE_ENTRY_MICRO_ADJUST_RADIUS
        if (
            not in_micro_adjust_zone
            and self._handle_precise_entry_blocked(w, location, direction, dist_to_entry)
        ):
            return

        if dist_to_entry > self.PRECISE_ENTRY_MICRO_ADJUST_RADIUS:
            if self.car_search_mode == self.CAR_SEARCH_GARAGE:
                self._ensure_precise_view(w)
                if self._attempt_drive_after_move(w, "靠近车库上车点时先检查驾驶按钮"):
                    return
                if self._find_largest_car(w):
                    print("[Running] 靠近车库上车点时已识别到车辆，提前视觉对车并尝试上车")
                    self._log_running_state(
                        "靠近车库上车点已识别到车辆",
                        location,
                        direction,
                        "不再强制到达上车点，直接视觉对车并前推上车",
                        self.CAR_ENTRY_POINT,
                        dist_to_entry,
                    )
                    if self._approach_visible_car(w):
                        return

            if self._update_precise_progress(dist_to_entry):
                self._handle_precise_entry_no_progress(w, location, direction, dist_to_entry)
                return

            if not self._align_to_point(w, location, direction, self.CAR_ENTRY_POINT, threshold=3):
                return
            self._tap_distance_forward_with_learning(
                w,
                self.CAR_ENTRY_POINT,
                dist_to_entry,
                "精调靠近上车点",
            )
            return

        print(
            f"[Running] 已进入上车点微调区 dist={dist_to_entry:.2f}，"
            f"先对准车库朝向 {self.CAR_FACE_DIRECTION}"
        )
        face_aligned = self._align_to_direction(w, direction, self.CAR_FACE_DIRECTION, threshold=3)
        if not face_aligned:
            return

        self._ensure_precise_view(w)

        if self._attempt_drive_after_move(w, "已到达上车点，先检查驾驶按钮"):
            return

        if self.car_search_mode == self.CAR_SEARCH_GARAGE and not self._find_largest_car(w):
            self._switch_to_roadside_car_search("已到达车库上车点，但当前画面未检测到车辆")
            return

        print("[Running] 已对准车库，开始视觉对车并前推尝试上车")
        self._log_running_state(
            "已对准车库朝向",
            location,
            direction,
            "切换第一人称后开始视觉寻车",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )
        if self._approach_visible_car(w):
            return

        self._handle_visual_entry_failure(w)

    def _handle_precise_invalid_direction(
        self,
        location: Tuple[int, int],
        direction: Optional[float],
        dist_to_entry: float,
    ) -> bool:
        if self.car_search_mode != self.CAR_SEARCH_GARAGE or direction != -1:
            return False

        self.precise_invalid_direction_count += 1
        print(
            f"[Running] 车库靠近/精调阶段方向值为 -1，累计 "
            f"{self.precise_invalid_direction_count}/{self.PRECISE_ENTRY_INVALID_DIRECTION_LIMIT}"
        )
        self._log_running_state(
            "车库靠近方向异常",
            location,
            direction,
            "累计方向 -1，暂不继续靠近车库点",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )
        if self.precise_invalid_direction_count >= self.PRECISE_ENTRY_INVALID_DIRECTION_LIMIT:
            self._switch_to_roadside_car_search("车库点方向多次为 -1")
        return True

    def _is_garage_entry_target(self, target: Tuple[int, int]) -> bool:
        return (
            self.finding_car
            and self.car_search_mode == self.CAR_SEARCH_GARAGE
            and tuple(map(int, target)) == tuple(map(int, self.CAR_ENTRY_POINT))
        )

    def _handle_precise_entry_blocked(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        dist_to_entry: float,
    ) -> bool:
        self._check_if_stuck(location)
        self._check_if_trapped(location)

        if self.trapped:
            if self.car_search_mode == self.CAR_SEARCH_GARAGE:
                self._switch_to_roadside_car_search("精调靠近车库上车点时局部打转")
                return True
            print("[Running] 精调上车阶段人物困死，结束当前局")
            self._log_running_state("精调上车困死", location, direction, "结束当前局", self.CAR_ENTRY_POINT, dist_to_entry)
            self._handle_death(w)
            return True

        if not self.stuck:
            return False

        self.precise_stuck_recoveries += 1
        print(
            f"[Running] 精调靠近上车点时卡住，执行脱困 "
            f"{self.precise_stuck_recoveries}/{self.PRECISE_ENTRY_STUCK_SWITCH_LIMIT}"
        )
        self._log_running_state(
            "精调上车卡住",
            location,
            direction,
            "执行脱困并重新逼近上车点",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )

        if (
            self.car_search_mode == self.CAR_SEARCH_GARAGE
            and self.precise_stuck_recoveries >= self.PRECISE_ENTRY_STUCK_SWITCH_LIMIT
        ):
            self._switch_to_roadside_car_search("多次靠近车库上车点卡住")
            return True

        self._reset_precise_entry_motion_state(location)
        self._perform_unstuck_action(w, location)
        self._reset_precise_entry_motion_state(self._get_location(w) or location)
        return True

    def _handle_precise_entry_no_progress(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        dist_to_entry: float,
    ):
        self.precise_stuck_recoveries += 1
        print(
            f"[Running] 靠近车库上车点距离无进展，执行脱困 "
            f"{self.precise_stuck_recoveries}/{self.PRECISE_ENTRY_STUCK_SWITCH_LIMIT}"
        )
        self._log_running_state(
            "精调上车无进展",
            location,
            direction,
            "执行脱困并重试上车点",
            self.CAR_ENTRY_POINT,
            dist_to_entry,
        )

        if (
            self.car_search_mode == self.CAR_SEARCH_GARAGE
            and self.precise_stuck_recoveries >= self.PRECISE_ENTRY_STUCK_SWITCH_LIMIT
        ):
            self._switch_to_roadside_car_search("多次靠近车库上车点无进展")
            return

        self._reset_precise_entry_motion_state(location)
        self._perform_unstuck_action(w, location)
        self._reset_precise_entry_motion_state(self._get_location(w) or location)

    def _reset_precise_entry_motion_state(self, location: Tuple[int, int]):
        self.precise_last_distance = None
        self.precise_idle_rounds = 0
        self.locations = [location]
        self.history_locations = [location]
        self.stuck = False
        self.trapped = False
        self.loading_road = False
        self.road_list = [self.CAR_ENTRY_POINT]
        self.current_segment_start = None

    def _update_precise_progress(self, dist_to_entry: float) -> bool:
        if self.precise_last_distance is None:
            self.precise_last_distance = dist_to_entry
            self.precise_idle_rounds = 0
            return False

        if dist_to_entry < self.precise_last_distance:
            self.precise_last_distance = dist_to_entry
            self.precise_idle_rounds = 0
            self.precise_stuck_recoveries = 0
            return False

        self.precise_idle_rounds += 1
        if self.precise_idle_rounds >= self.PRECISE_ENTRY_IDLE_UNSTUCK_ROUNDS:
            self.correct_position_times += 1
            self.precise_idle_rounds = 0
            print(f"[Running] 精调阶段长时间无进展，累计失败 {self.correct_position_times}")
            return True
        return False

    def _attempt_drive_after_move(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[Running] {reason}，尝试点击驾驶按钮")
        location = self._get_location(w)
        direction = self._get_scalar(w.get_info("direction"))
        if location is not None:
            self._log_running_state("执行上车尝试", location, direction, reason)

        drive_btn = w.get_info("驾驶")
        if not drive_btn:
            return False

        print("[Running] 检测到驾驶按钮，执行上车")
        if location is not None:
            self._log_running_state("检测到驾驶按钮", location, direction, "点击上车")
        w.click(drive_btn)
        return self._finish_drive_entry_click(w)

    def _click_drive_directly_after_move(self, w: "FrameWorker", reason: str) -> bool:
        print(f"[Running] {reason}，不预检查按钮，直接点击驾驶")
        location = self._get_location(w)
        direction = self._get_scalar(w.get_info("direction"))
        if location is not None:
            self._log_running_state("执行上车尝试", location, direction, reason)

        w.click("驾驶")
        return self._finish_drive_entry_click(w)

    def _finish_drive_entry_click(self, w: "FrameWorker") -> bool:
        time.sleep(1)
        w.refresh_frame()
        if self._is_in_vehicle(w):
            print("[Running] 上车成功")
            entry_source = self.active_vehicle_entry_source or (
                self.VEHICLE_ENTRY_GARAGE if self.precise_entering_car else self.VEHICLE_ENTRY_UNKNOWN
            )
            self.precise_entering_car = False
            self._restore_vehicle_view(w)
            self.stop_auto_forward(w)
            self.reset(finding_car=False)
            self.last_vehicle_entry_source = entry_source
            w.change_stage("开车阶段")
            return True

        print("[Running] 点击驾驶后仍未上车")
        return False

    def _handle_visual_entry_failure(self, w: "FrameWorker"):
        current_direction = self._get_current_precise_face_direction()
        self.precise_face_attempt_index += 1
        self.find_car_times = self.precise_face_attempt_index
        print(
            f"[Running] 车库朝向 {current_direction} 视觉寻车未成功，"
            f"累计失败 {self.find_car_times}/{len(self.PRECISE_FACE_DIRECTIONS)}"
        )
        if self.precise_face_attempt_index >= len(self.PRECISE_FACE_DIRECTIONS):
            return

        next_direction = self._get_current_precise_face_direction()
        print(f"[Running] 回到车库上车点后切换到朝向 {next_direction} 再试")
        self._return_to_entry_point(w)

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
        self.precise_stuck_recoveries = 0

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

    def _ensure_precise_view(self, w: "FrameWorker"):
        if self.precise_view_ready:
            return
        print("[Running] 到达上车点，切换第一人称以便视觉对车")
        self._switch_view_mode(
            w,
            self.VIEW_MODE_FIRST,
            "到达上车点，切换第一人称以便视觉对车",
        )
        self.precise_view_ready = True

    def _restore_vehicle_view(self, w: "FrameWorker"):
        print("[Running] 上车成功，切回第三人称")
        self._switch_view_mode(
            w,
            self.VIEW_MODE_THIRD,
            "上车成功，切回第三人称",
        )
        self.precise_view_ready = False

    def set_view_mode(self, mode: str):
        if mode in (self.VIEW_MODE_FIRST, self.VIEW_MODE_THIRD):
            self.current_view_mode = mode

    def _switch_view_mode(
        self,
        w: "FrameWorker",
        target_mode: str,
        reason: str,
    ) -> bool:
        if self.current_view_mode == target_mode:
            return False

        print(f"[Running] {reason}")
        w.click(self.VIEW_SWITCH_BUTTON)
        self.current_view_mode = target_mode
        time.sleep(0.2)
        w.refresh_frame()
        return True

    def _ensure_first_person_view(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        if self.current_view_mode == self.VIEW_MODE_FIRST:
            return False
        self._log_running_state("人称切换", location, direction, "切换第一人称")
        return self._switch_view_mode(w, self.VIEW_MODE_FIRST, "当前处于跑图阶段，切换第一人称")

    def _ensure_third_person_view(
        self,
        w: "FrameWorker",
        location: Optional[Tuple[int, int]] = None,
        direction: Optional[float] = None,
        reason: str = "切换第三人称",
    ) -> bool:
        if self.current_view_mode == self.VIEW_MODE_THIRD:
            return False
        if location is not None:
            self._log_running_state("人称切换", location, direction, "切换第三人称")
        return self._switch_view_mode(w, self.VIEW_MODE_THIRD, reason)

    def _find_largest_car(self, w: "FrameWorker"):
        scene = w.get_info("forward_scene")
        if not scene:
            return None

        cars = [
            obj for obj in scene
            if isinstance(obj, (list, tuple)) and len(obj) >= 6 and int(obj[5]) == 7
        ]
        if not cars:
            return None

        return max(cars, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def _get_detection_area_ratio(self, w: "FrameWorker", det) -> Optional[float]:
        frame = getattr(w, "frame", None)
        if frame is None or not hasattr(frame, "shape"):
            return None
        try:
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(max(1, int(frame_w) * int(frame_h)))
            box_area = max(0.0, float(det[2]) - float(det[0])) * max(0.0, float(det[3]) - float(det[1]))
            return box_area / frame_area
        except Exception:
            return None

    def _estimate_car_map_position(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        det,
    ) -> Optional[Tuple[Tuple[int, int], float, float]]:
        if direction is None or direction < 0:
            return None

        frame_w = self._get_visual_frame_width(w)
        if not frame_w:
            return None

        area_ratio = self._get_detection_area_ratio(w, det)
        if area_ratio is None or area_ratio <= 0:
            return None

        car_center_x = (float(det[0]) + float(det[2])) / 2.0
        center_offset_ratio = (car_center_x - (float(frame_w) / 2.0)) / max(1.0, float(frame_w) / 2.0)
        angle_offset = center_offset_ratio * (self.ROADSIDE_CAR_ESTIMATE_FOV_DEGREES / 2.0)
        estimated_direction = (float(direction) + angle_offset) % 360.0
        estimated_distance = self.ROADSIDE_CAR_DISTANCE_SCALE / math.sqrt(area_ratio)
        estimated_distance = max(
            self.ROADSIDE_CAR_MIN_ESTIMATED_DISTANCE,
            min(self.ROADSIDE_CAR_MAX_ESTIMATED_DISTANCE, estimated_distance),
        )

        rad = math.radians(estimated_direction - 90.0)
        estimated_x = int(round(float(location[0]) + estimated_distance * math.cos(rad)))
        estimated_y = int(round(float(location[1]) + estimated_distance * math.sin(rad)))
        return (estimated_x, estimated_y), estimated_distance, estimated_direction

    def _is_roadside_car_candidate(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        car,
    ) -> bool:
        player_road_point, player_road_dist = self.map_tool.nearest_walkable_within_radius(
            location,
            self.ROADSIDE_CAR_MAX_PLAYER_ROAD_DISTANCE,
        )
        if player_road_point is not None:
            print(
                f"[Running] 发现车辆且人物已在路边，当前位置 {location}, "
                f"附近 mask 道路点 {player_road_point}, player_road_dist={player_road_dist:.2f}，开始追车"
            )
            self._log_running_state(
                "路边车辆确认",
                location,
                direction,
                (
                    f"人物 {self.ROADSIDE_CAR_MAX_PLAYER_ROAD_DISTANCE:.2f} 距离内存在 mask 道路点，"
                    "放宽车辆路边判断"
                ),
                player_road_point,
                player_road_dist,
            )
            return True

        print(
            f"[Running] 发现车辆但人物不在路上，当前位置 {location}, "
            f"{self.ROADSIDE_CAR_MAX_PLAYER_ROAD_DISTANCE:.2f} 距离内无 mask 道路点，先停车不追车"
        )
        self._log_running_state(
            "路边车辆过滤",
            location,
            direction,
            (
                f"人物 {self.ROADSIDE_CAR_MAX_PLAYER_ROAD_DISTANCE:.2f} 距离内无 mask 道路点，暂不追车"
            ),
            player_road_point,
            player_road_dist,
        )
        return False

    def _get_dynamic_car_forward_motion(self) -> Tuple[int, int, int]:
        return (
            self._get_dynamic_car_forward_bias_y(),
            self._get_dynamic_car_forward_dura(),
            self._get_dynamic_car_forward_wait(),
        )

    def _get_lost_car_half_forward_motion(self) -> Tuple[int, int, int]:
        if self.roadside_car_last_forward_motion is None:
            return self._get_dynamic_car_forward_motion()

        bias_y, dura, wait = self.roadside_car_last_forward_motion
        half_dura = max(1, int(round(dura * 0.5)))
        half_wait = max(half_dura, int(round(wait * 0.5)))
        return bias_y, half_dura, half_wait

    def _get_dynamic_car_forward_bias_y(self) -> int:
        ratio = self.roadside_car_last_area_ratio
        if ratio is None:
            return self.CAR_VISUAL_FORWARD_BIAS_Y

        if ratio <= self.CAR_VISUAL_DYNAMIC_MID_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_FAR_BIAS_Y
        if ratio >= self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_MIN_BIAS_Y

        bias_y = self._interpolate_car_forward_wait(
            ratio,
            [
                (self.CAR_VISUAL_DYNAMIC_MID_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_FAR_BIAS_Y),
                (self.CAR_VISUAL_DYNAMIC_CLOSE_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_CLOSE_BIAS_Y),
                (self.CAR_VISUAL_DYNAMIC_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_NEAR_BIAS_Y),
                (self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_VERY_NEAR_BIAS_Y),
            ],
        )
        return int(round(bias_y))

    def _get_dynamic_car_forward_dura(self) -> int:
        ratio = self.roadside_car_last_area_ratio
        if ratio is None:
            return self.CAR_VISUAL_FORWARD_DURA

        if ratio <= self.CAR_VISUAL_DYNAMIC_MID_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_FAR_DURA
        if ratio >= self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_MIN_DURA

        dura = self._interpolate_car_forward_wait(
            ratio,
            [
                (self.CAR_VISUAL_DYNAMIC_MID_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_FAR_DURA),
                (self.CAR_VISUAL_DYNAMIC_CLOSE_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_CLOSE_DURA),
                (self.CAR_VISUAL_DYNAMIC_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_NEAR_DURA),
                (self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_VERY_NEAR_DURA),
            ],
        )
        return int(round(dura))

    def _get_dynamic_car_forward_wait(self) -> int:
        ratio = self.roadside_car_last_area_ratio
        if ratio is None:
            return self.CAR_VISUAL_FORWARD_WAIT

        if ratio <= self.CAR_VISUAL_DYNAMIC_FAR_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_MAX_WAIT
        if ratio >= self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO:
            return self.CAR_VISUAL_DYNAMIC_MIN_WAIT

        wait = self._interpolate_car_forward_wait(
            ratio,
            [
                (self.CAR_VISUAL_DYNAMIC_FAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_MAX_WAIT),
                (self.CAR_VISUAL_DYNAMIC_MID_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_FAR_WAIT),
                (self.CAR_VISUAL_DYNAMIC_CLOSE_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_CLOSE_WAIT),
                (self.CAR_VISUAL_DYNAMIC_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_NEAR_WAIT),
                (self.CAR_VISUAL_DYNAMIC_VERY_NEAR_AREA_RATIO, self.CAR_VISUAL_DYNAMIC_VERY_NEAR_WAIT),
            ],
        )
        return int(round(wait))

    def _interpolate_car_forward_wait(self, ratio: float, anchors) -> float:
        for (left_ratio, left_wait), (right_ratio, right_wait) in zip(anchors, anchors[1:]):
            if left_ratio <= ratio <= right_ratio:
                progress = (ratio - left_ratio) / max(0.000001, right_ratio - left_ratio)
                return left_wait + progress * (right_wait - left_wait)
        return anchors[-1][1]

    def _should_try_rear_car_recovery(self) -> bool:
        peak_ratio = self.roadside_car_peak_area_ratio or 0.0
        last_ratio = self.roadside_car_last_area_ratio or 0.0
        return (
            self.roadside_car_lost_after_forward_pushes >= 1
            and max(peak_ratio, last_ratio) >= self.ROADSIDE_CAR_OVERSHOOT_AREA_RATIO
        )

    def _try_recover_overshot_car_behind(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        print(
            "[Running] 近距离前推后车辆丢失，怀疑已经越过车辆，"
            "转身检查身后是否有车"
        )
        self._log_running_state("近车过冲复核", location, direction, "转身检查身后车辆")
        self.stop_auto_forward(w)

        original_direction = self._get_scalar(w.get_info("direction"))
        if original_direction is None:
            original_direction = self._get_scalar(direction)

        if original_direction is None:
            w.tap_single("视角", x_bias=self.UNSTUCK_TURN_BIAS * 2, dura=850, wait=500)
            w.refresh_frame()
        else:
            rear_direction = (float(original_direction) + 180.0) % 360.0
            print(
                f"[Running] 以当前角度 {float(original_direction):.1f} 为基准，"
                f"回头转向 {rear_direction:.1f}"
            )
            for _ in range(self.ROADSIDE_CAR_REAR_TURN_ATTEMPTS):
                current_direction = self._get_scalar(w.get_info("direction"))
                if current_direction is None:
                    break
                if self._align_to_direction(w, current_direction, rear_direction, threshold=8):
                    break
            w.refresh_frame()

        rear_car = self._find_largest_car(w)
        if not rear_car:
            print("[Running] 回头后未发现车辆")
            if original_direction is not None:
                for _ in range(2):
                    current_direction = self._get_scalar(w.get_info("direction"))
                    if current_direction is None:
                        break
                    if self._align_to_direction(w, current_direction, float(original_direction), threshold=10):
                        break
            return False

        area_ratio = self._get_detection_area_ratio(w, rear_car)
        if area_ratio is not None:
            self.roadside_car_last_area_ratio = area_ratio
            self.roadside_car_peak_area_ratio = max(self.roadside_car_peak_area_ratio or 0.0, area_ratio)
        print(f"[Running] 回头发现身后车辆，area_ratio={area_ratio}，执行微调上车")

        aligned = self._align_to_visible_car(w)
        if aligned is None:
            return False
        if not aligned:
            self.roadside_car_lost_rounds = 0
            self.roadside_car_lost_after_forward_pushes = 0
            return True

        w.tap_single(
            "摇杆",
            y_bias=self.ROADSIDE_CAR_REAR_FORWARD_BIAS_Y,
            dura=self.ROADSIDE_CAR_REAR_FORWARD_DURA,
            wait=self.ROADSIDE_CAR_REAR_FORWARD_WAIT,
        )
        w.refresh_frame()
        self.roadside_car_lost_rounds = 0
        self.roadside_car_lost_after_forward_pushes = 0
        return self._click_drive_directly_after_move(w, "回头发现身后车辆后微调点击驾驶") or True

    def _get_visual_frame_width(self, w: "FrameWorker") -> Optional[int]:
        frame = getattr(w, "frame", None)
        if frame is None:
            return None
        try:
            return int(frame.shape[1])
        except Exception:
            return None

    def _get_visible_car_center_offset(self, w: "FrameWorker") -> Optional[float]:
        car = self._find_largest_car(w)
        if not car:
            return None

        area_ratio = self._get_detection_area_ratio(w, car)
        if area_ratio is not None:
            self.roadside_car_last_area_ratio = area_ratio
            self.roadside_car_peak_area_ratio = max(self.roadside_car_peak_area_ratio or 0.0, area_ratio)

        frame_w = self._get_visual_frame_width(w)
        if not frame_w:
            return None

        screen_w = self.screen_w
        if not screen_w:
            screen_w, _ = get_resolution()
            self.screen_w = screen_w
        if not screen_w:
            return None

        car_center_x = (float(car[0]) + float(car[2])) / 2.0
        offset_real = (car_center_x - (frame_w / 2.0)) * (float(screen_w) / float(frame_w))
        print(f"[Running] 检测到车辆，视觉中心偏移 {offset_real:.2f}px")
        return offset_real

    def _align_to_visible_car(self, w: "FrameWorker") -> Optional[bool]:
        offset_real = self._get_visible_car_center_offset(w)
        if offset_real is None:
            return None

        if abs(offset_real) <= self.CAR_ALIGN_CENTER_THRESHOLD:
            return True

        adjust_val = int(offset_real * self.CAR_ALIGN_STEP_RATIO)
        adjust_val = max(-self.CAR_ALIGN_MAX_BIAS, min(self.CAR_ALIGN_MAX_BIAS, adjust_val))
        print(f"[Running] 使用视角对准车辆，x_bias={adjust_val}")
        w.tap_single("视角", x_bias=adjust_val, dura=self.CAR_ALIGN_DURA, wait=self.CAR_ALIGN_WAIT)
        w.refresh_frame()
        return False

    def _approach_visible_car(self, w: "FrameWorker") -> bool:
        if self.CAR_APPROACH_USE_SENDEVENT_UINPUT:
            if self._approach_visible_car_mixed_control(w):
                return True
            if not self.CAR_APPROACH_FALLBACK_TO_LEGACY:
                return False
            print("[Running] sendevent+uinput 视觉靠车未成功，回退原始单次前推方案")

        return self._approach_visible_car_legacy(w)

    def _approach_visible_car_mixed_control(self, w: "FrameWorker") -> bool:
        controller = getattr(w, "controller", None)
        if getattr(controller, "backend", None) != "sendevent":
            print("[Running] 当前触控后端不是 sendevent，跳过 sendevent+uinput 视觉靠车方案")
            return False

        joystick_pressed = False
        try:
            for step in range(self.CAR_APPROACH_MIXED_MAX_STEPS):
                step_text = f"{step + 1}/{self.CAR_APPROACH_MIXED_MAX_STEPS}"
                if self._attempt_drive_after_move(w, f"sendevent+uinput 靠车前检查驾驶按钮 {step_text}"):
                    return True

                offset_real = self._get_visible_car_center_offset(w)
                if offset_real is None:
                    print("[Running] sendevent+uinput 靠车时未检测到车辆，停止新方案")
                    return False

                forward_bias_y, _, _ = self._get_dynamic_car_forward_motion()
                print(
                    f"[Running] sendevent 按住摇杆靠车 {step_text}: "
                    f"y_bias={forward_bias_y}, car_area_ratio={self.roadside_car_last_area_ratio}"
                )

                if not joystick_pressed:
                    w.move_press(0, "摇杆")
                    joystick_pressed = True
                w.move_to(
                    0,
                    "摇杆",
                    y_bias=forward_bias_y,
                    duration_ms=self.CAR_APPROACH_MIXED_MOVE_DURA,
                )

                if abs(offset_real) > self.CAR_APPROACH_MIXED_CENTER_THRESHOLD:
                    view_bias = int(offset_real * self.CAR_APPROACH_MIXED_VIEW_STEP_RATIO)
                    view_bias = max(
                        -self.CAR_APPROACH_MIXED_MAX_VIEW_BIAS,
                        min(self.CAR_APPROACH_MIXED_MAX_VIEW_BIAS, view_bias),
                    )
                    print(f"[Running] uinput 同步调整视角靠车 {step_text}: x_bias={view_bias}")
                    w.uinput_tap_single(
                        "视角",
                        x_bias=view_bias,
                        dura=self.CAR_APPROACH_MIXED_VIEW_DURA,
                        wait=self.CAR_APPROACH_MIXED_VIEW_WAIT,
                    )

                w.refresh_frame()
                if w.get_info("驾驶"):
                    print("[Running] sendevent+uinput 靠车检测到驾驶按钮，松开摇杆后点击上车")
                    w.move_up(0)
                    joystick_pressed = False
                    return self._attempt_drive_after_move(w, f"sendevent+uinput 靠车后点击驾驶 {step_text}")

            print("[Running] sendevent+uinput 靠车达到步数上限，未检测到驾驶按钮")
            return False
        except Exception as exc:
            print(f"[Running] sendevent+uinput 靠车异常，准备回退原始方案: {exc}")
            return False
        finally:
            if joystick_pressed:
                try:
                    w.move_up(0)
                except Exception as exc:
                    print(f"[Running] sendevent+uinput 靠车松开摇杆失败: {exc}")

    def _approach_visible_car_legacy(self, w: "FrameWorker") -> bool:
        for step in range(self.CAR_VISUAL_SEARCH_MAX_STEPS):
            if self._attempt_drive_after_move(w, f"视觉寻车前检查驾驶按钮 {step + 1}/{self.CAR_VISUAL_SEARCH_MAX_STEPS}"):
                return True

            aligned = self._align_to_visible_car(w)
            if aligned is None:
                print("[Running] 当前画面未检测到车辆，保持朝向向前推进")
            elif not aligned:
                continue
            else:
                print(f"[Running] 已对准车辆，执行前推尝试上车 {step + 1}/{self.CAR_VISUAL_SEARCH_MAX_STEPS}")

            forward_bias_y, forward_dura, forward_wait = self._get_dynamic_car_forward_motion()
            print(
                f"[Running] 视觉对车前推 y_bias={forward_bias_y}, "
                f"dura={forward_dura}ms, wait={forward_wait}ms, "
                f"car_area_ratio={self.roadside_car_last_area_ratio}"
            )
            w.tap_single(
                "摇杆",
                y_bias=forward_bias_y,
                dura=forward_dura,
                wait=forward_wait,
            )
            w.refresh_frame()

            if self._attempt_drive_after_move(w, f"视觉对车后检查驾驶按钮 {step + 1}/{self.CAR_VISUAL_SEARCH_MAX_STEPS}"):
                return True

            if not self._find_largest_car(w):
                recover_result = self._recover_car_lost_after_forward_push(
                    w,
                    f"视觉对车前推后车辆消失 {step + 1}/{self.CAR_VISUAL_SEARCH_MAX_STEPS}",
                )
                if recover_result == "entered":
                    return True
                if recover_result == "visible":
                    continue
                return False

            if self._click_drive_directly_after_move(w, f"视觉对车后直接点击驾驶 {step + 1}/{self.CAR_VISUAL_SEARCH_MAX_STEPS}"):
                return True

        return False

    def _recover_car_lost_after_forward_push(self, w: "FrameWorker", reason: str) -> str:
        print(
            f"[Running] {reason}，后拉 {self.CAR_FORWARD_LOST_BACKOFF_WAIT}ms "
            "复核驾驶按钮和车辆位置"
        )
        w.tap_single(
            "摇杆",
            y_bias=self.CAR_FORWARD_LOST_BACKOFF_Y_BIAS,
            dura=self.CAR_FORWARD_LOST_BACKOFF_DURA,
            wait=self.CAR_FORWARD_LOST_BACKOFF_WAIT,
        )
        w.refresh_frame()

        if self._attempt_drive_after_move(w, f"{reason}后拉后检查驾驶按钮"):
            return "entered"

        car = self._find_largest_car(w)
        if car:
            area_ratio = self._get_detection_area_ratio(w, car)
            if area_ratio is not None:
                self.roadside_car_last_area_ratio = area_ratio
                self.roadside_car_peak_area_ratio = max(self.roadside_car_peak_area_ratio or 0.0, area_ratio)
            print(f"[Running] {reason}后拉后重新发现车辆，area_ratio={area_ratio}")
            return "visible"

        print(f"[Running] {reason}后拉后仍未发现车辆")
        return "lost"

    def _align_to_point(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: float,
        target: Tuple[int, int],
        threshold: int = 5,
        max_steps: int = 1,
    ) -> bool:
        target_dir = calculate_angle(location, target)
        return execute_view_turn(
            w,
            direction,
            target_dir,
            threshold=threshold,
            max_steps=max_steps,
            wait=250,
            log_prefix="[RunningAlign]",
        )

    def _align_to_direction(
        self,
        w: "FrameWorker",
        direction: float,
        target_direction: float,
        threshold: int = 5,
    ) -> bool:
        return execute_view_turn(
            w,
            direction,
            target_direction,
            threshold=threshold,
            max_steps=1,
            wait=250,
            log_prefix="[RunningDirection]",
        )

    def _precise_approach_waypoint(
        self,
        w: "FrameWorker",
        location: Tuple[int, int],
        direction: Optional[float],
        target: Tuple[int, int],
        dist: float,
    ):
        self.stop_auto_forward(w)

        if self._is_garage_entry_target(target) and direction == -1:
            self._handle_precise_invalid_direction(location, direction, dist)
            return

        if direction is None:
            print("[Running] 精确逼近时当前朝向无效，等待下一帧")
            self._log_running_state("精确逼近朝向无效", location, None, "等待下一帧", target, dist)
            return

        align_threshold = 2 if dist < 5 else 5
        align_max_steps = 3 if dist < 5 else 1
        aligned = self._align_to_point(
            w,
            location,
            direction,
            target,
            threshold=align_threshold,
            max_steps=align_max_steps,
        )
        if not aligned:
            self._log_running_state(
                "精确逼近调方向",
                location,
                direction,
                f"先对齐目标点 threshold={align_threshold}",
                target,
                dist,
            )
            return

        mode = self._forward_model_mode(dist)
        y_bias, dura, wait, model_dist = self._get_distance_forward_motion(mode, dist)
        self._log_running_state(
            "精确逼近目标点",
            location,
            direction,
            f"{mode}前推 y_bias={y_bias}, dura={dura}, wait={wait}, model_dist={model_dist}",
            target,
            dist,
        )
        self._tap_distance_forward_with_learning(w, target, dist, "精确逼近目标点")

    def _forward_model_mode(self, dist: float) -> str:
        try:
            dist_val = float(dist)
        except (TypeError, ValueError):
            dist_val = 0.0
        return "fast" if dist_val > self.WAYPOINT_PRECISE_APPROACH_DISTANCE else "slow"

    @staticmethod
    def _forward_model_fallback_wait(mode: str, dist: float) -> int:
        try:
            dist_val = max(0.0, float(dist))
        except (TypeError, ValueError):
            dist_val = 0.0
        if mode == "fast":
            return int(max(180, min(7000, dist_val * 32 + 220)))
        return int(max(180, min(7000, dist_val * 60 + 300)))

    def _get_distance_forward_motion(self, mode: str, dist: float):
        fallback_y_bias = -500 if mode == "fast" else -100
        fallback_dura = 300
        fallback_wait = self._forward_model_fallback_wait(mode, dist)
        return get_adaptive_forward_motion(
            mode,
            dist,
            fallback_y_bias,
            fallback_dura,
            fallback_wait,
        )

    def _tap_distance_forward_with_learning(
        self,
        w: "FrameWorker",
        target: Tuple[int, int],
        dist: float,
        reason: str,
    ):
        mode = self._forward_model_mode(dist)
        y_bias, dura, wait, model_dist = self._get_distance_forward_motion(mode, dist)
        print(
            f"[Running] {reason}: mode={mode}, dist={dist:.2f}, "
            f"model_dist={model_dist}, y_bias={y_bias}, dura={dura}, wait={wait}"
        )
        w.tap_single(
            "摇杆",
            y_bias=y_bias,
            dura=dura,
            wait=wait,
        )
        w.refresh_frame()
        after_location = self._get_location(w)
        after_dist = get_distance(after_location, target) if after_location is not None else None
        update_adaptive_forward_motion(mode, dist, dist, after_dist, y_bias, dura, wait)

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

        motion_ok = execute_view_turn(
            w,
            direction,
            target_dir,
            threshold=5,
            max_steps=1,
            wait=300,
            fallback_dura=max(400, int(pixel * 1.5)),
            log_prefix="[Correct Dire]",
        )
        self._log_running_state(
            "跑图方向偏移",
            location,
            direction,
            "统一角度模型调整视角",
            target,
            get_distance(location, target),
        )
        return motion_ok


    def stop_auto_forward(self, w: "FrameWorker"):
        if self.auto_forward:
            w.click("自动前进")
            self.auto_forward = False
