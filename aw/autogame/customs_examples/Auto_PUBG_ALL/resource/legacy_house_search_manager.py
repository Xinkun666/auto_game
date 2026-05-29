import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.map_navigation import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search import (
    EmbeddedHouseSearchAdapter,
    HouseSearchAdapter,
    QwenRoomSearchAgent,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation_geometry import *
from aw.autogame.tools.Utils import *

if TYPE_CHECKING:
    # 假设你的框架类定义在 framework.py 文件中
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class LegacyHouseSearchManager:
    HOUSE_INDOOR = 0
    HOUSE_OUTDOOR = 1
    HOUSE_ROOFTOP = 2

    STATUS_SCAN_ROOM = "SCAN_ROOM"
    STATUS_LOOT_ITEM = "LOOT_ITEM"
    STATUS_SELECT_DOOR = "SELECT_DOOR"
    STATUS_APPROACH_DOOR = "APPROACH_DOOR"
    STATUS_ENTER_NEXT_ROOM = "ENTER_NEXT_ROOM"
    STATUS_FINISHED = "FINISHED"

    SUPPLY_CLASS_IDS = {1}
    PICK_MENU_CLASS_IDS = {3}
    DOOR_CLASS_IDS = {0, 4}
    ROOM_CLUSTER_SIZE = 4.0
    ROOM_LOOT_LIMIT = 3
    ROOM_SCAN_STEPS = 6
    ROOM_SCAN_TURN_BIAS = 430
    ROOM_SCAN_TURN_DEGREES = 60.0
    TARGET_DEDUP_ANGLE = 8.0
    TARGET_DEDUP_BOX_H = 24.0
    TARGET_MATCH_ANGLE = 14.0
    ITEM_APPROACH_MAX_STEPS = 8
    PICK_MENU_AUTO_PICKUP_WAIT = 5.0
    DOOR_APPROACH_MAX_STEPS = 8
    ROOM_TRANSITION_DISTANCE = 1.6
    CENTER_TOLERANCE_PX = 90
    MAX_ROOMS_PER_FLOOR = 12

    def __init__(self):
        self.map_tool = MapNavigator()
        self.house_data = load_json(
            r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/house_entry/house_entries_summary.json')

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
        self.pubg_room_search_adapter: Optional[HouseSearchAdapter] = None
        self.embedded_room_search_adapter: Optional[EmbeddedHouseSearchAdapter] = None
        self.pubg_room_search_attempted = False
        self.qwen_room_search_agent: Optional[QwenRoomSearchAgent] = None

        self.reset()

    def reset(self):
        self.status = self.STATUS_SCAN_ROOM
        self.current_room_id = None
        self.current_room_location: Optional[Tuple[int, int]] = None
        self.visited_rooms = set()
        self.room_loot_counts: Dict[Any, int] = {}
        self.room_supplies: Dict[Any, List[Dict[str, Any]]] = {}
        self.room_doors: Dict[Any, List[Dict[str, Any]]] = {}
        self.visited_supply_ids = set()
        self.visited_door_ids = set()
        self.bad_door_ids = set()
        self.room_stack: List[Dict[str, Any]] = []
        self.active_supply: Optional[Dict[str, Any]] = None
        self.active_door: Optional[Dict[str, Any]] = None
        self.item_approach_steps = 0
        self.door_approach_steps = 0
        self.door_entry_start_location: Optional[Tuple[int, int]] = None
        self.rooms_searched = 0
        self.search_complete = False
        self.auto_forward = False
        self.history_locations = []
        self.supplies = []
        self.doors = []
        self.player_yaw = 0.0
        self.pubg_room_search_adapter = None
        self.embedded_room_search_adapter = None
        self.pubg_room_search_attempted = False
        if self.qwen_room_search_agent is not None:
            self.qwen_room_search_agent.reset()

    def process(self, w: 'FrameWorker'):
        location = self._get_location(w)
        direction = self._get_scalar(w.get_info("direction"))
        house_scene = self._get_house_scene(w)

        if location is None:
            print("[HouseLoot] 当前位置无效，小步移动刷新坐标")
            w.tap_single("摇杆", y_bias=-180, dura=180, wait=350)
            w.refresh_frame()
            return

        if self._try_qwen_room_search(w):
            return

        if house_scene == self.HOUSE_ROOFTOP:
            print("[HouseLoot] 检测到屋顶，停止一层搜房并切回跑图")
            self.reset()
            w.change_stage("跑图阶段")
            return

        if house_scene == self.HOUSE_OUTDOOR:
            print("[HouseLoot] 当前已在屋外，搜房结束，切回跑图")
            self.reset()
            w.change_stage("跑图阶段")
            return

        if house_scene != self.HOUSE_INDOOR:
            print(f"[HouseLoot] house_scene={house_scene}，等待进入屋内")
            return

        if self._try_pubg_room_search(w, source="indoor_process"):
            return

        if self.current_room_id is None:
            self._enter_room(location)

        if self.rooms_searched >= self.MAX_ROOMS_PER_FLOOR:
            print("[HouseLoot] 已达到单层最大房间数，结束搜房")
            self._finish_house_search(w)
            return

        if self.status == self.STATUS_SCAN_ROOM:
            self._scan_current_room(w, location)
            self.status = self.STATUS_LOOT_ITEM
            return

        if self.status == self.STATUS_LOOT_ITEM:
            if self._process_loot_item(w, location, direction):
                return
            self.status = self.STATUS_SELECT_DOOR
            return

        if self.status == self.STATUS_SELECT_DOOR:
            self._select_next_door_or_finish(w, location)
            return

        if self.status == self.STATUS_APPROACH_DOOR:
            self._process_door_approach(w, location, direction)
            return

        if self.status == self.STATUS_ENTER_NEXT_ROOM:
            self._confirm_enter_next_room(w, location)
            return

        if self.status == self.STATUS_FINISHED:
            self._finish_house_search(w)
            return

    def _try_qwen_room_search(self, w: 'FrameWorker') -> bool:
        agent = self.qwen_room_search_agent
        if agent is None:
            agent = QwenRoomSearchAgent.from_config(self)
            self.qwen_room_search_agent = agent
        if agent is None:
            return False
        return agent.process(w)

    def _enter_room(self, location: Tuple[int, int]):
        room_id = self._make_room_id(location)
        if room_id not in self.visited_rooms:
            self.rooms_searched += 1
        self.current_room_id = room_id
        self.current_room_location = location
        self.visited_rooms.add(room_id)
        self.room_loot_counts.setdefault(room_id, 0)
        self.room_supplies.setdefault(room_id, [])
        self.room_doors.setdefault(room_id, [])
        self.history_locations = []
        print(f"[HouseLoot] 进入房间 room={room_id}, loc={location}, rooms={self.rooms_searched}")

    def _scan_current_room(self, w: 'FrameWorker', location: Tuple[int, int]):
        room_id = self.current_room_id or self._make_room_id(location)
        self.current_room_id = room_id
        self.current_room_location = location
        self.room_supplies[room_id] = []
        self.room_doors[room_id] = []
        self.supplies = []
        self.doors = []

        print(f"[HouseLoot] 开始扫描当前房间 room={room_id}")
        for step in range(self.ROOM_SCAN_STEPS):
            w.refresh_frame()
            direction = self._get_scalar(w.get_info("direction"))
            scene = self._get_forward_scene(w)
            if direction is not None:
                self._collect_room_targets(room_id, scene, direction)
            if step < self.ROOM_SCAN_STEPS - 1:
                w.tap_single("视角", x_bias=self.ROOM_SCAN_TURN_BIAS, dura=800, wait=500)
                self.update_yaw(self.ROOM_SCAN_TURN_DEGREES)

        supplies = self.room_supplies.get(room_id, [])
        doors = self.room_doors.get(room_id, [])
        print(
            f"[HouseLoot] 房间扫描完成 room={room_id}, "
            f"supplies={len(supplies)}, doors={len(doors)}"
        )

    def _collect_room_targets(self, room_id, scene: Sequence[Sequence[float]], direction: float):
        for det in scene:
            if not self._valid_detection(det):
                continue

            class_id = int(det[5])
            if class_id in self.SUPPLY_CLASS_IDS:
                target = self._make_target(room_id, det, direction, "supply")
                if self._is_new_target(self.room_supplies[room_id], target):
                    self.room_supplies[room_id].append(target)
                    self.supplies.append((target["abs_angle"], target["box_h"]))
                    print(
                        f"[HouseLoot] 记录物资 angle={target['abs_angle']:.1f}, "
                        f"box_h={target['box_h']:.1f}, id={target['id']}"
                    )
            elif class_id in self.DOOR_CLASS_IDS:
                target = self._make_target(room_id, det, direction, "door")
                if self._is_new_target(self.room_doors[room_id], target):
                    self.room_doors[room_id].append(target)
                    self.doors.append((target["abs_angle"], target["box_h"]))
                    print(
                        f"[HouseLoot] 记录室内门 angle={target['abs_angle']:.1f}, "
                        f"box_h={target['box_h']:.1f}, id={target['id']}"
                    )

    def _process_loot_item(
        self,
        w: 'FrameWorker',
        location: Tuple[int, int],
        direction: Optional[float],
    ) -> bool:
        room_id = self.current_room_id
        if room_id is None:
            return False

        if self.room_loot_counts.get(room_id, 0) >= self.ROOM_LOOT_LIMIT:
            print(f"[HouseLoot] 当前房间已搜够 {self.ROOM_LOOT_LIMIT} 个物资")
            self.active_supply = None
            return False

        if self.active_supply is None:
            self.active_supply = self._select_next_supply(room_id)
            self.item_approach_steps = 0
            if self.active_supply is None:
                print("[HouseLoot] 当前房间没有可继续拾取的物资")
                return False

        if self._click_pickup_if_available(w):
            self._mark_supply_done(room_id, self.active_supply)
            return True

        if direction is None:
            print("[HouseLoot] 当前朝向无效，等待下一帧继续拾取")
            return True

        target = self._find_matching_target(w, self.active_supply, self.SUPPLY_CLASS_IDS)
        if target is None:
            if self.item_approach_steps == 0:
                self._align_direction_blocking(w, direction, self.active_supply["abs_angle"], tolerance=8)
                self.item_approach_steps += 1
                return True

            print("[HouseLoot] 物资目标丢失，跳过当前物资")
            self.visited_supply_ids.add(self.active_supply["id"])
            self.active_supply = None
            self.item_approach_steps = 0
            return True

        aligned = self._align_to_target(w, target, tolerance_px=self.CENTER_TOLERANCE_PX)
        w.refresh_frame()
        if not aligned:
            return True

        if self._click_pickup_if_available(w):
            self._mark_supply_done(room_id, self.active_supply)
            return True

        if self.item_approach_steps >= self.ITEM_APPROACH_MAX_STEPS:
            print("[HouseLoot] 靠近物资多次仍未拾取，跳过")
            self.visited_supply_ids.add(self.active_supply["id"])
            self.active_supply = None
            self.item_approach_steps = 0
            return True

        print(
            f"[HouseLoot] 靠近物资 step={self.item_approach_steps + 1}/"
            f"{self.ITEM_APPROACH_MAX_STEPS}"
        )
        self._move_forward(w, dura=260, wait=420)
        self.item_approach_steps += 1
        return True

    def _select_next_supply(self, room_id) -> Optional[Dict[str, Any]]:
        supplies = [
            target for target in self.room_supplies.get(room_id, [])
            if target["id"] not in self.visited_supply_ids
        ]
        if not supplies:
            return None

        supplies.sort(
            key=lambda target: (
                -target["area"],
                abs(target["center_x"] - self._frame_width() / 2.0),
                -target["center_y"],
            )
        )
        target = supplies[0]
        print(
            f"[HouseLoot] 选择物资 angle={target['abs_angle']:.1f}, "
            f"box_h={target['box_h']:.1f}"
        )
        return target

    def _mark_supply_done(self, room_id, target: Dict[str, Any]):
        self.visited_supply_ids.add(target["id"])
        self.room_loot_counts[room_id] = self.room_loot_counts.get(room_id, 0) + 1
        print(
            f"[HouseLoot] 完成拾取 room={room_id}, "
            f"count={self.room_loot_counts[room_id]}/{self.ROOM_LOOT_LIMIT}"
        )
        self.active_supply = None
        self.item_approach_steps = 0

    def _select_next_door_or_finish(self, w: 'FrameWorker', location: Tuple[int, int]):
        room_id = self.current_room_id
        if room_id is None:
            self.status = self.STATUS_SCAN_ROOM
            return

        door = self._select_next_door(room_id)
        if door is not None:
            self.active_door = door
            self.door_approach_steps = 0
            self.status = self.STATUS_APPROACH_DOOR
            print(f"[HouseLoot] 选择未探索门 angle={door['abs_angle']:.1f}, id={door['id']}")
            return

        if self.room_stack:
            backtrack = self.room_stack.pop()
            self.active_door = {
                "id": f"backtrack:{room_id}:{len(self.room_stack)}",
                "room_id": room_id,
                "kind": "backtrack",
                "abs_angle": backtrack["return_angle"],
                "target_room_id": backtrack["room_id"],
                "box_h": 0.0,
                "area": 0.0,
                "center_x": self._frame_width() / 2.0,
                "center_y": 0.0,
            }
            self.door_approach_steps = 0
            self.status = self.STATUS_APPROACH_DOOR
            print(
                f"[HouseLoot] 当前房间无新门，回退到上个房间 "
                f"target_room={backtrack['room_id']}, angle={backtrack['return_angle']:.1f}"
            )
            return

        print("[HouseLoot] 当前楼层没有可继续探索的门，搜房完成")
        self.status = self.STATUS_FINISHED

    def _select_next_door(self, room_id) -> Optional[Dict[str, Any]]:
        doors = [
            target for target in self.room_doors.get(room_id, [])
            if target["id"] not in self.visited_door_ids
            and target["id"] not in self.bad_door_ids
        ]
        if not doors:
            return None

        if self.room_stack:
            return_angle = self.room_stack[-1]["return_angle"]
            forward_doors = [
                target for target in doors
                if self._angle_diff(target["abs_angle"], return_angle) > 25.0
            ]
            if forward_doors:
                doors = forward_doors
            else:
                return None

        doors.sort(
            key=lambda target: (
                -target["area"],
                abs(target["center_x"] - self._frame_width() / 2.0),
            )
        )
        return doors[0]

    def _process_door_approach(
        self,
        w: 'FrameWorker',
        location: Tuple[int, int],
        direction: Optional[float],
    ):
        if self.active_door is None:
            self.status = self.STATUS_SELECT_DOOR
            return

        if direction is None:
            print("[HouseLoot] 当前朝向无效，等待下一帧继续靠门")
            return

        if self.active_door.get("kind") == "backtrack":
            self._align_direction_blocking(w, direction, self.active_door["abs_angle"], tolerance=8)
        else:
            target = self._find_matching_target(w, self.active_door, self.DOOR_CLASS_IDS)
            if target is not None:
                aligned = self._align_to_target(w, target, tolerance_px=self.CENTER_TOLERANCE_PX)
                w.refresh_frame()
                if not aligned:
                    return
            else:
                self._align_direction_blocking(w, direction, self.active_door["abs_angle"], tolerance=8)

        w.refresh_frame()
        if w.get_info("开门"):
            print("[HouseLoot] 检测到开门按钮，先开门")
            w.click("开门")
            time.sleep(0.7)
            w.refresh_frame()
        elif w.get_info("关门"):
            print("[HouseLoot] 门已经打开，准备进入")

        if self.door_approach_steps >= self.DOOR_APPROACH_MAX_STEPS:
            print("[HouseLoot] 靠近门多次无进展，标记该门不可用")
            if self.active_door.get("kind") != "backtrack":
                self.bad_door_ids.add(self.active_door["id"])
            self.active_door = None
            self.status = self.STATUS_SELECT_DOOR
            return

        if self.door_entry_start_location is None:
            self.door_entry_start_location = location

        print(
            f"[HouseLoot] 向门内推进 step={self.door_approach_steps + 1}/"
            f"{self.DOOR_APPROACH_MAX_STEPS}"
        )
        self._move_forward(w, dura=340, wait=650)
        self.door_approach_steps += 1
        self.status = self.STATUS_ENTER_NEXT_ROOM

    def _confirm_enter_next_room(self, w: 'FrameWorker', location: Tuple[int, int]):
        house_scene = self._get_house_scene(w)
        if self.active_door is None:
            self.status = self.STATUS_SELECT_DOOR
            return

        if house_scene == self.HOUSE_OUTDOOR:
            print("[HouseLoot] 通过该门到了屋外，停止搜房并切跑图")
            self._finish_house_search(w)
            return

        if house_scene == self.HOUSE_ROOFTOP:
            print("[HouseLoot] 通过该门到了屋顶，标记该门不可走")
            if self.active_door.get("kind") != "backtrack":
                self.bad_door_ids.add(self.active_door["id"])
            self._move_backward(w)
            self.active_door = None
            self.door_entry_start_location = None
            self.status = self.STATUS_SELECT_DOOR
            return

        if house_scene != self.HOUSE_INDOOR:
            print("[HouseLoot] 过门后室内状态不稳定，继续确认")
            return

        start_location = self.door_entry_start_location or location
        moved = get_distance(start_location, location)
        new_room_id = self._make_room_id(location)
        if moved < self.ROOM_TRANSITION_DISTANCE and new_room_id == self.current_room_id:
            self.status = self.STATUS_APPROACH_DOOR
            return

        if self.active_door.get("kind") == "backtrack":
            target_room_id = self.active_door.get("target_room_id")
            self.current_room_id = target_room_id or new_room_id
            self.current_room_location = location
            self.active_door = None
            self.door_entry_start_location = None
            self.status = self.STATUS_SCAN_ROOM
            print(f"[HouseLoot] 已回退到上一房间 room={self.current_room_id}")
            return

        current_room_id = self.current_room_id
        self.visited_door_ids.add(self.active_door["id"])
        return_angle = (self.active_door["abs_angle"] + 180.0) % 360.0
        self.room_stack.append({"room_id": current_room_id, "return_angle": return_angle})
        self.active_door = None
        self.door_entry_start_location = None
        self._enter_room(location)
        self.status = self.STATUS_SCAN_ROOM

    def _finish_house_search(self, w: 'FrameWorker'):
        print("[HouseLoot] 一层搜房完成，切回跑图阶段")
        self.stop_auto_forward(w)
        self.reset()
        w.change_stage("跑图阶段")

    def _click_pickup_if_available(self, w: 'FrameWorker') -> bool:
        if self._find_largest_target(self._get_forward_scene(w), self.PICK_MENU_CLASS_IDS):
            print(
                f"[HouseLoot] 检测到 pick_menu，已到物资跟前，"
                f"等待自动拾取 {self.PICK_MENU_AUTO_PICKUP_WAIT:.1f}s"
            )
            self.stop_auto_forward(w)
            time.sleep(self.PICK_MENU_AUTO_PICKUP_WAIT)
            w.refresh_frame()
            return True

        pickup = w.get_info("拾取首个物资")
        if not pickup:
            return False

        print("[HouseLoot] 检测到拾取按钮，点击拾取")
        w.click(pickup if not isinstance(pickup, bool) else "拾取首个物资")
        time.sleep(self.PICK_MENU_AUTO_PICKUP_WAIT)
        w.refresh_frame()
        return True

    def _find_matching_target(
        self,
        w: 'FrameWorker',
        expected: Dict[str, Any],
        class_ids: set,
    ) -> Optional[Sequence[float]]:
        direction = self._get_scalar(w.get_info("direction"))
        if direction is None:
            return self._find_largest_target(self._get_forward_scene(w), class_ids)

        candidates = []
        for det in self._get_forward_scene(w):
            if not self._valid_detection(det) or int(det[5]) not in class_ids:
                continue
            abs_angle = self._detection_abs_angle(det, direction)
            diff = self._angle_diff(abs_angle, expected["abs_angle"])
            if diff <= self.TARGET_MATCH_ANGLE:
                candidates.append((diff, self._detection_area(det), det))

        if candidates:
            candidates.sort(key=lambda item: (item[0], -item[1]))
            return candidates[0][2]
        return self._find_largest_target(self._get_forward_scene(w), class_ids)

    def _find_largest_target(
        self,
        detections: Sequence[Sequence[float]],
        class_ids: set,
    ) -> Optional[Sequence[float]]:
        candidates = [
            obj for obj in detections
            if self._valid_detection(obj) and int(obj[5]) in class_ids
        ]
        if not candidates:
            return None
        return max(candidates, key=self._detection_area)

    def _align_to_target(self, w: 'FrameWorker', target: Sequence[float], tolerance_px=80) -> bool:
        frame_w = self._frame_width()
        center_x = (float(target[0]) + float(target[2])) / 2.0
        offset = center_x - (frame_w / 2.0)
        if abs(offset) <= tolerance_px:
            return True

        bias = int(max(-400, min(400, offset * 0.33)))
        print(f"[HouseLoot] 视觉对齐目标 offset={offset:.1f}, bias={bias}")
        w.tap_single("视角", x_bias=bias, dura=500, wait=500)
        return False

    def _align_direction_blocking(self, w, current_dir, target_angle, tolerance=5):
        current_dir = self._get_scalar(current_dir)
        if current_dir is None:
            return False
        for _ in range(8):
            turn_dir, px, diff = calculate_move_count(current_dir, target_angle)
            if diff <= tolerance:
                return True
            x_bias = px if turn_dir == 'right' else -px
            w.tap_single('视角', x_bias=int(x_bias), dura=700, wait=450)
            w.refresh_frame()
            current_dir = self._get_scalar(w.get_info('direction'))
            if current_dir is None:
                return False
        return False

    def _move_forward(self, w: 'FrameWorker', dura=300, wait=500):
        self.stop_auto_forward(w)
        w.tap_single("摇杆", y_bias=-300, dura=dura, wait=wait)
        w.refresh_frame()

    def _move_backward(self, w: 'FrameWorker'):
        self.stop_auto_forward(w)
        w.tap_single("摇杆", y_bias=300, dura=320, wait=600)
        w.refresh_frame()

    def _get_house_scene(self, w: 'FrameWorker') -> Optional[int]:
        value = w.get_info("house_scene")
        if value is None:
            value = w.get_info("hosue_scene")
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_location(self, w: 'FrameWorker') -> Optional[Tuple[int, int]]:
        info = w.get_info("location")
        if info is None:
            return None
        if isinstance(info, (list, tuple)):
            if len(info) >= 2 and not isinstance(info[0], (list, tuple)):
                return check_location(info)
            if len(info) > 0:
                return check_location(info[0])
        return None

    def _get_forward_scene(self, w: 'FrameWorker') -> List[Sequence[float]]:
        scene = w.get_info("forward_scene")
        if not scene or isinstance(scene, bool):
            return []
        return list(scene)

    def _get_scalar(self, value) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, (int, float)):
                return float(first)
        return None

    def _make_room_id(self, location: Tuple[int, int]):
        return (
            int(round(float(location[0]) / self.ROOM_CLUSTER_SIZE)),
            int(round(float(location[1]) / self.ROOM_CLUSTER_SIZE)),
        )

    def _make_target(self, room_id, det: Sequence[float], direction: float, kind: str) -> Dict[str, Any]:
        abs_angle = self._detection_abs_angle(det, direction)
        box_h = max(0.0, float(det[3]) - float(det[1]))
        center_x = (float(det[0]) + float(det[2])) / 2.0
        center_y = (float(det[1]) + float(det[3])) / 2.0
        angle_bucket = int(round(abs_angle / self.TARGET_DEDUP_ANGLE))
        height_bucket = int(round(box_h / self.TARGET_DEDUP_BOX_H))
        return {
            "id": f"{kind}:{room_id}:{angle_bucket}:{height_bucket}:{int(det[5])}",
            "room_id": room_id,
            "kind": kind,
            "abs_angle": abs_angle,
            "box_h": box_h,
            "area": self._detection_area(det),
            "center_x": center_x,
            "center_y": center_y,
            "class_id": int(det[5]),
        }

    def _is_new_target(self, targets: List[Dict[str, Any]], target: Dict[str, Any]) -> bool:
        for item in targets:
            if self._angle_diff(item["abs_angle"], target["abs_angle"]) <= self.TARGET_DEDUP_ANGLE:
                if abs(item["box_h"] - target["box_h"]) <= self.TARGET_DEDUP_BOX_H:
                    return False
        return True

    def _valid_detection(self, det: Sequence[float]) -> bool:
        if not isinstance(det, (list, tuple)) or len(det) < 6:
            return False
        try:
            return float(det[2]) > float(det[0]) and float(det[3]) > float(det[1])
        except (TypeError, ValueError):
            return False

    def _detection_abs_angle(self, det: Sequence[float], direction: float) -> float:
        center_x = (float(det[0]) + float(det[2])) / 2.0
        rel_angle = self.pixel_to_angle(center_x)
        return (float(direction) + rel_angle) % 360.0

    def _detection_area(self, det: Sequence[float]) -> float:
        return max(0.0, float(det[2]) - float(det[0])) * max(0.0, float(det[3]) - float(det[1]))

    def _frame_width(self) -> float:
        try:
            inf_w, inf_h = get_wh()
            return float(max(inf_w, inf_h))
        except Exception:
            return float(self.screen_w)

    def _angle_diff(self, a: float, b: float) -> float:
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def searching_logic(self, w: 'FrameWorker', current_loc, current_direction):

        if self.searching_number == 5:
            print('已经搜满5个房间，切换到跑图阶段')
            self.searching_number = 0
            w.change_stage('跑图阶段')
            return

        # --- 智能选点 ---
        if self.current_house_id is None:
            self.select_smart_target(current_loc, current_direction)
            if not self.current_house_id:
                print("[Searching] 当前区域无合适目标或已搜完")
                return
            self.status = "FAST_NAV"
            print(f"[Searching] 锁定目标: {self.current_house_id} | 状态: 快速导航")
            self.history_locations = []  # 切换目标时清空历史

        target_loc = self.active_entry['location']
        dist = get_distance(current_loc, target_loc)

        # --- 快速前进 (距离 > 5.0) ---
        if self.status == "FAST_NAV":
            # 卡顿检测逻辑
            if self.update_and_check_stuck(current_loc):
                print("[Nav] 检测到人物卡死，启动避障程序...")
                self.execute_unstuck_logic(w, current_loc)
                self.history_locations = []
                return

            if dist <= 5.0:
                print(f"[Nav] 进入精细导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.align_direction(w, target_loc)

            if not self.auto_forward:
                w.click('自动前进')
                self.auto_forward = True

            self.handle_jump_logic(w)

        # --- 精细逼近 ---
        elif self.status == "PRECISE_NAV":
            # --- [修改 1] 在精细导航阶段加入卡顿检测 ---
            # 原因：即使在慢速移动时，也可能卡在树根或小障碍物上
            if self.update_and_check_stuck(current_loc):
                print("[Nav] (Precise) 检测到人物卡死，启动避障程序...")
                self.execute_unstuck_logic(w, current_loc)
                self.history_locations = []  # 清空历史，防止重复触发
                return
            # ----------------------------------------

            if dist <= 1:
                print(f"[Nav] 已到达进门点 (距离 {dist:.2f})")
                self.status = "SCANNING"
                return

            self.stop_auto_forward(w)
            self.align_direction(w, target_loc)
            press_duration = get_time_from_distance(dist)
            w.tap_single('摇杆', y_bias=-300, dura=300, wait=press_duration - 300)
            w.refresh_frame()
            self.handle_jump_logic(w)

        # --- 进门点扫描 ---
        elif self.status == "SCANNING":
            print("[Scan] 到达点位，开始门检测...")
            ideal_angle = self.active_entry['direction']
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if self._try_pubg_room_search(
                w,
                source="door_front",
                enter_after_refine=False,
                finish_to_next_target=True,
            ):
                return

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
            while True:
                door = self.find_largest_door(w)
                if not door:
                    print("[Visual] 丢失目标，重新扫描")
                    self.status = "INTERACT"
                    break

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

        # --- 交互逻辑 ---
        elif self.status == "INTERACT":
            print(f"[Interact] 尝试在 {self.current_house_id} 寻找交互按钮...")
            success = False
            for i in range(10):
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
                    w.click('关门')
                    time.sleep(1.2)
                    w.refresh_frame()
                    if w.get_info('开门'):
                        w.click('开门')
                        time.sleep(0.5)
                    success = True
                    break
                w.tap_single('摇杆', y_bias=-300, dura=300, wait=200)

            if success:
                print("[Interact] 交互成功，准备入户")
                self.status = "FINAL_ENTRY"
            else:
                print(f"[Interact] 警告：交互失败，舍弃进门点")
                ideal_angle = self.active_entry['direction']
                self.handle_failed_entry_logic(ideal_angle)
                self.status = "IDLE"
                return

        # --- 最终入户 ---
        elif self.status == "FINAL_ENTRY":
            ideal_angle = self.active_entry['direction']
            print(f"[Entry] 调整至进门角度: {ideal_angle}")
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)
            print("[Entry] 进门")
            w.tap_single('摇杆', y_bias=-300, dura=300, wait=1000)
            # w.tap_single('摇杆', y_bias=-500, dura=300, wait=1000)
            # time.sleep(2)
            # if w.get_info('关门') is None:
            #     # 当前可能没进入房屋成功
            #     print(f"[Entry] 第一次进门未成功，左移后重新进门")
            #     w.tap_single('摇杆', x_bias=-45, dura=300, wait=100)
            #     # 左移后重新进入房屋
            #     w.tap_single('摇杆', y_bias=-300, dura=300, wait=100)

            if self._try_pubg_room_search(
                w,
                source="final_entry",
                enter_after_refine=False,
                finish_to_next_target=True,
            ):
                return
            self.start_searching(w)
            self.completed_houses.add(self.current_house_id)
            print(f"[Finish] 房屋 {self.current_house_id} 完成")
            w.refresh_frame()
            exit_direction = w.get_info('direction')
            self.prepare_next_target_logic(exit_direction)
            self.current_house_id = None
            self.status = "IDLE"

    def _try_pubg_room_search(
        self,
        w: 'FrameWorker',
        source: str,
        *,
        enter_after_refine: Optional[bool] = None,
        finish_to_next_target: bool = False,
    ) -> bool:
        if self.pubg_room_search_attempted:
            return False

        embedded_result = self._try_embedded_room_search(
            w,
            source=source,
            enter_after_refine=enter_after_refine,
        )
        if embedded_result is not None:
            self.pubg_room_search_attempted = True
            if embedded_result.ok:
                self._finish_pubg_room_search_success(
                    w,
                    source=source,
                    finish_to_next_target=finish_to_next_target,
                )
                return True
            if not embedded_result.fallback_to_legacy:
                self._finish_pubg_room_search_terminal(w, source=source)
                return True
            print("[PubgRoomSearch] 内嵌核心回退旧搜房逻辑")
            self.pubg_room_search_attempted = False
            return False

        self.pubg_room_search_attempted = True
        adapter = self.pubg_room_search_adapter
        if adapter is None:
            adapter = HouseSearchAdapter.from_config(w)
            self.pubg_room_search_adapter = adapter
        if adapter is None:
            self.pubg_room_search_attempted = False
            return False

        print(f"[PubgRoomSearch] 开始接管搜房: source={source}")
        result = adapter.search_current_house()
        extra = f", reason={result.reason}" if result.reason else ""
        print(
            f"[PubgRoomSearch] 搜房返回: result={result.result_name}, "
            f"ok={result.ok}, fallback={result.fallback_to_legacy}{extra}"
        )

        if result.ok:
            self._finish_pubg_room_search_success(
                w,
                source=source,
                finish_to_next_target=finish_to_next_target,
                finish_stage=adapter.config.get("finish_stage") or "跑图阶段",
            )
            return True

        if result.fallback_to_legacy:
            print("[PubgRoomSearch] 切回旧搜房逻辑兜底")
            return False

        self._finish_pubg_room_search_terminal(
            w,
            source=source,
            finish_stage=adapter.config.get("finish_stage") or "跑图阶段",
        )
        return True

    def _try_embedded_room_search(
        self,
        w: 'FrameWorker',
        *,
        source: str,
        enter_after_refine: Optional[bool],
    ):
        if source != "door_front":
            return None

        adapter = self.embedded_room_search_adapter
        if adapter is None:
            adapter = EmbeddedHouseSearchAdapter.from_config(w)
            self.embedded_room_search_adapter = adapter
        if adapter is None:
            return None
        if not adapter.config.get("embedded_first", True):
            return None

        print(f"[EmbeddedRoomSearch] 开始内嵌接管搜房: source={source}")
        result = adapter.search_from_door_front(
            source=source,
            enter_after_refine=enter_after_refine,
        )
        extra = f", reason={result.reason}" if result.reason else ""
        print(
            f"[EmbeddedRoomSearch] 搜房返回: result={result.result_name}, "
            f"ok={result.ok}, fallback={result.fallback_to_legacy}{extra}"
        )
        return result

    def _finish_pubg_room_search_success(
        self,
        w: 'FrameWorker',
        *,
        source: str,
        finish_to_next_target: bool,
        finish_stage: str = "跑图阶段",
    ):
        self.stop_auto_forward(w)
        if finish_to_next_target and self.current_house_id is not None:
            house_id = self.current_house_id
            self.completed_houses.add(house_id)
            w.refresh_frame()
            exit_direction = w.get_info('direction')
            self.prepare_next_target_logic(exit_direction)
            self.current_house_id = None
            self.active_entry = None
            self.reset()
            self.status = "IDLE"
            print(f"[EmbeddedRoomSearch] 房屋 {house_id} 已完成，准备寻找下一个入口")
            return

        self.reset()
        w.change_stage(finish_stage)

    def _finish_pubg_room_search_terminal(
        self,
        w: 'FrameWorker',
        *,
        source: str,
        finish_stage: str = "跑图阶段",
    ):
        self.stop_auto_forward(w)
        self.reset()
        w.change_stage(finish_stage)

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

    def execute_unstuck_logic(self, w: 'FrameWorker', current_loc):
        self.stop_auto_forward(w)
        if w.get_info('跳跃'):
            print("[Unstuck] 尝试跳跃脱困")
            self.handle_jump_logic(w)
            w.tap_single('摇杆', y_bias=-300, dura=500, wait=1000)
            w.refresh_frame()
            new_loc = check_location(w.get_info('location')[0])
            if new_loc and get_distance(current_loc, new_loc) > self.stuck_threshold:
                print("[Unstuck] 跳跃脱困成功")
                return

        print("[Unstuck] 跳跃无效，进入 U 型避障移动...")
        while True:
            print("[Unstuck] 后退...")
            w.tap_single('摇杆', y_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_back = check_location(w.get_info('location')[0])
            if not loc_after_back: continue

            print("[Unstuck] 右移试探...")
            w.tap_single('摇杆', x_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_right = check_location(w.get_info('location')[0])

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
                loc_after_left = check_location(w.get_info('location')[0])

                if loc_after_left and get_distance(loc_after_right, loc_after_left) > 0.5:
                    print("[Unstuck] 左侧可通行")
                    side_way_clear = True
                    last_valid_loc = loc_after_left

            if not side_way_clear:
                print("[Unstuck] 左右均受阻 (U型死角)，再次后退...")
                continue

            print("[Unstuck] 尝试向前突破...")
            while True:
                w.tap_single('摇杆', y_bias=-300, dura=300, wait=2000)
                w.refresh_frame()
                loc_after_forward = check_location(w.get_info('location')[0])

                if loc_after_forward and get_distance(last_valid_loc, loc_after_forward) > 0.5:
                    print("[Unstuck] 脱困成功！")
                    return
                else:
                    print("[Unstuck] 前方依然受阻，继续侧向移动...")
                    moved_side = False
                    for bias in [300, -300]:
                        w.tap_single('摇杆', x_bias=bias, dura=300, wait=1500)
                        w.refresh_frame()
                        temp_loc = check_location(w.get_info('location')[0])
                        if temp_loc and get_distance(loc_after_forward, temp_loc) > 0.5:
                            last_valid_loc = temp_loc
                            moved_side = True
                            break

                    if not moved_side:
                        print("[Unstuck] 前方死路，重新执行后退逻辑")
                        break

    def handle_jump_logic(self, w: 'FrameWorker'):
        if w.get_info('跳跃'):
            print("[Jump] 检测到障碍，执行跳跃")
            self.stop_auto_forward(w)
            w.click('跳跃')
            time.sleep(0.2)
            w.tap_single('摇杆', y_bias=-400, dura=600)
            w.refresh_frame()

    def select_smart_target(self, current_loc, current_direction):
        best_dist = float('inf')
        best_id = None
        best_entry = None
        avoid_angle = getattr(self, 'avoid_angle_ref', None)
        avoid_mode = getattr(self, 'avoid_mode', None)

        for house_id, entries in self.house_data.items():
            if house_id in self.completed_houses: continue
            if house_id in self.temp_skip_houses: continue

            for entry in entries:
                dist = get_distance(current_loc, entry['location'])
                if avoid_angle is not None:
                    angle_to_target = calculate_angle(current_loc, entry['location'])
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

    def handle_failed_entry_logic(self, failed_entry_angle):
        print(f"[Smart] 进门失败，临时跳过 {self.current_house_id}")
        self.temp_skip_houses.add(self.current_house_id)
        self.current_house_id = None
        self.avoid_angle_ref = failed_entry_angle
        self.avoid_mode = 'SAME'

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

    def align_direction_blocking(self, w, current_dir, target_angle, tolerance=5):
        current_dir = self._get_scalar(current_dir)
        if current_dir is None:
            return False
        for _ in range(10):
            turn_dir, px, diff = calculate_move_count(current_dir, target_angle)
            if diff <= tolerance:
                return True
            x_bias = px if turn_dir == 'right' else - px
            w.tap_single('视角', x_bias=int(x_bias), dura=800, wait=500)
            w.refresh_frame()
            current_dir = self._get_scalar(w.get_info('direction'))
            if current_dir is None:
                return False
        return False

    def align_direction(self, w, tar_loc, threshold=5):
        while True:
            cur_loc = w.get_info('location')[0]
            cur_dir = w.get_info('direction')
            target_angle = calculate_angle(cur_loc, tar_loc)
            turn_dir, px, diff = calculate_move_count(cur_dir, target_angle)
            if abs(diff) <= threshold: break
            move_px = px if turn_dir == 'right' else -px
            w.tap_single('视角', x_bias=move_px, dura=800, wait=500)
            w.refresh_frame()

    def find_largest_door(self, w):
        """
          0: house
          1: door
          2: window
          3: open_door
          4: door_frame
        """
        scene = w.get_info('forward_scene')
        if not scene: return None
        doors = [obj for obj in scene if int(obj[5]) in [0]]
        if not doors: return None
        return max(doors, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def start_searching(self, w):

        # 旋转360度遍历当前房间，检查当前房间内是否有物资跟联通当前这个房间的门
        # 屏幕滑动430，角度大概旋转60度左右
        picked_points = []  # 当前待搜索的物资点
        not_indoors = []  # 当前进入房间搜索的门

        # 顺时针旋转视角
        time.sleep(2)

        for _ in range(6):
            w.tap_single('视角', x_bias=430, dura=800, wait=500)
            self.update_yaw(60)  # 更新绝对朝向
            time.sleep(1)
            # 获取当前画面中的门跟物资
            scene = w.get_info('forward_scene')
            picked_object = [obj for obj in scene if int(obj[5]) in [1]]
            in_door = [obj for obj in scene if int(obj[5]) in [4]]

            picked_points.extend(picked_object)
            not_indoors.extend(in_door)
            w.refresh_frame()

        time.sleep(5)

        # 获取当前画面中物资和门
        supplies_raw = self.get_targets_info(picked_points)  # [(rel_angle, box_h, det)]
        doors_raw = self.get_targets_info(not_indoors)

        print("[Searching] 搜寻物资。。。。。。。。。。。当前获房间取到物资点信息{}".format(supplies_raw))
        print("[Searching] 搜寻物资。。。。。。。。。。。当前获可进入其他房间门的信息{}".format(doors_raw))

        for rel_ang, box_h, _ in supplies_raw:
            abs_ang = (self.player_yaw + rel_ang) % 360
            if not self.same_target(self.supplies, abs_ang, box_h):
                self.supplies.append((abs_ang, box_h))
                print(f"  > 物资 {abs_ang:.1f}° 框高 {box_h}px")

        for rel_ang, box_h, _ in doors_raw:
            abs_ang = (self.player_yaw + rel_ang) % 360
            if not self.same_target(self.doors, abs_ang, box_h):
                self.doors.append((abs_ang, box_h))
                print(f"  > 门   {abs_ang:.1f}° 框高 {box_h}px")

        print(f"[扫描] 完成。物资: {len(self.supplies)}，门: {len(self.doors)}")

        # 当前房间内存在物资点，开始当前房间内物资点物资的拾取

        if self.supplies:
            supplies_sorted = sorted(self.supplies, key=lambda x: x[1], reverse=True)
            print(f"[物资] 共====== {len(supplies_sorted)} 个")
            for idx, (abs_ang, box_h) in enumerate(supplies_sorted, 1):
                while True:
                    print(f"==========物资{idx} {abs_ang:.1f}° 框高{box_h}px")
                    self.collect_item(w, abs_ang, box_h)

        else:
            print("[物资] 无")

        w.tap_single('视角', x_bias=430, dura=800, wait=500)

    def turn_to_absolute(self, w, target_abs: float):
        """让角色朝向指定的绝对方向，并更新 player_yaw"""
        delta = (target_abs - self.player_yaw + 180) % 360 - 180
        if abs(delta) < 0.3:
            return
        self.turn_by_angle(w, delta)
        self.update_yaw(delta)
        print(f"    [转向] 转动 {delta:.1f}°，当前 player_yaw = {self.player_yaw:.1f}°")

    def turn_by_angle(self, w, delta_angle: float, duration_ms: int = 200):
        """
        滑动右侧屏幕旋转视角，delta_angle > 0 右转，< 0 左转。
        """
        swipe_dist = delta_angle * 7.1
        # start_x = int(self.scr_w * 0.75)
        # start_y = int(self.scr_h * 0.5)
        # end_x = start_x + int(swipe_dist)
        # end_y = start_y
        # self._run_adb(f"adb shell input swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}")
        w.tap_single('视角', x_bias=int(swipe_dist), dura=800, wait=500)
        time.sleep(duration_ms / 800)
        w.refresh_frame()


    def collect_item(self, w, target_abs_angle: float, target_box_h: float):
        """锁定物资，闭环对准并靠近拾取"""
        print(f"    [搜集] 锁定物资 {target_abs_angle:.1f}°")
        self.current_target_abs = target_abs_angle  # 记录用于跟踪

        # 首次转向大致方向
        self.turn_to_absolute(w, target_abs_angle)


        for step in range(20):
            # 检查 UI 拾取按钮
            if w.get_info("拾取首个物资"):
                print("    出现可拾取物资的提示框信息")
                return

            # 重新检测并匹配当前物资，获取当前相对角度
            cur_rel_angle = self.find_target_relative_angle(w, target_box_h, 1)
            print("当前相对角度{}".format(cur_rel_angle))
            if cur_rel_angle is None:
                print("    [丢失] 未找到目标物资，放弃")
                return

            # 若偏差较大，微调朝向（闭环比例控制）
            if abs(cur_rel_angle) > 1.5:
                print(f"    [微调] 偏差 {cur_rel_angle:.1f}°")
                self.turn_by_angle(w, cur_rel_angle, duration_ms=150)
                self.update_yaw(cur_rel_angle)  # 更新 player_yaw
                time.sleep(0.15)
                continue

            # 对准后前进一小步
            w.tap_single('摇杆', y_bias=-400, dura=600)
            time.sleep(0.2)

        print("    -> 超时未拾取，放弃")

    def find_target_relative_angle(self, w, target_box_h, class_id):
        """
        在当前画面中寻找与 self.current_target_abs 匹配的物资，
        返回其相对角度（度），若未找到返回 None。
        """

        scene = w.get_info('forward_scene')
        detections = [obj for obj in scene if int(obj[5]) in [class_id]]
        print("调整人物转向过程中重新获取物资{}".format(detections))

        best = None
        best_diff = 999
        for det in detections:
            if det[5] not in [class_id]:
                continue
            cx = (det[0] + det[2]) / 2
            rel_ang = self.pixel_to_angle(cx)
            abs_ang = (self.player_yaw + rel_ang) % 360
            diff = abs((abs_ang - self.current_target_abs + 180) % 360 - 180)
            box_h = det[3] - det[1]
            # 角度差和框高差都要在阈值内
            if diff < 5 and abs(box_h - target_box_h) < 20:
                if diff < best_diff:
                    print("best_diff信息为{}".format(best_diff))
                    best_diff = diff
                    best = rel_ang
            print("调整人物转向过程中角度偏差{}以及宽高偏差{}".format(diff, abs(box_h - target_box_h)))

        return best

    def get_targets_info(self, targets):
        """
        从当前画面检测指定类别的目标，返回 [(rel_angle, box_height), ...]
        """
        info = []
        for target in targets:
            cx = (target[0] + target[2]) / 2
            bh = target[3] - target[1]
            rel_angle = self.pixel_to_angle(cx)
            info.append((rel_angle, bh, target))  # 保留原始框用于后续
        return info

    def pixel_to_angle(self, px: float) -> float:
        """像素水平坐标 -> 相对角度（度）"""
        frame_w = self._frame_width()
        center = frame_w / 2.0
        return (float(px) - center) / center * (80.0 / 2.0)

    def update_yaw(self, delta):
        """每次旋转后调用，更新绝对朝向"""
        self.player_yaw = (self.player_yaw + delta) % 360

    def same_target(self, target_list, abs_angle, box_h):
        """角度 & 框高双重去重"""
        for a, h in target_list:
            angle_diff = abs((abs_angle - a + 180) % 360 - 180)
            # TARGET_LOCK_ANGLE_THRESH 绝对方向匹配角度容差（度） 设置偏差5 TARGET_LOCK_BOX_THRESH = 20    # 框高匹配容差（像素）
            if angle_diff < 5 and abs(box_h - h) < 20:
                return True
        return False
