from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import (
    calculate_angle,
    calculate_move_count,
    get_distance,
    get_time_from_distance,
)


@dataclass
class QwenToolResult:
    ok: bool
    tool_name: str
    observation: Dict[str, Any]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QwenHouseSearchTools:
    """White-list tools for a Qwen-driven room-search agent.

    The model chooses one tool by name; this class validates the request and
    delegates to the existing deterministic room-search helpers.
    """

    TOOL_NAMES = (
        "get_game_state",
        "get_visible_objects",
        "select_next_house",
        "navigate_to_house_entry",
        "scan_entry_door",
        "approach_entry_door",
        "enter_house",
        "scan_room",
        "select_next_supply",
        "align_to_object",
        "move_to_object",
        "pickup_item",
        "select_next_door",
        "open_door",
        "enter_door",
        "check_stuck",
        "recover_from_stuck",
        "mark_house_done",
        "finish_house_search",
        "wait_and_refresh",
    )

    TOOL_SPECS = [
        {
            "name": "get_game_state",
            "description": "读取当前位置、朝向、屋内/屋外状态、搜房状态和房间记忆。",
            "args": {},
        },
        {
            "name": "get_visible_objects",
            "description": "读取 forward_scene 目标检测结果，归一成门、物资、拾取菜单等对象。",
            "args": {},
        },
        {
            "name": "select_next_house",
            "description": "从房屋入口数据里选择下一个未搜索房子。",
            "args": {},
        },
        {
            "name": "navigate_to_house_entry",
            "description": "低层闭环导航到当前房子的入户点；内部连续对齐、推进、刷新、避障，直到到达或失败后再返回。",
            "args": {},
        },
        {
            "name": "scan_entry_door",
            "description": "在房屋入户点按预设角度扫描并锁定可进入的门。",
            "args": {},
        },
        {
            "name": "approach_entry_door",
            "description": "视觉对齐入户门；对齐后可调用 enter_house。",
            "args": {},
        },
        {
            "name": "enter_house",
            "description": "执行开门/靠近/前推，直到进入室内或需要下一轮继续。",
            "args": {},
        },
        {
            "name": "scan_room",
            "description": "执行一轮确定性房间扫描，收集当前房间物资和门。",
            "args": {},
        },
        {
            "name": "select_next_supply",
            "description": "从当前房间记忆里选择下一个未拾取物资。",
            "args": {},
        },
        {
            "name": "align_to_object",
            "description": "对齐当前画面中的目标。target_type 可为 supply、door、pick_menu。",
            "args": {"target_type": "supply|door|pick_menu", "target_id": "optional"},
        },
        {
            "name": "move_to_object",
            "description": "先对齐目标，对齐后短前推靠近目标。",
            "args": {"target_type": "supply|door", "duration_ms": "optional"},
        },
        {
            "name": "pickup_item",
            "description": "点击拾取或等待 pick_menu 自动拾取，并标记当前物资完成。",
            "args": {},
        },
        {
            "name": "select_next_door",
            "description": "从当前房间记忆里选择下一个未探索门。",
            "args": {},
        },
        {
            "name": "open_door",
            "description": "如果出现开门/关门按钮，则执行门交互。",
            "args": {},
        },
        {
            "name": "enter_door",
            "description": "向当前门内短前推，并根据位置/house_scene 确认是否进入新房间。",
            "args": {"confirm": "optional bool"},
        },
        {
            "name": "check_stuck",
            "description": "根据位置历史判断人物是否卡住。",
            "args": {},
        },
        {
            "name": "recover_from_stuck",
            "description": "执行一次有界脱困动作：停止、跳跃/后退、侧移、刷新。",
            "args": {},
        },
        {
            "name": "mark_house_done",
            "description": "标记当前房子已完成，准备选择下一个房子。",
            "args": {"force": "optional bool"},
        },
        {
            "name": "finish_house_search",
            "description": "结束当前搜房任务并切回跑图阶段。",
            "args": {},
        },
        {
            "name": "wait_and_refresh",
            "description": "等待一小段时间并刷新当前帧。",
            "args": {"wait_sec": "optional float"},
        },
    ]

    def __init__(self, searcher: Any, worker: Any):
        self.searcher = searcher
        self.worker = worker
        self.config = getattr(searcher, "qwen_room_search_config", None) or {}

    @classmethod
    def tool_specs(cls) -> List[Dict[str, Any]]:
        return list(cls.TOOL_SPECS)

    def dispatch(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        if tool_name not in self.TOOL_NAMES:
            return self._with_feedback(
                self._result(False, tool_name, error=f"unknown tool: {tool_name}")
            ).to_dict()

        try:
            method = getattr(self, tool_name)
            if not isinstance(args, dict):
                raise ValueError("tool args must be a dict")
            result = method(**args)
            self._refresh_after_tool(tool_name)
            return self._with_feedback(result).to_dict()
        except Exception as exc:
            self._refresh_after_tool(tool_name)
            return self._with_feedback(self._result(False, tool_name, error=str(exc))).to_dict()

    def build_observation(self, task: str = "搜索当前房屋") -> Dict[str, Any]:
        state = self.get_game_state().observation
        objects = self.get_visible_objects().observation
        return {
            "task": task,
            "state": state,
            "visible_objects": objects.get("objects", []),
            "available_tools": self.tool_specs(),
        }

    def get_game_state(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        location = s._get_location(w)
        direction = s._get_scalar(w.get_info("direction"))
        house_scene = s._get_house_scene(w)
        room_id = s.current_room_id

        observation = {
            "location": self._json_point(location),
            "direction": direction,
            "house_scene": house_scene,
            "house_scene_name": self._house_scene_name(house_scene),
            "status": s.status,
            "current_room_id": self._json_key(room_id),
            "rooms_searched": s.rooms_searched,
            "max_rooms_per_floor": s.MAX_ROOMS_PER_FLOOR,
            "active_supply_id": self._target_id(s.active_supply),
            "active_door_id": self._target_id(s.active_door),
            "current_house_id": s.current_house_id,
            "active_entry": self._serialize_entry(s.active_entry),
            "completed_house_count": getattr(s, "searching_number", 0),
            "completed_house_ids": sorted(str(item) for item in s.completed_houses),
            "room_memory": self._room_memory(room_id),
            "interactions": self._interaction_state(),
        }
        return self._result(True, "get_game_state", observation)

    def get_visible_objects(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        direction = s._get_scalar(w.get_info("direction"))
        objects = [
            self._serialize_detection(det, direction)
            for det in s._get_forward_scene(w)
            if s._valid_detection(det)
        ]
        observation = {
            "objects": objects,
            "counts": {
                "supply": sum(1 for obj in objects if obj["type"] == "supply"),
                "door": sum(1 for obj in objects if obj["type"] == "door"),
                "pick_menu": sum(1 for obj in objects if obj["type"] == "pick_menu"),
            },
        }
        return self._result(True, "get_visible_objects", observation)

    def select_next_house(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        location = s._get_location(w)
        direction = s._get_scalar(w.get_info("direction"))
        if location is None:
            return self._result(False, "select_next_house", error="location unavailable")

        s.select_smart_target(location, direction)
        if s.current_house_id is None or s.active_entry is None:
            return self._result(
                False,
                "select_next_house",
                {"selected": None},
                error="no available house target",
            )

        s.status = "FAST_NAV"
        s.history_locations = []
        return self._result(
            True,
            "select_next_house",
            {
                "selected_house_id": s.current_house_id,
                "active_entry": self._serialize_entry(s.active_entry),
            },
        )

    def navigate_to_house_entry(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        location = s._get_location(w)
        if location is None:
            return self._result(False, "navigate_to_house_entry", error="location unavailable")
        if not s.active_entry:
            selected = self.select_next_house()
            if not selected.ok:
                return selected

        target_loc = tuple(s.active_entry["location"])
        start_location = location
        start_distance = get_distance(location, target_loc)
        arrival_distance = self._config_float("qwen_nav_arrival_distance", 1.0)
        precise_distance = self._config_float("qwen_nav_precise_distance", 5.0)
        max_steps = self._config_int("qwen_nav_max_steps", 8)
        timeout_sec = self._config_float("qwen_nav_timeout_sec", 12.0)
        no_progress_limit = self._config_int("qwen_nav_no_progress_limit", 2)

        if start_distance <= arrival_distance:
            s.stop_auto_forward(w)
            s.status = "SCANNING"
            return self._result(
                True,
                "navigate_to_house_entry",
                {
                    "at_entry": True,
                    "result_type": "arrived",
                    "start_location": self._json_point(start_location),
                    "end_location": self._json_point(start_location),
                    "before_distance": start_distance,
                    "after_distance": start_distance,
                    "distance_delta": 0.0,
                    "steps": [],
                    "status": s.status,
                    "action": "arrived",
                },
            )

        s.stop_auto_forward(w)
        deadline = time.time() + max(1.0, timeout_sec)
        steps: List[Dict[str, Any]] = []
        no_progress_count = 0
        last_distance = start_distance
        result_type = "max_steps"
        ok = False
        error = ""

        for step_index in range(max(1, max_steps)):
            if time.time() >= deadline:
                result_type = "timeout"
                error = "navigation timeout"
                break

            before_location = s._get_location(w)
            if before_location is None:
                result_type = "location_unavailable"
                error = "location unavailable during navigation"
                break

            house_scene = s._get_house_scene(w)
            if house_scene == s.HOUSE_INDOOR:
                s.status = s.STATUS_SCAN_ROOM
                result_type = "entered_indoor"
                ok = True
                break

            before_distance = get_distance(before_location, target_loc)
            if before_distance <= arrival_distance:
                s.stop_auto_forward(w)
                s.status = "SCANNING"
                result_type = "arrived"
                ok = True
                break

            if s.update_and_check_stuck(before_location):
                recovery = self.recover_from_stuck()
                steps.append({
                    "step": step_index + 1,
                    "action": "recover_from_stuck",
                    "recovery": recovery.observation,
                })
                result_type = "stuck"
                error = "stuck detected"
                break

            align_result = self._align_to_location_bounded(w, before_location, target_loc)
            precise = before_distance <= precise_distance
            duration_ms = self._nav_push_duration_ms(before_distance, precise=precise)
            y_bias = -240 if precise else -380
            wait_ms = 220 if precise else 300
            s.status = "PRECISE_NAV" if precise else "FAST_NAV"

            w.tap_single("摇杆", y_bias=y_bias, dura=duration_ms, wait=wait_ms)
            w.refresh_frame()
            s.handle_jump_logic(w)

            after_location = s._get_location(w)
            after_distance = None
            moved_distance = None
            distance_delta = None
            if after_location is not None:
                after_distance = get_distance(after_location, target_loc)
                moved_distance = get_distance(before_location, after_location)
                distance_delta = before_distance - after_distance

            steps.append({
                "step": step_index + 1,
                "action": "precise_push" if precise else "fast_push",
                "before_location": self._json_point(before_location),
                "after_location": self._json_point(after_location),
                "before_distance": before_distance,
                "after_distance": after_distance,
                "distance_delta": distance_delta,
                "moved_distance": moved_distance,
                "duration_ms": duration_ms,
                "y_bias": y_bias,
                "wait_ms": wait_ms,
                "align": align_result,
            })

            if after_distance is not None and after_distance <= arrival_distance:
                s.stop_auto_forward(w)
                s.status = "SCANNING"
                result_type = "arrived"
                ok = True
                break

            if moved_distance is None or moved_distance < 0.2 or (distance_delta is not None and distance_delta <= 0.05):
                no_progress_count += 1
            else:
                no_progress_count = 0
            if no_progress_count >= no_progress_limit:
                result_type = "no_progress"
                error = "navigation made no progress"
                break
            if after_distance is not None:
                last_distance = after_distance
        else:
            result_type = "max_steps"
            error = "navigation step limit reached"

        end_location = s._get_location(w)
        end_distance = get_distance(end_location, target_loc) if end_location is not None else None
        if end_distance is not None and end_distance <= arrival_distance:
            s.stop_auto_forward(w)
            s.status = "SCANNING"
            result_type = "arrived"
            ok = True
            error = ""

        return self._result(
            ok,
            "navigate_to_house_entry",
            {
                "at_entry": bool(ok and result_type == "arrived"),
                "result_type": result_type,
                "start_location": self._json_point(start_location),
                "end_location": self._json_point(end_location),
                "before_distance": start_distance,
                "after_distance": end_distance,
                "distance_delta": start_distance - end_distance if end_distance is not None else None,
                "moved_distance": get_distance(start_location, end_location) if end_location is not None else None,
                "target_location": self._json_point(target_loc),
                "status": s.status,
                "action": "closed_loop_navigation",
                "steps": steps,
                "step_count": len(steps),
                "last_distance": last_distance,
            },
            error=error,
        )

    def scan_entry_door(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        if not s.active_entry:
            return self._result(False, "scan_entry_door", error="active_entry unavailable")

        ideal_angle = s.active_entry["direction"]
        s.stop_auto_forward(w)
        s.align_direction_blocking(w, w.get_info("direction"), ideal_angle)
        w.refresh_frame()
        if s.check_and_lock_door(w):
            s.status = "VISUAL_APPROACH"
            return self._result(True, "scan_entry_door", {"found": True, "angle": ideal_angle})

        for offset in (30, -30):
            target_angle = (ideal_angle + offset) % 360
            s.align_direction_blocking(w, w.get_info("direction"), target_angle)
            w.refresh_frame()
            if s.check_and_lock_door(w):
                s.status = "VISUAL_APPROACH"
                return self._result(
                    True,
                    "scan_entry_door",
                    {"found": True, "angle": target_angle, "offset": offset},
                )

        s.handle_failed_entry_logic(ideal_angle)
        s.status = "IDLE"
        return self._result(True, "scan_entry_door", {"found": False, "angle": ideal_angle})

    def approach_entry_door(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        door = s.find_largest_door(w)
        if not door:
            s.status = "SCANNING"
            return self._result(False, "approach_entry_door", error="door not visible")

        aligned = s._align_to_target(w, door, tolerance_px=s.CENTER_TOLERANCE_PX)
        if not aligned:
            w.refresh_frame()
        else:
            s.status = "INTERACT"
        return self._result(
            True,
            "approach_entry_door",
            {
                "aligned": bool(aligned),
                "door": self._serialize_detection(door, s._get_scalar(w.get_info("direction"))),
                "status": s.status,
            },
        )

    def enter_house(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        house_scene = s._get_house_scene(w)
        if house_scene == s.HOUSE_INDOOR:
            s.stop_auto_forward(w)
            s.status = s.STATUS_SCAN_ROOM
            return self._result(True, "enter_house", {"entered": True, "house_scene": "indoor"})

        if w.get_info("跳跃"):
            s.handle_jump_logic(w)
            return self._result(True, "enter_house", {"entered": False, "action": "jump"})

        if w.get_info("开门"):
            w.click("开门")
            time.sleep(0.8)
            w.refresh_frame()
        elif w.get_info("关门"):
            pass
        else:
            door = s.find_largest_door(w)
            if door is not None:
                aligned = s._align_to_target(w, door, tolerance_px=s.CENTER_TOLERANCE_PX)
                if not aligned:
                    w.refresh_frame()
                    return self._result(True, "enter_house", {"entered": False, "action": "align_door"})

        s.stop_auto_forward(w)
        w.tap_single("摇杆", y_bias=-300, dura=320, wait=900)
        w.refresh_frame()
        house_scene = s._get_house_scene(w)
        entered = house_scene == s.HOUSE_INDOOR
        if entered:
            s.status = s.STATUS_SCAN_ROOM
            s.pubg_room_search_attempted = False
        return self._result(
            True,
            "enter_house",
            {"entered": bool(entered), "house_scene": self._house_scene_name(house_scene), "status": s.status},
        )

    def scan_room(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        location = s._get_location(w)
        if location is None:
            return self._result(False, "scan_room", error="location unavailable")

        if s.current_room_id is None:
            s._enter_room(location)
        s._scan_current_room(w, location)
        s.status = s.STATUS_LOOT_ITEM
        return self._result(True, "scan_room", self._room_memory(s.current_room_id))

    def select_next_supply(self) -> QwenToolResult:
        s = self.searcher
        room_id = self._ensure_room()
        if room_id is None:
            return self._result(False, "select_next_supply", error="room unavailable")

        target = s._select_next_supply(room_id)
        s.active_supply = target
        s.item_approach_steps = 0
        if target is None:
            s.status = s.STATUS_SELECT_DOOR
            return self._result(True, "select_next_supply", {"selected": None})

        s.status = s.STATUS_LOOT_ITEM
        return self._result(True, "select_next_supply", {"selected": self._serialize_target(target)})

    def align_to_object(
        self,
        target_type: str = "supply",
        target_id: Optional[str] = None,
        tolerance_px: Optional[float] = None,
    ) -> QwenToolResult:
        s = self.searcher
        target_type = self._normalize_target_type(target_type)
        target = self._resolve_visible_target(target_type, target_id)
        if target is None:
            return self._result(False, "align_to_object", error=f"no visible target: {target_type}")

        tolerance = float(tolerance_px or s.CENTER_TOLERANCE_PX)
        aligned = s._align_to_target(self.worker, target, tolerance_px=tolerance)
        if not aligned:
            self.worker.refresh_frame()
        return self._result(
            True,
            "align_to_object",
            {
                "target_type": target_type,
                "aligned": bool(aligned),
                "target": self._serialize_detection(target, s._get_scalar(self.worker.get_info("direction"))),
            },
        )

    def move_to_object(
        self,
        target_type: str = "supply",
        duration_ms: Optional[int] = None,
        wait_ms: Optional[int] = None,
    ) -> QwenToolResult:
        s = self.searcher
        align_result = self.align_to_object(target_type=target_type)
        aligned = bool(align_result.observation.get("aligned"))
        if not aligned:
            return self._result(True, "move_to_object", {"aligned": False, "moved": False})

        dura = int(duration_ms or (260 if target_type == "supply" else 340))
        wait = int(wait_ms or (420 if target_type == "supply" else 650))
        s._move_forward(self.worker, dura=dura, wait=wait)
        if target_type == "supply":
            s.item_approach_steps += 1
        elif target_type == "door":
            s.door_approach_steps += 1
        return self._result(
            True,
            "move_to_object",
            {"aligned": True, "moved": True, "duration_ms": dura, "wait_ms": wait},
        )

    def pickup_item(self) -> QwenToolResult:
        s = self.searcher
        room_id = self._ensure_room()
        if room_id is None:
            return self._result(False, "pickup_item", error="room unavailable")
        if s.active_supply is None:
            s.active_supply = s._select_next_supply(room_id)

        picked = s._click_pickup_if_available(self.worker)
        if picked and s.active_supply is not None:
            s._mark_supply_done(room_id, s.active_supply)
        return self._result(
            True,
            "pickup_item",
            {"picked": bool(picked), "room_memory": self._room_memory(room_id)},
        )

    def select_next_door(self) -> QwenToolResult:
        s = self.searcher
        room_id = self._ensure_room()
        if room_id is None:
            return self._result(False, "select_next_door", error="room unavailable")

        door = s._select_next_door(room_id)
        if door is None and s.room_stack:
            backtrack = s.room_stack.pop()
            door = {
                "id": f"backtrack:{room_id}:{len(s.room_stack)}",
                "room_id": room_id,
                "kind": "backtrack",
                "abs_angle": backtrack["return_angle"],
                "target_room_id": backtrack["room_id"],
                "box_h": 0.0,
                "area": 0.0,
                "center_x": s._frame_width() / 2.0,
                "center_y": 0.0,
            }
        s.active_door = door
        s.door_approach_steps = 0
        if door is None:
            s.status = s.STATUS_FINISHED
            return self._result(True, "select_next_door", {"selected": None})

        s.status = s.STATUS_APPROACH_DOOR
        return self._result(True, "select_next_door", {"selected": self._serialize_target(door)})

    def open_door(self) -> QwenToolResult:
        w = self.worker
        w.refresh_frame()
        if w.get_info("开门"):
            w.click("开门")
            time.sleep(0.7)
            w.refresh_frame()
            return self._result(True, "open_door", {"interacted": True, "button": "开门"})
        if w.get_info("关门"):
            return self._result(True, "open_door", {"interacted": False, "button": "关门"})
        return self._result(True, "open_door", {"interacted": False, "button": None})

    def enter_door(self, confirm: bool = True) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        location = s._get_location(w)
        if location is None:
            return self._result(False, "enter_door", error="location unavailable")

        if s.door_entry_start_location is None:
            s.door_entry_start_location = location
        s._move_forward(w, dura=340, wait=650)
        s.door_approach_steps += 1
        s.status = s.STATUS_ENTER_NEXT_ROOM

        if confirm:
            new_location = s._get_location(w)
            if new_location is not None:
                s._confirm_enter_next_room(w, new_location)

        return self._result(
            True,
            "enter_door",
            {
                "start_location": self._json_point(location),
                "current_location": self._json_point(s._get_location(w)),
                "status": s.status,
                "current_room_id": self._json_key(s.current_room_id),
            },
        )

    def check_stuck(self) -> QwenToolResult:
        s = self.searcher
        location = s._get_location(self.worker)
        if location is None:
            return self._result(False, "check_stuck", error="location unavailable")
        stuck = s.update_and_check_stuck(location)
        return self._result(
            True,
            "check_stuck",
            {
                "stuck": bool(stuck),
                "location": self._json_point(location),
                "history_len": len(s.history_locations),
            },
        )

    def recover_from_stuck(self) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        before = s._get_location(w)
        s.stop_auto_forward(w)

        if w.get_info("跳跃"):
            w.click("跳跃")
            time.sleep(0.2)
            w.tap_single("摇杆", y_bias=-360, dura=500, wait=700)
        else:
            w.tap_single("摇杆", y_bias=300, dura=320, wait=500)
            w.tap_single("摇杆", x_bias=260, dura=260, wait=450)
            w.tap_single("摇杆", y_bias=-300, dura=360, wait=700)
        w.refresh_frame()

        after = s._get_location(w)
        moved_distance = None
        if before is not None and after is not None:
            moved_distance = get_distance(before, after)
        s.history_locations = [after] if after is not None else []
        return self._result(
            True,
            "recover_from_stuck",
            {
                "before": self._json_point(before),
                "after": self._json_point(after),
                "moved_distance": moved_distance,
            },
        )

    def mark_house_done(self, force: bool = False) -> QwenToolResult:
        s = self.searcher
        w = self.worker
        house_scene = s._get_house_scene(w)
        if house_scene != s.HOUSE_OUTDOOR and not force:
            return self._result(
                False,
                "mark_house_done",
                {"house_scene": self._house_scene_name(house_scene)},
                error="not outdoors; pass force=true to mark anyway",
            )

        house_id = s.current_house_id
        if house_id is not None:
            s.completed_houses.add(house_id)
        s.searching_number = int(getattr(s, "searching_number", 0)) + 1
        exit_direction = w.get_info("direction")
        s.prepare_next_target_logic(exit_direction)
        s.current_house_id = None
        s.active_entry = None
        s.reset()
        s.status = "IDLE"
        return self._result(
            True,
            "mark_house_done",
            {
                "marked_house_id": house_id,
                "completed_house_count": s.searching_number,
                "status": s.status,
            },
        )

    def finish_house_search(self) -> QwenToolResult:
        self.searcher._finish_house_search(self.worker)
        return self._result(True, "finish_house_search", {"current_stage": self.worker.get_stage()})

    def wait_and_refresh(self, wait_sec: float = 0.2) -> QwenToolResult:
        wait_sec = max(0.0, min(float(wait_sec), 3.0))
        time.sleep(wait_sec)
        self.worker.refresh_frame()
        return self._result(True, "wait_and_refresh", {"wait_sec": wait_sec})

    def _ensure_room(self):
        s = self.searcher
        location = s._get_location(self.worker)
        if location is None:
            return None
        if s.current_room_id is None:
            s._enter_room(location)
        return s.current_room_id

    def _resolve_visible_target(self, target_type: str, target_id: Optional[str] = None):
        s = self.searcher
        w = self.worker
        class_ids = self._class_ids_for_type(target_type)

        expected = self._target_by_id(target_id) if target_id else None
        if expected is None:
            if target_type == "supply":
                expected = s.active_supply
            elif target_type == "door":
                expected = s.active_door

        if expected is not None:
            target = s._find_matching_target(w, expected, class_ids)
            if target is not None:
                return target
        return s._find_largest_target(s._get_forward_scene(w), class_ids)

    def _target_by_id(self, target_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not target_id:
            return None
        s = self.searcher
        for targets in list(s.room_supplies.values()) + list(s.room_doors.values()):
            for target in targets:
                if target.get("id") == target_id:
                    return target
        if s.active_supply and s.active_supply.get("id") == target_id:
            return s.active_supply
        if s.active_door and s.active_door.get("id") == target_id:
            return s.active_door
        return None

    def _class_ids_for_type(self, target_type: str) -> set:
        s = self.searcher
        if target_type == "supply":
            return set(s.SUPPLY_CLASS_IDS)
        if target_type == "door":
            return set(s.DOOR_CLASS_IDS)
        if target_type == "pick_menu":
            return set(s.PICK_MENU_CLASS_IDS)
        raise ValueError(f"unsupported target_type: {target_type}")

    def _normalize_target_type(self, target_type: str) -> str:
        value = str(target_type or "").lower().strip()
        aliases = {
            "item": "supply",
            "loot": "supply",
            "supply": "supply",
            "door": "door",
            "pick": "pick_menu",
            "pickup": "pick_menu",
            "pick_menu": "pick_menu",
        }
        if value not in aliases:
            raise ValueError(f"unsupported target_type: {target_type}")
        return aliases[value]

    def _serialize_detection(self, det: Sequence[float], direction: Optional[float]) -> Dict[str, Any]:
        s = self.searcher
        x1, y1, x2, y2 = [float(det[i]) for i in range(4)]
        class_id = int(det[5])
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        frame_w = s._frame_width()
        item = {
            "type": self._object_type(class_id),
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
            "center": [center_x, center_y],
            "center_offset_px": center_x - frame_w / 2.0,
            "area": s._detection_area(det),
            "box_h": max(0.0, y2 - y1),
        }
        if direction is not None:
            item["abs_angle"] = s._detection_abs_angle(det, direction)
        return item

    def _serialize_target(self, target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if target is None:
            return None
        keys = (
            "id",
            "kind",
            "abs_angle",
            "box_h",
            "area",
            "center_x",
            "center_y",
            "class_id",
            "target_room_id",
        )
        return {key: self._json_key(target.get(key)) for key in keys if key in target}

    def _serialize_entry(self, entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not entry:
            return None
        return {
            "location": self._json_point(tuple(entry["location"])) if entry.get("location") else None,
            "direction": entry.get("direction"),
        }

    def _room_memory(self, room_id) -> Dict[str, Any]:
        s = self.searcher
        return {
            "room_id": self._json_key(room_id),
            "loot_count": s.room_loot_counts.get(room_id, 0) if room_id is not None else 0,
            "supplies": [
                self._serialize_target(target)
                for target in s.room_supplies.get(room_id, [])
            ] if room_id is not None else [],
            "doors": [
                self._serialize_target(target)
                for target in s.room_doors.get(room_id, [])
            ] if room_id is not None else [],
            "visited_supply_ids": sorted(str(item) for item in s.visited_supply_ids),
            "visited_door_ids": sorted(str(item) for item in s.visited_door_ids),
            "bad_door_ids": sorted(str(item) for item in s.bad_door_ids),
            "room_stack_depth": len(s.room_stack),
        }

    def _interaction_state(self) -> Dict[str, bool]:
        w = self.worker
        return {
            "open_door": bool(w.get_info("开门")),
            "close_door": bool(w.get_info("关门")),
            "pickup_first": bool(w.get_info("拾取首个物资")),
            "jump": bool(w.get_info("跳跃")),
        }

    def _object_type(self, class_id: int) -> str:
        s = self.searcher
        if class_id in s.SUPPLY_CLASS_IDS:
            return "supply"
        if class_id in s.DOOR_CLASS_IDS:
            return "door"
        if class_id in s.PICK_MENU_CLASS_IDS:
            return "pick_menu"
        return "other"

    def _house_scene_name(self, house_scene: Optional[int]) -> str:
        s = self.searcher
        if house_scene == s.HOUSE_INDOOR:
            return "indoor"
        if house_scene == s.HOUSE_OUTDOOR:
            return "outdoor"
        if house_scene == s.HOUSE_ROOFTOP:
            return "rooftop"
        return "unknown"

    def _target_id(self, target: Optional[Dict[str, Any]]) -> Optional[str]:
        if not target:
            return None
        value = target.get("id")
        return str(value) if value is not None else None

    def _json_point(self, point: Optional[Tuple[int, int]]) -> Optional[List[int]]:
        if point is None:
            return None
        return [int(point[0]), int(point[1])]

    def _json_key(self, value):
        if isinstance(value, tuple):
            return list(value)
        return value

    def _nav_push_duration_ms(self, distance: float, *, precise: bool) -> int:
        """Return a bounded joystick hold time for one Qwen navigation step."""
        if precise:
            step_distance = min(max(float(distance) * 0.45, 0.25), 0.85)
            estimated = get_time_from_distance(step_distance)
            return self._bounded_int(estimated, 180, 650)

        step_distance = min(max(float(distance) * 0.35, 1.0), 3.0)
        estimated = get_time_from_distance(step_distance)
        return self._bounded_int(estimated, 520, 1800)

    def _align_to_location_bounded(
        self,
        worker: Any,
        current_location: Tuple[int, int],
        target_location: Tuple[int, int],
        *,
        tolerance: float = 6.0,
        max_turns: int = 3,
    ) -> Dict[str, Any]:
        turns = []
        aligned = False
        for turn_index in range(max(1, int(max_turns))):
            current_direction = self.searcher._get_scalar(worker.get_info("direction"))
            if current_direction is None:
                return {"aligned": False, "reason": "direction_unavailable", "turns": turns}
            target_angle = calculate_angle(current_location, target_location)
            turn_dir, px, diff = calculate_move_count(current_direction, target_angle)
            turns.append({
                "turn": turn_index + 1,
                "current_direction": current_direction,
                "target_angle": target_angle,
                "diff": diff,
                "x_bias": px if turn_dir == "right" else -px,
            })
            if abs(diff) <= tolerance:
                aligned = True
                break
            x_bias = px if turn_dir == "right" else -px
            worker.tap_single("视角", x_bias=int(x_bias), dura=520, wait=180)
            worker.refresh_frame()
        return {"aligned": aligned, "turns": turns}

    def _bounded_int(self, value: Any, min_value: int, max_value: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = min_value
        return max(min_value, min(max_value, number))

    def _config_float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _config_int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _with_feedback(self, result: QwenToolResult) -> QwenToolResult:
        observation = dict(result.observation or {})
        observation.setdefault("state_after", self._feedback_state())
        result.observation = observation
        return result

    def _refresh_after_tool(self, tool_name: str):
        if tool_name in {"get_game_state", "get_visible_objects"}:
            return
        try:
            self.worker.refresh_frame()
        except Exception:
            pass

    def _feedback_state(self) -> Dict[str, Any]:
        s = self.searcher
        w = self.worker
        try:
            location = s._get_location(w)
        except Exception:
            location = None
        try:
            house_scene = s._get_house_scene(w)
        except Exception:
            house_scene = None

        distance_to_entry = None
        if location is not None and s.active_entry:
            try:
                distance_to_entry = get_distance(location, tuple(s.active_entry["location"]))
            except Exception:
                distance_to_entry = None

        return {
            "location": self._json_point(location),
            "house_scene_name": self._house_scene_name(house_scene),
            "status": getattr(s, "status", None),
            "current_house_id": getattr(s, "current_house_id", None),
            "has_active_entry": bool(getattr(s, "active_entry", None)),
            "distance_to_entry": distance_to_entry,
            "active_supply_id": self._target_id(getattr(s, "active_supply", None)),
            "active_door_id": self._target_id(getattr(s, "active_door", None)),
            "completed_house_count": int(getattr(s, "searching_number", 0)),
            "auto_forward": bool(getattr(s, "auto_forward", False)),
        }

    def _result(
        self,
        ok: bool,
        tool_name: str,
        observation: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> QwenToolResult:
        return QwenToolResult(
            ok=bool(ok),
            tool_name=tool_name,
            observation=observation or {},
            error=error,
        )
