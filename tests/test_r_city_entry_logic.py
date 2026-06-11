import unittest

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control import (
    house_scene_search_manager as house_scene_module,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager import (
    HouseSceneSearchManager,
)


class RCityEntryLogicTests(unittest.TestCase):
    def setUp(self):
        self.manager = HouseSceneSearchManager.__new__(HouseSceneSearchManager)

    def test_blank_side_probes_forward_twice_before_switching_side(self):
        class Worker:
            def __init__(self):
                self.moves = []

            def tap_single(self, btn, **kwargs):
                self.moves.append((btn, kwargs))

        w = Worker()
        seen_targets = [(None, None), (None, None), ("door", None)]
        events = []
        self.manager.R_CITY_ENTRY_BLANK_PROBE_STEPS = 2
        self.manager.R_CITY_ENTRY_BLANK_PROBE_Y_BIAS = -200
        self.manager.R_CITY_ENTRY_BLANK_PROBE_DURA = 200
        self.manager.R_CITY_ENTRY_BLANK_PROBE_WAIT = 200
        self.manager._blocking_refresh_frame = lambda w, reason: True
        self.manager._is_indoor = lambda w: False
        self.manager._find_r_city_entry_visual_targets = lambda w: seen_targets.pop(0)
        self.manager._choose_r_city_entry_visual_target = (
            lambda door, window: ("door", door) if door else (None, None)
        )
        self.manager._handle_r_city_door_entry_step = (
            lambda w, door: events.append(("door", door)) or "retry"
        )

        result = self.manager._probe_blank_r_city_side_for_entry_targets(w)

        self.assertEqual(result, "retry")
        self.assertEqual(
            w.moves,
            [
                (
                    "摇杆",
                    {"y_bias": -200, "dura": 200, "wait": 200},
                ),
                (
                    "摇杆",
                    {"y_bias": -200, "dura": 200, "wait": 200},
                ),
            ],
        )
        self.assertEqual(events, [("door", "door")])

    def test_door_entry_uses_stone_wall_flow_before_long_forward_push(self):
        events = []
        self.manager._align_to_r_city_forward_target = (
            lambda w, target, label: events.append(("align", label)) or True
        )
        self.manager._refresh_and_settle_r_city_entry = (
            lambda w: events.append(("refresh", "door aligned"))
        )
        self.manager._find_largest_forward_target = (
            lambda w, class_ids: "stone_wall"
        )
        self.manager._handle_r_city_stone_wall_entry = (
            lambda w: events.append("stone_wall_flow") or True
        )
        self.manager._tap_r_city_entry_forward = (
            lambda *args, **kwargs: events.append("long_forward") or False
        )

        result = self.manager._handle_r_city_door_entry_step(object(), "door")

        self.assertEqual(result, "success")
        self.assertEqual(events, [("align", "门"), ("refresh", "door aligned"), "stone_wall_flow"])

    def test_stone_wall_flow_pushes_jumps_then_small_forward(self):
        class Worker:
            def __init__(self):
                self.clicks = []

            def click(self, btn):
                self.clicks.append(btn)

        w = Worker()
        pushes = []
        self.manager.R_CITY_ENTRY_STONE_WALL_FORWARD_Y_BIAS = -300
        self.manager.R_CITY_ENTRY_STONE_WALL_FORWARD_DURA = 100
        self.manager.R_CITY_ENTRY_STONE_WALL_FORWARD_WAIT = 300
        self.manager.R_CITY_ENTRY_JUMP_FORWARD_Y_BIAS = -200
        self.manager.R_CITY_ENTRY_JUMP_FORWARD_DURA = 200
        self.manager.R_CITY_ENTRY_JUMP_FORWARD_WAIT = 50
        self.manager.R_CITY_ENTRY_STONE_WALL_JUMP_SETTLE_SECONDS = 0
        self.manager._blocking_refresh_frame = lambda w, reason: True
        self.manager._tap_r_city_entry_forward = (
            lambda w, reason, y_bias, dura, wait, **kwargs:
            pushes.append((reason, y_bias, dura, wait, kwargs)) or False
        )

        self.assertFalse(self.manager._handle_r_city_stone_wall_entry(w))

        self.assertEqual(w.clicks, ["跳跃"])
        self.assertEqual(
            pushes,
            [
                ("stone_wall 前方遮挡，先短前推", -300, 100, 300, {"check_jump": False}),
                ("stone_wall 跳跃后小幅前推", -200, 200, 50, {"check_jump": False}),
            ],
        )

    def test_visible_door_in_r_city_interrupts_navigation_and_enters_door_flow(self):
        events = []
        target = {
            "id": "r_city_near",
            "location": (102, 100),
            "approach_location": (102, 100),
            "entry_direction": 90,
        }
        self.manager.current_house_id = "r_city_old"
        self.manager.status = "FAST_NAV"
        self.manager.r_city_near_distance = 30.0
        self.manager.history_locations = [(100, 100)]
        self.manager.r_city_route_target = "route"
        self.manager.r_city_route_path = [(100, 100)]
        self.manager.r_city_route_index = 1
        self.manager.r_city_entry_large_backoff_count = 2
        self.manager.r_city_side_probe_target = None
        self.manager.r_city_side_probe_count = 0
        self.manager.find_largest_door = lambda w: "door"
        self.manager._find_largest_forward_target = lambda w, class_ids: None
        self.manager._nearest_r_city_body_target = lambda loc, max_distance: (target, 2.0)
        self.manager.stop_auto_forward = lambda w: events.append("stop")
        self.manager._handle_r_city_door_entry_step = (
            lambda w, door: events.append(("door_step", door)) or "retry"
        )

        handled = self.manager._maybe_enter_visible_r_city_door(object(), (100, 100))

        self.assertTrue(handled)
        self.assertEqual(self.manager.current_house_id, "r_city_near")
        self.assertEqual(self.manager.status, self.manager.STATUS_SCENE_ENTRY)
        self.assertEqual(self.manager.history_locations, [])
        self.assertEqual(events, ["stop", ("door_step", "door")])

    def test_visible_window_in_r_city_interrupts_navigation_and_enters_window_flow(self):
        events = []
        target = {
            "id": "r_city_near",
            "location": (102, 100),
            "approach_location": (102, 100),
            "entry_direction": 90,
        }
        self.manager.current_house_id = "r_city_old"
        self.manager.status = "FAST_NAV"
        self.manager.r_city_near_distance = 30.0
        self.manager.history_locations = [(100, 100)]
        self.manager.r_city_route_target = "route"
        self.manager.r_city_route_path = [(100, 100)]
        self.manager.r_city_route_index = 1
        self.manager.r_city_entry_large_backoff_count = 2
        self.manager.r_city_side_probe_target = None
        self.manager.r_city_side_probe_count = 0
        self.manager.find_largest_door = lambda w: None
        self.manager._find_largest_forward_target = lambda w, class_ids: "window"
        self.manager._nearest_r_city_body_target = lambda loc, max_distance: (target, 2.0)
        self.manager.stop_auto_forward = lambda w: events.append("stop")
        self.manager._handle_r_city_window_entry_step = (
            lambda w, window: events.append(("window_step", window)) or "retry"
        )

        handled = self.manager._maybe_enter_visible_r_city_door(object(), (100, 100))

        self.assertTrue(handled)
        self.assertEqual(self.manager.current_house_id, "r_city_near")
        self.assertEqual(self.manager.status, self.manager.STATUS_SCENE_ENTRY)
        self.assertEqual(self.manager.history_locations, [])
        self.assertEqual(events, ["stop", ("window_step", "window")])

    def test_visual_alignment_refreshes_through_blocking_refresh_before_next_step(self):
        events = []
        sleep_calls = []
        original_sleep = house_scene_module.time.sleep
        self.manager.R_CITY_ENTRY_TARGET_ALIGN_TOLERANCE = 8
        self.manager._target_relative_angle = lambda target: 20
        self.manager._turn = lambda w, delta: events.append(("turn", delta))
        self.manager._blocking_refresh_frame = (
            lambda w, reason: events.append(("refresh", reason)) or True
        )

        house_scene_module.time.sleep = lambda seconds: sleep_calls.append(seconds)
        try:
            self.assertTrue(
                self.manager._align_to_r_city_forward_target(object(), "door", "门")
            )
        finally:
            house_scene_module.time.sleep = original_sleep

        self.assertEqual(events[0], ("turn", 20))
        self.assertEqual(events[1], ("refresh", "门视觉对齐后"))
        self.assertEqual(sleep_calls, [])

    def test_r_city_entry_refresh_does_not_sleep_after_blocking_frame(self):
        events = []
        sleep_calls = []
        original_sleep = house_scene_module.time.sleep
        self.manager._blocking_refresh_frame = (
            lambda w, reason: events.append(("refresh", reason)) or True
        )

        house_scene_module.time.sleep = lambda seconds: sleep_calls.append(seconds)
        try:
            self.manager._refresh_and_settle_r_city_entry(object())
        finally:
            house_scene_module.time.sleep = original_sleep

        self.assertEqual(events, [("refresh", "R城进房动作后")])
        self.assertEqual(sleep_calls, [])

    def test_choose_window_when_door_is_much_more_off_center(self):
        self.manager._target_relative_angle = lambda target: target["angle"]
        door = {"angle": 24}
        window = {"angle": 5}

        label, target = self.manager._choose_r_city_entry_visual_target(door, window)

        self.assertEqual(label, "window")
        self.assertIs(target, window)

    def test_choose_door_when_both_targets_are_off_center(self):
        self.manager._target_relative_angle = lambda target: target["angle"]
        door = {"angle": 28}
        window = {"angle": -26}

        label, target = self.manager._choose_r_city_entry_visual_target(door, window)

        self.assertEqual(label, "door")
        self.assertIs(target, door)

    def test_side_probe_point_rotates_clockwise_and_stays_near_house(self):
        point = self.manager._r_city_clockwise_side_probe_point(
            current_loc=(100, 96),
            house_loc=(100, 100),
        )

        self.assertEqual(point, (104, 100))

    def test_prefers_next_house_when_route_is_clear_and_not_more_than_double(self):
        current = (100, 96)
        house = {"id": "current", "location": (100, 100)}
        side_point = (104, 100)
        next_target = {"id": "next", "location": (106, 96)}

        self.assertTrue(
            self.manager._should_switch_to_next_r_city_house_after_blank_side(
                current,
                house,
                side_point,
                next_target,
            )
        )

    def test_stays_on_current_house_when_next_route_crosses_current_house(self):
        current = (100, 96)
        house = {"id": "current", "location": (100, 100)}
        side_point = (104, 100)
        next_target = {"id": "next", "location": (100, 108)}

        self.assertFalse(
            self.manager._should_switch_to_next_r_city_house_after_blank_side(
                current,
                house,
                side_point,
                next_target,
            )
        )


if __name__ == "__main__":
    unittest.main()
