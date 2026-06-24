import unittest

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager import (
    HouseSceneSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control import (
    house_search_manager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import (
    MapNavigator,
)


class ForbiddenLandingNavigationTests(unittest.TestCase):
    def test_map_navigator_distinguishes_separate_forbidden_regions(self):
        navigator = MapNavigator.__new__(MapNavigator)
        navigator.binary_map = np.full((5, 7), 255, dtype=np.uint8)
        navigator.binary_map[1, 1] = 0
        navigator.binary_map[1, 2] = 0
        navigator.binary_map[3, 5] = 0
        navigator.height, navigator.width = navigator.binary_map.shape
        navigator._forbidden_component_labels = None

        self.assertTrue(navigator.same_forbidden_region((1, 1), (2, 1)))
        self.assertFalse(navigator.same_forbidden_region((1, 1), (5, 3)))
        self.assertFalse(navigator.same_forbidden_region((1, 1), (0, 0)))

    def _make_forbidden_entry_manager(self, entry_location=(500, 0)):
        class MapTool:
            def __init__(self):
                self.safe_queries = []

            def is_walkable(self, pos):
                return tuple(pos) not in {(0, 0), tuple(entry_location)}

            def same_forbidden_region(self, start, end):
                return True

            def nearest_walkable_within_radius(self, pos, radius):
                self.safe_queries.append((tuple(pos), radius))
                return (10, 0), 10.0

        manager = HouseSceneSearchManager.__new__(HouseSceneSearchManager)
        manager.map_tool = MapTool()
        manager.house_data = {
            "house_forbidden_entry": [
                {"location": entry_location, "direction": 90},
            ],
        }
        manager.completed_houses = set()
        manager.excluded_house_ids = set()
        manager.temp_skip_houses = set()
        manager.current_house_id = None
        manager.active_entry = None
        manager.initial_target_pending = True
        manager.initial_location_samples = []
        manager.history_locations = []
        manager.status = "IDLE"
        manager.forbidden_escape_target = None
        manager.avoid_angle_ref = None
        manager.avoid_mode = None
        manager.route_stuck_reference_loc = None
        manager.route_stuck_bypass_attempts = 0
        manager.house_bypass_unstuck_pause_until = 0.0
        manager.entry_near_micro_adjust_attempts = 0
        manager.auto_forward = True
        manager.max_history_len = 5
        manager.stuck_threshold = 0.5
        manager._get_stable_initial_location = lambda loc: loc
        manager.stop_auto_forward = lambda w: setattr(manager, "auto_forward", False)
        manager._should_abort = lambda w: False
        manager._get_house_scene = lambda w: manager.HOUSE_OUTDOOR
        manager.update_and_check_stuck = lambda loc: False
        manager._maybe_bypass_front_house_on_route = lambda *args, **kwargs: False
        manager.align_direction = lambda *args, **kwargs: True
        manager.handle_jump_logic = lambda w: None
        return manager

    class Worker:
        current_stage = "搜房阶段"

        def __init__(self, info=None):
            self.actions = []
            self.info = dict(info or {})

        def click(self, name):
            self.actions.append(("click", name))

        def tap_single(self, name, **kwargs):
            self.actions.append(("tap", name, kwargs))

        def refresh_frame(self):
            self.actions.append(("refresh",))

        def get_info(self, name):
            return self.info.get(name)

    def _make_micro_adjust_manager(self):
        manager = HouseSceneSearchManager.__new__(HouseSceneSearchManager)
        manager.active_entry = {"direction": 30}
        manager.entry_near_micro_adjust_attempts = 0
        manager.history_locations = []
        manager._get_current_location = lambda w: (0, 0)
        return manager

    def test_entry_micro_adjust_uses_right_up_vector_inside_three_units(self):
        manager = self._make_micro_adjust_manager()
        worker = self.Worker({"direction": 30})
        original_calculate_angle = house_search_manager.calculate_angle
        house_search_manager.calculate_angle = lambda current, target: 360
        try:
            result = manager._micro_adjust_near_entry_point(worker, (0, 0), (1, 0), 2.0)
        finally:
            house_search_manager.calculate_angle = original_calculate_angle

        self.assertTrue(result)
        movement = [
            action[2] for action in worker.actions
            if action[0] == "tap" and action[1] == "摇杆"
        ][-1]
        self.assertGreater(movement["x_bias"], 0)
        self.assertLess(movement["y_bias"], 0)

    def test_entry_micro_adjust_turns_first_when_target_is_behind_side_range(self):
        manager = self._make_micro_adjust_manager()
        worker = self.Worker({"direction": 30})
        original_calculate_angle = house_search_manager.calculate_angle
        house_search_manager.calculate_angle = lambda current, target: 210
        try:
            result = manager._micro_adjust_near_entry_point(worker, (0, 0), (1, 0), 2.0)
        finally:
            house_search_manager.calculate_angle = original_calculate_angle

        self.assertTrue(result)
        self.assertTrue(any(action[0] == "tap" and action[1] == "视角" for action in worker.actions))
        self.assertFalse(any(action[0] == "tap" and action[1] == "摇杆" for action in worker.actions))

    def test_forbidden_landing_locks_nearest_forbidden_entry_without_safe_escape(self):
        manager = self._make_forbidden_entry_manager()

        result = manager._prepare_initial_forbidden_entry_route(object(), (0, 0))

        self.assertEqual(result, "locked")
        self.assertEqual(manager.map_tool.safe_queries, [])
        self.assertEqual(manager.current_house_id, "house_forbidden_entry")
        self.assertEqual(manager.active_entry["location"], (500, 0))
        self.assertEqual(manager.status, "FAST_NAV")
        self.assertFalse(manager.initial_target_pending)
        self.assertFalse(manager.auto_forward)

    def test_forbidden_entry_route_keeps_ignoring_escape_on_next_frame(self):
        manager = self._make_forbidden_entry_manager()
        manager._prepare_initial_forbidden_entry_route(object(), (0, 0))

        self.assertTrue(manager._should_skip_forbidden_escape_for_active_entry((0, 0)))
        self.assertEqual(manager.map_tool.safe_queries, [])

    def test_searching_logic_keeps_direct_route_inside_same_forbidden_region(self):
        manager = self._make_forbidden_entry_manager()
        worker = self.Worker()

        manager.searching_logic(worker, (0, 0), 0)
        manager.map_tool.safe_queries.clear()
        manager.searching_logic(worker, (0, 0), 0)

        self.assertEqual(manager.current_house_id, "house_forbidden_entry")
        self.assertEqual(manager.forbidden_entry_direct_target, (500, 0))
        self.assertEqual(manager.map_tool.safe_queries, [])

    def test_entering_target_forbidden_region_does_not_trigger_escape(self):
        manager = self._make_forbidden_entry_manager()
        manager.current_house_id = "house_forbidden_entry"
        manager.active_entry = {"location": (500, 0), "direction": 90}
        manager._mark_forbidden_entry_direct_route((500, 0))

        self.assertTrue(manager._should_skip_forbidden_escape_for_active_entry((0, 0)))
        self.assertEqual(manager.map_tool.safe_queries, [])

    def test_forbidden_landing_does_not_lock_entry_in_different_forbidden_region(self):
        manager = self._make_forbidden_entry_manager()
        manager.map_tool.same_forbidden_region = lambda start, end: False

        result = manager._prepare_initial_forbidden_entry_route(object(), (0, 0))

        self.assertEqual(result, "skip")
        self.assertIsNone(manager.current_house_id)
        self.assertIsNone(manager.active_entry)
        self.assertTrue(manager.initial_target_pending)

    def test_forbidden_escape_uses_auto_forward_instead_of_slow_tap(self):
        class Worker:
            def __init__(self):
                self.actions = []

            def click(self, name):
                self.actions.append(("click", name))

            def tap_single(self, name, **kwargs):
                self.actions.append(("tap", name, kwargs))

            def refresh_frame(self):
                self.actions.append(("refresh",))

        class MapTool:
            def is_walkable(self, pos):
                return tuple(pos) == (10, 0)

            def nearest_walkable_within_radius(self, pos, radius):
                return (10, 0), 10.0

        manager = HouseSceneSearchManager.__new__(HouseSceneSearchManager)
        manager.map_tool = MapTool()
        manager.forbidden_escape_target = None
        manager.initial_location_samples = []
        manager.history_locations = []
        manager.max_history_len = 5
        manager.stuck_threshold = 0.5
        manager.house_bypass_unstuck_pause_until = 0.0
        manager.auto_forward = False
        manager.aligned_targets = []
        manager._maybe_bypass_front_house_on_route = lambda *args, **kwargs: False
        manager.execute_unstuck_logic = lambda *args, **kwargs: False
        manager.align_direction = lambda w, target: manager.aligned_targets.append(target) or True
        manager.handle_jump_logic = lambda w: w.actions.append(("jump_check",))

        worker = Worker()

        self.assertTrue(manager._handle_forbidden_escape(worker, (0, 0), 0))
        self.assertEqual(manager.aligned_targets, [(10, 0)])
        self.assertIn(("click", "自动前进"), worker.actions)
        self.assertNotIn(
            "摇杆",
            [action[1] for action in worker.actions if action[0] == "tap"],
        )
        self.assertTrue(manager.auto_forward)


if __name__ == "__main__":
    unittest.main()
