import unittest

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager import (
    HouseSceneSearchManager,
)


class RCityEntryLogicTests(unittest.TestCase):
    def setUp(self):
        self.manager = HouseSceneSearchManager.__new__(HouseSceneSearchManager)

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
