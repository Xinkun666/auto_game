from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager import (
    HouseSceneSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control import house_search_manager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control import house_scene_search_manager


class FakeMapNavigator:
    def is_walkable(self, pos):
        return True

    def nearest_walkable_within_radius(self, pos, radius):
        return tuple(pos), 0.0

    def plan_path(self, start_pos, end_pos):
        return [tuple(start_pos), tuple(end_pos)]


def make_manager(monkeypatch):
    monkeypatch.setattr(house_search_manager, "MapNavigator", FakeMapNavigator)
    return HouseSceneSearchManager()


def test_loads_r_city_house_area_config(monkeypatch):
    manager = make_manager(monkeypatch)

    assert manager.r_city_center == (1036, 745)
    assert len(manager.r_city_targets) == 29
    assert manager.r_city_near_distance == 30.0


def test_selects_route_target_from_approach_side(monkeypatch):
    manager = make_manager(monkeypatch)

    west_target = manager._select_r_city_route_target((950, 746))
    east_target = manager._select_r_city_route_target((1115, 748))

    assert west_target["side"] == "west"
    assert east_target["side"] == "east"


def test_water_escape_keeps_same_side_after_forward_blocked(monkeypatch):
    manager = make_manager(monkeypatch)

    first_side = manager._choose_water_escape_side(
        current_loc=(990, 790),
        target_loc=(1036, 745),
        current_direction=0,
    )
    manager._record_water_escape_attempt(before_loc=(990, 790), after_loc=(990.2, 790.1))
    second_side = manager._choose_water_escape_side(
        current_loc=(990.2, 790.1),
        target_loc=(1036, 745),
        current_direction=0,
    )

    assert second_side == first_side


class FakeFrame:
    current_stage = "搜房阶段"

    def __init__(self, info=None):
        self.info = dict(info or {})
        self.actions = []

    def get_info(self, name):
        return self.info.get(name)

    def click(self, name):
        self.actions.append(("click", name))

    def tap_single(self, name, **kwargs):
        self.actions.append(("tap", name, kwargs))

    def refresh_frame(self):
        self.actions.append(("refresh",))


class RCityEntryGuardManager(HouseSceneSearchManager):
    def align_direction_blocking(self, *args, **kwargs):
        raise AssertionError("R City free entry should not align to configured entry direction")

    def align_direction(self, *args, **kwargs):
        return True

    def handle_jump_logic(self, w):
        w.click("跳跃")
        w.info["house_scene"] = self.HOUSE_INDOOR


def test_r_city_entry_uses_free_door_window_search_instead_of_entry_direction(monkeypatch):
    monkeypatch.setattr(house_search_manager, "MapNavigator", FakeMapNavigator)
    manager = RCityEntryGuardManager()
    target = {
        "id": "r_city_test",
        "location": (1000, 1000),
        "approach_location": (998, 1000),
        "entry_direction": 355,
    }
    manager.current_r_city_target = target
    manager.current_house_id = target["id"]
    manager.active_entry = {
        "location": target["approach_location"],
        "direction": target["entry_direction"],
        "r_city_target_id": target["id"],
    }
    frame = FakeFrame(
        {
            "location": [(998, 1000)],
            "direction": 90,
            "house_scene": manager.HOUSE_NEAR_WALL,
            "跳跃": True,
        }
    )

    assert manager._enter_house_by_scene(frame) is True
    assert ("click", "跳跃") in frame.actions


def test_rotate_search_uses_clockwise_then_counterclockwise_push_cycles(monkeypatch):
    monkeypatch.setattr(house_search_manager, "MapNavigator", FakeMapNavigator)
    monkeypatch.setattr(house_scene_search_manager.time, "sleep", lambda _: None)
    manager = HouseSceneSearchManager()
    frame = FakeFrame({"house_scene": manager.HOUSE_INDOOR})
    turns = []

    def fake_turn(w, signed_angle):
        turns.append(signed_angle)

    manager._turn_raw_pixels = fake_turn

    result = manager._rotate_search_inside_house(frame)

    assert result == manager.ROTATE_RESULT_FALLBACK_EXIT
    movement_actions = [
        action for action in frame.actions
        if action[0] == "tap" and action[1] == "摇杆"
    ]
    assert len(movement_actions) == 12
    assert movement_actions[0][2]["x_bias"] < 0
    assert movement_actions[6][2]["x_bias"] > 0
    assert turns[:6] == [500] * 6
    assert turns[6:] == [-500] * 6
