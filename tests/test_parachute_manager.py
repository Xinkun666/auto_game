import unittest
import sys
import types
from unittest import mock


sklearn_module = types.ModuleType("sklearn")
cluster_module = types.ModuleType("sklearn.cluster")
cluster_module.DBSCAN = object
sklearn_module.cluster = cluster_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules.setdefault("sklearn.cluster", cluster_module)

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control import parachute_manager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.parachute_manager import (
    ParachuteManager,
)


class FakeFrameWorker:
    def __init__(self, info=None):
        self.info = dict(info or {})
        self.events = []
        self.current_stage = "跳伞阶段"

    def get_info(self, name):
        return self.info.get(name)

    def click(self, name):
        self.events.append(("click", name))

    def tap_single(self, name, **kwargs):
        self.events.append(("tap_single", name, kwargs))

    def change_stage(self, stage):
        self.events.append(("change_stage", stage))
        self.current_stage = stage


class ParachuteManagerTests(unittest.TestCase):
    def test_invalid_location_does_not_enter_jump_confirmation_window(self):
        manager = ParachuteManager()
        w = FakeFrameWorker({"离开": True, "location": ((None, None), "unstable")})

        with mock.patch.object(parachute_manager, "align_direction"):
            for _ in range(3):
                manager.process(w)

        self.assertNotIn(("click", "跳伞"), w.events)
        self.assertEqual([], manager.jump_confirm_distances)

    def test_direct_coordinate_location_can_still_trigger_confirmed_jump(self):
        manager = ParachuteManager()
        w = FakeFrameWorker({"离开": True, "location": manager.target_pos})

        with mock.patch.object(parachute_manager, "align_direction"):
            for _ in range(3):
                manager.process(w)

        self.assertIn(("click", "跳伞"), w.events)
        self.assertIn(("change_stage", manager.landing_stage), w.events)

    def test_overshooting_far_from_target_jumps_instead_of_ending_round(self):
        manager = ParachuteManager()
        manager.OVERSHOOT_INCREASE_FRAMES = 2
        manager.DIVE_DURATION_MS = 0
        manager.configure(target_pos=(0, 0), landing_stage="搜房阶段")
        w = FakeFrameWorker({"离开": True, "location": (600, 0)})

        with mock.patch.object(parachute_manager, "align_direction"):
            manager.process(w)
            w.info["location"] = (610, 0)
            manager.process(w)
            w.info["location"] = (620, 0)
            manager.process(w)

        self.assertIn(("click", "跳伞"), w.events)
        self.assertIn(("change_stage", "搜房阶段"), w.events)
        self.assertNotIn(("change_stage", "结束阶段"), w.events)


if __name__ == "__main__":
    unittest.main()
