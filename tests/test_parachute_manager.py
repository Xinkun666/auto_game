import unittest
import importlib.util
import math
import pathlib
import sys
import types

PARACHUTE_MODULE = "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.parachute_manager"
MAP_UTILS_MODULE = "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_path_utils"
GEOMETRY_MODULE = "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry"
LOG_MODULE = "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log"


def load_parachute_manager_with_light_deps():
    sys.modules.pop(PARACHUTE_MODULE, None)

    map_utils = types.ModuleType(MAP_UTILS_MODULE)

    def get_distance(coord1, coord2):
        if coord1[0] is None or coord1[1] is None:
            return -1
        return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

    map_utils.get_distance = get_distance
    map_utils.__all__ = ["get_distance"]

    geometry = types.ModuleType(GEOMETRY_MODULE)
    geometry.align_direction = lambda w, target_pos: False
    geometry.__all__ = ["align_direction"]

    structured_log = types.ModuleType(LOG_MODULE)
    structured_log.autogame_print = lambda *args, **kwargs: None

    originals = {
        "cv2": sys.modules.get("cv2"),
        MAP_UTILS_MODULE: sys.modules.get(MAP_UTILS_MODULE),
        GEOMETRY_MODULE: sys.modules.get(GEOMETRY_MODULE),
        LOG_MODULE: sys.modules.get(LOG_MODULE),
    }

    sys.modules["cv2"] = types.ModuleType("cv2")
    sys.modules[MAP_UTILS_MODULE] = map_utils
    sys.modules[GEOMETRY_MODULE] = geometry
    sys.modules[LOG_MODULE] = structured_log
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "aw"
        / "autogame"
        / "customs_examples"
        / "Auto_PUBG_ALL"
        / "resource"
        / "control"
        / "parachute_manager.py"
    )
    spec = importlib.util.spec_from_file_location(PARACHUTE_MODULE, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PARACHUTE_MODULE] = module
    spec.loader.exec_module(module)
    return module.ParachuteManager, originals


def restore_modules(originals):
    sys.modules.pop(PARACHUTE_MODULE, None)
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class FakeFrameWorker:
    def __init__(self, location):
        self.location = location
        self.events = []

    def get_info(self, name):
        if name == "取消跟随":
            return False
        if name == "离开":
            return True
        if name == "location":
            return [self.location]
        if name == "direction":
            return None
        return None

    def click(self, name):
        self.events.append(("click", name))

    def tap_single(self, name, **kwargs):
        self.events.append(("tap_single", name, kwargs))

    def change_stage(self, stage):
        self.events.append(("change_stage", stage))


class ParachuteManagerTests(unittest.TestCase):
    def test_invalid_location_does_not_confirm_jump_distance(self):
        ParachuteManager, originals = load_parachute_manager_with_light_deps()
        manager = ParachuteManager()
        worker = FakeFrameWorker((None, None))

        try:
            for _ in range(3):
                manager.process(worker)
        finally:
            restore_modules(originals)

        self.assertNotIn(("click", "跳伞"), worker.events)
        self.assertEqual([], manager.jump_confirm_distances)
        self.assertIsNone(manager.last_dist)


if __name__ == "__main__":
    unittest.main()
