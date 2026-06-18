import os
import sys
import types
import unittest

import numpy as np

os.environ.setdefault("TARGET_PROJECT_CASE", "Auto_PUBG_ALL")
sys.modules.setdefault(
    "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.SpecialSceneHandler",
    types.ModuleType("aw.autogame.customs_examples.Auto_PUBG_ALL.resource.SpecialSceneHandler"),
)

from aw.autogame.tools.GameFrameWorker import FrameWorker
from aw.autogame.tools.GameSceneHandler import DEFAULT_GROUP_NAME, StageLogicController


class FakeProcessor:
    screen_w = 100
    screen_h = 100

    def __init__(self):
        self.last_tasks_config = None

    def process(self, _frame_img, tasks_config, buffer_ratio=0.1):
        self.last_tasks_config = tasks_config
        return {key: f"result-{key}" for key in tasks_config}


class StageGroupingTests(unittest.TestCase):
    def _controller(self):
        controller = StageLogicController.__new__(StageLogicController)
        controller.processor = FakeProcessor()
        controller.stage_info = {
            "开始阶段": {
                "groups": {
                    "默认": {"all": True},
                    "轻量识别": {
                        "items": [
                            {"scene": "大厅", "type": "area", "name": "开始"},
                            {"scene": "大厅", "type": "special_area", "name": "位置"},
                        ]
                    },
                },
                "scenes": {
                    "大厅": {
                        "width": 100,
                        "height": 100,
                        "areas": {
                            "开始": {
                                "rect": [0.0, 0.0, 0.1, 0.1],
                                "search_scope": [0.0, 0.0, 0.2, 0.2],
                                "template": "templates/start.png",
                            },
                            "关闭": {
                                "rect": [0.2, 0.2, 0.3, 0.3],
                                "search_scope": [0.2, 0.2, 0.4, 0.4],
                                "template": "templates/close.png",
                            },
                        },
                        "points": {"点击": {"rect": [0.0, 0.0, 0.1, 0.1]}},
                        "special_areas": {
                            "位置": {"rect": [0.4, 0.4, 0.5, 0.5]},
                            "方向": {"rect": [0.6, 0.6, 0.7, 0.7]},
                        },
                    }
                },
            }
        }
        return controller

    def test_default_group_runs_all_area_and_special_tasks(self):
        controller = self._controller()

        result = controller.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), "开始阶段")

        self.assertEqual(
            {"大厅__开始", "大厅__关闭", "大厅__位置", "大厅__方向"},
            set(result),
        )

    def test_custom_group_runs_only_selected_area_and_special_tasks(self):
        controller = self._controller()

        result = controller.process_frame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            "开始阶段",
            "轻量识别",
        )

        self.assertEqual({"大厅__开始", "大厅__位置"}, set(result))

    def test_frame_worker_change_group_switches_current_group_and_reprocesses_current_frame(self):
        worker = FrameWorker.__new__(FrameWorker)
        worker.current_stage = "开始阶段"
        worker.current_group = DEFAULT_GROUP_NAME
        worker.stage_info = {}
        worker.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        worker.frame_index = 0
        worker.viz_proc = None
        worker.viz_queue = None
        worker.stage_resolver = self._controller()

        changed = FrameWorker.change_group(worker, "轻量识别")

        self.assertTrue(changed)
        self.assertEqual("轻量识别", worker.current_group)
        self.assertEqual({"大厅__开始", "大厅__位置"}, set(worker.stage_info))
