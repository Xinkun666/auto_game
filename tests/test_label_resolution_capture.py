import unittest
import tempfile
from pathlib import Path

from aw.autogame.tools.Label import (
    AutoStudioWindow,
    DEFAULT_GLOBAL_SCENE_GROUP_NAME,
    DEFAULT_SCENE_GROUP_NAME,
    GroupData,
    GroupItemRef,
    ItemData,
    ProjectData,
    RectData,
    SceneData,
    SceneGroupData,
    StageData,
)


class FakePixmap:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def width(self):
        return self._width

    def height(self):
        return self._height

    def isNull(self):
        return False


class LabelResolutionCaptureTests(unittest.TestCase):
    def _window_with_stage(self, stage):
        window = AutoStudioWindow.__new__(AutoStudioWindow)
        window.project = ProjectData(name="demo", stages=[stage])
        AutoStudioWindow._ensure_project_scene_pool(window.project)
        window.current_stage = stage
        return window

    def test_capture_new_resolution_clones_items_from_existing_scene(self):
        source_scene = SceneData(
            id="scene-1",
            name="大厅",
            image_path="/tmp/old.jpeg",
            pixmap=FakePixmap(100, 50),
            image_width=100,
            image_height=50,
            items=[
                ItemData(
                    id="item-1",
                    name="开始",
                    item_type="area",
                    rect=RectData(10, 5, 20, 10),
                    search_scope=RectData(5, 2, 40, 20),
                    match_mode="gray",
                )
            ],
        )
        stage = StageData(id="stage-1", name="开始阶段", scenes=[source_scene])
        window = self._window_with_stage(stage)

        result = AutoStudioWindow._apply_capture_pixmap_to_scene_resolution(
            window,
            source_scene,
            "/tmp/new.jpeg",
            FakePixmap(200, 100),
        )

        self.assertEqual("created", result.action)
        self.assertEqual(2, len(stage.scenes))
        new_scene = result.scene
        self.assertIs(stage.scenes[1], new_scene)
        self.assertEqual("大厅", new_scene.name)
        self.assertEqual((200, 100), (new_scene.image_width, new_scene.image_height))
        self.assertEqual("/tmp/new.jpeg", new_scene.image_path)
        self.assertEqual(1, len(new_scene.items))
        cloned_item = new_scene.items[0]
        self.assertEqual("开始", cloned_item.name)
        expected_rect = AutoStudioWindow._scale_rect_between_images(
            window,
            source_scene.items[0].rect,
            100,
            50,
            200,
            100,
        )
        expected_scope = AutoStudioWindow._scale_rect_between_images(
            window,
            source_scene.items[0].search_scope,
            100,
            50,
            200,
            100,
        )
        self.assertEqual((expected_rect.x, expected_rect.y, expected_rect.w, expected_rect.h), (
            cloned_item.rect.x,
            cloned_item.rect.y,
            cloned_item.rect.w,
            cloned_item.rect.h,
        ))
        self.assertEqual((expected_scope.x, expected_scope.y, expected_scope.w, expected_scope.h), (
            cloned_item.search_scope.x,
            cloned_item.search_scope.y,
            cloned_item.search_scope.w,
            cloned_item.search_scope.h,
        ))

    def test_capture_existing_resolution_replaces_that_scene_image_only(self):
        source_scene = SceneData(
            id="scene-1",
            name="大厅",
            image_path="/tmp/old-100.jpeg",
            pixmap=FakePixmap(100, 50),
            image_width=100,
            image_height=50,
            items=[ItemData(id="item-1", name="开始", item_type="control", rect=RectData(10, 5, 20, 10))],
        )
        existing_scene = SceneData(
            id="scene-2",
            name="大厅",
            image_path="/tmp/old-200.jpeg",
            pixmap=FakePixmap(200, 100),
            image_width=200,
            image_height=100,
            items=[ItemData(id="item-2", name="开始", item_type="control", rect=RectData(20, 10, 40, 20))],
        )
        stage = StageData(id="stage-1", name="开始阶段", scenes=[source_scene, existing_scene])
        window = self._window_with_stage(stage)

        result = AutoStudioWindow._apply_capture_pixmap_to_scene_resolution(
            window,
            source_scene,
            "/tmp/replacement.jpeg",
            FakePixmap(200, 100),
        )

        self.assertEqual("replaced_existing", result.action)
        self.assertEqual(2, len(stage.scenes))
        self.assertIs(existing_scene, result.scene)
        self.assertEqual("/tmp/replacement.jpeg", existing_scene.image_path)
        self.assertEqual((200, 100), (existing_scene.image_width, existing_scene.image_height))
        self.assertEqual((20, 10, 40, 20), (
            existing_scene.items[0].rect.x,
            existing_scene.items[0].rect.y,
            existing_scene.items[0].rect.w,
            existing_scene.items[0].rect.h,
        ))
        self.assertEqual("/tmp/old-100.jpeg", source_scene.image_path)

    def test_stage_groups_serialize_default_and_custom_item_refs(self):
        stage = StageData(
            id="stage-1",
            name="开始阶段",
            groups=[
                GroupData(name="默认", includes_all=True),
                GroupData(
                    name="轻量识别",
                    items=[
                        GroupItemRef(scene_name="大厅", item_type="area", item_name="开始"),
                        GroupItemRef(scene_name="大厅", item_type="special_area", item_name="位置"),
                    ],
                ),
            ],
        )
        window = self._window_with_stage(stage)

        exported = AutoStudioWindow._serialize_stage_groups(window, stage)

        self.assertEqual({"all": True}, exported["默认"])
        self.assertEqual(
            [
                {"scene": "大厅", "type": "area", "name": "开始"},
                {"scene": "大厅", "type": "special_area", "name": "位置"},
            ],
            exported["轻量识别"]["items"],
        )

    def test_stage_groups_import_missing_groups_as_default(self):
        stage = StageData(id="stage-1", name="开始阶段")
        window = self._window_with_stage(stage)

        groups = AutoStudioWindow._deserialize_stage_groups(window, {}, stage)

        self.assertEqual(["默认"], [group.name for group in groups])
        self.assertTrue(groups[0].includes_all)

    def test_current_group_filters_area_and_special_but_keeps_controls_visible(self):
        scene = SceneData(
            id="scene-1",
            name="大厅",
            items=[
                ItemData(id="area-1", name="开始", item_type="area", rect=RectData(0, 0, 1, 1)),
                ItemData(id="special-1", name="位置", item_type="special_area", rect=RectData(0, 0, 1, 1)),
                ItemData(id="control-1", name="点击", item_type="control", rect=RectData(0, 0, 1, 1)),
            ],
        )
        stage = StageData(
            id="stage-1",
            name="开始阶段",
            scenes=[scene],
            groups=[
                GroupData(name="默认", includes_all=True),
                GroupData(name="轻量识别", items=[GroupItemRef("大厅", "area", "开始")]),
            ],
            active_group_name="轻量识别",
        )
        window = self._window_with_stage(stage)

        visible_names = [
            item.name
            for item in scene.items
            if AutoStudioWindow._is_item_visible_in_stage_group(window, stage, scene, item)
        ]

        self.assertEqual(["开始", "点击"], visible_names)

    def test_old_info_project_imports_into_default_scene_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "info.py").write_text(
                "\n".join(
                    [
                        "PROJECT_NAME = 'demo'",
                        "STAGE_INFO = {",
                        "    '开始阶段': {",
                        "        'scenes': {",
                        "            '大厅': {",
                        "                'image': '',",
                        "                'width': 100,",
                        "                'height': 50,",
                        "                'areas': {'开始': {'rect': [0, 0, 0.1, 0.2], 'search_scope': [0, 0, 0.2, 0.4]}},",
                        "                'points': {'点击': {'rect': [0.4, 0.4, 0.5, 0.5]}},",
                        "                'special_areas': {},",
                        "            }",
                        "        },",
                        "        'groups': {'默认': {'all': True}},",
                        "    },",
                        "    '跑圈阶段': {",
                        "        'scenes': {",
                        "            '大厅': {",
                        "                'image': '',",
                        "                'width': 100,",
                        "                'height': 50,",
                        "                'areas': {'开始': {'rect': [0, 0, 0.1, 0.2], 'search_scope': [0, 0, 0.2, 0.4]}},",
                        "                'points': {'点击': {'rect': [0.4, 0.4, 0.5, 0.5]}},",
                        "                'special_areas': {},",
                        "            }",
                        "        },",
                        "        'groups': {'默认': {'all': True}},",
                        "    },",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            project = AutoStudioWindow._load_project_model_from_dir(str(project_dir))

        self.assertEqual("demo", project.name)
        self.assertEqual(["待分组场景", "全局分组"], [group.name for group in project.scene_groups])
        self.assertEqual("待分组场景", DEFAULT_SCENE_GROUP_NAME)
        self.assertEqual(1, len(project.scene_groups[0].scenes))
        shared_scene = project.scene_groups[0].scenes[0]
        self.assertIs(project.stages[0].scenes[0], shared_scene)
        self.assertIs(project.stages[1].scenes[0], shared_scene)
        self.assertEqual(["开始", "点击"], [item.name for item in shared_scene.items])

    def test_existing_stage_scene_lists_are_registered_in_default_scene_pool(self):
        shared_scene = SceneData(
            id="scene-1",
            name="大厅",
            image_width=100,
            image_height=50,
            items=[ItemData(id="item-1", name="开始", item_type="area", rect=RectData(0, 0, 1, 1))],
        )
        project = ProjectData(
            name="demo",
            stages=[
                StageData(id="stage-1", name="开始阶段", scenes=[shared_scene]),
                StageData(id="stage-2", name="跑圈阶段", scenes=[shared_scene]),
            ],
        )

        AutoStudioWindow._ensure_project_scene_pool(project)

        self.assertEqual(["待分组场景", "全局分组"], [group.name for group in project.scene_groups])
        self.assertEqual([shared_scene], project.scene_groups[0].scenes)

    def test_legacy_default_global_group_is_migrated_to_global_group(self):
        global_scene = SceneData(id="scene-1", name="退出弹窗")
        project = ProjectData(
            name="demo",
            scene_groups=[
                SceneGroupData(id="group-1", name=DEFAULT_SCENE_GROUP_NAME),
                SceneGroupData(id="group-2", name="默认分组", scenes=[global_scene]),
            ],
        )

        AutoStudioWindow._ensure_project_scene_pool(project)

        self.assertEqual(["待分组场景", "全局分组"], [group.name for group in project.scene_groups])
        self.assertEqual([global_scene], project.scene_groups[1].scenes)

    def test_stage_scene_references_share_pool_scene_without_cloning(self):
        pool_scene = SceneData(
            id="scene-1",
            name="游戏内主界面",
            items=[ItemData(id="point-1", name="跳跃", item_type="control", rect=RectData(1, 2, 3, 4))],
        )
        stage_a = StageData(id="stage-a", name="搜房阶段")
        stage_b = StageData(id="stage-b", name="跑圈阶段")
        project = ProjectData(name="demo", stages=[stage_a, stage_b])

        AutoStudioWindow._add_scene_to_project_pool(project, pool_scene, "游戏场景")
        AutoStudioWindow._add_scene_reference_to_stage(stage_a, pool_scene)
        AutoStudioWindow._add_scene_reference_to_stage(stage_b, pool_scene)
        pool_scene.items[0].name = "跳跃按钮"

        self.assertIs(stage_a.scenes[0], pool_scene)
        self.assertIs(stage_b.scenes[0], pool_scene)
        self.assertEqual("跳跃按钮", stage_a.scenes[0].items[0].name)
        self.assertEqual(["游戏场景"], [group.name for group in project.scene_groups])

    def test_removing_stage_scene_reference_keeps_pool_scene(self):
        pool_scene = SceneData(id="scene-1", name="确认弹窗")
        stage = StageData(id="stage-1", name="启动阶段", scenes=[pool_scene])
        project = ProjectData(name="demo", stages=[stage])
        AutoStudioWindow._add_scene_to_project_pool(project, pool_scene, "弹窗")

        removed = AutoStudioWindow._remove_scene_reference_from_stage(stage, pool_scene)

        self.assertTrue(removed)
        self.assertEqual([], stage.scenes)
        self.assertEqual([pool_scene], project.scene_groups[0].scenes)

    def test_first_pool_scene_resolution_returns_first_matching_scene(self):
        first_resolution = SceneData(id="scene-1", name="大厅", image_width=100, image_height=50)
        second_resolution = SceneData(id="scene-2", name="大厅", image_width=200, image_height=100)
        other_scene = SceneData(id="scene-3", name="设置", image_width=100, image_height=50)
        scene_group = SceneGroupData(
            id="group-1",
            name="游戏场景",
            scenes=[first_resolution, second_resolution, other_scene],
        )

        selected = AutoStudioWindow._first_pool_scene_resolution(scene_group, "大厅")

        self.assertIs(first_resolution, selected)

    def test_move_selected_pending_scenes_into_target_group(self):
        pending_a = SceneData(id="scene-1", name="大厅", image_width=100, image_height=50)
        pending_b = SceneData(id="scene-2", name="设置", image_width=100, image_height=50)
        pending = SceneGroupData(id="group-1", name=DEFAULT_SCENE_GROUP_NAME, scenes=[pending_a, pending_b])
        target = SceneGroupData(id="group-2", name="游戏场景")
        project = ProjectData(name="demo", scene_groups=[pending, target])

        moved = AutoStudioWindow._move_pending_scenes_to_group(project, target, ["大厅"])

        self.assertEqual([pending_a], moved)
        self.assertEqual([pending_b], pending.scenes)
        self.assertEqual([pending_a], target.scenes)

    def test_moving_pending_scene_to_global_group_updates_existing_stages(self):
        global_scene = SceneData(id="scene-1", name="退出弹窗")
        pending = SceneGroupData(id="group-1", name=DEFAULT_SCENE_GROUP_NAME, scenes=[global_scene])
        global_group = SceneGroupData(id="group-2", name=DEFAULT_GLOBAL_SCENE_GROUP_NAME)
        stage = StageData(id="stage-1", name="搜房阶段")
        project = ProjectData(name="demo", stages=[stage], scene_groups=[pending, global_group])

        moved = AutoStudioWindow._move_pending_scenes_to_group(project, global_group, ["退出弹窗"])

        self.assertEqual([global_scene], moved)
        self.assertEqual([], pending.scenes)
        self.assertEqual([global_scene], global_group.scenes)
        self.assertIs(stage.scenes[0], global_scene)

    def test_global_scene_group_references_existing_and_future_stages(self):
        stage_a = StageData(id="stage-a", name="搜房阶段")
        stage_b = StageData(id="stage-b", name="跑圈阶段")
        global_scene = SceneData(id="scene-global", name="退出弹窗")
        project = ProjectData(name="demo", stages=[stage_a, stage_b])
        AutoStudioWindow._ensure_project_scene_pool(project)
        global_group = AutoStudioWindow._global_scene_group(project)

        AutoStudioWindow._add_scene_to_project_pool(project, global_scene, DEFAULT_GLOBAL_SCENE_GROUP_NAME)
        AutoStudioWindow._sync_global_scenes_to_stages(project)
        future_stage = AutoStudioWindow._new_stage_with_global_scenes(project, "结算阶段")

        self.assertIs(project.scene_groups[1], global_group)
        self.assertIs(stage_a.scenes[0], global_scene)
        self.assertIs(stage_b.scenes[0], global_scene)
        self.assertIs(future_stage.scenes[0], global_scene)

    def test_stage_pool_scene_choices_prioritize_global_then_pending_then_other_groups(self):
        pending_scene = SceneData(id="scene-pending", name="待分组场景")
        global_scene = SceneData(id="scene-global", name="全局弹窗")
        custom_scene = SceneData(id="scene-custom", name="游戏内主界面")
        project = ProjectData(
            name="demo",
            scene_groups=[
                SceneGroupData(id="group-pending", name=DEFAULT_SCENE_GROUP_NAME, scenes=[pending_scene]),
                SceneGroupData(id="group-global", name=DEFAULT_GLOBAL_SCENE_GROUP_NAME, scenes=[global_scene]),
                SceneGroupData(id="group-custom", name="游戏场景", scenes=[custom_scene]),
            ],
        )
        window = AutoStudioWindow.__new__(AutoStudioWindow)
        window.project = project

        choices, labels = AutoStudioWindow._pool_scene_choices(window)

        self.assertIs(choices[labels[0]], global_scene)
        self.assertIs(choices[labels[1]], pending_scene)
        self.assertIs(choices[labels[2]], custom_scene)

    def test_scene_pool_tree_groups_show_pending_then_global_as_siblings(self):
        pending = SceneGroupData(id="group-pending", name=DEFAULT_SCENE_GROUP_NAME)
        global_group = SceneGroupData(id="group-global", name=DEFAULT_GLOBAL_SCENE_GROUP_NAME)
        custom = SceneGroupData(id="group-custom", name="游戏场景")
        project = ProjectData(name="demo", scene_groups=[custom, global_group, pending])

        ordered_groups = AutoStudioWindow._ordered_scene_pool_groups_for_tree(project)

        self.assertEqual([pending, global_group, custom], ordered_groups)

    def test_direct_stage_scene_creation_uses_pending_pool_group(self):
        stage = StageData(id="stage-1", name="搜房阶段")
        scene = SceneData(id="scene-1", name="临时识别")
        project = ProjectData(name="demo", stages=[stage])
        AutoStudioWindow._ensure_project_scene_pool(project)

        AutoStudioWindow._add_scene_to_project_pool(project, scene)
        AutoStudioWindow._add_scene_reference_to_stage(stage, scene)

        self.assertIs(stage.scenes[0], scene)
        self.assertIn(scene, project.scene_groups[0].scenes)
        self.assertEqual(DEFAULT_SCENE_GROUP_NAME, project.scene_groups[0].name)

    def test_pool_scene_rename_updates_stage_references(self):
        scene = SceneData(
            id="scene-1",
            name="退出弹窗",
            items=[ItemData(id="item-1", name="确认", item_type="area", rect=RectData(0, 0, 1, 1))],
        )
        stage = StageData(
            id="stage-1",
            name="搜房阶段",
            scenes=[scene],
            groups=[GroupData(name="默认", includes_all=True), GroupData(name="弹窗识别", items=[
                GroupItemRef("退出弹窗", "area", "确认"),
            ])],
        )
        scene_group = SceneGroupData(id="group-1", name="弹窗", scenes=[scene])
        project = ProjectData(name="demo", stages=[stage], scene_groups=[scene_group])
        AutoStudioWindow._rename_pool_scene_group(project, scene_group, "退出弹窗", "返回弹窗")

        self.assertEqual("返回弹窗", scene.name)
        self.assertIs(stage.scenes[0], scene)
        self.assertEqual([GroupItemRef("返回弹窗", "area", "确认")], stage.groups[1].items)

    def test_pool_scene_delete_removes_all_stage_references(self):
        scene_a = SceneData(id="scene-1", name="退出弹窗")
        scene_b = SceneData(id="scene-2", name="退出弹窗", image_width=200, image_height=100)
        keep_scene = SceneData(id="scene-3", name="大厅")
        stage_a = StageData(id="stage-a", name="搜房阶段", scenes=[scene_a, keep_scene])
        stage_b = StageData(id="stage-b", name="跑圈阶段", scenes=[scene_b])
        scene_group = SceneGroupData(id="group-1", name="弹窗", scenes=[scene_a, scene_b, keep_scene])
        project = ProjectData(name="demo", stages=[stage_a, stage_b], scene_groups=[scene_group])

        deleted = AutoStudioWindow._delete_pool_scene_group(project, scene_group, "退出弹窗")

        self.assertEqual([scene_a, scene_b], deleted)
        self.assertEqual([keep_scene], scene_group.scenes)
        self.assertEqual([keep_scene], stage_a.scenes)
        self.assertEqual([], stage_b.scenes)

    def test_stage_scene_delete_removes_reference_but_keeps_pool_scene(self):
        scene = SceneData(id="scene-1", name="退出弹窗")
        stage = StageData(id="stage-1", name="搜房阶段", scenes=[scene])
        scene_group = SceneGroupData(id="group-1", name="弹窗", scenes=[scene])
        project = ProjectData(name="demo", stages=[stage], scene_groups=[scene_group])

        removed = AutoStudioWindow._remove_stage_scene_group_reference(project, stage, "退出弹窗")

        self.assertEqual([scene], removed)
        self.assertEqual([], stage.scenes)
        self.assertEqual([scene], scene_group.scenes)


if __name__ == "__main__":
    unittest.main()
