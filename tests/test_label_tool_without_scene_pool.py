import unittest

from aw.autogame.tools import Label


class LabelToolWithoutScenePoolTests(unittest.TestCase):
    def test_project_model_is_stage_scoped_without_scene_pool(self):
        project = Label.ProjectData(name="demo")

        self.assertFalse(hasattr(Label, "SceneGroupData"))
        self.assertFalse(hasattr(project, "scene_groups"))


if __name__ == "__main__":
    unittest.main()
