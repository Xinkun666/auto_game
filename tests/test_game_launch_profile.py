import unittest

from aw.autogame.tools.GameLaunchProfile import (
    cleanup_packages_for_test_profile,
    normalize_test_profile,
    should_use_sp_recording_for_profile,
)


class GameLaunchProfileTests(unittest.TestCase):
    def test_function_profile_disables_sp_recording(self):
        self.assertFalse(should_use_sp_recording_for_profile("function"))
        self.assertFalse(should_use_sp_recording_for_profile(" FUNCTION "))

    def test_power_and_unknown_profiles_keep_sp_recording_enabled(self):
        self.assertTrue(should_use_sp_recording_for_profile("power"))
        self.assertTrue(should_use_sp_recording_for_profile(""))
        self.assertTrue(should_use_sp_recording_for_profile(None))

    def test_normalize_test_profile_defaults_to_power(self):
        self.assertEqual("power", normalize_test_profile(None))
        self.assertEqual("power", normalize_test_profile("unexpected"))
        self.assertEqual("function", normalize_test_profile("Function"))

    def test_cleanup_packages_omit_sp_package_for_function_profile(self):
        self.assertEqual(
            ("com.tencent.tmgp.pubgmhd.hw",),
            cleanup_packages_for_test_profile(
                "function",
                game_package="com.tencent.tmgp.pubgmhd.hw",
                sp_package="com.huawei.hmsapp.hismartperf",
            ),
        )
        self.assertEqual(
            ("com.tencent.tmgp.pubgmhd.hw", "com.huawei.hmsapp.hismartperf"),
            cleanup_packages_for_test_profile(
                "power",
                game_package="com.tencent.tmgp.pubgmhd.hw",
                sp_package="com.huawei.hmsapp.hismartperf",
            ),
        )


if __name__ == "__main__":
    unittest.main()
