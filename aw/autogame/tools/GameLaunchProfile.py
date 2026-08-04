import os


TEST_PROFILE_POWER = "power"
TEST_PROFILE_FUNCTION = "function"
TEST_PROFILE_MARATHON = "marathon"
TEST_TYPE_POWER = "功耗测试"
TEST_TYPE_FUNCTION = "功能测试"
TEST_TYPE_MARATHON = "马拉松测试"
DEFAULT_MARATHON_DURATION_MINUTES = 60.0
DEFAULT_PUBG_GAME_PACKAGE = "com.tencent.tmgp.pubgmhd.hw"
DEFAULT_SP_PACKAGE = "com.huawei.hmsapp.hismartperf"

_TEST_TYPE_BY_PROFILE = {
    TEST_PROFILE_POWER: TEST_TYPE_POWER,
    TEST_PROFILE_FUNCTION: TEST_TYPE_FUNCTION,
    TEST_PROFILE_MARATHON: TEST_TYPE_MARATHON,
}


def normalize_test_profile(profile) -> str:
    value = str(profile or "").strip().lower()
    if value == TEST_PROFILE_FUNCTION:
        return TEST_PROFILE_FUNCTION
    if value == TEST_PROFILE_MARATHON:
        return TEST_PROFILE_MARATHON
    return TEST_PROFILE_POWER


def resolve_test_type(profile) -> str:
    """Return the worker-facing test type for a Launcher test profile."""
    return _TEST_TYPE_BY_PROFILE[normalize_test_profile(profile)]


def should_use_sp_recording_for_profile(profile) -> bool:
    return normalize_test_profile(profile) != TEST_PROFILE_FUNCTION


def should_preserve_game_process() -> bool:
    """Return the launcher-wide process preservation policy."""
    return os.environ.get("AUTOGAME_PRESERVE_GAME_PROCESS", "0").strip() == "1"


def cleanup_packages_for_test_profile(
    profile,
    game_package: str = DEFAULT_PUBG_GAME_PACKAGE,
    sp_package: str = DEFAULT_SP_PACKAGE,
) -> tuple[str, ...]:
    if should_preserve_game_process():
        return ()

    packages = [str(game_package).strip()]
    if should_use_sp_recording_for_profile(profile):
        packages.append(str(sp_package).strip())
    return tuple(package for package in packages if package)
