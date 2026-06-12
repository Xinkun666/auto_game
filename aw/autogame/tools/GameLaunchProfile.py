TEST_PROFILE_POWER = "power"
TEST_PROFILE_FUNCTION = "function"
DEFAULT_PUBG_GAME_PACKAGE = "com.tencent.tmgp.pubgmhd.hw"
DEFAULT_SP_PACKAGE = "com.huawei.hmsapp.hismartperf"


def normalize_test_profile(profile) -> str:
    value = str(profile or "").strip().lower()
    if value == TEST_PROFILE_FUNCTION:
        return TEST_PROFILE_FUNCTION
    return TEST_PROFILE_POWER


def should_use_sp_recording_for_profile(profile) -> bool:
    return normalize_test_profile(profile) != TEST_PROFILE_FUNCTION


def cleanup_packages_for_test_profile(
    profile,
    game_package: str = DEFAULT_PUBG_GAME_PACKAGE,
    sp_package: str = DEFAULT_SP_PACKAGE,
) -> tuple[str, ...]:
    packages = [str(game_package).strip()]
    if should_use_sp_recording_for_profile(profile):
        packages.append(str(sp_package).strip())
    return tuple(package for package in packages if package)
