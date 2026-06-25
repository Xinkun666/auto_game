import os
from aw.autogame.tools.Utils import archive_run_artifacts
from xdevice.__main__ import main_process


DEFAULT_TESTCASE_LABEL = "testcases/pubg/pubg_full_flow/auto_pubg"
DEFAULT_PROJECT_CASE = "Auto_PUBG_ALL"
DEFAULT_TARGET_CASE = "auto_pubg"

# 这种方式是通过SP打开想要的游戏，进入游戏后再启动游戏自动化
# if __name__ == '__main__':
#
#     # main_process("run -l testcases/pubg_test/pubg_1") # 启动和平精英测试
#     # main_process("run -l testcases/sanjiaozhou/sanjiaozhou_changgongxigu") # 启动三角洲
#     main_process("run -l testcases/pubg/pubg_full_flow/auto_pubg")


# 这种方式是直接开启游戏自动化，需要人工手动点开游戏
def run_direct():
    project_case = DEFAULT_PROJECT_CASE  # 这是你在标注工具导出的自动化资源目录名
    target_case = DEFAULT_TARGET_CASE  # 这是你编写的自动化用例脚本名
    os.environ["TARGET_PROJECT_CASE"] = project_case
    os.environ["TARGET_GAME_CASE"] = target_case
    from aw.autogame.tools.GameAutomator import GameAutomator
    automator = GameAutomator(driver=None, logger=None)
    automator.start()


def run_testcase():
    main_process(f"run -l {DEFAULT_TESTCASE_LABEL}")


def should_archive_run_outputs() -> bool:
    source = os.environ.get("AUTOGAME_RUN_SOURCE", "").strip().lower()
    vis_mode = os.environ.get("AUTOGAME_VIS_MODE", "").strip().lower()
    return source == "launcher" or vis_mode == "launcher"


def get_launcher_run_metadata(run_mode: str) -> dict:
    metadata = {
        "run_mode": run_mode,
        "project_case": os.environ.get("TARGET_PROJECT_CASE", ""),
        "target_case": os.environ.get("TARGET_GAME_CASE", ""),
        "testcase_label": DEFAULT_TESTCASE_LABEL if run_mode == "testcase" else None,
    }
    batch_start_timestamp = os.environ.get("AUTOGAME_BATCH_START_TIMESTAMP", "").strip()
    run_start_timestamp = os.environ.get("AUTOGAME_RUN_START_TIMESTAMP", "").strip()
    if batch_start_timestamp:
        metadata["batch_start_timestamp"] = batch_start_timestamp
    if run_start_timestamp:
        metadata["run_start_timestamp"] = run_start_timestamp
    return metadata


if __name__ == '__main__':
    run_mode = os.environ.get("AUTOGAME_MAIN_MODE", "direct").strip().lower()

    try:
        if run_mode == "testcase":
            run_testcase()
        else:
            run_direct()
    finally:
        if should_archive_run_outputs():
            run_index_text = os.environ.get("AUTOGAME_RUN_INDEX", "").strip()
            try:
                run_index = int(run_index_text) if run_index_text else 1
            except ValueError:
                run_index = 1
            archive_run_artifacts(
                run_index=run_index,
                source=f"main:{run_mode}",
                extra_metadata=get_launcher_run_metadata(run_mode),
                reuse_existing=bool(os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()),
            )
