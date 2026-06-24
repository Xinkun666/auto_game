from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import (
    autogame_print,
    format_log_line,
    LOG_CATEGORY_LOGIC,
)


def test_format_log_line_uses_fixed_four_columns():
    line = format_log_line(
        observation="house_scene=near_wall",
        target="R城房屋 house_1",
        action="转向绕墙",
        method="向右滑视角并前推摇杆",
        result="等待下一帧确认",
    )

    assert line == (
        "[AutoLog][逻辑日志] 当前状态=house_scene=near_wall | "
        "当前目标=R城房屋 house_1 | "
        "要做什么=转向绕墙 | "
        "怎么做=向右滑视角并前推摇杆 | "
        "结果=等待下一帧确认"
    )


def test_autogame_print_wraps_existing_message(capsys):
    autogame_print("[SceneSearch] 快速导航检测到卡住，启动避障")

    out = capsys.readouterr().out.strip()
    assert out == (
        "[AutoLog][逻辑日志] 当前状态=快速导航检测到卡住 | "
        "当前目标=搜房入门导航 | "
        "要做什么=启动避障 | "
        "怎么做=按当前脱困策略执行 | "
        "结果=等待后续判断"
    )


def test_format_log_line_allows_explicit_category():
    line = format_log_line(
        observation="进入搜房计时",
        target="搜房阶段",
        action="记录阶段开始",
        result="timer_started",
        category=LOG_CATEGORY_LOGIC,
    )

    assert line.startswith("[AutoLog][逻辑日志] ")


def test_autogame_print_explicit_method_field(capsys):
    autogame_print(
        "[Flow] 搜房结束",
        target="全流程阶段",
        action="切换到跑图阶段",
        method="停止搜房自动前进，刷新画面，必要时先出房",
        result="等待阶段切换",
    )

    out = capsys.readouterr().out.strip()
    assert out == (
        "[AutoLog][逻辑日志] 当前状态=搜房结束 | "
        "当前目标=全流程阶段 | "
        "要做什么=切换到跑图阶段 | "
        "怎么做=停止搜房自动前进，刷新画面，必要时先出房 | "
        "结果=等待阶段切换"
    )


def test_bracket_state_decision_message_maps_to_five_fields(capsys):
    autogame_print(
        "[情况:首次出库] "
        "[状态: speed=0, loc=(1, 2), dir=90.0] "
        "[决策:forward(500ms)+brake_click]"
    )

    out = capsys.readouterr().out.strip()
    assert out == (
        "[AutoLog][逻辑日志] 当前状态=首次出库；speed=0, loc=(1, 2), dir=90.0 | "
        "当前目标=开车阶段 | "
        "要做什么=执行决策 forward(500ms)+brake_click | "
        "怎么做=根据当前状态执行驾驶控制 | "
        "结果=等待后续判断"
    )


def test_status_only_message_still_has_action_method_and_result(capsys):
    autogame_print("[Parachute] 状态已重置!")

    out = capsys.readouterr().out.strip()
    assert out == (
        "[AutoLog][逻辑日志] 当前状态=状态已重置! | "
        "当前目标=跳伞阶段 | "
        "要做什么=记录当前状态 | "
        "怎么做=输出当前阶段的状态数据供后续判断 | "
        "结果=已记录"
    )
