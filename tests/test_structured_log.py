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
        result="等待下一帧确认",
    )

    assert line == (
        "[AutoLog][逻辑日志] 观察现象=house_scene=near_wall | "
        "当前目标=R城房屋 house_1 | "
        "要做的事=转向绕墙 | "
        "结果=等待下一帧确认"
    )


def test_autogame_print_wraps_existing_message(capsys):
    autogame_print("[SceneSearch] 快速导航检测到卡住，启动避障")

    out = capsys.readouterr().out.strip()
    assert out == (
        "[AutoLog][逻辑日志] 观察现象=快速导航检测到卡住 | "
        "当前目标=搜房入门导航 | "
        "要做的事=启动避障 | "
        "结果=-"
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
