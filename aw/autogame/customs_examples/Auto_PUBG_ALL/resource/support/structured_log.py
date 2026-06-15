import builtins
import re
import sys


LOG_PREFIX = "[AutoLog]"
LOG_CATEGORY_SYSTEM = "系统日志"
LOG_CATEGORY_TIME = "时间日志"
LOG_CATEGORY_LOGIC = "逻辑日志"
LOG_CATEGORY_UI = "UI和控点日志"
LOG_CATEGORY_OTHER = "其他日志"
LOG_CATEGORIES = {
    LOG_CATEGORY_SYSTEM,
    LOG_CATEGORY_TIME,
    LOG_CATEGORY_LOGIC,
    LOG_CATEGORY_UI,
    LOG_CATEGORY_OTHER,
}
EMPTY_FIELD = "-"
PREFIX_RE = re.compile(r"^\[(?P<prefix>[^\]]+)\]\s*(?P<body>.*)$")
PREFIX_TARGETS = {
    "Parachute": "跳伞阶段",
    "Searching": "搜房阶段",
    "搜房": "搜房阶段",
    "SceneSearch": "搜房入门导航",
    "SceneEntry": "进房流程",
    "SceneRotate": "室内旋转搜房",
    "SceneExit": "出房流程",
    "HouseExit": "出房流程",
    "Nav": "导航到目标",
    "NavBypass": "路线绕房避障",
    "Unstuck": "脱困避障",
    "Running": "跑图阶段",
    "Driving": "开车阶段",
    "Entry": "进门确认",
    "Interact": "门交互",
    "Scan": "门扫描",
    "Visual": "视觉对齐",
    "Finish": "阶段完成",
    "Jump": "跳跃翻越",
    "Smart": "智能导航",
    "TurnCalibration": "转向校准",
    "Flow": "全流程阶段",
}
ACTION_WORDS = (
    "启动",
    "开始",
    "执行",
    "切换",
    "进入",
    "准备",
    "尝试",
    "继续",
    "回退",
    "重置",
    "点击",
    "调整",
    "对齐",
    "绕",
    "前推",
    "后退",
    "等待",
    "保存",
    "停止",
    "跳过",
    "记录",
    "选择",
    "设置",
    "修正",
    "脱离",
    "导航",
    "寻找",
    "扫描",
    "补点",
    "转入",
    "保持",
)
RESULT_WORDS = (
    "成功",
    "失败",
    "完成",
    "结束",
    "已",
    "未",
    "仍",
    "无法",
    "丢失",
    "到达",
    "发现",
    "检测到",
    "判定",
    "确认",
)


def _clean_field(value):
    if value is None:
        return EMPTY_FIELD
    text = str(value)
    text = " ".join(text.splitlines())
    return text if text else EMPTY_FIELD


def _clean_category(category):
    text = _clean_field(category)
    return text if text in LOG_CATEGORIES else LOG_CATEGORY_OTHER


def _looks_like_action(text):
    return any(word in text for word in ACTION_WORDS)


def _looks_like_result(text):
    return any(word in text for word in RESULT_WORDS)


def infer_log_fields(message):
    text = _clean_field(message)
    target = None
    body = text

    match = PREFIX_RE.match(text)
    if match:
        prefix = match.group("prefix")
        body = match.group("body").strip() or text
        target = PREFIX_TARGETS.get(prefix, prefix)

    observation = body
    action = None
    result = None

    for separator in ("，", ",", "。", ";", "；"):
        if separator in body:
            first, second = body.split(separator, 1)
            observation = first.strip() or body
            remainder = second.strip()
            if remainder:
                if _looks_like_result(remainder) and not _looks_like_action(remainder):
                    result = remainder
                else:
                    action = remainder
            break

    return observation, target, action, result


def format_log_line(observation, target=None, action=None, result=None, *, category=LOG_CATEGORY_LOGIC):
    category_text = _clean_category(category)
    return (
        f"{LOG_PREFIX}[{category_text}] "
        f"观察现象={_clean_field(observation)} | "
        f"当前目标={_clean_field(target)} | "
        f"要做的事={_clean_field(action)} | "
        f"结果={_clean_field(result)}"
    )


def log_step(
    observation,
    target=None,
    action=None,
    result=None,
    *,
    category=LOG_CATEGORY_LOGIC,
    file=None,
    flush=False,
):
    builtins.print(
        format_log_line(
            observation,
            target=target,
            action=action,
            result=result,
            category=category,
        ),
        file=sys.stdout if file is None else file,
        flush=flush,
    )


def autogame_print(*values, sep=" ", end="\n", file=None, flush=False, category=LOG_CATEGORY_LOGIC):
    message = sep.join(str(value) for value in values)
    if message.startswith(LOG_PREFIX):
        builtins.print(message, end=end, file=sys.stdout if file is None else file, flush=flush)
        return
    observation, target, action, result = infer_log_fields(message)
    builtins.print(
        format_log_line(
            observation,
            target=target,
            action=action,
            result=result,
            category=category,
        ),
        end=end,
        file=sys.stdout if file is None else file,
        flush=flush,
    )
