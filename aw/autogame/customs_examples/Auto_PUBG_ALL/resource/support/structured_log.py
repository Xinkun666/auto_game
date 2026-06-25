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
    "RCitySearch": "R城搜房目标选择",
    "RCityRoute": "R城入门点导航",
    "RCityEntry": "R城进门流程",
    "RCityWater": "R城水区脱离",
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
    "Control": "设备控制",
    "HDC": "HDC控制命令",
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
BRACKET_FIELD_RE = re.compile(r"\[(?P<key>[^:\]]+):(?P<value>[^\]]*)\]")


OBSERVATION_KEYS = {"观察", "观察现象", "现象", "情况", "状态", "当前状态"}
TARGET_KEYS = {"目标", "当前目标"}
ACTION_KEYS = {"决策", "做的决策", "动作", "要做的事", "要做什么"}
METHOD_KEYS = {"控制", "具体控制", "怎么做", "实施", "method", "命令", "hdc"}
RESULT_KEYS = {"结果"}


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


def _assign_bracket_field(fields, key, value):
    key = key.strip()
    value = value.strip()
    if not value:
        return
    if key in OBSERVATION_KEYS:
        fields["observation"] = value
    elif key in TARGET_KEYS:
        fields["target"] = value
    elif key in ACTION_KEYS:
        fields["action"] = value
    elif key in METHOD_KEYS:
        fields["method"] = value
    elif key in RESULT_KEYS:
        fields["result"] = value


def _parse_bracket_fields(body):
    fields = {
        "observation": None,
        "target": None,
        "action": None,
        "method": None,
        "result": None,
    }
    for match in BRACKET_FIELD_RE.finditer(body):
        _assign_bracket_field(fields, match.group("key"), match.group("value"))
    return fields if any(fields.values()) else None


def infer_log_fields(message):
    text = _clean_field(message)
    target = None
    body = text

    match = PREFIX_RE.match(text)
    if match:
        prefix = match.group("prefix")
        body = match.group("body").strip() or text
        target = PREFIX_TARGETS.get(prefix, prefix)

    bracket_fields = _parse_bracket_fields(body)
    if bracket_fields:
        return (
            bracket_fields.get("observation") or body,
            bracket_fields.get("target") or target,
            bracket_fields.get("action"),
            bracket_fields.get("method"),
            bracket_fields.get("result"),
        )

    observation = body
    action = None
    method = None
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

    return observation, target, action, method, result


def format_log_line(
    observation,
    target=None,
    action=None,
    method=None,
    result=None,
    *,
    category=LOG_CATEGORY_LOGIC,
):
    category_text = _clean_category(category)
    return (
        f"{LOG_PREFIX}[{category_text}] "
        f"观察现象={_clean_field(observation)} | "
        f"当前目标={_clean_field(target)} | "
        f"做的决策={_clean_field(action)} | "
        f"具体控制={_clean_field(method)} | "
        f"结果={_clean_field(result)}"
    )


def log_step(
    observation,
    target=None,
    action=None,
    method=None,
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
            method=method,
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
    observation, target, action, method, result = infer_log_fields(message)
    builtins.print(
        format_log_line(
            observation,
            target=target,
            action=action,
            method=method,
            result=result,
            category=category,
        ),
        end=end,
        file=sys.stdout if file is None else file,
        flush=flush,
    )
