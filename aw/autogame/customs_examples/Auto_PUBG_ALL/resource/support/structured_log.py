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
BRACKET_FIELD_RE = re.compile(r"\[(?P<key>[^:\]]+):(?P<value>[^\]]*)\]")
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
    "NavAlign": "导航对齐",
    "NavBypass": "路线绕房避障",
    "Unstuck": "脱困避障",
    "AdaptiveMotion": "自适应动作参数",
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
    "Round": "全流程阶段",
    "Terminal": "结束阶段",
    "Popup": "关闭弹窗阶段",
    "StartGame": "开始游戏阶段",
    "Timer": "时间管理",
    "End": "结束阶段",
    "Resolution": "分辨率检测",
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


def _infer_method(action, target=None):
    action_text = _clean_field(action)
    target_text = _clean_field(target)
    if action_text == EMPTY_FIELD:
        return EMPTY_FIELD
    if "避障" in action_text or "脱困" in action_text:
        return "按当前脱困策略执行"
    if "跳跃" in action_text or "翻窗" in action_text:
        return "点击跳跃并按配置前推"
    if "出房" in action_text:
        return "按门窗识别和 house_scene 状态执行出房流程"
    if "进房" in action_text or "进门" in action_text:
        return "对齐入口后按进门流程执行"
    if "点击" in action_text:
        return "点击对应控件并刷新画面"
    if "切换" in action_text:
        return "调用阶段切换并同步状态"
    if "等待" in action_text:
        return "保持当前状态等待下一帧"
    if target_text != EMPTY_FIELD:
        return f"按{target_text}当前策略执行"
    return "按当前策略执行"


def _parse_bracket_state_decision(text):
    fields = {
        match.group("key").strip(): match.group("value").strip()
        for match in BRACKET_FIELD_RE.finditer(text)
    }
    if not {"情况", "状态", "决策"} & set(fields):
        return None

    situation = fields.get("情况")
    state = fields.get("状态")
    decision = fields.get("决策")
    status_parts = [part for part in (situation, state) if part]
    status = "；".join(status_parts) if status_parts else text
    action = f"执行决策 {decision}" if decision else None
    return (
        status,
        "开车阶段",
        action,
        "根据当前状态执行驾驶控制" if decision else _infer_method(action, "开车阶段"),
        "等待后续判断" if action else None,
    )


def infer_log_fields(message):
    text = _clean_field(message)
    target = None
    body = text

    bracket_fields = _parse_bracket_state_decision(text)
    if bracket_fields is not None:
        return bracket_fields

    match = PREFIX_RE.match(text)
    if match:
        prefix = match.group("prefix")
        body = match.group("body").strip() or text
        target = PREFIX_TARGETS.get(prefix, prefix)

    observation = body
    action = None
    result = None

    for separator in ("，", "。", "；"):
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

    if action is None:
        action = "记录当前状态"
        method = "输出当前阶段的状态数据供后续判断"
        result = result or "已记录"
    else:
        method = _infer_method(action, target)
        result = result or "等待后续判断"
    return observation, target, action, method, result


def format_log_line(observation, target=None, action=None, method=None, result=None, *, category=LOG_CATEGORY_LOGIC):
    category_text = _clean_category(category)
    return (
        f"{LOG_PREFIX}[{category_text}] "
        f"当前状态={_clean_field(observation)} | "
        f"当前目标={_clean_field(target)} | "
        f"要做什么={_clean_field(action)} | "
        f"怎么做={_clean_field(method)} | "
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


def autogame_print(
    *values,
    sep=" ",
    end="\n",
    file=None,
    flush=False,
    category=LOG_CATEGORY_LOGIC,
    target=None,
    action=None,
    method=None,
    result=None,
    status=None,
):
    message = sep.join(str(value) for value in values)
    if message.startswith(LOG_PREFIX):
        builtins.print(message, end=end, file=sys.stdout if file is None else file, flush=flush)
        return
    inferred_status, inferred_target, inferred_action, inferred_method, inferred_result = infer_log_fields(message)
    builtins.print(
        format_log_line(
            status if status is not None else inferred_status,
            target=target if target is not None else inferred_target,
            action=action if action is not None else inferred_action,
            method=method if method is not None else inferred_method,
            result=result if result is not None else inferred_result,
            category=category,
        ),
        end=end,
        file=sys.stdout if file is None else file,
        flush=flush,
    )
