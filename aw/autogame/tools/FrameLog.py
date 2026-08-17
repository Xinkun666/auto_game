import json
from enum import Enum


class FrameLogType(str, Enum):
    ALL = "总的"
    SYSTEM = "系统日志"
    TIME = "时间日志"
    LOGIC = "逻辑日志"
    UI_CONTROL = "UI和控点日志"
    OTHER = "其他日志"


DEFAULT_FRAME_LOG_TYPE = FrameLogType.LOGIC
FRAME_LOG_TRANSPORT_MARKER = "__AUTOGAME_FRAME_LOG__:"

_FRAME_LOG_TYPE_ALIASES = {
    "all": FrameLogType.ALL,
    "total": FrameLogType.ALL,
    "总的": FrameLogType.ALL,
    "system": FrameLogType.SYSTEM,
    "系统": FrameLogType.SYSTEM,
    "系统日志": FrameLogType.SYSTEM,
    "time": FrameLogType.TIME,
    "时间": FrameLogType.TIME,
    "时间日志": FrameLogType.TIME,
    "logic": FrameLogType.LOGIC,
    "逻辑": FrameLogType.LOGIC,
    "逻辑日志": FrameLogType.LOGIC,
    "control": FrameLogType.UI_CONTROL,
    "ui": FrameLogType.UI_CONTROL,
    "ui_control": FrameLogType.UI_CONTROL,
    "控制": FrameLogType.UI_CONTROL,
    "控点": FrameLogType.UI_CONTROL,
    "ui和控点日志": FrameLogType.UI_CONTROL,
    "other": FrameLogType.OTHER,
    "其他": FrameLogType.OTHER,
    "其他日志": FrameLogType.OTHER,
}


def normalize_frame_log_type(log_type) -> FrameLogType:
    if isinstance(log_type, FrameLogType):
        return log_type
    if log_type is None:
        return DEFAULT_FRAME_LOG_TYPE

    value = str(log_type).strip()
    normalized = _FRAME_LOG_TYPE_ALIASES.get(value.lower())
    if normalized is not None:
        return normalized
    raise ValueError(f"不支持的 frame_log 类型: {log_type}")


def build_frame_log_entry(message, log_type=DEFAULT_FRAME_LOG_TYPE) -> dict:
    return {
        "category": normalize_frame_log_type(log_type).value,
        "message": str(message or "").strip(),
    }


def encode_frame_log_transport(message, log_type=DEFAULT_FRAME_LOG_TYPE) -> str:
    payload = build_frame_log_entry(message, log_type)
    return FRAME_LOG_TRANSPORT_MARKER + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_frame_log_transport(line: str):
    text = str(line or "")
    marker_index = text.find(FRAME_LOG_TRANSPORT_MARKER)
    if marker_index < 0:
        return None

    payload_text = text[marker_index + len(FRAME_LOG_TRANSPORT_MARKER):].lstrip()
    try:
        payload, _ = json.JSONDecoder().raw_decode(payload_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    message = str(payload.get("message") or "").strip()
    if not message:
        return None
    try:
        category = normalize_frame_log_type(payload.get("category")).value
    except ValueError:
        category = DEFAULT_FRAME_LOG_TYPE.value
    return {
        "category": category,
        "message": message,
    }


def log_frame_event_on_change(
    worker,
    event_key,
    state,
    message,
    log_type=DEFAULT_FRAME_LOG_TYPE,
    repeat_after_seconds: float = 0.0,
):
    """Record a recurring runtime condition once per state transition.

    Test doubles and older workers retain the normal ``frame_log`` fallback.
    The real FrameWorker keeps the state cache for the lifetime of one run.
    """
    reporter = getattr(worker, "frame_log_state", None)
    if callable(reporter):
        return reporter(
            event_key,
            state,
            message,
            log_type=log_type,
            repeat_after_seconds=repeat_after_seconds,
        )
    return worker.frame_log(message)
