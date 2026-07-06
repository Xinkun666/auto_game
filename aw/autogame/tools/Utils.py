import re
import os
import cv2
import time
import math
import json
import shutil
import subprocess
import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from aw.autogame.tools.ProcessUtils import hdc_command_args, hidden_subprocess_kwargs


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT_DIR
TEMP_DIR = APP_DIR / "aw" / "autogame" / "temp"


def _resolve_env_path(env_name: str, default_path: Path) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return default_path


def resolve_log_dir() -> Path:
    return _resolve_env_path("AUTOGAME_LOG_DIR", TEMP_DIR / "logs")


LOG_DIR = resolve_log_dir()


def resolve_process_temp_logs_dir() -> Path:
    return _resolve_env_path("AUTOGAME_PREVIEW_DIR", resolve_log_dir() / "process_temp_logs")


def resolve_process_save_frames_dir() -> Path:
    return _resolve_env_path("AUTOGAME_SAVE_FRAMES_DIR", resolve_log_dir() / "process_save_frames")


def resolve_tmp_frames_dir() -> Path:
    return _resolve_env_path("AUTOGAME_TMP_FRAMES_DIR", TEMP_DIR / "tmp_frames")


PROCESS_TEMP_LOGS_DIR = resolve_process_temp_logs_dir()
PROCESS_SAVE_FRAMES_DIR = resolve_process_save_frames_dir()


def write_image_unicode(path, image, params=None) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [] if params is None else params
    path_text = str(path)

    try:
        if cv2.imwrite(path_text, image, params):
            return True
    except Exception:
        pass

    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        return False

    with open(path, "wb") as image_file:
        image_file.write(encoded.tobytes())
    return True


def _safe_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_top_level_log_files(dst_dir: Path) -> list[str]:
    copied = []
    if not LOG_DIR.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(LOG_DIR.iterdir()):
        if not path.is_file():
            continue
        shutil.copy2(path, dst_dir / path.name)
        copied.append(path.name)
    return copied


def _copy_process_temp_logs(dst_dir: Path) -> list[str]:
    copied = []
    if not PROCESS_TEMP_LOGS_DIR.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROCESS_TEMP_LOGS_DIR.iterdir()):
        target = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)
            copied.append(path.name)
        elif path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
            copied.append(path.name + "/")
    return copied


def _copy_process_save_frames(dst_dir: Path) -> list[str]:
    copied = []
    if not PROCESS_SAVE_FRAMES_DIR.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROCESS_SAVE_FRAMES_DIR.iterdir()):
        target = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)
            copied.append(path.name)
        elif path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
            copied.append(path.name + "/")
    return copied


def _sanitize_archive_name_part(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "unknown"
    value = Path(value).stem
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or "unknown"


def _resolve_batch_start_timestamp(extra_metadata: Optional[dict]) -> str:
    extra_metadata = extra_metadata or {}
    value = str(extra_metadata.get("batch_start_timestamp") or "").strip()
    if value:
        return value
    return time.strftime("%Y%m%d%H%M%S")


def _resolve_run_timestamp(extra_metadata: Optional[dict]) -> str:
    extra_metadata = extra_metadata or {}
    value = str(extra_metadata.get("run_start_timestamp") or "").strip()
    if value:
        return value
    return time.strftime("%Y%m%d%H%M%S")


def _resolve_archive_dir_path(run_index: int, extra_metadata: Optional[dict] = None) -> Path:
    batch_timestamp = _resolve_batch_start_timestamp(extra_metadata)
    run_timestamp = _resolve_run_timestamp(extra_metadata)

    batch_dir = TEMP_DIR / f"game_cases_{batch_timestamp}"
    return batch_dir / f"第{run_index}次_{run_timestamp}"


def resolve_run_archive_dir(
    run_index: int,
    extra_metadata: Optional[dict] = None,
    create: bool = False,
) -> Path:
    archive_dir = _resolve_archive_dir_path(run_index, extra_metadata=extra_metadata)
    if create:
        archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _is_timed_special_value_for_frame_log(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    timing = value[1]
    if not isinstance(timing, (list, tuple)) or len(timing) != 1:
        return False
    try:
        float(timing[0])
    except (TypeError, ValueError):
        return False
    return True


def _sanitize_frame_log_value(value):
    if _is_timed_special_value_for_frame_log(value):
        return [_sanitize_frame_log_value(value[0]), [value[1][0]]]
    if isinstance(value, dict):
        sanitized = {}
        for key, child in value.items():
            if key == "__visualizations__":
                sanitized[key] = f"{len(child) if isinstance(child, list) else 0} visualizations"
            else:
                sanitized[key] = _sanitize_frame_log_value(child)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_frame_log_value(child) for child in value]
    return value


def sanitize_frame_info_for_json(info):
    if isinstance(info, dict):
        safe_info = {}
        for key, value in info.items():
            safe_info[str(key)] = str(_sanitize_frame_log_value(value))
        return safe_info
    return str(info)


def _non_empty_info_keys(info):
    if not isinstance(info, dict):
        return []
    keys = []
    for key, value in info.items():
        text = str(value).strip()
        if not text or text in {"None", "[]", "{}", "False"}:
            continue
        keys.append(str(key))
    return keys


def _normalize_frame_log_entries(entries):
    if not isinstance(entries, list):
        return []
    normalized = []
    allowed_keys = {
        "seq",
        "created_at",
        "category",
        "message",
        "raw_message",
        "observation",
        "target",
        "action",
        "method",
        "result",
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append({str(key): value for key, value in entry.items() if key in allowed_keys})
    return normalized


def _normalize_frame_decision(runtime_logs):
    if not isinstance(runtime_logs, dict):
        return {}
    decision = runtime_logs.get("frame_decision")
    if not isinstance(decision, dict):
        return {}
    allowed_keys = {
        "observation",
        "target",
        "decision",
        "action",
        "method",
        "result",
        "next_action",
        "frame_log",
        "observed_infos",
        "control_actions",
    }
    return {str(key): value for key, value in decision.items() if key in allowed_keys}


def _semantic_text(value, default=""):
    text = str(value or "").strip()
    return text if text and text != "-" else default


def _infer_house_scene_state(value):
    text = str(value or "").strip().strip("[]()")
    mapping = {
        "0": "在屋内",
        "1": "在屋外",
        "2": "在楼顶/房顶",
        "3": "靠近门",
        "4": "靠近墙/贴墙/疑似撞墙",
        "HOUSE_INDOOR": "在屋内",
        "HOUSE_OUTDOOR": "在屋外",
        "HOUSE_ROOFTOP": "在楼顶/房顶",
        "HOUSE_NEAR_DOOR": "靠近门",
        "HOUSE_NEAR_WALL": "靠近墙/贴墙/疑似撞墙",
        "NEAR_DOOR": "靠近门",
        "NEAR_WALL": "靠近墙/贴墙/疑似撞墙",
    }
    return mapping.get(text)


def _infer_scene_state(safe_info, frame_decision, code_branch):
    if isinstance(safe_info, dict):
        house_scene_state = _infer_house_scene_state(safe_info.get("house_scene"))
        if house_scene_state:
            return house_scene_state
        if safe_info.get("上浮"):
            return "水中/水边"
        if safe_info.get("驾驶") or safe_info.get("喇叭"):
            return "车辆附近/车内"

    text_pool = " ".join(
        _semantic_text(value)
        for value in (
            frame_decision.get("target") if isinstance(frame_decision, dict) else "",
            frame_decision.get("observation") if isinstance(frame_decision, dict) else "",
            frame_decision.get("decision") if isinstance(frame_decision, dict) else "",
            code_branch.get("target") if isinstance(code_branch, dict) else "",
            code_branch.get("observation") if isinstance(code_branch, dict) else "",
        )
    )
    if "不可通行" in text_pool or "黑区" in text_pool:
        return "黑区中/不可通行区域"
    if "卡住" in text_pool:
        return "卡住/避障中"
    if "跑图" in text_pool or "导航" in text_pool:
        return "正常导航"
    if "跳伞" in text_pool:
        return "跳伞监控中"
    return "未明确子状态"


def _critical_frame_values(safe_info, info_keys):
    if not isinstance(safe_info, dict):
        return {}
    preferred = [
        "location",
        "direction",
        "forward_scene",
        "house_scene",
        "road_scene",
        "vehicle_scene",
        "上浮",
        "跳跃",
        "开门",
        "关门",
        "驾驶",
        "喇叭",
        "关闭活动",
        "关闭",
        "提示",
    ]
    critical = {}
    for key in preferred:
        if key in safe_info:
            critical[key] = safe_info[key]
    for key in info_keys:
        if key not in critical and len(critical) < 24:
            critical[key] = safe_info.get(key)
    return critical


def _normalize_semantic_actions(actions, default_reason=""):
    if not isinstance(actions, list):
        return []
    normalized = []
    for action in actions[-12:]:
        if not isinstance(action, dict):
            continue
        params = action.get("params")
        if not isinstance(params, dict):
            params = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
        control_trace = action.get("control_trace") if isinstance(action.get("control_trace"), dict) else {}
        merged_params = {str(key): str(value) for key, value in params.items()}
        for key in ("x_bias", "y_bias", "dura", "wait", "duration_ms", "finger_id", "backend"):
            if key in control_trace and key not in merged_params:
                merged_params[key] = str(control_trace[key])
        target = _semantic_text(action.get("target"))
        if not target and isinstance(action.get("args"), list) and action.get("args"):
            target = _semantic_text(action.get("args")[0])
        control_point = _semantic_text(action.get("control_point") or control_trace.get("control_point"), target)
        start_pos = _semantic_text(action.get("start_pos") or control_trace.get("start_pos"))
        end_pos = _semantic_text(action.get("end_pos") or control_trace.get("end_pos"))
        actual_pos = _semantic_text(action.get("actual_pos") or control_trace.get("actual_pos") or start_pos)
        normalized.append({
            "action": _semantic_text(action.get("action"), _semantic_text(action.get("name"), "unknown_action")),
            "name": _semantic_text(action.get("name"), "unknown_action"),
            "target": target,
            "control_point": control_point,
            "resolved_label": _semantic_text(action.get("resolved_label") or control_trace.get("resolved_label")),
            "actual_pos": actual_pos,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "description": _semantic_text(action.get("description")),
            "params": merged_params,
            "duration": _semantic_text(action.get("duration") or merged_params.get("dura") or merged_params.get("duration_ms") or merged_params.get("duration")),
            "reason": _semantic_text(action.get("reason"), default_reason),
            "control_trace": {str(key): str(value) for key, value in control_trace.items()},
        })
    return normalized


def _build_semantic_frame_log(stage, group_name, safe_info, info_keys, frame_decision, code_branch, next_action, index):
    frame_decision = frame_decision if isinstance(frame_decision, dict) else {}
    code_branch = code_branch if isinstance(code_branch, dict) else {}
    observed_infos = frame_decision.get("observed_infos") if isinstance(frame_decision.get("observed_infos"), list) else []
    scene_state = _infer_scene_state(safe_info, frame_decision, code_branch)
    perception_summary = "看到 " + ", ".join(info_keys) if info_keys else "未识别到有效 info"
    judgment_reason = (
        _semantic_text(frame_decision.get("observation"))
        or _semantic_text(code_branch.get("observation"))
        or perception_summary
    )
    judgment_decision = (
        _semantic_text(frame_decision.get("decision"))
        or _semantic_text(code_branch.get("action"))
        or _semantic_text(next_action, "等待下一轮逻辑判断")
    )
    branch_name = (
        _semantic_text(code_branch.get("target"))
        or _semantic_text(frame_decision.get("target"))
        or _semantic_text(stage, "未知分支")
    )
    actions = _normalize_semantic_actions(
        frame_decision.get("control_actions"),
        default_reason=judgment_decision or judgment_reason,
    )

    return {
        "frame_log": _semantic_text(frame_decision.get("frame_log")),
        "current_stage": {
            "stage": stage or "",
            "group": group_name or "",
            "scene_state": scene_state,
            "frame_index": index,
        },
        "perception": {
            "summary": perception_summary,
            "info_keys": info_keys,
            "critical_values": _critical_frame_values(safe_info, info_keys),
            "observed_infos": observed_infos,
        },
        "judgment": {
            "reason": judgment_reason,
            "decision": judgment_decision,
            "evidence": _semantic_text(frame_decision.get("method") or code_branch.get("method")),
            "result_expectation": _semantic_text(frame_decision.get("result") or code_branch.get("result")),
        },
        "branch": {
            "name": branch_name,
            "observation": _semantic_text(code_branch.get("observation") or frame_decision.get("observation")),
            "action": _semantic_text(code_branch.get("action") or frame_decision.get("action")),
            "method": _semantic_text(code_branch.get("method") or frame_decision.get("method")),
            "result": _semantic_text(code_branch.get("result") or frame_decision.get("result")),
        },
        "actions": actions,
    }


def _select_frame_code_branch(runtime_logs):
    if not isinstance(runtime_logs, dict):
        return {}
    frame_decision = _normalize_frame_decision(runtime_logs)
    if frame_decision:
        return {
            "target": frame_decision.get("target", ""),
            "observation": frame_decision.get("observation", ""),
            "action": frame_decision.get("action") or frame_decision.get("decision", ""),
            "method": frame_decision.get("method", ""),
            "result": frame_decision.get("result", ""),
        }
    branch = runtime_logs.get("current_branch")
    if isinstance(branch, dict) and branch:
        return dict(branch)
    logic_logs = _normalize_frame_log_entries(runtime_logs.get("logic_logs"))
    if logic_logs:
        return dict(logic_logs[-1])
    recent_logs = _normalize_frame_log_entries(runtime_logs.get("recent_logs"))
    return dict(recent_logs[-1]) if recent_logs else {}


def _select_next_action(runtime_logs, code_branch):
    if isinstance(runtime_logs, dict):
        frame_decision = _normalize_frame_decision(runtime_logs)
        if frame_decision:
            for key in ("next_action", "action", "decision"):
                value = str(frame_decision.get(key) or "").strip()
                if value and value != "-":
                    return value
        explicit = str(runtime_logs.get("next_action") or "").strip()
        if explicit and explicit != "-":
            return explicit
    if isinstance(code_branch, dict):
        for key in ("action", "result", "observation"):
            value = str(code_branch.get(key) or "").strip()
            if value and value != "-":
                return value
    return ""


def _build_frame_summary(stage, info_keys, code_branch, next_action):
    stage_text = str(stage or "未知阶段")
    if info_keys:
        seen_text = "看到 " + ", ".join(info_keys)
    else:
        seen_text = "未识别到有效 info"

    branch_target = ""
    branch_observation = ""
    if isinstance(code_branch, dict):
        branch_target = str(code_branch.get("target") or "").strip()
        branch_observation = str(code_branch.get("observation") or "").strip()

    branch_text = branch_target if branch_target and branch_target != "-" else "暂无明确代码分支"
    observation_text = branch_observation if branch_observation and branch_observation != "-" else seen_text
    action_text = next_action or "等待下一轮逻辑判断"
    return f"当前帧处于{stage_text}，{seen_text}；判断：{observation_text}；代码分支：{branch_text}；决策：{action_text}"


def build_frame_log_payload(stage, info, index, runtime_logs=None, group_name=None, frame_name=None):
    safe_info = sanitize_frame_info_for_json(info)
    info_keys = _non_empty_info_keys(safe_info)
    runtime_logs = runtime_logs if isinstance(runtime_logs, dict) else {}
    frame_decision = _normalize_frame_decision(runtime_logs)
    code_branch = _select_frame_code_branch(runtime_logs)
    next_action = _select_next_action(runtime_logs, code_branch)
    semantic_log = _build_semantic_frame_log(
        stage,
        group_name,
        safe_info,
        info_keys,
        frame_decision,
        code_branch,
        next_action,
        index,
    )
    frame_name = frame_name or f"frame_{int(index):05d}.jpg"

    return {
        "schema_version": 2,
        "index": index,
        "frame": {
            "index": index,
            "image": frame_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "stage": {
            "name": stage or "",
            "group": group_name or "",
        },
        "phase": {
            "stage": stage or "",
            "group": group_name or "",
        },
        "info": safe_info,
        "seen": {
            "summary": "看到 " + ", ".join(info_keys) if info_keys else "未识别到有效 info",
            "info_keys": info_keys,
        },
        "time_logs": _normalize_frame_log_entries(runtime_logs.get("time_logs")),
        "logic_logs": _normalize_frame_log_entries(runtime_logs.get("logic_logs")),
        "recent_logs": _normalize_frame_log_entries(runtime_logs.get("recent_logs")),
        "decision": frame_decision,
        "code_branch": code_branch,
        "next_action": next_action,
        "semantic_log": semantic_log,
        "frame_summary": _build_frame_summary(stage, info_keys, code_branch, next_action),
    }


def _build_archive_dir(
    run_index: int,
    extra_metadata: Optional[dict] = None,
    reuse_existing: bool = False,
) -> Path:
    base_dir = _resolve_archive_dir_path(run_index, extra_metadata=extra_metadata)
    batch_dir = base_dir.parent
    batch_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    archive_dir = base_dir
    suffix = 1
    while archive_dir.exists():
        archive_dir = batch_dir / f"{base_dir.name}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _frame_sort_key(path: Path):
    match = re.search(r"frame_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return path.name


def _read_image_quietly(path: Path) -> Optional[np.ndarray]:
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.stat().st_size <= 0:
            return None
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _create_preview_video(src_dir: Path, output_path: Path, fps: int = 10, pattern: str = "frame_*.jpg") -> Optional[str]:
    frame_paths = sorted(src_dir.glob(pattern), key=_frame_sort_key)
    if not frame_paths:
        return None

    first_frame = None
    for frame_path in frame_paths:
        first_frame = _read_image_quietly(frame_path)
        if first_frame is not None:
            break
    if first_frame is None:
        return None

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    if not writer.isOpened():
        return None

    try:
        for frame_path in frame_paths:
            frame = _read_image_quietly(frame_path)
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    if output_path.exists():
        return output_path.name
    return None


def archive_run_artifacts(
    run_index: int,
    source: str,
    extra_text_files: Optional[dict[str, str]] = None,
    extra_log_files: Optional[dict[str, str]] = None,
    extra_metadata: Optional[dict] = None,
    reuse_existing: bool = False,
    generate_preview_video: bool = False,
) -> Path:
    archive_dir = _build_archive_dir(
        run_index,
        extra_metadata=extra_metadata,
        reuse_existing=reuse_existing,
    )
    log_archive_dir = archive_dir / "logs"
    process_archive_dir = archive_dir / "process_temp_logs"
    save_frame_archive_dir = archive_dir / "process_save_frames"

    copied_log_files = _copy_top_level_log_files(log_archive_dir)
    copied_process_files = _copy_process_temp_logs(process_archive_dir)
    copied_process_save_frames = _copy_process_save_frames(save_frame_archive_dir)
    video_source = None
    video_file = None
    if generate_preview_video:
        video_file = _create_preview_video(
            process_archive_dir,
            archive_dir / "preview_10fps.mp4",
            fps=10,
        )
        if video_file:
            video_source = "process_temp_logs"
        elif copied_process_save_frames:
            video_file = _create_preview_video(
                save_frame_archive_dir,
                archive_dir / "preview_10fps.mp4",
                fps=10,
                pattern="*.jpg",
            )
            if video_file:
                video_source = "process_save_frames"

    if extra_text_files:
        for name, content in extra_text_files.items():
            if not name:
                continue
            _safe_write_text(log_archive_dir / name, content)

    copied_extra_log_files = []
    if extra_log_files:
        log_archive_dir.mkdir(parents=True, exist_ok=True)
        for archive_name, source_path in extra_log_files.items():
            if not archive_name or not source_path:
                continue
            src = Path(source_path)
            if not src.exists() or not src.is_file():
                continue
            safe_name = _sanitize_archive_name_part(archive_name) + src.suffix
            target = log_archive_dir / safe_name
            shutil.copy2(src, target)
            copied_extra_log_files.append(safe_name)

    metadata = {
        "source": source,
        "run_index": run_index,
        "archive_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archive_dir": str(archive_dir),
        "batch_dir": str(archive_dir.parent),
        "copied_log_files": copied_log_files,
        "copied_process_temp_logs": copied_process_files,
        "copied_process_save_frames": copied_process_save_frames,
        "copied_extra_log_files": copied_extra_log_files,
        "generate_preview_video": bool(generate_preview_video),
        "preview_video": video_file,
        "preview_video_source": video_source,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    _safe_write_text(
        archive_dir / "archive_info.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    return archive_dir

TEMPLATE_MATCH_MODES = ("gray", "rgb", "hsv", "edge", "clahe_gray")


def find_template_center_multiscale(target_img, template_input, threshold=0.7, match_mode="gray"):
    def _normalize_match_mode(mode):
        mode = str(mode or "gray").strip().lower()
        return mode if mode in TEMPLATE_MATCH_MODES else "gray"

    def _to_gray(img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _apply_clahe(gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _edge_image(img):
        gray = _apply_clahe(_to_gray(img))
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, 5, 15)

    def _prepare_match_image(img_data, mode):
        if isinstance(img_data, str):
            try:
                arr = np.fromfile(img_data, dtype=np.uint8)
                flag = cv2.IMREAD_GRAYSCALE if mode == "gray" else cv2.IMREAD_COLOR
                img = cv2.imdecode(arr, flag)
            except Exception:
                return None
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            return None

        if img is None:
            return None

        if mode == "gray":
            return _to_gray(img)
        if mode == "clahe_gray":
            return _apply_clahe(_to_gray(img))
        if mode == "edge":
            return _edge_image(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if mode == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mode == "hsv":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _match_template(search_img, tpl_img, mode):
        method = cv2.TM_CCORR_NORMED if mode == "edge" else cv2.TM_CCOEFF_NORMED
        if len(search_img.shape) == 2:
            return cv2.matchTemplate(search_img, tpl_img, method)

        channel_scores = []
        for channel_idx in range(search_img.shape[2]):
            channel_scores.append(
                cv2.matchTemplate(
                    search_img[:, :, channel_idx],
                    tpl_img[:, :, channel_idx],
                    method,
                )
            )
        return np.mean(channel_scores, axis=0)

    match_mode = _normalize_match_mode(match_mode)
    prepared_target = _prepare_match_image(target_img, match_mode)
    prepared_template = _prepare_match_image(template_input, match_mode)

    if prepared_target is None or prepared_template is None:
        return None

    tH, tW = prepared_template.shape[:2]
    best_match = None

    # 缩放范围：由于主缩放已在外部完成，这里仅作微调
    scales = np.linspace(0.8, 1.2, 10)

    for scale in scales:
        new_w = int(tW * scale)
        new_h = int(tH * scale)
        if new_w < 10 or new_h < 10 or new_h > prepared_target.shape[0] or new_w > prepared_target.shape[1]:
            continue

        resized_tpl = cv2.resize(prepared_template, (new_w, new_h))
        res = _match_template(prepared_target, resized_tpl, match_mode)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best_match is None or max_val > best_match[0]:
            cX = max_loc[0] + new_w // 2
            cY = max_loc[1] + new_h // 2
            best_match = (max_val, (cX, cY))

    if best_match and best_match[0] >= threshold:
        return best_match[1]
    return None

def run_shell(cmd: str, r = False):
    try:
        hdc_args = hdc_command_args(cmd)
        if r:
            result = subprocess.run(
                hdc_args or cmd,
                shell=hdc_args is None,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                **hidden_subprocess_kwargs(),
            )
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            if result.returncode != 0 and not output:
                print(f"命令执行失败: {cmd}\nreturncode={result.returncode}")
                return None
            return output or None
        if hdc_args is not None:
            subprocess.run(
                hdc_args,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **hidden_subprocess_kwargs(),
            )
            return None
        subprocess.run(cmd, shell=True, check=True, **hidden_subprocess_kwargs())
    except Exception as e:
        print(f"命令执行失败: {cmd}\n{e}")
        if r:
            return None


def _parse_screen_resolution(screen_info: str):
    if not screen_info:
        return None

    patterns = (
        r'activeMode:\s*(\d+)\s*x\s*(\d+)',
        r'render\s+resolution\s*=\s*(\d+)\s*x\s*(\d+)',
        r'physical\s+resolution\s*=\s*(\d+)\s*x\s*(\d+)',
        r'supportedMode\[\d+\]:\s*(\d+)\s*x\s*(\d+)',
    )
    for pattern in patterns:
        match = re.search(pattern, screen_info, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


_AUTO_ROTATION = object()


def get_resolution(r = True, rotation=_AUTO_ROTATION):
    resolution_mode = run_shell('hdc shell hidumper -s RenderService -a screen', r)
    resolution = _parse_screen_resolution(resolution_mode)
    if resolution:
        width, height = resolution
        if rotation is _AUTO_ROTATION:
            rotation = normalize_rotation(get_display_rotation())
        else:
            rotation = normalize_rotation(rotation)
        if rotation is None:
            return max(width, height), min(width, height)
        return normalize_resolution_by_rotation(width, height, rotation)

    print('未能获取分辨率信息!')
    if resolution_mode:
        print(f"[Resolution] RenderService 输出片段: {resolution_mode[:500]}")
    return None, None


def set_runtime_screen_resolution_env(width=None, height=None):
    if width and height:
        os.environ["AUTOGAME_SCREEN_WIDTH"] = str(int(width))
        os.environ["AUTOGAME_SCREEN_HEIGHT"] = str(int(height))


def wait_for_landscape_resolution_stable(timeout=12, stable_rounds=3, interval=0.5):
    """
    Wait until the device reports a stable landscape resolution, then return
    the real display resolution normalized to landscape orientation.
    """
    deadline = time.time() + timeout
    last_state = None
    stable_count = 0
    latest_resolution = (None, None)

    while time.time() < deadline:
        rotation = normalize_rotation(get_display_rotation())
        width, height = get_resolution(rotation=rotation)
        latest_resolution = (width, height)
        state = (rotation, width, height)

        if width and height and width > height and (rotation in (90, 270) or rotation is None):
            if state == last_state:
                stable_count += 1
            else:
                stable_count = 1
                last_state = state

            if stable_count >= stable_rounds:
                set_runtime_screen_resolution_env(width, height)
                print(f"[Resolution] 横屏分辨率已稳定: {width}x{height}, rotation={rotation}")
                return width, height
        else:
            last_state = state
            stable_count = 0

        time.sleep(interval)

    width, height = latest_resolution
    if width and height:
        set_runtime_screen_resolution_env(width, height)
        print(f"[Resolution] 等待横屏稳定超时，使用当前分辨率: {width}x{height}")
    return width, height

def get_wh():
    resolution = get_resolution()
    assert resolution[0] is not None, '分辨率获取失败'
    if resolution[0] > resolution[1]:
        w_h = resolution[0] / resolution[1]
    else:
        w_h = resolution[1] / resolution[0]
    with open(r'aw\autogame\config\config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        width = config["width"]
        height = int(width * w_h)
        return width, height

def normalize_rotation(rotation):
    if rotation is None:
        return None
    try:
        value = int(rotation)
    except (TypeError, ValueError):
        return None
    if value in (0, 90, 180, 270):
        return value
    mapping = {1: 90, 2: 180, 3: 270}
    return mapping.get(value)

def normalize_resolution_by_rotation(width, height, rotation):
    if width is None or height is None:
        return width, height

    width = int(width)
    height = int(height)
    rotation = normalize_rotation(rotation)

    if rotation in (90, 270):
        return (max(width, height), min(width, height))
    if rotation in (0, 180):
        return (min(width, height), max(width, height))
    return width, height

def infer_landscape_rotation(width, height, default=90):
    if width is None or height is None:
        return default
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return default
    if width > height:
        return default
    return 0

def get_natural_resolution_by_rotation(width, height, rotation):
    if width is None or height is None:
        return width, height

    width = int(width)
    height = int(height)
    rotation = normalize_rotation(rotation)

    if rotation in (90, 270):
        return height, width
    return width, height

def is_landscape(width, height):
    return width >= height

def _clamp_point(x, y, width, height):
    x = int(round(x))
    y = int(round(y))
    x = min(max(x, 0), max(int(width) - 1, 0))
    y = min(max(y, 0), max(int(height) - 1, 0))
    return x, y

def _round_point(x, y):
    return int(round(x)), int(round(y))

def scale_point(x, y, src_width, src_height, dst_width, dst_height):
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return int(round(x)), int(round(y))
    dst_x = float(x) * float(dst_width) / float(src_width)
    dst_y = float(y) * float(dst_height) / float(src_height)
    return _clamp_point(dst_x, dst_y, dst_width, dst_height)

def convert_display_point_by_rotation(x, y, screen_width, screen_height, current_rotation):
    """
    将“当前画面左上角坐标系”中的点，转换为设备物理屏幕左上角坐标系。

    约定:
    - x, y 是已经映射到 screen_size 下的画面坐标
    - current_rotation 表示设备当前屏幕旋转角
    - rotation=90/270 时需要交换宽高轴来做映射
    """
    if screen_width <= 0 or screen_height <= 0:
        return int(round(x)), int(round(y))

    current_rotation = normalize_rotation(current_rotation)
    if current_rotation == 90:
        dst_x = y
        dst_y = screen_width - x
        return _round_point(dst_x, dst_y)

    if current_rotation == 180:
        dst_x = screen_width - x
        dst_y = screen_height - y
        return _round_point(dst_x, dst_y)

    if current_rotation == 270:
        dst_x = screen_height - y
        dst_y = x
        return _round_point(dst_x, dst_y)

    return _round_point(x, y)

def convert_scene_point_by_current_rotation(x, y, scene_width, scene_height,
                                            current_width, current_height, current_rotation):
    if (
        scene_width <= 0 or scene_height <= 0
        or current_width <= 0 or current_height <= 0
    ):
        return int(round(x)), int(round(y))
    scaled_x, scaled_y = scale_point(
        x, y,
        scene_width, scene_height,
        current_width, current_height,
    )
    return convert_display_point_by_rotation(
        scaled_x, scaled_y,
        current_width, current_height,
        current_rotation,
    )

def get_runtime_screen_resolution():
    env_w = os.environ.get("AUTOGAME_SCREEN_WIDTH")
    env_h = os.environ.get("AUTOGAME_SCREEN_HEIGHT")
    if env_w and env_h:
        try:
            return int(env_w), int(env_h)
        except ValueError:
            pass
    screen_w, screen_h = get_resolution()
    if screen_w and screen_h:
        return int(screen_w), int(screen_h)
    return None, None


def select_scene_resolution(scene_content, screen_width=None, screen_height=None):
    if not isinstance(scene_content, dict) or "resolutions" not in scene_content:
        return scene_content

    resolutions = scene_content.get("resolutions", {})
    if not isinstance(resolutions, dict) or not resolutions:
        return {}

    if screen_width and screen_height:
        exact_key = f"{int(screen_width)}_{int(screen_height)}"
        if exact_key in resolutions:
            return resolutions[exact_key]
        for value in resolutions.values():
            if (
                isinstance(value, dict)
                and int(value.get("width") or 0) == int(screen_width)
                and int(value.get("height") or 0) == int(screen_height)
            ):
                return value

    return next(iter(resolutions.values()))


def extract_absolute_points(stage_info):
    """
    将游戏各阶段场景中的控点（Points）从百分比归一化坐标转换为屏幕绝对像素坐标。

    在自动化标注过程中，为了适配不同分辨率的屏幕，我们通常使用 0.0 到 1.0 之间的浮点数（归一化坐标）
    来表示按钮的位置。但在实际执行点击操作（如使用 hdc 或 uinput）时，系统需要具体的像素坐标。
    该函数会自动遍历整个配置表，读取每个场景的原始宽高，计算出每个按钮中心点的像素位置，
    并生成一个方便直接查询的扁平化字典。

    参数:
        stage_info (dict): 包含游戏阶段、场景、长宽信息及控点矩形区域的原始嵌套字典。
                           要求每个场景必须包含 'width' 和 'height' 键。

    返回:
        dict: 转换后的绝对坐标字典。
              键格式为: '阶段名_控点名' (str)
              值格式为:
              {
                  "pos": (x, y),
                  "scene_width": int,
                  "scene_height": int,
              }
    """
    absolute_points = {}
    screen_w, screen_h = get_runtime_screen_resolution()

    for stage_name, stage_content in stage_info.items():
        scenes = stage_content.get('scenes', {})

        for scene_name, scene_content in scenes.items():
            scene_content = select_scene_resolution(scene_content, screen_w, screen_h)
            # 获取当前场景的画布大小（标注时的原始分辨率）
            img_w = scene_content.get('width', 1)
            img_h = scene_content.get('height', 1)
            points = scene_content.get('points', {})

            for point_name, point_content in points.items():
                # 获取归一化矩形区域 [x_start, y_start, x_end, y_end]
                rect = point_content.get('rect', [0, 0, 0, 0])

                # 计算中心点的归一化位置
                norm_x = (rect[0] + rect[2]) / 2
                norm_y = (rect[1] + rect[3]) / 2

                # 核心转换：归一化比例 * 原始分辨率 = 绝对像素坐标
                abs_x = int(norm_x * img_w)
                abs_y = int(norm_y * img_h)

                # 扁平化存储：使用 阶段_控点 作为唯一索引，方便 Controller 直接调用
                key = f"{stage_name}_{point_name}"
                absolute_points[key] = {
                    "pos": (abs_x, abs_y),
                    "norm_pos": (float(norm_x), float(norm_y)),
                    "rect": list(rect),
                    "scene_width": int(img_w),
                    "scene_height": int(img_h),
                }
                if "anchor" in point_content:
                    absolute_points[key]["anchor"] = point_content.get("anchor")
                    absolute_points[key]["offset"] = dict(point_content.get("offset", {}))
                    absolute_points[key]["size"] = dict(point_content.get("size", {}))

    return absolute_points

def get_formatted_time():
    # 1. 获取当前本地时间
    now = datetime.now()

    # 2. 使用 strftime 格式化，%f 代表微秒（6位）
    # 格式含义： %m月, %d日, %H时, %M分, %S秒, %f微秒
    time_str = now.strftime("%m-%d %H:%M:%S.%f")

    # 3. 因为你要的是 3 位毫秒，所以截取掉末尾的 3 位微秒
    # 原字符串：01-22 09:31:44.732000 -> 截取后：01-22 09:31:44.732
    final_time = time_str[:-3]

    return final_time

def analyze_fps_value(log_path, time_txt_path):
    """
    在 log.txt 中寻找每个时间区间内【第一个】出现的 'targetFps = 数值'。
    返回结果将阶段名中的 Time 替换为 Fps。
    """
    # 1. 解析 time.txt
    time_data = {}
    time_keys = []
    if not os.path.exists(time_txt_path):
        print(f" [错误] 找不到文件: {time_txt_path}")
        return

    with open(time_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                time_data[key.strip()] = val.strip()
                time_keys.append(key.strip())

    # 2. 读取 log.txt
    if not os.path.exists(log_path):
        print(f" [错误] 找不到日志文件: {log_path}")
        return
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_lines = f.readlines()

    results = {}

    # 3. 按区间搜索第一个匹配项
    for idx, key in enumerate(time_keys):
        start_time_str = time_data[key]
        end_time_str = time_data[time_keys[idx+1]] if idx + 1 < len(time_keys) else "99-99 99:99:99.999"

        found_value = "No Matching"

        # 转换键名: Hall Time -> Hall Fps
        new_key_name = key.replace("Time", "Fps")

        for line in log_lines:
            log_time_str = line[:18]

            # 时间区间判断 (log_time 必须在当前起始点和下一个起始点之间)
            if start_time_str <= log_time_str < end_time_str:
                # 匹配 targetFps = 数字
                if "targetFps" in line:
                    match = re.search(r'targetFps\s*=\s*(\d+)', line)
                    if match:
                        found_value = match.group(1)
                        break # 只记录第一次出现，跳出当前日志循环

            # 性能优化
            if log_time_str >= end_time_str:
                break

        results[new_key_name] = found_value

    # 4. 写入结果到 results.txt
    output_path = os.path.join(os.path.dirname(time_txt_path), "results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 按照转换后的 Fps 键名写入
        for key in results:
            f.write(f"{key}: {results[key]}\n")

    print(f" [完成] 分析报告已生成: {output_path}")

def draw_chinese_text(img, text, position, font_path="msyh.ttc", font_size=25, color=(0, 255, 0)):
    """
    使用 PIL 在图像上绘制中文
    """
    # 1. 先切断与外部 numpy/PIL buffer 的共享关系，再转为 PIL
    img_bgr = np.ascontiguousarray(img).copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb, mode="RGB").copy()
    draw = ImageDraw.Draw(img_pil)

    # 2. 加载字体
    try:
        # Windows常用字体: "msyh.ttc" (微软雅黑), "simhei.ttf" (黑体)
        # 如果是Linux，通常路径为: "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 如果找不到指定字体，使用默认字体（可能还是不支持中文，建议放一个ttf在工程目录）
        font = ImageFont.load_default()

    # 3. 绘制文字 (PIL使用RGB，但我们传进来的是BGR转RGB，所以color(0,255,0)依然是绿色)
    draw.text(position, text, font=font, fill=color)

    # 4. PIL 转回 OpenCV (BGR)，显式拷贝避免持有 PIL 内部内存视图
    out_rgb = np.array(img_pil, copy=True)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def visualizer_process(queue, visual=True):
    """
    独立进程：处理图像旋转、JSON保存及目标检测框可视化
    """
    vis_mode = os.environ.get("AUTOGAME_VIS_MODE", "window").strip().lower()
    show_window = visual and vis_mode != "launcher"
    print(f"[Visualizer] 显示进程已启动, 可视化状态: {visual}, mode: {vis_mode}")
    window_name = "Frame Monitor"
    log_dir = str(resolve_process_temp_logs_dir())
    print(f"[Visualizer] 预览帧输出目录: {log_dir}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 统一显示尺寸 (826 * 2) * (384 * 2) = 1652 * 768
    target_width = 826 * 2
    target_height = 384 * 2

    def is_timed_special_value(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        timing = value[1]
        if not isinstance(timing, (list, tuple)) or len(timing) != 1:
            return False
        try:
            float(timing[0])
        except (TypeError, ValueError):
            return False
        return True

    def unwrap_timed_special_value(value):
        if is_timed_special_value(value):
            return value[0]
        return value

    def is_detection_list(value):
        value = unwrap_timed_special_value(value)
        if not isinstance(value, list) or not value:
            return False

        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                return False
            try:
                float(item[0])
                float(item[1])
                float(item[2])
                float(item[3])
                float(item[4])
            except (TypeError, ValueError):
                return False
        return True

    def draw_detection_list(frame, detections, font_size_getter, font_size):
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_id} {conf:.2f}"
            frame = draw_chinese_text(
                frame,
                label,
                (x1, max(font_size_getter(25), y1 - font_size_getter(10))),
                "simhei.ttf",
                font_size,
                (0, 255, 0),
            )
        return frame

    def collect_visualizations(value):
        visuals = []
        if is_timed_special_value(value):
            return collect_visualizations(value[0])
        if isinstance(value, dict):
            own_visuals = value.get("__visualizations__")
            if isinstance(own_visuals, list):
                visuals.extend(item for item in own_visuals if isinstance(item, dict))
            for key, child in value.items():
                if key == "__visualizations__":
                    continue
                visuals.extend(collect_visualizations(child))
        elif isinstance(value, list):
            for child in value:
                visuals.extend(collect_visualizations(child))
        return visuals

    def _as_bgr_color(value, fallback):
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return fallback
        try:
            return tuple(int(max(0, min(255, v))) for v in value[:3])
        except (TypeError, ValueError):
            return fallback

    def draw_algorithm_visualizations(frame, visuals, font_size_getter, font_size):
        for visual_item in visuals:
            visual_type = str(visual_item.get("type") or "").lower()
            bbox = visual_item.get("bbox_xyxy")
            label = str(visual_item.get("label") or visual_type or "visual")
            score = visual_item.get("score")
            mask_color = _as_bgr_color(visual_item.get("color_bgr"), (0, 255, 255))
            bbox_color = _as_bgr_color(visual_item.get("bbox_color_bgr"), (0, 255, 0))

            contours = visual_item.get("contours")
            if isinstance(contours, list) and contours:
                overlay = frame.copy()
                contour_arrays = []
                for contour in contours:
                    if not isinstance(contour, list) or len(contour) < 3:
                        continue
                    arr = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                    contour_arrays.append(arr)
                if contour_arrays:
                    alpha = float(visual_item.get("alpha", 0.45) or 0.45)
                    cv2.fillPoly(overlay, contour_arrays, mask_color)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
                    cv2.polylines(frame, contour_arrays, True, bbox_color, 2)

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                if score is not None:
                    try:
                        label = f"{label} {float(score):.2f}"
                    except (TypeError, ValueError):
                        pass
                frame = draw_chinese_text(
                    frame,
                    label,
                    (x1, max(font_size_getter(25), y1 - font_size_getter(10))),
                    "simhei.ttf",
                    font_size,
                    bbox_color,
                )
        return frame

    def sanitize_info_for_log(value):
        if is_timed_special_value(value):
            return [sanitize_info_for_log(value[0]), [value[1][0]]]
        if isinstance(value, dict):
            sanitized = {}
            for key, child in value.items():
                if key == "__visualizations__":
                    sanitized[key] = f"{len(child) if isinstance(child, list) else 0} visualizations"
                else:
                    sanitized[key] = sanitize_info_for_log(child)
            return sanitized
        if isinstance(value, list):
            return [sanitize_info_for_log(child) for child in value]
        return value

    while True:
        try:
            data = queue.get()
            if data == "STOP":
                break

            if isinstance(data, (list, tuple)) and len(data) == 5:
                frame_rgb, stage, info, index, frame_meta = data
            else:
                frame_rgb, stage, info, index = data
                frame_meta = {}
            frame_meta = frame_meta if isinstance(frame_meta, dict) else {}
            if not isinstance(frame_rgb, np.ndarray):
                frame_rgb = np.array(frame_rgb, copy=True)
            else:
                frame_rgb = np.ascontiguousarray(frame_rgb).copy()

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_rotated = frame_bgr.copy()

            # --- 关键修改：计算字体缩放比 ---
            # 获取旋转后原图的尺寸
            orig_h, orig_w = frame_rotated.shape[:2]
            # 计算缩放比例 (基于宽度的倍数)
            scale = target_width / orig_w

            # 根据比例动态调整字号
            # 如果原图很小，scale > 1，字号变大；如果原图很大，scale < 1，字号减小，从而在 resize 后保持视觉一致
            def get_scaled_size(base_size):
                return int(base_size / scale)

            # 调整后的基础参数。图片不再叠加 ID / Stage / info 文本，只保留检测框标注字号。
            font_info = get_scaled_size(18)
            # ---------------------------

            # 2. 可视化检测框
            detection_keys = set()
            sorted_info_items = []
            if isinstance(info, dict):
                sorted_info_items = sorted(info.items(), key=lambda item: str(item[0]).lower())

            if visual and isinstance(info, dict):
                for k, v in sorted_info_items:
                    if is_detection_list(v):
                        detection_keys.add(k)
                        detections = unwrap_timed_special_value(v)
                        frame_rotated = draw_detection_list(frame_rotated, detections, get_scaled_size, font_info)
                visual_items = collect_visualizations(info)
                if visual_items:
                    frame_rotated = draw_algorithm_visualizations(
                        frame_rotated,
                        visual_items,
                        get_scaled_size,
                        font_info,
                    )

            # 4. 存储原始图
            base_filename = os.path.join(log_dir, f"frame_{index:05d}")
            frame_name = f"frame_{index:05d}.jpg"
            if not write_image_unicode(f"{base_filename}.jpg", frame_rotated):
                raise RuntimeError(f"preview image write failed: {base_filename}.jpg")
            runtime_logs = frame_meta.get("runtime_logs")
            runtime_logs = dict(runtime_logs) if isinstance(runtime_logs, dict) else {}
            if isinstance(frame_meta.get("frame_decision"), dict):
                runtime_logs["frame_decision"] = frame_meta.get("frame_decision")
            payload = build_frame_log_payload(
                stage,
                info,
                index,
                runtime_logs=runtime_logs,
                group_name=frame_meta.get("group_name"),
                frame_name=frame_name,
            )
            with open(f"{base_filename}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)

            # 5. 缩放显示 (此时文字会因为前面的反向补偿，在显示窗口中看起来大小适中)
            if show_window:
                frame_display = cv2.resize(frame_rotated, (target_width, target_height))
                cv2.imshow(window_name, frame_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"\n[Visualizer Error] {e}")
            break

    if show_window:
        cv2.destroyAllWindows()

def _read_autogame_config(config_path="aw/autogame/config/config.json"):
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_screen_mode(config_path="aw/autogame/config/config.json"):
    """
    仅读取并返回 config.json 中的 screen_mode 字段
    """
    config = _read_autogame_config(config_path)
    return str(config.get("screen_mode", "0"))


def get_touch_backend(config_path="aw/autogame/config/config.json"):
    """
    读取触控后端配置。

    统一从 config.json 中读取 touch_backend。
    可选值：sendevent / uinput
    未配置或配置非法时，默认使用 uinput。
    """
    config = _read_autogame_config(config_path)
    backend = str(config.get("touch_backend", "")).strip().lower()
    if backend in {"sendevent", "uinput"}:
        return backend

    return "uinput"

def _parse_display_rotation(output: str):
    if not output:
        return None

    blocks = []
    block_match = re.search(
        r"\[SCREEN PROPERTY\](.*?)(\n\[|\Z)",
        output,
        re.S
    )
    if block_match:
        blocks.append(block_match.group(1))
    blocks.append(output)

    patterns = (
        r"^\s*Rotation\s*[:=]\s*(\d+)\s*$",
        r"\brotation\b\s*[:=]?\s*(\d+)",
    )
    for block in blocks:
        for pattern in patterns:
            match = re.search(pattern, block, re.IGNORECASE | re.MULTILINE)
            if match:
                return normalize_rotation(match.group(1))
    return None


def get_display_rotation():
    """
    获取屏幕真实旋转角度（0/90/180/270）。
    """
    candidates = (
        (
            ["hdc", "shell", "hidumper", "-s", "DisplayManagerService", "-a", "-a"],
            "DisplayManagerService",
        ),
        (
            ["hdc", "shell", "hidumper", "-s", "WindowManagerService", "-a", "-a"],
            "WindowManagerService",
        ),
        (
            ["hdc", "shell", "snapshot_display"],
            "snapshot_display",
        ),
        (
            ["hdc", "shell", "hidumper", "-s", "RenderService", "-a", "screen"],
            "RenderService",
        ),
    )
    last_output = None
    last_source = None

    for cmd, source in candidates:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=5,
                **hidden_subprocess_kwargs(),
            )
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            if output:
                last_output = output
                last_source = source

            rotation = _parse_display_rotation(output)
            if rotation is not None:
                return rotation
        except Exception as e:
            last_output = str(e)
            last_source = source

    print("[Rotation] Rotation not found")
    if last_output:
        print(f"[Rotation] {last_source} 输出片段: {last_output[:500]}")
    return None

def insert_logs(log_name, time_dura, *key_words):
    """
    在指定日志文件中追加一行记录，并休眠 time_dura 秒。

    Args:
        log_name: 任务名称/日志标识
        time_dura: 持续时间（单位：秒）
        *key_words: 额外的关键词列表
    """
    # 1. 获取环境变量中的 task_name
    task_name = os.environ.get("TARGET_GAME_CASE", "default_task")

    # 2. 准备文件路径
    log_dir = os.path.join(r"aw/autogame/temp/results", task_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = os.path.join(log_dir, "time.txt")

    # 3. 处理时间格式
    start_dt = datetime.now()
    # 格式化 start_time: 月-日 时:分:秒.毫秒
    start_time_str = start_dt.strftime("%m-%d %H:%M:%S") + f".{int(start_dt.microsecond / 1000):03d}"

    # 4. 计算 end_time: start_time + time_dura (秒)
    # 修改点：这里改为 seconds=time_dura
    end_dt = start_dt + timedelta(seconds=time_dura)
    end_time_str = end_dt.strftime("%m-%d %H:%M:%S") + f".{int(end_dt.microsecond / 1000):03d}"

    # 5. 组合关键词字符串
    keywords_str = " ".join(map(str, key_words))

    # 6. 构建整行数据
    log_entry = f"{log_name} {start_time_str} {end_time_str} {keywords_str}\n"

    # 7. 追加模式写入文件
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"写入日志失败: {e}")

    # 8. 执行休眠
    # 修改点：直接 sleep time_dura，不需要再乘 60
    print(f"[Log] 已记录 {log_name}，开始休眠 {time_dura} 秒...")
    time.sleep(time_dura)

def analyze_txt(log_path, frame_path, time_txt_path, result_path):
    """
    解析 time.txt，根据时间段分析帧数据并将最终汇总结果记录在 result_path。

    :param log_path: 供子函数使用的日志路径
    :param frame_path: 截图帧所在的目录
    :param time_txt_path: 输入的包含时间段信息的 time.txt 路径
    :param result_path: 最终结果汇总输出路径 (results.txt)
    """
    if not os.path.exists(time_txt_path):
        print(f"Error: {time_txt_path} not found.")
        return

    # 定义关键字与函数的映射关系
    func_map = {
        "帧率": analyze_fps,
        "插帧": analyze_insert_frame,
        "超分": analyze_super_resolution
    }

    results_to_write = []

    with open(time_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # 基础格式要求：log_name, start_date, start_time, end_date, end_time (共5个parts)
            if len(parts) < 5:
                continue

            # 1. 解析基础字段
            # time.txt 格式：log_name 02-24 14:24:07.574 02-24 14:24:12.574 帧率 插帧
            log_name = parts[0]
            start_str = f"{parts[1]} {parts[2]}" # 拼接日期和时间
            end_str = f"{parts[3]} {parts[4]}"
            keywords = parts[5:]

            # 2. 执行分析逻辑
            current_line_results = [log_name]

            for kw in keywords:
                if kw in func_map:
                    # 按照你更新后的逻辑，传入 log_path 供子函数参考
                    res = func_map[kw](log_path, frame_path, start_str, end_str)
                    current_line_results.append(f"{kw}:{res}")

            # 3. 组合成一行：log_name 关键字1:结果1 关键字2:结果2
            results_to_write.append(" ".join(current_line_results))

    # 4. 写入最终结果文件 (使用 result_path)
    # 先确保输出目录存在
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_to_write) + "\n")

    print(f"Analysis complete. Summary saved to {result_path}")

def analyze_fps(log_path, frame_path, start_time, end_time):
    """
    读取 log_path，提取 start_time 到 end_time 之间所有的 targetFps 值。

    :param log_path: 日志文件路径 (例如 hdc log 产生的文本)
    :param frame_path: 帧目录 (此函数暂不用)
    :param start_time: 开始时间字符串 "02-24 14:24:07.574"
    :param end_time: 结束时间字符串 "02-24 14:24:12.574"
    :return: 包含所有 targetFps 的字符串，例如 "60 90"
    """
    if not os.path.exists(log_path):
        return "LogNotFound"

    # 匹配 "targetFps":60 或 "targetFps": 60 或 targetFps:60 等格式
    # 使用正则表达式匹配 targetFps 后面跟着的数字
    fps_pattern = re.compile(r'targetFps["\']?\s*:\s*(\d+)')

    found_fps = []

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. 提取行首时间 (前 18 位左右: 02-24 15:12:25.942)
                # 假设日志格式固定，前18位是时间
                line_time = line[:18]

                # 2. 时间区间过滤 (利用字符串直接比较大小)
                if start_time <= line_time <= end_time:
                    # 3. 搜索该行是否存在 targetFps
                    matches = fps_pattern.findall(line)
                    for val in matches:
                        if val not in found_fps:  # 去重，只记录出现过的不同帧率
                            found_fps.append(val)

                # 如果当前行时间已经超过了 end_time，可以提前结束读取提高效率
                elif line_time > end_time:
                    # 注意：如果日志不是严格按时间顺序写的，请删掉这两行
                    break

    except Exception as e:
        print(f"Error reading log: {e}")
        return "Error"

    # 返回结果，如果没有找到则返回空字符串或特定的默认值
    return " ".join(found_fps) if found_fps else "None"

def analyze_insert_frame(log_path, frame_path, start_time, end_time):
    """
    分析指定时间段内的截图，只要有一张图片红色像素占比 > 0.1 则立即返回“生效”。
    """
    if not os.path.exists(frame_path):
        return "目录不存在"

    # 1. 筛选时间段内的图片
    all_files = os.listdir(frame_path)
    target_frames = []

    # 将标准时间格式中的 : 替换为 - 以便匹配文件名 (02-24 13-56-42.982.jpg)
    s_time_cmp = start_time.replace(":", "-")
    e_time_cmp = end_time.replace(":", "-")

    for f in all_files:
        if f.endswith((".jpg", ".png")):
            file_name_no_ext = os.path.splitext(f)[0]
            if s_time_cmp <= file_name_no_ext <= e_time_cmp:
                target_frames.append(f)

    if not target_frames:
        return "未找到帧"

    # 2. 逐张分析，一旦发现满足条件的图片立即返回
    for f in target_frames:
        file_path = os.path.join(frame_path, f)
        img = cv2.imread(file_path)
        if img is None:
            continue

        # 获取总像素数
        height, width = img.shape[:2]
        total_pixels = height * width

        # --- 核心红色提取逻辑 ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义红色范围 (S/V 下限 200/220)
        lower_red1 = np.array([0, 200, 220])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 200, 220])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)

        # 形态学处理：3x3 核腐蚀再膨胀
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        # 统计处理后的红色像素点
        red_pixel_count = cv2.countNonZero(red_mask)

        # 计算比例
        if red_pixel_count > 3000:
            # --- 关键改进：找到第一个满足条件的就停止后续所有计算 ---
            return "生效"

    # 如果遍历完所有图片都没有满足条件的
    return "失效"

def analyze_super_resolution(log_path, frame_path, start_time, end_time):
    return "Done"

if __name__ == '__main__':
    print(get_resolution())
