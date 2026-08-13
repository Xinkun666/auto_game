"""手动到达房屋门前后，按南大原方案匹配并搜索当前一栋房子。

这个入口保留南大动态抬头、房屋分割失败时最多三次取景和位置恢复，
不重复用户已完成的门前校准；匹配成功后，使用迁移后的 HOS 单摇杆
执行器回放房屋库中的南大搜房 DSL，匹配失败则直接结束。
"""

from __future__ import annotations

from datetime import datetime
import cv2
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_house_search_strategy import (
    NandaRoomMatch,
    NandaSearchContext,
    NandaSearchResult,
    NandaSearchStatus,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_latest_house_search import (
    NandaCurrentViewMatchResult,
    NandaHosJoystickReplayExecutor,
    NandaLatestSettings,
    NandaLocalRoomMatcher,
)
from aw.autogame.tools.Utils import _read_project_config, write_image_unicode


if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT / "aw" / "autogame" / "temp" / "results" / "room_match_once"
)
DETAILS_FILENAME = "匹配详情.json"
SUMMARY_FILENAME = "匹配概要.txt"
ORIGINAL_IMAGE_FILENAME = "原始图片.png"
MATCH_IMAGE_FILENAME = "匹配结果.png"
RESULT_MARKER = "__AUTOGAME_ROOM_MATCH_RESULT__:"

_matcher: Optional[NandaLocalRoomMatcher] = None
_replay_executor: Optional[NandaHosJoystickReplayExecutor] = None
_running = False
_completed = False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _new_timestamp_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate = base_dir / f"{timestamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"无法创建本次匹配结果目录: {base_dir}")


def _result_dir() -> Path:
    explicit = os.environ.get("AUTOGAME_ROOM_MATCH_OUTPUT_DIR", "").strip()
    if explicit:
        result_dir = Path(explicit).expanduser().resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
    archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
    if archive_dir:
        return _new_timestamp_dir(Path(archive_dir).expanduser().resolve())
    return _new_timestamp_dir(DEFAULT_RESULT_ROOT)


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _frame_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """HOScrcpy 输出 RGB；OpenCV 写盘前必须转成 BGR。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"原始画面必须是 HxWx3 RGB 图像: {frame.shape}")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _save_original_image(path: Path, frame: np.ndarray) -> None:
    if not write_image_unicode(path, _frame_rgb_to_bgr(frame)):
        raise RuntimeError(f"无法保存原始图片: {path}")


def _save_match_image(
    path: Path,
    result: NandaCurrentViewMatchResult,
) -> Path:
    decision = result.decision if isinstance(result.decision, Mapping) else {}
    template_path_text = str(decision.get("template_path") or "").strip()
    if not template_path_text and isinstance(result.top_candidates, (list, tuple)):
        for candidate in result.top_candidates:
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("room_id") or "") != str(result.room_id or ""):
                continue
            template_path_text = str(candidate.get("template_path") or "").strip()
            if template_path_text:
                break
    if not template_path_text:
        raise RuntimeError(f"匹配到 {result.room_id}，但结果中没有模板视角路径")

    template_path = Path(template_path_text).expanduser().resolve()
    template_dir = template_path.parent
    room_dir = template_dir.parent.parent
    capture_dir = room_dir / "captures"
    source_path: Optional[Path] = None
    for suffix in (".jpeg", ".jpg", ".png"):
        candidate = capture_dir / f"{template_dir.name}{suffix}"
        if candidate.is_file():
            source_path = candidate
            break
    if source_path is None and template_path.is_file():
        source_path = template_path
    if source_path is None:
        raise FileNotFoundError(
            f"匹配到 {result.room_id}，但找不到对应视角图片: "
            f"capture={capture_dir / template_dir.name}, template={template_path}"
        )

    try:
        encoded = np.fromfile(str(source_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        image = None
    if image is None:
        raise RuntimeError(f"无法读取匹配视角图片: {source_path}")
    if not write_image_unicode(path, image):
        raise RuntimeError(f"无法保存匹配结果图片: {path}")
    return source_path


def _summary_text(result: NandaCurrentViewMatchResult) -> str:
    lines = [f"匹配成功：{'是' if result.matched else '否'}"]
    if result.matched:
        lines.append(f"匹配房型：{result.room_id}")
    return "\n".join(lines) + "\n"


def _load_settings() -> NandaLatestSettings:
    config = _read_project_config("Auto_PUBG_ALL")
    section = config.get("nanda_house_search", config)
    if not isinstance(section, Mapping):
        section = {}
    return NandaLatestSettings.from_mapping(section)


def _get_matcher() -> NandaLocalRoomMatcher:
    global _matcher
    if _matcher is not None:
        return _matcher
    matcher = NandaLocalRoomMatcher(_load_settings())
    if not matcher.is_available():
        raise RuntimeError(matcher.unavailable_reason)
    matcher.warmup()
    _matcher = matcher
    return matcher


def _get_replay_executor() -> NandaHosJoystickReplayExecutor:
    global _replay_executor
    if _replay_executor is None:
        _replay_executor = NandaHosJoystickReplayExecutor(_load_settings())
    return _replay_executor


def preload_runtime() -> None:
    """可选预热；只加载模型和房型索引，不读取画面也不控制设备。"""
    _get_matcher()


def _build_context(worker: "FrameWorker", frame: np.ndarray) -> NandaSearchContext:
    def refresh_frame(_reason: str = "") -> bool:
        refresh = getattr(worker, "refresh_frame", None)
        return bool(refresh()) if callable(refresh) else False

    def should_abort() -> bool:
        return not bool(getattr(worker, "running", True))

    return NandaSearchContext(
        worker=worker,
        frame=frame,
        house_id=None,
        entry={},
        entry_location=None,
        entry_direction=None,
        current_location=None,
        current_direction=None,
        distance_to_entry=None,
        door_box=None,
        door_center_offset_px=None,
        door_area_ratio=None,
        phase_label="manual_nanda_room_match_once",
        refresh_frame=refresh_frame,
        should_abort=should_abort,
        should_abort_replay=should_abort,
        is_outside=lambda: True,
    )


def _build_replay_match(
    result: NandaCurrentViewMatchResult,
) -> NandaRoomMatch:
    if not result.matched or not result.room_id or not result.replay_path:
        raise ValueError("只有包含房型和回放路径的成功匹配才能执行搜房")
    return NandaRoomMatch(
        room_id=str(result.room_id),
        replay_path=str(result.replay_path),
        score=result.score,
        metadata={
            "decision": result.decision,
            "selection_reason": result.selection_reason,
            "requires_pose_realign": result.requires_pose_realign,
            "matching_attempts": result.matching_attempts,
            "view_preparation": result.view_preparation,
        },
    )


def _match_payload(
    result: NandaCurrentViewMatchResult,
    *,
    result_dir: Path,
    details_path: Path,
    original_image_path: Path,
    summary_path: Path,
    match_image_path: Optional[Path],
    match_image_source_path: Optional[Path],
    frame: np.ndarray,
    elapsed_seconds: float,
    search_result: Optional[NandaSearchResult],
    search_elapsed_seconds: Optional[float],
) -> dict[str, Any]:
    expected_room_id = os.environ.get("AUTOGAME_EXPECTED_ROOM_ID", "").strip()
    search_status = (
        search_result.status.value if search_result is not None else "skipped"
    )
    status = "matched" if result.matched else "no_match"
    if result.matched and (
        search_result is None
        or search_result.status != NandaSearchStatus.COMPLETED
    ):
        status = "search_failed"
    return {
        "status": status,
        "match_status": "matched" if result.matched else "no_match",
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "manual_nanda_original_match_and_search_once",
        "movement_enabled": result.movement_enabled,
        "attempt_count": result.attempt_count,
        "room_id": result.room_id,
        "expected_room_id": expected_room_id or None,
        "correct": (
            result.room_id == expected_room_id
            if expected_room_id
            else None
        ),
        "score": result.score,
        "no_match_reason": result.no_match_reason or None,
        "decision": result.decision,
        "top_candidates": result.top_candidates,
        "top2_margin": result.top2_margin,
        "matcher_elapsed_ms": result.matcher_elapsed_ms,
        "sam3_score": result.sam3_score,
        "crop_xyxy": result.crop_xyxy,
        "query_structure": result.query_structure,
        "thresholds": result.thresholds,
        "matching_attempts": result.matching_attempts,
        "view_preparation": result.view_preparation,
        "selection_reason": result.selection_reason,
        "requires_pose_realign": result.requires_pose_realign,
        "replay_path": result.replay_path,
        "frame_shape": list(frame.shape),
        "elapsed_seconds": elapsed_seconds,
        "search_performed": search_result is not None,
        "search_status": search_status,
        "search_message": (
            search_result.message if search_result is not None else None
        ),
        "search_elapsed_seconds": search_elapsed_seconds,
        "search_metadata": (
            dict(search_result.metadata) if search_result is not None else None
        ),
        "total_algorithm_seconds": (
            elapsed_seconds + search_elapsed_seconds
            if search_elapsed_seconds is not None
            else elapsed_seconds
        ),
        "result_dir": str(result_dir),
        "original_image_path": str(original_image_path),
        "query_frame_path": str(original_image_path),
        "summary_path": str(summary_path),
        "match_image_path": (
            str(match_image_path) if match_image_path is not None else None
        ),
        "match_image_source_path": (
            str(match_image_source_path)
            if match_image_source_path is not None
            else None
        ),
        "result_path": str(details_path),
    }


def on_stage(worker: "FrameWorker") -> None:
    global _running, _completed
    if _completed or _running:
        return
    _running = True
    result_dir = _result_dir()
    details_path = result_dir / DETAILS_FILENAME
    summary_path = result_dir / SUMMARY_FILENAME
    original_image_path = result_dir / ORIGINAL_IMAGE_FILENAME
    started_at = time.monotonic()
    match_image_path: Optional[Path] = None
    match_image_source_path: Optional[Path] = None
    search_result: Optional[NandaSearchResult] = None
    search_elapsed_seconds: Optional[float] = None
    try:
        if worker.current_stage != "搜房阶段":
            worker.change_stage("搜房阶段")
        if worker.current_stage != "搜房阶段":
            raise RuntimeError("无法切换到只用于感知配置的搜房阶段")
        if worker.current_group != "other":
            if worker.change_group("other") is not True:
                raise RuntimeError("无法切换到房型匹配的基础感知分组")

        frame = getattr(worker, "frame", None)
        if frame is None:
            raise RuntimeError("当前没有可用游戏画面")
        frame = np.ascontiguousarray(frame).copy()
        _save_original_image(original_image_path, frame)

        matcher = _get_matcher()
        context = _build_context(worker, frame)
        result = matcher.match_original_nanda_flow(context)
        match_elapsed_seconds = time.monotonic() - started_at
        if result.matched:
            match_image_path = result_dir / MATCH_IMAGE_FILENAME
            match_image_source_path = _save_match_image(
                match_image_path,
                result,
            )
            search_started_at = time.monotonic()
            try:
                search_result = _get_replay_executor().replay(
                    context,
                    _build_replay_match(result),
                )
            except Exception as exc:
                search_result = NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    f"南大搜房回放执行异常: {exc}",
                    room_id=result.room_id,
                    replay_path=result.replay_path,
                    metadata={
                        "phase": "replay",
                        "exception": type(exc).__name__,
                    },
                )
            finally:
                search_elapsed_seconds = time.monotonic() - search_started_at
        payload = _match_payload(
            result,
            result_dir=result_dir,
            details_path=details_path,
            original_image_path=original_image_path,
            summary_path=summary_path,
            match_image_path=match_image_path,
            match_image_source_path=match_image_source_path,
            frame=frame,
            elapsed_seconds=match_elapsed_seconds,
            search_result=search_result,
            search_elapsed_seconds=search_elapsed_seconds,
        )
        _write_payload(details_path, payload)
        _write_text(summary_path, _summary_text(result))
        worker.frame_log(
            f"[NandaMatchSearchOnce] 单次匹配与搜房结束：status={payload['status']}，"
            f"room={payload['room_id']}，score={payload['score']}，"
            f"expected={payload['expected_room_id']}，correct={payload['correct']}，"
            f"search_status={payload['search_status']}，"
            f"search_elapsed={payload['search_elapsed_seconds']}，"
            f"result_dir={result_dir}"
        )
        if payload["status"] == "search_failed":
            mark_failed = getattr(worker, "mark_failed", None)
            if callable(mark_failed):
                mark_failed(
                    "room_search_once_failed",
                    payload.get("search_message") or "匹配成功后搜房未完成",
                    result_path=str(details_path),
                    room_id=result.room_id,
                    search_status=payload.get("search_status"),
                )
        print(RESULT_MARKER + json.dumps(_json_safe(payload), ensure_ascii=False))
    except Exception as exc:
        payload = {
            "status": "error",
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": "manual_nanda_original_match_and_search_once",
            "movement_enabled": True,
            "attempt_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started_at,
            "search_performed": search_result is not None,
            "search_status": (
                search_result.status.value if search_result is not None else "skipped"
            ),
            "search_elapsed_seconds": search_elapsed_seconds,
            "result_dir": str(result_dir),
            "original_image_path": (
                str(original_image_path) if original_image_path.is_file() else None
            ),
            "summary_path": str(summary_path),
            "match_image_path": None,
            "match_image_source_path": None,
            "result_path": str(details_path),
        }
        _write_payload(details_path, payload)
        _write_text(summary_path, "匹配成功：否\n")
        worker.frame_log(
            f"[NandaMatchSearchOnce] 单次执行失败：{type(exc).__name__}: {exc}；"
            f"result_dir={result_dir}"
        )
        mark_failed = getattr(worker, "mark_failed", None)
        if callable(mark_failed):
            mark_failed(
                "room_match_once_failed",
                str(exc),
                result_path=str(details_path),
                exception=type(exc).__name__,
            )
        print(RESULT_MARKER + json.dumps(_json_safe(payload), ensure_ascii=False))
    finally:
        _completed = True
        _running = False
        worker.stop()


__all__ = ["on_stage", "preload_runtime"]
