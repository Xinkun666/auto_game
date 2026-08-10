"""手动到达房屋门前后，只对当前画面执行一次房型匹配。

这个入口不校准人物位置，不移动视角，不后拉，不重试，不执行进屋回放。
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_house_search_strategy import (
    NandaSearchContext,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_latest_house_search import (
    NandaCurrentViewMatchResult,
    NandaLatestSettings,
    NandaLocalRoomMatcher,
)
from aw.autogame.tools.Utils import _read_project_config, write_image_unicode


if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "aw" / "autogame" / "temp" / "results" / "room_match_once"
)
RESULT_MARKER = "__AUTOGAME_ROOM_MATCH_RESULT__:"

_matcher: Optional[NandaLocalRoomMatcher] = None
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


def _result_path() -> Path:
    explicit = os.environ.get("AUTOGAME_ROOM_MATCH_OUTPUT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
    if archive_dir:
        return Path(archive_dir).expanduser().resolve() / "room_match_result.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_RESULT_DIR / f"room_match_{timestamp}.json"


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def preload_runtime() -> None:
    """可选预热；只加载模型和房型索引，不读取画面也不控制设备。"""
    _get_matcher()


def _build_context(worker: "FrameWorker", frame: np.ndarray) -> NandaSearchContext:
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
        phase_label="manual_room_match_once",
        refresh_frame=lambda _reason: True,
        should_abort=lambda: False,
        is_outside=lambda: True,
    )


def _match_payload(
    result: NandaCurrentViewMatchResult,
    *,
    result_path: Path,
    query_frame_path: Optional[Path],
    frame: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    expected_room_id = os.environ.get("AUTOGAME_EXPECTED_ROOM_ID", "").strip()
    status = "matched" if result.matched else "no_match"
    return {
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "manual_current_view_once",
        "movement_enabled": False,
        "attempt_count": 1,
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
        "replay_path": result.replay_path,
        "frame_shape": list(frame.shape),
        "elapsed_seconds": elapsed_seconds,
        "query_frame_path": (
            str(query_frame_path) if query_frame_path is not None else None
        ),
        "result_path": str(result_path),
    }


def on_stage(worker: "FrameWorker") -> None:
    global _running, _completed
    if _completed or _running:
        return
    _running = True
    result_path = _result_path()
    started_at = time.monotonic()
    query_frame_path: Optional[Path] = None
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
        query_frame_path = result_path.with_suffix(".query.png")
        if not write_image_unicode(query_frame_path, frame):
            query_frame_path = None

        matcher = _get_matcher()
        result = matcher.match_current_view(_build_context(worker, frame))
        payload = _match_payload(
            result,
            result_path=result_path,
            query_frame_path=query_frame_path,
            frame=frame,
            elapsed_seconds=time.monotonic() - started_at,
        )
        _write_payload(result_path, payload)
        worker.frame_log(
            f"[NandaMatchOnly] 单次匹配结束：status={payload['status']}，"
            f"room={payload['room_id']}，score={payload['score']}，"
            f"expected={payload['expected_room_id']}，correct={payload['correct']}，"
            f"result={result_path}"
        )
        print(RESULT_MARKER + json.dumps(_json_safe(payload), ensure_ascii=False))
    except Exception as exc:
        payload = {
            "status": "error",
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": "manual_current_view_once",
            "movement_enabled": False,
            "attempt_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started_at,
            "result_path": str(result_path),
        }
        _write_payload(result_path, payload)
        worker.frame_log(
            f"[NandaMatchOnly] 单次匹配失败：{type(exc).__name__}: {exc}；"
            f"result={result_path}"
        )
        mark_failed = getattr(worker, "mark_failed", None)
        if callable(mark_failed):
            mark_failed(
                "room_match_once_failed",
                str(exc),
                result_path=str(result_path),
                exception=type(exc).__name__,
            )
        print(RESULT_MARKER + json.dumps(_json_safe(payload), ensure_ascii=False))
    finally:
        _completed = True
        _running = False
        worker.stop()


__all__ = ["on_stage", "preload_runtime"]
