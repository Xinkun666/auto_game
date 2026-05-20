from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.action_adapter import (
    AutoGameActionProxy,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.config import (
    ensure_pubg_test_import_path,
    get_pubg_room_search_config,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.frame_adapter import (
    AutoGameRoomPicCapture,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3_embedded import (
    get_sam3_perception,
)


BUTTONS_TO_PROCESS = ("jump", "map", "attack", "pick_btn", "door")
SLIDE_TIME_INTERVAL = 0.3
SLIDE_RATE = 1.0
LENGTH_FOR_HORIZONTAL_15_DEGREE = 0.0545

_ROOM_LIBRARY_CACHE: Dict[Tuple[str, str], Any] = {}
_ROOM_LIBRARY_CACHE_LOCK = threading.Lock()


class ReplayValidationError(ValueError):
    """Raised when an action DSL payload does not match the pubg_test schema."""


def _ensure_binary_flag(name: str, value: int) -> int:
    if value not in (0, 1):
        raise ReplayValidationError(f"{name} must be 0 or 1, got {value!r}")
    return value


def _ensure_angle(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise ReplayValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < 0 or value > 360:
        raise ReplayValidationError(f"{name} must be in [0, 360], got {value}")
    return value


@dataclass
class HdcReplaySearchResult:
    ok: bool
    result_name: str
    fallback_to_legacy: bool
    reason: str = ""
    room_id: Optional[str] = None
    dsl_record_path: Optional[str] = None
    debug_payload: Optional[Dict[str, Any]] = None


@dataclass
class PubgActionFlags:
    do_move: int
    view_left: int
    view_right: int
    view_up: int
    view_down: int
    jump: int
    pick_btn: int
    map: int
    door: int
    attack: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _ensure_binary_flag(item.name, getattr(self, item.name))
        if self.view_left and self.view_right:
            raise ReplayValidationError("view_left and view_right cannot both be 1")
        if self.view_up and self.view_down:
            raise ReplayValidationError("view_up and view_down cannot both be 1")

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PubgActionFlags":
        if not isinstance(data, dict):
            raise ReplayValidationError("actions must be a dict")
        lower_keys = {str(k).lower(): k for k in data.keys()}
        required = {item.name for item in fields(cls)}
        missing = required - set(lower_keys.keys())
        if missing:
            raise ReplayValidationError(f"Missing action flags: {sorted(missing)}")
        return cls(**{name: int(data[lower_keys[name]]) for name in required})

    @classmethod
    def initial_actions(cls) -> "PubgActionFlags":
        return cls(**{item.name: 0 for item in fields(cls)})


@dataclass
class PubgStepAction:
    move_direction: int
    actions: PubgActionFlags = field(default_factory=PubgActionFlags.initial_actions)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.move_direction = _ensure_angle("move_direction", int(self.move_direction))
        if self.params is None:
            self.params = {}

    def copy(self) -> "PubgStepAction":
        return self.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "move_direction": self.move_direction,
            "actions": self.actions.to_dict(),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PubgStepAction":
        if not isinstance(data, dict):
            raise ReplayValidationError("StepAction must be built from a dict")
        lower_keys = {str(k).lower(): k for k in data.keys()}
        if "move_direction" not in lower_keys or "actions" not in lower_keys:
            raise ReplayValidationError("move_direction and actions are required")
        raw_params = data.get(lower_keys.get("params"), {})
        if raw_params is None:
            params = {}
        elif isinstance(raw_params, dict):
            params = dict(raw_params)
        else:
            raise ReplayValidationError("params must be a dict if provided")
        return cls(
            move_direction=int(data[lower_keys["move_direction"]]),
            actions=PubgActionFlags.from_dict(data[lower_keys["actions"]]),
            params=params,
        )

    @classmethod
    def initial_step(cls) -> "PubgStepAction":
        return cls(move_direction=0, actions=PubgActionFlags.initial_actions())

    def is_idle_step(self) -> bool:
        return all(value == 0 for value in asdict(self.actions).values())


@dataclass
class PubgDslRecord:
    entries: List[Tuple[float, PubgStepAction]]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        preserve_duplicate_timestamps: bool = True,
    ) -> "PubgDslRecord":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ReplayValidationError("dsl_record.json must be a list")

        entries: List[Tuple[float, PubgStepAction]] = []
        collapsed: Dict[float, PubgStepAction] = {}
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ReplayValidationError(f"Item {index} is not a dict")
            if "time" not in item:
                raise ReplayValidationError(f"Item {index} missing time")
            try:
                timestamp = float(item["time"])
            except (TypeError, ValueError) as exc:
                raise ReplayValidationError(f"Invalid time at index {index}") from exc
            step = PubgStepAction.from_dict(item)
            if preserve_duplicate_timestamps:
                entries.append((timestamp, step))
            else:
                collapsed[timestamp] = step

        if not preserve_duplicate_timestamps:
            entries = list(collapsed.items())
        return cls(entries=entries)


class HdcPubgActionRunner:
    """pubg_test-style state-diff runner backed by auto_game HDC actions."""

    def __init__(self, proxy: AutoGameActionProxy, *, stop_timeout_sec: float = 3.0):
        self._proxy = proxy
        self._stop_timeout_sec = max(0.2, float(stop_timeout_sec))
        self._lock = threading.Lock()
        self._exec_lock = threading.Lock()
        self._started = False
        self._stop_event = threading.Event()
        self._queue: queue.Queue[Optional[PubgStepAction]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._current_state = PubgStepAction.initial_step().to_dict()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop_event.clear()
            self._started = True
            self._thread = threading.Thread(
                target=self._exec_loop,
                daemon=True,
                name="AutoGamePubgHdcActionRunner",
            )
            self._thread.start()

    def stop(self) -> bool:
        with self._lock:
            self._started = False
            self._stop_event.set()
            thread = self._thread
        self._queue.put(None)
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=self._stop_timeout_sec)
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self.execute_step(PubgStepAction.initial_step())
        try:
            self._proxy.release_all()
        except Exception:
            pass
        return not (thread is not None and thread.is_alive())

    def submit_steps(self, step_list: Sequence[PubgStepAction]) -> bool:
        with self._lock:
            if not self._started:
                return False
            for step in step_list:
                self._queue.put(step.copy())
        return True

    def get_current_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._current_state)

    def set_current_state(self, state: PubgStepAction) -> None:
        with self._lock:
            self._current_state = state.to_dict()

    def execute_step(self, step: PubgStepAction) -> None:
        with self._exec_lock:
            self._exec_step(step)

    def _exec_loop(self) -> None:
        while True:
            action_step = None
            try:
                action_step = self._queue.get(timeout=0.1)
                if action_step is None:
                    self._queue.task_done()
                    if self._stop_event.is_set():
                        break
                    continue
                if self._stop_event.is_set():
                    continue
                self.execute_step(action_step)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
            finally:
                if action_step is not None:
                    self._queue.task_done()

    def _exec_step(self, step: PubgStepAction) -> None:
        current = PubgStepAction.from_dict(self.get_current_state())
        actions_to_send: List[Dict[str, Any]] = []
        self._gen_actions_for_move(actions_to_send, current, step)
        self._gen_actions_for_view(actions_to_send, current, step)
        self._gen_actions_for_btn(actions_to_send, current, step)
        self.set_current_state(step)
        if actions_to_send:
            self._proxy.send_actions(actions_to_send)

    @staticmethod
    def _gen_actions_for_btn(
        actions: List[Dict[str, Any]], current: PubgStepAction, next_state: PubgStepAction
    ) -> None:
        for action_name in BUTTONS_TO_PROCESS:
            cur_active = getattr(current.actions, action_name) == 1
            nxt_active = getattr(next_state.actions, action_name) == 1
            if nxt_active and not cur_active:
                actions.append({"method": f"{action_name}_touch", "args": {}})
            elif not nxt_active and cur_active:
                actions.append({"method": f"{action_name}_release", "args": {}})

    @staticmethod
    def _gen_actions_for_view(
        actions: List[Dict[str, Any]], current: PubgStepAction, next_state: PubgStepAction
    ) -> None:
        cur_view = (
            current.actions.view_left,
            current.actions.view_right,
            current.actions.view_up,
            current.actions.view_down,
        )
        nxt_view = (
            next_state.actions.view_left,
            next_state.actions.view_right,
            next_state.actions.view_up,
            next_state.actions.view_down,
        )
        cur_view_speed = current.params.get("view_speed_rate")
        nxt_view_speed = next_state.params.get("view_speed_rate")
        if cur_view == nxt_view and cur_view_speed == nxt_view_speed:
            return
        cur_active = any(cur_view)
        nxt_active = any(nxt_view)
        if cur_active and not nxt_active:
            actions.append({"method": "view_release", "args": {}})
            return
        if not nxt_active:
            return

        slide_rate = (
            float(nxt_view_speed)
            if nxt_view_speed is not None
            else SLIDE_RATE
        )
        x_rate = 0.0
        y_rate = 0.0
        if nxt_view[0]:
            x_rate -= slide_rate
        if nxt_view[1]:
            x_rate += slide_rate
        if nxt_view[2]:
            y_rate -= slide_rate
        if nxt_view[3]:
            y_rate += slide_rate
        actions.append(
            {
                "method": "view_keep_slide",
                "args": {
                    "x_speed": x_rate,
                    "y_speed": y_rate,
                    "slide_length": LENGTH_FOR_HORIZONTAL_15_DEGREE,
                    "target_interval": SLIDE_TIME_INTERVAL,
                },
            }
        )

    @staticmethod
    def _gen_actions_for_move(
        actions: List[Dict[str, Any]], current: PubgStepAction, next_state: PubgStepAction
    ) -> None:
        cur_move_on = current.actions.do_move == 1
        nxt_move_on = next_state.actions.do_move == 1
        cur_angle = _move_direction_to_stick_angle(current.move_direction)
        nxt_angle = _move_direction_to_stick_angle(next_state.move_direction)
        if nxt_move_on and not cur_move_on:
            actions.extend(
                [
                    {"method": "move_release", "args": {}},
                    {"method": "move_press", "args": {"init_angle": nxt_angle}},
                ]
            )
        elif nxt_move_on and cur_move_on:
            delta = nxt_angle - cur_angle
            if abs(delta) > 0.5:
                actions.append(
                    {"method": "move_slide_plus", "args": {"angle_step": delta}}
                )
        elif cur_move_on and not nxt_move_on:
            actions.append({"method": "move_release", "args": {}})


class AutoGamePubgDslReplayer:
    _STOP_POLL_INTERVAL_SEC = 0.05

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_pubg_room_search_config()
        self.is_replaying = False
        self._replay_lock = threading.Lock()

    def stop_replay(self) -> None:
        with self._replay_lock:
            self.is_replaying = False

    def is_running(self) -> bool:
        with self._replay_lock:
            return self.is_replaying

    def replay_block(
        self,
        action_runner: HdcPubgActionRunner,
        dsl_record_path: str | Path,
        *,
        skip_idle: Optional[bool] = None,
    ) -> bool:
        with self._replay_lock:
            if self.is_replaying:
                print("[ReplayRoomSearch] 正在回放中，跳过新的回放请求")
                return False
            self.is_replaying = True
        try:
            self._do_replay(action_runner, dsl_record_path, skip_idle=skip_idle)
            return True
        finally:
            with self._replay_lock:
                self.is_replaying = False

    def _do_replay(
        self,
        action_runner: HdcPubgActionRunner,
        dsl_record_path: str | Path,
        *,
        skip_idle: Optional[bool],
    ) -> None:
        preserve_duplicates = bool(
            self.config.get("replay_preserve_duplicate_timestamps", True)
        )
        record = PubgDslRecord.load(
            dsl_record_path,
            preserve_duplicate_timestamps=preserve_duplicates,
        )
        skip_idle = bool(
            self.config.get("replay_skip_idle", True)
            if skip_idle is None
            else skip_idle
        )
        time_scale = max(0.05, float(self.config.get("replay_time_scale", 1.0)))
        min_wait_sec = max(0.0, float(self.config.get("replay_min_wait_sec", 0.001)))

        print(
            f"[ReplayRoomSearch] 开始回放 {dsl_record_path}, "
            f"steps={len(record.entries)}, skip_idle={skip_idle}"
        )
        last_exec_time = 0.0
        prev_was_idle = True
        try:
            for start_time, action_step in record.entries:
                if not self.is_running():
                    print("[ReplayRoomSearch] 回放中止")
                    break
                current_time = float(start_time)
                wait_time = max(0.0, current_time - last_exec_time) * time_scale
                last_exec_time = current_time

                dont_wait = skip_idle and prev_was_idle
                prev_was_idle = action_step.is_idle_step()

                if not dont_wait and wait_time >= min_wait_sec:
                    self._sleep_while_replaying(wait_time)
                if not self.is_running():
                    break
                action_runner.submit_steps([action_step])
        finally:
            action_runner.submit_steps([PubgStepAction.initial_step()])
            print("[ReplayRoomSearch] 回放结束")

    def _sleep_while_replaying(self, wait_time: float) -> None:
        deadline = time.perf_counter() + max(0.0, wait_time)
        while time.perf_counter() < deadline and self.is_running():
            time.sleep(min(self._STOP_POLL_INTERVAL_SEC, deadline - time.perf_counter()))


class AutoGameReplayRoomSearcher:
    """Match the current house facade and replay the matched room DSL on HDC."""

    def __init__(self, worker, config: Optional[Dict[str, Any]] = None):
        self.worker = worker
        self.config = config or get_pubg_room_search_config()
        self.capture = AutoGameRoomPicCapture(
            worker,
            control_proxy=None,
            frame_color=self.config.get("frame_color", "rgb"),
            refresh_interval_sec=float(
                self.config.get("frame_refresh_interval_sec", 0.12)
            ),
            refresh_mode=self.config.get("frame_refresh_mode", "worker_refresh"),
        )
        self.action_proxy = AutoGameActionProxy(worker, self.config)
        self.action_runner = HdcPubgActionRunner(
            self.action_proxy,
            stop_timeout_sec=float(self.config.get("replay_queue_stop_timeout_sec", 3.0)),
        )
        self.replayer = AutoGamePubgDslReplayer(self.config)

    def search_current_house(self, *, source: str = "") -> HdcReplaySearchResult:
        print(f"[ReplayRoomSearch] 启动内嵌房型匹配+DSL回放 source={source}")
        try:
            frame = self.capture.get_current_frame(force_refresh=True)
            if frame is None:
                return self._result(False, "CAPTURE_FAILED", "无法获取当前画面")

            segmentation = self._segment_house(frame)
            if segmentation is None:
                return self._result(False, "SEGMENT_FAILED", "SAM3 房屋分割失败")

            room_id, dsl_record_path, debug_payload = self._match_room(segmentation)
            if room_id is None:
                return self._result(
                    False,
                    "NO_MATCH",
                    "未匹配到房型",
                    debug_payload=debug_payload,
                )
            if not dsl_record_path:
                return self._result(
                    False,
                    "NO_DSL",
                    f"房型 {room_id} 没有 DSL 回放文件",
                    room_id=room_id,
                    debug_payload=debug_payload,
                )
            decision = (debug_payload or {}).get("decision") or {}
            if decision.get("replay_allow_actions") is False:
                return self._result(
                    False,
                    "REPLAY_REJECTED",
                    f"房型 {room_id} 的 metadata 禁止回放",
                    room_id=room_id,
                    dsl_record_path=dsl_record_path,
                    debug_payload=debug_payload,
                )

            replay_ok = self._replay_dsl(dsl_record_path)
            if not replay_ok:
                return self._result(
                    False,
                    "REPLAY_FAILED",
                    f"房型 {room_id} 回放失败",
                    room_id=room_id,
                    dsl_record_path=dsl_record_path,
                    debug_payload=debug_payload,
                )
            return self._result(
                True,
                "SUCCESS",
                f"房型 {room_id} 已完成匹配和回放",
                room_id=room_id,
                dsl_record_path=dsl_record_path,
                debug_payload=debug_payload,
            )
        except Exception as exc:
            return self._result(False, "EXCEPTION", str(exc))

    def _segment_house(self, frame):
        try:
            return get_sam3_perception().segment_house(frame)
        except Exception as exc:
            print(f"[ReplayRoomSearch] SAM3 房屋分割不可用: {exc}")
            return None

    def _match_room(self, segmentation) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        room_lib = self._load_room_library()
        sample_started_at = datetime.now()
        return room_lib.search_house(
            segmentation.segmented_bgr,
            segmented_house_mask=getattr(segmentation, "cropped_mask", None),
            original_house_image=getattr(segmentation, "cropped_bgr", None),
            sample_started_at=sample_started_at,
        )

    def _load_room_library(self):
        pubg_test_root = ensure_pubg_test_import_path(self.config)
        match_dump_dir = str(self.config.get("room_match_dump_dir") or "")
        cache_key = (str(pubg_test_root), match_dump_dir)
        with _ROOM_LIBRARY_CACHE_LOCK:
            cached = _ROOM_LIBRARY_CACHE.get(cache_key)
            if cached is not None:
                return cached

            from gametest_proxy.pubg_room_explore.img_similarity.room_library_process import (
                RoomLibrary,
            )
            from gametest_proxy.pubg_room_explore.img_similarity.similarity_utils import (
                ImgSimilarityWithDinoV3,
            )

            room_lib = RoomLibrary(
                extractor=ImgSimilarityWithDinoV3(),
                match_dump_dir=match_dump_dir or None,
            )
            _ROOM_LIBRARY_CACHE[cache_key] = room_lib
            return room_lib

    def _replay_dsl(self, dsl_record_path: str | Path) -> bool:
        self.action_runner.start()
        try:
            return self.replayer.replay_block(
                self.action_runner,
                dsl_record_path,
                skip_idle=bool(self.config.get("replay_skip_idle", True)),
            )
        finally:
            self.action_runner.stop()

    def _result(
        self,
        ok: bool,
        result_name: str,
        reason: str = "",
        *,
        room_id: Optional[str] = None,
        dsl_record_path: Optional[str] = None,
        debug_payload: Optional[Dict[str, Any]] = None,
    ) -> HdcReplaySearchResult:
        if reason:
            print(f"[ReplayRoomSearch] {result_name}: {reason}")
        return HdcReplaySearchResult(
            ok=ok,
            result_name=result_name,
            fallback_to_legacy=not ok
            and bool(self.config.get("embedded_allow_legacy_fallback", True)),
            reason=reason,
            room_id=room_id,
            dsl_record_path=str(dsl_record_path) if dsl_record_path else None,
            debug_payload=debug_payload,
        )


def _move_direction_to_stick_angle(move_direction: int) -> float:
    angle = (int(move_direction) - 90) % 360
    if angle > 180:
        angle -= 360
    return float(angle)
