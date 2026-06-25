import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set


PHASE_RUNNING = "跑图"
PHASE_DRIVING = "开车"
PHASE_SEARCHING = "搜房"
STAGE_TIME_CONFIG_KEY = "stage_time_minutes"
DEFAULT_PHASE_DURATIONS_IN_MINUTES = {
    PHASE_SEARCHING: 10.0,
    PHASE_RUNNING: 10.0,
    PHASE_DRIVING: 10.0,
}
_PHASE_DURATION_CONFIG_ALIASES = {
    PHASE_SEARCHING: PHASE_SEARCHING,
    f"{PHASE_SEARCHING}阶段": PHASE_SEARCHING,
    PHASE_RUNNING: PHASE_RUNNING,
    f"{PHASE_RUNNING}阶段": PHASE_RUNNING,
    PHASE_DRIVING: PHASE_DRIVING,
    f"{PHASE_DRIVING}阶段": PHASE_DRIVING,
}


def _parse_positive_minutes(value: Any, default: float) -> float:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return float(default)
    if minutes <= 0:
        return float(default)
    return minutes


def load_phase_durations_from_config(config: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    durations = dict(DEFAULT_PHASE_DURATIONS_IN_MINUTES)
    if not isinstance(config, Mapping):
        return durations

    raw_stage_times = config.get(STAGE_TIME_CONFIG_KEY)
    if not isinstance(raw_stage_times, Mapping):
        return durations

    for raw_key, raw_value in raw_stage_times.items():
        phase_name = _PHASE_DURATION_CONFIG_ALIASES.get(str(raw_key).strip())
        if phase_name not in durations:
            continue
        durations[phase_name] = _parse_positive_minutes(raw_value, durations[phase_name])
    return durations


@dataclass
class PhaseState:
    name: str
    duration: float
    elapsed: float = 0.0
    started: bool = False
    completed: bool = False


def parse_case_loop_count(value, default: int = 1) -> int:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)
    return max(1, count)


class PhaseTimeManager:
    def __init__(self, durations_in_minutes: Dict[str, float], stage_phase_map: Dict[str, str]):
        self.phase_states = {
            phase_name: PhaseState(name=phase_name, duration=float(duration) * 60.0)
            for phase_name, duration in durations_in_minutes.items()
        }
        self.stage_phase_map = dict(stage_phase_map)

        self.last_stage: Optional[str] = None
        self.active_phase: Optional[str] = None
        self.active_since: Optional[float] = None
        self.total_duration = sum(state.duration for state in self.phase_states.values())
        self.total_elapsed = 0.0
        self.total_active_since: Optional[float] = None

        self.round_index = 0
        self.landed = False
        self.start_game_time: Optional[float] = None
        self.sp_started_ever = False
        self.sp_recording = False
        self.sp_saved = False
        self.case_loop_count = 1
        self.case_loop_index = 1

    def _format_phase_minutes(self, seconds: float) -> str:
        minutes = float(seconds) / 60.0
        if minutes.is_integer():
            return str(int(minutes))
        return f"{minutes:.1f}".rstrip("0").rstrip(".")

    def get_duration_minutes_label(self, phase_name: str) -> str:
        state = self.phase_states[phase_name]
        return self._format_phase_minutes(state.duration)

    def _phase_label(self, phase_name: str) -> str:
        return f"{phase_name}阶段"

    def _mark_phase_started(self, phase_name: str):
        state = self.phase_states[phase_name]
        if state.started:
            return
        state.started = True
        print(f"[Timer] {self._phase_label(phase_name)}开始，计划 {self._format_phase_minutes(state.duration)} 分钟")

    def _mark_phase_completed(self, phase_name: str) -> bool:
        state = self.phase_states[phase_name]
        if state.completed:
            return False
        state.completed = True
        state.elapsed = state.duration
        print(f"[Timer] {self._phase_label(phase_name)}结束，已累计 {self._format_phase_minutes(state.duration)} 分钟")
        return True

    def _write_sp_state(self, event_name: str):
        archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if not archive_dir:
            return

        try:
            os.makedirs(archive_dir, exist_ok=True)
            payload = {
                "event": event_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "round_index": self.round_index,
                "sp_started_ever": self.sp_started_ever,
                "sp_recording": self.sp_recording,
                "sp_saved": self.sp_saved,
                "last_stage": self.last_stage,
                "active_phase": self.active_phase,
            }
            signal_path = os.path.join(archive_dir, "sp_recording_state.json")
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as exc:
            print(f"[Timer] 写入 sp 状态失败: {exc}")

    def _phase_for_stage(self, stage_name: Optional[str]) -> Optional[str]:
        return self.stage_phase_map.get(stage_name)

    def configure_case_loop_count(self, count: int):
        self.case_loop_count = parse_case_loop_count(count)
        if self.case_loop_index > self.case_loop_count:
            self.case_loop_index = self.case_loop_count
        print(f"[Timer] 单次用例循环次数: {self.case_loop_count}")

    def _reset_phase_progress(self):
        for state in self.phase_states.values():
            state.elapsed = 0.0
            state.started = False
            state.completed = False
        self.last_stage = None
        self.active_phase = None
        self.active_since = None
        self.total_elapsed = 0.0
        self.total_active_since = None
        self.round_index = 0
        self.landed = False
        self.start_game_time = None

    def _effective_elapsed(self, phase_name: str, now: Optional[float] = None) -> float:
        state = self.phase_states[phase_name]
        elapsed = state.elapsed
        if self.active_phase == phase_name and self.active_since is not None and not state.completed:
            now = time.time() if now is None else now
            elapsed += max(0.0, now - self.active_since)
        return elapsed

    def _sync_completed_flag(self, phase_name: str, now: Optional[float] = None) -> bool:
        state = self.phase_states[phase_name]
        if state.completed:
            return False
        if self._effective_elapsed(phase_name, now=now) >= state.duration:
            return self._mark_phase_completed(phase_name)
        return False

    def _pause_active_phase(self, now: Optional[float] = None) -> Set[str]:
        events: Set[str] = set()
        if self.active_phase is None or self.active_since is None:
            if self.total_active_since is not None:
                now = time.time() if now is None else now
                self.total_elapsed = min(
                    self.total_duration,
                    self.total_elapsed + max(0.0, now - self.total_active_since),
                )
                self.total_active_since = None
            return events

        now = time.time() if now is None else now
        if self.total_active_since is not None:
            self.total_elapsed = min(
                self.total_duration,
                self.total_elapsed + max(0.0, now - self.total_active_since),
            )
            self.total_active_since = None

        state = self.phase_states[self.active_phase]
        if not state.completed:
            state.elapsed = min(state.duration, state.elapsed + max(0.0, now - self.active_since))
            if state.elapsed >= state.duration:
                self._mark_phase_completed(self.active_phase)
                events.add(f"completed_{self.active_phase}")

        self.active_phase = None
        self.active_since = None
        return events

    def sync_stage(self, stage_name: Optional[str]) -> Set[str]:
        now = time.time()
        events: Set[str] = set()

        if stage_name == self.last_stage:
            return events

        previous_stage = self.last_stage
        events |= self._pause_active_phase(now=now)
        self.last_stage = stage_name

        if previous_stage == "跳伞阶段" and stage_name in ("跑图阶段", "搜房阶段"):
            self.landed = True
            self.start_game_time = now
            events.add("landed")

        new_phase = self._phase_for_stage(stage_name)
        if new_phase and not self.phase_states[new_phase].completed:
            self.active_phase = new_phase
            self.active_since = now
            self._mark_phase_started(new_phase)
            events.add(f"enter_{new_phase}")
        elif new_phase:
            self.active_phase = None
            self.active_since = None

        if new_phase and self.total_elapsed < self.total_duration:
            self.total_active_since = now

        return events

    def refresh(self) -> Set[str]:
        now = time.time()
        events: Set[str] = set()
        if self.active_phase is None:
            return events

        if self._sync_completed_flag(self.active_phase, now=now):
            events.add(f"completed_{self.active_phase}")
        return events

    def start_new_round(self):
        self.round_index += 1
        self.landed = False
        self.start_game_time = None
        print(f"[Timer] 开始第 {self.case_loop_index}/{self.case_loop_count} 次循环，第 {self.round_index} 局")

    def has_next_case_loop(self) -> bool:
        return self.case_loop_index < self.case_loop_count

    def advance_case_loop(self) -> bool:
        if not self.has_next_case_loop():
            return False

        self._reset_phase_progress()
        self.case_loop_index += 1
        self.sp_recording = False
        self.sp_saved = False
        print(f"[Timer] 准备进入第 {self.case_loop_index}/{self.case_loop_count} 次循环")
        self._write_sp_state("case_loop_advanced")
        return True

    def get_remaining(self, phase_name: str) -> float:
        state = self.phase_states[phase_name]
        return max(0.0, state.duration - self._effective_elapsed(phase_name))

    def get_total_elapsed(self) -> float:
        elapsed = self.total_elapsed
        if self.total_active_since is not None:
            elapsed += max(0.0, time.time() - self.total_active_since)
        return min(self.total_duration, elapsed)

    def get_total_remaining(self) -> float:
        return max(0.0, self.total_duration - self.get_total_elapsed())

    def is_completed(self, phase_name: str) -> bool:
        self._sync_completed_flag(phase_name)
        return self.phase_states[phase_name].completed

    def all_done(self) -> bool:
        return self.get_total_elapsed() >= self.total_duration

    def need_drive(self) -> bool:
        return not self.is_completed(PHASE_DRIVING)

    def should_start_sp(self) -> bool:
        return self.landed and not self.sp_recording and not self.sp_saved

    def get_match_elapsed(self) -> float:
        if self.start_game_time is None:
            return 0.0
        return max(0.0, time.time() - self.start_game_time)

    def mark_sp_started(self):
        self.sp_started_ever = True
        self.sp_recording = True
        print("[Timer] sp 记录已开始")
        self._write_sp_state("sp_started")

    def mark_sp_stopped(self):
        if self.sp_recording:
            print("[Timer] sp 记录已停止")
        self.sp_recording = False
        self._write_sp_state("sp_stopped")

    def mark_sp_saved(self):
        self.sp_saved = True
        print("[Timer] sp 数据已保存")
        self._write_sp_state("sp_saved")


def format_phase_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


class PhaseTimeReporter:
    def __init__(self, report_interval: float = 5.0):
        self.report_interval = float(report_interval)
        self.next_report_time = 0.0
        self.all_done_reported = False

    def reset(self):
        self.next_report_time = 0.0
        self.all_done_reported = False

    def maybe_report(self, timer: PhaseTimeManager):
        if timer.start_game_time is None:
            return

        now = time.time()
        if self.next_report_time <= 0.0:
            self.next_report_time = now + self.report_interval

        if now >= self.next_report_time:
            self._print_remaining(timer)
            self.next_report_time = now + self.report_interval

        if timer.all_done() and not self.all_done_reported:
            self._print_all_done(timer)
            self.all_done_reported = True

    def _remaining_parts(self, timer: PhaseTimeManager) -> str:
        return (
            f"总计={format_phase_seconds(timer.get_total_remaining())} | "
            f"搜房={format_phase_seconds(timer.get_remaining(PHASE_SEARCHING))} | "
            f"跑图={format_phase_seconds(timer.get_remaining(PHASE_RUNNING))} | "
            f"开车={format_phase_seconds(timer.get_remaining(PHASE_DRIVING))}"
        )

    def _print_remaining(self, timer: PhaseTimeManager):
        print(f"[Timer] 阶段剩余时间 | {self._remaining_parts(timer)}")

    def _print_all_done(self, timer: PhaseTimeManager):
        print(f"[Timer] 30 分钟总时长已圆满结束 | {self._remaining_parts(timer)}")
