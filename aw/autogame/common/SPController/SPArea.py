import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from aw.autogame.tools.FrameLog import FrameLogType

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


SP_SAVE_LONG_PRESS_MS = 3000
SP_DEFAULT_NORM_POSITION = (0.048, 0.295)
MARATHON_DURATION_ENV = "AUTOGAME_MARATHON_DURATION_MINUTES"
MARATHON_END_BATTERY_ENV = "AUTOGAME_MARATHON_END_BATTERY_PERCENT"
SP_CONTROLLER_STATE_FILE = "sp_controller_state.json"
BATTERY_LOG_FILE = "battery.log"
BATTERY_POLL_INTERVAL_SECONDS = 60.0


def parse_marathon_duration_minutes(value: Any) -> float:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, minutes)


def parse_battery_percent(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        percent = int(float(str(value).strip().rstrip("%")))
    except (TypeError, ValueError):
        return None
    return percent if 0 <= percent <= 100 else None


def parse_marathon_end_battery_percent(value: Any) -> int:
    percent = parse_battery_percent(value)
    return percent if percent is not None and percent > 0 else 0


def build_sp_save_shell_command(
    screen_width: int,
    screen_height: int,
    norm_position=SP_DEFAULT_NORM_POSITION,
    duration_ms: int = SP_SAVE_LONG_PRESS_MS,
):
    x = int(round(int(screen_width) * float(norm_position[0])))
    y = int(round(int(screen_height) * float(norm_position[1])))
    duration_ms = max(1, int(duration_ms))
    command = f"uinput -T -d {x} {y} -i {duration_ms} -u {x} {y}"
    return command, x, y, duration_ms


class SPControllerBase:
    """管理 SP 录制的启动、暂停、恢复和保存。"""

    def __init__(
        self,
        w: "FrameWorker",
        marathon_duration_minutes: Optional[float] = None,
        marathon_end_battery_percent: Optional[int] = None,
        battery_poll_interval_seconds: float = BATTERY_POLL_INTERVAL_SECONDS,
    ):
        self.w = w
        if marathon_duration_minutes is None:
            marathon_duration_minutes = os.environ.get(MARATHON_DURATION_ENV, "")
        self._target_duration_seconds = (
            parse_marathon_duration_minutes(marathon_duration_minutes) * 60.0
        )
        if marathon_end_battery_percent is None:
            marathon_end_battery_percent = os.environ.get(MARATHON_END_BATTERY_ENV, "")
        self._end_battery_percent = parse_marathon_end_battery_percent(
            marathon_end_battery_percent
        )
        self._battery_poll_interval_seconds = max(
            0.01,
            float(battery_poll_interval_seconds),
        )
        self._start_time: Optional[float] = None
        self._paused_time = 0.0
        self._pause_start: Optional[float] = None
        self._is_paused = False
        self._area: Any = None
        self._effective_time_at_stop: Optional[float] = None
        self._started_ever = False
        self._battery_monitor_stop_event = threading.Event()
        self._battery_stop_requested_event = threading.Event()
        self._battery_monitor_thread: Optional[threading.Thread] = None
        self._battery_log_lock = threading.Lock()
        self._last_battery_percent: Optional[int] = None
        self._battery_stop_requested_at = ""

    @property
    def area(self):
        return self._area

    @property
    def is_paused(self):
        return self._is_paused

    @property
    def is_stoped(self):
        return self._effective_time_at_stop is not None

    @property
    def has_started(self):
        return self._started_ever

    @property
    def is_recording(self):
        return (
            self._start_time is not None
            and self._effective_time_at_stop is None
            and not self._is_paused
        )

    @property
    def is_saved(self):
        return self._effective_time_at_stop is not None

    @property
    def marathon_enabled(self):
        return self._target_duration_seconds > 0

    @property
    def target_duration_seconds(self):
        return self._target_duration_seconds

    @property
    def target_reached(self):
        return self.marathon_enabled and self.effective_time >= self._target_duration_seconds

    @property
    def remaining_time(self):
        if not self.marathon_enabled:
            return 0.0
        return max(0.0, self._target_duration_seconds - self.effective_time)

    @property
    def end_battery_percent(self):
        return self._end_battery_percent

    @property
    def last_battery_percent(self):
        return self._last_battery_percent

    @property
    def battery_stop_requested(self):
        return self._battery_stop_requested_event.is_set()

    @property
    def effective_time(self):
        if self._effective_time_at_stop is not None:
            return self._effective_time_at_stop
        if self._start_time is None:
            return 0.0

        now = time.monotonic()
        total = now - self._start_time
        if self._is_paused and self._pause_start is not None:
            total -= now - self._pause_start
        return max(0.0, total - self._paused_time)

    @staticmethod
    def _control_executed(result):
        if not isinstance(result, dict):
            return True
        return str(result.get("executed", "True")).strip().lower() != "false"

    def _log_missing(self):
        self._frame_log("找不到SP")

    def _frame_log(self, message: str):
        try:
            self.w.frame_log(message, log_type=FrameLogType.TIME)
        except TypeError as exc:
            if "log_type" not in str(exc) and "keyword" not in str(exc):
                raise
            self.w.frame_log(message)

    def snapshot(self, event_name: str = "snapshot"):
        stopped = self._effective_time_at_stop is not None
        started = self._start_time is not None
        return {
            "event": event_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sp_started_ever": self.has_started,
            "sp_recording": self.is_recording,
            "sp_paused": started and not stopped and self._is_paused,
            "sp_saved": self.is_saved,
            "marathon_enabled": self.marathon_enabled,
            "target_duration_seconds": self.target_duration_seconds,
            "effective_time_seconds": self.effective_time,
            "remaining_time_seconds": self.remaining_time,
            "target_reached": self.target_reached,
            "end_battery_percent": self.end_battery_percent,
            "last_battery_percent": self.last_battery_percent,
            "battery_stop_requested": self.battery_stop_requested,
            "battery_stop_requested_at": self._battery_stop_requested_at,
        }

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _resolve_battery_log_path(self) -> Optional[Path]:
        batch_dir = os.environ.get("AUTOGAME_BATCH_ARCHIVE_DIR", "").strip()
        if batch_dir:
            return Path(batch_dir) / BATTERY_LOG_FILE
        run_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if run_dir:
            return Path(run_dir).parent / BATTERY_LOG_FILE
        return None

    def _append_battery_log(self, message: str):
        log_path = self._resolve_battery_log_path()
        if log_path is None:
            return
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            run_index = os.environ.get("AUTOGAME_RUN_INDEX", "").strip() or "-"
            line = f"[{self._timestamp()}] run={run_index} {str(message).strip()}\n"
            with self._battery_log_lock:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            print(f"[SPController] 写入 battery.log 失败: {exc}")

    def _read_battery_percent(self) -> Optional[int]:
        try:
            getter = self.w.driver.System.get_battery_percent
            return parse_battery_percent(getter())
        except Exception as exc:
            self._append_battery_log(f"battery=unknown error={exc}")
            return None

    def _request_battery_stop(self, percent: int):
        if self._battery_stop_requested_event.is_set():
            return
        self._battery_stop_requested_at = self._timestamp()
        self._battery_stop_requested_event.set()
        self._append_battery_log(
            f"end_battery_reached battery={percent}% "
            f"threshold={self.end_battery_percent}%"
        )
        self._write_state("end_battery_reached")

    def _battery_monitor_loop(self):
        threshold_text = (
            f"{self.end_battery_percent}%"
            if self.end_battery_percent > 0
            else "disabled"
        )
        self._append_battery_log(
            f"monitor_started end_battery={threshold_text} "
            f"sp_target_minutes={self.target_duration_seconds / 60:g}"
        )
        while not self._battery_monitor_stop_event.is_set():
            percent = self._read_battery_percent()
            if percent is not None:
                self._last_battery_percent = percent
                self._append_battery_log(f"battery={percent}%")
                if (
                    self._started_ever
                    and self.end_battery_percent > 0
                    and percent <= self.end_battery_percent
                ):
                    self._request_battery_stop(percent)
                    return
            if self._battery_monitor_stop_event.wait(self._battery_poll_interval_seconds):
                return

    def start_battery_monitor(self):
        if not self.marathon_enabled:
            return
        thread = self._battery_monitor_thread
        if thread is not None and thread.is_alive():
            return
        self._battery_monitor_stop_event.clear()
        self._battery_monitor_thread = threading.Thread(
            target=self._battery_monitor_loop,
            daemon=True,
            name="MarathonBatteryMonitor",
        )
        self._battery_monitor_thread.start()

    def shutdown(self):
        self._battery_monitor_stop_event.set()
        thread = self._battery_monitor_thread
        if (
            thread is not None
            and thread.is_alive()
            and threading.current_thread() is not thread
        ):
            thread.join(timeout=1.0)
        self._battery_monitor_thread = None

    def _write_state(self, event_name: str):
        archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if not archive_dir:
            return
        try:
            os.makedirs(archive_dir, exist_ok=True)
            signal_path = os.path.join(archive_dir, SP_CONTROLLER_STATE_FILE)
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(event_name), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as exc:
            print(f"[SPController] 写入 SP 控制状态失败: {exc}")

    # 手动设置area
    def set_area(self, sp_area, force=False):
        if self._area is None or force:
            self._area = sp_area

    def start(self, sp_area_name=None):
        if self._is_paused:
            return self.resume()
        if self._start_time is not None:  # 已经启动过就不再启动
            return True
        if self._effective_time_at_stop is not None:  # 已经停止后sp就退出了，需要直接返回
            self._log_missing()
            return False
        if sp_area_name is not None:
            # sp_area_name 只能是sp区域名。
            sp_area = self.w.get_info(sp_area_name)
            if sp_area:
                self._area = sp_area
            else:
                self._area = sp_area_name
        result = self.w.click(self._area)
        if not self._control_executed(result):
            self._area = None
            self._log_missing()
            return False

        self._start_time = time.monotonic()
        self._paused_time = 0.0
        self._pause_start = None
        self._is_paused = False
        self._effective_time_at_stop = None
        self._started_ever = True
        self._frame_log("sp 记录已开始")
        self._write_state("sp_started")
        self.start_battery_monitor()
        if (
            self.last_battery_percent is not None
            and self.end_battery_percent > 0
            and self.last_battery_percent <= self.end_battery_percent
        ):
            self._request_battery_stop(self.last_battery_percent)
        return True

    def pause(self):
        if self._area is None or self._start_time is None or self._effective_time_at_stop is not None:
            self._log_missing()
            return False
        if self._is_paused:
            return True

        result = self.w.click(self._area)
        if not self._control_executed(result):
            self._log_missing()
            return False

        self._is_paused = True
        self._pause_start = time.monotonic()
        self._frame_log("sp 记录已暂停")
        self._write_state("sp_paused")
        return True

    def resume(self):
        if self._area is None or not self._is_paused:
            return False

        result = self.w.click(self._area)
        if not self._control_executed(result):
            self._log_missing()
            return False

        now = time.monotonic()
        if self._pause_start is not None:
            self._paused_time += now - self._pause_start
        self._is_paused = False
        self._pause_start = None
        self._frame_log("sp 记录已恢复")
        self._write_state("sp_resumed")
        return True

    def get_effective_time(self):
        return self.effective_time

    def stop(self):
        if self._area is None or self._start_time is None or self._effective_time_at_stop is not None:
            self._log_missing()
            return False

        result = self.w.click_down(self._area, dura=SP_SAVE_LONG_PRESS_MS)
        if not self._control_executed(result):
            self._log_missing()
            return False

        self._effective_time_at_stop = self.effective_time
        self._is_paused = False
        self._pause_start = None
        self._frame_log("sp 数据已保存")
        self._write_state("sp_saved")
        self.shutdown()
        return True
