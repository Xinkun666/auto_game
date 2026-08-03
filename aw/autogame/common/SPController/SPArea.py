import json
import os
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


SP_SAVE_LONG_PRESS_MS = 3000
MARATHON_DURATION_ENV = "AUTOGAME_MARATHON_DURATION_MINUTES"
SP_CONTROLLER_STATE_FILE = "sp_controller_state.json"


def parse_marathon_duration_minutes(value: Any) -> float:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, minutes)


class SPControllerBase:
    """管理 SP 录制的启动、暂停、恢复和保存。"""

    def __init__(self, w: "FrameWorker", marathon_duration_minutes: Optional[float] = None):
        self.w = w
        if marathon_duration_minutes is None:
            marathon_duration_minutes = os.environ.get(MARATHON_DURATION_ENV, "")
        self._target_duration_seconds = (
            parse_marathon_duration_minutes(marathon_duration_minutes) * 60.0
        )
        self._start_time: Optional[float] = None
        self._paused_time = 0.0
        self._pause_start: Optional[float] = None
        self._is_paused = False
        self._area: Any = None
        self._effective_time_at_stop: Optional[float] = None
        self._started_ever = False

    @property
    def area(self):
        return self._area

    @property
    def is_paused(self):
        return self._is_paused

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
        self.w.frame_log("找不到SP")

    def snapshot(self, event_name: str = "snapshot"):
        stopped = self._effective_time_at_stop is not None
        started = self._start_time is not None
        return {
            "event": event_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sp_started_ever": self._started_ever,
            "sp_recording": started and not stopped and not self._is_paused,
            "sp_paused": started and not stopped and self._is_paused,
            "sp_saved": stopped,
            "marathon_enabled": self.marathon_enabled,
            "target_duration_seconds": self.target_duration_seconds,
            "effective_time_seconds": self.effective_time,
            "remaining_time_seconds": self.remaining_time,
            "target_reached": self.target_reached,
        }

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
            self.w.frame_log(f"写入 SP 控制状态失败: {exc}")

    # 手动设置area
    def set_area(self, sp_area, force=False):
        if self._area is None or force:
            self._area = sp_area

    def start(self, sp_area_name=None):
        if self._is_paused:
            return self.resume()
        if self._start_time is not None and self._effective_time_at_stop is None:
            return True
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
        self.w.frame_log("sp start")
        self._write_state("sp_started")
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
        self.w.frame_log("sp paused")
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
        self.w.frame_log("sp resumed")
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
        self.w.frame_log("sp end")
        self._write_state("sp_saved")
        return True
