import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


SP_SAVE_LONG_PRESS_MS = 3000


class SPControllerBase:
    """管理 SP 录制的启动、暂停、恢复和保存。"""

    def __init__(self, w: "FrameWorker"):
        self.w = w
        self._start_time: Optional[float] = None
        self._paused_time = 0.0
        self._pause_start: Optional[float] = None
        self._is_paused = False
        self._area: Any = None
        self._effective_time_at_stop: Optional[float] = None

    @property
    def area(self):
        return self._area

    @property
    def is_paused(self):
        return self._is_paused

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

    def start(self, sp_area_name):
        if self._is_paused:
            return self.resume()
        if self._start_time is not None and self._effective_time_at_stop is None:
            return True

        # sp_area_name 只能是sp区域名。
        sp_area = self.w.get_info(sp_area_name)
        if sp_area:
            result = self.w.click(sp_area)
        else:
            sp_area = sp_area_name
            result = self.w.click(sp_area_name)

        if not self._control_executed(result):
            self._area = None
            self._log_missing()
            return False

        self._area = sp_area
        self._start_time = time.monotonic()
        self._paused_time = 0.0
        self._pause_start = None
        self._is_paused = False
        self._effective_time_at_stop = None
        self.w.frame_log("sp start")
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
        return True
