import time
from dataclasses import dataclass
from typing import Optional


def now() -> float:
    return time.time()


def monotonic_now() -> float:
    return time.monotonic()


@dataclass
class Stopwatch:
    started_at: Optional[float] = None
    monotonic: bool = False

    def _now(self) -> float:
        return monotonic_now() if self.monotonic else now()

    def start(self, started_at: Optional[float] = None) -> float:
        self.started_at = self._now() if started_at is None else float(started_at)
        return self.started_at

    def ensure_started(self, started_at: Optional[float] = None) -> bool:
        if self.started_at is not None:
            return False
        self.start(started_at)
        return True

    def reset(self):
        self.started_at = None

    def is_running(self) -> bool:
        return self.started_at is not None

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self._now() - self.started_at)


class TimeoutTracker:
    def __init__(self, duration: float, *, monotonic: bool = False):
        self.duration = float(duration)
        self.clock = Stopwatch(monotonic=monotonic)
        self.reported = False

    def start(self):
        self.clock.start()
        self.reported = False

    def start_if_needed(self):
        if self.clock.ensure_started():
            self.reported = False

    def reset(self):
        self.clock.reset()
        self.reported = False

    def elapsed(self) -> float:
        return self.clock.elapsed()

    def expired(self) -> bool:
        return self.clock.is_running() and self.elapsed() > self.duration

    def should_report_expired(self) -> bool:
        if not self.expired() or self.reported:
            return False
        self.reported = True
        return True


@dataclass
class Cooldown:
    last_triggered_at: Optional[float] = None

    def reset(self):
        self.last_triggered_at = None

    def ready(self, interval: float) -> bool:
        if self.last_triggered_at is None:
            return True
        return now() - self.last_triggered_at >= float(interval)

    def trigger(self):
        self.last_triggered_at = now()

    def try_acquire(self, interval: float) -> bool:
        if not self.ready(interval):
            return False
        self.trigger()
        return True


@dataclass
class ActiveWindow:
    active_until: float = 0.0

    def reset(self):
        self.active_until = 0.0

    def start(self, duration: float):
        self.active_until = now() + max(0.0, float(duration))

    def active(self) -> bool:
        return now() < self.active_until

    def remaining(self) -> float:
        return max(0.0, self.active_until - now())
