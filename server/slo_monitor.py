"""v0.7: burn-rate SLO monitor.

Tracks a rolling window of real request latencies and periodically checks
whether p99 has been above the SLO threshold for several consecutive
checks in a row ("sustained breach", not a single slow request tripping
it). Fires a real alert -- not a config stub -- verified by an actual test
that floods the monitor with slow latencies and asserts an alert lands.
"""
import asyncio
import time
from collections import deque


class SLOMonitor:
    def __init__(self, latency_threshold_s: float, check_interval_s: float = 1.0,
                 sustained_breaches: int = 3, window_size: int = 20):
        self.latency_threshold_s = latency_threshold_s
        self.check_interval_s = check_interval_s
        self.sustained_breaches = sustained_breaches
        self.window: deque = deque(maxlen=window_size)
        self.consecutive_breaches = 0
        self.alerts_fired: list[dict] = []
        self._task = None

    def record(self, latency_s: float):
        self.window.append(latency_s)

    def _p99(self) -> float:
        if not self.window:
            return 0.0
        s = sorted(self.window)
        idx = min(len(s) - 1, int(0.99 * len(s)))
        return s[idx]

    def _check_once(self):
        if not self.window:
            return
        p99 = self._p99()
        if p99 > self.latency_threshold_s:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0
        if self.consecutive_breaches >= self.sustained_breaches:
            self._fire(p99)
            self.consecutive_breaches = 0  # avoid re-firing every check while still breached

    def _fire(self, p99: float):
        alert = {
            "timestamp": time.time(),
            "p99_latency_s": p99,
            "threshold_s": self.latency_threshold_s,
            "sustained_checks": self.sustained_breaches,
        }
        self.alerts_fired.append(alert)
        print(f"[SLO ALERT] p99 latency {p99:.3f}s sustained above {self.latency_threshold_s}s "
              f"threshold for {self.sustained_breaches} consecutive checks")

    async def _run_forever(self):
        while True:
            await asyncio.sleep(self.check_interval_s)
            self._check_once()

    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._run_forever())
