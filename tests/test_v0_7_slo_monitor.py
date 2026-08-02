"""v0.7: the burn-rate alert has to actually fire in a test, not just exist
as config. These tests run the real SLOMonitor background loop (real
asyncio task, real timing, just a short interval so the test finishes
quickly) and assert on real fired alerts.
"""
import asyncio

import pytest

from server.slo_monitor import SLOMonitor


@pytest.mark.asyncio
async def test_sustained_breach_fires_alert():
    mon = SLOMonitor(latency_threshold_s=1.0, check_interval_s=0.02,
                      sustained_breaches=3, window_size=10)
    mon.start()

    for _ in range(10):
        mon.record(5.0)  # way above the 1.0s threshold, every request

    await asyncio.sleep(0.15)  # >= 3 check intervals with margin
    assert len(mon.alerts_fired) >= 1, "sustained breach should have fired at least one alert"
    assert mon.alerts_fired[0]["p99_latency_s"] > 1.0


@pytest.mark.asyncio
async def test_healthy_latencies_never_fire():
    mon = SLOMonitor(latency_threshold_s=1.0, check_interval_s=0.02,
                      sustained_breaches=3, window_size=10)
    mon.start()

    for _ in range(10):
        mon.record(0.1)  # well under threshold

    await asyncio.sleep(0.15)
    assert mon.alerts_fired == [], "no alert should fire when latencies are healthy"


@pytest.mark.asyncio
async def test_single_slow_request_does_not_fire_alone():
    """One slow request shouldn't trip the alert -- it has to be sustained."""
    mon = SLOMonitor(latency_threshold_s=1.0, check_interval_s=0.02,
                      sustained_breaches=3, window_size=10)
    mon.start()

    for _ in range(10):
        mon.record(0.1)
    mon.record(5.0)  # one spike

    await asyncio.sleep(0.03)  # only ~1 check interval, not enough for 3 sustained breaches
    assert mon.alerts_fired == [], "a single check-cycle breach shouldn't fire the sustained alert"
