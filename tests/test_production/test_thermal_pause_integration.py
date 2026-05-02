"""Thermal pause + training-loop integration tests.

Production scenario: a long training run hits sustained throttling on
an air-cooled MacBook. ThermalPauseController must:
  - pause when SERIOUS or CRITICAL pressure observed
  - hold the pause for at least min_pause_sec even if pressure drops fast
  - resume once pressure returns to <= resume_at
  - never engage when the OS thermal probes are reporting NOMINAL

Hysteresis is critical — without it the FSM oscillates per second on
boundary noise.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.thermal import ThermalState
from macfleet.monitoring.thermal_pause import (
    PauseState,
    ThermalPauseConfig,
    ThermalPauseController,
)


class _FakeReader:
    def __init__(self, pressures: list[ThermalPressure]):
        self.pressures = pressures
        self.calls = 0

    def __call__(self) -> ThermalState:
        idx = min(self.calls, len(self.pressures) - 1)
        self.calls += 1
        return ThermalState(pressure=self.pressures[idx])


class TestPauseEngagesUnderLoad:
    """Sustained SERIOUS pressure must engage the pause."""

    def test_sustained_serious_pauses_within_one_tick(self):
        reader = _FakeReader([
            ThermalPressure.NOMINAL,
            ThermalPressure.SERIOUS,
        ])
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert not ctrl.is_paused()
        ctrl.tick()
        assert ctrl.is_paused()


class TestHysteresisPreventsOscillation:
    """Boundary noise (SERIOUS → FAIR → SERIOUS → FAIR) must not cause
    flapping under hysteresis."""

    def test_no_oscillation_at_boundary(self):
        # Alternating SERIOUS / FAIR — should pause once, stay paused
        # until FAIR is sustained for at least min_pause_sec.
        sequence = [
            ThermalPressure.SERIOUS,  # tick 1: PAUSED
            ThermalPressure.FAIR,     # tick 2: still PAUSED (min_pause_sec)
            ThermalPressure.SERIOUS,  # tick 3: still PAUSED
            ThermalPressure.FAIR,     # tick 4: now eligible to resume
        ]
        reader = _FakeReader(sequence)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.05,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()
        ctrl.tick()  # FAIR but min_pause_sec hasn't elapsed
        assert ctrl.is_paused()
        time.sleep(0.06)  # past min_pause_sec
        ctrl.tick()  # SERIOUS — keeps paused
        assert ctrl.is_paused()
        ctrl.tick()  # FAIR — eligible, resumes
        assert not ctrl.is_paused()


class TestNominalDoesNotPause:
    def test_nominal_never_pauses(self):
        reader = _FakeReader([ThermalPressure.NOMINAL] * 10)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        for _ in range(10):
            ctrl.tick()
        assert ctrl.state == PauseState.RUNNING

    def test_fair_below_pause_threshold(self):
        reader = _FakeReader([ThermalPressure.FAIR] * 10)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        for _ in range(10):
            ctrl.tick()
        assert ctrl.state == PauseState.RUNNING


class TestAsyncWaitForResume:
    """Async training loops use async_wait_for_resume — verify it yields
    the loop while polling."""

    @pytest.mark.asyncio
    async def test_async_wait_returns_when_pressure_drops(self):
        # Start at SERIOUS, drop to NOMINAL after 100ms in another task.
        reader = _FakeReader([
            ThermalPressure.SERIOUS, ThermalPressure.SERIOUS,
            ThermalPressure.NOMINAL, ThermalPressure.NOMINAL,
        ])
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.05, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()  # First tick: NOMINAL? No — first reading is SERIOUS.
        assert ctrl.is_paused()

        start = time.monotonic()
        # async_wait_for_resume polls the FakeReader. After 2 calls
        # the FakeReader returns NOMINAL.
        result = await ctrl.async_wait_for_resume(timeout_sec=2.0)
        elapsed = time.monotonic() - start
        assert result is True
        assert not ctrl.is_paused()
        # Should have resumed quickly (within a few poll intervals).
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_async_wait_times_out_if_pressure_stays_high(self):
        reader = _FakeReader([ThermalPressure.CRITICAL] * 100)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.02, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()
        result = await ctrl.async_wait_for_resume(timeout_sec=0.2)
        assert result is False
        assert ctrl.is_paused()

    @pytest.mark.asyncio
    async def test_async_wait_does_not_block_event_loop(self):
        """Other coroutines must continue to run during async_wait_for_resume."""
        reader = _FakeReader([
            ThermalPressure.SERIOUS, ThermalPressure.NOMINAL,
        ])
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.05, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()

        ticks = []

        async def background_ticker():
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks.append(time.monotonic())

        bg = asyncio.create_task(background_ticker())
        await ctrl.async_wait_for_resume(timeout_sec=2.0)
        await bg

        # The background coroutine made progress while we were waiting.
        assert len(ticks) == 10


class TestPauseCallbacks:
    """on_pause / on_resume callbacks must fire exactly once per transition."""

    def test_on_pause_fires_on_engage(self):
        events = []
        reader = _FakeReader([
            ThermalPressure.NOMINAL, ThermalPressure.SERIOUS,
            ThermalPressure.SERIOUS,
        ])
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
            on_pause=lambda evt: events.append(("pause", evt.state)),
        )
        ctrl.tick()
        ctrl.tick()  # transition NOMINAL → SERIOUS
        ctrl.tick()  # already paused — should NOT fire again
        assert len(events) == 1
        assert events[0] == ("pause", PauseState.PAUSED)

    def test_on_resume_fires_on_release(self):
        events = []
        reader = _FakeReader([
            ThermalPressure.SERIOUS, ThermalPressure.NOMINAL,
            ThermalPressure.NOMINAL,
        ])
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
            on_resume=lambda evt: events.append(("resume", evt.state)),
        )
        ctrl.tick()  # PAUSED
        ctrl.tick()  # NOMINAL — RUNNING
        ctrl.tick()  # already running — should NOT fire again
        assert len(events) == 1


class TestThermalReadFailureSafety:
    def test_holds_state_on_probe_failure(self):
        def boom():
            raise RuntimeError("pmset hiccup")
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=boom,
        )
        ctrl.tick()
        assert ctrl.state == PauseState.RUNNING
