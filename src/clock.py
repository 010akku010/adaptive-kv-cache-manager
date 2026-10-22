"""
clock.py

A tiny time abstraction so that every time-dependent computation in the
project (block recency, block age, workload generation) reads "now" from
an injected clock instead of calling ``time.time()`` directly.

Why this matters
----------------
* **Reproducibility.** With a ``ManualClock`` a simulation run is a pure
  function of its configuration and random seed: running it twice gives
  bit-identical results. Experiments, figures and unit tests in the final
  report can therefore be regenerated exactly.
* **Testability.** Tests can move time forward explicitly (``advance``)
  instead of sleeping or back-dating timestamps by hand.
* **Production.** ``SystemClock`` keeps the real wall-clock behaviour for
  the eventual vLLM integration, with no change to calling code.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Anything that can report the current time in seconds."""

    def now(self) -> float:
        ...


class SystemClock:
    """Real wall-clock time (Unix seconds). Default for production use."""

    def now(self) -> float:
        return time.time()


class ManualClock:
    """
    Deterministic clock whose time only changes when told to.

    Args:
        start: initial time in seconds.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        """Move time forward by ``seconds`` (must be >= 0); returns new time."""
        if seconds < 0:
            raise ValueError("ManualClock cannot move backwards")
        self._now += seconds
        return self._now

    def set(self, timestamp: float) -> None:
        """Jump to an absolute time (must not be earlier than the current time)."""
        if timestamp < self._now:
            raise ValueError("ManualClock cannot move backwards")
        self._now = float(timestamp)
