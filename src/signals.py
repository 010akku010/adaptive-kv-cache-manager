"""
signals.py

Lightweight, *always-on* observations of the running system that the
Adaptive Weight Tuner learns from.

Why a separate object from ``metrics.SimMetrics``?
--------------------------------------------------
``SimMetrics`` is for *reporting*: it ignores the warm-up period so that
cold-start effects do not distort results. A tuner is different: it must
keep learning from the very first request, warm-up included. Keeping the two
apart also guarantees that the tuner can only see what a real deployment
could see (counters and timestamps), never the evaluation metrics.

``WindowStats`` is the difference between two snapshots: what happened in
the last ``window`` seconds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
from typing import Deque, List

_COUNTER_FIELDS = (
    "requests",
    "accesses",
    "gpu_hits",
    "cpu_hits",
    "recomputes",
    "compressed_reads",
    "forced_evictions",
    "cpu_evictions",
    "oom_events",
    "pcie_wait_ms",
)


@dataclass(frozen=True)
class CounterSnapshot:
    """Cumulative counter values at one instant."""

    requests: int = 0
    accesses: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    recomputes: int = 0
    compressed_reads: int = 0
    forced_evictions: int = 0
    cpu_evictions: int = 0
    oom_events: int = 0
    pcie_wait_ms: float = 0.0

    def minus(self, earlier: "CounterSnapshot") -> "WindowStats":
        """What happened between ``earlier`` and this snapshot."""
        return WindowStats(**{f: getattr(self, f) - getattr(earlier, f) for f in _COUNTER_FIELDS})


@dataclass(frozen=True)
class WindowStats:
    """Activity over one observation window."""

    requests: int = 0
    accesses: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    recomputes: int = 0
    compressed_reads: int = 0
    forced_evictions: int = 0
    cpu_evictions: int = 0
    oom_events: int = 0
    pcie_wait_ms: float = 0.0

    @property
    def recompute_ratio(self) -> float:
        """Share of block reads that had to be recomputed (true misses)."""
        return self.recomputes / self.accesses if self.accesses else 0.0

    @property
    def gpu_hit_ratio(self) -> float:
        return self.gpu_hits / self.accesses if self.accesses else 0.0

    @property
    def compressed_ratio(self) -> float:
        """Share of block reads served from lossy (compressed) data: the quality-exposure proxy."""
        return self.compressed_reads / self.accesses if self.accesses else 0.0


@dataclass
class Signals:
    """Mutable counters plus a bounded memory of recent block re-use gaps."""

    max_gaps: int = 1024
    requests: int = 0
    accesses: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    recomputes: int = 0
    compressed_reads: int = 0
    forced_evictions: int = 0
    cpu_evictions: int = 0
    oom_events: int = 0
    pcie_wait_ms: float = 0.0
    _gaps: Deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._gaps = deque(self._gaps, maxlen=self.max_gaps)

    # -- recording ---------------------------------------------------------

    def record_request(self) -> None:
        self.requests += 1

    def record_gpu_hit(self, compressed: bool = False) -> None:
        self.accesses += 1
        self.gpu_hits += 1
        if compressed:
            self.compressed_reads += 1

    def record_cpu_hit(self, wait_ms: float, compressed: bool = False) -> None:
        self.accesses += 1
        self.cpu_hits += 1
        self.pcie_wait_ms += wait_ms
        if compressed:
            self.compressed_reads += 1

    def record_recompute(self) -> None:
        self.accesses += 1
        self.recomputes += 1

    def record_forced_eviction(self) -> None:
        self.forced_evictions += 1

    def record_cpu_eviction(self) -> None:
        self.cpu_evictions += 1

    def record_oom(self) -> None:
        self.oom_events += 1

    def record_reuse_gap(self, gap_s: float) -> None:
        """Seconds between two consecutive reads of the same block."""
        if gap_s >= 0.0:
            self._gaps.append(gap_s)

    # -- reading -----------------------------------------------------------

    def snapshot(self) -> CounterSnapshot:
        return CounterSnapshot(**{f: getattr(self, f) for f in _COUNTER_FIELDS})

    def reuse_gaps(self) -> List[float]:
        """Recent re-use gaps (oldest first); bounded to ``max_gaps``."""
        return list(self._gaps)


assert {f.name for f in fields(CounterSnapshot)} == set(_COUNTER_FIELDS)
