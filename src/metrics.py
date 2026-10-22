"""
metrics.py

Metrics collection for the event-driven simulator.

What is reported, and why
-------------------------
* **TTFT percentiles (P50 / P90 / P99).** Time-to-first-token is the
  user-visible latency that KV-cache placement affects. Averages hide the
  tail, so percentiles are reported (literature review, Part 4).
* **Two hit ratios.** ``gpu_hit_ratio`` = share of block reads served
  straight from GPU memory. ``cache_hit_ratio`` = share served *without
  recomputation* (GPU hit or CPU swap-in). The gap between them is the cost
  of using the slower tier.
* **Compressed-read ratio.** The simulator does not measure model quality,
  so this is reported as an *exposure* proxy: the fraction of attention
  reads that were served from lossy (compressed) data.
* **Overheads.** Bytes moved over PCIe, time spent recomputing and
  swapping in, forced evictions (policy failed to make room), and
  out-of-memory events.

A warm-up period can be excluded: while ``measuring`` is False nothing is
recorded, so cold-start effects do not distort steady-state results.

Not modelled (and therefore not reported): request queueing and batching,
so there is no throughput metric yet; GPU compute contention between
concurrent requests.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.placement import PlacementResult


def percentile(data: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (``p`` in [0, 100]); 0.0 for empty input."""
    if not data:
        return 0.0
    ordered = sorted(data)
    rank = max(1, math.ceil((p / 100.0) * len(ordered)))
    return ordered[rank - 1]


def _latency_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean_ms": round(statistics.fmean(values), 4) if values else 0.0,
        f"{prefix}_p50_ms": round(percentile(values, 50), 4),
        f"{prefix}_p90_ms": round(percentile(values, 90), 4),
        f"{prefix}_p99_ms": round(percentile(values, 99), 4),
    }


@dataclass
class SimMetrics:
    """Accumulates everything one simulation run reports."""

    measuring: bool = True

    ttft_ms: List[float] = field(default_factory=list)
    reuse_ttft_ms: List[float] = field(default_factory=list)  # requests on an existing conversation
    requests: int = 0
    reuse_requests: int = 0

    gpu_hits: int = 0
    cpu_hits: int = 0  # served by swapping in from CPU
    recomputes: int = 0  # served by recomputation (a true miss)
    compressed_reads: int = 0
    oom_events: int = 0  # a block could not be given GPU room even after forced eviction

    evictions: int = 0
    migrations: int = 0
    compressions: int = 0
    fallbacks: int = 0  # policy action executed differently than requested
    forced_evictions: int = 0  # evictions the simulator had to force because the policy freed too little
    cpu_evictions: int = 0  # CPU-tier cleanup evictions

    migrated_out_mb: float = 0.0
    swapped_in_mb: float = 0.0
    migrate_out_ms: float = 0.0  # PCIe time spent offloading (D2H), off the request path
    swap_in_ms: float = 0.0
    pcie_wait_ms: float = 0.0  # time requests waited behind other swap-ins
    recompute_ms: float = 0.0

    gpu_util_samples: List[float] = field(default_factory=list)

    # -- recording ---------------------------------------------------------

    def record_request(self, ttft_ms: float, is_reuse: bool) -> None:
        if not self.measuring:
            return
        self.requests += 1
        self.ttft_ms.append(ttft_ms)
        if is_reuse:
            self.reuse_requests += 1
            self.reuse_ttft_ms.append(ttft_ms)

    def record_gpu_hit(self, compressed: bool) -> None:
        if self.measuring:
            self.gpu_hits += 1
            if compressed:
                self.compressed_reads += 1

    def record_cpu_hit(self, size_mb: float, swap_ms: float, wait_ms: float, compressed: bool) -> None:
        if self.measuring:
            self.cpu_hits += 1
            self.swapped_in_mb += size_mb
            self.swap_in_ms += swap_ms
            self.pcie_wait_ms += wait_ms
            if compressed:
                self.compressed_reads += 1

    def record_recompute(self, recompute_ms: float) -> None:
        if self.measuring:
            self.recomputes += 1
            self.recompute_ms += recompute_ms

    def record_oom(self) -> None:
        if self.measuring:
            self.oom_events += 1

    def record_action(self, result: PlacementResult, transfer_ms: float = 0.0) -> None:
        """Record one executed placement action from a policy."""
        if not self.measuring:
            return
        executed = result.executed_action
        if executed == "EVICT":
            self.evictions += 1
        elif executed == "MIGRATE":
            self.migrations += 1
            self.migrated_out_mb += result.freed_gpu_mb
            self.migrate_out_ms += transfer_ms
        elif executed == "COMPRESS":
            self.compressions += 1
        if result.was_fallback and executed != "KEEP":
            self.fallbacks += 1

    def record_forced_eviction(self) -> None:
        if self.measuring:
            self.forced_evictions += 1

    def record_cpu_eviction(self) -> None:
        if self.measuring:
            self.cpu_evictions += 1

    def sample_gpu_util(self, utilization: float) -> None:
        if self.measuring:
            self.gpu_util_samples.append(utilization)

    # -- reporting ---------------------------------------------------------

    @property
    def accesses(self) -> int:
        return self.gpu_hits + self.cpu_hits + self.recomputes

    def summary(self) -> Dict[str, Any]:
        n = self.accesses or 1
        util = self.gpu_util_samples
        out: Dict[str, Any] = {
            "requests": self.requests,
            "reuse_requests": self.reuse_requests,
            **_latency_stats(self.ttft_ms, "ttft"),
            **_latency_stats(self.reuse_ttft_ms, "reuse_ttft"),
            "accesses": self.accesses,
            "gpu_hit_ratio": round(self.gpu_hits / n, 4),
            "cache_hit_ratio": round((self.gpu_hits + self.cpu_hits) / n, 4),
            "recompute_ratio": round(self.recomputes / n, 4),
            "compressed_read_ratio": round(self.compressed_reads / n, 4),
            "evictions": self.evictions,
            "migrations": self.migrations,
            "compressions": self.compressions,
            "fallbacks": self.fallbacks,
            "forced_evictions": self.forced_evictions,
            "cpu_evictions": self.cpu_evictions,
            "oom_events": self.oom_events,
            "migrated_out_mb": round(self.migrated_out_mb, 2),
            "swapped_in_mb": round(self.swapped_in_mb, 2),
            "migrate_out_ms": round(self.migrate_out_ms, 2),
            "swap_in_ms": round(self.swap_in_ms, 2),
            "pcie_wait_ms": round(self.pcie_wait_ms, 2),
            "recompute_ms": round(self.recompute_ms, 2),
            "mean_gpu_util": round(statistics.fmean(util), 4) if util else 0.0,
            "p95_gpu_util": round(percentile(util, 95), 4),
            "saturation_ratio": round(sum(1 for u in util if u >= 0.95) / len(util), 4) if util else 0.0,
        }
        return out
