"""
performance_monitor.py

The Performance Monitor: the module responsible for turning raw
per-request events (which tier a block was found in, which action the
Decision Engine took) into the metrics the project promises to report:
GPU memory utilization, cache hit ratio, time-to-first-token (TTFT),
throughput, and migration overhead (Zeroth Review, Slide 9/12).

Key terms (used throughout this module and the evaluation harness):

- Cache hit / miss: a "hit" means the KV-cache block a request needs is
  already available (GPU or CPU); a "miss" means it was evicted and must
  be recomputed from scratch, which is far more expensive.
- TTFT (time-to-first-token): the latency between a request arriving and
  the model producing its first output token. KV-cache placement affects
  this directly, because a GPU-resident block is read almost instantly, a
  CPU-resident one needs a migration-back transfer first, and a missing
  (evicted) one needs full recomputation.
- Percentile latency (P50 / P90 / P99): the latency value below which
  50% / 90% / 99% of requests fall. Reporting only the *average* latency
  hides how bad the worst 1-10% of requests are, which is exactly the
  "misleading normalization" anti-pattern the literature review (Part 4)
  warns against -- so this module reports percentiles, not just a mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.metadata_collector import Tier

# Simulated per-access latency (milliseconds) by where the block was
# found. These stand in for real measurements until vLLM integration:
# GPU access is near-instant; a CPU-resident block needs a migrate-back
# transfer; an evicted block must be recomputed from the prompt, which is
# by far the most expensive case.
LATENCY_MS_BY_TIER = {
    Tier.GPU: 2.0,
    Tier.CPU: 12.0,
    Tier.EVICTED: 80.0,  # recompute penalty
}


@dataclass
class PerformanceMonitor:
    """Accumulates access latencies and decision counts for one run."""

    latencies_ms: List[float] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0  # actions executed differently than requested (e.g. CPU full)
    action_counts: Dict[str, int] = field(default_factory=lambda: {
        "KEEP": 0, "MIGRATE": 0, "COMPRESS": 0, "EVICT": 0,
    })

    def record_access(self, tier: Tier) -> float:
        """
        Record a simulated read of a block currently in `tier`. Returns
        the simulated latency (ms) so callers can log/print it if useful.
        A GPU or CPU hit counts as a "hit"; an EVICTED lookup is a "miss"
        (the block must be recomputed).
        """
        latency = LATENCY_MS_BY_TIER[tier]
        self.latencies_ms.append(latency)
        if tier == Tier.EVICTED:
            self.misses += 1
        else:
            self.hits += 1
        return latency

    def record_decision(self, action: str) -> None:
        """Record which action the Decision Engine (or baseline) chose."""
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def record_fallback(self) -> None:
        """Record that an action had to be downgraded (e.g. MIGRATE -> EVICT)."""
        self.fallbacks += 1

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def percentile(self, p: float) -> float:
        """
        Nearest-rank percentile of recorded latencies. `p` is in [0, 100].
        e.g. percentile(50) is the median, percentile(99) is the P99 tail
        latency that matters for SLO-style reporting.
        """
        if not self.latencies_ms:
            return 0.0
        data = sorted(self.latencies_ms)
        rank = max(1, math.ceil((p / 100.0) * len(data)))
        return data[rank - 1]

    def average_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def summary(self) -> Dict[str, Any]:
        """Single dict of everything the evaluation harness needs to report."""
        return {
            "requests": len(self.latencies_ms),
            "hit_ratio": round(self.hit_ratio(), 4),
            "avg_latency_ms": round(self.average_latency_ms(), 3),
            "p50_latency_ms": round(self.percentile(50), 3),
            "p90_latency_ms": round(self.percentile(90), 3),
            "p99_latency_ms": round(self.percentile(99), 3),
            "actions": dict(self.action_counts),
            "fallbacks": self.fallbacks,
        }
