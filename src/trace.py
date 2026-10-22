"""
trace.py

Request-trace generator for the event-driven simulator.

A trace is a *policy-independent* list of timestamped requests, generated
once from a seed and then replayed identically against every policy. That
independence is what removes the circularity of the earlier snapshot
harness (where the replayed accesses were derived from the very
``access_count`` that our own importance score consumes).

Workload model (a conversation / prefix-reuse workload)
-------------------------------------------------------
* There is a fixed population of ``num_contexts`` conversations. Each has
  an initial length (log-normal, in blocks) and **grows** as turns are
  added.
* Each request targets one conversation and needs *all* of its existing
  KV blocks (the prefix) plus a few new ones (the new turn).
* Which conversation a request targets follows a **Zipf** law: a few
  conversations are very hot, most are rarely revisited, which is the
  shape reported for real chat traffic (ShareGPT-style). ``zipf_s = 0``
  gives uniform popularity.
* Arrivals are a **Poisson** process, or a **bursty** on/off Markov-
  modulated Poisson process that keeps the same mean rate.

This is a *model* of reuse structure, not a replay of real data. Replaying
public traces (ShareGPT / LMSYS-Chat / BurstGPT) is a later milestone; the
generator is deliberately parameterised so it can be fitted to them.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import Dict, List


@dataclass(frozen=True)
class WorkloadConfig:
    name: str = "zipf"
    duration_s: float = 400.0
    arrival_rate: float = 5.0  # mean requests per second
    arrival: str = "poisson"  # "poisson" | "bursty"
    # Bursty parameters (ignored for poisson): mean rate is preserved.
    burst_rate_multiplier: float = 4.0
    burst_on_mean_s: float = 10.0
    burst_off_mean_s: float = 30.0
    # Conversation population and popularity.
    num_contexts: int = 200
    zipf_s: float = 1.0  # 0 = uniform popularity
    mean_context_blocks: float = 40.0
    context_sigma: float = 0.6  # log-normal spread of initial context length
    min_context_blocks: int = 4
    max_context_blocks: int = 256
    # Per-request growth and generation.
    growth_blocks_mean: float = 2.0
    mean_output_tokens: float = 64.0
    seed: int = 0

    def with_seed(self, seed: int) -> "WorkloadConfig":
        return replace(self, seed=seed)


@dataclass(frozen=True)
class Request:
    request_id: int
    time_s: float
    context_id: int
    new_blocks: int
    output_tokens: int


@dataclass
class Trace:
    config: WorkloadConfig
    requests: List[Request] = field(default_factory=list)
    initial_blocks: List[int] = field(default_factory=list)  # per context id

    @property
    def total_initial_blocks(self) -> int:
        return sum(self.initial_blocks)


# ---------------------------------------------------------------------------
# Presets used by the experiments
# ---------------------------------------------------------------------------

WORKLOADS: Dict[str, WorkloadConfig] = {
    # Skewed reuse, steady load: the "typical chat" case.
    "zipf": WorkloadConfig(name="zipf", zipf_s=1.0),
    # No popularity skew: every conversation equally likely (hardest for any
    # recency/frequency heuristic; there is little to learn).
    "uniform": WorkloadConfig(name="uniform", zipf_s=0.0),
    # Same skew, but traffic arrives in bursts (4x rate for ~10 s at a time).
    "bursty": WorkloadConfig(name="bursty", zipf_s=1.0, arrival="bursty"),
    # Few, very long conversations (document / RAG-style): big blocks sets, high pressure.
    "long_context": WorkloadConfig(
        name="long_context", zipf_s=0.8, num_contexts=60, mean_context_blocks=160.0,
        max_context_blocks=512, arrival_rate=2.0, growth_blocks_mean=3.0,
    ),
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _arrival_times(cfg: WorkloadConfig, rng: random.Random) -> List[float]:
    times: List[float] = []
    if cfg.arrival == "poisson":
        t = 0.0
        while True:
            t += rng.expovariate(cfg.arrival_rate)
            if t >= cfg.duration_s:
                return times
            times.append(t)

    if cfg.arrival == "bursty":
        on_mean, off_mean = cfg.burst_on_mean_s, cfg.burst_off_mean_s
        p_on = on_mean / (on_mean + off_mean)
        hi = cfg.arrival_rate * cfg.burst_rate_multiplier
        lo = (cfg.arrival_rate - p_on * hi) / (1.0 - p_on)
        if lo < 0:
            raise ValueError(
                "burst_rate_multiplier too large to preserve the mean arrival rate "
                f"(would need a negative off-rate: {lo:.3f})"
            )
        t = 0.0
        on = rng.random() < p_on
        state_end = rng.expovariate(1.0 / (on_mean if on else off_mean))
        while t < cfg.duration_s:
            rate = hi if on else lo
            if rate > 0:
                dt = rng.expovariate(rate)
                if t + dt < state_end:
                    t += dt
                    if t < cfg.duration_s:
                        times.append(t)
                    continue
            # State ends before (or without) another arrival: memorylessness
            # of the exponential lets us simply jump to the state boundary.
            t = state_end
            on = not on
            state_end = t + rng.expovariate(1.0 / (on_mean if on else off_mean))
        return times

    raise ValueError(f"Unknown arrival process: {cfg.arrival!r}")


def generate_trace(cfg: WorkloadConfig) -> Trace:
    """Build a deterministic trace for ``cfg`` (same config + seed -> same trace)."""
    rng = random.Random(cfg.seed)

    # Initial conversation lengths: log-normal with the requested *mean*.
    sigma = cfg.context_sigma
    mu = math.log(cfg.mean_context_blocks) - sigma * sigma / 2.0
    initial = [
        int(min(cfg.max_context_blocks, max(cfg.min_context_blocks, round(rng.lognormvariate(mu, sigma)))))
        for _ in range(cfg.num_contexts)
    ]

    # Zipf popularity over a random rank -> context assignment.
    ranks = list(range(cfg.num_contexts))
    rng.shuffle(ranks)  # ranks[r] = context id holding popularity rank r
    weights = [1.0 / ((r + 1) ** cfg.zipf_s) for r in range(cfg.num_contexts)]
    cum = list(accumulate(weights))
    total = cum[-1]

    requests: List[Request] = []
    for i, t in enumerate(_arrival_times(cfg, rng)):
        rank = bisect_left(cum, rng.random() * total)
        rank = min(rank, cfg.num_contexts - 1)
        new_blocks = max(1, round(rng.expovariate(1.0 / cfg.growth_blocks_mean)))
        out_tokens = int(min(4 * cfg.mean_output_tokens, max(8, rng.expovariate(1.0 / cfg.mean_output_tokens))))
        requests.append(Request(i, t, ranks[rank], new_blocks, out_tokens))

    return Trace(config=cfg, requests=requests, initial_blocks=initial)
