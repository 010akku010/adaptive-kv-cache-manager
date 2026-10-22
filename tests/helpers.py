"""Shared test fixtures: a toy hardware profile with round numbers for hand-checkable maths."""

from __future__ import annotations

from typing import List

from src.eventsim import EventSimulator, SimConfig, SimResult
from src.hardware import ModelProfile, ServingProfile
from src.metadata_collector import Tier
from src.policies import PolicySpec
from src.trace import Request, Trace, WorkloadConfig

# 25 layers x 25 KV heads x 25 head_dim: 2*25*25*25*2 B = 62,500 B/token,
# so a 16-token block is EXACTLY 1.0 MB.
TOY_MODEL = ModelProfile("toy", num_layers=25, num_kv_heads=25, head_dim=25)


def toy_profile(pcie_gb_per_s: float = 1.0) -> ServingProfile:
    """1 MB block: transfer = 1.0 ms (at 1 GB/s); recompute of a block = 10 ms."""
    return ServingProfile(
        name="toy",
        model=TOY_MODEL,
        pcie_gb_per_s=pcie_gb_per_s,  # 1 GB/s == 1 MB/ms
        prefill_tokens_per_s=1600.0,  # 16 tokens -> 10 ms
        decode_ms_per_token=1.0,
        pcie_setup_ms=0.0,
        decompress_ms_per_mb=0.5,
        compression_savings=0.5,
        request_overhead_ms=0.0,
    )


def toy_config(**overrides) -> SimConfig:
    params = dict(
        profile=toy_profile(),
        gpu_pool_mb=100.0,
        cpu_pool_mb=100.0,
        tick_interval_s=0.1,
        warmup_s=0.0,
    )
    params.update(overrides)
    return SimConfig(**params)


def make_trace(requests: List[Request], initial_blocks: List[int], duration_s: float = 3.0,
               max_context_blocks: int = 64) -> Trace:
    cfg = WorkloadConfig(name="hand", duration_s=duration_s, max_context_blocks=max_context_blocks)
    return Trace(config=cfg, requests=requests, initial_blocks=initial_blocks)


def run(trace: Trace, spec: PolicySpec, config: SimConfig = None):
    """Run and return (simulator, result) so tests can inspect internal state."""
    sim = EventSimulator(trace, config or toy_config(), spec)
    result: SimResult = sim.run()
    return sim, result


# --- Test-double policies -------------------------------------------------

class ActOnAll:
    """Every call, apply ``action`` to every GPU block (the placement manager refuses pinned ones)."""

    def __init__(self, ctx, action: str) -> None:
        self.ctx, self.action, self.name = ctx, action, f"{action.lower()}-all"

    def relieve(self):
        return [
            self.ctx.placement.apply(b.block_id, self.action)
            for b in list(self.ctx.collector.blocks_in_tier(Tier.GPU))
        ]


class DoNothing:
    name = "null"

    def __init__(self, ctx) -> None:
        pass

    def relieve(self):
        return []


EVICT_ALL = PolicySpec("evict-all", lambda ctx: ActOnAll(ctx, "EVICT"))
MIGRATE_ALL = PolicySpec("migrate-all", lambda ctx: ActOnAll(ctx, "MIGRATE"))
NULL_POLICY = PolicySpec("null", DoNothing)
