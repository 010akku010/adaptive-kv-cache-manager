"""
workload.py

Synthetic KV-block workload generation and loading, shared by the
end-to-end simulator and the policy-comparison harness so that both start
from exactly the same initial state.

This is the *snapshot* workload used for Review-1-level demonstration
(a set of blocks already resident on the GPU at a given pressure). A
time-evolving, request-driven workload is the job of the event-driven
simulator planned as the next milestone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from src.clock import Clock, ManualClock
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier


@dataclass
class SimulationConfig:
    num_blocks: int = 200
    gpu_capacity_mb: float = 4096.0
    cpu_capacity_mb: float = 16384.0
    target_gpu_fill_ratio: float = 0.85
    random_seed: int = 42
    # Fixed simulated "now" so runs are reproducible (see src/clock.py).
    start_time: float = 1_700_000_000.0

    def make_clock(self) -> ManualClock:
        return ManualClock(self.start_time)


def build_synthetic_workload(config: SimulationConfig) -> List[dict]:
    """Generate randomized-but-reproducible synthetic block specs."""
    rng = random.Random(config.random_seed)
    blocks = []
    for i in range(config.num_blocks):
        size_mb = rng.uniform(4.0, 64.0)
        last_access_offset_s = rng.uniform(0.0, 120.0)
        blocks.append(
            {
                "block_id": f"block-{i:03d}",
                "size_mb": size_mb,
                "last_access_offset_s": last_access_offset_s,
                "access_count": rng.randint(1, 30),
                "migration_cost_ms": size_mb * rng.uniform(0.3, 0.8),
                "compression_benefit_mb": size_mb * rng.uniform(0.2, 0.5),
                # A block cannot have been created after its last access:
                # creation is `extra` seconds before the last access. This
                # gives FIFO a meaningful, distinct creation order.
                "created_offset_s": last_access_offset_s + rng.uniform(0.0, 300.0),
            }
        )
    return blocks


def load_blocks(
    collector: MetadataCollector,
    tracker: MemoryTracker,
    config: SimulationConfig,
    clock: Optional[Clock] = None,
) -> None:
    """
    Populate the GPU with synthetic blocks until it reaches the target
    fill ratio (or the GPU is full). ``collector`` should have been built
    with the same ``clock`` so timestamps are consistent.
    """
    clock = clock if clock is not None else collector.clock
    now = clock.now()
    for spec in build_synthetic_workload(config):
        if not tracker.can_fit(Tier.GPU, spec["size_mb"]):
            break
        block = collector.register_block(
            block_id=spec["block_id"],
            size_mb=spec["size_mb"],
            migration_cost_ms=spec["migration_cost_ms"],
            compression_benefit_mb=spec["compression_benefit_mb"],
        )
        block.created_at = now - spec["created_offset_s"]
        block.last_access = now - spec["last_access_offset_s"]
        block.access_count = spec["access_count"]
        tracker.add_block(block)
        if tracker.gpu_pressure() >= config.target_gpu_fill_ratio:
            break
