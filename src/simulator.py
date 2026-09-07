"""
simulator.py

A synthetic KV-cache simulation environment that wires together the
four modules implemented so far:

    MetadataCollector -> MemoryTracker -> AdaptiveImportanceEngine
        -> DecisionEngine -> (simulated Tier Placement Manager)

This lets the group demonstrate the core decision logic end-to-end
*before* vLLM integration, per the project roadmap: build and validate
the core chain (Metadata -> Importance -> Cost/Benefit -> Decision ->
Placement) on synthetic blocks first, then connect it to a real serving
engine.

Run directly for a demo:

    python -m src.simulator
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List

from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier


@dataclass
class SimulationConfig:
    num_blocks: int = 200
    gpu_capacity_mb: float = 4096.0
    cpu_capacity_mb: float = 16384.0
    target_gpu_fill_ratio: float = 0.85
    random_seed: int = 42


class TierPlacementManager:
    """
    Executes a Decision Engine's action for a block. In this simulator
    the "execution" is bookkeeping-only (updating MetadataCollector and
    MemoryTracker); real GPU<->CPU tensor movement is out of scope until
    vLLM integration.
    """

    def __init__(self, collector: MetadataCollector, tracker: MemoryTracker) -> None:
        self.collector = collector
        self.tracker = tracker

    def apply(self, block_id: str, action: str) -> None:
        block = self.collector.get(block_id)

        if action == "KEEP":
            return  # no-op

        if action == "MIGRATE":
            self.tracker.move_block(block, Tier.CPU)
            self.collector.update_location(block_id, Tier.CPU)

        elif action == "COMPRESS":
            new_size = max(0.0, block.size_mb - block.compression_benefit_mb)
            self.tracker.remove_block(block)
            self.collector.mark_compressed(block_id, new_size)
            self.tracker.add_block(block)

        elif action == "EVICT":
            self.tracker.remove_block(block)
            self.collector.update_location(block_id, Tier.EVICTED)

        else:
            raise ValueError(f"Unknown action: {action}")


def build_synthetic_workload(config: SimulationConfig) -> List[dict]:
    """Generate randomized-but-reproducible synthetic block specs."""
    rng = random.Random(config.random_seed)
    now = time.time()
    blocks = []
    for i in range(config.num_blocks):
        size_mb = rng.uniform(4.0, 64.0)
        blocks.append(
            {
                "block_id": f"block-{i:03d}",
                "size_mb": size_mb,
                "last_access_offset_s": rng.uniform(0.0, 120.0),
                "access_count": rng.randint(1, 30),
                "migration_cost_ms": size_mb * rng.uniform(0.3, 0.8),
                "compression_benefit_mb": size_mb * rng.uniform(0.2, 0.5),
            }
        )
    return blocks


def run_simulation(config: SimulationConfig = None) -> None:
    config = config or SimulationConfig()

    collector = MetadataCollector()
    tracker = MemoryTracker(
        gpu_capacity_mb=config.gpu_capacity_mb,
        cpu_capacity_mb=config.cpu_capacity_mb,
    )
    importance_engine = AdaptiveImportanceEngine(weights=ImportanceWeights())
    decision_engine = DecisionEngine()
    placement_manager = TierPlacementManager(collector, tracker)

    # 1. Populate synthetic blocks until GPU reaches the target fill ratio.
    now = time.time()
    for spec in build_synthetic_workload(config):
        block = collector.register_block(
            block_id=spec["block_id"],
            size_mb=spec["size_mb"],
            migration_cost_ms=spec["migration_cost_ms"],
            compression_benefit_mb=spec["compression_benefit_mb"],
        )
        block.last_access = now - spec["last_access_offset_s"]
        block.access_count = spec["access_count"]
        tracker.add_block(block)

        if tracker.gpu_pressure() >= config.target_gpu_fill_ratio:
            break

    print(f"Loaded {len(collector)} blocks onto GPU.")
    print(f"Initial state: {tracker.snapshot()}\n")

    # 2. For every GPU-resident block, compute action values and decide.
    header = f"{'BLOCK':<12}{'SIZE(MB)':>10}{'IMPORTANCE':>12}{'ACTION':>12}{'NET UTILITY':>14}"
    print(header)
    print("-" * len(header))

    for block in list(collector.blocks_in_tier(Tier.GPU)):
        gpu_pressure = tracker.gpu_pressure()
        action_values = importance_engine.compute_action_values(
            block=block,
            gpu_pressure=gpu_pressure,
        )
        action, utility = decision_engine.decide(action_values)
        importance = importance_engine.compute_importance(block)

        print(
            f"{block.block_id:<12}{block.size_mb:>10.1f}{importance:>12.3f}"
            f"{action:>12}{utility:>14.4f}"
        )

        placement_manager.apply(block.block_id, action)

    print(f"\nFinal state: {tracker.snapshot()}")


if __name__ == "__main__":
    run_simulation()
