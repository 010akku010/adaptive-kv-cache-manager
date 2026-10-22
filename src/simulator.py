"""
simulator.py

A synthetic KV-cache simulation environment that wires together the
modules implemented so far:

    MetadataCollector -> MemoryTracker -> AdaptiveImportanceEngine
        -> DecisionEngine -> PressureDrivenController -> TierPlacementManager

This lets the group demonstrate the core decision logic end-to-end
*before* vLLM integration, per the project roadmap: build and validate
the core chain (Metadata -> Importance -> Cost/Benefit -> Decision ->
Placement) on synthetic blocks first, then connect it to a real serving
engine.

Run directly for a demo:

    python -m src.simulator
"""

from __future__ import annotations

from typing import Optional

from src.controller import ControllerConfig, PressureDrivenController
from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.placement import TierPlacementManager
from src.workload import SimulationConfig, build_synthetic_workload, load_blocks

# Re-exported so existing imports (`from src.simulator import ...`) keep working.
__all__ = [
    "SimulationConfig",
    "TierPlacementManager",
    "build_synthetic_workload",
    "run_simulation",
]


def run_simulation(
    config: Optional[SimulationConfig] = None,
    controller_config: Optional[ControllerConfig] = None,
) -> None:
    config = config or SimulationConfig()

    clock = config.make_clock()
    collector = MetadataCollector(clock=clock)
    tracker = MemoryTracker(
        gpu_capacity_mb=config.gpu_capacity_mb,
        cpu_capacity_mb=config.cpu_capacity_mb,
    )
    importance_engine = AdaptiveImportanceEngine(weights=ImportanceWeights(), clock=clock)
    controller = PressureDrivenController(
        collector=collector,
        tracker=tracker,
        importance_engine=importance_engine,
        decision_engine=DecisionEngine(),
        placement=TierPlacementManager(collector, tracker),
        config=controller_config,
    )

    # 1. Populate synthetic blocks until the GPU reaches the target fill ratio.
    load_blocks(collector, tracker, config, clock)
    initial_gpu_blocks = sorted(b.block_id for b in collector.blocks_in_tier(Tier.GPU))
    cfg = controller.config

    print(f"Loaded {len(collector)} blocks onto GPU.")
    print(f"Initial state: {tracker.snapshot()}")
    print(
        f"Controller: start shedding at {cfg.high_watermark:.0%} GPU utilization, "
        f"stop at {cfg.low_watermark:.0%}\n"
    )

    # 2. One control cycle: shed pressure using cost-benefit decisions.
    decisions = controller.step()
    acted = {d.block_id: d for d in decisions}

    header = f"{'BLOCK':<12}{'SIZE(MB)':>10}{'IMPORTANCE':>12}{'ACTION':>12}{'NET UTILITY':>14}"
    print(header)
    print("-" * len(header))
    for d in decisions:
        block = collector.get(d.block_id)
        note = f"  (fallback: {d.fallback_reason})" if d.fallback_reason else ""
        print(
            f"{d.block_id:<12}{block.size_mb:>10.1f}{d.importance:>12.3f}"
            f"{d.executed_action:>12}{d.net_utility:>14.4f}{note}"
        )

    kept = [bid for bid in initial_gpu_blocks if bid not in acted]
    print(f"\n{len(decisions)} blocks acted on; {len(kept)} blocks kept untouched on GPU.")
    print(f"Final state: {tracker.snapshot()}")


if __name__ == "__main__":
    run_simulation()
