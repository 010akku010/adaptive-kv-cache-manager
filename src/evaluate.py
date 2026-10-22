"""
evaluate.py

Preliminary comparison harness: runs the SAME synthetic workload through
four policies -- our adaptive Cost-Benefit Decision Engine (driven by the
pressure controller), FIFO, LRU, and a simplified MTDS-style
static-threshold baseline -- and reports hit ratio, latency percentiles,
and final GPU utilization for each.

Fairness guarantees (new in this version)
-----------------------------------------
* All policies start from an identical workload built with a fixed clock,
  so runs are fully reproducible.
* All policies execute their actions through the *same*
  ``TierPlacementManager`` (no duplicated execution code).
* Ours, FIFO and LRU use the same trigger (start shedding at the high
  watermark) and the same target (stop at the low watermark), so they
  differ only in *which* blocks they pick and *what* they do with them.

This is Zeroth Review roadmap steps 8-12 (baseline implementation +
preliminary comparison) at the synthetic-simulation level, ahead of vLLM
integration. All numbers come from the synthetic workload generator in
`src/workload.py` -- a methodology demonstration, not yet a claim about
real serving traces. Part 4 of the literature review (evaluation
methodology) is exactly about not overclaiming from a synthetic or
unrepresentative benchmark.

Run with:
    python -m src.evaluate
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.baselines import (
    BaselinePolicy,
    FIFOPolicy,
    LRUPolicy,
    MTDSStaticThresholdPolicy,
)
from src.controller import ControllerConfig, PressureDrivenController
from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.performance_monitor import PerformanceMonitor
from src.placement import PlacementResult, TierPlacementManager
from src.workload import SimulationConfig, load_blocks


def _record(monitor: PerformanceMonitor, result: PlacementResult) -> None:
    monitor.record_decision(result.executed_action)
    if result.was_fallback:
        monitor.record_fallback()


def _replay_access_trace(collector: MetadataCollector, monitor: PerformanceMonitor) -> None:
    """
    Replay simulated future requests against the post-decision block
    placement: each block is "requested" a number of times proportional
    to its original access_count (hotter blocks get requested more), and
    the monitor records the resulting latency based on wherever that
    block ended up (GPU / CPU / EVICTED).
    """
    for block in collector.all_blocks():
        for _ in range(max(1, block.access_count // 5)):
            monitor.record_access(block.location)


def _setup(config: SimulationConfig):
    clock = config.make_clock()
    collector = MetadataCollector(clock=clock)
    tracker = MemoryTracker(config.gpu_capacity_mb, config.cpu_capacity_mb)
    load_blocks(collector, tracker, config, clock)
    return clock, collector, tracker, TierPlacementManager(collector, tracker), PerformanceMonitor()


def _finish(name: str, collector: MetadataCollector, tracker: MemoryTracker, monitor: PerformanceMonitor) -> Dict:
    _replay_access_trace(collector, monitor)
    return {"name": name, "summary": monitor.summary(), "final_state": tracker.snapshot()}


def run_our_engine(config: SimulationConfig, controller_config: Optional[ControllerConfig] = None) -> Dict:
    clock, collector, tracker, placement, monitor = _setup(config)
    controller = PressureDrivenController(
        collector=collector,
        tracker=tracker,
        importance_engine=AdaptiveImportanceEngine(weights=ImportanceWeights(), clock=clock),
        decision_engine=DecisionEngine(),
        placement=placement,
        config=controller_config,
    )
    initial_gpu_blocks = {b.block_id for b in collector.blocks_in_tier(Tier.GPU)}

    acted = set()
    for d in controller.step():
        acted.add(d.block_id)
        monitor.record_decision(d.executed_action)
        if d.fallback_reason:
            monitor.record_fallback()
    for _ in initial_gpu_blocks - acted:
        monitor.record_decision("KEEP")

    return _finish("Ours (Adaptive)", collector, tracker, monitor)


def run_baseline(
    policy: BaselinePolicy,
    config: SimulationConfig,
    controller_config: Optional[ControllerConfig] = None,
) -> Dict:
    cc = controller_config or ControllerConfig()
    _, collector, tracker, placement, monitor = _setup(config)

    gpu_blocks = list(collector.blocks_in_tier(Tier.GPU))
    actions = policy.select_actions(
        gpu_blocks,
        tracker.gpu_pressure(),
        config.gpu_capacity_mb,
        target_pressure=cc.low_watermark,
        trigger_pressure=cc.high_watermark,
    )
    for block in gpu_blocks:
        _record(monitor, placement.apply(block.block_id, actions[block.block_id]))

    return _finish(policy.name, collector, tracker, monitor)


def run_comparison(
    config: Optional[SimulationConfig] = None,
    controller_config: Optional[ControllerConfig] = None,
) -> List[Dict]:
    config = config or SimulationConfig()
    return [
        run_our_engine(config, controller_config),
        run_baseline(FIFOPolicy(), config, controller_config),
        run_baseline(LRUPolicy(), config, controller_config),
        run_baseline(MTDSStaticThresholdPolicy(), config, controller_config),
    ]


def print_comparison(results: List[Dict]) -> None:
    header = f"{'POLICY':<16}{'HIT RATIO':>11}{'AVG(ms)':>10}{'P99(ms)':>10}{'GPU UTIL':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        s, fs = r["summary"], r["final_state"]
        print(
            f"{r['name']:<16}{s['hit_ratio']:>11.3f}{s['avg_latency_ms']:>10.2f}"
            f"{s['p99_latency_ms']:>10.2f}{fs['gpu_utilization']:>11.3f}"
        )
        print(f"{'':<16}actions: {s['actions']}  fallbacks: {s['fallbacks']}")


if __name__ == "__main__":
    print_comparison(run_comparison())
