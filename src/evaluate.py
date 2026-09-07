"""
evaluate.py

Preliminary comparison harness: runs the SAME synthetic workload through
four policies -- our adaptive Cost-Benefit Decision Engine, FIFO, LRU,
and a simplified MTDS-style static-threshold baseline -- and reports
hit ratio, latency percentiles, and final GPU utilization for each.

This directly implements Zeroth Review roadmap steps 8-12 (baseline
implementation + preliminary comparison) at the synthetic-simulation
level, ahead of vLLM integration. All numbers below come from the
synthetic workload generator in `src/simulator.py` -- they are a
methodology demonstration, not yet a claim about real serving traces.
That distinction matters for the eventual report: Part 4 of the
literature review (evaluation methodology) is exactly about not
overclaiming from a synthetic or unrepresentative benchmark.

Run with:
    python -m src.evaluate
"""

from __future__ import annotations

import random
from typing import Dict, List

from src.baselines import BaselinePolicy, FIFOPolicy, LRUPolicy, MTDSStaticThresholdPolicy
from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, KVBlock, Tier
from src.performance_monitor import PerformanceMonitor
from src.simulator import SimulationConfig, build_synthetic_workload


def _load_blocks(collector: MetadataCollector, tracker: MemoryTracker, config: SimulationConfig) -> None:
    """Populate GPU with the same synthetic block set, up to target pressure."""
    import time
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


def _apply_action(collector: MetadataCollector, tracker: MemoryTracker, block: KVBlock, action: str) -> None:
    """Same bookkeeping-only execution used by TierPlacementManager in simulator.py."""
    if action == "KEEP":
        return
    if action == "MIGRATE":
        tracker.move_block(block, Tier.CPU)
    elif action == "COMPRESS":
        new_size = max(0.0, block.size_mb - block.compression_benefit_mb)
        tracker.remove_block(block)
        block.compressed = True
        block.size_mb = new_size
        tracker.add_block(block)
    elif action == "EVICT":
        tracker.remove_block(block)
        block.location = Tier.EVICTED


def _simulate_access_trace(blocks: List[KVBlock], monitor: PerformanceMonitor, seed: int = 7) -> None:
    """
    Replay simulated future requests against the post-decision block
    placement: each block is "requested" a number of times proportional
    to its original access_count (hotter blocks get requested more),
    and the monitor records the resulting latency based on wherever
    that block ended up (GPU / CPU / EVICTED).
    """
    rng = random.Random(seed)
    for block in blocks:
        num_requests = max(1, block.access_count // 5)
        for _ in range(num_requests):
            monitor.record_access(block.location)
    rng.shuffle(monitor.latencies_ms)  # order doesn't matter for percentiles; kept for realism


def run_our_engine(config: SimulationConfig) -> Dict:
    collector = MetadataCollector()
    tracker = MemoryTracker(config.gpu_capacity_mb, config.cpu_capacity_mb)
    _load_blocks(collector, tracker, config)

    importance_engine = AdaptiveImportanceEngine(weights=ImportanceWeights())
    decision_engine = DecisionEngine()
    monitor = PerformanceMonitor()

    for block in list(collector.blocks_in_tier(Tier.GPU)):
        values = importance_engine.compute_action_values(block, tracker.gpu_pressure())
        action, _ = decision_engine.decide(values)
        monitor.record_decision(action)
        _apply_action(collector, tracker, block, action)

    _simulate_access_trace(list(collector.all_blocks()), monitor)
    return {"name": "Ours (Adaptive)", "summary": monitor.summary(), "final_state": tracker.snapshot()}


def run_baseline(policy: BaselinePolicy, config: SimulationConfig) -> Dict:
    collector = MetadataCollector()
    tracker = MemoryTracker(config.gpu_capacity_mb, config.cpu_capacity_mb)
    _load_blocks(collector, tracker, config)

    monitor = PerformanceMonitor()
    gpu_blocks = list(collector.blocks_in_tier(Tier.GPU))
    actions = policy.select_actions(gpu_blocks, tracker.gpu_pressure(), config.gpu_capacity_mb)

    for block in gpu_blocks:
        action = actions[block.block_id]
        monitor.record_decision(action)
        _apply_action(collector, tracker, block, action)

    _simulate_access_trace(list(collector.all_blocks()), monitor)
    return {"name": policy.name, "summary": monitor.summary(), "final_state": tracker.snapshot()}


def run_comparison(config: SimulationConfig = None) -> List[Dict]:
    config = config or SimulationConfig()
    results = [
        run_our_engine(config),
        run_baseline(FIFOPolicy(), config),
        run_baseline(LRUPolicy(), config),
        run_baseline(MTDSStaticThresholdPolicy(), config),
    ]
    return results


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
        print(f"{'':<16}actions: {s['actions']}")


if __name__ == "__main__":
    results = run_comparison()
    print_comparison(results)
