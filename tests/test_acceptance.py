"""
Acceptance tests: the project's headline claims, checked end to end on short held-out-style runs.

Each test states the requirement it verifies. Seeds 0-1 and 150 s traces keep the suite fast; the
full evidence (5 seeds, 400 s, paired statistics) is in results/analysis.txt.
"""

import statistics
import unittest
from dataclasses import replace

from src.demo import baseline_spec, ours_spec
from src.eventsim import EventSimulator, SimConfig
from src.policies import fifo_spec, lru_spec, ms_adaptive_spec
from src.trace import WORKLOADS, generate_trace
from src.tuner import TunerConfig


def ttft(workload, spec, seed, config):
    cfg = replace(WORKLOADS[workload].with_seed(seed), duration_s=150.0)
    return EventSimulator(generate_trace(cfg), config, spec).run().summary


DEFAULT = SimConfig(warmup_s=30.0)
STARVED = SimConfig(cpu_pool_mb=512.0, warmup_s=30.0)


class TestAcceptance(unittest.TestCase):
    def test_AC1_using_the_host_tier_beats_eviction_only_policies(self):
        for policy in (fifo_spec(), lru_spec()):
            slow = ttft("zipf", policy, 0, DEFAULT)["reuse_ttft_mean_ms"]
            fast = ttft("zipf", ours_spec(), 0, DEFAULT)["reuse_ttft_mean_ms"]
            self.assertLess(fast, slow / 3, policy.name)

    def test_AC2_adaptive_engine_beats_lru_offload_when_host_memory_is_scarce(self):
        wins = 0
        for seed in (0, 1):
            base = ttft("zipf", baseline_spec(), seed, STARVED)["reuse_ttft_mean_ms"]
            ours = ttft("zipf", ours_spec(), seed, STARVED)["reuse_ttft_mean_ms"]
            wins += ours < base
        self.assertEqual(wins, 2)

    def test_AC3_adaptive_engine_beats_lru_offload_with_plentiful_host_memory_on_uniform(self):
        base = [ttft("uniform", baseline_spec(), s, DEFAULT)["reuse_ttft_mean_ms"] for s in (0, 1)]
        ours = [ttft("uniform", ours_spec(), s, DEFAULT)["reuse_ttft_mean_ms"] for s in (0, 1)]
        self.assertLess(statistics.fmean(ours), 0.85 * statistics.fmean(base))

    def test_AC4_tail_latency_improves_not_just_the_mean(self):
        base = ttft("uniform", baseline_spec(), 0, DEFAULT)["reuse_ttft_p99_ms"]
        ours = ttft("uniform", ours_spec(), 0, DEFAULT)["reuse_ttft_p99_ms"]
        self.assertLess(ours, base)

    def test_AC5_a_smaller_quality_budget_means_less_compressed_reading(self):
        # The budget is a soft target (README, finding 5), so test the property that holds: more budget, more exposure.
        def exposure(budget):
            spec = ms_adaptive_spec(f"b={budget}", tuner_config=TunerConfig(max_compressed_read_ratio=budget))
            return ttft("zipf", spec, 0, STARVED)["compressed_read_ratio"]

        self.assertLess(exposure(0.1), exposure(1.0))

    def test_AC6_no_request_fails_for_lack_of_memory(self):
        for workload in ("uniform", "bursty"):
            self.assertEqual(ttft(workload, ours_spec(), 0, DEFAULT)["oom_events"], 0, workload)


if __name__ == "__main__":
    unittest.main()
