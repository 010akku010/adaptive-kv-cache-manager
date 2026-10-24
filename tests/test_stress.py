"""Stress and performance tests: heavy traffic must keep every invariant and finish in bounded time."""

import time
import unittest
from dataclasses import replace

from src.demo import ours_spec
from src.eventsim import EventSimulator, SimConfig
from src.policies import fifo_spec, ladder_spec, lru_offload_spec, ms_adaptive_spec
from src.trace import WORKLOADS, generate_trace


def run(workload, spec, config=None, seed=0, duration=80.0, rate=None):
    cfg = replace(WORKLOADS[workload].with_seed(seed), duration_s=duration)
    if rate:
        cfg = replace(cfg, arrival_rate=rate)
    sim = EventSimulator(generate_trace(cfg), config or SimConfig(warmup_s=20.0), spec)
    result = sim.run()
    return sim, result


class TestStress(unittest.TestCase):
    def assert_consistent(self, sim):
        self.assertEqual(sim.check_invariants(), [])
        self.assertTrue(all(b.pin_count == 0 for b in sim.collector.all_blocks() if b.pin_count))

    def test_every_policy_stays_consistent_under_heavy_load(self):
        # 3x the normal request rate on a tiny GPU pool: constant pressure.
        config = SimConfig(gpu_pool_mb=256.0, cpu_pool_mb=512.0, warmup_s=20.0)
        for spec in (fifo_spec(), lru_offload_spec(), ladder_spec(max_compressed_fraction=0.5, name="Ladder"), ours_spec()):
            with self.subTest(policy=spec.name):
                sim, result = run("zipf", spec, config, rate=10.0)
                self.assert_consistent(sim)
                self.assertGreater(result.summary["reuse_ttft_mean_ms"], 0)

    def test_long_context_with_a_tiny_host_pool_does_not_break_invariants(self):
        config = SimConfig(gpu_pool_mb=512.0, cpu_pool_mb=64.0, warmup_s=20.0)
        sim, _ = run("long_context", ours_spec(), config)
        self.assert_consistent(sim)

    def test_bursty_traffic_keeps_the_tuner_stable(self):
        sim, result = run("bursty", ours_spec(), SimConfig(cpu_pool_mb=512.0, warmup_s=20.0), duration=120.0)
        self.assert_consistent(sim)
        history = sim.policy.tuner.history
        self.assertGreater(len(history), 5)
        for h in history:  # anti-windup bounds hold throughout
            self.assertGreaterEqual(h.quality_penalty_ms, 0.25 - 1e-9)

    def test_same_seed_gives_identical_results(self):
        a = run("uniform", ours_spec(), duration=60.0)[1].summary
        b = run("uniform", ours_spec(), duration=60.0)[1].summary
        self.assertEqual(a, b)

    def test_oom_events_stay_rare_in_a_normal_configuration(self):
        _, result = run("uniform", ours_spec(), duration=80.0)
        self.assertEqual(result.summary["oom_events"], 0)


class TestPerformance(unittest.TestCase):
    """Wall-clock budgets generous enough for slow CI runners but tight enough to catch a quadratic blow-up."""

    def test_simulating_100_seconds_of_traffic_finishes_quickly(self):
        start = time.perf_counter()
        run("uniform", ours_spec(), duration=100.0)
        self.assertLess(time.perf_counter() - start, 30.0)

    def test_decision_cost_does_not_explode_with_many_blocks(self):
        # long_context has ~10k blocks; the controller must still make decisions in near-linear time.
        start = time.perf_counter()
        sim, _ = run("long_context", ms_adaptive_spec("x"), duration=80.0)
        elapsed = time.perf_counter() - start
        blocks = sum(1 for _ in sim.collector.all_blocks())
        self.assertGreater(blocks, 1000)
        self.assertLess(elapsed / blocks * 1000, 30.0, "more than 30 ms of wall time per block")


if __name__ == "__main__":
    unittest.main()
