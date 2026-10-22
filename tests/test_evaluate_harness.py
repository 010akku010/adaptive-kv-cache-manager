"""Tests for the workload generator and the policy-comparison harness."""

import unittest

from src.evaluate import run_comparison
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector
from src.workload import SimulationConfig, build_synthetic_workload, load_blocks


class TestWorkload(unittest.TestCase):
    def test_workload_is_reproducible_for_a_seed(self):
        cfg = SimulationConfig()
        self.assertEqual(build_synthetic_workload(cfg), build_synthetic_workload(cfg))

    def test_different_seeds_differ(self):
        a = build_synthetic_workload(SimulationConfig(random_seed=1))
        b = build_synthetic_workload(SimulationConfig(random_seed=2))
        self.assertNotEqual(a, b)

    def test_blocks_are_created_no_later_than_their_last_access(self):
        for spec in build_synthetic_workload(SimulationConfig()):
            self.assertGreaterEqual(spec["created_offset_s"], spec["last_access_offset_s"])

    def test_load_blocks_reaches_target_fill_without_overflow(self):
        cfg = SimulationConfig()
        clock = cfg.make_clock()
        collector = MetadataCollector(clock=clock)
        tracker = MemoryTracker(cfg.gpu_capacity_mb, cfg.cpu_capacity_mb)
        load_blocks(collector, tracker, cfg, clock)
        self.assertGreaterEqual(tracker.gpu_pressure(), cfg.target_gpu_fill_ratio)
        self.assertLessEqual(tracker.gpu.used_mb, tracker.gpu.capacity_mb)

    def test_load_blocks_stops_when_gpu_is_too_small(self):
        cfg = SimulationConfig(gpu_capacity_mb=100.0, target_gpu_fill_ratio=1.0)
        clock = cfg.make_clock()
        collector = MetadataCollector(clock=clock)
        tracker = MemoryTracker(cfg.gpu_capacity_mb, cfg.cpu_capacity_mb)
        load_blocks(collector, tracker, cfg, clock)  # must not raise CapacityError
        self.assertLessEqual(tracker.gpu.used_mb, 100.0)


class TestComparisonHarness(unittest.TestCase):
    def setUp(self):
        self.results = run_comparison()
        self.by_name = {r["name"]: r for r in self.results}

    def test_runs_are_fully_reproducible(self):
        self.assertEqual(self.results, run_comparison())

    def test_all_four_policies_are_reported(self):
        self.assertEqual(
            set(self.by_name), {"Ours (Adaptive)", "FIFO", "LRU", "MTDS-static"}
        )

    def test_ours_stops_shedding_at_the_target(self):
        cfg = SimulationConfig()
        largest_block_fraction = 64.0 / cfg.gpu_capacity_mb
        util = self.by_name["Ours (Adaptive)"]["final_state"]["gpu_utilization"]
        self.assertLessEqual(util, 0.70 + 1e-9)
        # Regression test for the original bug (finished at ~34%): at most one
        # block's worth of overshoot below the target is acceptable.
        self.assertGreaterEqual(util, 0.70 - largest_block_fraction)

    def test_ours_fifo_lru_end_at_comparable_utilization(self):
        utils = [self.by_name[n]["final_state"]["gpu_utilization"] for n in ("Ours (Adaptive)", "FIFO", "LRU")]
        self.assertLess(max(utils) - min(utils), 0.03)

    def test_every_policy_accounts_for_every_gpu_block(self):
        totals = {sum(r["summary"]["actions"].values()) for r in self.results}
        self.assertEqual(len(totals), 1)

    def test_tiers_never_exceed_capacity(self):
        for r in self.results:
            fs = r["final_state"]
            self.assertLessEqual(fs["gpu_utilization"], 1.0)
            self.assertLessEqual(fs["cpu_utilization"], 1.0)

    def test_small_cpu_tier_forces_mtds_fallbacks(self):
        results = run_comparison(SimulationConfig(cpu_capacity_mb=500.0))
        mtds = next(r for r in results if r["name"] == "MTDS-static")
        self.assertGreater(mtds["summary"]["fallbacks"], 0)
        self.assertLessEqual(mtds["final_state"]["cpu_utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
