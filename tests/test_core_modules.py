"""
Unit tests for the four core modules implemented so far:
MetadataCollector, MemoryTracker, AdaptiveImportanceEngine, DecisionEngine.

Run with:
    python -m pytest tests/ -v
or, with no extra dependencies:
    python -m unittest discover -s tests -v
"""

import time
import unittest

from src.decision_engine import DecisionEngine
from src.importance_engine import ActionEstimate, AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier


class TestMetadataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = MetadataCollector()

    def test_register_and_get_block(self):
        block = self.collector.register_block("b1", size_mb=10.0)
        self.assertEqual(block.block_id, "b1")
        self.assertEqual(block.location, Tier.GPU)
        self.assertEqual(self.collector.get("b1"), block)

    def test_duplicate_registration_raises(self):
        self.collector.register_block("b1", size_mb=10.0)
        with self.assertRaises(ValueError):
            self.collector.register_block("b1", size_mb=5.0)

    def test_record_access_updates_recency_and_frequency(self):
        block = self.collector.register_block("b1", size_mb=10.0)
        block.last_access = time.time() - 100  # simulate staleness
        initial_count = block.access_count

        self.collector.record_access("b1")

        self.assertEqual(block.access_count, initial_count + 1)
        self.assertLess(block.age_seconds(), 1.0)

    def test_unknown_block_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.collector.get("does-not-exist")

    def test_remove_block(self):
        self.collector.register_block("b1", size_mb=10.0)
        self.collector.remove_block("b1")
        self.assertEqual(len(self.collector), 0)

    def test_blocks_in_tier_filters_correctly(self):
        self.collector.register_block("gpu-block", size_mb=1.0, location=Tier.GPU)
        self.collector.register_block("cpu-block", size_mb=1.0, location=Tier.CPU)
        gpu_ids = {b.block_id for b in self.collector.blocks_in_tier(Tier.GPU)}
        self.assertEqual(gpu_ids, {"gpu-block"})


class TestMemoryTracker(unittest.TestCase):
    def setUp(self):
        self.collector = MetadataCollector()
        self.tracker = MemoryTracker(gpu_capacity_mb=100.0, cpu_capacity_mb=1000.0)

    def test_add_block_updates_usage(self):
        block = self.collector.register_block("b1", size_mb=20.0)
        self.tracker.add_block(block)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 20.0)
        self.assertAlmostEqual(self.tracker.gpu_pressure(), 0.2)

    def test_move_block_between_tiers(self):
        block = self.collector.register_block("b1", size_mb=20.0)
        self.tracker.add_block(block)

        self.tracker.move_block(block, Tier.CPU)

        self.assertEqual(block.location, Tier.CPU)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 0.0)
        self.assertAlmostEqual(self.tracker.cpu.used_mb, 20.0)

    def test_gpu_pressure_bounded_and_monotonic(self):
        for i in range(10):
            block = self.collector.register_block(f"b{i}", size_mb=10.0)
            self.tracker.add_block(block)
        # 10 * 10MB = 100MB == capacity -> pressure should saturate at 1.0
        self.assertAlmostEqual(self.tracker.gpu_pressure(), 1.0)

    def test_remove_block_never_goes_negative(self):
        block = self.collector.register_block("b1", size_mb=20.0)
        self.tracker.add_block(block)
        self.tracker.remove_block(block)
        self.tracker.remove_block(block)  # double-remove should not go negative
        self.assertGreaterEqual(self.tracker.gpu.used_mb, 0.0)


class TestAdaptiveImportanceEngine(unittest.TestCase):
    def setUp(self):
        self.collector = MetadataCollector()
        self.engine = AdaptiveImportanceEngine(weights=ImportanceWeights())

    def test_recently_and_frequently_accessed_block_scores_higher(self):
        hot = self.collector.register_block("hot", size_mb=10.0)
        hot.access_count = 50
        hot.last_access = time.time()

        cold = self.collector.register_block("cold", size_mb=10.0)
        cold.access_count = 1
        cold.last_access = time.time() - 600  # long idle

        self.assertGreater(
            self.engine.compute_importance(hot),
            self.engine.compute_importance(cold),
        )

    def test_importance_is_bounded_zero_to_one(self):
        block = self.collector.register_block("b1", size_mb=10.0)
        block.access_count = 10_000  # extreme frequency
        block.last_access = time.time()

        importance = self.engine.compute_importance(block)
        self.assertGreaterEqual(importance, 0.0)
        self.assertLessEqual(importance, 1.0)

    def test_high_gpu_pressure_increases_evict_and_migrate_benefit(self):
        block = self.collector.register_block("b1", size_mb=100.0)

        low_pressure = self.engine.compute_action_values(block, gpu_pressure=0.1)
        high_pressure = self.engine.compute_action_values(block, gpu_pressure=0.9)

        self.assertGreater(high_pressure["EVICT"].benefit, low_pressure["EVICT"].benefit)
        self.assertGreater(high_pressure["MIGRATE"].benefit, low_pressure["MIGRATE"].benefit)

    def test_hot_block_favors_keep_over_evict(self):
        hot = self.collector.register_block("hot", size_mb=10.0)
        hot.access_count = 100
        hot.last_access = time.time()

        values = self.engine.compute_action_values(hot, gpu_pressure=0.3)
        decision_engine = DecisionEngine()
        action, _ = decision_engine.decide(values)

        self.assertEqual(action, "KEEP")

    def test_cold_low_priority_block_under_high_pressure_is_not_kept(self):
        cold = self.collector.register_block("cold", size_mb=200.0)
        cold.access_count = 1
        cold.last_access = time.time() - 3600  # idle for an hour

        values = self.engine.compute_action_values(cold, gpu_pressure=0.95)
        decision_engine = DecisionEngine()
        action, _ = decision_engine.decide(values)

        self.assertNotEqual(action, "KEEP")


class TestDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()

    def test_calculate_utility(self):
        self.assertEqual(self.engine.calculate_utility(benefit=8, cost=2), 6)

    def test_decide_picks_highest_net_utility_with_plain_dicts(self):
        actions = {
            "KEEP": {"benefit": 8, "cost": 2},
            "MIGRATE": {"benefit": 6, "cost": 1},
            "COMPRESS": {"benefit": 5, "cost": 4},
            "EVICT": {"benefit": 3, "cost": 0},
        }
        action, utility = self.engine.decide(actions)
        self.assertEqual(action, "KEEP")
        self.assertEqual(utility, 6)

    def test_decide_accepts_action_estimate_objects(self):
        actions = {
            "KEEP": ActionEstimate(benefit=1.0, cost=0.9),
            "MIGRATE": ActionEstimate(benefit=1.0, cost=0.1),
        }
        action, utility = self.engine.decide(actions)
        self.assertEqual(action, "MIGRATE")
        self.assertAlmostEqual(utility, 0.9)

    def test_decide_raises_on_empty_actions(self):
        with self.assertRaises(ValueError):
            self.engine.decide({})


if __name__ == "__main__":
    unittest.main()
