"""Tests for MemoryTracker capacity enforcement."""

import unittest

from src.memory_tracker import CapacityError, MemoryTracker
from src.metadata_collector import MetadataCollector, Tier


class TestCapacityEnforcement(unittest.TestCase):
    def setUp(self):
        self.collector = MetadataCollector()
        self.tracker = MemoryTracker(gpu_capacity_mb=100.0, cpu_capacity_mb=50.0)

    def test_exact_fit_is_allowed(self):
        block = self.collector.register_block("b", size_mb=100.0)
        self.tracker.add_block(block)
        self.assertAlmostEqual(self.tracker.gpu_pressure(), 1.0)

    def test_overfilling_gpu_raises_and_leaves_state_unchanged(self):
        self.tracker.add_block(self.collector.register_block("a", size_mb=80.0))
        too_big = self.collector.register_block("b", size_mb=30.0)
        with self.assertRaises(CapacityError):
            self.tracker.add_block(too_big)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 80.0)

    def test_can_fit_reports_headroom(self):
        self.tracker.add_block(self.collector.register_block("a", size_mb=80.0))
        self.assertTrue(self.tracker.can_fit(Tier.GPU, 20.0))
        self.assertFalse(self.tracker.can_fit(Tier.GPU, 20.5))
        self.assertTrue(self.tracker.can_fit(Tier.EVICTED, 1e9))

    def test_move_to_full_tier_is_atomic(self):
        block = self.collector.register_block("a", size_mb=60.0)
        self.tracker.add_block(block)  # GPU
        with self.assertRaises(CapacityError):
            self.tracker.move_block(block, Tier.CPU)  # CPU capacity is only 50
        self.assertEqual(block.location, Tier.GPU)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 60.0)
        self.assertAlmostEqual(self.tracker.cpu.used_mb, 0.0)

    def test_in_place_shrink_works_on_a_full_tier(self):
        block = self.collector.register_block("a", size_mb=100.0)
        self.tracker.add_block(block)
        self.tracker.move_block(block, Tier.GPU, new_size_mb=60.0)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 60.0)
        self.assertEqual(block.size_mb, 60.0)

    def test_in_place_growth_beyond_capacity_is_rejected(self):
        block = self.collector.register_block("a", size_mb=90.0)
        self.tracker.add_block(block)
        with self.assertRaises(CapacityError):
            self.tracker.move_block(block, Tier.GPU, new_size_mb=120.0)
        self.assertEqual(block.size_mb, 90.0)

    def test_evict_block_frees_memory_and_marks_evicted(self):
        block = self.collector.register_block("a", size_mb=40.0)
        self.tracker.add_block(block)
        self.tracker.evict_block(block)
        self.assertEqual(block.location, Tier.EVICTED)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
