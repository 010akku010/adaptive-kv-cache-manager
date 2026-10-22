"""Tests for the Tier Placement Manager (execution + fallback rules)."""

import unittest

from src.importance_engine import AdaptiveImportanceEngine
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.placement import TierPlacementManager


class TestPlacementManager(unittest.TestCase):
    def setUp(self):
        self.collector = MetadataCollector()
        self.tracker = MemoryTracker(gpu_capacity_mb=100.0, cpu_capacity_mb=30.0)
        self.placement = TierPlacementManager(self.collector, self.tracker)

    def _gpu_block(self, block_id="b1", size=20.0, compression_benefit=8.0):
        block = self.collector.register_block(
            block_id, size_mb=size, compression_benefit_mb=compression_benefit
        )
        self.tracker.add_block(block)
        return block

    def test_keep_is_a_noop(self):
        block = self._gpu_block()
        result = self.placement.apply("b1", "KEEP")
        self.assertEqual(result.executed_action, "KEEP")
        self.assertEqual(block.location, Tier.GPU)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 20.0)

    def test_migrate_moves_gpu_to_cpu(self):
        block = self._gpu_block()
        result = self.placement.apply("b1", "MIGRATE")
        self.assertEqual(result.executed_action, "MIGRATE")
        self.assertFalse(result.was_fallback)
        self.assertEqual(block.location, Tier.CPU)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 0.0)
        self.assertAlmostEqual(self.tracker.cpu.used_mb, 20.0)
        self.assertAlmostEqual(result.freed_gpu_mb, 20.0)

    def test_migrate_with_full_cpu_falls_back_to_evict(self):
        self._gpu_block("a", size=25.0)
        self._gpu_block("b", size=25.0)
        first = self.placement.apply("a", "MIGRATE")
        second = self.placement.apply("b", "MIGRATE")  # only 5 MB of CPU left

        self.assertEqual(first.executed_action, "MIGRATE")
        self.assertEqual(second.requested_action, "MIGRATE")
        self.assertEqual(second.executed_action, "EVICT")
        self.assertEqual(second.fallback_reason, "cpu_full")
        self.assertEqual(self.collector.get("b").location, Tier.EVICTED)
        self.assertAlmostEqual(self.tracker.cpu.used_mb, 25.0)  # never overfilled

    def test_migrate_of_non_gpu_block_is_noop(self):
        self._gpu_block()
        self.placement.apply("b1", "MIGRATE")
        result = self.placement.apply("b1", "MIGRATE")
        self.assertEqual(result.executed_action, "KEEP")
        self.assertEqual(result.fallback_reason, "not_on_gpu")

    def test_compress_shrinks_block_and_marks_it(self):
        block = self._gpu_block(size=20.0, compression_benefit=8.0)
        result = self.placement.apply("b1", "COMPRESS")
        self.assertEqual(result.executed_action, "COMPRESS")
        self.assertTrue(block.compressed)
        self.assertAlmostEqual(block.size_mb, 12.0)
        self.assertAlmostEqual(block.compression_benefit_mb, 0.0)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 12.0)
        self.assertAlmostEqual(result.freed_gpu_mb, 8.0)

    def test_compress_only_once(self):
        block = self._gpu_block(size=20.0, compression_benefit=8.0)
        self.placement.apply("b1", "COMPRESS")
        result = self.placement.apply("b1", "COMPRESS")
        self.assertEqual(result.executed_action, "KEEP")
        self.assertEqual(result.fallback_reason, "already_compressed")
        self.assertAlmostEqual(block.size_mb, 12.0)  # unchanged by the second call

    def test_evict_frees_gpu_memory(self):
        block = self._gpu_block()
        result = self.placement.apply("b1", "EVICT")
        self.assertEqual(block.location, Tier.EVICTED)
        self.assertAlmostEqual(self.tracker.gpu.used_mb, 0.0)
        self.assertAlmostEqual(result.freed_gpu_mb, 20.0)

    def test_unknown_action_and_block_are_rejected(self):
        self._gpu_block()
        with self.assertRaises(ValueError):
            self.placement.apply("b1", "TELEPORT")
        with self.assertRaises(KeyError):
            self.placement.apply("missing", "KEEP")


class TestCompressOnceInImportanceEngine(unittest.TestCase):
    def test_compressed_block_is_not_offered_compress(self):
        collector = MetadataCollector()
        engine = AdaptiveImportanceEngine()
        block = collector.register_block("b", size_mb=20.0, compression_benefit_mb=8.0)
        self.assertIn("COMPRESS", engine.compute_action_values(block, 0.9))
        block.compressed = True
        self.assertNotIn("COMPRESS", engine.compute_action_values(block, 0.9))


if __name__ == "__main__":
    unittest.main()
