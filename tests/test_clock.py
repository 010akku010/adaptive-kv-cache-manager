"""Tests for the injectable clock and its use by the collector / importance engine."""

import math
import time
import unittest

from src.clock import ManualClock, SystemClock
from src.importance_engine import AdaptiveImportanceEngine
from src.metadata_collector import MetadataCollector


class TestClocks(unittest.TestCase):
    def test_manual_clock_starts_where_told_and_advances(self):
        clock = ManualClock(100.0)
        self.assertEqual(clock.now(), 100.0)
        self.assertEqual(clock.advance(5.5), 105.5)
        self.assertEqual(clock.now(), 105.5)

    def test_manual_clock_refuses_to_go_backwards(self):
        clock = ManualClock(10.0)
        with self.assertRaises(ValueError):
            clock.advance(-1.0)
        with self.assertRaises(ValueError):
            clock.set(5.0)

    def test_manual_clock_set_forward(self):
        clock = ManualClock(10.0)
        clock.set(50.0)
        self.assertEqual(clock.now(), 50.0)

    def test_system_clock_tracks_wall_time(self):
        self.assertLess(abs(SystemClock().now() - time.time()), 1.0)


class TestClockInjection(unittest.TestCase):
    def test_collector_stamps_blocks_from_its_clock(self):
        clock = ManualClock(1000.0)
        collector = MetadataCollector(clock=clock)
        block = collector.register_block("b1", size_mb=10.0)
        self.assertEqual(block.created_at, 1000.0)
        self.assertEqual(block.last_access, 1000.0)

        clock.advance(42.0)
        collector.record_access("b1")
        self.assertEqual(block.last_access, 1042.0)
        self.assertEqual(block.access_count, 2)

    def test_recency_decays_with_injected_clock_not_wall_time(self):
        clock = ManualClock(0.0)
        collector = MetadataCollector(clock=clock)
        engine = AdaptiveImportanceEngine(clock=clock, recency_half_life_s=30.0)
        block = collector.register_block("b1", size_mb=10.0)

        self.assertAlmostEqual(engine.recency_score(block), 1.0)
        clock.advance(30.0)  # one time-constant later
        self.assertAlmostEqual(engine.recency_score(block), math.exp(-1.0))

    def test_identical_runs_give_identical_scores(self):
        def score():
            clock = ManualClock(500.0)
            collector = MetadataCollector(clock=clock)
            engine = AdaptiveImportanceEngine(clock=clock)
            block = collector.register_block("b", size_mb=8.0)
            clock.advance(12.0)
            return engine.compute_importance(block)

        self.assertEqual(score(), score())


if __name__ == "__main__":
    unittest.main()
