"""Tests for the pressure-driven controller (trigger, hysteresis, ordering, fallbacks)."""

import unittest

from src.clock import ManualClock
from src.controller import ControllerConfig, PressureDrivenController
from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.placement import TierPlacementManager


class _FixedActionEngine(DecisionEngine):
    """Test double: always chooses one action, regardless of the numbers."""

    def __init__(self, action):
        self._action = action

    def decide(self, actions):
        return self._action, 0.0


def build(n_blocks, block_mb=10.0, gpu_mb=100.0, cpu_mb=1000.0,
          config=None, decision_engine=None, compression_benefit=0.0):
    """n identical GPU-resident blocks; hotness can be tweaked by the caller afterwards."""
    clock = ManualClock(10_000.0)
    collector = MetadataCollector(clock=clock)
    tracker = MemoryTracker(gpu_mb, cpu_mb)
    for i in range(n_blocks):
        block = collector.register_block(
            f"b{i:02d}", size_mb=block_mb, compression_benefit_mb=compression_benefit,
            migration_cost_ms=10.0,
        )
        tracker.add_block(block)
    controller = PressureDrivenController(
        collector=collector,
        tracker=tracker,
        importance_engine=AdaptiveImportanceEngine(clock=clock),
        decision_engine=decision_engine or DecisionEngine(),
        placement=TierPlacementManager(collector, tracker),
        config=config,
    )
    return controller, collector, tracker, clock


class TestConfigValidation(unittest.TestCase):
    def test_low_must_be_below_high(self):
        with self.assertRaises(ValueError):
            ControllerConfig(high_watermark=0.7, low_watermark=0.7)
        with self.assertRaises(ValueError):
            ControllerConfig(high_watermark=0.6, low_watermark=0.8)

    def test_watermarks_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            ControllerConfig(high_watermark=1.2, low_watermark=0.5)
        with self.assertRaises(ValueError):
            ControllerConfig(high_watermark=0.9, low_watermark=0.0)

    def test_action_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            ControllerConfig(max_actions_per_step=0)


class TestTriggerAndStop(unittest.TestCase):
    def test_no_action_below_high_watermark(self):
        controller, _, tracker, _ = build(8)  # 80% < 85%
        self.assertEqual(controller.step(), [])
        self.assertFalse(controller.shedding)
        self.assertAlmostEqual(tracker.gpu_pressure(), 0.8)

    def test_sheds_down_to_low_watermark_and_no_further(self):
        controller, _, tracker, _ = build(9)  # 90% >= 85%
        decisions = controller.step()
        self.assertTrue(decisions)
        self.assertLessEqual(tracker.gpu_pressure(), 0.70 + 1e-9)
        # One block fewer would still have been above the target: no over-eviction.
        self.assertGreater(tracker.gpu_pressure() + 0.10, 0.70)
        self.assertFalse(controller.shedding)

    def test_decisions_report_pressure_trajectory(self):
        controller, _, _, _ = build(9)
        decisions = controller.step()
        for d in decisions:
            self.assertLess(d.pressure_after, d.pressure_before)
            self.assertGreater(d.freed_gpu_mb, 0.0)
        self.assertGreater(decisions[0].pressure_before, 0.85)

    def test_never_requests_keep(self):
        controller, _, _, _ = build(10)
        for d in controller.step():
            self.assertNotEqual(d.requested_action, "KEEP")

    def test_gpu_never_exceeds_capacity(self):
        controller, _, tracker, _ = build(10)  # exactly full
        controller.step()
        self.assertLessEqual(tracker.gpu.used_mb, tracker.gpu.capacity_mb)


class TestHysteresis(unittest.TestCase):
    def test_rate_limited_shedding_continues_below_high_watermark(self):
        cfg = ControllerConfig(high_watermark=0.85, low_watermark=0.70, max_actions_per_step=1)
        controller, _, tracker, _ = build(9, config=cfg)  # 90%

        first = controller.step()
        self.assertEqual(len(first), 1)
        self.assertAlmostEqual(tracker.gpu_pressure(), 0.8)  # below high, above low
        self.assertTrue(controller.shedding)  # keeps going: that's the hysteresis

        second = controller.step()
        self.assertEqual(len(second), 1)
        self.assertAlmostEqual(tracker.gpu_pressure(), 0.7)
        self.assertFalse(controller.shedding)

    def test_does_not_restart_between_low_and_high(self):
        controller, collector, tracker, _ = build(9)
        controller.step()  # settles at/below 70%
        # Refill to ~80%: inside the dead band, so the controller must stay idle.
        block = collector.register_block("refill", size_mb=10.0)
        tracker.add_block(block)
        self.assertGreater(tracker.gpu_pressure(), 0.70)
        self.assertLess(tracker.gpu_pressure(), 0.85)
        self.assertEqual(controller.step(), [])


class TestVictimSelection(unittest.TestCase):
    def test_coldest_block_is_shed_first(self):
        controller, collector, _, clock = build(9)
        for b in collector.all_blocks():
            b.access_count = 100
            b.last_access = clock.now()  # everyone hot
        cold = collector.get("b04")
        cold.access_count = 1
        cold.last_access = clock.now() - 3600.0

        decisions = controller.step()
        self.assertEqual(decisions[0].block_id, "b04")

    def test_value_density_prefers_big_cold_block_over_small_cold_block(self):
        controller, collector, tracker, clock = build(9)
        for b in collector.all_blocks():
            b.access_count = 100
            b.last_access = clock.now()
        small = collector.get("b01")
        big = collector.get("b02")
        for b in (small, big):
            b.access_count = 1
            b.last_access = clock.now() - 3600.0
        # Same importance, but the 10 MB block holds more memory per unit of
        # value than the 5 MB block, so it should be shed first.
        tracker.move_block(small, Tier.GPU, new_size_mb=5.0)  # total now 85 MB

        decisions = controller.step()
        self.assertEqual(decisions[0].block_id, "b02")

    def test_deterministic_across_identical_runs(self):
        def run():
            controller, _, _, _ = build(9)
            return [(d.block_id, d.executed_action) for d in controller.step()]

        self.assertEqual(run(), run())


class TestFallbackAndGuards(unittest.TestCase):
    def test_cpu_full_migrations_fall_back_to_evict(self):
        controller, _, tracker, _ = build(
            10, cpu_mb=15.0, decision_engine=_FixedActionEngine("MIGRATE")
        )
        decisions = controller.step()
        executed = [d.executed_action for d in decisions]
        self.assertEqual(executed[0], "MIGRATE")
        self.assertIn("EVICT", executed[1:])
        self.assertTrue(any(d.fallback_reason == "cpu_full" for d in decisions))
        self.assertLessEqual(tracker.cpu.used_mb, tracker.cpu.capacity_mb)
        self.assertLessEqual(tracker.gpu_pressure(), 0.70 + 1e-9)

    def test_compress_only_policy_terminates_and_compresses_each_block_once(self):
        # Compression frees only 2 MB per block, so 10 blocks can reach just 80%,
        # never the 70% target. The controller must give up cleanly, not loop.
        controller, _, tracker, _ = build(
            10, decision_engine=_FixedActionEngine("COMPRESS"), compression_benefit=2.0
        )
        decisions = controller.step()
        compressed = [d for d in decisions if d.executed_action == "COMPRESS"]
        self.assertEqual(len(compressed), 10)
        self.assertEqual(len({d.block_id for d in compressed}), 10)  # once each
        self.assertAlmostEqual(tracker.gpu_pressure(), 0.8)
        self.assertTrue(controller.shedding)  # still above target; will retry next cycle

    def test_real_engine_uses_cheap_actions_for_hot_blocks_when_evict_is_costly(self):
        controller, collector, _, clock = build(9, compression_benefit=4.0)
        for b in collector.all_blocks():
            b.access_count = 200
            b.last_access = clock.now()
        decisions = controller.step()
        self.assertTrue(decisions)
        # Hot blocks carry a high recompute penalty: not everything should be evicted.
        self.assertTrue(any(d.executed_action != "EVICT" for d in decisions))


if __name__ == "__main__":
    unittest.main()
