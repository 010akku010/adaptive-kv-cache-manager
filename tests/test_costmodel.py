"""
Tests for the millisecond cost model and engine.

Uses the toy hardware profile from tests/helpers.py (1 MB block, 1 ms swap-in,
10 ms recompute, 0.5 ms/MB decompress, compression removes 50% of a block) so
every expected number below can be checked by hand.

For a block with importance ~= 1 (just accessed, accessed 100 times) and
reuse_scale = 2, the expected number of future reads is 2, and at GPU
pressure 1.0 the memory price is 1 x 1^2 x 10 ms/MB x 2 = 20 ms/MB:

    EVICT    benefit 20 x 1.0  = 20    cost 2 x 10            = 20   -> net  0
    MIGRATE  benefit 20 x 1.0  = 20    cost 2 x 1             =  2   -> net 18
    COMPRESS benefit 20 x 0.5  = 10    cost 2 x (0.5 + 0.5)   =  2   -> net  8
"""

import unittest

from src.clock import ManualClock
from src.costmodel import CostModel, MsImportanceEngine, MsParams
from src.decision_engine import DecisionEngine
from src.metadata_collector import MetadataCollector
from tests.helpers import toy_profile


def make_engine(cpu_free=None, params=None, tau=30.0):
    clock = ManualClock(1000.0)
    collector = MetadataCollector(clock=clock)
    engine = MsImportanceEngine(
        CostModel(toy_profile(), block_tokens=16),
        params=params or MsParams(reuse_scale=2.0, price_multiplier=2.0, quality_penalty_ms=0.5),
        recency_tau_s=tau,
        clock=clock,
        cpu_free_mb=cpu_free,
    )
    return engine, collector, clock


def hot_block(collector, bid="hot", size=1.0):
    block = collector.register_block(bid, size_mb=size, migration_cost_ms=1.0, compression_benefit_mb=0.5)
    block.access_count = 100  # frequency score ~ 1
    return block  # just registered: recency score = 1


def cold_block(collector, clock, bid="cold", size=1.0):
    block = collector.register_block(bid, size_mb=size, migration_cost_ms=1.0, compression_benefit_mb=0.5)
    block.access_count = 1
    block.last_access = clock.now() - 100_000.0  # recency score ~ 0
    return block


class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.cm = CostModel(toy_profile(), block_tokens=16)
        self.collector = MetadataCollector()

    def test_block_geometry_and_unit_costs(self):
        self.assertAlmostEqual(self.cm.block_mb, 1.0)
        self.assertAlmostEqual(self.cm.recompute_ms_per_mb, 10.0)

    def test_recompute_is_proportional_to_full_size(self):
        block = self.collector.register_block("b", size_mb=1.0)
        self.assertAlmostEqual(self.cm.recompute_ms(block), 10.0)
        block.size_mb = 0.5
        block.compressed = True  # a compressed 0.5 MB block was 1.0 MB at full precision
        self.assertAlmostEqual(self.cm.uncompressed_mb(block), 1.0)
        self.assertAlmostEqual(self.cm.recompute_ms(block), 10.0)

    def test_swap_in_prefers_the_recorded_migration_cost(self):
        block = self.collector.register_block("b", size_mb=1.0, migration_cost_ms=3.0)
        self.assertAlmostEqual(self.cm.swap_in_ms(block), 3.0)
        block.migration_cost_ms = 0.0
        self.assertAlmostEqual(self.cm.swap_in_ms(block), 1.0)  # falls back to size / bandwidth

    def test_decompress_cost_scales_with_stored_size(self):
        block = self.collector.register_block("b", size_mb=0.5)
        self.assertAlmostEqual(self.cm.decompress_ms(block), 0.25)


class TestActionValues(unittest.TestCase):
    def test_hand_computed_estimates_for_a_hot_block(self):
        engine, collector, _ = make_engine()
        v = engine.compute_action_values(hot_block(collector), gpu_pressure=1.0)
        self.assertAlmostEqual(v["KEEP"].benefit - v["KEEP"].cost, 0.0)
        self.assertAlmostEqual(v["EVICT"].benefit, 20.0, places=3)
        self.assertAlmostEqual(v["EVICT"].cost, 20.0, places=3)
        self.assertAlmostEqual(v["MIGRATE"].benefit, 20.0, places=3)
        self.assertAlmostEqual(v["MIGRATE"].cost, 2.0, places=3)
        self.assertAlmostEqual(v["COMPRESS"].benefit, 10.0, places=3)
        self.assertAlmostEqual(v["COMPRESS"].cost, 2.0, places=3)

    def test_decision_for_a_hot_block_with_room_on_cpu_is_migrate(self):
        engine, collector, _ = make_engine(cpu_free=lambda: 100.0)
        v = engine.compute_action_values(hot_block(collector), 1.0)
        v.pop("KEEP")
        action, utility = DecisionEngine().decide(v)
        self.assertEqual(action, "MIGRATE")
        self.assertAlmostEqual(utility, 18.0, places=3)

    def test_full_cpu_removes_migrate_so_compress_wins_for_hot_blocks(self):
        engine, collector, _ = make_engine(cpu_free=lambda: 0.0)
        v = engine.compute_action_values(hot_block(collector), 1.0)
        self.assertNotIn("MIGRATE", v)
        v.pop("KEEP")
        action, utility = DecisionEngine().decide(v)
        self.assertEqual(action, "COMPRESS")
        self.assertAlmostEqual(utility, 8.0, places=3)

    def test_full_cpu_still_evicts_a_cold_block(self):
        # Cold block: ~0.18 expected reads, so evicting it costs ~1.8 ms while freeing twice
        # as much memory as compressing it (and MIGRATE is not on offer).
        engine, collector, clock = make_engine(cpu_free=lambda: 0.0)
        v = engine.compute_action_values(cold_block(collector, clock), 1.0)
        v.pop("KEEP")
        self.assertEqual(DecisionEngine().decide(v)[0], "EVICT")

    def test_low_pressure_keeps_everything(self):
        engine, collector, _ = make_engine()
        v = engine.compute_action_values(hot_block(collector), gpu_pressure=0.1)
        action, utility = DecisionEngine().decide(v)  # KEEP is a candidate here
        self.assertEqual(action, "KEEP")
        self.assertAlmostEqual(utility, 0.0)

    def test_compress_is_not_offered_for_a_compressed_block(self):
        engine, collector, _ = make_engine()
        block = hot_block(collector)
        block.compressed = True
        self.assertNotIn("COMPRESS", engine.compute_action_values(block, 1.0))

    def test_compress_is_not_offered_when_nothing_would_be_saved(self):
        engine, collector, _ = make_engine()
        block = hot_block(collector)
        block.compression_benefit_mb = 0.0
        self.assertNotIn("COMPRESS", engine.compute_action_values(block, 1.0))

    def test_quality_penalty_can_price_compression_out(self):
        # compress net = 10 - 2 x (0.5 + q); evict net (cpu full) = 0. Break-even at q = 4.5.
        engine, collector, _ = make_engine(
            cpu_free=lambda: 0.0,
            params=MsParams(reuse_scale=2.0, price_multiplier=2.0, quality_penalty_ms=6.0),
        )
        v = engine.compute_action_values(hot_block(collector), 1.0)
        v.pop("KEEP")
        self.assertNotEqual(DecisionEngine().decide(v)[0], "COMPRESS")

    def test_memory_price_rises_with_pressure(self):
        engine, _, _ = make_engine()
        prices = [engine.memory_price_ms_per_mb(p) for p in (0.0, 0.5, 0.85, 1.0)]
        self.assertEqual(prices, sorted(prices))
        self.assertAlmostEqual(prices[0], 0.0)
        self.assertAlmostEqual(prices[-1], 20.0)
        self.assertAlmostEqual(engine.memory_price_ms_per_mb(5.0), 20.0)  # pressure is clamped


class TestVictimScore(unittest.TestCase):
    def test_colder_blocks_score_lower_and_are_shed_first(self):
        engine, collector, clock = make_engine()
        self.assertLess(
            engine.victim_score(cold_block(collector, clock)),
            engine.victim_score(hot_block(collector)),
        )

    def test_ranking_is_independent_of_reuse_scale(self):
        def order(scale):
            engine, collector, clock = make_engine(params=MsParams(reuse_scale=scale))
            blocks = [hot_block(collector, "a"), cold_block(collector, clock, "b")]
            return sorted(blocks, key=engine.victim_score)[0].block_id

        self.assertEqual(order(0.5), order(8.0))

    def test_score_is_the_cheapest_relief_per_mb(self):
        # hot block, cpu has room: migrate costs 2 ms for 1 MB = 2 ms/MB (cheapest);
        # evict would be 20 ms/MB, compress 2 ms / 0.5 MB = 4 ms/MB.
        engine, collector, _ = make_engine(cpu_free=lambda: 100.0)
        self.assertAlmostEqual(engine.victim_score(hot_block(collector)), 2.0, places=3)
        # cpu full: no migrate; compress (4 ms/MB) beats evict (20 ms/MB).
        engine, collector, _ = make_engine(cpu_free=lambda: 0.0)
        self.assertAlmostEqual(engine.victim_score(hot_block(collector)), 4.0, places=3)

    def test_expected_reads_scale_with_importance(self):
        engine, collector, clock = make_engine()
        self.assertGreater(engine.expected_reads(hot_block(collector)), 1.9)
        self.assertLess(engine.expected_reads(cold_block(collector, clock)), 0.5)


if __name__ == "__main__":
    unittest.main()


class TestHostMemoryPrice(unittest.TestCase):
    """The host tier has a shadow price too (cpu_price_multiplier; 0 = disabled)."""

    def engine(self, multiplier, utilization=1.0):
        return make_engine(
            cpu_free=lambda: 100.0,
            params=MsParams(reuse_scale=2.0, price_multiplier=2.0, quality_penalty_ms=0.5,
                            cpu_price_multiplier=multiplier),
        )[:3], utilization

    def build(self, multiplier, utilization=1.0):
        engine, collector, clock = make_engine(
            cpu_free=lambda: 100.0,
            params=MsParams(reuse_scale=2.0, price_multiplier=2.0, quality_penalty_ms=0.5,
                            cpu_price_multiplier=multiplier),
        )
        engine._cpu_utilization = lambda: utilization
        return engine, collector, clock

    def test_disabled_by_default_so_earlier_behaviour_is_unchanged(self):
        self.assertEqual(MsParams().cpu_price_multiplier, 0.0)
        engine, collector, _ = self.build(0.0)
        self.assertEqual(engine.cpu_price_ms_per_mb(), 0.0)
        v = engine.compute_action_values(hot_block(collector), 1.0)
        self.assertAlmostEqual(v["MIGRATE"].cost, 2.0, places=3)

    def test_price_rises_with_host_utilisation(self):
        prices = []
        for u in (0.0, 0.5, 0.9, 1.0):
            engine, _, _ = self.build(1.0, utilization=u)
            prices.append(engine.cpu_price_ms_per_mb())
        self.assertEqual(prices, sorted(prices))
        self.assertAlmostEqual(prices[0], 0.0)
        self.assertAlmostEqual(prices[-1], 10.0)  # 1 x 1^2 x 10 ms/MB

    def test_unobservable_host_gives_no_price(self):
        engine, _, _ = make_engine(params=MsParams(cpu_price_multiplier=5.0))
        self.assertEqual(engine.cpu_price_ms_per_mb(), 0.0)

    def test_migrate_cost_includes_the_host_price(self):
        engine, collector, _ = self.build(2.0)  # price 20 ms/MB at full host utilisation
        v = engine.compute_action_values(hot_block(collector), 1.0)
        self.assertAlmostEqual(v["MIGRATE"].cost, 2.0 + 20.0, places=3)

    def test_expensive_host_pushes_hot_blocks_to_compress_and_cold_blocks_to_evict(self):
        engine, collector, clock = self.build(2.0)
        hot = engine.compute_action_values(hot_block(collector), 1.0)
        hot.pop("KEEP")
        self.assertEqual(DecisionEngine().decide(hot)[0], "COMPRESS")  # nets: migrate -2, compress 8, evict 0
        cold = engine.compute_action_values(cold_block(collector, clock), 1.0)
        cold.pop("KEEP")
        self.assertEqual(DecisionEngine().decide(cold)[0], "EVICT")  # not worth parking on a full host

    def test_cheap_host_still_prefers_migrate(self):
        engine, collector, _ = self.build(2.0, utilization=0.1)  # price 0.2 ms/MB
        v = engine.compute_action_values(hot_block(collector), 1.0)
        v.pop("KEEP")
        self.assertEqual(DecisionEngine().decide(v)[0], "MIGRATE")

    def test_victim_score_accounts_for_the_host_price(self):
        engine, collector, _ = self.build(2.0)
        # migrate per MB = 2 + 20 = 22; compress per MB = 4 (cheapest); evict = 20.
        self.assertAlmostEqual(engine.victim_score(hot_block(collector)), 4.0, places=3)
        engine, collector, _ = self.build(0.0)
        self.assertAlmostEqual(engine.victim_score(hot_block(collector)), 2.0, places=3)
