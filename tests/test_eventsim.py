"""
Tests for the event-driven simulator.

The core scenarios use a toy hardware profile with round numbers (1 MB block,
1 ms transfer, 10 ms recompute) so every expected latency is computed by hand.
"""

import unittest

from src.eventsim import EventSimulator, SimConfig
from src.metadata_collector import Tier
from src.policies import default_policy_specs
from src.trace import Request, WorkloadConfig, generate_trace
from tests.helpers import (
    EVICT_ALL,
    MIGRATE_ALL,
    NULL_POLICY,
    make_trace,
    run,
    toy_config,
    toy_profile,
)


def req(i, t, ctx=0, new=1, out=10):
    return Request(request_id=i, time_s=t, context_id=ctx, new_blocks=new, output_tokens=out)


class TestLatencyPhysics(unittest.TestCase):
    def test_cold_request_pays_prefill_for_all_new_tokens(self):
        # 3 initial + 1 grown block = 4 blocks = 64 tokens -> 64/1600 s = 40 ms.
        _, result = run(make_trace([req(0, 0.0)], [3]), NULL_POLICY)
        self.assertEqual(result.metrics.ttft_ms, [40.0])
        self.assertEqual(result.metrics.reuse_ttft_ms, [])  # first touch is not a "reuse"
        self.assertEqual(result.summary["accesses"], 0)  # nothing was read from cache

    def test_gpu_resident_reuse_costs_only_the_new_tokens(self):
        sim, result = run(make_trace([req(0, 0.0), req(1, 1.0)], [3]), NULL_POLICY)
        self.assertEqual(result.metrics.reuse_ttft_ms, [10.0])  # one new block = 16 tokens = 10 ms
        self.assertEqual(result.metrics.gpu_hits, 4)
        self.assertEqual(result.summary["gpu_hit_ratio"], 1.0)
        self.assertEqual(sim.check_invariants(), [])

    def test_evicted_blocks_are_recomputed(self):
        sim, result = run(make_trace([req(0, 0.0), req(1, 1.0)], [3]), EVICT_ALL)
        # 4 blocks x 10 ms recompute + 10 ms for the one new block.
        self.assertEqual(result.metrics.reuse_ttft_ms, [50.0])
        self.assertEqual(result.metrics.recomputes, 4)
        self.assertEqual(result.summary["recompute_ratio"], 1.0)
        self.assertEqual(sim.check_invariants(), [])

    def test_migrated_blocks_pay_pcie_swap_in(self):
        sim, result = run(make_trace([req(0, 0.0), req(1, 1.0)], [3]), MIGRATE_ALL)
        # 4 blocks x 1 ms transfer + 10 ms for the new block.
        self.assertAlmostEqual(result.metrics.reuse_ttft_ms[0], 14.0, places=6)
        self.assertEqual(result.metrics.cpu_hits, 4)
        self.assertAlmostEqual(result.metrics.swapped_in_mb, 4.0)
        self.assertEqual(sim.check_invariants(), [])

    def test_concurrent_swap_ins_queue_on_the_pcie_link(self):
        trace = make_trace(
            [req(0, 0.0, ctx=0, new=0), req(1, 0.0, ctx=1, new=0),
             req(2, 1.0, ctx=0, new=0), req(3, 1.0, ctx=1, new=0)],
            [2, 2],
        )
        _, result = run(trace, MIGRATE_ALL)
        first, second = result.metrics.reuse_ttft_ms
        self.assertAlmostEqual(first, 2.0, places=6)  # 2 blocks x 1 ms, link idle
        # The second request waits 2 ms for the link, then transfers 2 blocks.
        self.assertAlmostEqual(second, 4.0, places=6)
        self.assertAlmostEqual(result.metrics.pcie_wait_ms, 2.0, places=6)

    def test_selective_load_recomputes_when_transfer_is_slower(self):
        slow = toy_config(profile=toy_profile(pcie_gb_per_s=0.01))  # 1 MB takes 100 ms
        trace = make_trace([req(0, 0.0), req(1, 1.0)], [3])
        _, result = run(trace, MIGRATE_ALL, slow)
        self.assertEqual(result.metrics.cpu_hits, 0)
        self.assertEqual(result.metrics.recomputes, 4)
        self.assertAlmostEqual(result.metrics.reuse_ttft_ms[0], 50.0, places=6)

    def test_selective_load_can_be_disabled(self):
        slow = toy_config(profile=toy_profile(pcie_gb_per_s=0.01), selective_load=False)
        trace = make_trace([req(0, 0.0), req(1, 1.0)], [3])
        _, result = run(trace, MIGRATE_ALL, slow)
        self.assertEqual(result.metrics.cpu_hits, 4)
        self.assertAlmostEqual(result.metrics.reuse_ttft_ms[0], 4 * 100.0 + 10.0, places=6)


class TestPinningAndCapacity(unittest.TestCase):
    def test_blocks_of_a_running_request_are_never_moved(self):
        # 1,000,000 output tokens => the request holds its blocks for the whole run.
        trace = make_trace([req(0, 0.0, new=0, out=1_000_000)], [2], duration_s=2.0)
        sim, result = run(trace, EVICT_ALL)
        self.assertEqual(result.metrics.evictions, 0)
        self.assertTrue(all(b.location == Tier.GPU for b in sim.collector.all_blocks()))
        # ...and everything is unpinned again once the request completes.
        self.assertTrue(all(b.pin_count == 0 for b in sim.collector.all_blocks()))

    def test_policy_failing_to_free_memory_triggers_forced_eviction(self):
        cfg = toy_config(gpu_pool_mb=3.0)
        trace = make_trace([req(0, 0.0, ctx=0, new=0), req(1, 1.0, ctx=1, new=0)], [2, 2])
        sim, result = run(trace, NULL_POLICY, cfg)
        self.assertEqual(result.metrics.forced_evictions, 1)
        self.assertEqual(result.metrics.oom_events, 0)
        self.assertEqual(sim.check_invariants(), [])

    def test_oom_when_everything_is_pinned(self):
        cfg = toy_config(gpu_pool_mb=3.0)
        trace = make_trace([req(0, 0.0, new=0)], [4])  # one request needs 4 MB, pool has 3
        sim, result = run(trace, NULL_POLICY, cfg)
        self.assertEqual(result.metrics.oom_events, 1)
        self.assertEqual(result.metrics.forced_evictions, 0)
        self.assertEqual(sim.check_invariants(), [])

    def test_cpu_tier_cleanup_keeps_cpu_within_capacity(self):
        # 8 contexts x 2 blocks = 16 MB migrated into a 10 MB CPU pool.
        cfg = toy_config(cpu_pool_mb=10.0)
        reqs = [req(i, 0.01 * i, ctx=i, new=0) for i in range(8)]
        sim, result = run(make_trace(reqs, [2] * 8, duration_s=3.0), MIGRATE_ALL, cfg)
        self.assertLessEqual(sim.tracker.cpu.used_mb, 10.0)
        self.assertGreater(result.metrics.cpu_evictions + result.metrics.evictions, 0)
        self.assertEqual(sim.check_invariants(), [])


class TestMeasurementWindow(unittest.TestCase):
    def test_warmup_requests_are_excluded(self):
        cfg = toy_config(warmup_s=5.0)
        trace = make_trace([req(0, 1.0), req(1, 10.0)], [3], duration_s=12.0)
        _, result = run(trace, NULL_POLICY, cfg)
        self.assertEqual(result.metrics.requests, 1)  # only the t=10 request counts


class TestEndToEndConsistency(unittest.TestCase):
    CFG = SimConfig(gpu_pool_mb=128.0, cpu_pool_mb=512.0, warmup_s=10.0)
    WORKLOAD = WorkloadConfig(name="small", duration_s=60.0, num_contexts=40, arrival_rate=4.0, seed=3)

    def test_every_policy_leaves_consistent_bookkeeping(self):
        trace = generate_trace(self.WORKLOAD)
        for spec in default_policy_specs():
            with self.subTest(policy=spec.name):
                sim = EventSimulator(trace, self.CFG, spec)
                result = sim.run()
                self.assertEqual(sim.check_invariants(), [])
                self.assertTrue(all(b.pin_count == 0 for b in sim.collector.all_blocks()))
                self.assertGreater(result.summary["requests"], 0)

    def test_runs_are_deterministic(self):
        trace = generate_trace(self.WORKLOAD)
        for spec in default_policy_specs():
            with self.subTest(policy=spec.name):
                a = EventSimulator(trace, self.CFG, spec).run().summary
                b = EventSimulator(trace, self.CFG, spec).run().summary
                self.assertEqual(a, b)

    def test_all_policies_see_the_same_requests(self):
        trace = generate_trace(self.WORKLOAD)
        counts = {EventSimulator(trace, self.CFG, s).run().summary["requests"] for s in default_policy_specs()}
        self.assertEqual(len(counts), 1)

    def test_offload_beats_evict_only_when_transfers_are_cheap(self):
        # Physics sanity check: swap-in (~0.06 ms) is far cheaper than recompute (~2.7 ms).
        trace = generate_trace(self.WORKLOAD)
        by_name = {s.name: EventSimulator(trace, self.CFG, s).run().summary for s in default_policy_specs()}
        self.assertGreater(by_name["LRU+offload"]["cache_hit_ratio"], by_name["LRU"]["cache_hit_ratio"])
        self.assertLess(by_name["LRU+offload"]["reuse_ttft_mean_ms"], by_name["LRU"]["reuse_ttft_mean_ms"])


if __name__ == "__main__":
    unittest.main()


class TestSignalsMatchMetrics(unittest.TestCase):
    """The tuner's always-on counters must agree with the reporting metrics (warm-up off)."""

    def _check(self, sim, result):
        s, m = sim.signals, result.metrics
        self.assertEqual(s.requests, m.requests)
        self.assertEqual(s.gpu_hits, m.gpu_hits)
        self.assertEqual(s.cpu_hits, m.cpu_hits)
        self.assertEqual(s.recomputes, m.recomputes)
        self.assertEqual(s.compressed_reads, m.compressed_reads)
        self.assertEqual(s.forced_evictions, m.forced_evictions)
        self.assertEqual(s.cpu_evictions, m.cpu_evictions)
        self.assertEqual(s.oom_events, m.oom_events)
        self.assertAlmostEqual(s.pcie_wait_ms, m.pcie_wait_ms, places=6)
        self.assertEqual(s.accesses, m.accesses)

    def test_counters_agree_on_hand_built_scenarios(self):
        cases = [
            (make_trace([req(0, 0.0), req(1, 1.0)], [3]), NULL_POLICY, toy_config()),
            (make_trace([req(0, 0.0), req(1, 1.0)], [3]), MIGRATE_ALL, toy_config()),
            (make_trace([req(0, 0.0), req(1, 1.0)], [3]), EVICT_ALL, toy_config()),
            (make_trace([req(0, 0.0, new=0)], [4]), NULL_POLICY, toy_config(gpu_pool_mb=3.0)),  # OOM
            (make_trace([req(0, 0.0, ctx=0, new=0), req(1, 1.0, ctx=1, new=0)], [2, 2]),
             NULL_POLICY, toy_config(gpu_pool_mb=3.0)),  # forced eviction
        ]
        for trace, spec, cfg in cases:
            with self.subTest(policy=spec.name, gpu=cfg.gpu_pool_mb):
                sim, result = run(trace, spec, cfg)
                self._check(sim, result)

    def test_swap_ins_are_not_counted_as_oom(self):
        # Regression: an OOM counter once fired on every CPU swap-in.
        sim, result = run(make_trace([req(0, 0.0), req(1, 1.0)], [3]), MIGRATE_ALL)
        self.assertGreater(result.metrics.cpu_hits, 0)
        self.assertEqual(sim.signals.oom_events, 0)

    def test_counters_agree_on_a_generated_workload(self):
        cfg = SimConfig(gpu_pool_mb=64.0, cpu_pool_mb=128.0, warmup_s=0.0)
        trace = generate_trace(WorkloadConfig(name="s", duration_s=40.0, num_contexts=30, seed=5))
        for spec in default_policy_specs():
            with self.subTest(policy=spec.name):
                sim = EventSimulator(trace, cfg, spec)
                self._check(sim, sim.run())

    def test_reuse_gaps_are_recorded(self):
        sim, _ = run(make_trace([req(0, 0.0), req(1, 5.0), req(2, 8.0)], [2]), NULL_POLICY)
        gaps = sim.signals.reuse_gaps()
        self.assertTrue(gaps)
        self.assertTrue(all(g >= 0.0 for g in gaps))
        self.assertIn(3.0, [round(g, 6) for g in gaps])  # the gap between t=5 and t=8
