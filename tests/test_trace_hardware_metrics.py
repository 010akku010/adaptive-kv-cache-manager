"""Tests for the trace generator, the hardware cost model and the metrics maths."""

import statistics
import unittest
from collections import Counter

from src.hardware import (
    A10_LLAMA_3_8B,
    DEFAULT_PROFILE,
    LLAMA_3_2_1B,
    PROFILES,
)
from src.metrics import SimMetrics, percentile
from src.trace import WORKLOADS, WorkloadConfig, generate_trace


class TestTraceGenerator(unittest.TestCase):
    CFG = WorkloadConfig(name="t", duration_s=300.0, arrival_rate=5.0, num_contexts=50, seed=7)

    def test_same_seed_gives_identical_trace(self):
        a, b = generate_trace(self.CFG), generate_trace(self.CFG)
        self.assertEqual(a.requests, b.requests)
        self.assertEqual(a.initial_blocks, b.initial_blocks)

    def test_different_seeds_differ(self):
        self.assertNotEqual(
            generate_trace(self.CFG).requests, generate_trace(self.CFG.with_seed(8)).requests
        )

    def test_arrivals_are_sorted_and_inside_the_window(self):
        times = [r.time_s for r in generate_trace(self.CFG).requests]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(0.0 <= t < self.CFG.duration_s for t in times))

    def test_poisson_mean_rate_is_close_to_target(self):
        n = len(generate_trace(self.CFG).requests)
        self.assertAlmostEqual(n / self.CFG.duration_s, self.CFG.arrival_rate, delta=0.5)

    def test_bursty_preserves_mean_rate_but_is_burstier(self):
        # A single bursty trace is noisy (only ~50 on/off cycles), so check the
        # mean rate over several seeds. (Over 360 seeds the mean is 5.05 +/- 0.04.)
        def trace_for(seed, arrival):
            return generate_trace(WorkloadConfig(
                name="b", duration_s=2000.0, arrival=arrival, num_contexts=20, seed=seed))

        seeds = range(12)
        rates = [len(trace_for(s, "bursty").requests) / 2000.0 for s in seeds]
        self.assertAlmostEqual(statistics.fmean(rates), 5.0, delta=0.4)

        def dispersion(ts):  # variance-to-mean ratio of per-second counts
            c = Counter(int(t) for t in ts)
            counts = [c.get(i, 0) for i in range(2000)]
            return statistics.pvariance(counts) / statistics.fmean(counts)

        bursty = dispersion([r.time_s for r in trace_for(1, "bursty").requests])
        poisson = dispersion([r.time_s for r in trace_for(1, "poisson").requests])
        # ~1 for Poisson, clearly larger when bursty.
        self.assertGreater(bursty, 1.5 * poisson)

    def test_impossible_burst_multiplier_is_rejected(self):
        # 10 s on / 30 s off with a 10x burst would need a negative off-rate.
        cfg = WorkloadConfig(name="bad", arrival="bursty", burst_rate_multiplier=10.0)
        with self.assertRaises(ValueError):
            generate_trace(cfg)

    def test_unknown_arrival_process_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_trace(WorkloadConfig(name="x", arrival="weekly"))

    def test_zipf_concentrates_requests_and_uniform_does_not(self):
        def top_share(s):
            cfg = WorkloadConfig(name="z", duration_s=2000.0, num_contexts=100, zipf_s=s, seed=2)
            counts = Counter(r.context_id for r in generate_trace(cfg).requests)
            top10 = sum(c for _, c in counts.most_common(10))
            return top10 / sum(counts.values())

        self.assertGreater(top_share(1.0), 0.5)  # a few conversations dominate
        self.assertLess(top_share(0.0), 0.2)  # ~10% of conversations get ~10% of requests

    def test_context_lengths_respect_bounds(self):
        cfg = WorkloadConfig(name="l", num_contexts=500, min_context_blocks=4, max_context_blocks=64, seed=3)
        blocks = generate_trace(cfg).initial_blocks
        self.assertGreaterEqual(min(blocks), 4)
        self.assertLessEqual(max(blocks), 64)

    def test_every_request_targets_a_real_conversation_and_adds_blocks(self):
        trace = generate_trace(self.CFG)
        for r in trace.requests:
            self.assertLess(r.context_id, self.CFG.num_contexts)
            self.assertGreaterEqual(r.new_blocks, 1)
            self.assertGreaterEqual(r.output_tokens, 8)

    def test_presets_are_valid(self):
        for name, cfg in WORKLOADS.items():
            with self.subTest(workload=name):
                self.assertEqual(cfg.name, name)
                self.assertGreater(len(generate_trace(cfg.with_seed(0)).requests), 0)


class TestHardwareModel(unittest.TestCase):
    def test_kv_bytes_per_token_matches_the_formula(self):
        # 2 (K,V) x 16 layers x 8 KV heads x 64 dim x 2 bytes = 32,768 B
        self.assertEqual(LLAMA_3_2_1B.kv_bytes_per_token, 32_768)
        # Llama-3-8B: 2 x 32 x 8 x 128 x 2 = 131,072 B (128 KiB) per token
        self.assertEqual(A10_LLAMA_3_8B.model.kv_bytes_per_token, 131_072)

    def test_block_size(self):
        self.assertAlmostEqual(LLAMA_3_2_1B.block_size_mb(16), 0.524288)

    def test_transfer_time_is_setup_plus_size_over_bandwidth(self):
        p = DEFAULT_PROFILE  # 12 GB/s == 12 MB/ms, 0.02 ms setup
        self.assertAlmostEqual(p.transfer_ms(12.0), 0.02 + 1.0)
        self.assertAlmostEqual(p.transfer_ms(0.0), 0.02)

    def test_prefill_time_scales_linearly_with_tokens(self):
        p = DEFAULT_PROFILE  # 6000 tokens/s
        self.assertAlmostEqual(p.prefill_ms(6000), 1000.0)
        self.assertAlmostEqual(p.prefill_ms(16), 16 / 6000 * 1000.0)

    def test_swapping_a_block_in_is_cheaper_than_recomputing_it_on_default_hardware(self):
        # The physical fact that makes MIGRATE worthwhile at all.
        p = DEFAULT_PROFILE
        block_mb = p.block_size_mb(16)
        self.assertLess(p.transfer_ms(block_mb), p.prefill_ms(16) / 10)

    def test_profiles_registry(self):
        self.assertIn(DEFAULT_PROFILE.name, PROFILES)
        for name, prof in PROFILES.items():
            self.assertEqual(name, prof.name)


class TestMetrics(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        data = list(range(1, 101))
        self.assertEqual(percentile(data, 50), 50)
        self.assertEqual(percentile(data, 99), 99)
        self.assertEqual(percentile(data, 100), 100)
        self.assertEqual(percentile([], 99), 0.0)
        self.assertEqual(percentile([7.0], 99), 7.0)

    def test_nothing_is_recorded_while_not_measuring(self):
        m = SimMetrics(measuring=False)
        m.record_request(5.0, True)
        m.record_gpu_hit(False)
        m.record_recompute(3.0)
        m.record_oom()
        m.sample_gpu_util(0.9)
        self.assertEqual((m.requests, m.accesses, m.oom_events, m.gpu_util_samples), (0, 0, 0, []))

    def test_hit_ratios_and_exposure(self):
        m = SimMetrics()
        for _ in range(6):
            m.record_gpu_hit(False)
        m.record_gpu_hit(True)  # a compressed GPU hit
        m.record_cpu_hit(0.5, 1.0, 0.0, False)
        m.record_recompute(10.0)
        s = m.summary()
        self.assertEqual(s["accesses"], 9)
        self.assertAlmostEqual(s["gpu_hit_ratio"], 7 / 9, places=3)
        self.assertAlmostEqual(s["cache_hit_ratio"], 8 / 9, places=3)
        self.assertAlmostEqual(s["recompute_ratio"], 1 / 9, places=3)
        self.assertAlmostEqual(s["compressed_read_ratio"], 1 / 9, places=3)

    def test_reuse_latency_is_tracked_separately_from_cold_requests(self):
        m = SimMetrics()
        m.record_request(100.0, False)
        m.record_request(10.0, True)
        s = m.summary()
        self.assertEqual((s["requests"], s["reuse_requests"]), (2, 1))
        self.assertEqual(s["reuse_ttft_p50_ms"], 10.0)

    def test_empty_summary_does_not_divide_by_zero(self):
        s = SimMetrics().summary()
        self.assertEqual(s["gpu_hit_ratio"], 0.0)
        self.assertEqual(s["ttft_p99_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
