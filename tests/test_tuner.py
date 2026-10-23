"""Tests for the signals module, the Adaptive Weight Tuner and the ladder baseline."""

import random
import unittest
from dataclasses import replace

from src.clock import ManualClock
from src.controller import ControllerConfig
from src.costmodel import CostModel, MsImportanceEngine, MsParams
from src.eventsim import EventSimulator, SimConfig
from src.policies import ladder_spec, ms_adaptive_spec
from src.signals import CounterSnapshot, Signals
from src.trace import WorkloadConfig, generate_trace
from src.tuner import AdaptiveWeightTuner, TunerConfig, quantile
from tests.helpers import make_trace, run, toy_config, toy_profile


class TestSignals(unittest.TestCase):
    def test_counters_and_window_difference(self):
        s = Signals()
        before = s.snapshot()
        s.record_request()
        s.record_gpu_hit(compressed=True)
        s.record_cpu_hit(wait_ms=2.0, compressed=False)
        s.record_recompute()
        s.record_forced_eviction()
        s.record_oom()
        w = s.snapshot().minus(before)
        self.assertEqual((w.requests, w.accesses, w.gpu_hits, w.cpu_hits, w.recomputes), (1, 3, 1, 1, 1))
        self.assertEqual((w.compressed_reads, w.forced_evictions, w.oom_events), (1, 1, 1))
        self.assertAlmostEqual(w.pcie_wait_ms, 2.0)
        self.assertAlmostEqual(w.compressed_ratio, 1 / 3)
        self.assertAlmostEqual(w.recompute_ratio, 1 / 3)
        self.assertAlmostEqual(w.gpu_hit_ratio, 1 / 3)

    def test_window_ratios_with_no_accesses_are_zero(self):
        w = CounterSnapshot().minus(CounterSnapshot())
        self.assertEqual((w.compressed_ratio, w.recompute_ratio, w.gpu_hit_ratio), (0.0, 0.0, 0.0))

    def test_reuse_gaps_are_bounded_and_ignore_negatives(self):
        s = Signals(max_gaps=3)
        for g in (1.0, 2.0, 3.0, 4.0, -5.0):
            s.record_reuse_gap(g)
        self.assertEqual(s.reuse_gaps(), [2.0, 3.0, 4.0])


class TestQuantile(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertEqual(quantile([1.0], 0.9), 1.0)
        self.assertEqual(quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0)
        self.assertAlmostEqual(quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(quantile([5.0, 1.0, 3.0], 1.0), 5.0)


def make_tuner(config=None, params=None, controller=None):
    clock = ManualClock(0.0)
    signals = Signals()
    engine = MsImportanceEngine(
        CostModel(toy_profile(), 16), params=params or MsParams(quality_penalty_ms=0.5), clock=clock
    )
    cc = controller or ControllerConfig(high_watermark=0.85, low_watermark=0.70)
    cfg = config or TunerConfig(adapt_tau=False)
    return AdaptiveWeightTuner(engine, cc, signals, clock, cfg), engine, cc, signals, clock


def feed(signals, accesses=100, compressed=0, forced=0, oom=0, recomputes=0):
    """Add one window's worth of activity to the signals."""
    for i in range(accesses - recomputes):
        signals.record_gpu_hit(compressed=i < compressed)
    for _ in range(recomputes):
        signals.record_recompute()
    for _ in range(forced):
        signals.record_forced_eviction()
    for _ in range(oom):
        signals.record_oom()


class TestConfigValidation(unittest.TestCase):
    def test_rejects_bad_values(self):
        bad = [
            dict(window_s=0), dict(max_compressed_read_ratio=1.5), dict(quality_slack=1.0),
            dict(quality_raise_factor=1.0), dict(quality_min_ms=0.0),
            dict(quality_min_ms=2.0, quality_max_ms=1.0), dict(tau_quantile=1.0),
            dict(tau_smoothing=0.0), dict(tau_min_s=10.0, tau_max_s=5.0),
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                TunerConfig(**kwargs)


class TestWindowing(unittest.TestCase):
    def test_steps_only_after_a_full_window(self):
        tuner, _, _, _, clock = make_tuner(TunerConfig(window_s=10.0, adapt_tau=False))
        self.assertIsNone(tuner.maybe_step())  # first call opens the first window
        clock.advance(9.9)
        self.assertIsNone(tuner.maybe_step())
        clock.advance(0.2)
        self.assertIsNotNone(tuner.maybe_step())
        self.assertIsNone(tuner.maybe_step())  # a new window has just started
        self.assertEqual(len(tuner.history), 1)

    def test_each_step_sees_only_its_own_window(self):
        tuner, _, _, signals, clock = make_tuner()
        tuner.maybe_step()
        feed(signals, accesses=100, compressed=40)
        clock.advance(10.0)
        first = tuner.maybe_step()
        feed(signals, accesses=60, compressed=0)
        clock.advance(10.0)
        second = tuner.maybe_step()
        self.assertEqual(first.stats.accesses, 100)
        self.assertEqual(second.stats.accesses, 60)
        self.assertEqual(second.stats.compressed_reads, 0)


class TestQualityBudget(unittest.TestCase):
    def setUp(self):
        self.cfg = TunerConfig(adapt_tau=False, adapt_watermarks=False, max_compressed_read_ratio=0.20)
        self.tuner, self.engine, _, self.signals, self.clock = make_tuner(self.cfg)

    def step(self, **kw):
        feed(self.signals, **kw)
        return self.tuner.step()

    def test_over_budget_raises_the_penalty(self):
        rec = self.step(accesses=100, compressed=40)
        self.assertAlmostEqual(self.engine.params.quality_penalty_ms, 0.5 * 1.5)
        self.assertAlmostEqual(rec.quality_penalty_ms, 0.75)
        self.assertTrue(any("over budget" in r for r in rec.reasons))

    def test_well_under_budget_lowers_it_slowly(self):
        self.step(accesses=100, compressed=0)
        self.assertAlmostEqual(self.engine.params.quality_penalty_ms, 0.5 / 1.15)

    def test_inside_the_dead_band_nothing_changes(self):
        rec = self.step(accesses=100, compressed=15)  # 0.15: between 0.5 x 0.20 and 0.20
        self.assertEqual(self.engine.params.quality_penalty_ms, 0.5)
        self.assertEqual(rec.reasons, ())

    def test_penalty_is_clamped_to_its_useful_range(self):
        # Range = [0.25 ms, recompute cost of a block = 10 ms on the toy profile].
        for _ in range(30):
            self.step(accesses=100, compressed=90)
        self.assertAlmostEqual(self.engine.params.quality_penalty_ms, 10.0)
        for _ in range(100):
            self.step(accesses=100, compressed=0)
        self.assertAlmostEqual(self.engine.params.quality_penalty_ms, 0.25)

    def test_explicit_upper_bound_is_respected(self):
        cfg = replace(self.cfg, quality_max_ms=1.0)
        tuner, engine, _, signals, _ = make_tuner(cfg)
        for _ in range(10):
            feed(signals, accesses=100, compressed=90)
            tuner.step()
        self.assertAlmostEqual(engine.params.quality_penalty_ms, 1.0)

    def test_windows_with_too_little_data_are_ignored(self):
        rec = self.step(accesses=10, compressed=10)
        self.assertEqual(self.engine.params.quality_penalty_ms, 0.5)
        self.assertEqual(rec.reasons, ())

    def test_disabled_mechanism_does_nothing(self):
        cfg = replace(self.cfg, adapt_quality=False)
        tuner, engine, _, signals, _ = make_tuner(cfg)
        feed(signals, accesses=100, compressed=90)
        tuner.step()
        self.assertEqual(engine.params.quality_penalty_ms, 0.5)


class TestWatermarks(unittest.TestCase):
    def setUp(self):
        self.cfg = TunerConfig(adapt_tau=False, adapt_quality=False)
        self.tuner, _, self.cc, self.signals, _ = make_tuner(self.cfg)

    def test_forced_evictions_lower_both_watermarks(self):
        feed(self.signals, accesses=0, forced=3)
        rec = self.tuner.step()
        self.assertAlmostEqual(self.cc.high_watermark, 0.83)
        self.assertAlmostEqual(self.cc.low_watermark, 0.68)
        self.assertTrue(any("forced evictions" in r for r in rec.reasons))

    def test_oom_also_counts_as_too_lazy(self):
        feed(self.signals, accesses=0, oom=1)
        self.tuner.step()
        self.assertAlmostEqual(self.cc.high_watermark, 0.83)

    def test_floors_and_minimum_gap_are_respected(self):
        for _ in range(100):
            feed(self.signals, accesses=0, forced=1)
            self.tuner.step()
        self.assertGreaterEqual(self.cc.high_watermark, self.cfg.high_floor - 1e-9)
        self.assertGreaterEqual(self.cc.low_watermark, self.cfg.low_floor - 1e-9)
        self.assertGreaterEqual(self.cc.high_watermark - self.cc.low_watermark, self.cfg.min_watermark_gap - 1e-9)

    def test_quiet_windows_relax_back_toward_home_but_not_past_it(self):
        feed(self.signals, accesses=0, forced=1)
        self.tuner.step()  # 0.83 / 0.68
        for _ in range(2):
            self.tuner.step()
        self.assertAlmostEqual(self.cc.high_watermark, 0.83)  # two quiet windows: not yet
        self.tuner.step()  # third quiet window
        self.assertAlmostEqual(self.cc.high_watermark, 0.84)
        for _ in range(50):
            self.tuner.step()
        self.assertAlmostEqual(self.cc.high_watermark, 0.85)  # never above the starting value
        self.assertAlmostEqual(self.cc.low_watermark, 0.70)

    def test_a_trouble_window_resets_the_quiet_counter(self):
        feed(self.signals, accesses=0, forced=1)
        self.tuner.step()
        self.tuner.step()
        self.tuner.step()
        feed(self.signals, accesses=0, forced=1)
        self.tuner.step()  # trouble again
        self.tuner.step()
        self.tuner.step()
        self.assertAlmostEqual(self.cc.high_watermark, 0.81)  # no relaxation in between


class TestRecencyTau(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(TunerConfig().adapt_tau)

    def test_moves_toward_the_gap_quantile_smoothly_and_within_bounds(self):
        cfg = TunerConfig(adapt_quality=False, adapt_watermarks=False, adapt_tau=True,
                          tau_quantile=0.5, tau_smoothing=0.5, min_gap_samples=10)
        tuner, engine, _, signals, _ = make_tuner(cfg)
        start = engine.recency_half_life_s  # 30
        for _ in range(20):
            signals.record_reuse_gap(100.0)
        tuner.step()
        self.assertAlmostEqual(engine.recency_half_life_s, 0.5 * start + 0.5 * 100.0)
        for _ in range(60):
            tuner.step()
        self.assertAlmostEqual(engine.recency_half_life_s, 100.0, places=3)

    def test_waits_for_enough_samples(self):
        cfg = TunerConfig(adapt_quality=False, adapt_watermarks=False, adapt_tau=True, min_gap_samples=10)
        tuner, engine, _, signals, _ = make_tuner(cfg)
        for _ in range(5):
            signals.record_reuse_gap(500.0)
        tuner.step()
        self.assertEqual(engine.recency_half_life_s, 30.0)

    def test_target_is_clamped(self):
        cfg = TunerConfig(adapt_quality=False, adapt_watermarks=False, adapt_tau=True,
                          tau_smoothing=1.0, min_gap_samples=5, tau_max_s=60.0)
        tuner, engine, _, signals, _ = make_tuner(cfg)
        for _ in range(10):
            signals.record_reuse_gap(10_000.0)
        tuner.step()
        self.assertEqual(engine.recency_half_life_s, 60.0)


class TestTunerInvariants(unittest.TestCase):
    def test_random_signal_sequences_never_break_any_bound(self):
        rng = random.Random(1234)
        cfg = TunerConfig(adapt_tau=True, min_gap_samples=5)
        tuner, engine, cc, signals, clock = make_tuner(cfg)
        qmax = engine.cost_model.recompute_ms_per_mb * engine.cost_model.block_mb
        for _ in range(300):
            feed(signals, accesses=rng.randint(0, 200), compressed=rng.randint(0, 200),
                 forced=rng.choice([0, 0, 0, 1, 5]), oom=rng.choice([0, 0, 2]))
            for _ in range(rng.randint(0, 5)):
                signals.record_reuse_gap(rng.expovariate(0.02))
            clock.advance(10.0)
            tuner.maybe_step()
            self.assertGreaterEqual(engine.params.quality_penalty_ms, cfg.quality_min_ms - 1e-9)
            self.assertLessEqual(engine.params.quality_penalty_ms, qmax + 1e-9)
            self.assertTrue(0.0 < cc.low_watermark < cc.high_watermark <= 1.0)
            self.assertGreaterEqual(cc.high_watermark - cc.low_watermark, cfg.min_watermark_gap - 1e-9)
            self.assertGreaterEqual(engine.recency_half_life_s, cfg.tau_min_s - 1e-9)
            self.assertLessEqual(engine.recency_half_life_s, cfg.tau_max_s + 1e-9)

    def test_identical_inputs_give_identical_histories(self):
        def trajectory():
            tuner, _, _, signals, clock = make_tuner()
            rng = random.Random(7)
            tuner.maybe_step()
            for _ in range(50):
                feed(signals, accesses=100, compressed=rng.randint(0, 60), forced=rng.choice([0, 1]))
                clock.advance(10.0)
                tuner.maybe_step()
            return [(r.quality_penalty_ms, r.high_watermark, r.reasons) for r in tuner.history]

        self.assertEqual(trajectory(), trajectory())


class TestLadderBaseline(unittest.TestCase):
    """Static compress-first ladder (LRU order): compress, then offload."""

    def _run(self, spec, gpu_mb=10.0, contexts=4, blocks=3):
        # contexts x blocks of 1 MB each (+1 grown block each) fill most of a 10 MB GPU.
        reqs = [
            *[make_req(i, 0.01 * i, ctx=i) for i in range(contexts)],
            *[make_req(100 + i, 5.0 + 0.01 * i, ctx=i) for i in range(contexts)],
        ]
        trace = make_trace(reqs, [blocks] * contexts, duration_s=8.0)
        cfg = toy_config(gpu_pool_mb=gpu_mb, cpu_pool_mb=100.0)
        return run(trace, spec, cfg)

    def test_compresses_before_offloading(self):
        sim, result = self._run(ladder_spec())
        self.assertGreater(result.metrics.compressions, 0)
        self.assertEqual(sim.check_invariants(), [])

    def test_zero_cap_never_compresses(self):
        sim, result = self._run(ladder_spec(max_compressed_fraction=0.0))
        self.assertEqual(result.metrics.compressions, 0)
        self.assertEqual(result.metrics.compressed_reads, 0)

    def test_rejects_invalid_cap(self):
        with self.assertRaises(ValueError):
            ladder_spec(max_compressed_fraction=1.5).build(None)  # validated before use

    def test_cap_limits_compressed_share(self):
        _, loose = self._run(ladder_spec(max_compressed_fraction=1.0))
        _, tight = self._run(ladder_spec(max_compressed_fraction=0.25))
        self.assertLessEqual(tight.metrics.compressions, loose.metrics.compressions)


def make_req(i, t, ctx=0):
    from src.trace import Request

    return Request(request_id=i, time_s=t, context_id=ctx, new_blocks=1, output_tokens=10)


class TestTunerInTheSimulator(unittest.TestCase):
    CFG = SimConfig(gpu_pool_mb=96.0, cpu_pool_mb=48.0, warmup_s=10.0)
    WORKLOAD = WorkloadConfig(name="s", duration_s=120.0, num_contexts=40, arrival_rate=4.0, seed=11)

    def run_spec(self, spec):
        sim = EventSimulator(generate_trace(self.WORKLOAD), self.CFG, spec)
        return sim, sim.run()

    def test_tuned_policy_leaves_consistent_bookkeeping_and_records_history(self):
        spec = ms_adaptive_spec("t", tuner_config=TunerConfig())
        sim, result = self.run_spec(spec)
        self.assertEqual(sim.check_invariants(), [])
        self.assertTrue(all(b.pin_count == 0 for b in sim.collector.all_blocks()))
        self.assertGreater(len(sim.policy.tuner.history), 5)
        self.assertGreater(result.summary["requests"], 0)

    def test_runs_are_deterministic(self):
        spec = ms_adaptive_spec("t", tuner_config=TunerConfig())
        self.assertEqual(self.run_spec(spec)[1].summary, self.run_spec(spec)[1].summary)

    def test_tuner_state_does_not_leak_between_runs_of_the_same_spec(self):
        params = MsParams(quality_penalty_ms=0.5)
        spec = ms_adaptive_spec("t", params=params, controller_config=ControllerConfig(),
                                tuner_config=TunerConfig(max_compressed_read_ratio=0.0))
        self.run_spec(spec)
        self.assertEqual(params.quality_penalty_ms, 0.5)  # the spec's template is untouched
        first_start = self.run_spec(spec)[0].policy.tuner.history[0]
        self.assertAlmostEqual(first_start.quality_penalty_ms, 0.5 * 1.5)  # fresh start each run

    def test_a_zero_budget_yields_less_compressed_exposure_than_a_full_budget(self):
        none = self.run_spec(ms_adaptive_spec("b0", tuner_config=TunerConfig(max_compressed_read_ratio=0.0)))[1]
        full = self.run_spec(ms_adaptive_spec("b1", tuner_config=TunerConfig(max_compressed_read_ratio=1.0)))[1]
        self.assertLessEqual(
            none.summary["compressed_read_ratio"], full.summary["compressed_read_ratio"]
        )


if __name__ == "__main__":
    unittest.main()
