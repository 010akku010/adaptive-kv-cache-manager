"""Tests for the statistics used to compare policies."""

import math
import statistics
import unittest

from src.analysis import (
    format_frontier,
    format_report,
    frontier_points,
    interpolate,
    mean_metric,
    paired_difference,
    t_critical,
)


def row(policy, seed, ttft, exposure=0.0, workload="w", regime="r"):
    return {
        "regime": regime, "workload": workload, "policy": policy, "seed": seed,
        "summary": {"reuse_ttft_mean_ms": ttft, "compressed_read_ratio": exposure},
    }


class TestTCritical(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(t_critical(4), 2.776)  # 5 paired seeds
        self.assertAlmostEqual(t_critical(1), 12.706)

    def test_untabulated_df_uses_the_conservative_lower_entry(self):
        self.assertAlmostEqual(t_critical(11), t_critical(10))
        self.assertAlmostEqual(t_critical(14), t_critical(12))

    def test_large_df_approaches_normal(self):
        self.assertAlmostEqual(t_critical(500), 1.96)

    def test_needs_at_least_one_df(self):
        with self.assertRaises(ValueError):
            t_critical(0)


class TestPairedDifference(unittest.TestCase):
    def test_hand_computed_interval(self):
        a = [10.0, 12.0, 11.0, 13.0, 9.0]
        b = [12.0, 15.0, 12.0, 16.0, 13.0]
        rows = [row("A", s, v) for s, v in enumerate(a)] + [row("B", s, v) for s, v in enumerate(b)]
        diffs = [x - y for x, y in zip(a, b)]  # -2, -3, -1, -3, -4
        mean, half, n = paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "w", "r")
        self.assertEqual(n, 5)
        self.assertAlmostEqual(mean, -2.6)
        self.assertAlmostEqual(half, 2.776 * statistics.stdev(diffs) / math.sqrt(5))
        self.assertLess(abs(mean) - half, 0.0 + 1e-9 + abs(mean))  # sanity: finite interval

    def test_pairing_removes_trace_to_trace_variation(self):
        # Traces differ hugely (10 vs 100 ms) but A is always exactly 1 ms faster.
        base = [10.0, 40.0, 70.0, 100.0]
        rows = [row("A", s, v - 1.0) for s, v in enumerate(base)] + [row("B", s, v) for s, v in enumerate(base)]
        mean, half, _ = paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "w", "r")
        self.assertAlmostEqual(mean, -1.0)
        self.assertAlmostEqual(half, 0.0)  # no variance in the differences

    def test_only_common_seeds_are_paired(self):
        rows = [row("A", s, 1.0) for s in (0, 1, 2, 3)] + [row("B", s, 2.0) for s in (2, 3, 4)]
        _, _, n = paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "w", "r")
        self.assertEqual(n, 2)

    def test_filters_by_workload_and_regime(self):
        rows = (
            [row("A", s, 1.0, workload="x") for s in (0, 1)] + [row("B", s, 5.0, workload="x") for s in (0, 1)]
            + [row("A", s, 9.0, workload="y") for s in (0, 1)] + [row("B", s, 9.5, workload="y") for s in (0, 1)]
        )
        self.assertAlmostEqual(paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "x", "r")[0], -4.0)
        self.assertAlmostEqual(paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "y", "r")[0], -0.5)

    def test_too_few_pairs_is_an_error(self):
        rows = [row("A", 0, 1.0), row("B", 0, 2.0)]
        with self.assertRaises(ValueError):
            paired_difference(rows, "A", "B", "reuse_ttft_mean_ms", "w", "r")


class TestInterpolation(unittest.TestCase):
    def test_linear_between_points_and_exact_at_them(self):
        pts = [(0.0, 100.0), (0.5, 60.0), (1.0, 40.0)]
        self.assertAlmostEqual(interpolate(pts, 0.25), 80.0)
        self.assertAlmostEqual(interpolate(pts, 0.5), 60.0)
        self.assertAlmostEqual(interpolate(pts, 0.75), 50.0)

    def test_outside_the_sampled_range_is_none(self):
        pts = [(0.1, 1.0), (0.9, 2.0)]
        self.assertIsNone(interpolate(pts, 0.05))
        self.assertIsNone(interpolate(pts, 0.95))
        self.assertIsNone(interpolate([], 0.5))

    def test_input_order_does_not_matter(self):
        self.assertAlmostEqual(interpolate([(1.0, 0.0), (0.0, 10.0)], 0.5), 5.0)


class TestFrontier(unittest.TestCase):
    def rows(self):
        rows = []
        for s in range(3):
            rows += [
                row("LRU+offload", s, 100.0, 0.0),
                row("Ladder 0.5", s, 60.0, 0.5),
                row("Ladder 1", s, 40.0, 1.0),
                row("Ours-ms+tuner b=0.2", s, 70.0, 0.2),   # ladder at 0.2 would be 84
                row("Ours-ms+tuner b=0.9", s, 45.0, 0.9),   # ladder at 0.9 would be 44
                row("Ours-ms+tuner b=0.0", s, 100.0, -0.1),  # outside the ladder's range
            ]
        return rows

    def test_mean_metric_and_points(self):
        rows = self.rows()
        self.assertAlmostEqual(mean_metric(rows, "Ladder 0.5", "reuse_ttft_mean_ms", "w", "r"), 60.0)
        pts = frontier_points(rows, ["LRU+offload", "Ladder 0.5", "Ladder 1", "missing"], "w", "r")
        self.assertEqual(sorted(pts), [(0.0, 100.0), (0.5, 60.0), (1.0, 40.0)])

    def test_gap_against_the_ladder_at_equal_exposure(self):
        lines = format_frontier(self.rows(), "w", "r")
        b02 = next(line for line in lines if "b=0.2" in line)
        self.assertIn("ladder at same exposure    84.0", b02)
        self.assertIn("-14.0", b02)  # ours is 14 ms faster
        b09 = next(line for line in lines if "b=0.9" in line)
        self.assertIn("+1.0", b09)  # ours is 1 ms slower than the ladder at 0.9
        self.assertIn("outside the ladder's range", next(line for line in lines if "b=0.0" in line))

    def test_report_runs_on_realistic_rows(self):
        rows = []
        for s in range(3):
            for name, v in (("Ours-ms", 50.0 + s), ("LRU+offload", 80.0 + s), ("MTDS-style", 70.0), ("Ladder 0.25", 75.0)):
                rows.append(row(name, s, v))
        text = format_report(rows)
        self.assertIn("Ours-ms", text)
        self.assertIn("*", text)  # a clear win is flagged as significant


if __name__ == "__main__":
    unittest.main()
