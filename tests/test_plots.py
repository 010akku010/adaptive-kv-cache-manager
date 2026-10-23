"""Smoke tests for the figure code (skipped when matplotlib is not installed)."""

import os
import tempfile
import unittest

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

from src.plots import plot_frontier


def row(policy, seed, ttft, exposure, workload="zipf", regime="host-starved"):
    return {
        "regime": regime, "workload": workload, "policy": policy, "seed": seed,
        "summary": {"reuse_ttft_mean_ms": ttft, "compressed_read_ratio": exposure},
    }


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class TestFrontierPlot(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        rows = []
        for s in range(3):
            rows += [
                row("LRU+offload", s, 90.0 + s, 0.0), row("Ladder 0.5", s, 65.0 + s, 0.45),
                row("Ladder 1", s, 45.0 + s, 0.87), row("Ours-ms+tuner b=0.2", s, 66.0 + s, 0.28),
                row("Ours-ms+tuner b=1", s, 52.0 + s, 0.67),
            ]
        with tempfile.TemporaryDirectory() as d:
            path = plot_frontier(rows, "host-starved", os.path.join(d, "nested", "f.png"))
            self.assertTrue(os.path.getsize(path) > 1000)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
