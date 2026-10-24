import contextlib
import io
import json
import unittest

from src.demo import DemoSetup, baseline_spec, format_frame, main, ours_spec, record, record_pair, replay

SHORT = DemoSetup(workload="uniform", seed=1, duration_s=90.0, warmup_s=20.0)


class TestRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = record_pair(SHORT)

    def test_records_both_policies_on_the_same_trace(self):
        names = [r["policy"] for r in self.data["runs"]]
        self.assertEqual(names, ["LRU+offload", "Ours"])
        a, b = (r["frames"] for r in self.data["runs"])
        self.assertEqual([f["t"] for f in a], [f["t"] for f in b])

    def test_frames_are_one_per_second_and_physically_consistent(self):
        for run in self.data["runs"]:
            frames = run["frames"]
            self.assertGreaterEqual(len(frames), 88)
            times = [f["t"] for f in frames]
            self.assertEqual(times, sorted(times))
            for f in frames:
                self.assertLessEqual(f["gpu_used_mb"], run["gpu_capacity_mb"] + 1e-6)
                self.assertLessEqual(f["cpu_used_mb"], run["cpu_capacity_mb"] + 1e-6)
                self.assertLessEqual(f["gpu_compressed"], f["gpu_blocks"])
                self.assertLessEqual(f["cpu_compressed"], f["cpu_blocks"])

    def test_cumulative_counters_never_decrease(self):
        for run in self.data["runs"]:
            for key in ("migrations", "compressions", "evictions", "requests"):
                values = [f[key] for f in run["frames"]]
                self.assertEqual(values, sorted(values), key)

    def test_baseline_never_compresses_and_ours_does(self):
        base, ours = self.data["runs"]
        self.assertEqual(base["action_totals"]["COMPRESS"], 0)
        self.assertGreater(ours["action_totals"]["COMPRESS"], 0)

    def test_action_log_is_capped_per_second(self):
        for run in self.data["runs"]:
            per_second = {}
            for a in run["actions"]:
                per_second[int(a["t"])] = per_second.get(int(a["t"]), 0) + 1
            self.assertLessEqual(max(per_second.values(), default=0), 6)

    def test_recording_is_json_serialisable_and_deterministic(self):
        text = json.dumps(self.data)
        self.assertEqual(json.loads(text)["runs"][1]["policy"], "Ours")
        again = record(ours_spec(), SHORT)
        self.assertEqual(again["summary"], self.data["runs"][1]["summary"])

    def test_ours_is_not_slower_than_the_baseline_on_uniform(self):
        base, ours = (r["summary"]["reuse_ttft_mean_ms"] for r in self.data["runs"])
        self.assertLess(ours, base)

    def test_ttft_is_reported_in_every_frame_that_saw_requests(self):
        frames = self.data["runs"][0]["frames"]
        measured = [f for f in frames if f["t"] > 22]
        self.assertTrue(any(f["ttft_recent_ms"] is not None for f in measured))
        self.assertGreater(sum(f["ttft_recent_ms"] is not None for f in measured), len(measured) // 2)


class TestReplay(unittest.TestCase):
    def test_replay_prints_a_result_table_and_the_quality_caveat(self):
        data = record_pair(SHORT)
        out = io.StringIO()
        replay(data, speed=0, out=out)
        text = out.getvalue()
        self.assertIn("=== Result", text)
        self.assertIn("Mean TTFT change vs LRU+offload", text)
        self.assertIn("accuracy cost is not measured", text)

    def test_format_frame_shows_both_tiers(self):
        data = record_pair(SHORT)
        run = data["runs"][1]
        line = format_frame(run, run["frames"][-1])
        self.assertIn("GPU", line)
        self.assertIn("CPU", line)

    def test_cli_main_writes_json(self):
        import os
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "demo.json")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(["--duration", "30", "--speed", "0", "--json", path])
        with open(path) as fh:
            self.assertEqual(len(json.load(fh)["runs"]), 2)


class TestSpecs(unittest.TestCase):
    def test_spec_names(self):
        self.assertEqual(baseline_spec().name, "LRU+offload")
        self.assertEqual(ours_spec().name, "Ours")


if __name__ == "__main__":
    unittest.main()
