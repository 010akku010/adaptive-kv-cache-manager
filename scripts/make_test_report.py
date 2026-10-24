"""
Run the whole test suite and write docs/TEST_REPORT.md.

Every test is assigned to one of the categories the Review 2 guidelines ask for (unit, integration,
functional, stress, performance, acceptance) by the table below, so the report shows evidence per category.

Usage:
    python scripts/make_test_report.py            # runs everything (about 2 minutes)
"""

from __future__ import annotations

import os
import platform
import sys
import time
import unittest
from collections import OrderedDict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CATEGORY_OF_MODULE = {
    "test_clock": "Unit",
    "test_core_modules": "Unit",
    "test_costmodel": "Unit",
    "test_tuner": "Unit",
    "test_placement": "Unit",
    "test_memory_capacity": "Unit",
    "test_analysis": "Unit",
    "test_plots": "Unit",
    "test_trace_hardware_metrics": "Unit",
    "test_controller": "Integration",
    "test_evaluate_harness": "Integration",
    "test_eventsim": "Integration",
    "test_demo": "Functional",
    "test_acceptance": "Acceptance",
}
CATEGORY_ORDER = ["Unit", "Integration", "Functional", "Stress", "Performance", "Acceptance"]
DESCRIPTION = {
    "Unit": "One module in isolation, with hand-computed expected values.",
    "Integration": "Several modules wired together: controller + tracker + placement, the event simulator, the evaluation harness.",
    "Functional": "The user-facing demo: recording, replay and export behave as documented.",
    "Stress": "Heavy load, tiny memory pools and bursty traffic; the simulator's invariants must still hold.",
    "Performance": "Wall-clock budgets that catch a quadratic slow-down.",
    "Acceptance": "The project's headline claims (AC1-AC6) checked end to end.",
}


def category(test_id: str) -> str:
    module = test_id.split(".")[1] if test_id.startswith("tests.") else test_id.split(".")[0]
    if module == "test_stress":
        return "Performance" if "TestPerformance" in test_id else "Stress"
    return CATEGORY_OF_MODULE.get(module, "Unit")


class Collector(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows: list[tuple[str, str, float]] = []
        self._t0 = 0.0

    def startTest(self, test):
        self._t0 = time.perf_counter()
        super().startTest(test)

    def _add(self, test, outcome):
        self.rows.append((test.id(), outcome, time.perf_counter() - self._t0))

    def addSuccess(self, test):
        super().addSuccess(test)
        self._add(test, "pass")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._add(test, "FAIL")

    def addError(self, test, err):
        super().addError(test, err)
        self._add(test, "ERROR")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._add(test, "skip")


def main() -> int:
    suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    runner = unittest.TextTestRunner(resultclass=Collector, verbosity=0, stream=open(os.devnull, "w"))
    start = time.perf_counter()
    result = runner.run(suite)
    wall = time.perf_counter() - start

    per_cat: dict[str, list[tuple[str, str, float]]] = OrderedDict((c, []) for c in CATEGORY_ORDER)
    per_module: dict[str, list[tuple[str, str, float]]] = {}
    for row in result.rows:
        per_cat.setdefault(category(row[0]), []).append(row)
        per_module.setdefault(row[0].split(".")[1], []).append(row)

    def counts(rows):
        return (
            len(rows),
            sum(r[1] == "pass" for r in rows),
            sum(r[1] in ("FAIL", "ERROR") for r in rows),
            sum(r[1] == "skip" for r in rows),
            sum(r[2] for r in rows),
        )

    lines = [
        "# Test Report",
        "",
        f"Generated {date.today().isoformat()} by `scripts/make_test_report.py`. "
        f"Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{os.cpu_count()} CPU cores. All runs are deterministic (fixed seeds).",
        "",
        "## Summary",
        "",
        f"**{len(result.rows)} tests, {counts(result.rows)[1]} passed, {counts(result.rows)[2]} failed, "
        f"{counts(result.rows)[3]} skipped, in {wall:.0f} s.**",
        "",
        "| Category | What it checks | Tests | Passed | Failed | Skipped | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cat, rows in per_cat.items():
        n, p, f, s, t = counts(rows)
        lines.append(f"| {cat} | {DESCRIPTION[cat]} | {n} | {p} | {f} | {s} | {t:.1f} |")

    lines += ["", "## By test module", "", "| Module | Category | Tests | Passed | Time (s) |", "|---|---|---:|---:|---:|"]
    for module, rows in sorted(per_module.items()):
        n, p, _, _, t = counts(rows)
        cats = sorted({category(r[0]) for r in rows}, key=CATEGORY_ORDER.index)
        lines.append(f"| `{module}` | {' + '.join(cats)} | {n} | {p} | {t:.1f} |")

    lines += [
        "",
        "## Acceptance criteria",
        "",
        "| ID | Requirement | Result |",
        "|---|---|---|",
    ]
    ac = {
        "AC1": "Using the host tier beats eviction-only policies (FIFO, LRU) by more than 3x on zipf traffic",
        "AC2": "The adaptive engine beats LRU+offload on zipf traffic when host memory is scarce (2 of 2 seeds)",
        "AC3": "With plentiful host memory the engine is at least 15% faster than LRU+offload on uniform traffic",
        "AC4": "Tail latency (P99 TTFT) improves, not only the mean",
        "AC5": "A smaller quality budget leads to less compressed reading",
        "AC6": "No request fails for lack of memory (zero OOM events)",
    }
    outcomes = {r[0].split("test_")[-1][:3]: r[1] for r in result.rows if "test_acceptance" in r[0]}
    for key, text in ac.items():
        lines.append(f"| {key} | {text} | {outcomes.get(key, 'not run')} |")

    lines += [
        "",
        "## How the tests were kept honest",
        "",
        "- Expected values in cost-model, tuner and simulator tests are computed by hand on a toy hardware "
        "profile (1 MB blocks, 1 ms transfers, 10 ms recompute) so a reader can check the arithmetic.",
        "- Each bug found during development has a regression test. The regression test for phantom out-of-memory "
        "events (which made the tuner collapse its watermarks) was confirmed by re-introducing the bug and watching it fail.",
        "- The simulator checks its own invariants after stress runs: tracked memory equals the sum of block sizes "
        "in each tier, no tier exceeds capacity, and no block stays pinned.",
        "- Acceptance tests use short traces (80-150 s, 1-2 seeds) to stay fast. The full evidence (5 held-out "
        "seeds, 400 s, paired 95% intervals) is in `results/analysis.txt`.",
        "",
        "## Not covered",
        "",
        "- No accuracy or quality measurement: the simulator has no model of compression error.",
        "- No test against a real inference engine or real GPU (planned for the integration phase).",
        "- Wall-clock budgets are generous so they do not fail on slow CI machines; they are not benchmarks.",
        "",
    ]
    failed = [r for r in result.rows if r[1] in ("FAIL", "ERROR")]
    if failed:
        lines += ["## Failures", ""] + [f"- `{r[0]}`" for r in failed] + [""]

    out = os.path.join(ROOT, "docs", "TEST_REPORT.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"{len(result.rows)} tests, {len(failed)} failed; wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
