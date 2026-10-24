# Test Report

Generated 2026-10-24 by `scripts/make_test_report.py`. Python 3.13.16, Linux 6.18.44-fc-v153, 2 CPU cores. All runs are deterministic (fixed seeds).

## Summary

**218 tests, 218 passed, 0 failed, 0 skipped, in 102 s.**

| Category | What it checks | Tests | Passed | Failed | Skipped | Time (s) |
|---|---|---:|---:|---:|---:|---:|
| Unit | One module in isolation, with hand-computed expected values. | 145 | 145 | 0 | 0 | 34.9 |
| Integration | Several modules wired together: controller + tracker + placement, the event simulator, the evaluation harness. | 48 | 48 | 0 | 0 | 17.7 |
| Functional | The user-facing demo: recording, replay and export behave as documented. | 12 | 12 | 0 | 0 | 2.9 |
| Stress | Heavy load, tiny memory pools and bursty traffic; the simulator's invariants must still hold. | 5 | 5 | 0 | 0 | 34.8 |
| Performance | Wall-clock budgets that catch a quadratic slow-down. | 2 | 2 | 0 | 0 | 1.1 |
| Acceptance | The project's headline claims (AC1-AC6) checked end to end. | 6 | 6 | 0 | 0 | 10.2 |

## By test module

| Module | Category | Tests | Passed | Time (s) |
|---|---|---:|---:|---:|
| `test_acceptance` | Acceptance | 6 | 6 | 10.2 |
| `test_analysis` | Unit | 16 | 16 | 0.0 |
| `test_clock` | Unit | 7 | 7 | 0.0 |
| `test_controller` | Integration | 16 | 16 | 0.0 |
| `test_core_modules` | Unit | 26 | 26 | 0.0 |
| `test_costmodel` | Unit | 24 | 24 | 0.0 |
| `test_demo` | Functional | 12 | 12 | 2.9 |
| `test_evaluate_harness` | Integration | 12 | 12 | 0.0 |
| `test_eventsim` | Integration | 20 | 20 | 17.6 |
| `test_memory_capacity` | Unit | 7 | 7 | 0.0 |
| `test_placement` | Unit | 9 | 9 | 0.0 |
| `test_plots` | Unit | 1 | 1 | 0.7 |
| `test_stress` | Stress + Performance | 7 | 7 | 35.9 |
| `test_trace_hardware_metrics` | Unit | 22 | 22 | 0.3 |
| `test_tuner` | Unit | 33 | 33 | 33.8 |

## Acceptance criteria

| ID | Requirement | Result |
|---|---|---|
| AC1 | Using the host tier beats eviction-only policies (FIFO, LRU) by more than 3x on zipf traffic | pass |
| AC2 | The adaptive engine beats LRU+offload on zipf traffic when host memory is scarce (2 of 2 seeds) | pass |
| AC3 | With plentiful host memory the engine is at least 15% faster than LRU+offload on uniform traffic | pass |
| AC4 | Tail latency (P99 TTFT) improves, not only the mean | pass |
| AC5 | A smaller quality budget leads to less compressed reading | pass |
| AC6 | No request fails for lack of memory (zero OOM events) | pass |

## How the tests were kept honest

- Expected values in cost-model, tuner and simulator tests are computed by hand on a toy hardware profile (1 MB blocks, 1 ms transfers, 10 ms recompute) so a reader can check the arithmetic.
- Each bug found during development has a regression test. The regression test for phantom out-of-memory events (which made the tuner collapse its watermarks) was confirmed by re-introducing the bug and watching it fail.
- The simulator checks its own invariants after stress runs: tracked memory equals the sum of block sizes in each tier, no tier exceeds capacity, and no block stays pinned.
- Acceptance tests use short traces (80-150 s, 1-2 seeds) to stay fast. The full evidence (5 held-out seeds, 400 s, paired 95% intervals) is in `results/analysis.txt`.

## Not covered

- No accuracy or quality measurement: the simulator has no model of compression error.
- No test against a real inference engine or real GPU (planned for the integration phase).
- Wall-clock budgets are generous so they do not fail on slow CI machines; they are not benchmarks.
