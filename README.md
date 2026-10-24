# Adaptive Multi-Tier KV Cache Manager

A cost-benefit approach to managing the KV cache of LLM serving systems across
GPU memory, CPU memory and compressed storage.

When an LLM generates text it keeps a **KV cache** of everything processed so
far. That cache is large and GPU memory is scarce, so when the GPU fills up a
server must decide, per block of cache, whether to **keep** it, **migrate** it to
CPU memory, **compress** it, or **evict** it (and recompute it later if needed).
Static-threshold systems apply one fixed rule to every block. This project
instead scores each block and chooses the action with the best **Net Utility =
Benefit − Cost**, using recency, frequency, migration cost, compression benefit
and GPU memory pressure.

## Quick start

```bash
python -m unittest discover -s tests -v   # run the tests (no dependencies needed)
python -m src.simulator                   # end-to-end demo of our engine
python -m src.evaluate                    # snapshot comparison (Review-1 demo)
python -m src.experiment --seeds 3        # event-driven experiment, all workloads and policies
python -m src.analysis results/full_host_starved.json   # paired statistics on saved results
```

Requires Python 3.10+. Runs are deterministic.

## Status

Core decision pipeline, pressure-driven controller, baselines and an event-driven
simulator (request traces, PCIe/recompute cost model, TTFT percentiles), a cost model
in milliseconds and an Adaptive Weight Tuner (quality-budget feedback loop) are
implemented, tested and evaluated on held-out seeds with paired statistics. Still in
progress: replay of real traces, a quality (accuracy) measurement and vLLM integration. See [`src/README.md`](src/README.md) for the module map,
current results and their known limitations.
