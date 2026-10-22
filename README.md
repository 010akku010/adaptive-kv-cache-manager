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
python -m src.evaluate                    # compare Ours vs FIFO vs LRU vs MTDS-static
```

Requires Python 3.9+. Runs are deterministic.

## Status

Core decision pipeline, pressure-driven controller, baselines and comparison
harness are implemented and tested on a synthetic workload. Still in progress:
event-driven simulator, Adaptive Weight Tuner, evaluation on realistic traces and
vLLM integration. See [`src/README.md`](src/README.md) for the module map,
current results and their known limitations.
