# Adaptive Multi-Tier KV Cache Manager — Core Modules

Implementation status: foundation hardening (Step 1) and the event-driven
simulator (Step 2) are complete. The Adaptive Weight Tuner, cost formulas in a
common unit (milliseconds) and vLLM integration are still to come.

## Modules in this package

| Module | File | Status |
|---|---|---|
| Clock abstraction (reproducibility) | `clock.py` | ✅ Implemented + tested |
| Metadata Collector | `metadata_collector.py` | ✅ Implemented + tested |
| Memory Tracker (capacity-enforcing) | `memory_tracker.py` | ✅ Implemented + tested |
| Adaptive Importance Engine | `importance_engine.py` | ✅ Implemented + tested |
| Cost-Benefit Decision Engine | `decision_engine.py` | ✅ Implemented + tested |
| Tier Placement Manager | `placement.py` | ✅ Implemented + tested (single shared implementation) |
| Pressure-driven Controller (hysteresis) | `controller.py` | ✅ Implemented + tested |
| Synthetic workload generator/loader | `workload.py` | ✅ Implemented + tested |
| End-to-end synthetic demo | `simulator.py` | ✅ Implemented |
| Performance Monitor | `performance_monitor.py` | ✅ Implemented + tested |
| FIFO / LRU / MTDS-static baselines | `baselines.py` | ✅ Implemented + tested |
| Snapshot comparison harness (Review-1 demo) | `evaluate.py` | ✅ Implemented + tested |
| Hardware cost model (KV size, PCIe, recompute) | `hardware.py` | ✅ Implemented + tested |
| Request-trace generator (Poisson/bursty, Zipf) | `trace.py` | ✅ Implemented + tested |
| Event-driven simulator | `eventsim.py` | ✅ Implemented + tested |
| Metrics (TTFT percentiles, hit ratios, overheads) | `metrics.py` | ✅ Implemented + tested |
| Policy adapters (Ours, FIFO, LRU, LRU+offload, MTDS-style) | `policies.py` | ✅ Implemented + tested |
| Multi-workload, multi-seed experiment runner | `experiment.py` | ✅ Implemented |
| Adaptive Weight Tuner | — | ❌ Not yet implemented |
| Cost formulas in milliseconds | — | ❌ Not yet implemented |
| vLLM integration | — | ❌ Not yet implemented |

## How the pieces fit together

```
MetadataCollector          (per-block metadata: recency, frequency, size, tier)
        │
        ▼
MemoryTracker               (GPU/CPU capacity accounting, enforced limits, gpu_pressure())
        │
        ▼
PressureDrivenController    (when to act: high/low watermarks with hysteresis;
        │                    whom to act on: lowest importance-per-MB first)
        ▼
AdaptiveImportanceEngine    (5-factor score -> per-action benefit/cost)
        │
        ▼
DecisionEngine               (Net Utility = Benefit − Cost, picks best relief action)
        │
        ▼
TierPlacementManager         (executes MIGRATE / COMPRESS / EVICT with explicit fallbacks)
```

This is the architecture proposed in the Zeroth Review, implemented first as a
self-contained, vLLM-free simulation so the core decision logic can be
validated on its own before the higher-risk vLLM integration.

## Running it

```bash
# Run the full unit test suite (no external dependencies required)
python -m unittest discover -s tests -v

# Run the end-to-end synthetic demo (our engine only)
python -m src.simulator

# Snapshot comparison (old Review-1 harness): Ours vs FIFO vs LRU vs MTDS-static
python -m src.evaluate

# Event-driven experiment: 4 workloads x 5 policies x N seeds (about 1 minute per seed)
python -m src.experiment --seeds 3 --out results/event_sim_3seeds.json
python -m src.experiment --workloads zipf --seeds 5 --duration 200   # quicker
```

All runs are deterministic: the workload uses a fixed seed and a fixed
simulated clock, so the output is identical every time.

## What changed in the foundation-hardening step

| Problem found in the first version | Fix |
|---|---|
| Engine had no stopping rule and over-evicted (final GPU utilization ≈ 34% vs ≈ 70% for FIFO/LRU) | `PressureDrivenController`: start shedding at the high watermark (85%), stop at the low watermark (70%) |
| Result depended on the arbitrary order blocks were visited | Victims are ranked by importance per MB; ties broken by block id |
| Thresholds could flip-flop around one value | Hysteresis: a dead band between the two watermarks; optional per-step action limit |
| Memory tiers could be silently overfilled | `MemoryTracker` raises `CapacityError`; moves are atomic |
| MIGRATE with a full CPU tier was undefined | Explicit fallback to EVICT, recorded as `fallback_reason="cpu_full"` |
| A block could be compressed repeatedly | Compress-once guard (engine omits COMPRESS; placement manager no-ops it) |
| Simulator and harness each had their own (already diverging) action-execution code | One `TierPlacementManager` used by every policy |
| Runs depended on `time.time()` | Injectable `Clock`; `ManualClock` makes runs reproducible |
| Baselines were compared under different triggers/targets | Ours, FIFO and LRU share the same trigger and target |

## Event-driven simulator

Unlike the snapshot harness (one decision on a frozen set of blocks), the event
simulator replays a **policy-independent request trace** while time passes:

- **Workload.** A population of conversations; each request needs *all* of its
  conversation's existing KV blocks plus a few new ones. Which conversation is hit
  follows a Zipf law; arrivals are Poisson or bursty. The trace is generated once
  per (workload, seed) and replayed identically against every policy, which removes
  the old circularity (replayed accesses used to be derived from the same
  `access_count` our own score reads).
- **Physical costs.** A GPU hit is free. A CPU-resident block is swapped in over PCIe:
  `setup + size / bandwidth`, queued behind other swap-ins on the same link. An
  evicted block is recomputed: `tokens / prefill throughput`. Block size comes from
  the model's real architecture (`2 × layers × KV heads × head dim × bytes`).
- **Pinning.** A running request's blocks cannot be moved or evicted.
- **Fairness.** All policies share the same platform behaviour: forced LRU eviction
  when a policy frees too little, CPU-tier cleanup, and the "load vs recompute"
  choice for CPU blocks. Ours, FIFO, LRU and LRU+offload share the same trigger
  (85%) and target (70%).
- **Baselines.** FIFO and LRU (eviction only), **LRU+offload** (LRU order but moves
  victims to CPU: the fair tier-aware classical baseline), and **MTDS-style** (a
  simplified static-threshold re-implementation of the base paper's *idea*, not of
  the full MTDS system).

## Current results (3 seeds, 400 s traces, modelled T4 + ~1B model)

Reuse-request TTFT in ms (mean ± std over seeds) and share of block reads served
without recomputation. Raw numbers: `results/event_sim_3seeds.json`.

| Workload | Policy | TTFT P50 | TTFT P99 | GPU hit | Cache hit |
|---|---|---|---|---|---|
| zipf | Ours | 7.6 ± 0.2 | 161 ± 18 | 0.730 ± 0.021 | 0.982 |
| zipf | LRU+offload | 8.3 ± 0.5 | 166 ± 8 | 0.674 ± 0.031 | 0.982 |
| zipf | MTDS-style | 7.9 ± 0.3 | 187 ± 12 | 0.711 ± 0.027 | 0.975 |
| zipf | LRU | 68.7 ± 13.9 | 630 ± 64 | 0.675 ± 0.032 | 0.675 |
| zipf | FIFO | 123.8 ± 11.1 | 675 ± 15 | 0.546 ± 0.026 | 0.546 |
| long_context | Ours | 18.9 ± 1.6 | 933 ± 170 | 0.389 ± 0.012 | 0.890 |
| long_context | LRU+offload | 20.2 ± 1.8 | 976 ± 227 | 0.312 ± 0.015 | 0.891 |
| long_context | LRU | 406 ± 27 | 1367 | 0.317 ± 0.019 | 0.317 |

(`bursty` and `uniform` follow the same pattern; see the JSON.)

### What these results do and do not show

1. **Using the CPU tier is the big win.** Every policy that offloads instead of
   evicting cuts P50 TTFT by roughly 5-20x versus FIFO/LRU, because swapping a
   block in over PCIe (~0.05 ms) is far cheaper than recomputing it (~2.7 ms). This
   is a property of the hardware model, and it is exactly the effect the base paper
   exploits. It is **not** a contribution of our scoring.
2. **Our engine keeps more blocks on the GPU than LRU+offload.** The GPU hit ratio
   is higher in all 12 paired (workload, seed) comparisons, by 0.2 to 8 points, and
   it moves about 18% less data over PCIe on `zipf`. This is the genuine, modest
   advantage of importance-aware victim selection.
3. **That does not yet translate into a reliable latency win.** TTFT differences
   between Ours and LRU+offload have mixed signs across seeds and are within noise.
   We should not claim a latency improvement over LRU+offload from these results.
4. **`COMPRESS` is never chosen** (0 compressions in every run). With the current
   formulas compression never beats eviction or migration, so one of the four
   actions is effectively dead. The cost formulas mix unitless importance with
   size-scaled terms and hand-picked constants; re-expressing everything in
   milliseconds of expected delay (next step) should fix this, and the compression
   path then needs a quality-loss model.
5. **Weights are still static**: the Adaptive Weight Tuner is not built, so "adaptive"
   currently describes the scoring, not a feedback loop.

### Known simplifications (state these in the report)

- No request queueing, batching or GPU compute contention, so there is **no
  throughput metric** and TTFT excludes scheduling delay.
- Memory freed by MIGRATE becomes available immediately (real systems free it when
  the device-to-host copy finishes); offload time is tracked but kept off the request path.
- Recomputing a block costs `tokens / prefill throughput` regardless of position.
- All blocks are the same size; pools are scaled down (1 GB GPU / 4 GB CPU) so the
  *ratio* of working set to GPU memory creates pressure while runs stay fast.
- Bandwidth and throughput values are **assumed typical figures, not measurements**
  (`hardware.py`); calibrate them on real hardware during vLLM integration.
- The workload is a *model* of conversation reuse, not a replay of real traces.

### Earlier snapshot harness (`evaluate.py`)

Kept as a Review-1 demonstration of the decision logic on a frozen set of blocks.
Its latency model is flat (2 / 12 / 80 ms per tier) and its replayed accesses derive
from `access_count`, so its numbers should **not** be quoted as evidence; use the
event-driven results above instead.

## Design rationale (for the Review presentation)

- **Why a constrained, pressure-driven controller instead of independent
  per-block decisions:** a cache manager's real job is "keep utilization under a
  limit while losing as little future value as possible". The controller states
  that constraint explicitly; the Decision Engine then picks the cheapest way to
  shed each chosen block. Excluding KEEP in this phase is deliberate: the
  pressure constraint has already decided that memory must be freed.

- **Why value density (importance ÷ size) picks victims:** shedding the block
  with the least keep-worth per megabyte frees the most memory for the least
  value lost, the standard greedy rule for knapsack-style problems.

- **Why two watermarks:** with a single threshold, utilization hovering near it
  would trigger shedding, stop, and trigger again. The gap (hysteresis) makes
  shedding decisive and stable.

- **Why five factors feed one importance score, not five separate rules:**
  recency and frequency combine into a single bounded importance value; migration
  cost, compression benefit and GPU pressure then shape the benefit/cost of each
  *specific action*. This keeps the scoring interpretable.

- **Why costs are normalized against a reference block size instead of total
  GPU capacity:** a single block is a tiny fraction of GPU capacity, which would
  make the benefit of freeing it negligible next to a [0, 1] importance score
  and bias the engine toward keeping everything. (This normalization is one of
  the items to be revisited when costs move to a common time unit.)

- **Why `DecisionEngine` still accepts plain `{"benefit", "cost"}` dicts:**
  backward compatibility with the module as first committed; it also accepts
  `ActionEstimate` objects from `AdaptiveImportanceEngine`.
