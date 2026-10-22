# Adaptive Multi-Tier KV Cache Manager — Core Modules

Implementation status: foundation hardening (Step 1) complete; event-driven
simulator, Adaptive Weight Tuner and vLLM integration are still to come.

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
| Comparison harness | `evaluate.py` | ✅ Implemented + tested |
| Adaptive Weight Tuner | — | ❌ Not yet implemented |
| Event-driven (time-evolving) simulator | — | ❌ Not yet implemented |
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

# Run the comparison harness: Ours vs FIFO vs LRU vs MTDS-static
python -m src.evaluate
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

## Current results and how far to trust them

```
POLICY            HIT RATIO   AVG(ms)   P99(ms)   GPU UTIL
Ours (Adaptive)       0.908      9.14     80.00      0.691
FIFO                  0.795     18.00     80.00      0.699
LRU                   0.828     15.43     80.00      0.699
MTDS-static           1.000     12.00     12.00      0.000
```

These numbers show the *mechanics work* (all policies now shed to the same
target and the engine no longer over-evicts). They are **not** yet evidence for
the project's claims, for reasons we know and plan to fix:

1. **Circular workload.** The replayed requests are derived from each block's
   `access_count`, the same quantity our importance score uses, so our engine is
   partly graded on its own inputs. An independent, time-evolving request stream
   is required (event-driven simulator milestone).
2. **Flat latency model.** Latency is a constant per tier (2 / 12 / 80 ms) and
   ignores block size and migration cost, so MIGRATE looks almost free. This is
   why MTDS-static scores a "perfect" hit ratio: it moves every block to an
   unbounded-in-practice CPU tier. Realistic transfer and recompute costs will
   remove this artifact.
3. **Our engine currently chooses EVICT for every victim.** With the present
   cost formulas, MIGRATE never beats EVICT for cold blocks. The formulas mix
   unitless importance with size-scaled terms and hand-picked constants; they
   should be re-expressed in one currency (expected milliseconds of delay).
   The Adaptive Weight Tuner is also not built yet, so weights are still static.
4. **Single snapshot.** One decision cycle on one set of blocks, one seed.
   Final claims need multiple workloads, many seeds and mean ± std reporting.

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
