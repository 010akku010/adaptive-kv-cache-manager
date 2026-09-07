# Adaptive Multi-Tier KV Cache Manager — Core Modules

Implementation status as of this commit (post-Zeroth Review, working toward
Review 1's 50–60% implementation target).

## Modules in this package

| Module | File | Status |
|---|---|---|
| Metadata Collector | `metadata_collector.py` | ✅ Implemented + tested |
| Memory Tracker | `memory_tracker.py` | ✅ Implemented + tested |
| Adaptive Importance Engine | `importance_engine.py` | ✅ Implemented + tested |
| Cost-Benefit Decision Engine | `decision_engine.py` | ✅ Implemented + tested (extended, backward-compatible with the original committed version) |
| Tier Placement Manager | `simulator.py` (`TierPlacementManager`) | ✅ Simulation-level implementation |
| End-to-end synthetic simulator | `simulator.py` | ✅ Implemented |
| Performance Monitor | — | ❌ Not yet implemented |
| vLLM integration | — | ❌ Not yet implemented |
| FIFO / LRU / MTDS baselines | — | ❌ Not yet implemented |

## How the pieces fit together

```
MetadataCollector          (per-block metadata: recency, frequency, size, tier)
        │
        ▼
MemoryTracker               (GPU/CPU capacity accounting, gpu_pressure())
        │
        ▼
AdaptiveImportanceEngine    (5-factor score -> per-action benefit/cost)
        │
        ▼
DecisionEngine               (Net Utility = Benefit − Cost, picks best action)
        │
        ▼
TierPlacementManager         (executes KEEP / MIGRATE / COMPRESS / EVICT)
```

This is exactly the architecture proposed in the Zeroth Review (Slide 8/10),
implemented first as a self-contained, vLLM-free simulation so the core
decision logic can be demonstrated and validated on its own before the
(higher-risk) vLLM integration step.

## Running it

```bash
# Run the full unit test suite (no external dependencies required)
python -m unittest discover -s tests -v

# Run the end-to-end synthetic demo
python -m src.simulator
```

The demo populates a synthetic GPU with randomized KV blocks until it hits
~85% utilization, then runs the full Metadata → Importance → Decision →
Placement pipeline on every GPU-resident block, printing each block's
importance score, chosen action, and resulting Net Utility — followed by the
before/after memory snapshot.

## Design rationale (for the Review 1 presentation)

- **Why five factors feed one importance score, not five separate rules:**
  recency and frequency combine into a single bounded "importance" value
  (how valuable a block is to keep instantly accessible); migration cost,
  compression benefit, and GPU pressure then shape the benefit/cost of each
  *specific action*, rather than all five being mashed into one undifferentiated
  number. This keeps the scoring interpretable and matches how the Zeroth
  Review frames the five factors as informing different parts of the decision.

- **Why costs are normalized against a reference block size instead of raw
  memory or total GPU capacity:** a single block's memory footprint is a tiny
  fraction of total GPU capacity, which would make "benefit of freeing this
  block" numerically negligible next to a [0, 1] importance score — biasing
  the engine toward always keeping everything, regardless of pressure. Scaling
  against a typical block size keeps benefit and cost on a comparable, unitless
  scale so GPU pressure can meaningfully tip the decision for large, cold
  blocks, which is the behavior the base paper's static-threshold approach
  cannot express.

- **Why `DecisionEngine` still accepts plain `{"benefit", "cost"}` dicts:**
  this preserves compatibility with the module as it was first committed and
  tested in the Zeroth-to-Review-1 transition, so existing tests/demos from
  that commit continue to work unchanged even as the engine now also accepts
  richer `ActionEstimate` objects from `AdaptiveImportanceEngine`.
