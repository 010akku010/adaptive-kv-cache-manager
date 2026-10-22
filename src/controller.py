"""
controller.py

The pressure-driven control loop that sits on top of the Adaptive
Importance Engine and the Cost-Benefit Decision Engine.

Problem it solves
-----------------
The first version of the pipeline asked, for every block independently,
"which action has the best Net Utility?" and applied the answer. That has
two flaws:

1. **No stopping rule.** Nothing says "enough memory has been freed". The
   engine kept evicting after pressure had already eased, finishing at
   ~34% GPU utilization while FIFO/LRU settled near their 70% target. That
   wastes cache and makes the comparison unfair.
2. **Order dependence.** Because pressure is re-read after each block, the
   outcome depended on the arbitrary order blocks were visited in.

How this controller works
-------------------------
A cache manager is really solving a *constrained* problem: keep GPU
utilization below a limit while losing as little future value as
possible. The controller turns the per-block cost-benefit logic into that:

1. **Trigger with hysteresis.** Shedding starts when utilization reaches
   ``high_watermark`` and continues until it falls to ``low_watermark``.
   The gap between the two prevents flip-flopping (start/stop/start) when
   utilization hovers around a single threshold.
2. **Choose victims by value density.** Blocks are ranked by
   ``importance / size`` -- how much "keep-worth" each megabyte holds.
   Shedding the lowest-density block first frees the most memory for the
   least value lost (the classic greedy rule for knapsack-style problems).
3. **Choose the action by Net Utility.** For the selected victim the
   Decision Engine compares MIGRATE / COMPRESS / EVICT (KEEP is excluded,
   because the pressure constraint has already decided that this block
   must give up GPU memory). Hot blocks tend to be migrated or compressed
   (cheap to recover); cold blocks tend to be evicted.
4. **Stop at the low watermark**, re-reading real utilization after every
   executed action.

Everything is deterministic: ties are broken by block id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine
from src.memory_tracker import MemoryTracker
from src.metadata_collector import KVBlock, MetadataCollector, Tier
from src.placement import PlacementResult, TierPlacementManager

_MIN_SIZE_MB = 1e-6  # avoids division by zero for fully compressed blocks


@dataclass
class ControllerConfig:
    """Tunable thresholds for the pressure-driven controller."""

    high_watermark: float = 0.85
    low_watermark: float = 0.70
    # Optional rate limit: at most this many actions per ``step()`` call.
    # When set, shedding can span several steps, and the hysteresis state
    # keeps the controller shedding until it reaches ``low_watermark``.
    max_actions_per_step: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.low_watermark < self.high_watermark <= 1.0):
            raise ValueError(
                "watermarks must satisfy 0 < low_watermark < high_watermark <= 1 "
                f"(got low={self.low_watermark}, high={self.high_watermark})"
            )
        if self.max_actions_per_step is not None and self.max_actions_per_step < 1:
            raise ValueError("max_actions_per_step must be >= 1 when set")


@dataclass(frozen=True)
class ControlDecision:
    """One action taken by the controller, with the context it was taken in."""

    block_id: str
    requested_action: str
    executed_action: str
    net_utility: float
    importance: float
    pressure_before: float
    pressure_after: float
    freed_gpu_mb: float
    fallback_reason: Optional[str] = None


class PressureDrivenController:
    """Sheds GPU memory pressure using cost-benefit decisions, with hysteresis."""

    def __init__(
        self,
        collector: MetadataCollector,
        tracker: MemoryTracker,
        importance_engine: AdaptiveImportanceEngine,
        decision_engine: DecisionEngine,
        placement: TierPlacementManager,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.collector = collector
        self.tracker = tracker
        self.importance_engine = importance_engine
        self.decision_engine = decision_engine
        self.placement = placement
        self.config = config if config is not None else ControllerConfig()
        self._shedding = False

    @property
    def shedding(self) -> bool:
        """True while the controller is actively relieving pressure."""
        return self._shedding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> List[ControlDecision]:
        """
        Run one control cycle and return the actions taken (empty if the
        GPU is not under enough pressure to act).
        """
        cfg = self.config
        pressure = self.tracker.gpu_pressure()

        # Hysteresis state machine.
        if not self._shedding and pressure >= cfg.high_watermark:
            self._shedding = True
        elif self._shedding and pressure <= cfg.low_watermark:
            self._shedding = False

        if not self._shedding:
            return []

        decisions: List[ControlDecision] = []
        no_progress: Set[str] = set()  # blocks whose action was a no-op
        gpu_block_count = sum(1 for _ in self.collector.blocks_in_tier(Tier.GPU))
        # Every block can be acted on at most twice (compress, then
        # migrate/evict), so this bound guarantees termination.
        max_iterations = 2 * gpu_block_count + 1

        for _ in range(max_iterations):
            if self.tracker.gpu_pressure() <= cfg.low_watermark:
                break
            if cfg.max_actions_per_step is not None and len(decisions) >= cfg.max_actions_per_step:
                break

            candidates = [
                b for b in self.collector.blocks_in_tier(Tier.GPU)
                if b.block_id not in no_progress
            ]
            if not candidates:
                break

            victim = min(candidates, key=self._victim_key)
            decision = self._relieve(victim)
            decisions.append(decision)
            if decision.executed_action == "KEEP":
                no_progress.add(victim.block_id)

        if self.tracker.gpu_pressure() <= cfg.low_watermark:
            self._shedding = False
        return decisions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _victim_key(self, block: KVBlock):
        """Lowest importance-per-MB first; block id breaks ties deterministically."""
        density = self.importance_engine.compute_importance(block) / max(block.size_mb, _MIN_SIZE_MB)
        return (density, block.block_id)

    def _relieve(self, block: KVBlock) -> ControlDecision:
        pressure_before = self.tracker.gpu_pressure()
        values = self.importance_engine.compute_action_values(block, pressure_before)
        values.pop("KEEP", None)  # the pressure constraint already rules KEEP out
        action, utility = self.decision_engine.decide(values)
        importance = self.importance_engine.compute_importance(block)

        result: PlacementResult = self.placement.apply(block.block_id, action)
        return ControlDecision(
            block_id=block.block_id,
            requested_action=result.requested_action,
            executed_action=result.executed_action,
            net_utility=utility,
            importance=importance,
            pressure_before=pressure_before,
            pressure_after=self.tracker.gpu_pressure(),
            freed_gpu_mb=result.freed_gpu_mb,
            fallback_reason=result.fallback_reason,
        )
