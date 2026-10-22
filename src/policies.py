"""
policies.py

Adapters that let every cache-management policy -- ours and the baselines
-- plug into the event-driven simulator through one tiny interface:

    policy.relieve() -> List[PlacementResult]

The simulator calls ``relieve()`` periodically (every tick) and whenever an
allocation does not fit. The policy inspects the shared bookkeeping
modules, executes whatever actions it wants through the shared
``TierPlacementManager`` and returns what it did.

Policies provided
-----------------
* ``Ours (Adaptive)``  -- the cost-benefit pipeline: pressure-driven
  controller + importance engine + decision engine.
* ``FIFO`` / ``LRU``   -- classical queue-ordered *eviction only* (the
  original baselines from the Zeroth Review).
* ``LRU+offload``      -- LRU victim order but MIGRATE (offload to CPU)
  instead of EVICT. This is the fair "classical but tier-aware" baseline:
  without it our engine would be compared against policies that are not
  even allowed to use the CPU tier.
* ``MTDS-style``       -- a simplified, static-threshold policy following
  the structure of the base paper's hierarchical eviction (two capacity
  levels, each with a stricter per-block utilization threshold). It is a
  re-implementation of the *idea*, not of the full MTDS system -- see the
  class docstring for exactly what is and is not modelled.

Policy-independent platform behaviour (CPU-tier cleanup, forced eviction
when a policy fails to free enough, load-vs-recompute choice) lives in the
simulator so that every policy gets identical help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from src.baselines import BaselinePolicy, FIFOPolicy, LRUPolicy
from src.clock import Clock
from src.controller import ControllerConfig, PressureDrivenController
from src.decision_engine import DecisionEngine
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.placement import PlacementResult, TierPlacementManager


@dataclass
class PolicyContext:
    """Everything a policy may look at or act on."""

    collector: MetadataCollector
    tracker: MemoryTracker
    placement: TierPlacementManager
    clock: Clock
    block_mb: float  # standard block size (MB)
    requests_seen: Callable[[], int]  # requests processed so far


class CachePolicy(Protocol):
    name: str

    def relieve(self) -> List[PlacementResult]:
        """Free GPU memory if the policy decides it should; return actions taken."""
        ...


@dataclass(frozen=True)
class PolicySpec:
    """A named recipe for building a policy once the simulator's context exists."""

    name: str
    build: Callable[[PolicyContext], CachePolicy]


# ---------------------------------------------------------------------------
# Ours
# ---------------------------------------------------------------------------

class AdaptivePolicy:
    """The project's cost-benefit engine behind the pressure-driven controller."""

    def __init__(
        self,
        ctx: PolicyContext,
        weights: Optional[ImportanceWeights] = None,
        controller_config: Optional[ControllerConfig] = None,
        name: str = "Ours (Adaptive)",
    ) -> None:
        self.name = name
        # The importance engine normalises sizes against a "reference block";
        # in the simulator that reference is the real block size.
        engine = AdaptiveImportanceEngine(
            weights=weights or ImportanceWeights(),
            clock=ctx.clock,
            reference_block_mb=ctx.block_mb,
        )
        self.controller = PressureDrivenController(
            collector=ctx.collector,
            tracker=ctx.tracker,
            importance_engine=engine,
            decision_engine=DecisionEngine(),
            placement=ctx.placement,
            config=controller_config,
        )

    def relieve(self) -> List[PlacementResult]:
        return [
            PlacementResult(
                block_id=d.block_id,
                requested_action=d.requested_action,
                executed_action=d.executed_action,
                freed_gpu_mb=d.freed_gpu_mb,
                fallback_reason=d.fallback_reason,
            )
            for d in self.controller.step()
        ]


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

class QueuePolicy:
    """
    FIFO / LRU style shedding: when GPU pressure reaches the trigger, take
    victims from the front of a queue until the target is reached. The
    victims are *evicted* (classic behaviour) or *offloaded to CPU*
    (``action="MIGRATE"``). Uses the same trigger/target watermarks as our
    controller so the comparison is apples to apples.
    """

    def __init__(
        self,
        ctx: PolicyContext,
        baseline: BaselinePolicy,
        action: str = "EVICT",
        controller_config: Optional[ControllerConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        if action not in ("EVICT", "MIGRATE"):
            raise ValueError("QueuePolicy action must be 'EVICT' or 'MIGRATE'")
        self.ctx = ctx
        self.baseline = baseline
        self.action = action
        self.config = controller_config or ControllerConfig()
        self.name = name or baseline.name

    def relieve(self) -> List[PlacementResult]:
        tracker = self.ctx.tracker
        pressure = tracker.gpu_pressure()
        if pressure < self.config.high_watermark:
            return []
        blocks = [b for b in self.ctx.collector.blocks_in_tier(Tier.GPU) if not b.pinned]
        actions = self.baseline.select_actions(
            blocks,
            pressure,
            tracker.gpu.capacity_mb,
            target_pressure=self.config.low_watermark,
            trigger_pressure=self.config.high_watermark,
        )
        results = []
        for block in blocks:
            if actions[block.block_id] != "KEEP":
                results.append(self.ctx.placement.apply(block.block_id, self.action))
        return results


# ---------------------------------------------------------------------------
# MTDS-style static thresholds
# ---------------------------------------------------------------------------

class MTDSStylePolicy:
    """
    Simplified static-threshold policy modelled on the base paper's
    *hierarchical eviction* (Wang et al., Complex & Intelligent Systems,
    2026, "Adaptive eviction design"):

    * Two capacity levels L1 < L2 (the paper's V1, V2).
    * At level L1, entries whose **utilization** is below U1 are moved down
      a tier; at level L2 a stricter threshold U2 > U1 applies. Below L1
      nothing happens. The paper's example values are U1 = 0.01, U2 = 0.1.
    * Utilization here is the block's *historical* share of requests,
      ``access_count / requests_seen``.

    What this is NOT (deliberately simplified, to be stated in the report):
    no SSD tier (we only have GPU and CPU), no sliding-window look-ahead
    term (xi = 0), no multi-GPU offload scheduler, and the thresholds are
    applied to GPU->CPU placement rather than DRAM->SSD. The *static*
    thresholds -- the property the project contrasts itself with -- are
    preserved: they do not react to workload, concurrency or pressure
    beyond the two fixed levels.
    """

    name = "MTDS-style"

    def __init__(
        self,
        ctx: PolicyContext,
        level1: float = 0.75,
        level2: float = 0.90,
        util_threshold1: float = 0.01,
        util_threshold2: float = 0.10,
    ) -> None:
        if not (0.0 < level1 < level2 <= 1.0 and 0.0 < util_threshold1 < util_threshold2):
            raise ValueError("require 0 < level1 < level2 <= 1 and 0 < U1 < U2")
        self.ctx = ctx
        self.level1, self.level2 = level1, level2
        self.u1, self.u2 = util_threshold1, util_threshold2

    def relieve(self) -> List[PlacementResult]:
        pressure = self.ctx.tracker.gpu_pressure()
        if pressure >= self.level2:
            threshold = self.u2
        elif pressure >= self.level1:
            threshold = self.u1
        else:
            return []
        n = max(1, self.ctx.requests_seen())
        victims = sorted(
            (b for b in self.ctx.collector.blocks_in_tier(Tier.GPU)
             if not b.pinned and b.access_count / n < threshold),
            key=lambda b: b.block_id,
        )
        return [self.ctx.placement.apply(b.block_id, "MIGRATE") for b in victims]


# ---------------------------------------------------------------------------
# Spec constructors
# ---------------------------------------------------------------------------

def adaptive_spec(
    weights: Optional[ImportanceWeights] = None,
    controller_config: Optional[ControllerConfig] = None,
    name: str = "Ours (Adaptive)",
) -> PolicySpec:
    return PolicySpec(name, lambda ctx: AdaptivePolicy(ctx, weights, controller_config, name))


def fifo_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec("FIFO", lambda ctx: QueuePolicy(ctx, FIFOPolicy(), "EVICT", controller_config))


def lru_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec("LRU", lambda ctx: QueuePolicy(ctx, LRUPolicy(), "EVICT", controller_config))


def lru_offload_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec(
        "LRU+offload",
        lambda ctx: QueuePolicy(ctx, LRUPolicy(), "MIGRATE", controller_config, name="LRU+offload"),
    )


def mtds_spec(**kwargs: float) -> PolicySpec:
    return PolicySpec("MTDS-style", lambda ctx: MTDSStylePolicy(ctx, **kwargs))


def default_policy_specs() -> List[PolicySpec]:
    """Ours plus the four baselines, in report order."""
    return [adaptive_spec(), fifo_spec(), lru_spec(), lru_offload_spec(), mtds_spec()]
