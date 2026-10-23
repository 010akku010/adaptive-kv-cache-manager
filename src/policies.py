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

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Protocol

from src.baselines import BaselinePolicy, FIFOPolicy, LRUPolicy
from src.clock import Clock
from src.controller import ControllerConfig, PressureDrivenController
from src.costmodel import CostModel, MsImportanceEngine, MsParams
from src.decision_engine import DecisionEngine
from src.hardware import DEFAULT_PROFILE, ServingProfile
from src.importance_engine import AdaptiveImportanceEngine, ImportanceWeights
from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier
from src.placement import PlacementResult, TierPlacementManager
from src.signals import Signals
from src.tuner import AdaptiveWeightTuner, TunerConfig


@dataclass
class PolicyContext:
    """Everything a policy may look at or act on."""

    collector: MetadataCollector
    tracker: MemoryTracker
    placement: TierPlacementManager
    clock: Clock
    block_mb: float  # standard block size (MB)
    requests_seen: Callable[[], int]  # requests processed so far
    signals: Signals = field(default_factory=Signals)  # always-on observations (see signals.py)
    profile: ServingProfile = DEFAULT_PROFILE  # hardware model (for the millisecond cost model)
    block_tokens: int = 16


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


class MsAdaptivePolicy:
    """
    The cost-benefit engine with every quantity in milliseconds
    (``costmodel.MsImportanceEngine``), optionally with the Adaptive Weight
    Tuner closing the feedback loop (``tuner.AdaptiveWeightTuner``).
    """

    def __init__(
        self,
        ctx: PolicyContext,
        name: str,
        weights: Optional[ImportanceWeights] = None,
        params: Optional[MsParams] = None,
        controller_config: Optional[ControllerConfig] = None,
        tuner_config: Optional[TunerConfig] = None,
        recency_tau_s: float = 30.0,
    ) -> None:
        self.name = name
        engine = MsImportanceEngine(
            cost_model=CostModel(ctx.profile, ctx.block_tokens),
            params=params,
            weights=weights,
            recency_tau_s=recency_tau_s,
            clock=ctx.clock,
            cpu_free_mb=lambda: ctx.tracker.cpu.free_mb,
            cpu_utilization=lambda: ctx.tracker.cpu.utilization,
        )
        self.controller = PressureDrivenController(
            collector=ctx.collector,
            tracker=ctx.tracker,
            importance_engine=engine,
            decision_engine=DecisionEngine(),
            placement=ctx.placement,
            config=controller_config,
        )
        self.engine = engine
        self.tuner: Optional[AdaptiveWeightTuner] = None
        if tuner_config is not None:
            self.tuner = AdaptiveWeightTuner(
                engine=engine,
                controller_config=self.controller.config,
                signals=ctx.signals,
                clock=ctx.clock,
                config=tuner_config,
            )

    def relieve(self) -> List[PlacementResult]:
        if self.tuner is not None:
            self.tuner.maybe_step()
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


class LadderPolicy:
    """
    Static, tier-aware baseline that uses ALL of our actions but with a fixed
    rule instead of cost-benefit reasoning: LRU order, and a fixed ladder
    "compress first, then offload".

    When GPU pressure reaches the trigger, LRU blocks are compressed (frees
    ``compression_savings`` of each block while keeping it on the GPU) until
    the target is reached; if every candidate is already compressed and the
    target is still not met, the least-recently-used blocks are offloaded to
    CPU. It exists so that gains attributed to our engine are not just gains
    from *having* compression available: this baseline has it too, but cannot
    reason about which blocks deserve it. Compression here is unconditional,
    so it ignores any accuracy budget.
    """

    name = "LRU+compress+offload"

    def __init__(
        self,
        ctx: PolicyContext,
        controller_config: Optional[ControllerConfig] = None,
        max_compressed_fraction: float = 1.0,
    ) -> None:
        if not 0.0 <= max_compressed_fraction <= 1.0:
            raise ValueError("max_compressed_fraction must be in [0, 1]")
        self.ctx = ctx
        self.config = controller_config or ControllerConfig()
        # Static cap on the share of GPU-resident blocks that may be compressed.
        # 1.0 = unconstrained; smaller values give a fixed quality budget, so the
        # ladder can be compared with our engine at equal compressed-read exposure.
        self.max_compressed_fraction = max_compressed_fraction

    def relieve(self) -> List[PlacementResult]:
        tracker, placement = self.ctx.tracker, self.ctx.placement
        if tracker.gpu_pressure() < self.config.high_watermark:
            return []
        target = self.config.low_watermark
        results: List[PlacementResult] = []
        lru = sorted(
            (b for b in self.ctx.collector.blocks_in_tier(Tier.GPU) if not b.pinned),
            key=lambda b: (b.last_access, b.block_id),
        )
        resident = list(self.ctx.collector.blocks_in_tier(Tier.GPU))
        compressed = sum(1 for b in resident if b.compressed)
        cap = self.max_compressed_fraction * len(resident)
        for block in lru:  # pass 1: compress (up to the static cap)
            if tracker.gpu_pressure() <= target:
                return results
            if compressed >= cap:
                break
            if not block.compressed and block.compression_benefit_mb > 0.0:
                results.append(placement.apply(block.block_id, "COMPRESS"))
                compressed += 1
        for block in lru:  # pass 2: offload (compressed blocks move smaller)
            if tracker.gpu_pressure() <= target:
                break
            if block.location == Tier.GPU:
                results.append(placement.apply(block.block_id, "MIGRATE"))
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


def ms_adaptive_spec(
    name: str = "Ours-ms",
    weights: Optional[ImportanceWeights] = None,
    params: Optional[MsParams] = None,
    controller_config: Optional[ControllerConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    recency_tau_s: float = 30.0,
) -> PolicySpec:
    """Millisecond cost model; pass ``tuner_config`` to enable the Adaptive Weight Tuner."""

    def build(ctx: PolicyContext) -> CachePolicy:
        # Fresh copies per run: the tuner mutates params and controller config in place.
        return MsAdaptivePolicy(
            ctx, name, weights, replace(params) if params else None,
            replace(controller_config) if controller_config else None,
            tuner_config, recency_tau_s,
        )

    return PolicySpec(name, build)


def fifo_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec("FIFO", lambda ctx: QueuePolicy(ctx, FIFOPolicy(), "EVICT", controller_config))


def lru_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec("LRU", lambda ctx: QueuePolicy(ctx, LRUPolicy(), "EVICT", controller_config))


def lru_offload_spec(controller_config: Optional[ControllerConfig] = None) -> PolicySpec:
    return PolicySpec(
        "LRU+offload",
        lambda ctx: QueuePolicy(ctx, LRUPolicy(), "MIGRATE", controller_config, name="LRU+offload"),
    )


def ladder_spec(
    controller_config: Optional[ControllerConfig] = None,
    max_compressed_fraction: float = 1.0,
    name: str = "LRU+compress+offload",
) -> PolicySpec:
    return PolicySpec(name, lambda ctx: LadderPolicy(ctx, controller_config, max_compressed_fraction))


def mtds_spec(**kwargs: float) -> PolicySpec:
    return PolicySpec("MTDS-style", lambda ctx: MTDSStylePolicy(ctx, **kwargs))


def default_policy_specs() -> List[PolicySpec]:
    """Ours plus the four baselines, in report order."""
    return [adaptive_spec(), fifo_spec(), lru_spec(), lru_offload_spec(), mtds_spec()]
